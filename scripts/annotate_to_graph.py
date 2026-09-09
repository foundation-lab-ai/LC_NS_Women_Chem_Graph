#!/usr/bin/env python3
"""
annotate_to_graph.py
--------------------
Turn an untargeted LC-HRMS feature table into GNN-ready node features + edges.

Design principle: EVERY node gets chemistry-derived features, whether or not it
can be named. Database annotation is an optional enrichment layer, not a filter.

Inputs
------
--features  CSV/TSV with columns: mz, rt, and one column per sample (intensities)
--db        (optional) compound DB CSV: name, formula, monoisotopic_mass, main_class
            Get RefMet: metabolomicsworkbench.org/databases/refmet/refmet_download.php
            (columns there: refmet_name, formula, exactmass, main_class, super_class)

Outputs (--outdir)
------------------
node_features.csv    one row per feature -> x matrix for PyG
edges.csv            source,target,edge_type,delta,transformation
annotations.csv      long format: every candidate match per feature
redundancy.csv       adduct/isotope grouping detail

Usage
-----
python annotate_to_graph.py --features feats.csv --db refmet.csv --outdir out/
python annotate_to_graph.py --demo            # runs on synthetic data
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

PROTON = 1.007276
ELECTRON = 0.000549

# Adducts: name -> (mass shift from neutral M, charge, multiplier on M)
ADDUCTS_POS = {
    "[M+H]+":    (1.007276, 1, 1),
    "[M+Na]+":   (22.989218, 1, 1),
    "[M+NH4]+":  (18.033823, 1, 1),
    "[M+K]+":    (38.963158, 1, 1),
    "[M+H-H2O]+": (-17.002740, 1, 1),
    "[2M+H]+":   (1.007276, 1, 2),
}
ADDUCTS_NEG = {
    "[M-H]-":    (-1.007276, 1, 1),
    "[M+Cl]-":   (34.969402, 1, 1),
    "[M+HCOO]-": (44.998201, 1, 1),
    "[M-H2O-H]-": (-19.017840, 1, 1),
    "[2M-H]-":   (-1.007276, 1, 2),
}

# Redundancy offsets: same compound, different ion.
# MODE MATTERS. Na-H and K-H describe [M+Na]+ vs [M+H]+ and cannot occur in
# negative mode; applying them there discards real metabolites. Likewise
# formate and chloride adducts only exist in negative mode.
ISOTOPE_C13 = 1.003355

ISOTOPE_OFFSETS = {"13C1": ISOTOPE_C13, "13C2": 2 * ISOTOPE_C13}

ADDUCT_OFFSETS_POS = {
    "Na-H": 21.981944,
    "K-H": 37.955882,
    "NH4-H": 17.026549,
    "H2O_loss": -18.010565,
}
ADDUCT_OFFSETS_NEG = {
    "Cl-H": 35.976678,      # [M+Cl]- vs [M-H]-
    "HCOO-H": 46.005479,    # [M+HCOO]- vs [M-H]-
}

# Biotransformations: name -> exact mass difference (Da)
TRANSFORMATIONS = {
    "methylation":      14.015650,
    "hydroxylation":    15.994915,
    "desaturation":     -2.015650,
    "acetylation":      42.010565,
    "carboxylation":    43.989829,
    "phosphorylation":  79.966331,
    "sulfation":        79.956815,
    "glycine_conj":     57.021464,
    "taurine_conj":     107.004099,
    "glucuronidation":  176.032088,
    "hexose":           162.052824,
    "amination":        15.010899,
    "deamination":      -16.018724,
    "C2H4_chain":       28.031300,
    "glutathione":      305.068156,
    "malonyl":          86.000394,
    "acetyl_CoA_unit":  42.010565,
    "reduction_2H":     2.015650,
    "oxidation_O":      15.994915,
    "CO2":              43.989829,
    "NH3":              17.026549,
}

# Transformations that are numerically confusable with MS artifacts.
# Flagged so you can down-weight or drop them.
ARTIFACT_RISK = {"hydroxylation", "oxidation_O", "reduction_2H", "desaturation"}


# ----------------------------------------------------------------------------
# 1. Load + QC
# ----------------------------------------------------------------------------

def load_feature_table(path, mz_col=None, rt_col=None):
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(path, sep=sep)
    cols = {c.lower().strip(): c for c in df.columns}

    mz_col = mz_col or next((cols[k] for k in ("mz", "m/z", "mass", "mz_medn")
                             if k in cols), None)
    rt_col = rt_col or next((cols[k] for k in ("rt", "time", "retention_time",
                                               "rt_medn", "rtmed")
                             if k in cols), None)
    if mz_col is None or rt_col is None:
        sys.exit(f"Could not find mz/rt columns. Saw: {list(df.columns)[:12]}")

    df = df.rename(columns={mz_col: "mz", rt_col: "rt"})
    sample_cols = [c for c in df.columns
                   if c not in ("mz", "rt") and pd.api.types.is_numeric_dtype(df[c])]
    df["feature_id"] = [f"F{i:06d}" for i in range(len(df))]
    return df, sample_cols


def qc_filter(df, sample_cols, min_presence=0.8, min_cv=None):
    """Drop features detected in too few samples. Optionally drop low-variance."""
    X = df[sample_cols].to_numpy(dtype=float)
    detected = np.isfinite(X) & (X > 0)
    presence = detected.mean(axis=1)
    keep = presence >= min_presence

    if min_cv is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.nanmean(np.where(detected, X, np.nan), axis=1)
            sd = np.nanstd(np.where(detected, X, np.nan), axis=1)
            cv = np.where(mu > 0, sd / mu, 0)
        keep &= cv >= min_cv

    out = df.loc[keep].copy()
    out["presence"] = presence[keep]
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Intrinsic chemistry features  (available for EVERY node, named or not)
# ----------------------------------------------------------------------------

def intrinsic_features(df):
    """
    Physicochemical descriptors computable from m/z and RT alone.
    These are the workhorse node features for unannotated peaks.
    """
    mz = df["mz"].to_numpy(dtype=float)
    rt = df["rt"].to_numpy(dtype=float)

    # Mass defect = decimal part. Tracks H/C ratio -> proxy for compound class.
    mass_defect = mz - np.floor(mz)
    # Relative mass defect (ppm-scaled) separates lipids from polar metabolites.
    rel_mass_defect = mass_defect / mz * 1000.0
    # Kendrick mass defect (CH2 base): homologous series collapse to same KMD.
    kendrick = mz * (14.0 / 14.015650)
    kmd = np.round(kendrick) - kendrick
    # Same idea on an O base -> oxidation series
    kendrick_o = mz * (16.0 / 15.994915)
    kmd_o = np.round(kendrick_o) - kendrick_o

    out = pd.DataFrame({
        "feature_id": df["feature_id"].values,
        "mz": mz,
        "rt": rt,
        "log_mz": np.log10(mz),
        "mass_defect": mass_defect,
        "rel_mass_defect": rel_mass_defect,
        "kendrick_md_ch2": kmd,
        "kendrick_md_o": kmd_o,
        "rt_norm": (rt - rt.min()) / max(float(np.ptp(rt)), 1e-9),
    })
    return out


# ----------------------------------------------------------------------------
# 3. Redundancy grouping (isotopes / adducts / dimers)
# ----------------------------------------------------------------------------

def group_redundant(df, sample_cols=None, mode="positive", rt_tol=10.0,
                    ppm=5.0, max_isotope_ratio=0.9):
    """
    Flag isotope / adduct / dimer ions so they don't masquerade as
    biotransformations in the graph.

    Three rules that matter, learned the hard way:

    1. MODE-DEPENDENT ADDUCTS. Na-H and K-H are positive-mode only. Screening
       for them in negative mode discards genuine metabolites (an LPE 22:6 was
       being deleted as a sodium adduct in a negative-mode run).

    2. ISOTOPES NEED AN INTENSITY CHECK. A real 13C1 peak is ~1.1% x nC of its
       monoisotopic peak -- typically 20-40%, and never larger. If the
       candidate "isotope" is comparable to or more intense than the supposed
       parent, the 1.0034 spacing is coincidence and these are two compounds.

    3. FLAGGING IS NOT DELETION. Redundant ions stay as graph nodes; the flag
       is recorded in ion_role so downstream code can choose. Dropping them
       from edge construction orphans real metabolites into singleton modules.
    """
    df = df.sort_values("rt").reset_index(drop=True)
    rt = df["rt"].to_numpy(dtype=float)
    mz = df["mz"].to_numpy(dtype=float)

    inten = None
    if sample_cols:
        X = df[sample_cols].to_numpy(dtype=float)
        inten = np.nanmedian(np.where(X > 0, X, np.nan), axis=1)
        inten = np.nan_to_num(inten, nan=0.0)

    adducts = ADDUCT_OFFSETS_POS if mode == "positive" else ADDUCT_OFFSETS_NEG

    parent = np.arange(len(df))
    role = np.array(["monoisotopic"] * len(df), dtype=object)
    reason = np.array([""] * len(df), dtype=object)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(len(df)):
        j = i + 1
        while j < len(df) and rt[j] - rt[i] <= rt_tol:
            d = abs(mz[j] - mz[i])
            tol = ppm * 1e-6 * max(mz[i], mz[j])
            lo, hi = (i, j) if mz[j] > mz[i] else (j, i)
            matched = False

            for name, off in ISOTOPE_OFFSETS.items():
                if abs(d - off) <= tol:
                    # intensity gate: isotope must be clearly weaker than parent
                    if inten is not None and inten[lo] > 0:
                        if inten[hi] / inten[lo] > max_isotope_ratio:
                            break          # too intense -- a separate compound
                    union(i, j); role[hi] = "redundant"; reason[hi] = name
                    matched = True
                    break
            if matched:
                j += 1
                continue

            for name, off in adducts.items():
                if abs(d - abs(off)) <= tol:
                    union(i, j); role[hi] = "redundant"; reason[hi] = name
                    matched = True
                    break
            if not matched and abs(d - (min(mz[i], mz[j]) - PROTON)) <= tol * 2:
                union(i, j); role[hi] = "redundant"; reason[hi] = "dimer"
            j += 1

    df["rt_cluster"] = np.digitize(rt, np.arange(rt.min(), rt.max() + rt_tol,
                                                 rt_tol))
    df["redundancy_group"] = [f"G{find(i):06d}" for i in range(len(df))]
    df["ion_role"] = role
    df["redundancy_reason"] = reason
    return df


# ----------------------------------------------------------------------------
# 4. Database matching
# ----------------------------------------------------------------------------

def load_db(path):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    name = next((cols[k] for k in ("name", "refmet_name", "compound") if k in cols), None)
    mass = next((cols[k] for k in ("monoisotopic_mass", "exactmass", "exact_mass",
                                   "mass") if k in cols), None)
    klass = next((cols[k] for k in ("main_class", "class", "sub_class")
                  if k in cols), None)
    sup = next((cols[k] for k in ("super_class", "superclass") if k in cols),
               None)
    formula = next((cols[k] for k in ("formula", "chemical_formula")
                    if k in cols), None)
    if name is None or mass is None:
        sys.exit("DB needs a name column and a monoisotopic mass column.")

    out = df[[c for c in (name, mass, klass, sup, formula)
              if c is not None]].copy()
    out.columns = ["name", "mass"] + (["main_class"] if klass else []) + \
                  (["super_class"] if sup else []) + \
                  (["formula"] if formula else [])
    if "main_class" not in out:
        out["main_class"] = "unknown"
    if "super_class" not in out:
        out["super_class"] = out["main_class"]
    if "formula" not in out:
        out["formula"] = ""
    out["mass"] = pd.to_numeric(out["mass"], errors="coerce")
    return out.dropna(subset=["mass"]).reset_index(drop=True)


def match_database(df, db, mode="positive", ppm=5.0):
    """
    For each feature, enumerate neutral-mass hypotheses under each adduct and
    look for DB compounds within tolerance. Returns long-format candidates.
    """
    adducts = ADDUCTS_POS if mode == "positive" else ADDUCTS_NEG
    db_mass = db["mass"].to_numpy()
    order = np.argsort(db_mass)
    db_sorted = db_mass[order]

    rows = []
    for fid, mzv in zip(df["feature_id"], df["mz"].astype(float)):
        for aname, (shift, charge, mult) in adducts.items():
            neutral = (mzv * charge - shift) / mult
            if neutral <= 0:
                continue
            tol = ppm * 1e-6 * mzv
            lo = np.searchsorted(db_sorted, neutral - tol, "left")
            hi = np.searchsorted(db_sorted, neutral + tol, "right")
            for j in order[lo:hi]:
                rows.append({
                    "feature_id": fid,
                    "adduct": aname,
                    "neutral_mass": neutral,
                    "db_name": db.at[j, "name"],
                    "db_formula": db.at[j, "formula"],
                    "main_class": db.at[j, "main_class"],
                    "super_class": db.at[j, "super_class"],
                    "ppm_error": (neutral - db.at[j, "mass"]) / neutral * 1e6,
                })
    return pd.DataFrame(rows)


def score_annotations(cand, df):
    """
    Collapse candidates -> one row per feature with a confidence tier.

    MSI-style tiers reachable from MS1 only:
      3  single class-consistent candidate, [M+H]+/[M-H]- adduct  -> class-level
      4a few candidates, all same class                           -> class-level, weak
      4b many candidates / conflicting classes                    -> unknown
      5  no match
    Level 2 requires MS/MS; level 1 requires an authentic standard. Say so.
    """
    if cand.empty:
        base = pd.DataFrame({"feature_id": df["feature_id"]})
        base["n_candidates"] = 0
        base["annotated"] = 0
        base["class_consistent"] = 0
        base["main_class"] = "unknown"
        base["super_class"] = "unknown"
        base["super_consistent"] = 0
        base["best_name"] = ""
        base["msi_level"] = 5
        base["annot_confidence"] = 0.0
        return base

    g = cand.groupby("feature_id")
    recs = []
    primary = {"[M+H]+", "[M-H]-"}
    for fid, sub in g:
        classes = sub["main_class"].dropna().unique()
        supers = (sub["super_class"].dropna().unique()
                  if "super_class" in sub else classes)
        n = len(sub)
        consistent = int(len(classes) == 1)
        sup_consistent = int(len(supers) == 1)
        has_primary = int(bool(primary & set(sub["adduct"])))
        best = sub.iloc[sub["ppm_error"].abs().to_numpy().argmin()]

        if n == 1 and has_primary:
            level, conf = 3, 0.9
        elif consistent and n <= 5:
            level, conf = 4, 0.6
        elif consistent:
            level, conf = 4, 0.4
        else:
            level, conf = 4, 0.2
        recs.append({
            "feature_id": fid,
            "n_candidates": n,
            "annotated": 1,
            "class_consistent": consistent,
            "main_class": classes[0] if consistent else "ambiguous",
            # a feature whose candidates disagree on the exact class but all
            # sit in one super class is still usable: "a lipid" is a real
            # constraint, and dropping it discards most of the annotation
            "super_class": supers[0] if sup_consistent else "ambiguous",
            "super_consistent": sup_consistent,
            "best_name": best["db_name"],
            "msi_level": level,
            "annot_confidence": conf,
        })

    out = pd.DataFrame(recs)
    out = df[["feature_id"]].merge(out, on="feature_id", how="left")
    out["n_candidates"] = out["n_candidates"].fillna(0).astype(int)
    out["annotated"] = out["annotated"].fillna(0).astype(int)
    out["class_consistent"] = out["class_consistent"].fillna(0).astype(int)
    out["main_class"] = out["main_class"].fillna("unknown")
    out["super_class"] = out["super_class"].fillna("unknown")
    out["super_consistent"] = out["super_consistent"].fillna(0).astype(int)
    out["best_name"] = out["best_name"].fillna("")
    out["msi_level"] = out["msi_level"].fillna(5).astype(int)
    out["annot_confidence"] = out["annot_confidence"].fillna(0.0)
    return out


# ----------------------------------------------------------------------------
# 5. Mass-difference edges
# ----------------------------------------------------------------------------

def mass_difference_edges(df, ppm=5.0, rt_max_shift=None,
                          exclude_redundant=False, drop_artifact_risk=False):
    """
    Edge between features whose m/z difference matches a biotransformation.
    Operates on representative ions only (redundant ions dropped).
    """
    work = df
    if exclude_redundant and "ion_role" in df:
        work = df[df["ion_role"] == "monoisotopic"]
    work = work.sort_values("mz").reset_index(drop=True)

    mz = work["mz"].to_numpy(dtype=float)
    rt = work["rt"].to_numpy(dtype=float)
    fid = work["feature_id"].to_numpy()

    trans = {k: v for k, v in TRANSFORMATIONS.items()
             if not (drop_artifact_risk and k in ARTIFACT_RISK)}

    rows = []
    for name, delta in trans.items():
        d = abs(delta)
        # exploit sorted order: for each i find window matching mz[i] + d
        targets = mz + d
        tol = ppm * 1e-6 * targets
        lo = np.searchsorted(mz, targets - tol, "left")
        hi = np.searchsorted(mz, targets + tol, "right")
        for i in range(len(mz)):
            for j in range(lo[i], hi[i]):
                if i == j:
                    continue
                if rt_max_shift is not None and abs(rt[j] - rt[i]) > rt_max_shift:
                    continue
                rows.append({
                    "source": fid[i],
                    "target": fid[j],
                    "edge_type": "mass_difference",
                    "transformation": name,
                    "delta": mz[j] - mz[i],
                    "artifact_risk": int(name in ARTIFACT_RISK),
                })
    edges = pd.DataFrame(rows)
    if not edges.empty:
        edges = edges.drop_duplicates(subset=["source", "target", "transformation"])
    return edges


def correlation_edges(df, sample_cols, top_k=10, min_r=0.5):
    """Second edge layer: co-abundance. Keeps top_k partners per node."""
    X = df[sample_cols].to_numpy(dtype=float)
    X = np.log1p(np.nan_to_num(X, nan=0.0))
    Xc = X - X.mean(axis=1, keepdims=True)
    sd = Xc.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    Z = Xc / sd
    R = (Z @ Z.T) / X.shape[1]
    np.fill_diagonal(R, 0)

    fid = df["feature_id"].to_numpy()
    rows = []
    k = min(top_k, R.shape[0] - 1)
    for i in range(R.shape[0]):
        part = np.argpartition(-np.abs(R[i]), k)[:k]
        for j in part:
            if abs(R[i, j]) >= min_r:
                rows.append({
                    "source": fid[i], "target": fid[j],
                    "edge_type": "correlation",
                    "transformation": "",
                    "delta": float(R[i, j]),
                    "artifact_risk": 0,
                })
    e = pd.DataFrame(rows)
    if not e.empty:
        # dedupe undirected
        key = e.apply(lambda r: tuple(sorted((r["source"], r["target"]))), axis=1)
        e = e.loc[~key.duplicated()]
    return e


# ----------------------------------------------------------------------------
# 6. Assemble node feature matrix
# ----------------------------------------------------------------------------

def build_node_features(intr, annot, redun, edges, min_class_count=20):
    """
    Final x matrix. Columns:
      - intrinsic chemistry (always populated)
      - annotation flags + confidence
      - one-hot compound class (rare classes folded into 'other')
      - graph-derived degree per edge type
    """
    nf = intr.merge(annot, on="feature_id", how="left")
    nf = nf.merge(
        redun[["feature_id", "ion_role", "presence"]],
        on="feature_id", how="left"
    )
    nf["is_monoisotopic"] = (nf["ion_role"] == "monoisotopic").astype(int)
    nf = nf.drop(columns=["ion_role"])

    # class one-hot, rare classes folded
    counts = nf["main_class"].value_counts()
    keep = set(counts[counts >= min_class_count].index) - {"unknown", "ambiguous"}
    nf["class_grp"] = nf["main_class"].where(nf["main_class"].isin(keep),
                                             "other")
    oh = pd.get_dummies(nf["class_grp"], prefix="class").astype(int)
    nf = pd.concat([nf.drop(columns=["class_grp"]), oh], axis=1)

    # degree per edge type
    if edges is not None and not edges.empty:
        for et in edges["edge_type"].unique():
            sub = edges[edges["edge_type"] == et]
            deg = pd.concat([sub["source"], sub["target"]]).value_counts()
            nf[f"deg_{et}"] = nf["feature_id"].map(deg).fillna(0).astype(int)
        # transformation-type degree: mechanistic signal without identity
        mdi = edges[edges["edge_type"] == "mass_difference"]
        for t in ("methylation", "hydroxylation", "desaturation",
                  "glucuronidation", "sulfation", "C2H4_chain"):
            sub = mdi[mdi["transformation"] == t]
            deg = pd.concat([sub["source"], sub["target"]]).value_counts()
            nf[f"deg_{t}"] = nf["feature_id"].map(deg).fillna(0).astype(int)

    return nf


# ----------------------------------------------------------------------------
# Demo data
# ----------------------------------------------------------------------------

def make_demo(n=400, n_samples=40, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(120, 900, n // 2)
    mz = list(base)
    rt = list(rng.uniform(30, 600, n // 2))
    # add real relationships so the pipeline has something to find
    for i in range(n - n // 2):
        p = rng.integers(0, len(base))
        kind = rng.integers(0, 3)
        if kind == 0:      # methyl derivative, similar RT
            mz.append(base[p] + 14.015650); rt.append(rt[p] + rng.normal(0, 5))
        elif kind == 1:    # 13C isotope, same RT
            mz.append(base[p] + 1.003355); rt.append(rt[p] + rng.normal(0, 1))
        else:              # Na adduct, same RT
            mz.append(base[p] + 21.981944); rt.append(rt[p] + rng.normal(0, 1))
    df = pd.DataFrame({"mz": mz, "rt": rt})
    for s in range(n_samples):
        df[f"S{s:03d}"] = rng.lognormal(10, 1, len(df))
    return df


DEMO_DB = pd.DataFrame({
    "name": ["Choline", "Betaine", "Taurine", "LPE(18:0)", "LPE(16:0)", "Valine"],
    "mass": [103.099714, 117.078979, 125.014664, 481.316, 453.285, 117.078979],
    "main_class": ["Cholines", "Betaines", "Sulfonic acids",
                   "Lysophosphatidylethanolamines",
                   "Lysophosphatidylethanolamines", "Amino acids"],
    "formula": ["C5H13NO", "C5H11NO2", "C2H7NO3S", "C23H48NO7P",
                "C21H44NO7P", "C5H11NO2"],
})


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features")
    ap.add_argument("--db")
    ap.add_argument("--outdir", default="graph_out")
    ap.add_argument("--mode", choices=["positive", "negative"], default="positive")
    ap.add_argument("--ppm", type=float, default=5.0)
    ap.add_argument("--rt-tol", type=float, default=10.0,
                    help="seconds; RT window for adduct/isotope grouping")
    ap.add_argument("--rt-max-shift", type=float, default=None,
                    help="max RT difference allowed for a biotransformation edge")
    ap.add_argument("--min-presence", type=float, default=0.8)
    ap.add_argument("--top-k-corr", type=int, default=10)
    ap.add_argument("--no-corr-edges", action="store_true")
    ap.add_argument("--drop-artifact-risk", action="store_true",
                    help="drop transformations confusable with MS artifacts")
    ap.add_argument("--max-isotope-ratio", type=float, default=0.9,
                    help="a 13C peak more intense than this fraction of its "
                         "supposed parent is treated as a separate compound")
    ap.add_argument("--exclude-redundant", action="store_true",
                    help="drop flagged ions from the graph entirely. OFF by "
                         "default: excluding them orphans real metabolites "
                         "into singleton modules.")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.demo:
        raw = make_demo()
        sample_cols = [c for c in raw.columns if c.startswith("S")]
        raw["feature_id"] = [f"F{i:06d}" for i in range(len(raw))]
        db = DEMO_DB
    else:
        if not args.features:
            sys.exit("--features required (or use --demo)")
        raw, sample_cols = load_feature_table(args.features)
        db = load_db(args.db) if args.db else None

    print(f"[1/6] loaded {len(raw)} features x {len(sample_cols)} samples")

    df = qc_filter(raw, sample_cols, min_presence=args.min_presence)
    print(f"[2/6] QC: {len(df)} features retained "
          f"(presence >= {args.min_presence})")

    df = group_redundant(df, sample_cols=sample_cols, mode=args.mode,
                         rt_tol=args.rt_tol, ppm=args.ppm,
                         max_isotope_ratio=args.max_isotope_ratio)
    n_red = (df["ion_role"] == "redundant").sum()
    print(f"[3/6] redundancy ({args.mode}): {n_red} flagged "
          f"({n_red/len(df):.1%}); {df['redundancy_group'].nunique()} groups; "
          f"kept in graph: {'no' if args.exclude_redundant else 'yes'}")
    if n_red:
        print("      " + ", ".join(
            f"{k}={v}" for k, v in
            df.loc[df.ion_role == "redundant", "redundancy_reason"]
              .value_counts().head(6).items()))

    intr = intrinsic_features(df)
    print(f"[4/6] intrinsic features: {intr.shape[1]-1} descriptors, all nodes")

    if db is not None:
        cand = match_database(df, db, mode=args.mode, ppm=args.ppm)
        annot = score_annotations(cand, df)
        cand.to_csv(f"{args.outdir}/annotations.csv", index=False)
        n_ann = int(annot["annotated"].sum())
        n_mc = int((~annot["main_class"].isin(
            ["unknown", "ambiguous"])).sum())
        n_sc = int((~annot["super_class"].isin(
            ["unknown", "ambiguous"])).sum())
        print(f"[5/6] annotation: {n_ann}/{len(df)} matched "
              f"({n_ann/len(df):.1%}); {int((annot['msi_level']==3).sum())} "
              f"at MSI level 3")
        print(f"      usable class labels: main_class {n_mc} "
              f"({n_mc/len(df):.1%}), super_class {n_sc} "
              f"({n_sc/len(df):.1%})")
    else:
        annot = score_annotations(pd.DataFrame(), df)
        print("[5/6] annotation: skipped (no --db)")

    md = mass_difference_edges(df, ppm=args.ppm,
                               rt_max_shift=args.rt_max_shift,
                               exclude_redundant=args.exclude_redundant,
                               drop_artifact_risk=args.drop_artifact_risk)
    parts = [md]
    if not args.no_corr_edges:
        ce = correlation_edges(df, sample_cols, top_k=args.top_k_corr)
        parts.append(ce)
    edges = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    print(f"[6/6] edges: {len(md)} mass-difference, "
          f"{len(edges)-len(md)} correlation")

    nf = build_node_features(intr, annot, df, edges)

    nf.to_csv(f"{args.outdir}/node_features.csv", index=False)
    edges.to_csv(f"{args.outdir}/edges.csv", index=False)
    df[["feature_id", "mz", "rt", "redundancy_group", "ion_role",
        "redundancy_reason"]].to_csv(f"{args.outdir}/redundancy.csv", index=False)

    print(f"\nwrote -> {args.outdir}/")
    print(f"  node_features.csv  {nf.shape[0]} x {nf.shape[1]}")
    print(f"  edges.csv          {edges.shape[0]}")
    print("\nSanity checks to run before modeling:")
    deg = edges[edges.edge_type == "mass_difference"]
    if not deg.empty:
        d = pd.concat([deg.source, deg.target]).value_counts()
        print(f"  mass-diff degree: median {d.median():.0f}, "
              f"max {d.max()}, mean {d.mean():.1f}")
        print("  -> want heavy-tailed. If flat, tighten --ppm.")
    print(f"  redundancy flagged: {n_red/len(df):.1%} "
          f"(expect 30-60% in real Orbitrap data; if <10%, check --rt-tol)")


if __name__ == "__main__":
    main()
