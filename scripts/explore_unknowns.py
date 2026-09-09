#!/usr/bin/env python3
"""
explore_unknowns.py — turn unidentified features into testable hypotheses

An unannotated feature one mass-difference edge from an annotated compound has
an implied identity: neighbor + transformation. "LPE 18:2 plus a methylene" is
a specific, falsifiable claim about a peak that no database could name.

Chromatography provides an independent check. On reversed phase, adding CH2
makes a compound more hydrophobic so it elutes LATER; adding a hydroxyl or a
double bond makes it more polar so it elutes EARLIER. On HILIC the polar
direction reverses. A proposed relationship whose retention shift runs the
wrong way is probably a mass coincidence, and this flags it.

Output ranks candidates by association strength, direction agreement with the
neighbor, and retention-time plausibility -- an MS/MS shortlist.

Usage
-----
python explore_unknowns.py --data out/c18_neg --disc disc/c18_neg/features.csv \\
    --mode rp --seed-annotations lipids --out unknowns_c18.csv
"""

import argparse
import re
import sys
import numpy as np
import pandas as pd

# expected retention shift when the transformation is ADDED (higher mass)
#   +1 later, -1 earlier, 0 unclear. Reversed-phase convention.
RT_RP = {
    "C2H4_chain": +1, "acetyl_CoA_unit": +1, "methylation": +1,
    "acetylation": +1, "malonyl": 0,
    "hydroxylation": -1, "oxidation_O": -1, "carboxylation": -1,
    "glucuronidation": -1, "sulfation": -1, "taurine_conj": -1,
    "glycine_conj": -1, "phosphorylation": -1, "hexose": -1,
    "amination": -1, "deamination": +1,
    "desaturation": -1, "reduction_2H": +1,
    "CO2": -1, "NH3": 0, "glutathione": -1,
}
LIPID = ("LPE", "LPC", "LPA", "CPA", "PE", "PC", "PA", "PI", "PS", "SM",
         "Cer", "TG", "DG", "MG", "FA", "CAR", "LysoP")


def is_lipid(n, min_chain=12):
    """
    Prefix alone is not enough. RefMet enumerates lipids combinatorially, so
    'LPC 4:1' and 'MG 7:0' match the prefix but do not occur at meaningful
    abundance in human plasma, where acyl chains are typically 14-22 carbons.
    """
    if not isinstance(n, str) or not any(n.strip().startswith(p)
                                         for p in LIPID):
        return False
    chains = [int(c) for c in re.findall(r"(\d{1,2}):\d", n)]
    return bool(chains) and max(chains) >= min_chain


def tidy(n):
    if not isinstance(n, str):
        return ""
    return re.sub(r"\s+", " ", n.replace("(:", "(").replace(":)", ")")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--disc", required=True,
                    help="features.csv written by discover.py")
    ap.add_argument("--mode", choices=["rp", "hilic"], default="rp",
                    help="chromatography, for the retention-shift check")
    ap.add_argument("--seed-annotations", choices=["lipids", "any"],
                    default="lipids",
                    help="which annotated features may seed a proposal")
    ap.add_argument("--seed-list",
                    help="CSV with feature_id (and optionally annotation) of "
                         "TRUSTED seeds. Strongly preferred: RefMet contains "
                         "in-silico combinatorial lipids and buffers such as "
                         "PIPES that pass a name-prefix test but are not "
                         "plausible plasma metabolites.")
    ap.add_argument("--max-delta-rt", type=float, default=20.0,
                    help="seconds; a proposal whose retention shift exceeds "
                         "this is rejected. One reaction step does not move a "
                         "peak most of the way across the gradient.")
    ap.add_argument("--min-chain", type=int, default=12,
                    help="reject lipid seeds whose acyl chains are shorter "
                         "than this many carbons")
    ap.add_argument("--min-abs-t", type=float, default=1.5,
                    help="only report unknowns at least this strongly "
                         "associated")
    ap.add_argument("--same-direction", action="store_true", default=True)
    ap.add_argument("--any-direction", dest="same_direction",
                    action="store_false")
    ap.add_argument("--out", default="unknowns.csv")
    a = ap.parse_args()

    nf = pd.read_csv(f"{a.data}/node_features.csv")
    ed = pd.read_csv(f"{a.data}/edges.csv")
    md = ed[ed.edge_type == "mass_difference"] if "edge_type" in ed else ed
    if "transformation" not in md:
        sys.exit("edges.csv has no transformation column")
    disc = pd.read_csv(a.disc)

    d = nf.merge(disc[["feature_id", "t", "p", "q"]], on="feature_id",
                 how="left")
    name_col = "best_name" if "best_name" in d else None
    if name_col is None:
        sys.exit("node_features.csv has no best_name column -- rerun "
                 "annotate_to_graph.py with --db")
    d["name"] = d[name_col].apply(tidy)
    d["is_lipid"] = d["name"].apply(lambda n: is_lipid(n, a.min_chain))
    if a.seed_list:
        sl = pd.read_csv(a.seed_list)
        keep = set(sl.feature_id.astype(str))
        d["seed"] = d.feature_id.astype(str).isin(keep)
        if "annotation" in sl:
            nm = dict(zip(sl.feature_id.astype(str),
                          sl.annotation.apply(tidy)))
            d["name"] = d.apply(
                lambda r: nm.get(str(r.feature_id), r["name"]), axis=1)
        print(f"seeding from {int(d.seed.sum())}/{len(sl)} curated features")
    else:
        d["seed"] = d["is_lipid"] if a.seed_annotations == "lipids" \
            else d["name"].ne("")
        print(f"seeding from {int(d.seed.sum())} annotated features "
              f"(pass --seed-list for a curated set)")

    info = d.set_index("feature_id")[["mz", "rt", "t", "p", "q", "name",
                                      "seed"]].to_dict("index")
    rank = disc.set_index("feature_id").p.rank(pct=True) * 100

    rows = []
    for s, tgt, tr in zip(md.source, md.target, md.transformation):
        for u, v in ((s, tgt), (tgt, s)):
            if u not in info or v not in info:
                continue
            A, B = info[u], info[v]          # A proposes an identity for B
            if not A["seed"] or B["seed"] or B["name"]:
                continue
            if not isinstance(A["name"], str) or not A["name"]:
                continue
            if not np.isfinite(B["t"]) or abs(B["t"]) < a.min_abs_t:
                continue
            if a.same_direction and np.sign(B["t"]) != np.sign(A["t"]):
                continue

            if abs(B["rt"] - A["rt"]) > a.max_delta_rt:
                continue                      # too far apart to be one step
            heavier = B["mz"] > A["mz"]
            exp = RT_RP.get(tr, 0)
            if a.mode == "hilic":
                exp = -exp
            if not heavier:
                exp = -exp
            obs = np.sign(B["rt"] - A["rt"]) if abs(B["rt"] - A["rt"]) > 1 else 0
            rt_ok = ("consistent" if exp == 0 or obs == 0 or exp == obs
                     else "INCONSISTENT")

            rows.append({
                "unknown_id": v, "mz": B["mz"], "rt": B["rt"],
                "t": B["t"], "q": B["q"],
                "rank_pct": rank.get(v, np.nan),
                "proposed": f"{A['name']} {'+' if heavier else '−'} {tr}",
                "neighbor_id": u, "neighbor": A["name"],
                "neighbor_t": A["t"], "transformation": tr,
                "delta_mz": round(B["mz"] - A["mz"], 4),
                "delta_rt": round(B["rt"] - A["rt"], 1),
                "rt_check": rt_ok,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        sys.exit("no candidates -- lower --min-abs-t or use "
                 "--seed-annotations any")

    # one row per unknown: keep its strongest-associated neighbor, but note
    # how many independent annotated neighbors point at it
    out["n_neighbors"] = out.groupby("unknown_id").unknown_id.transform("size")
    out["consistent_neighbors"] = out.groupby("unknown_id").rt_check.transform(
        lambda s: (s == "consistent").sum())
    best = (out.sort_values(["rt_check", "neighbor_t"],
                            ascending=[True, True])
            .drop_duplicates("unknown_id"))
    best = best.sort_values("t")

    print(f"{len(best)} unannotated features proposed from "
          f"{out.neighbor_id.nunique()} annotated seeds")
    print(f"  retention-time consistent: "
          f"{(best.rt_check == 'consistent').sum()}/{len(best)}")
    print(f"  in the top 5% of the association ranking: "
          f"{(best.rank_pct <= 5).sum()}\n")

    show = best.head(20)[["unknown_id", "mz", "rt", "t", "rank_pct",
                          "proposed", "delta_rt", "rt_check", "n_neighbors"]]
    print(show.to_string(index=False))
    best.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    print("\nThese are hypotheses, not identifications. Confirm by MS/MS: "
          "acquire the precursor and compare fragmentation with the proposed "
          "neighbor, which should share head-group fragments.")


if __name__ == "__main__":
    main()
