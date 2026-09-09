#!/usr/bin/env python3
"""
run_me8.py — apply the pipeline to a second published module

Demonstrating a method on one module leaves generality untested. Rahman et
al. report a second, ME8, of 440 features associated with lung
adenocarcinoma at FDR 0.196. It differs from ME16 in three ways that make it
the harder case, and therefore the more informative one.

It is predominantly positive mode, 399 features against 40, which reverses
the balance of ME16. Only 17 of 440 carry an annotation, against 27 of 121.
Most importantly the odds ratios run in both directions, whereas every ME16
feature was reported below unity. A sign test is the correct instrument only
when the published direction is uniform, so agreement is assessed here by
concordance between the published odds ratio and the within-pair statistic,
feature by feature, against a permutation null.

Usage
-----
python run_me8.py --xlsx raw/ijc34929-sup-0001-tables.xlsx \\
    --disc-pos disc/hilic_pos --disc-neg disc/c18_neg --outdir me8
"""

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats


def extract(xlsx, out):
    x = pd.ExcelFile(xlsx)
    sheet = next((s for s in x.sheet_names
                  if re.search(r"Table\s*2", s, re.I)), None)
    if sheet is None:
        sys.exit(f"no Supplementary Table 2 among {x.sheet_names}")
    d = pd.read_excel(xlsx, sheet_name=sheet, skiprows=1)
    # the published sheets carry their footnotes inside the m.z column, so
    # dropping missing values is not enough and rows are kept only when m.z
    # parses as a number
    d = d[pd.to_numeric(d["m.z"], errors="coerce").notna()].reset_index(
        drop=True)
    orc = next((c for c in d.columns if c.startswith("OR")), None)
    d = d.rename(columns={"m.z": "mz_pub", "rt": "rt_pub",
                          "Annotation": "annotation", orc: "OR_pub"})
    d["or"] = d.OR_pub.map(
        lambda v: float(m.group(1))
        if (m := re.match(r"\s*([\d.]+)", str(v))) else np.nan)
    d.to_csv(out, index=False)
    print(f"ME8: {len(d)} features from '{sheet}'")
    print(f"  modes: {d['mode'].value_counts().to_dict()}")
    print(f"  annotated: {int(d.annotation.notna().sum())}")
    print(f"  odds ratio below unity: {int((d['or'] < 1).sum())}, "
          f"above: {int((d['or'] > 1).sum())}")
    return d


def match(mod, disc_dir, mode, ppm, rt_tol):
    f = f"{disc_dir}/features.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    sub = mod[mod["mode"].astype(str).str.lower()
              .str.startswith(mode[:3])].copy()
    for c in ("mz_pub", "rt_pub"):
        if c in sub:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=["mz_pub"])
    d = d.copy()
    for c in ("mz", "rt", "t", "p", "q"):
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["mz"])
    rank = d.p.rank(pct=True) * 100
    rows = []
    for _, r in sub.iterrows():
        c = d[(d.mz - r.mz_pub).abs() < ppm * 1e-6 * r.mz_pub]
        if rt_tol is not None and "rt" in d:
            c = c[(c.rt - r.rt_pub).abs() < rt_tol]
        if not len(c):
            continue
        i = (c.mz - r.mz_pub).abs().idxmin()
        rows.append({"feature_id": d.feature_id[i], "mz": d.mz[i],
                     "rt": d.rt[i], "t": d.t[i], "p": d.p[i], "q": d.q[i],
                     "rank_pct": rank[i], "mz_pub": r.mz_pub,
                     "rt_pub": r.rt_pub, "annotation": r.annotation,
                     "OR_pub": r.OR_pub, "or": r["or"]})
    return pd.DataFrame(rows)


def concordance(m, n_perm=20000, seed=0):
    """
    Agreement between the published direction and ours, tested by
    permutation. A published odds ratio below unity predicts a negative
    statistic here, so concordance is the fraction of features whose signs
    agree. The null shuffles our signs, which preserves the marginal
    imbalance of the published directions and asks only whether the pairing
    carries information.
    """
    d = m.dropna(subset=["or", "t"])
    if len(d) < 5:
        return None
    pub = np.sign(np.log(d["or"].to_numpy()))
    obs = np.sign(d.t.to_numpy())
    agree = int((pub == obs).sum())
    rng = np.random.default_rng(seed)
    null = np.array([(pub == rng.permutation(obs)).sum()
                     for _ in range(n_perm)])
    p = max(float((null >= agree).mean()), 1 / n_perm)
    r, p_r = stats.spearmanr(np.log(d["or"]), d.t)
    return {"n": len(d), "agree": agree, "frac": agree / len(d),
            "null_mean": float(null.mean()), "p_concordance": p,
            "spearman_rho": float(r), "p_spearman": float(p_r)}


def enrichment(m, disc_dir, n_perm=10000, seed=0):
    d = pd.read_csv(f"{disc_dir}/features.csv")
    obs = float(np.median(m.rank_pct))
    rng = np.random.default_rng(seed)
    rank = (d.p.rank(pct=True) * 100).to_numpy()
    null = np.array([np.median(rng.choice(rank, len(m), replace=False))
                     for _ in range(n_perm)])
    return obs, max(float((null <= obs).mean()), 1 / n_perm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx",
                    help="the published supporting-information workbook")
    ap.add_argument("--csv",
                    help="a module table already extracted, with columns "
                         "mz_pub, rt_pub, mode, annotation, OR_pub. Use this "
                         "when the workbook is unavailable.")
    ap.add_argument("--disc-pos", default="disc/hilic_pos")
    ap.add_argument("--disc-neg", default="disc/c18_neg")
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--rt-tol", type=float, default=30.0)
    ap.add_argument("--no-rt", action="store_true",
                    help="match on mass only, since the deposited retention "
                         "times may sit on a different scale")
    ap.add_argument("--outdir", default="me8")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    if a.csv:
        mod = pd.read_csv(a.csv)
        # column names differ between the workbook export and any table
        # prepared by hand, so they are resolved rather than assumed
        ren = {}
        for want, cands in (("mz_pub", ("mz_pub", "m.z", "mz", "m/z")),
                            ("rt_pub", ("rt_pub", "rt", "RT", "time")),
                            ("mode", ("mode", "Mode", "ion_mode")),
                            ("annotation", ("annotation", "Annotation",
                                            "name", "best_name"))):
            if want not in mod.columns:
                hit = next((c for c in cands if c in mod.columns), None)
                if hit:
                    ren[hit] = want
        orc = next((c for c in mod.columns
                    if str(c).upper().startswith("OR")), None)
        if orc and orc != "OR_pub":
            ren[orc] = "OR_pub"
        mod = mod.rename(columns=ren)
        missing = [c for c in ("mz_pub", "mode") if c not in mod.columns]
        if missing:
            sys.exit(f"{a.csv} lacks {missing}. Columns present: "
                     f"{list(mod.columns)}. Download ME8_features.csv again, "
                     f"or supply --xlsx.")
        if "annotation" not in mod:
            mod["annotation"] = np.nan
        if "rt_pub" not in mod:
            mod["rt_pub"] = np.nan
        if "or" not in mod.columns:
            if "OR_pub" not in mod.columns:
                sys.exit(f"{a.csv} carries no odds ratio column, so the "
                         f"concordance test cannot run. Columns present: "
                         f"{list(mod.columns)}")
            mod["or"] = mod.OR_pub.map(
                lambda v: float(m.group(1))
                if (m := re.match(r"\s*([\d.]+)", str(v))) else np.nan)
        print(f"module: {len(mod)} features from {a.csv}")
        print(f"  modes: {mod['mode'].value_counts().to_dict()}")
        print(f"  odds ratio below unity: {int((mod['or'] < 1).sum())}, "
              f"above: {int((mod['or'] > 1).sum())}")
    elif a.xlsx:
        mod = extract(a.xlsx, f"{a.outdir}/ME8_features.csv")
    else:
        sys.exit("supply either --xlsx or --csv")
    rt_tol = None if a.no_rt else a.rt_tol

    summary = []
    for mode, disc, tag in (("positive", a.disc_pos, "hilic_pos"),
                            ("negative", a.disc_neg, "c18_neg")):
        m = match(mod, disc, mode, a.ppm, rt_tol)
        if m is None or m.empty:
            print(f"\n{tag}: no matches, check the paths")
            continue
        n_pub = int((mod["mode"].str.lower().str.startswith(mode[:3])).sum())
        m.to_csv(f"{a.outdir}/me8_{tag}.csv", index=False)

        print(f"\n=== {tag}  matched {len(m)} of {n_pub} published features")
        med, p_enr = enrichment(m, disc)
        print(f"  median rank {med:.1f}% of features, enrichment "
              f"p = {p_enr:.4f}")
        print(f"  mean t {m.t.mean():+.3f}, negative {int((m.t<0).sum())} "
              f"of {len(m)}")
        c = concordance(m)
        if c:
            print(f"  direction concordance {c['agree']}/{c['n']} "
                  f"({c['frac']:.0%}), null {c['null_mean']:.1f}, "
                  f"p = {c['p_concordance']:.4f}")
            print(f"  log(OR) against t, Spearman rho = "
                  f"{c['spearman_rho']:+.3f}, p = {c['p_spearman']:.4f}")
        else:
            print("  too few features carry an odds ratio for a "
                  "concordance test")
        summary.append({"column": tag, "published": n_pub, "matched": len(m),
                        "median_rank_pct": round(med, 2),
                        "p_enrichment": p_enr,
                        "mean_t": round(float(m.t.mean()), 3),
                        "n_negative": int((m.t < 0).sum()),
                        **({k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in c.items()} if c else {})})

    if summary:
        pd.DataFrame(summary).to_csv(f"{a.outdir}/me8_summary.csv",
                                     index=False)
        print(f"\nwrote {a.outdir}/me8_summary.csv")

    print("""
How to read this. ME8 is the harder case. Its published association is
weaker, at FDR 0.196 against 0.028 for ME16, its odds ratios run in both
directions, and only 4 percent of its features carry an annotation. A
concordance near chance would show that the recovery of ME16 depended on the
strength and uniformity of that module, which is worth reporting either way.
Run figure_module_full.py on the matched file next to see whether the
matched features connect on the graph.""")


if __name__ == "__main__":
    main()
