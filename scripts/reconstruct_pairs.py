#!/usr/bin/env python3
"""
reconstruct_pairs.py — recover matched case-control sets from ST002773 metadata

The deposit has no matched-set ID, but pairs were matched on age and appear to be
block-randomized within batch (a case and its control injected close together).
This solves an optimal assignment within each batch:

    cost(case i, control j) = AGE_WEIGHT * |age_i - age_j| + |injection_i - injection_j|

Pairs with an age difference above --max-age-diff are left unmatched rather than
forced, so you can report exactly how much of the design was recovered.

Usage
-----
python reconstruct_pairs.py data/hilic_pos/metadata_aligned.csv \
    --out data/hilic_pos/metadata_paired.csv

Outputs a matched_set column (NaN where unmatched) plus a diagnostic report.
"""

import argparse
import re
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

AGE_WEIGHT = 1000.0     # age dominates; injection order only breaks ties


def injection_index(name):
    """Trailing numeric field of the raw file name = position in the run."""
    m = re.findall(r"(\d+)", str(name))
    return int(m[-1]) if m else np.nan


def match_batch(sub, max_age_diff, use_injection=True):
    cases = sub[sub.label == "case"].reset_index(drop=True)
    ctrls = sub[sub.label == "control"].reset_index(drop=True)
    if len(cases) == 0 or len(ctrls) == 0:
        return []

    age_d = np.abs(cases.AGE.to_numpy()[:, None] - ctrls.AGE.to_numpy()[None, :])
    cost = AGE_WEIGHT * age_d
    if use_injection:
        inj_d = np.abs(cases.inj.to_numpy()[:, None] - ctrls.inj.to_numpy()[None, :])
        cost = cost + np.nan_to_num(inj_d, nan=0.0)

    # forbid pairs beyond the age tolerance
    cost = np.where(age_d > max_age_diff, 1e9, cost)

    ri, ci = linear_sum_assignment(cost)
    out = []
    for a, b in zip(ri, ci):
        if cost[a, b] >= 1e9:
            continue
        out.append((cases.sample_id[a], ctrls.sample_id[b],
                    age_d[a, b],
                    abs(cases.inj[a] - ctrls.inj[b]) if use_injection else np.nan))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata")
    ap.add_argument("--out", default="metadata_paired.csv")
    ap.add_argument("--label-col", default="LungCancer")
    ap.add_argument("--age-col", default="AGE")
    ap.add_argument("--batch-col", default="Batch")
    ap.add_argument("--rawfile-col", default="RAW_FILE_NAME")
    ap.add_argument("--max-age-diff", type=float, default=2.0,
                    help="years; pairs beyond this are left unmatched")
    ap.add_argument("--no-injection", action="store_true",
                    help="match on age only, ignore run order")
    args = ap.parse_args()

    m = pd.read_csv(args.metadata)
    m = m.rename(columns={args.label_col: "label", args.age_col: "AGE",
                          args.batch_col: "Batch"})
    m["label"] = m.label.astype(str).str.strip().str.lower()
    m = m[m.label.isin(["case", "control"])].copy()
    m["inj"] = m[args.rawfile_col].map(injection_index) \
        if args.rawfile_col in m else np.nan

    print(f"{len(m)} labeled samples | "
          f"{(m.label=='case').sum()} case / {(m.label=='control').sum()} control")
    print(f"{m.Batch.nunique()} batches | "
          f"age {m.AGE.min():.0f}-{m.AGE.max():.0f}")

    # is the design plausibly balanced within batch?
    bal = m.groupby(["Batch", "label"]).size().unstack(fill_value=0)
    imbal = (bal.get("case", 0) - bal.get("control", 0)).abs()
    print(f"per-batch case/control imbalance: median {imbal.median():.0f}, "
          f"max {imbal.max()}")

    pairs, rows = [], []
    for b, sub in m.groupby("Batch"):
        for c, k, ad, idd in match_batch(sub, args.max_age_diff,
                                         not args.no_injection):
            pairs.append((c, k))
            rows.append({"batch": b, "case": c, "control": k,
                         "age_diff": ad, "inj_diff": idd})

    rep = pd.DataFrame(rows)
    n_pairs = len(rep)
    print(f"\nrecovered {n_pairs} pairs "
          f"({2*n_pairs}/{len(m)} samples = {2*n_pairs/len(m):.1%})")
    if n_pairs:
        print(f"  age diff: exact {int((rep.age_diff==0).sum())} "
              f"({(rep.age_diff==0).mean():.1%}), "
              f"mean {rep.age_diff.mean():.2f} yr")
        if rep.inj_diff.notna().any():
            print(f"  injection gap: median {rep.inj_diff.median():.0f}, "
                  f"within 20 slots {(rep.inj_diff<=20).mean():.1%}")
            print("  -> a tight injection gap supports block randomisation "
                  "(and the batch-cancellation argument)")

    setid = {}
    for i, (c, k) in enumerate(pairs):
        setid[c] = f"P{i:04d}"
        setid[k] = f"P{i:04d}"
    m["matched_set"] = m.sample_id.map(setid)

    m.to_csv(args.out, index=False)
    if n_pairs:
        rep.to_csv(args.out.replace(".csv", "_pairs.csv"), index=False)
    print(f"\nwrote {args.out}")
    print(f"  matched_set populated for {m.matched_set.notna().sum()} samples")
    print(f"  unmatched: {m.matched_set.isna().sum()} "
          f"(exclude these from the conditional loss)")

    print("\nCAVEAT: these sets are inferred, not the study's true matching. "
          "Report them as reconstructed, run the unmatched model as a "
          "sensitivity analysis, and request the real IDs from the authors.")


if __name__ == "__main__":
    main()
