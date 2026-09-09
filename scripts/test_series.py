#!/usr/bin/env python3
"""
test_series.py — test enumerated series as pre-specified node sets

find_series.py defines its node sets from masses and retention times, with
no reference to any outcome. They are therefore pre-specified in the sense
that matters, and testing them costs one test per series instead of one per
feature.

Two statistics are reported per series. The mean within-pair t measures a
shift shared by the whole set, which is what a co-regulated series would
produce. The mean of t squared measures dispersion, which detects a series
whose members move in opposite directions. Both use the same exact
sign-flip null as the feature-level analysis, so the series statistic and
its null are computed on the same permutations and remain comparable.

A series shares members with other series, since one feature can sit on
several chains, so the tests are correlated and Benjamini-Hochberg is
conservative here rather than anti-conservative.

Usage
-----
python test_series.py --series series_c18.csv --data out/c18_neg \\
    --intensities data/c18_neg/features.csv \\
    --metadata data/c18_neg/metadata_paired.csv --out series_tests_c18.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from common import normalize, build_D, signflip_t, bh






def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--intensities", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--all-series", action="store_true",
                    help="test every enumerated series, not only those "
                         "retained on retention order")
    ap.add_argument("--out", default="series_tests.csv")
    a = ap.parse_args()

    s = pd.read_csv(a.series)
    if not a.all_series and "retained" in s:
        n0 = len(s)
        s = s[s.retained]
        print(f"testing {len(s)} of {n0} series retained on retention order")

    D, _nf, ids = build_D(a.data, a.intensities, a.metadata)
    idx = {f: i for i, f in enumerate(ids)}
    print(f"{D.shape[0]} matched pairs, {D.shape[1]} features")

    t, T = signflip_t(D, a.n_perm)

    rows = []
    for _, r in s.iterrows():
        mem = [idx[f] for f in r.members.split(",") if f in idx]
        if len(mem) < a.min_size:
            continue
        mt = float(t[mem].mean())
        st = float((t[mem] ** 2).mean())
        null_mean = T[:, mem].mean(1)
        null_sq = (T[:, mem] ** 2).mean(1)
        p_shift = float(max((np.abs(null_mean) >= abs(mt)).mean(),
                            1 / a.n_perm))
        p_disp = float(max((null_sq >= st).mean(), 1 / a.n_perm))
        rows.append({
            "series_id": r.series_id, "transformation": r.transformation,
            "size": len(mem), "mean_t": round(mt, 3),
            "mean_t2": round(st, 3),
            "p_shift": p_shift, "p_dispersion": p_disp,
            "n_negative": int((t[mem] < 0).sum()),
            "rho_rt": r.get("rho_rt"),
            "annotations": r.get("annotations"),
        })

    d = pd.DataFrame(rows)
    if d.empty:
        sys.exit("no series of the required size survived")
    d["q_shift"] = bh(d.p_shift.to_numpy())
    d["q_dispersion"] = bh(d.p_dispersion.to_numpy())
    d = d.sort_values("p_shift")
    d.to_csv(a.out, index=False)

    n = len(d)
    print(f"\n{n} series tested")
    for col, lab in (("p_shift", "shared shift"),
                     ("p_dispersion", "dispersion")):
        k = int((d[col] < 0.05).sum())
        q = int((d[col.replace('p_', 'q_')] < 0.20).sum())
        print(f"  {lab:<14} {k} at p<0.05 (chance expects {0.05*n:.1f}), "
              f"{q} at q<0.20")

    print("\nstrongest series")
    cols = ["series_id", "transformation", "size", "mean_t", "n_negative",
            "p_shift", "q_shift"]
    print(d.head(10)[cols].to_string(index=False))
    print(f"\nwrote {a.out}")
    print("\nSeries overlap, because one feature can lie on several chains, "
          "so these tests are positively correlated and the "
          "Benjamini-Hochberg estimate is conservative.")


if __name__ == "__main__":
    main()
