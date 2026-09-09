#!/usr/bin/env python3
"""
edge_fdr.py — how many mass-difference edges are coincidence?

An edge is drawn when a mass gap falls within 5 ppm of a transformation mass.
With thousands of features, some gaps land there by accident. Reporting an
edge count without an estimate of that accidental rate leaves the central
quantity of the method unquantified.

The estimate uses decoy transformations. Each real mass is displaced by an
offset that no chemistry produces, for example +0.35 Da, which is far outside
the 5 ppm window yet close enough that the local density of the mass
distribution is unchanged. Edges drawn on a decoy are coincidental by
construction, so

    FDR(tau)  =  mean decoy edges  /  observed edges

Reported per transformation, since a small mass such as desaturation at
2.0157 Da sits in a denser part of the pairwise gap distribution than
glucuronidation at 176.0321 Da and accumulates coincidences at a different
rate.

A degree-preserving null is unsuitable here. The question is not whether the
topology is unusual but whether an individual edge reflects chemistry, and
that depends on the mass axis rather than on connectivity.

Usage
-----
python edge_fdr.py --data out/c18_neg --features data/c18_neg/features.csv \\
    --out edge_fdr_c18.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

DEFAULT_DECOYS = (0.35, -0.35, 0.47, -0.47, 0.61, -0.61, 0.73, -0.73)


def count_edges(mz_sorted, delta, ppm, rt=None, rt_tol=None):
    """Pairs whose gap matches delta within tolerance, by binary search."""
    d = abs(delta)
    target = mz_sorted + d
    tol = ppm * 1e-6 * target
    lo = np.searchsorted(mz_sorted, target - tol, "left")
    hi = np.searchsorted(mz_sorted, target + tol, "right")
    if rt is None:
        return int(np.maximum(hi - lo, 0).sum())
    n = 0
    for i in range(len(mz_sorted)):
        for j in range(lo[i], hi[i]):
            if i != j and abs(rt[j] - rt[i]) <= rt_tol:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="directory holding node_features.csv and edges.csv")
    ap.add_argument("--features",
                    help="feature table, used only if node_features lacks m/z")
    ap.add_argument("--ppm", type=float, default=5.0)
    ap.add_argument("--rt-tol", type=float, default=None,
                    help="seconds; apply the same retention constraint the "
                         "graph used, if any")
    ap.add_argument("--decoys", type=float, nargs="*",
                    default=list(DEFAULT_DECOYS))
    ap.add_argument("--src", default=".",
                    help="directory containing annotate_to_graph.py")
    ap.add_argument("--out", default="edge_fdr.csv")
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    try:
        from annotate_to_graph import TRANSFORMATIONS as T
    except ImportError as e:
        sys.exit(f"could not import annotate_to_graph from {a.src}: {e}")

    nf = pd.read_csv(f"{a.data}/node_features.csv")
    if "mz" not in nf and a.features:
        nf = pd.read_csv(a.features)
    mz = np.sort(nf.mz.to_numpy(dtype=float))
    rt = nf.sort_values("mz").rt.to_numpy(dtype=float) if a.rt_tol else None
    print(f"{len(mz)} features, m/z {mz.min():.1f} to {mz.max():.1f}")

    ed = pd.read_csv(f"{a.data}/edges.csv")
    md = ed[ed.edge_type == "mass_difference"] if "edge_type" in ed else ed
    obs_by_t = md.transformation.value_counts().to_dict() \
        if "transformation" in md else {}

    rows = []
    for name, delta in sorted(T.items(), key=lambda kv: abs(kv[1])):
        obs = count_edges(mz, delta, a.ppm, rt, a.rt_tol)
        dec = [count_edges(mz, delta + off, a.ppm, rt, a.rt_tol)
               for off in a.decoys]
        m, s = float(np.mean(dec)), float(np.std(dec))
        rows.append({
            "transformation": name,
            "delta": delta,
            "observed": obs,
            "in_graph": int(obs_by_t.get(name, 0)),
            "decoy_mean": round(m, 1),
            "decoy_sd": round(s, 1),
            "fdr": round(m / obs, 4) if obs else np.nan,
            "excess": obs - m,
        })

    d = pd.DataFrame(rows)
    tot_o, tot_d = d.observed.sum(), d.decoy_mean.sum()
    d.to_csv(a.out, index=False)

    print(f"\n{'transformation':<18}{'Δm':>10}{'observed':>10}"
          f"{'decoy':>9}{'FDR':>8}")
    print("-" * 55)
    for _, r in d.sort_values("fdr").iterrows():
        print(f"{r.transformation:<18}{r.delta:>10.4f}{r.observed:>10,}"
              f"{r.decoy_mean:>9,.0f}{r.fdr:>8.2f}")
    print("-" * 55)
    print(f"{'TOTAL':<18}{'':>10}{tot_o:>10,.0f}{tot_d:>9,.0f}"
          f"{tot_d/tot_o:>8.2f}")
    print(f"\nAbout {tot_o - tot_d:,.0f} edges ({(1-tot_d/tot_o)*100:.0f}%) "
          f"exceed what coincidence produces.")
    print(f"wrote {a.out}")
    print("\nReading this table. A transformation with FDR near 1 contributes "
          "little beyond chance and is a candidate for removal. A small Δm "
          "sits in a denser region of the gap distribution and will show a "
          "higher FDR than a large one at equal tolerance.")


if __name__ == "__main__":
    main()
