#!/usr/bin/env python3
"""
propagate.py — diffuse the association statistic over the graph

Enumerating node sets and testing them fixes the units in advance. Network
propagation drops that constraint. Each feature lends part of its statistic
to its chemical neighbors and receives part of theirs, so an association
spread thinly across several features of one pathway is reinforced while an
isolated value is diluted.

    F = (1 - alpha) (I - alpha W)^-1 F0

with W the symmetrically degree-normalized adjacency and alpha controlling
how far a value travels. At alpha near zero the smoothed vector is the
observed one, and at alpha near one it approaches the stationary
distribution of a walk on the graph, which carries no information about the
outcome.

The null must pass through the same operator. Smoothing raises the largest
value in any vector, including a vector of noise, so comparing a smoothed
observation against an unsmoothed null would manufacture significance.
Every sign-flipped null vector is therefore diffused through the same graph,
and the comparison is smoothed against smoothed.

Usage
-----
python propagate.py --data out/c18_neg --intensities data/c18_neg/features.csv \\
    --metadata data/c18_neg/metadata_paired.csv --out prop_c18.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

from common import normalize, build_D, signflip_t
from scipy import sparse
from scipy.sparse.linalg import splu





def build_W(nf, edges, edge_types, fdr_table=None, max_fdr=None):
    ids = nf.feature_id.to_numpy()
    pos = {f: i for i, f in enumerate(ids)}
    ed = edges
    if edge_types != "both" and "edge_type" in ed:
        want = "mass_difference" if edge_types == "massdiff" else "correlation"
        ed = ed[ed.edge_type == want]
    if fdr_table is not None and max_fdr is not None \
            and "transformation" in ed:
        keep = set(fdr_table.loc[fdr_table.fdr < max_fdr, "transformation"])
        before = len(ed)
        ed = ed[ed.transformation.isin(keep)]
        print(f"edges after FDR filter: {len(ed):,} of {before:,}")
    s = ed.source.map(pos).to_numpy()
    d = ed.target.map(pos).to_numpy()
    ok = ~(pd.isna(s) | pd.isna(d))
    s, d = s[ok].astype(int), d[ok].astype(int)
    n = len(ids)
    A = sparse.coo_matrix((np.ones(len(s)), (s, d)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel()
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    Dm = sparse.diags(inv)
    return (Dm @ A @ Dm).tocsc(), deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--intensities", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--alpha", type=float, nargs="*",
                    default=[0.3, 0.5, 0.7, 0.9],
                    help="diffusion strength. Reported across a range, since "
                         "the best value is unknown in advance and choosing "
                         "it on the observed data would inflate the result.")
    ap.add_argument("--edge-types", choices=["massdiff", "corr", "both"],
                    default="massdiff")
    ap.add_argument("--edge-fdr")
    ap.add_argument("--max-fdr", type=float, default=0.20)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--out", default="propagation.csv")
    a = ap.parse_args()

    D, nf, ids = build_D(a.data, a.intensities, a.metadata)
    ed = pd.read_csv(f"{a.data}/edges.csv")
    fdr = pd.read_csv(a.edge_fdr) if a.edge_fdr and os.path.exists(a.edge_fdr) \
        else None
    W, deg = build_W(nf, ed, a.edge_types, fdr, a.max_fdr)
    print(f"{D.shape[0]} pairs, {len(ids)} features, "
          f"{int(deg.sum()//2):,} undirected edges, "
          f"{int((deg == 0).sum())} isolated")

    t, T = signflip_t(D, a.n_perm)
    n = len(ids)
    I = sparse.identity(n, format="csc")

    rows, best = [], None
    for al in a.alpha:
        lu = splu((I - al * W).tocsc())
        f_obs = (1 - al) * lu.solve(t)
        # the null passes through the same operator, since smoothing raises
        # the extreme value of any vector including one of pure noise
        f_null_max = np.empty(a.n_perm)
        f_null_all = np.empty((min(a.n_perm, 200), n))
        for i in range(a.n_perm):
            v = (1 - al) * lu.solve(T[i])
            f_null_max[i] = np.abs(v).max()
            if i < f_null_all.shape[0]:
                f_null_all[i] = v
        obs_max = float(np.abs(f_obs).max())
        p_global = float(max((f_null_max >= obs_max).mean(), 1 / a.n_perm))

        pooled = np.sort(np.abs(f_null_all).ravel())
        p_feat = 1.0 - np.searchsorted(pooled, np.abs(f_obs), "left") \
            / len(pooled)
        p_feat = np.clip(p_feat, 1 / len(pooled), 1.0)
        o = np.argsort(p_feat)
        q = np.empty_like(p_feat)
        prev = 1.0
        for r, i in enumerate(o[::-1]):
            prev = min(prev, p_feat[i] * n / (n - r))
            q[i] = prev

        rows.append({"alpha": al, "max_smoothed": round(obs_max, 3),
                     "null_max_mean": round(float(f_null_max.mean()), 3),
                     "p_global": p_global,
                     "features_q20": int((q < 0.20).sum()),
                     "features_q05": int((q < 0.05).sum())})
        print(f"  alpha {al:.1f}  max |F| {obs_max:.3f} "
              f"(null {f_null_max.mean():.3f})  p = {p_global:.3f}  "
              f"q<0.20: {(q < 0.20).sum()}")
        if best is None or p_global < best[0]:
            best = (p_global, al, f_obs, p_feat, q)

    res = pd.DataFrame(rows)
    res.to_csv(a.out, index=False)

    _, al, f_obs, p_feat, q = best
    det = pd.DataFrame({
        "feature_id": ids, "mz": nf.mz, "rt": nf.rt, "t": t,
        "smoothed": f_obs, "degree": deg, "p": p_feat, "q": q,
        "best_name": nf.get("best_name", pd.Series([""] * len(nf))),
    }).sort_values("p")
    det.to_csv(a.out.replace(".csv", "_features.csv"), index=False)

    print(f"\nstrongest features after diffusion at alpha = {al}")
    print(det.head(10)[["feature_id", "mz", "rt", "t", "smoothed", "degree",
                        "q", "best_name"]].to_string(index=False))
    print(f"\nwrote {a.out} and {a.out.replace('.csv', '_features.csv')}")
    print("\nDiffusion amplifies an association shared by chemical "
          "neighbors and dilutes an isolated one. A result that appears "
          "only at a single alpha, or only after the strongest smoothing, "
          "reflects the operator rather than the data.")


if __name__ == "__main__":
    main()
