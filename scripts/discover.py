#!/usr/bin/env python3
"""
discover.py — what the graph learned: features, modules, and interactions

Prediction is weak in pre-diagnostic metabolomics, but discovery is testable.
This script asks four questions, each with an exact permutation null.

THE NULL
--------
In a matched pair the null hypothesis is that case/control orientation is
arbitrary, so permuting labels is exactly flipping the sign of the pair
difference D. Under a sign flip, D^2 is unchanged, so the variance term is
invariant and thousands of permutations reduce to one matrix product.
This is an exact test, not an asymptotic approximation.

QUESTIONS
---------
1. FEATURES    which individual features differ within pairs (sign-flip FDR)
2. MODULES     which connected subgraphs of the mass-difference network are
               enriched for signal -- the chemistry-derived analogue of a
               WGCNA module, and the part no prior work has tested
3. INTERACTIONS which connected feature PAIRS carry joint signal beyond either
               alone (2-df Wald statistic minus the better marginal)
4. CHEMISTRY   which biotransformation types are over-represented among the
               top features, using a DEGREE-MATCHED null so hub features
               don't manufacture enrichment

Plus: recovery of a user-supplied panel (e.g. the LPE list) and, optionally,
agreement with GNN attention.

Usage
-----
python discover.py --data out/hilic_pos --intensities data/hilic_pos/features.csv \
    --metadata data/hilic_pos/metadata_paired.csv --outdir disc/hilic_pos \
    --panel lpe_panel.csv --attention v5/gin/attention_hilic_pos.csv
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

from common import normalize, build_D, signflip_t


# ----------------------------------------------------------------- data




def empirical_fdr(obs_t, T):
    """BH on p-values from the pooled permutation null."""
    null = np.sort(np.abs(T).ravel())
    p = 1.0 - np.searchsorted(null, np.abs(obs_t), "left") / len(null)
    p = np.clip(p, 1.0 / len(null), 1.0)
    o = np.argsort(p)
    q = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * m / (m - rank))
        q[i] = prev
    return p, q


def louvain_modules(edges_ij, n, resolution=1.0, seed=0, min_size=3):
    """
    Community detection instead of connected components.

    Connected components of a mass-difference graph percolate: one giant
    component of thousands of features plus size-3 fragments. Neither has
    power -- the giant blob averages signal away, the fragments are noise.
    Louvain partitions the giant component into modules on the scale WGCNA
    produces (tens to hundreds of features), which is where signal is
    detectable. Higher --resolution gives smaller modules.
    """
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges_ij)
    comm = nx.community.louvain_communities(G, resolution=resolution,
                                            seed=seed)
    lab = -np.ones(n, dtype=int)
    for k, c in enumerate(comm):
        for v in c:
            lab[v] = k
    return lab


def components(edges_ij, n):
    """Connected components via union-find."""
    par = np.arange(n)

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in edges_ij:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def module_test(obs_t, T, comp, min_size=3):
    """Module statistic = mean t^2 over members; same sign-flip null."""
    rows = []
    for c in np.unique(comp):
        idx = np.where(comp == c)[0]
        if len(idx) < min_size:
            continue
        obs = float((obs_t[idx] ** 2).mean())
        null = (T[:, idx] ** 2).mean(1)
        p = float((null >= obs).mean())
        rows.append({"module": int(c), "size": len(idx),
                     "stat": obs, "p": max(p, 1.0 / len(null)),
                     "mean_t": float(obs_t[idx].mean()),
                     "n_pos": int((obs_t[idx] > 0).sum())})
    if not rows:
        return pd.DataFrame(columns=["module", "size", "stat", "p",
                                     "mean_t", "n_pos", "q"])
    df = pd.DataFrame(rows).sort_values("p")
    if len(df):
        m = len(df)
        df["q"] = np.minimum.accumulate(
            (df.p.values * m / np.arange(1, m + 1))[::-1])[::-1]
    return df


def interaction_test(D, obs_t, T, src, dst, n_perm_keep=500, top_only=None):
    """
    For each edge, the 2-df Wald statistic for the connected pair, minus the
    better of the two marginals. Positive synergy means the pair carries joint
    information neither feature has alone.

    NOTE ON SELECTION: edges must NOT be pre-filtered by observed |t|. Doing so
    compares statistics selected for being extreme against a null that went
    through no selection, which makes almost every edge look significant. All
    edges are tested; multiplicity is handled by BH afterwards.
    """
    n = D.shape[0]
    if top_only is not None:
        raise ValueError("top_only is unsafe: selecting edges on observed |t| "
                         "biases the null. Test all edges instead.")
    if len(src) == 0:
        return pd.DataFrame()

    Ds = D[:, src]; Dd = D[:, dst]
    r = ((Ds * Dd).mean(0) - Ds.mean(0) * Dd.mean(0)) / \
        (Ds.std(0) * Dd.std(0) + 1e-12)
    r = np.clip(r, -0.99, 0.99)
    ti, tj = obs_t[src], obs_t[dst]
    joint = (ti ** 2 + tj ** 2 - 2 * r * ti * tj) / (1 - r ** 2)
    syn = joint - np.maximum(ti ** 2, tj ** 2)

    Tn = T[:n_perm_keep]
    tin, tjn = Tn[:, src], Tn[:, dst]
    jn = (tin ** 2 + tjn ** 2 - 2 * r[None, :] * tin * tjn) / (1 - r[None, :] ** 2)
    sn = jn - np.maximum(tin ** 2, tjn ** 2)
    p = (sn >= syn[None, :]).mean(0)

    out = pd.DataFrame({"i": src, "j": dst, "r": r, "t_i": ti, "t_j": tj,
                        "joint": joint, "synergy": syn,
                        "p": np.maximum(p, 1.0 / n_perm_keep)})
    key = np.minimum(out.i, out.j).astype(str) + "_" + \
        np.maximum(out.i, out.j).astype(str)
    out = out.loc[~key.duplicated()].sort_values("p")
    m = len(out)
    out["q"] = np.minimum.accumulate(
        (out.p.values * m / np.arange(1, m + 1))[::-1])[::-1]
    return out.sort_values("synergy", ascending=False)


def transformation_enrichment(ed, ids, top_idx, degree, n_perm=2000, seed=0):
    """
    Which transformation types are over-represented on edges touching the top
    features? Null draws feature sets matched on degree decile, because hub
    features touch more edges of every type and would fake enrichment.
    """
    md = ed[ed.edge_type == "mass_difference"] if "edge_type" in ed else ed
    if md.empty or "transformation" not in md.columns:
        return pd.DataFrame()
    pos = {f: i for i, f in enumerate(ids)}
    si = md.source.map(pos).to_numpy(); di = md.target.map(pos).to_numpy()
    ok = ~(pd.isna(si) | pd.isna(di))
    si, di = si[ok].astype(int), di[ok].astype(int)
    tr = md.transformation.to_numpy()[ok]

    def counts(sel):
        mask = np.zeros(len(ids), bool); mask[sel] = True
        hit = mask[si] | mask[di]
        return pd.Series(tr[hit]).value_counts()

    obs = counts(top_idx)
    dec = pd.qcut(pd.Series(degree).rank(method="first"), 10,
                  labels=False).to_numpy()
    by_dec = {d: np.where(dec == d)[0] for d in np.unique(dec)}
    want = pd.Series(dec[top_idx]).value_counts()

    rng = np.random.default_rng(seed)
    null = {t: [] for t in obs.index}
    for _ in range(n_perm):
        sel = np.concatenate([rng.choice(by_dec[d], k, replace=False)
                              for d, k in want.items()])
        c = counts(sel)
        for t in obs.index:
            null[t].append(c.get(t, 0))
    rows = []
    for t in obs.index:
        nl = np.array(null[t])
        rows.append({"transformation": t, "observed": int(obs[t]),
                     "expected": float(nl.mean()),
                     "fold": obs[t] / max(nl.mean(), 1e-9),
                     "p": float(max((nl >= obs[t]).mean(), 1.0 / n_perm))})
    return pd.DataFrame(rows).sort_values("p")


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--intensities", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--outdir", default="disc")
    ap.add_argument("--panel", help="CSV with feature_id column to check recovery")
    ap.add_argument("--attention", help="attention_*.csv from a model run")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--top-k", type=int, default=200,
                    help="features treated as 'top' for enrichment")
    ap.add_argument("--module-edges", choices=["massdiff", "both"],
                    default="massdiff")
    ap.add_argument("--modules", choices=["louvain", "components"],
                    default="louvain",
                    help="louvain avoids the giant-component problem")
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="higher = smaller modules")
    ap.add_argument("--min-module", type=int, default=10,
                    help="modules smaller than this are skipped (no power)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    D, nf, ids = build_D(args.data, args.intensities, args.metadata)
    ed = pd.read_csv(f"{args.data}/edges.csv")
    print(f"{D.shape[0]} pairs x {D.shape[1]} features")

    # ---- 1. features ------------------------------------------------------
    obs_t, T = signflip_t(D, args.n_perm)
    p, q = empirical_fdr(obs_t, T)
    feat = pd.DataFrame({"feature_id": ids, "mz": nf.mz, "rt": nf.rt,
                         "t": obs_t, "p": p, "q": q,
                         "main_class": nf.get("main_class", ""),
                         "best_name": nf.get("best_name", "")})
    feat = feat.sort_values("p")
    feat.to_csv(f"{args.outdir}/features.csv", index=False)
    print(f"\n1. FEATURES   q<0.20: {(q < 0.20).sum()}   q<0.05: {(q < 0.05).sum()}")
    print(f"   strongest |t| = {np.abs(obs_t).max():.2f} "
          f"(null max {np.abs(T).max():.2f})")
    print(feat.head(5)[["feature_id", "mz", "rt", "t", "q", "best_name"]]
          .to_string(index=False))

    # ---- 2. modules -------------------------------------------------------
    sub = ed if args.module_edges == "both" else ed[ed.edge_type == "mass_difference"]
    pos = {f: i for i, f in enumerate(ids)}
    si = sub.source.map(pos).to_numpy(); di = sub.target.map(pos).to_numpy()
    ok = ~(pd.isna(si) | pd.isna(di))
    si, di = si[ok].astype(int), di[ok].astype(int)
    eij = list(zip(si, di))
    if args.modules == "louvain":
        comp = louvain_modules(eij, len(ids), args.resolution)
    else:
        comp = components(eij, len(ids))
    mod = module_test(obs_t, T, comp, min_size=args.min_module)
    mod.to_csv(f"{args.outdir}/modules.csv", index=False)
    sizes = pd.Series(comp[comp >= 0]).value_counts()
    big = sizes[sizes >= args.min_module]
    if len(big):
        print(f"\n2. MODULES ({args.modules})  {len(big)} of size>="
              f"{args.min_module} | sizes: median {int(big.median())}, "
              f"max {int(big.max())}")
    else:
        largest = int(sizes.max()) if len(sizes) else 0
        print(f"\n2. MODULES ({args.modules})  none reached size "
              f"{args.min_module} (largest {largest}). Lower --min-module or "
              f"--resolution on a sparse graph.")
    if len(mod):
        print(f"   significant: p<0.05 {(mod.p < 0.05).sum()} "
              f"(chance expects {0.05*len(mod):.1f}), "
              f"q<0.20 {(mod.q < 0.20).sum()}")
        print(mod.head(5).to_string(index=False))

    # ---- 3. interactions --------------------------------------------------
    inter = interaction_test(D, obs_t, T, si, di)
    if len(inter):
        inter["feature_i"] = ids[inter.i.to_numpy()]
        inter["feature_j"] = ids[inter.j.to_numpy()]
        inter.to_csv(f"{args.outdir}/interactions.csv", index=False)
        exp = 0.05 * len(inter)
        print(f"\n3. INTERACTIONS  {len(inter)} unique connected pairs tested")
        print(f"   p<0.05: {(inter.p < 0.05).sum()} (chance expects {exp:.0f})"
              f"   q<0.20: {(inter.q < 0.20).sum()}")
        print(inter.head(5)[["feature_i", "feature_j", "r", "t_i", "t_j",
                             "synergy", "p", "q"]].to_string(index=False))

    # ---- 4. chemistry -----------------------------------------------------
    deg = np.zeros(len(ids))
    np.add.at(deg, si, 1); np.add.at(deg, di, 1)
    top_idx = np.argsort(-np.abs(obs_t))[:args.top_k]
    enr = transformation_enrichment(ed, ids, top_idx, deg)
    if len(enr):
        enr.to_csv(f"{args.outdir}/chemistry.csv", index=False)
        print(f"\n4. CHEMISTRY  degree-matched enrichment among top "
              f"{args.top_k} features")
        print(enr.head(6).to_string(index=False))

    # ---- panel recovery ---------------------------------------------------
    if args.panel:
        pan = pd.read_csv(args.panel)
        # feature_ids are positional PER COLUMN, so a panel spanning both
        # columns must be filtered to this one or IDs collide across datasets
        if "column" in pan.columns:
            tag = os.path.basename(args.data.rstrip("/"))
            before = len(pan)
            pan = pan[pan.column.astype(str) == tag]
            print(f"\n[panel filtered to column '{tag}': "
                  f"{len(pan)}/{before} entries]")
            if pan.empty:
                print("  no panel entries for this column -- check the "
                      "'column' values match the --data folder name")
        pids = set(pan.feature_id) if "feature_id" in pan else set()
        if pids:
            r = feat.reset_index(drop=True)
            r["rank_pct"] = (r.index + 1) / len(r) * 100
            hit = r[r.feature_id.isin(pids)]
            print(f"\nPANEL RECOVERY ({len(hit)} of {len(pids)} present)")
            print(hit[["feature_id", "mz", "rt", "t", "q", "rank_pct"]]
                  .to_string(index=False))
            if len(hit):
                med = hit.rank_pct.median()
                # is the panel enriched toward the top?
                rng = np.random.default_rng(0)
                nullmed = [np.median(rng.choice(r.rank_pct, len(hit)))
                           for _ in range(5000)]
                pv = float(np.mean(np.array(nullmed) <= med))
                print(f"  median rank {med:.1f}% of features, "
                      f"enrichment p = {max(pv, 2e-4):.4f}")

    # ---- attention agreement ---------------------------------------------
    if args.attention and os.path.exists(args.attention):
        att = pd.read_csv(args.attention)
        j = feat.merge(att, on="feature_id")
        if len(j) and j.attention.std() > 0:
            rk = lambda v: pd.Series(v).rank().to_numpy()
            rho = np.corrcoef(rk(np.abs(j.t)), rk(j.attention))[0, 1]
            print(f"\nATTENTION vs sign-flip t: Spearman rho = {rho:.3f}")
            print("  high rho => the network found the same features the "
                  "statistics do (good); near zero => it found noise.")

    json.dump(vars(args), open(f"{args.outdir}/config.json", "w"), indent=2)
    print(f"\nwrote {args.outdir}/")


if __name__ == "__main__":
    main()
