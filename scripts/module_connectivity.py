#!/usr/bin/env python3
"""
module_connectivity.py — is a module's connectivity on the graph unusual?

Reporting that 33 of 64 module features are joined by a single reaction step
invites the question of what fraction any set of 64 features would reach.
Mass-difference graphs are dense, so a large connected fraction may say more
about the graph than about the module. The comparison below draws random
feature sets matched to the module and measures the same quantities.

Matching on mass matters. A module concentrated in a crowded region of the
mass axis has more potential partners than one spread thinly, so sets drawn
uniformly would understate the baseline. Random sets are therefore drawn
within mass deciles, preserving the module's position on that axis.

Three statistics are reported. The connected fraction is the proportion of
module features with at least one edge to another module feature. The
component count is how many separate pieces those features form. The
largest-component size is how much of the module falls into one piece. The
first is expected to be close to baseline for a dense graph. The second and
third distinguish a module that forms a few coherent structures from one
that fragments into pairs.

Usage
-----
python module_connectivity.py --data out/c18_neg --matched me16_c18.csv \\
    --label ME16 --out conn_me16_c18.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx


def stats_for(G, nodes):
    sub = G.subgraph(nodes)
    comps = [c for c in nx.connected_components(sub) if len(c) > 1]
    connected = set().union(*comps) if comps else set()
    return {
        "n": len(nodes),
        "connected": len(connected),
        "frac": len(connected) / max(len(nodes), 1),
        "components": len(comps),
        "largest": max((len(c) for c in comps), default=0),
        "largest_frac": max((len(c) for c in comps), default=0)
        / max(len(nodes), 1),
        # components per connected feature. A raw component count is
        # uninformative against a null whose sets are almost entirely
        # unconnected, since such a null has few components for the trivial
        # reason that it has few edges. Dividing by the number of connected
        # features asks the meaningful question, which is whether the
        # connectivity present assembles into large pieces or into pairs.
        "comp_per_conn": (len(comps) / len(connected)) if connected else
        np.nan,
        "mean_comp_size": (len(connected) / len(comps)) if comps else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--matched", required=True)
    ap.add_argument("--label", default="module")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--deciles", type=int, default=10,
                    help="mass strata for matching. Set to 1 to draw "
                         "uniformly, which overstates how unusual a module "
                         "in a crowded mass region appears.")
    ap.add_argument("--edge-fdr",
                    help="restrict to transformations below --max-fdr")
    ap.add_argument("--max-fdr", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="connectivity.csv")
    a = ap.parse_args()

    nf = pd.read_csv(f"{a.data}/node_features.csv")
    ed = pd.read_csv(f"{a.data}/edges.csv")
    md = ed[ed.edge_type == "mass_difference"] if "edge_type" in ed else ed
    if a.edge_fdr and os.path.exists(a.edge_fdr):
        f = pd.read_csv(a.edge_fdr)
        keep = set(f.loc[f.fdr < a.max_fdr, "transformation"])
        before = len(md)
        md = md[md.transformation.isin(keep)]
        print(f"edges after FDR filter: {len(md):,} of {before:,}")

    G = nx.Graph()
    G.add_nodes_from(nf.feature_id)
    G.add_edges_from(zip(md.source, md.target))

    m = pd.read_csv(a.matched)
    mod = [f for f in m.feature_id if f in G]
    if len(mod) < 5:
        sys.exit(f"only {len(mod)} module features found in the graph")
    obs = stats_for(G, mod)
    print(f"\n{a.label}: {obs['n']} features in the graph")
    print(f"  connected {obs['connected']} ({obs['frac']:.0%}) in "
          f"{obs['components']} components, largest {obs['largest']}, "
          f"mean component size {obs['mean_comp_size']:.1f}")

    # random sets matched on the mass distribution of the module
    mz = dict(zip(nf.feature_id, nf.mz))
    allf = np.array([f for f in nf.feature_id if f in G])
    allmz = np.array([mz[f] for f in allf])
    edges = np.quantile(allmz, np.linspace(0, 1, a.deciles + 1))
    edges[-1] += 1e-6
    bin_all = np.digitize(allmz, edges[1:-1])
    want = np.digitize(np.array([mz[f] for f in mod]), edges[1:-1])
    need = pd.Series(want).value_counts().to_dict()
    pool = {b: allf[bin_all == b] for b in need}
    for b, k in need.items():
        if len(pool[b]) < k:
            sys.exit(f"stratum {b} holds {len(pool[b])} features but the "
                     f"module needs {k}. Lower --deciles.")

    rng = np.random.default_rng(a.seed)
    null = []
    for _ in range(a.n_perm):
        draw = np.concatenate([rng.choice(pool[b], k, replace=False)
                               for b, k in need.items()])
        null.append(stats_for(G, list(draw)))
    N = pd.DataFrame(null)

    rows = []
    for key, direction, meaning in (
            ("frac", "greater", "more of the module is connected"),
            ("largest_frac", "greater",
             "more of the module falls in one piece"),
            ("mean_comp_size", "greater",
             "the connected part assembles into larger pieces"),
            ("comp_per_conn", "less",
             "the connected part fragments less"),
            ("components", "greater",
             "reported for completeness, see the note below")):
        v = obs[key]
        col = N[key].dropna()
        if not len(col) or not np.isfinite(v):
            continue
        p = ((col >= v).mean() if direction == "greater"
             else (col <= v).mean())
        p = max(float(p), 1 / a.n_perm)
        rows.append({"statistic": key, "observed": round(v, 4),
                     "null_mean": round(float(col.mean()), 4),
                     "null_sd": round(float(col.std()), 4),
                     "p": p, "interpretation": meaning})
    r = pd.DataFrame(rows)
    r.insert(0, "module", a.label)
    r.to_csv(a.out, index=False)

    print(f"\nagainst {a.n_perm} mass-matched random sets of the same size")
    print(r[["statistic", "observed", "null_mean", "null_sd", "p"]]
          .to_string(index=False))
    print(f"\nwrote {a.out}")
    print("""
Reading this. The null accounts for set size, which matters more than it
appears. A draw of 64 features from several thousand rarely connects to
itself, whereas a draw of 400 often does, so a connected fraction can only be
judged against a null of the same size.

The raw component count is listed for completeness and should not be read as
a test. A null whose sets are almost unconnected has almost no components,
so a module with genuine structure records more components than the null for
the trivial reason that it has more edges. Mean component size and
components per connected feature ask the intended question, which is whether
the connectivity present assembles into large pieces or into pairs.""")


if __name__ == "__main__":
    main()
