#!/usr/bin/env python3
"""
find_series.py — labeled node sets from the graph alone

Testing an externally published module makes the result depend on that
publication. The graph can instead define its own node sets, because a
metabolic network contains homologous series, chains in which the same
transformation repeats. Fatty acyl species differing by successive CH2
groups form one. Lipids differing by successive double bonds form another.

Enumerating these requires no outcome, no correlation structure and no
annotation. Validation comes from chromatography. A genuine homologous
series elutes in an order determined by physical chemistry, since adding
CH2 makes a compound more hydrophobic and later-eluting on reversed phase
while adding a double bond does the reverse. A chain assembled from
coincidental mass matches has no reason to obey that ordering, so the
Spearman correlation between position in the chain and retention time
separates real series from artifacts.

Each surviving series is a labeled node set, ready to be tested against an
outcome as a pre-specified unit in the same way a WGCNA module would be.

Usage
-----
python find_series.py --data out/c18_neg --mode rp \\
    --edge-fdr edge_fdr_c18.csv --max-fdr 0.20 --out series_c18.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

# expected sign of the retention shift when the transformation is added,
# on reversed phase. Reversed for HILIC.
RT_DIR = {
    "C2H4_chain": +1, "acetyl_CoA_unit": +1, "methylation": +1,
    "acetylation": +1, "reduction_2H": +1, "deamination": +1,
    "desaturation": -1, "hydroxylation": -1, "oxidation_O": -1,
    "carboxylation": -1, "glucuronidation": -1, "sulfation": -1,
    "taurine_conj": -1, "glycine_conj": -1, "phosphorylation": -1,
    "hexose": -1, "amination": -1, "CO2": -1, "glutathione": -1,
    "malonyl": 0, "NH3": 0,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["rp", "hilic"], default="rp")
    ap.add_argument("--edge-fdr",
                    help="output of edge_fdr.py, used to drop unreliable "
                         "transformations before enumeration")
    ap.add_argument("--max-fdr", type=float, default=0.20)
    ap.add_argument("--min-length", type=int, default=3,
                    help="features per series. Three is the shortest chain "
                         "whose retention ordering carries information.")
    ap.add_argument("--min-rho", type=float, default=0.95,
                    help="a homologous series elutes in near-perfect order, "
                         "so the default is strict")

    ap.add_argument("--src", default=".",
                    help="directory containing annotate_to_graph.py, used to "
                         "collapse transformations that share a mass")
    ap.add_argument("--out", default="series.csv")
    a = ap.parse_args()

    nf = pd.read_csv(f"{a.data}/node_features.csv")
    ed = pd.read_csv(f"{a.data}/edges.csv")
    md = ed[ed.edge_type == "mass_difference"] if "edge_type" in ed else ed
    if "transformation" not in md:
        sys.exit("edges.csv carries no transformation column")

    if a.edge_fdr and os.path.exists(a.edge_fdr):
        f = pd.read_csv(a.edge_fdr)
        keep = set(f.loc[f.fdr < a.max_fdr, "transformation"])
        before = len(md)
        md = md[md.transformation.isin(keep)]
        print(f"transformations at FDR < {a.max_fdr}: {len(keep)}, "
              f"edges {before:,} to {len(md):,}")

    mz = dict(zip(nf.feature_id, nf.mz))
    rt = dict(zip(nf.feature_id, nf.rt))
    name = dict(zip(nf.feature_id,
                    nf.get("best_name", pd.Series([""] * len(nf)))))

    # transformations that share a mass are the same relation under two
    # names, for instance hydroxylation and oxidation both add 15.9949 Da.
    # Grouping by name would report each series twice.
    sys.path.insert(0, a.src)
    TT = {}
    try:
        from annotate_to_graph import TRANSFORMATIONS as TT
        md = md.copy()
        if not TT:
            raise ImportError("empty")
        md["delta_key"] = md.transformation.map(
            lambda t: round(abs(TT.get(t, 0.0)), 4))
        alias = (md.groupby("delta_key").transformation
                 .agg(lambda v: "/".join(sorted(set(v)))).to_dict())
        md["relation"] = md.delta_key.map(alias)
        print(f"{md.transformation.nunique()} labels collapse to "
              f"{md.relation.nunique()} distinct mass relations")
        group_col = "relation"
    except Exception:
        group_col = "transformation"

    rows, sid = [], 0
    for tr, sub in md.groupby(group_col):
        # one graph per transformation. A homologous series is a connected
        # component in that single-relation graph.
        G = nx.Graph()
        G.add_edges_from(zip(sub.source, sub.target))
        for comp in nx.connected_components(G):
            if len(comp) < a.min_length:
                continue
            sub_g = G.subgraph(comp)
            # A homologous series is a path. A component in which features
            # have three or more neighbors at the same mass difference is a
            # dense cluster of coincidental gaps, common in the crowded
            # low-mass region, and sorting it by mass imposes an order the
            # data does not contain. The longest induced path is extracted
            # instead, and the component is skipped when no path of the
            # required length exists.
            if max(dict(sub_g.degree()).values()) > 2:
                ends = [n for n in sub_g if sub_g.degree(n) <= 2]
                best = []
                for u in ends[:40]:
                    lengths = nx.single_source_shortest_path(sub_g, u)
                    cand = max(lengths.values(), key=len)
                    if len(cand) > len(best):
                        best = cand
                if len(best) < a.min_length:
                    continue
                comp = set(best)
                branchy = True
            else:
                branchy = False
            nodes = sorted(comp, key=lambda f: mz[f])
            # consecutive members must differ by one step, otherwise the
            # ordering skips a rung and the chain is not homologous
            gaps = np.diff([mz[f] for f in nodes])
            if len(gaps) and (gaps.max() > 1.6 * np.median(gaps)):
                continue
            order = np.arange(len(nodes))
            rts = np.array([rt[f] for f in nodes], dtype=float)
            if np.ptp(rts) < 1e-9:
                continue
            rho, p = stats.spearmanr(order, rts)
            # Edges are oriented by increasing mass, so the direction to use
            # is that of the mass-increasing member of the pair. Desaturation
            # and reduction share a mass of 2.0157 Da and describe opposite
            # chemistry, and going up in mass is the reduction.
            names = tr.split("/")
            exp = 0
            for nmx in names:
                d_ = TT.get(nmx, 0.0)
                if d_ > 0 and nmx in RT_DIR:
                    exp = RT_DIR[nmx]
                    break
            if exp == 0:
                for nmx in names:
                    if nmx in RT_DIR and RT_DIR[nmx] != 0:
                        exp = RT_DIR[nmx]
                        break
            if a.mode == "hilic":
                exp = -exp
            # a chain is retained when its retention order follows the
            # direction the chemistry predicts, at the required strength
            ok = exp != 0 and np.sign(rho) == exp and abs(rho) >= a.min_rho
            ann = [name.get(f) for f in nodes
                   if isinstance(name.get(f), str) and name.get(f).strip()
                   and name.get(f) != "nan"]
            sid += 1
            rows.append({
                "series_id": f"S{sid:04d}", "transformation": tr,
                "length": len(nodes), "rho_rt": round(float(rho), 3),
                "p_rt": float(p), "expected_sign": exp,
                "retained": bool(ok), "branch_trimmed": branchy,
                "mz_min": round(min(mz[f] for f in nodes), 4),
                "mz_max": round(max(mz[f] for f in nodes), 4),
                "rt_min": round(rts.min(), 1), "rt_max": round(rts.max(), 1),
                "n_annotated": len(ann),
                "annotations": "; ".join(ann[:4]),
                "members": ",".join(nodes),
            })

    d = pd.DataFrame(rows)
    if d.empty:
        sys.exit("no series of the requested length were found")
    d = d.sort_values(["retained", "length"], ascending=[False, False])
    d.to_csv(a.out, index=False)

    r = d[d.retained]
    n_feat = len(set(",".join(r.members).split(","))) if len(r) else 0
    print(f"\n{len(d)} chains of length >= {a.min_length}, "
          f"{len(r)} retained on retention order")
    print(f"  covering {n_feat:,} of {len(nf):,} features "
          f"({n_feat/len(nf):.0%})")
    print(f"  with an annotation: {int((r.n_annotated > 0).sum())} series")

    # how often does the ordering hold by chance? Shuffle retention times.
    rng = np.random.default_rng(0)
    shuffled = []
    for _ in range(200):
        perm = dict(zip(nf.feature_id, rng.permutation(nf.rt.to_numpy())))
        hit = 0
        for _, row in d.iterrows():
            nodes = row.members.split(",")
            rr = np.array([perm[f] for f in nodes], dtype=float)
            if np.ptp(rr) < 1e-9:
                continue
            rho, _ = stats.spearmanr(np.arange(len(nodes)), rr)
            e = row.expected_sign
            if e != 0 and np.sign(rho) == e and abs(rho) >= a.min_rho:
                hit += 1
        shuffled.append(hit)
    exp_hits = float(np.mean(shuffled))
    print(f"  expected under shuffled retention times: {exp_hits:.0f} "
          f"({exp_hits/max(len(r),1):.0%} of the observed)")

    print("\nlongest retained series")
    cols = ["series_id", "transformation", "length", "rho_rt",
            "mz_min", "mz_max", "n_annotated", "annotations"]
    print(r.head(12)[cols].to_string(index=False))
    print(f"\nwrote {a.out}. Each retained series is a pre-specified node "
          f"set defined without reference to any outcome.")


if __name__ == "__main__":
    main()
