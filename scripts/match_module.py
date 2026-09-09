#!/usr/bin/env python3
"""
match_module.py — test the PUBLISHED module against your own analysis

Rahman et al. (Int J Cancer 2024;155:508-518) report module ME16: 121 metabolic
features, all with OR<1, enriched in lyso-glycerophospholipids. Supplemental
Table 1 gives m/z, retention time and ionization mode for every one.

This matches those features to yours on m/z (ppm) and retention time, then asks
whether they behave the same way in your independent analysis. Because the
feature set is pre-specified by someone else, there is no multiplicity penalty:
one hypothesis, one test.

Usage
-----
python match_module.py --module ME16_features.csv \
    --disc disc/c18_neg/features.csv --mode negative --ppm 10 --rt-tol 30
"""
import argparse, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--module", required=True)
ap.add_argument("--disc", required=True, help="features.csv from discover.py")
ap.add_argument("--mode", choices=["negative", "positive"], required=True)
ap.add_argument("--ppm", type=float, default=10.0)
ap.add_argument("--rt-tol", type=float, default=30.0,
                help="seconds; set large or use --no-rt if gradients differ")
ap.add_argument("--no-rt", action="store_true")
ap.add_argument("--out")
a = ap.parse_args()

mod = pd.read_csv(a.module)
mod = mod[mod["mode"].astype(str).str.lower() == a.mode]
mine = pd.read_csv(a.disc)
print(f"module: {len(mod)} {a.mode}-mode features | yours: {len(mine)}")

rows = []
for _, r in mod.iterrows():
    tol = a.ppm * 1e-6 * r["m.z"]
    c = mine[(mine.mz - r["m.z"]).abs() <= tol]
    if not a.no_rt and len(c):
        c = c[(c.rt - r["rt"]).abs() <= a.rt_tol]
    if len(c):
        b = c.iloc[(c.mz - r["m.z"]).abs().to_numpy().argmin()]
        rows.append({"mz_pub": r["m.z"], "rt_pub": r["rt"],
                     "annotation": r.get("Annotation"),
                     "OR_pub": r.get("OR (95% CI)4"),
                     "feature_id": b.feature_id, "mz": b.mz, "rt": b.rt,
                     "t": b.t, "p": b.p, "q": b.q,
                     "rank_pct": (mine.p.rank().loc[b.name]) / len(mine) * 100})
m = pd.DataFrame(rows)
print(f"matched: {len(m)}/{len(mod)} ({len(m)/max(len(mod),1):.0%})")
if not len(m):
    raise SystemExit("no matches -- loosen --ppm or pass --no-rt")

neg = int((m.t < 0).sum())
from math import comb
n = len(m)
sign_p = 2 * sum(comb(n, k) for k in range(max(neg, n - neg), n + 1)) / 2 ** n
print(f"\nDIRECTION  {neg}/{n} negative (published: all 121 OR<1)")
print(f"  sign test p = {min(sign_p,1):.4g}")

# are they enriched toward the top of your ranking?
rng = np.random.default_rng(0)
allr = np.arange(1, len(mine) + 1) / len(mine) * 100
obs = m.rank_pct.median()
null = [np.median(rng.choice(allr, n)) for _ in range(20000)]
print(f"\nRANK       median {obs:.1f}% of features "
      f"(enrichment p = {max(np.mean(np.array(null) <= obs), 5e-5):.4g})")

# mean t against a random-set null (uses magnitude and direction together)
tn = [np.mean(rng.choice(mine.t.to_numpy(), n)) for _ in range(20000)]
print(f"MEAN t     {m.t.mean():+.3f} "
      f"(p = {max(np.mean(np.array(tn) <= m.t.mean()), 5e-5):.4g})")

print("\nstrongest matched features:")
print(m.nsmallest(10, "p")[["mz", "rt", "annotation", "t", "rank_pct"]]
      .to_string(index=False))
if a.out:
    m.to_csv(a.out, index=False); print(f"\nwrote {a.out}")
