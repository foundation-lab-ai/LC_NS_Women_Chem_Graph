"""
common.py — the pieces every analysis shares

Three operations recur across the analyses and are defined once here. All of
them treat the matched pair as the unit, which is what the study design
requires and what makes the permutation null exact.
"""

import numpy as np
import pandas as pd


def normalize(X, batches=None):
    """
    Per-sample median normalization, log transform, batch centering and
    Pareto scaling. Rows are samples, columns are features.
    """
    X = np.clip(X.astype(np.float64), 0, None)
    med = np.median(np.where(X > 0, X, np.nan), axis=1, keepdims=True)
    med = np.nan_to_num(med, nan=1.0)
    med[med <= 0] = 1.0
    X = np.log1p(X / med * np.median(med))
    if batches is not None:
        for b in np.unique(batches):
            m = batches == b
            if m.sum() > 1:
                X[m] -= np.median(X[m], axis=0, keepdims=True)
    mu = X.mean(0, keepdims=True)
    sd = np.sqrt(X.std(0, keepdims=True) + 1e-8)
    return (X - mu) / sd


def build_D(data_dir, intensities, metadata):
    """
    Within-pair differences, case minus control, one row per matched pair.
    Returns the difference matrix, the node table and the feature
    identifiers in column order.
    """
    nf = pd.read_csv(f"{data_dir}/node_features.csv")
    it = pd.read_csv(intensities).reset_index(drop=True)
    it = it.set_axis([f"F{i:06d}" for i in range(len(it))], axis=0)
    it = it.loc[nf.feature_id.tolist()]
    samples = [c for c in it.columns if c not in ("mz", "rt")]

    meta = pd.read_csv(metadata)
    if "label" not in meta:
        meta["label"] = meta.LungCancer
    meta["label"] = (meta.label.astype(str).str.lower() == "case").astype(int)
    meta = meta[meta.matched_set.notna()]
    meta = meta[meta.sample_id.isin(samples)].reset_index(drop=True)

    pos = {s: i for i, s in enumerate(samples)}
    X = normalize(
        it[samples].to_numpy(np.float64).T[[pos[s] for s in meta.sample_id]],
        meta.Batch.to_numpy() if "Batch" in meta else None)

    ci, ki = [], []
    for s in meta.matched_set.unique():
        i = np.where(meta.matched_set == s)[0]
        p = i[meta.label.to_numpy()[i] == 1]
        q = i[meta.label.to_numpy()[i] == 0]
        if len(p) and len(q):
            ci.append(p[0]); ki.append(q[0])
    return (X[np.array(ci)] - X[np.array(ki)], nf,
            nf.feature_id.to_numpy())


def signflip_t(D, n_perm=2000, seed=0):
    """
    Paired t statistic per feature and its exact null.

    Which member of a pair is the case is arbitrary under the null, so
    flipping the sign of a row of D produces a dataset as likely as the one
    observed. Var(sD) does not depend on the flip pattern, so all
    permutations reduce to one matrix product.
    """
    n = D.shape[0]
    m2 = (D ** 2).mean(0)
    om = D.mean(0)
    t = om / np.sqrt(np.maximum(m2 - om ** 2, 1e-12) / (n - 1))
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(n_perm, n))
    M = (S @ D) / n
    T = M / np.sqrt(np.maximum(m2[None, :] - M ** 2, 1e-12) / (n - 1))
    return t, T


def bh(p):
    """Benjamini-Hochberg adjusted p values."""
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    q = np.empty_like(p)
    m, prev = len(p), 1.0
    for r, i in enumerate(o[::-1]):
        prev = min(prev, p[i] * m / (m - r))
        q[i] = prev
    return q
