#!/usr/bin/env python3
"""
prep_results.py — ST002773 *_Results.txt -> features.csv for annotate_to_graph.py

Splits the mz_time column, separates QC/NIST injections from study samples,
uses the pooled QC replicates to filter unreliable features, and aligns
columns to metadata.csv.

Usage
-----
python prep_results.py ST002773_AN004513_Results.txt \
    --metadata data/hilic_pos/metadata.csv --outdir data/hilic_pos
"""

import argparse
import os
import re
import warnings
import numpy as np
import pandas as pd

QC_PATTERNS = [r"^nist", r"^q3june", r"^pool", r"^qc", r"^blank"]


def is_qc(name):
    n = str(name).strip().lower()
    return any(re.match(p, n) for p in QC_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--metadata")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--max-qc-cv", type=float, default=None,
                    help="OPTIONAL: drop features with pooled-QC CV above this. "
                         "Off by default -- across-batch QC CV conflates drift "
                         "with noise and over-filters badly.")
    ap.add_argument("--min-presence", type=float, default=0.50,
                    help="drop features detected in fewer than this fraction")
    ap.add_argument("--drop-unlabeled", action="store_true",
                    help="drop study samples with no case/control label")
    ap.add_argument("--label-col", default="LungCancer")
    ap.add_argument("--keep-qc", action="store_true",
                    help="also write a table containing the QC injections")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("reading (may take a minute)...")
    df = pd.read_csv(args.results, sep="\t", low_memory=False)
    idcol = df.columns[0]
    print(f"  {df.shape[0]} features x {df.shape[1]-1} injections")

    # ---- split mz_time -----------------------------------------------------
    parts = df[idcol].astype(str).str.split("_", n=1, expand=True)
    df["mz"] = pd.to_numeric(parts[0], errors="coerce")
    df["rt"] = pd.to_numeric(parts[1], errors="coerce")
    bad = df["mz"].isna() | df["rt"].isna()
    if bad.any():
        print(f"  dropped {bad.sum()} rows with unparseable {idcol}")
        df = df[~bad]

    unit = "minutes" if df["rt"].max() < 60 else "seconds"
    print(f"  m/z {df.mz.min():.1f}-{df.mz.max():.1f} | "
          f"RT {df.rt.min():.1f}-{df.rt.max():.1f} ({unit})")

    inj = [c for c in df.columns if c not in (idcol, "mz", "rt")]
    qc_cols = [c for c in inj if is_qc(c)]
    smp_cols = [c for c in inj if not is_qc(c)]
    print(f"  {len(smp_cols)} study samples | {len(qc_cols)} QC injections")

    # ---- QC-based feature filtering ---------------------------------------
    n0 = len(df)
    if qc_cols:
        Q = df[qc_cols].to_numpy(dtype=float)
        Q = np.where(Q > 0, Q, np.nan)
        with np.errstate(invalid="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mu, sd = np.nanmean(Q, axis=1), np.nanstd(Q, axis=1)
            cv = np.where(mu > 0, sd / mu, np.inf)
        df["qc_cv"] = cv
        med = np.nanmedian(cv)
        if args.max_qc_cv is None:
            keep_cv = np.ones(n0, dtype=bool)
            print(f"  QC CV: median {med:.2f} (filter OFF; "
                  f"--max-qc-cv to enable)")
        else:
            keep_cv = np.nan_to_num(cv, nan=np.inf) <= args.max_qc_cv
            print(f"  QC CV <= {args.max_qc_cv}: {keep_cv.sum()}/{n0} "
                  f"(median CV {med:.2f})")
    else:
        df["qc_cv"] = np.nan
        keep_cv = np.ones(n0, dtype=bool)
        print("  no QC injections found — skipping CV filter")

    S = df[smp_cols].to_numpy(dtype=float)
    presence = (np.isfinite(S) & (S > 0)).mean(axis=1)
    keep_pres = presence >= args.min_presence
    print(f"  presence >= {args.min_presence}: {keep_pres.sum()}/{n0}")

    keep = keep_cv & keep_pres
    out = df.loc[keep].copy()
    print(f"  retained {len(out)}/{n0} features ({len(out)/n0:.1%})")

    # ---- metadata alignment ------------------------------------------------
    if args.metadata:
        meta = pd.read_csv(args.metadata)
        meta["sample_id"] = meta["sample_id"].astype(str).str.strip()
        matched = [c for c in smp_cols if c in set(meta.sample_id)]
        missing = set(smp_cols) - set(meta.sample_id)
        print(f"  metadata match: {len(matched)}/{len(smp_cols)}")
        if missing:
            print(f"    unmatched (first 5): {sorted(missing)[:5]}")
        m = meta[meta.sample_id.isin(matched)]
        lc = args.label_col
        if lc in m:
            print(f"    labels: {m[lc].value_counts(dropna=False).to_dict()}")
            if args.drop_unlabeled:
                lab = m[lc].astype(str).str.strip().str.lower()
                ok = ~lab.isin(["nan", "na", "", "none"])
                dropped = (~ok).sum()
                m = m[ok]
                matched = [c for c in matched if c in set(m.sample_id)]
                print(f"    dropped {dropped} unlabeled -> {len(matched)} samples")
        smp_cols = matched
        m.to_csv(f"{args.outdir}/metadata_aligned.csv", index=False)

    # ---- write -------------------------------------------------------------
    feats = out[["mz", "rt"] + smp_cols]
    feats.to_csv(f"{args.outdir}/features.csv", index=False)
    print(f"\nwrote {args.outdir}/features.csv  "
          f"{feats.shape[0]} x {len(smp_cols)} samples")

    if args.keep_qc and qc_cols:
        out[["mz", "rt", "qc_cv"] + qc_cols].to_csv(
            f"{args.outdir}/features_qc.csv", index=False)

    tol = "0.2" if unit == "minutes" else "10"
    print(f"\nnext:\n  python annotate_to_graph.py "
          f"--features {args.outdir}/features.csv \\\n"
          f"      --mode {'positive' if 'AN004513' in args.results else 'negative'} "
          f"--rt-tol {tol} --outdir out/")


if __name__ == "__main__":
    main()
