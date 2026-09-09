#!/usr/bin/env python3
"""
parse_mwtab.py — mwTab -> feature table + sample metadata

Usage
-----
python parse_mwtab.py data/ST002773_AN004513.txt --outdir data/hilic_pos
python parse_mwtab.py data/ST002773_AN004514.txt --outdir data/c18_neg

Writes
------
features.csv    mz, rt, <one column per sample>   -> feed to annotate_to_graph.py
metadata.csv    sample_id + every SUBJECT_SAMPLE_FACTORS field, one col each
report.txt      what was found, what's missing
"""

import argparse
import os
import re
import sys
import pandas as pd


# ---------------------------------------------------------------- factors

def parse_factors(lines):
    """
    SUBJECT_SAMPLE_FACTORS lines look like:
      SUBJECT_SAMPLE_FACTORS  -  SampleID  Factor1:val | Factor2:val  RAWFILE=x
    Tab-delimited, factor field is pipe-separated key:value pairs.
    """
    recs = []
    for ln in lines:
        if not ln.startswith("SUBJECT_SAMPLE_FACTORS"):
            continue
        parts = ln.rstrip("\n").split("\t")
        parts = [p.strip() for p in parts]
        if len(parts) < 4:
            continue
        rec = {"subject_id": parts[1], "sample_id": parts[2]}
        for kv in parts[3].split("|"):
            if ":" in kv:
                k, v = kv.split(":", 1)
                rec[k.strip()] = v.strip()
        # trailing additional-data field (raw file names etc.)
        if len(parts) > 4 and parts[4]:
            for kv in parts[4].split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    rec[k.strip()] = v.strip()
        recs.append(rec)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------- data block

def extract_block(lines, start_tag, end_tag):
    out, on = [], False
    for ln in lines:
        s = ln.strip()
        if s == start_tag:
            on = True
            continue
        if s == end_tag:
            break
        if on:
            out.append(ln.rstrip("\n"))
    return out


def parse_data_block(block):
    """First row = Samples header, optional Factors row, then data rows."""
    if not block:
        return None
    rows = [ln.split("\t") for ln in block if ln.strip()]
    header = [c.strip() for c in rows[0]]
    samples = header[1:]

    data = []
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        if r[0].strip().lower() in ("factors", "factor"):
            continue
        name = r[0].strip()
        vals = r[1:len(samples) + 1]
        vals += [""] * (len(samples) - len(vals))
        data.append([name] + vals)

    df = pd.DataFrame(data, columns=["feature_name"] + samples)
    for c in samples:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------- mz / rt

MZRT_PATTERNS = [
    re.compile(r"^\s*(\d+\.?\d*)\s*[_/]\s*(\d+\.?\d*)\s*$"),        # 85.0284_44.5
    re.compile(r"^\s*mz\s*(\d+\.?\d*)\D+rt\s*(\d+\.?\d*)", re.I),   # mz85.02_rt44.5
    re.compile(r"^\s*(\d+\.?\d*)\s+(\d+\.?\d*)\s*$"),               # 85.0284 44.5
]


def split_mz_rt(names):
    mz, rt = [], []
    for n in names:
        hit = None
        for p in MZRT_PATTERNS:
            m = p.match(str(n))
            if m:
                hit = (float(m.group(1)), float(m.group(2)))
                break
        mz.append(hit[0] if hit else None)
        rt.append(hit[1] if hit else None)
    return mz, rt


def mz_rt_from_metabolites_block(lines):
    """Fallback: METABOLITES block may carry explicit mz / rt columns."""
    block = extract_block(lines, "METABOLITES_START", "METABOLITES_END")
    if not block:
        return None
    rows = [ln.split("\t") for ln in block if ln.strip()]
    hdr = [c.strip().lower() for c in rows[0]]
    df = pd.DataFrame(rows[1:], columns=hdr)
    mzc = next((c for c in hdr if c in ("mz", "m/z", "moverz_quant", "mass")), None)
    rtc = next((c for c in hdr if c in ("rt", "ri", "retention_time",
                                        "retention_index", "rt_min")), None)
    if not mzc or not rtc:
        return None
    out = pd.DataFrame({
        "feature_name": df[hdr[0]].values,
        "mz": pd.to_numeric(df[mzc], errors="coerce"),
        "rt": pd.to_numeric(df[rtc], errors="coerce"),
    })
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mwtab")
    ap.add_argument("--outdir", default="parsed")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.mwtab, errors="ignore") as f:
        lines = f.readlines()

    rep = []

    meta = parse_factors(lines)
    if meta.empty:
        sys.exit("No SUBJECT_SAMPLE_FACTORS found — wrong file?")
    meta.to_csv(f"{args.outdir}/metadata.csv", index=False)
    rep.append(f"samples: {len(meta)}")
    rep.append(f"factor fields: {[c for c in meta.columns]}")

    block = extract_block(lines, "MS_METABOLITE_DATA_START", "MS_METABOLITE_DATA_END")
    df = parse_data_block(block)
    if df is None or df.empty:
        sys.exit("MS_METABOLITE_DATA block empty — this file may be raw-only.")
    rep.append(f"features: {len(df)}")

    mz, rt = split_mz_rt(df["feature_name"])
    n_ok = sum(v is not None for v in mz)
    if n_ok < 0.5 * len(df):
        alt = mz_rt_from_metabolites_block(lines)
        if alt is not None:
            df = df.merge(alt, on="feature_name", how="left")
            n_ok = df["mz"].notna().sum()
            rep.append("mz/rt source: METABOLITES block")
        else:
            rep.append("!! could not parse mz/rt from feature names")
            rep.append(f"   examples: {list(df['feature_name'][:5])}")
    else:
        df["mz"], df["rt"] = mz, rt
        rep.append("mz/rt source: feature names")

    rep.append(f"mz/rt resolved: {n_ok}/{len(df)}")

    if "mz" in df:
        df = df.dropna(subset=["mz", "rt"])
        sample_cols = [c for c in df.columns
                       if c not in ("feature_name", "mz", "rt")]
        out = df[["mz", "rt"] + sample_cols]
        out.to_csv(f"{args.outdir}/features.csv", index=False)
        rep.append(f"wrote features.csv: {out.shape[0]} x {len(sample_cols)} samples")

        rtv = df["rt"]
        unit = "minutes" if rtv.max() < 60 else "seconds"
        rep.append(f"RT range {rtv.min():.1f}-{rtv.max():.1f} -> looks like {unit}")
        rep.append(f"  => use --rt-tol {'0.2' if unit=='minutes' else '10'}")

    # what's usable for the study design
    lower = {c.lower(): c for c in meta.columns}
    rep.append("\ndesign check:")
    for want, keys in [
        ("case/control", ("group", "diagnosis", "status", "case", "phenotype")),
        ("matched pair", ("match", "pair", "set", "stratum", "matchid")),
        ("time to dx",   ("time", "year", "lag", "interval", "followup", "days")),
        ("histology",    ("histology", "subtype", "adenocarcinoma")),
        ("batch",        ("batch", "plate", "run", "sequence")),
    ]:
        hits = [lower[k] for k in lower if any(x in k for x in keys)]
        rep.append(f"  {want:<14} {'FOUND: ' + str(hits) if hits else 'absent'}")

    txt = "\n".join(rep)
    open(f"{args.outdir}/report.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
