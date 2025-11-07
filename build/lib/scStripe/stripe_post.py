#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import ast
import math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def _clean_group(
    n: int,
    subdf: pd.DataFrame,
    c_idx: np.ndarray,
    pos1: np.ndarray,
    pos2: np.ndarray,
    pos3: np.ndarray,
    pos4: np.ndarray,
    height: np.ndarray,              # ← 接受 height
    width: np.ndarray,               # ← 接受 width
    split_length: int,
    step_size: int,
) -> List[int]:
    to_remove: List[int] = []

    span = max(int(math.ceil(split_length / max(step_size, 1))), 1)
    all_n = [n + i for i in range(span)]
    if "split_mat_id" in subdf.columns:
        idx = np.where(subdf["split_mat_id"].isin(all_n))[0]
    else:
        idx = np.arange(len(subdf))

    for i in range(len(idx) - 1):
        for j in range(i + 1, len(idx)):
            ii_local = idx[i]
            jj_local = idx[j]

            A = [pos1[ii_local], pos2[ii_local], pos3[ii_local], pos4[ii_local]]
            B = [pos1[jj_local], pos2[jj_local], pos3[jj_local], pos4[jj_local]]

            inter_x = max(0, min(A[1], B[1]) - max(A[0], B[0]) + 1)
            inter_y = max(0, min(A[3], B[3]) - max(A[2], B[2]) + 1)

            lenA_x = max(A[1] - A[0] + 1, 1)
            lenB_x = max(B[1] - B[0] + 1, 1)
            lenA_y = max(A[3] - A[2] + 1, 1)
            lenB_y = max(B[3] - B[2] + 1, 1)

            s_x = inter_x / min(lenA_x, lenB_x)
            s_y = inter_y / min(lenA_y, lenB_y)

            if s_x > 0.2 and s_y > 0.2:
                ratio_i = height[ii_local] / max(width[ii_local], 1)
                ratio_j = height[jj_local] / max(width[jj_local], 1)
                # 记录“全局行号”
                to_remove.append(c_idx[ii_local] if ratio_i <= ratio_j else c_idx[jj_local])

    return list(set(to_remove))

def remove_redundant(
    df: pd.DataFrame,
    split_length: int,
    step_size: int,
    n_jobs: int = 4,
) -> pd.DataFrame:
    if df.empty:
        return df

    keep = np.ones(len(df), dtype=bool)

    for chrom in df["chr"].unique():
        c_idx = np.where(df["chr"] == chrom)[0]
        subdf = df.iloc[c_idx]
        if subdf.empty:
            continue

        pos1 = df.loc[c_idx, "pos1"].values
        pos2 = df.loc[c_idx, "pos2"].values
        pos3 = df.loc[c_idx, "pos3"].values
        pos4 = df.loc[c_idx, "pos4"].values
        height = df.loc[c_idx, "length"].values
        width = df.loc[c_idx, "width"].values

        if "split_mat_id" in subdf.columns:
            split_ids = sorted(set(subdf["split_mat_id"].dropna().astype(int).tolist()))
            if not split_ids:
                split_ids = [0]
        else:
            split_ids = [0]

        results = Parallel(n_jobs=n_jobs)(
            delayed(_clean_group)(
                n=int(n),
                subdf=subdf,
                c_idx=c_idx,
                pos1=pos1, pos2=pos2, pos3=pos3, pos4=pos4,
                height=height, width=width,
                split_length=split_length,
                step_size=step_size,
            )
            for n in split_ids
        )
        for r in results:
            keep[r] = False

    return df[keep]


def parse_stripe(file_path: Path, chr_num: str, resolution: int) -> Optional[pd.DataFrame]:
    df = pd.read_csv(file_path)
    if df.empty:
        return None

    required = {"pass_t", "pass_fc", "pass_dip", "stripe_width", "stripe_len_anchor_and_length", "direction"}
    if not required.issubset(df.columns):
        return None

    df["pass_t"] = df["pass_t"].apply(ast.literal_eval)
    df = df[df["pass_t"].apply(lambda x: isinstance(x, list) and len(x) > 0 and x[0] == 1)]
    if df.empty:
        return None

    df["pass_fc"] = pd.to_numeric(df["pass_fc"], errors="coerce")
    df = df[(df["pass_fc"] == 1) & (df["pass_dip"] == 1)]
    if df.empty:
        return None

    df["stripe_width"] = df["stripe_width"].apply(ast.literal_eval)
    width_ok = df["stripe_width"].dropna().apply(lambda x: isinstance(x, list) and len(x) == 2)
    if width_ok.sum() != len(df):
        return None
    df[["pos1", "pos2"]] = pd.DataFrame(df["stripe_width"][width_ok].tolist(), index=df.index)

    df = df.dropna(subset=["stripe_len_anchor_and_length"]).copy()
    df["pos3"] = df["stripe_len_anchor_and_length"].apply(
        lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else None
    )

    df["chr"] = f"chr{chr_num}"
    df["chr2"] = df["chr"]
    df["pos4"] = df["pos3"]

    df.loc[df["direction"] == "left", "pos3"] = df["pos1"]
    df.loc[df["direction"] == "right", "pos4"] = df["pos2"]

    for col in ["pos1", "pos2", "pos3", "pos4"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        df[col] = df[col] * int(resolution)

    df["pos1"] = df["pos1"] + 1
    df["pos3"] = df["pos3"] + 1

    df["length"] = df["pos4"] - df["pos3"] + 1
    df["width"] = df["pos2"] - df["pos1"] + 1

    cols = ["chr", "pos1", "pos2", "chr2", "pos3", "pos4", "length", "width"]
    if "split_mat_id" in df.columns:
        cols.append("split_mat_id")
    return df[cols]


def write_bedpe_and_bed(df: pd.DataFrame, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    """
    Export BEDPE (chr,pos1,pos2,chr2,pos3,pos4) and BED.
    BED is always based on (chr2,pos3,pos4).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bedpe_path = out_dir / f"{stem}.bedpe"
    bed_path = out_dir / f"{stem}.bed"

    bedpe = df[["chr", "pos1", "pos2", "chr2", "pos3", "pos4"]]
    bedpe.to_csv(bedpe_path, sep="\t", header=False, index=False)

    bed = df[["chr2", "pos3", "pos4"]]
    bed.to_csv(bed_path, sep="\t", header=False, index=False)

    return bedpe_path, bed_path

# ------------------------- Main Entry ------------------------- #

def process_all(
    results_dir: Path,
    split_length: int,
    resolution: int,
    step_size: int,
    n_jobs: int = 4,
) -> Tuple[Path, Path]:
    """
    Traverse chromosome folders under results_dir, parse all *unidentified_stripe.csv,
    concatenate, then remove redundant stripes.
    Returns (raw_tsv, final_tsv).
    """
    results_dir = Path(results_dir)
    all_dfs: List[pd.DataFrame] = []

    for chr_dir in sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("chr")):
        chr_num = chr_dir.name[3:]
        for f in chr_dir.iterdir():
            if f.name.endswith("unidentified_stripe.csv"):
                df = parse_stripe(f, chr_num, resolution)
                if df is not None:
                    all_dfs.append(df)

    if not all_dfs:
        raise RuntimeError(f"No valid stripes found under: {results_dir}")

    combined = pd.concat(all_dfs, ignore_index=True)

    raw_tsv = results_dir / "processed_results_raw.tsv"
    combined.to_csv(raw_tsv, sep="\t", index=False)

    final_df = remove_redundant(combined, split_length, step_size, n_jobs=n_jobs)
    final_tsv = results_dir / "processed_results.tsv"
    final_df.to_csv(final_tsv, sep="\t", index=False)

    # export BEDPE & BED here
    write_bedpe_and_bed(final_df, results_dir, stem="processed_results")

    return raw_tsv, final_tsv