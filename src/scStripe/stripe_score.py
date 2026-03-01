# stripe_score.py
from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
from joblib import Parallel, delayed
import math

def _process_cell(
    cellname: str,
    df_stripe: pd.DataFrame,
    celltype: str,
    pairs_dir: Path,
    chr1_col: int,
    pos1_col: int,
    chr2_col: int,
    pos2_col: int,
    log_path: Path,
    q1: float,
    q2: float,
    flanking: str, 
    flanking_range: int | None,  
) -> pd.Series:
    flanking = str(flanking).lower().strip()
    if flanking_range is not None:
        flanking_range = int(flanking_range)
        if flanking_range <= 0:
            raise ValueError("flanking_range must be a positive integer, or None")
        
    if flanking not in ("both", "inside", "outside"):
        raise ValueError(f"Invalid flanking={flanking}, must be one of both/inside/outside")

    if not (0.0 <= q1 < q2 <= 1.0):
        raise ValueError(f"Invalid q1,q2: {q1},{q2} (require 0 <= q1 < q2 <= 1)")

    with open(log_path, "a") as logf:
        logf.write(f"Processing {cellname}\n")

    pair_file = pairs_dir / f"{cellname}.txt"
    if not pair_file.exists():
        print(f"⚠️ Skipping {cellname} (file not found: {pair_file})")
        return pd.Series([0] * len(df_stripe), name=f"{cellname}")

    print(f"Processing {cellname}")
    usecols = [chr1_col, pos1_col, chr2_col, pos2_col]

    df_pairs = pd.read_csv(
        pair_file,
        sep=r"\s+",
        header=None,
        usecols=usecols,
        dtype=str,        
        comment="#",
        engine="python",
    )
    df_pairs.columns = ["chr1", "pos1", "chr2", "pos2"]

    for c in ("pos1", "pos2"):
        s = df_pairs[c].fillna("").str.replace(",", "", regex=False).str.strip()
        df_pairs[c] = pd.to_numeric(s, errors="coerce")

    df_pairs = df_pairs.dropna(subset=["pos1", "pos2"]).copy()
    df_pairs["pos1"] = df_pairs["pos1"].astype("int64")
    df_pairs["pos2"] = df_pairs["pos2"].astype("int64")

    for c in ("chr1", "chr2"):
        df_pairs[c] = df_pairs[c].astype(str).str.strip()

    ratios: List[float] = []
    for _, row in df_stripe.iterrows():
        chr1, p1, p2, chr2, p3, p4 = (
            row["chr"], row["pos1"], row["pos2"], row["chr2"], row["pos3"], row["pos4"]
        )

        if chr1 != chr2:
            ratios.append(0)
            continue

        # horizontal stripe (p1 == p3)
        if p1 == p3:
            row_start, row_end = p1, p2
            col_start, col_end = p3, p4
            wid_L = row_end - row_start
            len_L = col_end - col_start
            flank_L = wid_L if flanking_range is None else flanking_range  
            sub_start = col_start + int(math.floor(len_L * (1.0 - q2)))
            sub_end = col_end - int(math.floor(len_L * q1))

            if sub_start > sub_end:
                main_mask = pd.Series(False, index=df_pairs.index)
                flank_mask = pd.Series(False, index=df_pairs.index)
            else:
                main_mask = (
                    (df_pairs["chr1"] == chr1)
                    & (df_pairs["pos1"].between(row_start, row_end))
                    & (df_pairs["chr2"] == chr2)
                    & (df_pairs["pos2"].between(sub_start, sub_end))
                )
                # --- define two flanks separately ---
                outside_mask = (
                    (df_pairs["chr1"] == chr1)
                    & (df_pairs["pos1"].between(row_start - flank_L, row_start - 1))  
                    & (df_pairs["chr2"] == chr2)
                    & (df_pairs["pos2"].between(sub_start, sub_end))
                )

                inside_mask = (
                    (df_pairs["chr1"] == chr1)
                    & (df_pairs["pos1"].between(row_end + 1, row_end + flank_L))     
                    & (df_pairs["chr2"] == chr2)
                    & (df_pairs["pos2"].between(sub_start, sub_end))
                )

                if flanking == "both":
                    flank_mask = outside_mask | inside_mask
                elif flanking == "inside":
                    flank_mask = inside_mask
                else:  # flanking == "outside"
                    flank_mask = outside_mask

        # vertical stripe (p2 == p4)
        elif p2 == p4:
            col_start, col_end = p1, p2
            row_start, row_end = p3, p4
            wid_L = col_end - col_start
            len_L = row_end - row_start
            flank_L = wid_L if flanking_range is None else flanking_range

            sub_start = row_start + int(math.floor(len_L * q1))
            sub_end = row_end - int(math.floor(len_L * (1.0 - q2)))

            if sub_start > sub_end:
                main_mask = pd.Series(False, index=df_pairs.index)
                flank_mask = pd.Series(False, index=df_pairs.index)
            else:
                main_mask = (
                    (df_pairs["chr1"] == chr1)
                    & (df_pairs["pos1"].between(sub_start, sub_end))
                    & (df_pairs["chr2"] == chr2)
                    & (df_pairs["pos2"].between(col_start, col_end))
                )

                outside_mask = (
                    (df_pairs["chr1"] == chr1)
                    & (df_pairs["pos1"].between(sub_start, sub_end))
                    & (df_pairs["chr2"] == chr2)
                    & (df_pairs["pos2"].between(col_end + 1, col_end + flank_L))      
                )

                inside_mask = (
                    (df_pairs["chr1"] == chr1)
                    & (df_pairs["pos1"].between(sub_start, sub_end))
                    & (df_pairs["chr2"] == chr2)
                    & (df_pairs["pos2"].between(col_start - flank_L, col_start - 1))
                )

                if flanking == "both":
                    flank_mask = outside_mask | inside_mask
                elif flanking == "inside":
                    flank_mask = inside_mask
                else:  # flanking == "outside"
                    flank_mask = outside_mask

        else:
            ratios.append(0.0)
            continue

        main_count = main_mask.sum()
        flank_count = flank_mask.sum()

        if flanking == "both":
            denom = (flank_count / 2.0) + 1.0
        else:
            denom = float(flank_count) + 1.0

        ratio = (main_count + 1.0) / denom
        ratios.append(ratio)

    return pd.Series(ratios, name=f"{cellname}")


def run_stripe_scores(
    *,
    celltype: str,
    pairs_dir: Path,
    meta_file: Path,
    output_dir: Path,
    stripe_file: Path,
    chr1_col: int,
    pos1_col: int,
    chr2_col: int,
    pos2_col: int,
    q1: float = 0.0,
    q2: float = 0.5,
    flanking: str,  
    flanking_range: int | None = None,
    n_jobs: int = 40,
) -> Path:
    """Compute stripe scores per cell and save a TSV; returns output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    if not (0.0 <= q1 < q2 <= 1.0):
        raise ValueError(f"Invalid q1,q2: {q1},{q2} (require 0 <= q1 < q2 <= 1)")

    # metadata
    meta_df = pd.read_csv(meta_file, sep="\t")
    meta_df.columns = meta_df.columns.str.strip()
    meta_df["Celltype"] = meta_df["Celltype"].str.strip()
    meta_df["Cellname"] = meta_df["Cellname"].str.strip()

    # stripes table
    df_stripe = pd.read_csv(stripe_file, sep="\t")
    print(f"Loaded stripe file: {stripe_file}, total {len(df_stripe)} entries")

    # cells for this celltype
    if celltype.upper() == "ALL":
        cellnames = meta_df["Cellname"].tolist()
        out_tag = "ALLCELLS"
        print(f"Found {len(cellnames)} cells (ALL from metadata)")
    else:
        cellnames = meta_df.loc[meta_df["Celltype"] == celltype, "Cellname"].tolist()
        out_tag = celltype
        print(f"Found {len(cellnames)} cells in {celltype}")

    log_path = output_dir / f"log_{out_tag}.txt"

    # parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_cell)(
            cellname=cellname,
            df_stripe=df_stripe.copy(),
            celltype=celltype,
            pairs_dir=pairs_dir,
            chr1_col=chr1_col,
            pos1_col=pos1_col,
            chr2_col=chr2_col,
            pos2_col=pos2_col,
            log_path=log_path,
            q1=q1,
            q2=q2,
            flanking=flanking,
            flanking_range=flanking_range
        )
        for cellname in cellnames
    )


    for series in results:
        df_stripe[series.name] = series

    df_stripe["celltype_stripe"] = celltype
    out_file = output_dir / f"stripe_score_{celltype}.tsv"
    df_stripe.to_csv(out_file, sep="\t", index=False)
    print(f"Saved to: {out_file}")
    return out_file