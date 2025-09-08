# stripe_score.py
from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd
from joblib import Parallel, delayed

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
) -> pd.Series:
    with open(log_path, "a") as logf:
        logf.write(f"Processing {cellname}\n")

    pair_file = pairs_dir / f"{cellname}.txt"
    if not pair_file.exists():
        print(f"⚠️ Skipping {cellname} (file not found: {pair_file})")
        return pd.Series([0] * len(df_stripe), name=f"{cellname}")

    print(f"Processing {cellname}")
    usecols = [chr1_col, pos1_col, chr2_col, pos2_col]
    df_pairs = pd.read_csv(pair_file, sep=r"\s+", header=None, usecols=usecols)
    df_pairs.columns = ["chr1", "pos1", "chr2", "pos2"]

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

            main_mask = (
                (df_pairs["chr1"] == chr1)
                & (df_pairs["pos1"].between(row_start, row_end))
                & (df_pairs["chr2"] == chr2)
                & (df_pairs["pos2"].between(col_start, col_end))
            )
            flank_mask = (
                (df_pairs["chr1"] == chr1)
                & (
                    df_pairs["pos1"].between(row_start - wid_L, row_start - 1)
                    | df_pairs["pos1"].between(row_end + 1, row_end + wid_L)
                )
                & (df_pairs["chr2"] == chr2)
                & (df_pairs["pos2"].between(col_start, col_end))
            )

        # vertical stripe (p2 == p4)
        elif p2 == p4:
            col_start, col_end = p1, p2
            row_start, row_end = p3, p4
            wid_L = col_end - col_start

            main_mask = (
                (df_pairs["chr1"] == chr1)
                & (df_pairs["pos1"].between(row_start, row_end))
                & (df_pairs["chr2"] == chr2)
                & (df_pairs["pos2"].between(col_start, col_end))
            )
            flank_mask = (
                (df_pairs["chr1"] == chr1)
                & (df_pairs["pos1"].between(row_start, row_end))
                & (df_pairs["chr2"] == chr2)
                & (
                    df_pairs["pos2"].between(col_start - wid_L, col_start - 1)
                    | df_pairs["pos2"].between(col_end + 1, col_end + wid_L)
                )
            )
        else:
            ratios.append(0)
            continue

        main_count = main_mask.sum()
        flank_count = flank_mask.sum()
        ratio = (main_count + 1) / (flank_count / 2 + 1)
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
    n_jobs: int = 40,
) -> Path:
    """Compute stripe scores per cell and save a TSV; returns output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # metadata
    meta_df = pd.read_csv(meta_file, sep="\t")
    meta_df.columns = meta_df.columns.str.strip()
    meta_df["Celltype"] = meta_df["Celltype"].str.strip()
    meta_df["Cellname"] = meta_df["Cellname"].str.strip()

    # stripes table
    df_stripe = pd.read_csv(stripe_file, sep="\t")
    print(f"Loaded stripe file: {stripe_file}, total {len(df_stripe)} entries")

    # cells for this celltype
    cellnames = meta_df.loc[meta_df["Celltype"] == celltype, "Cellname"].tolist()
    print(f"Found {len(cellnames)} cells in {celltype}")

    log_path = output_dir / f"log_{celltype}.txt"
    # parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_cell)(
            cellname, df_stripe.copy(), celltype,
            pairs_dir, chr1_col, pos1_col, chr2_col, pos2_col, log_path
        )
        for cellname in cellnames
    )

    # merge back
    for series in results:
        df_stripe[series.name] = series

    df_stripe["celltype_stripe"] = celltype
    out_file = output_dir / f"stripe_score_{celltype}.tsv"
    df_stripe.to_csv(out_file, sep="\t", index=False)
    print(f"Saved to: {out_file}")
    return out_file