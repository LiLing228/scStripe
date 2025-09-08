#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import cooler
from .Stripenn_stripiness_pvalue.getStripe_class import getStripe  

REQUIRED_COLS = ["chr", "pos1", "pos2", "chr2", "pos3", "pos4"]

# ----------------------------- helpers ----------------------------- #

def read_tsv(path: Path, required_cols: Iterable[str]) -> pd.DataFrame:
    """Read a TSV and ensure required columns exist; coerces pos* to numeric."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep="\t")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")
    num_cols = [c for c in ["pos1", "pos2", "pos3", "pos4"] if c in df.columns]
    if num_cols:
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df


def write_bedpe_and_bed(df: pd.DataFrame, out_dir: Path, stem: str, bed_axis: str = "col") -> Tuple[Path, Path]:
    """
    Export BEDPE (chr,pos1,pos2,chr2,pos3,pos4) and BED.
    bed_axis: 'row' -> BED uses (chr,pos1,pos2); 'col' -> BED uses (chr2,pos3,pos4).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bedpe_path = out_dir / f"{stem}.bedpe"
    bed_path = out_dir / f"{stem}.bed"

    bedpe = df[["chr", "pos1", "pos2", "chr2", "pos3", "pos4"]]
    if bed_axis == "row":
        bed = df[["chr", "pos1", "pos2"]]
    else:
        bed = df[["chr2", "pos3", "pos4"]]

    bedpe.to_csv(bedpe_path, sep="\t", header=False, index=False)
    bed.to_csv(bed_path,   sep="\t", header=False, index=False)
    return bedpe_path, bed_path


def unique_sorted_chroms(df: pd.DataFrame, col: str = "chr") -> List[str]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    return sorted(df[col].astype(str).unique())


def filter_default_chroms(chroms: List[str]) -> List[str]:
    bad = ("JH5", "GL4")
    keep: List[str] = []
    for c in chroms:
        if any(b in c for b in bad):
            continue
        if c in {"M", "chrM", "Y", "chrY"}:
            continue
        keep.append(c)
    return keep


# ----------------------------- core API ----------------------------- #

def compute_stripiness_table(
    cool_path: Path,
    candidates: pd.DataFrame,
    *,
    norm: str | bool = "None",
    chroms: List[str] | str = "all",
    numcores: int = 10,
    mask: str = "0",
    bfilter: int = 3,
) -> pd.DataFrame:
    """
    Core function: given cooler and candidate stripes table, return a new table with stripiness/p-values.
    """
    np.seterr(divide="ignore", invalid="ignore")

    # parse chromosomes
    chrom_list = chroms if isinstance(chroms, list) else chroms.split(",")

    # open cooler
    lib = cooler.Cooler(str(cool_path))
    possible_norm = lib.bins().columns

    # normalization
    if norm == "None":
        balance_flag: bool | str = False
    elif norm == "weight":
        balance_flag = True
    elif norm in possible_norm:
        balance_flag = norm  # cooler accepts bins() column name
    else:
        print("Possible normalization methods include:")
        print("  - None")
        for col in possible_norm[3:]:
            print(f"  - {col}")
        print("Invalid normalization; falling back to 'None'.")
        balance_flag = False
    print(f"[INFO] Using balance={balance_flag!r}, binsize={lib.binsize}")

    # filter chromosomes
    all_chromnames = filter_default_chroms(list(lib.chromnames))
    if not all_chromnames:
        raise RuntimeError("All chromosomes are shorter than 50kb or filtered out.")
    # align sizes
    all_chromsizes = lib.chromsizes.reindex(all_chromnames)

    # user subset
    warnflag = False
    if chrom_list and chrom_list[0] != "all":
        idx_map = {c: i for i, c in enumerate(all_chromnames)}
        missing: List[str] = []
        ordered: List[str] = []
        for c in chrom_list:
            if c in idx_map:
                ordered.append(c)
            else:
                missing.append(c)
        if missing:
            warnings.warn("Missing chromosomes in .cool: " + ", ".join(missing))
            warnings.warn("Available: " + ", ".join(all_chromnames))
        chromnames = ordered if ordered else all_chromnames
    else:
        chromnames = all_chromnames
    chromsizes = all_chromsizes.loc[chromnames]

    # matrix accessor
    mat_accessor = lib.matrix(balance=balance_flag)
    resol = lib.binsize

    # init getStripe object
    obj = getStripe(mat_accessor, resol, all_chromnames, chromnames, lib.chromsizes, chromsizes, numcores, bfilter)

    print("2. Expected value calculation ...")
    ev = obj.mpmean()

    print("3. Background distribution estimation ...")
    bgleft_up, bgright_up, bgleft_down, bgright_down = obj.nulldist()

    # evaluate candidates
    print("4. Evaluating candidate stripes ...")
    cand_pval = obj.extract(candidates, bgleft_up, bgright_up, bgleft_down, bgright_down)

    print("5. Stripiness calculation ...")
    print("Chromosomes after filtering:", obj.chromnames)
    res = obj.scoringstripes(cand_pval, ev, mask)
    assert isinstance(res, (list, tuple)) and len(res) > 0, "Unexpected return from scoringstripes"
    scores = res[0]
    cand_pval.insert(cand_pval.shape[1], "Stripiness", scores, True)

    cand_pval = cand_pval.sort_values(by=["Stripiness"], ascending=False)
    return cand_pval


def run(
    *,
    cool: Path,
    candidates_path: Path,
    outdir: Path,
    norm: str | bool = "None",
    chrom: str = "all",
    numcores: int = 10,
    mask: str = "0",
    bfilter: int = 3,
    stem: str = "processed_results",
    bed_axis: str = "col",
) -> Tuple[Path, Path, Path]:
    """
    High-level runner: load candidates -> BED/BEDPE -> compute stripiness -> write TSV.

    Returns
    -------
    (bedpe_path, bed_path, out_tsv_path)
    """
    # load candidates
    cand_df = read_tsv(candidates_path, REQUIRED_COLS)

    # write BED/BEDPE
    bedpe_path, bed_path = write_bedpe_and_bed(cand_df, outdir, stem, bed_axis=bed_axis)

    # chrom list
    chroms_for_calc = (",".join(unique_sorted_chroms(cand_df, "chr"))
                       if chrom.strip().lower() == "from_file"
                       else chrom)

    # compute
    out_df = compute_stripiness_table(
        cool_path=cool,
        candidates=cand_df,
        norm=norm,
        chroms=chroms_for_calc,
        numcores=numcores,
        mask=mask,
        bfilter=bfilter,
    )

    out_tsv = outdir / f"{stem}_add_stripiness_pvalue.tsv"
    outdir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_tsv, sep="\t", index=False)

    return bedpe_path, bed_path, out_tsv