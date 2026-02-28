# scStripe/cli.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import typer

from .stripe_detect import run_pipeline
from .stripe_score import run_stripe_scores
from .stripe_post import process_all
from .add_stripiness import run as run_stripiness 

app = typer.Typer(add_completion=False, help="Stripe toolkit CLI")

@app.command("detect")
def detect(
    input: Path = typer.Option(..., "--input", help="Path to input contact matrix (.txt or .cool)"),
    output: Path = typer.Option(..., "--output", "-o", help="Output base directory (results will go into OUTPUT/CHROM/)"),
    penalty: float = typer.Option(0.1, "--penalty"),
    fold_threshold1: float = typer.Option(1.3, "--fold-threshold1"),
    split_length: int = typer.Option(200, "--split-length"),
    step_size: int = typer.Option(50, "--step-size"),
    max_width: int = typer.Option(8, "--max-width"),
    min_length: int = typer.Option(20, "--min-length"),
    p_thresh_wid: float = typer.Option(1e-3, "--p-thresh-wid"),
    p_thresh_len: float = typer.Option(5e-2, "--p-thresh-len"),
    fc_thresh_wid: float = typer.Option(1.1, "--fc-thresh-wid"),
    chrom: str = typer.Option(..., "--chrom", help="Chromosome name, e.g. 'chr1'"),
    cool_norm: str = typer.Option("weight", "--norm", help="'weight' | 'None' | a bins() column in the .cool"),
):
    """
    Run stripe detection on a contact matrix.
    - If --input is a .cool file: we will fetch the submatrix of the given --chrom.
    - Results are written under OUTPUT/CHROM/...
    """
    out_path = run_pipeline(
        input_matrix=input,
        output_dir=output,         
        penalty=penalty,
        fold_threshold1=fold_threshold1,
        split_length=split_length,
        step_size=step_size,
        max_width=max_width,
        min_length=min_length,
        p_thresh_wid=p_thresh_wid,
        p_thresh_len=p_thresh_len,
        fc_thresh_wid=fc_thresh_wid,
        chrom=chrom,           
        cool_norm=cool_norm,
    )
    typer.echo(f"[OK] Finished. Results written to: {out_path}")


@app.command("postprocess")
def postprocess(
    results_dir: Path = typer.Option(..., "--results-dir", help="Root directory containing per-chromosome results (chr1/, chr2/, ...)"),
    split_length: int = typer.Option(200, "--split-length", help="Split size used during detection"),
    resolution: int = typer.Option(10000, "--resolution", help="Bin size used in detection"),
    step_size: int = typer.Option(50, "--step-size", help="Step size used during detection"),
    n_jobs: int = typer.Option(4, "--n-jobs", help="Parallel jobs for redundancy removal"),
):
    """Organize chromosome-wise results, filter, and output."""
    raw_tsv, final_tsv = process_all(
        results_dir=results_dir,
        split_length=split_length,
        resolution=resolution,
        step_size=step_size,
        n_jobs=n_jobs,
    )
    typer.echo(f"[OK] Raw combined table: {raw_tsv}")
    typer.echo(f"[OK] Final deduplicated table: {final_tsv}")

@app.command("score-cells")
def score_cells(
    celltype: str = typer.Option(..., "--celltype"),
    pairs_dir: Path = typer.Option(..., "--pairs-dir"),
    meta_file: Path = typer.Option(..., "--meta-file"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    stripe_file: Path = typer.Option(..., "--stripe-file",
        help="Path to stripe result file for the given celltype (columns: chr,pos1,pos2,chr2,pos3,pos4)"),
    chr1_col: int = typer.Option(..., "--chr1-col", help="0-based index in pairs txt"),
    pos1_col: int = typer.Option(..., "--pos1-col", help="0-based index in pairs txt"),
    chr2_col: int = typer.Option(..., "--chr2-col", help="0-based index in pairs txt"),
    pos2_col: int = typer.Option(..., "--pos2-col", help="0-based index in pairs txt"),
    q1: float = typer.Option(0.0, "--q1", help="Fraction along stripe (distal end) to begin the subregion (0..1)"),
    q2: float = typer.Option(0.5, "--q2", help="Fraction along stripe (distal end) to end the subregion (0..1), must be > q1"),
    flanking: str = typer.Option("both", "--flanking", help="Which flanking region to use: both / inside / outside"),
    n_jobs: int = typer.Option(40, "--n-jobs", help="Parallel workers"),
):
    """Compute stripe scores per cell for a given cell type using per-cell pairs .txt files."""

    out = run_stripe_scores(
        celltype=celltype,
        pairs_dir=pairs_dir,
        meta_file=meta_file,
        output_dir=output_dir,
        stripe_file=stripe_file,
        chr1_col=chr1_col,
        pos1_col=pos1_col,
        chr2_col=chr2_col,
        pos2_col=pos2_col,
        q1=q1,
        q2=q2,
        flanking=flanking,
        n_jobs=n_jobs,
    )
    typer.echo(f"[OK] Saved to: {out}")

@app.command("add-stripiness")
def add_stripiness_cmd(
    cool: Path = typer.Option(..., "--cool", help="Path to input .cool file"),
    stripe_file: Path = typer.Option(..., "--stripe-file", help="Path to stripes TSV"),
    output: Path = typer.Option(..., "--output", help="Path to output TSV (stripiness + p-values)"),
    norm: str = typer.Option("None", "--cool-norm", help="None | weight | <bins column in .cool>"),
    chrom: str = typer.Option("from_file", "--chrom", help="'all', comma list, or 'from_file'"),
    numcores: int = typer.Option(10, "--numcores", help="Parallel jobs"),
    mask: str = typer.Option("0", "--mask", help="Mask region like 'chr1:100000-200000' or '0'"),
    bfilter: int = typer.Option(3, "--bfilter", help="Background filter level"),
):
    """
    Compute Stripenn stripiness & p-values for stripes.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    out_path = run_stripiness(
        cool=cool,
        stripe_file=stripe_file,
        stripes_add_stripiness_pvalue_path=output,
        norm=norm,
        chrom=chrom,
        numcores=numcores,
        mask=mask,
        bfilter=bfilter,
    )

    typer.echo(f"[OK] Stripiness TSV: {out_path}")


def main():
    app()

if __name__ == "__main__":
    main()