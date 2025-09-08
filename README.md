# scStripe

**scStripe** is a toolkit for stripe detection and analysis in single-cell Hi-C (scHi-C) data.  

---

## Features

- **Stripe Detection**: Matrix segmentation and changepoint-based detection; supports both `.txt` contact matrices and `.cool` files.  
- **Post-processing**: Merge results from chromosome-level detection, remove redundant/overlapping stripes, and produce unified outputs (BED/BEDPE optional).  
- **Cell-level Scoring**: Compute stripe scores for each cell from per-cell pairs files.  
- **Statistical Analysis**: Evaluate stripiness and p-values (two metrics originally proposed in Stripenn).  

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/LiLing228/scStripe.git
cd scStripe
pip install -e .
```

---

## Usage

After installation, the command-line tool `scstripe` will be available.

### 1. Stripe Detection
```bash
scstripe detect \
  --input /path/to/sample_10kb.cool \
  --chrom chr1 \
  --cool-norm weight \
  --output /path/to/outdir \
  --split-length 200 --step-size 50 \
  --max-width 8 --min-length 20
```

### 2. Post-processing
```bash
scstripe postprocess \
  --results-dir /path/to/outdir \
  --resolution 10000 \
  --split-length 200 \
  --step-size 50
```

### 3. Cell-level Scoring
```bash
scstripe score-cells \
  --celltype Astrocytes \
  --pairs-dir /path/to/pairs_txts \
  --meta-file /path/to/metadata.tsv \
  --output-dir /path/to/outdir \
  --stripe-file /path/to/outdir/processed_results.tsv \
  --chr1-col 0 --pos1-col 1 --chr2-col 2 --pos2-col 3
```

### 4. Statistical Analysis
```bash
scstripe add-stripiness \
  --cool /path/to/sample_10kb.cool \
  --stripe-file /path/to/outdir/processed_results.tsv \
  --output /path/to/outdir/processed_results_add_stripiness_pvalue.tsv \
  --cool-norm weight \
  --chrom from_file
```

---


