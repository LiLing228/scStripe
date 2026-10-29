# Tiled Per-Cell Scoring of the EBF1 Stripe

## Overview

This tutorial divides the EBF1 stripe into six non-overlapping subregions and calculates a per-cell stripe score for each subregion with `scstripe score-cells`.

The six analyzed intervals are:

```text
0.00-0.20
0.20-0.40
0.40-0.50
0.50-0.60
0.60-0.70
0.70-0.80
```


## Input

Three inputs are required:

1. A directory containing one raw contact-pair file per cell.
2. A metadata file containing the cells to be analyzed.
3. A stripe file containing the EBF1 stripe coordinates.

The contact-pair files must contain chromosome and position columns. In this example, the first four columns are:

```text
chr1    pos1    chr2    pos2
chr11   120000  chr11   640000
chr11   125000  chr11   645000
```

The cell identifiers in the metadata file must correspond to the cell-level contact files expected by scStripe.

## Run tiled per-cell scoring

```bash
#!/usr/bin/env bash
set -euo pipefail

# Input
PAIRS_DIR="/path/to/per_cell_pairs"
METADATA_FILE="/path/to/metadata.tsv"
STRIPE_FILE="/path/to/processed_results.tsv"

# Output
OUTPUT_DIR="ebf1_tiled_scores"

# Contact-pair column positions
CHR1_COL=1
POS1_COL=2
CHR2_COL=3
POS2_COL=4

# q1, q2, and output label for each EBF1 subregion
TILES=(
  "0.00 0.20 0_20per"
  "0.20 0.40 20_40per"
  "0.40 0.50 40_50per"
  "0.50 0.60 50_60per"
  "0.60 0.70 60_70per"
  "0.70 0.80 70_80per"
)

mkdir -p "${OUTPUT_DIR}"

for tile in "${TILES[@]}"; do
  read -r q1 q2 label <<< "${tile}"

  tile_dir="${OUTPUT_DIR}/${label}"
  mkdir -p "${tile_dir}"

  echo "Scoring EBF1 subregion ${q1}-${q2}"

  scstripe score-cells \
    --q1 "${q1}" \
    --q2 "${q2}" \
    --flanking both \
    --celltype scMicroC \
    --pairs-dir "${PAIRS_DIR}" \
    --meta-file "${METADATA_FILE}" \
    --stripe-file "${STRIPE_FILE}" \
    --output-dir "${tile_dir}" \
    --chr1-col "${CHR1_COL}" \
    --pos1-col "${POS1_COL}" \
    --chr2-col "${CHR2_COL}" \
    --pos2-col "${POS2_COL}"

  source_file="${tile_dir}/stripe_score_scMicroC.tsv"
  output_file="${OUTPUT_DIR}/stripe_score_scMicroC_${label}.tsv"

  if [[ ! -s "${source_file}" ]]; then
    echo "ERROR: expected output not found: ${source_file}" >&2
    exit 1
  fi

  mv "${source_file}" "${output_file}"
done
```

## Output

The script produces six per-cell stripe-score files:

```text
stripe_score_scMicroC_0_20per.tsv
stripe_score_scMicroC_20_40per.tsv
stripe_score_scMicroC_40_50per.tsv
stripe_score_scMicroC_50_60per.tsv
stripe_score_scMicroC_60_70per.tsv
stripe_score_scMicroC_70_80per.tsv
```

Together, the six files provide one local EBF1 stripe-score profile per cell and can be used for downstream clustering or visualization.

## Downstream grouping

The six tiled stripe scores were used for hierarchical clustering. Cells were first divided into 13 preliminary clusters, which were then manually combined into seven final groups based on their stripe-score patterns. A subregion was considered retained when its stripe score was at least 1.5.

## Group-average contact maps

After assigning cells to the seven final groups, average their contact matrices to visualize the contact pattern of each group.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

groups = pd.read_csv("cell_to_group.tsv", sep="\t")
group_ids = sorted(groups["group"].unique())
fig, axes = plt.subplots(1, len(group_ids), figsize=(2.2 * len(group_ids), 2.2))

for ax, group_id in zip(np.atleast_1d(axes), group_ids):
    cells = groups.loc[groups["group"] == group_id, "cell"]
    matrices = [
        np.loadtxt(f"per_cell_matrices/{cell}.tsv", delimiter="\t")
        for cell in cells
    ]

    average = np.mean([m + m.T for m in matrices], axis=0)
    ax.imshow(average, cmap="rainbow")
    ax.set_title(f"Group {group_id}")
    ax.axis("off")

fig.savefig("group_average_contact_maps.pdf", bbox_inches="tight")
plt.close(fig)
```
