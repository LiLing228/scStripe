========
scStripe
========
--------------------------------------------------------------------
A toolkit for stripe detection and analysis in single-cell Hi-C (scHi-C) data. 
--------------------------------------------------------------------

Contents
########
* Introduction
* Installation
* Example run
* Usage

## Introduction

scStripe is an imputation-free two-stage statistical framework for stripe detection and quantification from single-cell Hi-C data. 
In stage~1, pseudo-bulk contact maps are analyzed to identify candidate stripe endpoints using a Random-Matrix-Theory–guided spectral selection, followed by endpoint assembly and statistical validation of candidate stripes. 
In stage~2, these validated pseudo-bulk stripes serve as a reference to compute per-cell stripe scores directly from raw, unbalanced single-cell contact matrices, thereby quantifying the presence and strength of each stripe across individual cells.

Requirement
############
**Python 3.8 or higher version**


Installation
############

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/LiLing228/scStripe.git
   cd scStripe
   pip install -e .


## Example run
#########################
Let's check if scstripe is working or not with a simple example. This example .cool file is Blood_10kb of only chr19 of mouse (`Liu et al., Science, 2025 <(https://www.science.org/doi/10.1126/science.adg3797)>`_).
::

   cd <Test_Directory> # Move to your test directory
   wget https://www.dropbox.com/scl/fi/5uxbr6lu4sjt0x0n4hfcu/Blood_10kb.cool?rlkey=2fl6dyzhcby0rbr3gt13gmtpb&st=ebjkayev&dl=1 -O test.cool --no-check-certificate
   scstripe detect --input test.cool --chrom chr19 --norm weight --output <Test_Directory> --split-length 200 --step-size 50 --max-width 8 --min-length 20
   scstripe postprocess --results-dir <Test_Directory> --resolution 10000 --split-length 200 --step-size 50

**Output1**
An example of the post-processed stripe summary table is shown below:
.. csv-table:: processed_results.tsv
   :header: "chr","pos1","pos2","chr2","pos3","pos4","length","width","split_mat_id"

   chr19	4930001	5010000	chr19	4400001	5010000	610000	80000	9
   chr19	5850001	5910000	chr19	5480001	5910000	430000	60000	10
   chr19	5910001	5990000	chr19	5910001	6380000	470000	80000	12
   chr19	7370001	7450000	chr19	6900001	7450000	550000	80000	12
   chr19	7210001	7290000	chr19	7210001	7500000	290000	80000	13
   chr19	7020001	7100000	chr19	7020001	7500000	480000	80000	14
   chr19	9070001	9140000	chr19	8460001	9140000	680000	70000	16
   chr19	9400001	9480000	chr19	9400001	9860000	460000	80000	19
   chr19	9850001	9930000	chr19	9850001	10980000	1130000	80000	19

Each line represents the coordinates of a stripe.

.. image:: https://github.com/LiLing228/scStripe/blob/main/image/github_x1.pdf

* chr: chromosome
* pos1: x1 position of stripe
* pos2: x2 position of stripe
* chr2: chromosome (same as chr_1)
* pos3: y1 position of stripe
* pos4: y2 position of stripe
* length: Length of vertical stripe (y2-y1+1)
* width: Width of vertical stripe (x2-x1+1)
* split_mat_id: Index of the sub-matrix (generated during window splitting) within which this stripe was detected.


::

   # download
   cd <Test_Directory>
   wget https://www.dropbox.com/scl/fi/03bf33khvuavwwf7b7j68/Blood_metadata.txt?rlkey=ccd5df79njsc8p6tpcx9laldm&st=xwolg695&dl=1 -O Blood_metadata.txt --no-check-certificate
   wget https://www.dropbox.com/scl/fi/qno3gfg5mcx1zwkevy347/Blood_pairs_txt.tar.gz?rlkey=2z7klftlctrhissizp8008t5b&st=2lqqcu9y&dl=1 -O Blood_pairs_txt.tar.gz --no-check-certificate
   
   tar -xzf Blood_pairs_txt.tar.gz && find embryo_pairs_by_cell -name "*.txt.gz" -exec gunzip {} \;

   PAIRS_DIR=<Test_Directory>/embryo_pairs_by_cell
   META=<Test_Directory>/Blood_metadata.txt
   STRIPE_FILE=<Test_Directory>/processed_results.tsv
   OUTROOT=<Test_Directory>
   scstripe score-cells --celltype Blood --meta-file ${META} \
                        --stripe-file ${STRIPE_FILE} \
                        --pairs-dir ${PAIRS_DIR} \
                        --chr1-col 1 --pos1-col 2 --chr2-col 3 --pos2-col 4 \
                        --n-jobs 40 --output-dir ${OUTROOT} 
    
**Output2**
An example of the single-cell stripe score summary table is shown below:
.. csv-table:: stripe_score_Blood.tsv
   :header: "chr","pos1","pos2","chr2","pos3","pos4","length","width","split_mat_id","GasaE751008","GasaE751023","GasaE751026","GasaE751027","GasaE751028","GasaE751029"

   "chr19","4930001","5010000","chr19","4400001","5010000","610000","80000","9","0.67","1.0","2.0","1.0","1.2","2.0"
   "chr19","5850001","5910000","chr19","5480001","5910000","430000","60000","10","0.33","1.0","1.0","0.5","1.0","3.0"
   "chr19","5910001","5990000","chr19","5910001","6380000","470000","80000","12","0.57","1.0","1.0","0.5","1.0","1.0"
   "chr19","7370001","7450000","chr19","6900001","7450000","550000","80000","12","1.0","1.0","1.33","3.0","0.67","1.0"
   "chr19","7210001","7290000","chr19","7210001","7500000","290000","80000","13","1.08","0.5","0.67","0.33","1.0","0.5"
   "chr19","7020001","7100000","chr19","7020001","7500000","480000","80000","14","1.33","0.67","0.67","1.67","0.67","1.0"
   "chr19","9070001","9140000","chr19","8460001","9140000","680000","70000","16","2.0","1.0","1.2","1.0","1.0","1.0"

Each line represents the coordinates of a stripe.




## Usage

scStripe has four functions.

* detect
* postprocess
* score-cells
* add-stripiness

**detect**
:Matrix segmentation and changepoint-based detection; supports both `.txt` contact matrices and `.cool` files.  

Options:
  --input PATH              Path to input contact matrix (.txt or .cool).
                            [required]

  -o, --output PATH         Output base directory. Results will be written
                            into OUTPUT/CHROM/.  [required]

  --chrom TEXT              Chromosome name, e.g. 'chr1'.  [required]

  --norm TEXT               Normalization method for .cool input. It should be
                            'weight', 'None', or the name of a column in
                            cooler.bins() (e.g., KR, VC, VC_SQRT).
                            [default: weight]

  --penalty FLOAT           Penalty parameter for change-point detection
                            (larger values yield fewer segments).
                            [default: 0.1]

  --fold-threshold1 FLOAT   Fold-change threshold used to pre-select stripe
                            candidates.  [default: 1.3]

  --split-length INTEGER    Window length (in bins) used to tile each
                            chromosome into sub-matrices.  [default: 200]

  --step-size INTEGER       Step size (in bins) between adjacent windows
                            (controls window overlap).  [default: 50]

  --max-width INTEGER       Maximum stripe width (in bins).  [default: 8]

  --min-length INTEGER      Minimum stripe length (in bins).  [default: 20]

  --p-thresh-wid FLOAT      P-value threshold for stripe width.  [default:
                            1e-3]

  --p-thresh-len FLOAT      P-value threshold for stripe length.  [default:
                            5e-2]

  --fc-thresh-wid FLOAT     Fold-change threshold for stripe width.
                            [default: 1.1]

  --fc-thresh-len FLOAT     Fold-change threshold for stripe length.
                            [default: 3.0]

  --add-dip TEXT            Whether to also detect “dip” patterns inside
                            stripes ('Y' or 'N').  [default: N]




**postprocess**
:Merge results from chromosome-level detection, remove redundant stripes, and produce unified outputs.  
Options:
  --results-dir PATH     Root directory containing per-chromosome results
                         (subdirectories such as chr1/, chr2/, ...).
                         [required]

  --split-length INTEGER Split size (in bins) used during detection. Must
                         match the value used in `scstripe detect`.
                         [default: 200]

  --resolution INTEGER   Bin size (in base pairs) used in detection. Must
                         match the resolution of the input Hi-C matrix
                         (e.g., 10000 for 10 kb).  [default: 10000]

  --step-size INTEGER    Step size (in bins) between adjacent windows used
                         during detection. Must match the value used in
                         `scstripe detect`.  [default: 50]

  --n-jobs INTEGER       Number of parallel jobs used for redundancy removal
                         and stripe merging.  [default: 4]


**score-cells**
:Compute stripe scores for each cell from per-cell pairs files.  
Options:
  --celltype TEXT        Cell type label for which cell-wise stripe scores
                         will be computed.  [required]

  --pairs-dir PATH       Directory containing per-cell pairs files
                         (e.g., one <cell>.txt per cell).  [required]

  --meta-file PATH       Metadata table describing cells (e.g., Cellname and
                         Celltype annotation).  [required]

  --output-dir PATH      Output directory for per-cell stripe scores and
                         summary tables.  [required]

  --stripe-file PATH     Path to stripe result file for the given cell type
                         (columns: chr,pos1,pos2,chr2,pos3,pos4).  [required]

  --chr1-col INTEGER     0-based column index of chr1 in the pairs .txt file.
                         [required]

  --pos1-col INTEGER     0-based column index of pos1 in the pairs .txt file.
                         [required]

  --chr2-col INTEGER     0-based column index of chr2 in the pairs .txt file.
                         [required]

  --pos2-col INTEGER     0-based column index of pos2 in the pairs .txt file.
                         [required]

  --q1 FLOAT             Fraction along the stripe (measured from the distal
                         end) at which the scoring subregion begins
                         (0 ≤ q1 < q2 ≤ 1).  [default: 0.0]

  --q2 FLOAT             Fraction along the stripe (measured from the distal
                         end) at which the scoring subregion ends
                         (must be > q1).  [default: 0.5]

  --n-jobs INTEGER       Number of parallel workers used for per-cell
                         scoring.  [default: 40]


**add-stripiness**
:Evaluate stripiness and p-values (two metrics originally proposed in Stripenn).  
These metrics can also be computed using the `stripenn score` command  
(`Yoon et al., Nature Communications, 2022 <(https://www.nature.com/articles/s41467-022-29258-9)>`_).
Options:
  --cool PATH            Path to input .cool file.  [required]

  --stripe-file PATH     Path to the stripe summary table (TSV) for which
                         stripiness scores and p-values will
                         be computed.  [required]

  --output PATH          Path to output TSV file containing the original
                         stripe table augmented with stripiness metrics
                         and p-values.  [required]

  --cool-norm TEXT       Normalization to use for the .cool matrix:
                         "None", "weight", or the name of a column in
                         cooler.bins() (e.g., KR, VC, VC_SQRT).
                         [default: None]

  --chrom TEXT           Chromosome(s) to process: "all" (all chromosomes),
                         a comma-separated list (e.g., "chr1,chr2,chr19"),
                         or "from_file" to infer from the stripe_file.
                         [default: from_file]

  --numcores INTEGER     Number of parallel jobs used to compute stripiness.
                         [default: 10]

  --mask TEXT            Genomic region to be masked when computing
                         background, e.g. "chr1:100000-200000"; use "0"
                         to disable masking.  [default: 0]

  --bfilter INTEGER      Background filter level (kernel size) applied to
                         smooth local contact frequencies.  [default: 3]


