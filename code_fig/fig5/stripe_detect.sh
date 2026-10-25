scstripe detect \
    --input /data_disk/data/liling/scMicroC_data/EBF1_submat_5kb/submat.txt \
    --chrom chr5 \
    --norm weight \
    --output /data_disk/data/liling/scMicroC_data/EBF1_submat_5kb_wid6 \
    --split-length 400 \
    --step-size 100 \
    --max-width 6 \
    --min-length 40
scstripe postprocess \
    --results-dir /data_disk/data/liling/scMicroC_data/EBF1_submat_5kb_wid6 \
    --resolution 5000 \
    --split-length 400 \
    --step-size 100