import os
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

outdir = "/data_disk/liling/Stripe_topn4000"
os.makedirs(outdir, exist_ok=True)

expr_path = "/data_disk/liling/hires_embryo_combined_stripe_gene_matrix_1102.tsv"
raw_df = pd.read_csv(expr_path, sep="\t", index_col=0)

gene_expr_df = raw_df[raw_df.index.str.startswith("stripe")].copy()
gene_expr_df.index = gene_expr_df.index.str.replace("^stripe[_-]", "", regex=True)

meta_df = pd.read_csv("/data_disk/liling/GSE223917_HiRES_emb_metadata.tsv", sep="\t")
meta_df["Cell.type"] = meta_df["Cell type"].str.replace(" ", "_")
meta_df = meta_df.set_index("Cellname")

common_cells = list(set(gene_expr_df.columns) & set(meta_df.index))
gene_expr_df = gene_expr_df[common_cells]
meta_df = meta_df.loc[common_cells]

# === Binarization ===
binary_matrix = gene_expr_df.gt(gene_expr_df.median(axis=1), axis=0).astype(int)
expr_frac = binary_matrix.mean(axis=1)
filtered = binary_matrix[(expr_frac > 0.05) & (expr_frac < 0.95)] 

ad = sc.AnnData(filtered.T)
ad.obs = meta_df.loc[ad.obs_names].copy()
ad.obs["Cell.type"] = pd.Categorical(ad.obs["Cell.type"])
ad.raw = ad.copy()

ad.obs["celltype"] = ad.obs["Cell.type"]  
ad = ad[~ad.obs["celltype"].isna()].copy()

ad.obs["celltype"] = ad.obs["celltype"].astype("category")
ad.obs["celltype"] = ad.obs["celltype"].cat.remove_unused_categories()


sc.pp.normalize_total(ad, target_sum=1e4)
sc.pp.log1p(ad)
sc.pp.highly_variable_genes(ad, n_top_genes=4000, subset=False)
sc.tl.rank_genes_groups(ad, groupby="celltype", method="t-test", only_positive=True)

markers_df = sc.get.rank_genes_groups_df(ad, group=None)
markers_df.to_csv(f"{outdir}/filtered_0_marker_genes.tsv", sep="\t", index=False)
markers_df_logfc = markers_df[markers_df["logfoldchanges"] > 0].copy()

significant_markers_df_logfc = markers_df_logfc[markers_df_logfc["pvals_adj"] < 0.05].copy()
top10_df = significant_markers_df_logfc.groupby("group").apply(
    lambda x: x.nsmallest(10, "pvals_adj")
).reset_index(drop=True)

top10_genes = top10_df["names"].unique().tolist()
top10_df[["names", "group"]].rename(columns={"names": "gene", "group": "cluster"}).to_csv(
    f"{outdir}/top10_gene_cluster_map.tsv", sep="\t", index=False
)


selected_celltypes = [
  "early_mesenchyme", "early_mesoderm","intermediate_mesoderm",
  "mix_late_mesenchyme", "myocytes", "neural_ectoderm", "neural_tube",
  "early_neurons", "radial_glias", "oligodendrocytes_and_progenitors",
  "notochord", "NMP", "epithelial_cells", "epiblast_and_PS", "mitosis",
  "ExE_mesoderm", "endoderm", "schwann_cell_precursors",
  "ExE_endoderm", "ExE_ectoderm",
  "blood"
]  

ad_sub = ad[ad.obs["celltype"].isin(selected_celltypes)].copy()

ad_sub.obs["celltype"] = ad_sub.obs["celltype"].map(
    lambda x: x.replace("_", " ").capitalize()
)
ad_sub.obs["celltype"] = pd.Categorical(ad_sub.obs["celltype"])
ad_sub.obs["celltype"] = ad_sub.obs["celltype"].astype("category")
ad_sub.obs["celltype"] = ad_sub.obs["celltype"].cat.remove_unused_categories()

sc.pp.scale(ad_sub, zero_center=True, max_value=5)

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import matplotlib.font_manager as fm


ARIAL_PATH = "/data_disk/liling/packages/Arial.ttf"
if os.path.exists(ARIAL_PATH):
    fm.fontManager.addfont(ARIAL_PATH)
    ARIAL = fm.FontProperties(fname=ARIAL_PATH).get_name()
else:
    ARIAL = "Arial"


mpl.rcParams.update({
    "font.family": ARIAL,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 100,
})

def style_like_before(fig):
    for ax in fig.axes:
        ax.tick_params(axis="both", which="major", length=2.0, width=0.5,
                       direction="out", labelsize=7.2, color="black", labelcolor="black")
        for sp in ["left", "bottom"]:
            if sp in ax.spines:
                ax.spines[sp].set_linewidth(0.6)
                ax.spines[sp].set_color("black")
        for sp in ["right", "top"]:
            if sp in ax.spines:
                ax.spines[sp].set_visible(False)
        if ax.get_xlabel(): ax.set_xlabel(ax.get_xlabel(), fontsize=7.2, fontfamily=ARIAL)
        if ax.get_ylabel(): ax.set_ylabel(ax.get_ylabel(), fontsize=7.2, fontfamily=ARIAL)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontsize(7.2)
            t.set_fontfamily(ARIAL)
        for t in ax.get_xticklabels():
            t.set_rotation(45)
            t.set_ha('right')
            t.set_rotation_mode('anchor')  
    for t in fig.texts:
        t.set_fontsize(7.2)
        t.set_fontfamily(ARIAL)

for rng in [2, 3, 4, 5]:
    ax = sc.pl.heatmap(
        ad_sub,
        var_names=top10_genes,
        groupby="celltype",
        use_raw=False,
        show=False,
        cmap="YlGnBu",
        vmin=-rng, vmax=rng,
        swap_axes=True,        
        dendrogram=False,
        figsize=(12, 4)        
    )
    fig = plt.gcf()
    style_like_before(fig)
    plt.tight_layout()
    plt.savefig(f"{outdir}/scale_range0{rng}_group4_heatmap.pdf",
                bbox_inches="tight", dpi=300)
    plt.close(fig)

top10_df[["names", "group"]].rename(columns={"names": "gene", "group": "cluster"}).to_csv(
    f"{outdir}/top10_gene_cluster_map.tsv", sep="\t", index=False
)

