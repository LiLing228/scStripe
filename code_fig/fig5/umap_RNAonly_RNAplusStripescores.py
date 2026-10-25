import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


EXPRESSION_FILE = "/data_disk/liling/hires_embryo_combined_stripe_gene_matrix_1102.tsv"
METADATA_FILE = "/data_disk/liling/GSE223917_HiRES_emb_metadata.tsv"
OUTPUT_DIR = "/data_disk/liling/fig5/grid_rna_stripe"

MIN_GENES = 2500
MIN_CELLS = 300
N_TOP_FEATURES = 5000
TOTAL_DIM = 35
GENE_DIM = 20
STRIPE_DIM = TOTAL_DIM - GENE_DIM
N_PCS = 128
N_NEIGHBORS = 35
RESOLUTION = 1.5
RANDOM_SEED = 0

TARGET_CELLTYPES = {
    "early_mesenchyme",
    "early_mesoderm",
    "intermediate_mesoderm",
    "mix_late_mesenchyme",
    "myocytes",
    "neural_ectoderm",
    "neural_tube",
    "early_neurons",
    "radial_glias",
    "oligodendrocytes_and_progenitors",
    "notochord",
    "NMP",
    "epithelial_cells",
    "epiblast_and_PS",
    "mitosis",
    "ExE_mesoderm",
    "endoderm",
    "schwann_cell_precursors",
    "ExE_endoderm",
    "ExE_ectoderm",
    "blood",
}

CELLTYPE_PALETTE = {
    "mitosis": "#b9babc",
    "blood": "#f3995c",
    "ExE_endoderm": "#f4be79",
    "ExE_ectoderm": "#bdabcd",
    "epiblast_and_PS": "#a6d29d",
    "neural_ectoderm": "#a3c1da",
    "NMP": "#94b4d0",
    "neural_tube": "#85a7c5",
    "notochord": "#7298bb",
    "radial_glias": "#668db2",
    "oligodendrocytes_and_progenitors": "#577faa",
    "early_neurons": "#557aa6",
    "schwann_cell_precursors": "#466e9f",
    "early_mesoderm": "#f29584",
    "ExE_mesoderm": "#e48b7d",
    "early_mesenchyme": "#d87c72",
    "intermediate_mesoderm": "#ca6e69",
    "myocytes": "#c0635f",
    "mix_late_mesenchyme": "#b65857",
    "endoderm": "#f59c62",
    "epithelial_cells": "#f09eb0",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_label(label):
    words = str(label).replace("_", " ").split()

    if words and words[0] not in {"NMP", "ExE"}:
        words[0] = words[0].capitalize()

    return " ".join(words)


def get_palette(values, preferred=None):
    labels = sorted(pd.Series(values).astype(str).unique())

    if preferred is None:
        colors = sns.color_palette("tab20", len(labels))
        return labels, dict(zip(labels, colors))

    order = [label for label in preferred if label in labels]
    extras = [label for label in labels if label not in preferred]

    palette = {label: preferred[label] for label in order}
    palette.update(dict(zip(extras, sns.color_palette("tab20", len(extras)))))

    return order + extras, palette


def save_legend(order, palette, output_file):
    fig, ax = plt.subplots(figsize=(2.0, max(2.5, 0.25 * len(order))))
    ax.axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=palette[label],
            markeredgecolor="none",
            label=format_label(label),
        )
        for label in order
    ]

    ax.legend(handles=handles, loc="center", frameon=False, fontsize=7)
    fig.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_umap(adata, color_key, output_file, title, palette=None):
    data = pd.DataFrame(
        adata.obsm["X_umap"],
        columns=["UMAP_1", "UMAP_2"],
        index=adata.obs_names,
    )
    data["label"] = adata.obs[color_key].astype(str).values

    order, colors = get_palette(data["label"], palette)

    fig, ax = plt.subplots(figsize=(3, 3))

    sns.scatterplot(
        data=data,
        x="UMAP_1",
        y="UMAP_2",
        hue="label",
        hue_order=order,
        palette=colors,
        s=5,
        linewidth=0,
        legend=False,
        rasterized=True,
        ax=ax,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    save_legend(
        order,
        colors,
        output_file.replace(".pdf", "_legend.pdf"),
    )


def preprocess(adata):
    adata = adata.copy()

    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)

    if adata.n_obs < 3 or adata.n_vars < 2:
        raise ValueError("Too few cells or features after filtering.")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_top = min(N_TOP_FEATURES, adata.n_vars)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top,
        flavor="seurat",
    )

    if adata.var["highly_variable"].sum() > 0:
        adata = adata[:, adata.var["highly_variable"]].copy()

    max_pcs = min(N_PCS, adata.n_obs - 1, adata.n_vars)

    if max_pcs < 2:
        raise ValueError("Too few principal components.")

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(
        adata,
        n_comps=max_pcs,
        svd_solver="arpack",
        random_state=RANDOM_SEED,
    )

    return adata


def knn_purity(embedding, labels, k):
    labels = np.asarray(labels).astype(str)

    if len(labels) < 2:
        return np.nan

    k = min(k, len(labels) - 1)

    neighbors = NearestNeighbors(n_neighbors=k + 1)
    indices = neighbors.fit(embedding).kneighbors(
        embedding,
        return_distance=False,
    )[:, 1:]

    return float((labels[indices] == labels[:, None]).mean())


def silhouette(embedding, labels):
    labels = np.asarray(labels)
    groups, counts = np.unique(labels, return_counts=True)

    if len(groups) < 2 or np.any(counts < 2):
        return np.nan

    return float(silhouette_score(embedding, labels))


def run_method(name, embedding, celltypes):
    adata = sc.AnnData(embedding)
    adata.obs["celltype"] = np.asarray(celltypes).astype(str)

    n_neighbors = min(N_NEIGHBORS, adata.n_obs - 1)

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X",
    )

    sc.tl.louvain(
        adata,
        resolution=RESOLUTION,
        random_state=RANDOM_SEED,
        key_added="cluster",
    )

    sc.tl.umap(
        adata,
        min_dist=0.01,
        spread=5,
        random_state=RANDOM_SEED,
    )

    labels = adata.obs["celltype"].to_numpy()
    clusters = adata.obs["cluster"].to_numpy()

    metrics = {
        "method": name,
        "n_cells": adata.n_obs,
        "ARI": adjusted_rand_score(labels, clusters),
        "NMI": normalized_mutual_info_score(labels, clusters),
        "FMI": fowlkes_mallows_score(labels, clusters),
        "ASW": silhouette(adata.X, labels),
        "kNN_purity": knn_purity(adata.X, labels, n_neighbors),
    }

    method_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(method_dir, exist_ok=True)

    metric_text = (
        f'ARI={metrics["ARI"]:.3f}, '
        f'NMI={metrics["NMI"]:.3f}, '
        f'FMI={metrics["FMI"]:.3f}'
    )

    plot_umap(
        adata,
        "celltype",
        os.path.join(method_dir, "umap_celltype.pdf"),
        f"{name}: {metric_text}",
        CELLTYPE_PALETTE,
    )

    plot_umap(
        adata,
        "cluster",
        os.path.join(method_dir, "umap_cluster.pdf"),
        f"{name}: Louvain clusters",
    )

    return metrics


print("Loading data...")

expression = pd.read_csv(
    EXPRESSION_FILE,
    sep="\t",
    index_col=0,
)

metadata = pd.read_csv(
    METADATA_FILE,
    sep="\t",
)

expression = expression.iloc[:, 1:]
expression.columns = expression.columns.astype(str)

metadata["Cellname"] = metadata["Cellname"].astype(str)
metadata["celltype"] = (
    metadata["Cell type"].astype(str).str.replace(" ", "_")
)

metadata = metadata[
    metadata["celltype"].isin(TARGET_CELLTYPES)
].set_index("Cellname")

common_cells = expression.columns.intersection(metadata.index)
expression = expression.loc[:, common_cells]
metadata = metadata.loc[common_cells]

stripe_mask = expression.index.str.startswith("stripe")

stripe_matrix = expression.loc[stripe_mask].astype(float)
gene_matrix = expression.loc[~stripe_mask].astype(float)

stripe_matrix = stripe_matrix.mask(stripe_matrix.eq(1), 0.0)

stripe_adata = sc.AnnData(stripe_matrix.T)
gene_adata = sc.AnnData(gene_matrix.T)

stripe_adata.obs["celltype"] = metadata["celltype"].values
gene_adata.obs["celltype"] = metadata["celltype"].values

stripe_adata.var_names = stripe_matrix.index.astype(str)
gene_adata.var_names = gene_matrix.index.astype(str)

print("Preprocessing stripe data...")
stripe_adata = preprocess(stripe_adata)

print("Preprocessing RNA data...")
gene_adata = preprocess(gene_adata)

common_cells = stripe_adata.obs_names.intersection(gene_adata.obs_names)

if len(common_cells) < 50:
    raise ValueError("Fewer than 50 shared cells remain after filtering.")

stripe_pca = stripe_adata[common_cells].obsm["X_pca"]
gene_pca = gene_adata[common_cells].obsm["X_pca"]

joint_embedding = np.concatenate(
    [
        stripe_pca[:, :STRIPE_DIM],
        gene_pca[:, :GENE_DIM],
    ],
    axis=1,
)

rna_embedding = gene_adata.obsm["X_pca"][:, :TOTAL_DIM]

joint_celltypes = gene_adata.obs.loc[common_cells, "celltype"]
rna_celltypes = gene_adata.obs["celltype"]

print("Running RNA plus Stripe...")
joint_metrics = run_method(
    "joint_RNAplusStripe",
    joint_embedding,
    joint_celltypes,
)

print("Running RNA only...")
rna_metrics = run_method(
    "rna_only",
    rna_embedding,
    rna_celltypes,
)

summary = pd.DataFrame([joint_metrics, rna_metrics])
summary.to_csv(
    os.path.join(OUTPUT_DIR, "summary_metrics.csv"),
    index=False,
)

print(summary)
print("Done.")