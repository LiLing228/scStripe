import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm
from sklearn.metrics import adjusted_rand_score

ARIAL_PATH = "/data_disk/liling/packages/Arial.ttf"
EXPRESSION_FILE = "/data_disk/liling/fig5/RNA_data_code/hires_RNA_data/all_stripe_signal_ratios_all_celltypes_concatenated.tsv"
METADATA_FILE = "/data_disk/liling/fig5/RNA_data_code/hires_RNA_data/GSE223917_HiRES_emb_metadata.tsv"
OUTPUT_DIR = "/data_disk/liling/fig5/umap_stability5"

PARAMS = {
    "tag": "strict_filter",
    "min_genes": 5000,
    "min_cells": 500,
    "n_neighbors": 35,
    "n_pcs": 20,
    "resolution": 1.0,
}

STAGE_ORDER = ["E70", "E75", "E80", "E85", "E95", "EX05", "EX15"]

EMBRYONIC_TYPES = {
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

CELLCYCLE_PALETTE = {
    "G0": "#8E4B9F",
    "G1": "#C8B0D4",
    "Early-S": "#8FD1C4",
    "Mid-S": "#63C0A8",
    "Late-S": "#4BAF7B",
    "G2": "#3F944F",
    "M": "#E7793E",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

font_name = "Arial"
if os.path.exists(ARIAL_PATH):
    fm.fontManager.addfont(ARIAL_PATH)
    font_name = fm.FontProperties(fname=ARIAL_PATH).get_name()

plt.rcParams.update({
    "font.family": font_name,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def display_label(label):
    words = str(label).replace("_", " ").split()
    if words and words[0] not in {"NMP", "ExE"}:
        words[0] = words[0].capitalize()
    return " ".join(words)


def style_axis(ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_palette(values, preferred=None):
    labels = sorted(values.astype(str).unique())

    if preferred is None:
        colors = sns.color_palette("tab20", len(labels))
        return labels, dict(zip(labels, colors))

    order = [label for label in preferred if label in labels]
    extras = [label for label in labels if label not in preferred]
    colors = sns.color_palette("tab20", len(extras))

    palette = {label: preferred[label] for label in order}
    palette.update(dict(zip(extras, colors)))

    return order + extras, palette


def save_legend(order, palette, output_file):
    fig, ax = plt.subplots(figsize=(2.0, max(2.5, 0.28 * len(order))))
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
            label=display_label(label),
        )
        for label in order
    ]

    ax.legend(handles=handles, loc="center", frameon=False)
    fig.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_umap(data, label, output_file, title, palette=None):
    order, colors = get_palette(data[label], palette)

    fig, ax = plt.subplots(figsize=(3, 3))

    sns.scatterplot(
        data=data,
        x="UMAP_1",
        y="UMAP_2",
        hue=label,
        hue_order=order,
        palette=colors,
        s=6,
        linewidth=0,
        legend=False,
        rasterized=True,
        ax=ax,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    style_axis(ax)

    fig.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    save_legend(order, colors, output_file.replace(".pdf", "_legend.pdf"))


def plot_stage_panels(data, output_file):
    order, colors = get_palette(data["celltype"], CELLTYPE_PALETTE)

    fig, axes = plt.subplots(
        1,
        len(STAGE_ORDER),
        figsize=(2.1 * len(STAGE_ORDER), 2.2),
    )

    for ax, stage in zip(axes, STAGE_ORDER):
        subset = data[data["Stage"] == stage]

        if not subset.empty:
            sns.scatterplot(
                data=subset,
                x="UMAP_1",
                y="UMAP_2",
                hue="celltype",
                hue_order=order,
                palette=colors,
                s=6,
                linewidth=0,
                legend=False,
                rasterized=True,
                ax=ax,
            )

        ax.set_title(stage)
        ax.set_xlabel("")
        ax.set_ylabel("")
        style_axis(ax)

    fig.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

    save_legend(order, colors, output_file.replace(".pdf", "_legend.pdf"))


def prepare_data():
    expression = pd.read_csv(EXPRESSION_FILE, sep="\t", index_col=0)
    expression = expression.loc[expression["pvalue"].le(1)]

    matrix = expression.iloc[:, 12:-1].astype(float)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.mask(np.isclose(matrix, 1.0), 0.0)

    metadata = pd.read_csv(METADATA_FILE, sep="\t")
    metadata["Cellname"] = metadata["Cellname"].astype(str)
    metadata = metadata.set_index("Cellname")

    common_cells = matrix.columns.intersection(metadata.index)
    matrix = matrix.loc[:, common_cells]
    metadata = metadata.loc[common_cells].copy()

    metadata["celltype"] = (
        metadata["Cell type"].astype(str).str.replace(" ", "_")
    )

    if "Cellcycle phase" in metadata.columns:
        metadata["Cellcycle_phase"] = (
            metadata["Cellcycle phase"].fillna("Unknown").astype(str)
        )
    else:
        metadata["Cellcycle_phase"] = "Unknown"

    if "Stage" in metadata.columns:
        metadata["Stage"] = metadata["Stage"].fillna("Unknown").astype(str)
    else:
        metadata["Stage"] = "Unknown"

    metadata["maincelltype"] = "other"
    metadata.loc[
        metadata["celltype"].isin(EMBRYONIC_TYPES),
        "maincelltype",
    ] = "embryonic_tissues"

    for label in ["ExE_endoderm", "ExE_ectoderm", "blood"]:
        metadata.loc[
            metadata["celltype"].eq(label),
            "maincelltype",
        ] = label

    keep = metadata["maincelltype"].isin([
        "embryonic_tissues",
        "ExE_endoderm",
        "ExE_ectoderm",
        "blood",
    ])

    return matrix.loc[:, keep], metadata.loc[keep]


def run_umap(matrix, metadata):
    adata = sc.AnnData(matrix.T)
    adata.obs = metadata.loc[adata.obs_names].copy()

    sc.pp.filter_cells(adata, min_genes=PARAMS["min_genes"])
    sc.pp.filter_genes(adata, min_cells=PARAMS["min_cells"])

    if adata.n_obs < 3 or adata.n_vars < 2:
        raise ValueError("Too few cells or features after filtering.")

    if np.nanmin(adata.X) >= 0:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    sc.pp.scale(adata, max_value=10)

    max_pcs = min(100, adata.n_obs - 1, adata.n_vars)
    sc.tl.pca(adata, n_comps=max_pcs, svd_solver="arpack")

    n_pcs = min(PARAMS["n_pcs"], adata.obsm["X_pca"].shape[1])

    sc.pp.neighbors(
        adata,
        n_neighbors=PARAMS["n_neighbors"],
        n_pcs=n_pcs,
    )

    sc.tl.louvain(
        adata,
        resolution=PARAMS["resolution"],
        random_state=0,
        key_added="cluster",
    )

    sc.tl.umap(
        adata,
        min_dist=0.01,
        spread=5,
        random_state=0,
    )

    result = adata.obs.copy()
    result[["UMAP_1", "UMAP_2"]] = adata.obsm["X_umap"]
    result["cluster"] = result["cluster"].astype(str)

    return result, adata.n_obs, adata.n_vars, n_pcs


matrix, metadata = prepare_data()
result, n_cells, n_features, n_pcs = run_umap(matrix, metadata)

ari_main = adjusted_rand_score(result["maincelltype"], result["cluster"])
ari_celltype = adjusted_rand_score(result["celltype"], result["cluster"])

base = (
    f'{PARAMS["tag"]}_minf{PARAMS["min_genes"]}'
    f'_minc{PARAMS["min_cells"]}'
    f'_n{PARAMS["n_neighbors"]}'
    f'_pca{n_pcs}'
    f'_res{PARAMS["resolution"]}'
)

plot_umap(
    result,
    "maincelltype",
    f"{OUTPUT_DIR}/umap_maincelltype_{base}.pdf",
    "Stripe score - 4 main groups",
)

plot_umap(
    result,
    "Cellcycle_phase",
    f"{OUTPUT_DIR}/umap_cellcycle_phase_{base}.pdf",
    "Stripe score - cell cycle phase",
    CELLCYCLE_PALETTE,
)

plot_stage_panels(
    result,
    f"{OUTPUT_DIR}/umap_celltype_by_stage_{base}.pdf",
)

plot_umap(
    result,
    "cluster",
    f"{OUTPUT_DIR}/umap_cluster_{base}.pdf",
    f"UMAP clusters (ARI main={ari_main:.3f}, celltype={ari_celltype:.3f})",
)

summary = pd.DataFrame([{
    **PARAMS,
    "n_pcs_used": n_pcs,
    "n_cells_after_filter": n_cells,
    "n_features_after_filter": n_features,
    "ARI_maincelltype": ari_main,
    "ARI_celltype": ari_celltype,
}])

summary.to_csv(
    f"{OUTPUT_DIR}/ARI_results.tsv",
    sep="\t",
    index=False,
)

print("Done.")