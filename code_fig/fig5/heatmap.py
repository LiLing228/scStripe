import os, sys, re, gzip
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ARIAL_PATH = "/data_disk/liling/packages/Arial.ttf"
if os.path.exists(ARIAL_PATH):
    fm.fontManager.addfont(ARIAL_PATH)
    ARIAL_NAME = fm.FontProperties(fname=ARIAL_PATH).get_name()
else:
    ARIAL_NAME = "Arial"

plt.rcParams.update({
    "font.family": ARIAL_NAME,
    "font.size": 8,               
    "axes.unicode_minus": False     
})

# ========= USER CONFIG =========
out = "/data_disk/liling/fig6/both_EBF1_submat_5kb_wid6"

GROUPS_TSV   = f"{out}/hierarchical_0_20_40_50_60_70_80/cell_to_group.tsv"
PERCELL_DIR  = "/data_disk/data/liling/scMicroC_data/EBF1_submat_5kb_percell"
OUT_DIR      = f"{out}/hierarchical_0_20_40_50_60_70_80"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG      = os.path.join(OUT_DIR, "groups_avg_heatmaps.png")
OUT_PDF      = os.path.join(OUT_DIR, "groups_avg_heatmaps.pdf")

EXPECTED_SHAPE: Tuple[int,int] = (120, 120)
BOXES: List[Tuple[int,int,int,int]] = [(96, 102, 16, 102)]   
CMAP            = "rainbow"        
USER_VMAX: Optional[float] = None 
VMAX_PERCENTILE = 95              
PANEL_SIZE = (2.2, 2.6)           
DPI = 300

def find_percell_file(percell_dir: str, cellname: str) -> Optional[str]:
    path = os.path.join(percell_dir, f"{cellname}.tsv")
    return path if os.path.isfile(path) else None


def load_matrix(path: str) -> Optional[np.ndarray]:
    try:
        return np.loadtxt(path, delimiter="\t", dtype=float)
    except Exception as e:
        print(f"error {path}: {e}", file=sys.stderr)
        return None

def group_mean_matrix(cells: List[str]) -> Tuple[Optional[np.ndarray], int]:
    acc, n = None, 0
    for c in cells:
        p = find_percell_file(PERCELL_DIR, c)
        if not p: 
            continue
        m = load_matrix(p)
        if m is None or m.shape != EXPECTED_SHAPE:
            continue
        acc = m if acc is None else (acc + m)
        n += 1
    if acc is None or n == 0:
        return None, 0
    acc = acc + acc.T
    return acc / float(n), n

# ========= plotting =========
def plot_heatmaps_right_cbar_narrow(
    mats, labels, out_png, out_pdf,
    counts=None, boxes=None,
    cmap="rainbow", user_vmax=None, vmax_percentile=95,
    panel_size=(2.2, 2.2),  # (width, height)
    dpi=300,
    cbar_width_ratio=0.045,
    wspace=0.05,
    cbar_label="average contact value",
    cbar_tick_size=8, title_size=9, ann_size=8
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors, patches as mpatches

    n = len(mats)
    if n == 0: 
        return

    valid = [m for m in mats if m is not None]
    if not valid: 
        return

    all_vals = np.concatenate([m.ravel() for m in valid])
    all_vals = all_vals[~np.isnan(all_vals)]
    pos = all_vals[all_vals > 0]
    if user_vmax is not None:
        use_vmax = max(float(user_vmax),
                       float(np.percentile(pos, vmax_percentile)) if pos.size else float(user_vmax))
    else:
        use_vmax = float(np.percentile(pos, vmax_percentile)) if pos.size else float(np.nanmax(all_vals))
    if not np.isfinite(use_vmax) or use_vmax <= 0: 
        use_vmax = 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=use_vmax)

    if counts is None: 
        counts = [None]*n

    fig_w = panel_size[0]*n + 0.6
    fig_h = panel_size[1]
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    width_ratios = [1.0]*n + [cbar_width_ratio]
    gs = fig.add_gridspec(nrows=1, ncols=n+1, width_ratios=width_ratios, wspace=wspace)

    axs = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cax = fig.add_subplot(gs[0, -1])

    last_im = None
    for ax, lbl, mat, cnt in zip(axs, labels, mats, counts):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): 
            sp.set_visible(False)
        title_txt = f"Group {lbl}" + (f" (n={int(cnt)})" if cnt is not None else "")
        ax.set_title(title_txt, fontsize=title_size, pad=6)

        if mat is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=ann_size, transform=ax.transAxes)
        else:
            disp = np.nan_to_num(np.asarray(mat, float), nan=0.0)
            last_im = ax.imshow(disp, origin="upper", cmap=cmap, norm=norm,
                                interpolation="nearest", aspect="equal")
            if boxes:
                for (r0, r1, c0, c1) in boxes:
                    rr0 = max(0, int(r0)); rr1 = min(disp.shape[0]-1, int(r1))
                    cc0 = max(0, int(c0)); cc1 = min(disp.shape[1]-1, int(c1))
                    if rr0 <= rr1 and cc0 <= cc1:
                        rect = mpatches.Rectangle((cc0-0.5, rr0-0.5),
                                                  cc1-cc0+1, rr1-rr0+1,
                                                  linewidth=1.0, edgecolor='black', facecolor='none')
                        ax.add_patch(rect)

    if last_im is not None:
        cb = fig.colorbar(last_im, cax=cax, orientation="vertical")
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=cbar_tick_size)

    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ========= main =========
def main():
    if not os.path.exists(GROUPS_TSV):
        print(f"ERROR: groups file not found: {GROUPS_TSV}", file=sys.stderr); sys.exit(1)

    df = pd.read_csv(GROUPS_TSV, sep="\t")

    df["group"] = pd.to_numeric(df["group"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["cell","group"]).copy()
    df["group"] = df["group"].astype(int)

    groups = sorted(df["group"].unique().tolist())

    mats, counts, labels = [], [], []
    for g in groups:
        cells = df.loc[df["group"] == g, "cell"].tolist()
        print(f"Group {g}: {len(cells)} cells")
        mean_mat, n = group_mean_matrix(cells)
        mats.append(mean_mat)
        counts.append(n)
        labels.append(str(g))
        if mean_mat is not None:
            np.savez_compressed(os.path.join(OUT_DIR, f"group_{g}_avg_submat.npz"), matrix=mean_mat)
            np.savetxt(os.path.join(OUT_DIR, f"group_{g}_avg_submat.tsv"),
                       mean_mat, fmt="%.6g", delimiter="\t")

    plot_heatmaps_right_cbar_narrow(
        mats=mats,
        labels=labels,
        counts=counts,
        out_png=OUT_PNG,
        out_pdf=OUT_PDF,
        boxes=BOXES,
        cmap=CMAP,
        user_vmax=USER_VMAX,
        vmax_percentile=VMAX_PERCENTILE,
        panel_size=PANEL_SIZE,
        dpi=DPI,
        cbar_width_ratio=0.045 
    )

if __name__ == "__main__":
    main()