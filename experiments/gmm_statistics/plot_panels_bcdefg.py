# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
Figure 2 — Panels b, c, d, e, f, g.

Per-tile GMM phenotype-score (responsibility) distributions (K=6 clusters),
aggregated by split:
  Panels b (density) + c (violin) — train
  Panels d (density) + e (violin) — val
  Panels f (density) + g (violin) — test

Convention: label 0 = non-amp, label 1 = MYCN-amp.
Colours match the later figures (Figs 4/5/6): non-amplified = blue #4F74C8,
MYCN-amplified = red #D94F3D. All six panels use the same blue–red scheme and
the same fill transparency (alpha 0.55) for the density and violin panels.
Legends: 'MYCN+' (amp) and 'MYCN−' (non-amp), rendered in Arial.

Layout: b/d/f are short-height long-width (they sit beside c/e/g in one row in
Illustrator). All six panels share the same height so they line up. No titles,
no axis labels (added manually in Illustrator). Tick marks/values retained.

Inputs (resolved under $PHENO_MYCN_ROOT; not shipped — patient-derived):
    olga_refactered/data/cv_splits/neuroblastoma/fold9.csv          — CV splits
    olga_refactered/results/slide_inference/fold_9/pt_outputs/*.pt  — per-tile GMM

Outputs (results/, git-ignored):
    panel_{b,d,f}_density_{train,val,test}.pdf  — KDE/histogram, per cluster
    panel_{c,e,g}_violin_{train,val,test}.pdf   — split violins, per cluster

Run:
    export PHENO_MYCN_ROOT=/path/to/pheno_mycn
    python plot_panels_bcdefg.py
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.join(os.environ.get("PHENO_MYCN_ROOT", "/path/to/pheno_mycn"), "olga_refactered")
FOLD_CSV = os.path.join(BASE, "data/cv_splits/neuroblastoma/fold9.csv")
PT_DIR   = os.path.join(BASE, "results/slide_inference/fold_9/pt_outputs")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

N_COMPONENTS = 6

# Colour convention — matched to later figures (Figs 4/5/6):
#   non-amplified (label 0) = blue #4F74C8 ; MYCN-amplified (label 1) = red #D94F3D
# (see e.g. latent_space/allcluster_softlabel_summary.py: C_NON, C_AMP = "#4F74C8", "#D94F3D";
#  cell_level_stats/mixed_effects_stats.py; shap_analysis/run_shap.py.)
# All six panels (b–g) use the same blue–red scheme and the same fill transparency.
FILL_ALPHA = 0.55

# Density-panel colours
COLOR_NEG_DENS = "#4F74C8"   # blue — MYCN−  (non-amp, label 0)
COLOR_POS_DENS = "#D94F3D"   # red  — MYCN+ (amp,     label 1)

# Violin colours
COLOR_POS_VIO = "#D94F3D"    # red  — MYCN+ (amp,     label 1)
COLOR_NEG_VIO = "#4F74C8"    # blue — MYCN− (non-amp, label 0)

LABEL_POS = "MYCN+"
LABEL_NEG = "MYCN−"   # unicode minus

# Figure sizes — both panel types share height 3.5 for side-by-side row layout.
# Width:height ratios are 100:35 for densities and 80:35 for violins.
DENS_FIGSIZE = (10.0, 3.5)   # width:height = 100:35
VIOLIN_FIGSIZE = (8.0, 3.5)  # width:height = 80:35

# Arial font for legend
ARIAL = FontProperties(family=["Arial", "Liberation Sans", "DejaVu Sans"], size=10)


def load_split_labels():
    """fold9.csv → dict slide_name → (split, label)."""
    df = pd.read_csv(FOLD_CSV, index_col=0)
    out = {}
    for split in ("train", "val", "test"):
        names = df[split].dropna()
        labels = df[f"{split}_label"].dropna()
        for s, lbl in zip(names, labels):
            out[s.strip()] = (split, int(float(lbl)))
    return out


def load_responsibilities(slide_meta):
    """Return DataFrame: rows = tiles, columns = c1..c6 + split + label."""
    rows = []
    t0 = time.time()
    n_loaded = 0
    n_missing = 0
    for slide_name, (split, label) in slide_meta.items():
        pt_path = os.path.join(PT_DIR, f"{slide_name}_gmm.pt")
        if not os.path.isfile(pt_path):
            n_missing += 1
            continue
        t = torch.load(pt_path, map_location="cpu", weights_only=False)
        if t.ndim == 3:
            t = t.squeeze(0)
        arr = t.cpu().numpy()
        n_loaded += 1
        for i in range(arr.shape[0]):
            rows.append((split, label, *arr[i].tolist()))
        if n_loaded % 50 == 0:
            print(f"  loaded {n_loaded}/{len(slide_meta)} slides "
                  f"({time.time()-t0:.1f}s elapsed)")
    cols = ["split", "label"] + [f"c{i+1}" for i in range(N_COMPONENTS)]
    df = pd.DataFrame(rows, columns=cols)
    print(f"Loaded {n_loaded} slides ({n_missing} missing). Total tiles: {len(df):,}")
    return df


def plot_density_panel(df_split, out_path):
    """One PDF with N_COMPONENTS sub-panels. Legend per sub-panel in upper-left."""
    fig, axes = plt.subplots(1, N_COMPONENTS, figsize=DENS_FIGSIZE, sharey=False)

    neg = df_split[df_split["label"] == 0]
    pos = df_split[df_split["label"] == 1]

    for i in range(N_COMPONENTS):
        ax = axes[i]
        col = f"c{i+1}"
        eps = 1e-12
        neg_vals = np.clip(neg[col].values, eps, 1.0)
        pos_vals = np.clip(pos[col].values, eps, 1.0)

        sns.histplot(neg_vals, kde=True, bins=30, color=COLOR_NEG_DENS,
                     label=LABEL_NEG, ax=ax, log_scale=True,
                     stat="density", alpha=FILL_ALPHA, edgecolor="none")
        sns.histplot(pos_vals, kde=True, bins=30, color=COLOR_POS_DENS,
                     label=LABEL_POS, ax=ax, log_scale=True,
                     stat="density", alpha=FILL_ALPHA, edgecolor="none")
        # No titles, no axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Legend in upper-left (empty space), Arial
        ax.legend(loc="upper left", prop=ARIAL, frameon=False,
                  handlelength=1.2, handletextpad=0.5, borderaxespad=0.3)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_violin_panel(df_split, out_path):
    """One PDF, single split-violin with N_COMPONENTS on x-axis. Legend upper-right.

    Violin bodies are drawn at the same fill transparency (FILL_ALPHA) as the
    density panels so the two panel types match visually.
    """
    long_rows = []
    for i in range(N_COMPONENTS):
        col = f"c{i+1}"
        for v in df_split.loc[df_split["label"] == 1, col].values:
            long_rows.append((f"C{i+1}", LABEL_POS, v))
        for v in df_split.loc[df_split["label"] == 0, col].values:
            long_rows.append((f"C{i+1}", LABEL_NEG, v))
    long_df = pd.DataFrame(long_rows, columns=["Component", "Class", "Responsibility"])

    fig, ax = plt.subplots(figsize=VIOLIN_FIGSIZE)
    sns.violinplot(
        x="Component", y="Responsibility", hue="Class",
        data=long_df,
        split=True,
        inner="quartile",
        density_norm="width",
        hue_order=[LABEL_POS, LABEL_NEG],
        palette={LABEL_POS: COLOR_POS_VIO, LABEL_NEG: COLOR_NEG_VIO},
        linewidth=0.8,
        ax=ax,
    )
    # Match the density panels' fill transparency: the violin bodies are the
    # PolyCollections on the axes (the inner quartile lines are Line2D and keep
    # full opacity, mirroring the density KDE lines).
    for coll in ax.collections:
        coll.set_alpha(FILL_ALPHA)
    # No titles, no axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend in upper-right with frame; Arial
    leg = ax.legend(loc="upper right", prop=ARIAL, title=None, frameon=True,
                    edgecolor="black", fancybox=False, handlelength=1.2,
                    handletextpad=0.5, borderaxespad=0.5)
    # Make the legend swatches semi-transparent too, to match the bodies.
    for handle in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
        handle.set_alpha(FILL_ALPHA)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    print("Loading split labels...")
    meta = load_split_labels()
    print(f"  {len(meta)} slides "
          f"({sum(1 for _,(s,_) in meta.items() if s=='train')} train, "
          f"{sum(1 for _,(s,_) in meta.items() if s=='val')} val, "
          f"{sum(1 for _,(s,_) in meta.items() if s=='test')} test)")

    print("Loading per-tile GMM responsibilities...")
    df = load_responsibilities(meta)
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print("Tile counts by split × label:")
    print(counts)

    plans = [
        ("train", "panel_b_density_train.pdf", "panel_c_violin_train.pdf"),
        ("val",   "panel_d_density_val.pdf",   "panel_e_violin_val.pdf"),
        ("test",  "panel_f_density_test.pdf",  "panel_g_violin_test.pdf"),
    ]
    for split, dens_name, vio_name in plans:
        sub = df[df["split"] == split]
        print(f"\n[{split}] tiles={len(sub):,}")
        plot_density_panel(sub, os.path.join(OUT_DIR, dens_name))
        plot_violin_panel(sub, os.path.join(OUT_DIR, vio_name))


if __name__ == "__main__":
    main()
