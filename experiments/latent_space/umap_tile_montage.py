# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
umap_tile_montage.py — enlarged UMAP + representative tile montage (EXP-2 / Comment 6).

For each phenotype cluster (C2,C3,C5,C6) this reproduces the SAME preprocessing,
soft-label logistic regression and UMAP embedding (random_state=42) as
cell_latent_analysis.py, then:

  - draws an ENLARGED UMAP coloured by soft label P(MYCN-amp);
  - selects representative tiles two ways (Comment 6 = "both"):
       * REGION samples  : KMeans(k=6) on the 2-D embedding, tile nearest each
                           region centroid (what each morphological sub-population
                           looks like);
       * EXTREME samples : tiles at the lowest / ~0.5 / highest soft label;
  - marks the selected tiles on the UMAP (numbered) and shows the matching
    H&E (or cell-overlay) thumbnails in a labelled grid below.

Tile-image lookup (index 'class_X|S_x_A_y_B'):
  C3 -> prototype_2/patches/class_X/S_x_A_y_B.png
  C2,C5,C6 -> prototype_N/patches_cell_property/class_X/overlay/S_x_A_y_B.png

Outputs (results/):
  component{2,3,5,6}_umap_tilemontage.pdf
  component{2,3,5,6}_tile_selection.csv
  component{2,3,5,6}_umap_softlabel_large.pdf   (enlarged scatter only)
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import umap

warnings.filterwarnings("ignore")

BASE = os.path.join(os.environ.get("PHENO_MYCN_ROOT", "/path/to/pheno_mycn"), "olga_refactered")
IMG  = os.path.join(BASE, "results/prototype_analysis")
OUT  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

# comp_name -> (cell_info csv prototype, image prototype dir, image-kind)
COMPONENTS = {
    "component2": dict(proto=1, kind="overlay"),
    "component3": dict(proto=2, kind="plain"),
    "component5": dict(proto=4, kind="overlay"),
    "component6": dict(proto=5, kind="overlay"),
}
NAN_THRESH = 0.50
N_REGIONS  = 6


# NOTE on tile images (verified 2026-06-15):
#   The Section-2.4 latent-space analysis uses the TIER-2 table under
#   slide_inference/.../prototype_N/features/ (1465/1574 tiles) — but those tiles
#   have NO saved patch images and no source WSIs are available to crop them.
#   The TIER-1 pathology-review table under prototype_analysis/.../features/
#   (229/382/131/217 tiles) fully matches its overlay/H&E patch images, so the
#   image-backed montage is built on the TIER-1 set. This is a DIFFERENT (smaller,
#   highest-confidence) tile set than the main Section-2.4 UMAP — see EXP-2 note
#   in manuscript_correction.md; final-figure choice deferred to user.
def csv_path(proto):
    return os.path.join(BASE, f"results/prototype_analysis/prototype_{proto}/features/cell_info_updated.csv")


def tile_image_path(comp, idx):
    """Resolve 'class_X|S_x_A_y_B' to its patch png path; None if missing."""
    cfg = COMPONENTS[comp]
    cls, rest = idx.split("|", 1)
    if cfg["kind"] == "plain":
        p = os.path.join(IMG, f"prototype_{cfg['proto']}", "patches", cls, rest + ".png")
    else:
        p = os.path.join(IMG, f"prototype_{cfg['proto']}", "patches_cell_property", cls, "overlay", rest + ".png")
    return p if os.path.exists(p) else None


def parse_meta(index_series):
    split     = index_series.str.split("|", n=1, expand=True)
    class_str = split[0]
    slide_idx = split[1].str.split("_x_", n=1).str[0]
    return class_str, class_str + "_" + slide_idx


for comp, cfg in COMPONENTS.items():
    print(f"\n{'='*60}\n{comp} (prototype_{cfg['proto']})\n{'='*60}")
    df = pd.read_csv(csv_path(cfg["proto"]), index_col=0)
    class_str, slide_id = parse_meta(pd.Series(df.index))
    y = (class_str.values == "class_1").astype(int)

    meta = []
    feat_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    amp_m, non_m = y == 1, y == 0
    feat_cols = [c for c in feat_cols
                 if df.loc[amp_m, c].isna().mean() <= NAN_THRESH
                 and df.loc[non_m, c].isna().mean() <= NAN_THRESH
                 and df[c].std() > 1e-9]

    X = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(df[feat_cols].values.astype(float)))

    lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, solver="lbfgs").fit(X, y)
    soft = lr.predict_proba(X)[:, 1]
    emb = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    random_state=42, verbose=False).fit_transform(X)

    # ── selection ───────────────────────────────────────────────────────────────
    km = KMeans(n_clusters=N_REGIONS, random_state=42, n_init=10).fit(emb)
    region_pts = []
    for k in range(N_REGIONS):
        members = np.where(km.labels_ == k)[0]
        d = np.linalg.norm(emb[members] - km.cluster_centers_[k], axis=1)
        region_pts.append(members[d.argmin()])
    extreme_pts = [int(np.argmin(soft)),
                   int(np.argmin(np.abs(soft - 0.5))),
                   int(np.argmax(soft))]
    sel = [("R%d" % (i + 1), p, "region") for i, p in enumerate(region_pts)] + \
          [("E-low", extreme_pts[0], "extreme"),
           ("E-mid", extreme_pts[1], "extreme"),
           ("E-high", extreme_pts[2], "extreme")]

    sel_rows = []
    for lbl, p, kind in sel:
        idx = df.index[p]
        sel_rows.append({"label": lbl, "kind": kind, "tile_index": idx,
                         "class": class_str.values[p], "soft_label": float(soft[p]),
                         "umap1": float(emb[p, 0]), "umap2": float(emb[p, 1]),
                         "image": tile_image_path(comp, idx)})
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(os.path.join(OUT, f"{comp}_tile_selection_pathreview.csv"), index=False)
    n_found = sel_df.image.notna().sum()
    print(f"  selected {len(sel_df)} tiles; {n_found} images found")

    # ── enlarged scatter only ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=soft, cmap="RdBu_r", vmin=0, vmax=1,
                    s=14, alpha=0.6, linewidths=0, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0 (non-amp)", "0.5", "1 (MYCN-amp)"])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    fig.savefig(os.path.join(OUT, f"{comp}_umap_softlabel_large_pathreview.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)

    # ── composite: scatter (numbered) + thumbnail grid ───────────────────────────
    ncols = len(sel_df)
    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, ncols, height_ratios=[3, 1.3], hspace=0.25, wspace=0.15)

    axs = fig.add_subplot(gs[0, :])
    sc = axs.scatter(emb[:, 0], emb[:, 1], c=soft, cmap="RdBu_r", vmin=0, vmax=1,
                     s=14, alpha=0.55, linewidths=0, rasterized=True)
    cb = plt.colorbar(sc, ax=axs, fraction=0.03, pad=0.01)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0", "0.5", "1"])
    cb.set_label("P(MYCN-amp)", fontsize=9)
    for _, r in sel_df.iterrows():
        mk = "o" if r["kind"] == "region" else "*"
        axs.scatter([r.umap1], [r.umap2], marker=mk, s=160, facecolors="none",
                    edgecolors="black", linewidths=1.6, zorder=5)
        axs.annotate(r["label"], (r.umap1, r.umap2), fontsize=8, fontweight="bold",
                     xytext=(4, 4), textcoords="offset points", zorder=6)
    axs.set_xlabel("UMAP 1"); axs.set_ylabel("UMAP 2")
    axs.set_title(f"{comp}: soft-label UMAP with representative tiles "
                  f"(○ sub-population centroids, ★ soft-label extremes)", fontsize=10)

    for j, (_, r) in enumerate(sel_df.iterrows()):
        axt = fig.add_subplot(gs[1, j])
        if r["image"] and os.path.exists(r["image"]):
            axt.imshow(mpimg.imread(r["image"]))
        else:
            axt.text(0.5, 0.5, "image\nN/A", ha="center", va="center", fontsize=7)
        axt.set_xticks([]); axt.set_yticks([])
        cls_short = "amp" if r["class"] == "class_1" else "non"
        axt.set_title(f"{r['label']}\n{cls_short}, p={r.soft_label:.2f}", fontsize=7)

    fig.savefig(os.path.join(OUT, f"{comp}_umap_tilemontage_pathreview.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {comp}_umap_tilemontage_pathreview.pdf  ({n_found}/{len(sel_df)} images)")

print("\nDone.")
