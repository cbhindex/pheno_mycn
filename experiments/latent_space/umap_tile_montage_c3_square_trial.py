# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
umap_tile_montage_c3_square_trial.py — TRIAL layout for the main Fig 4k hero panel.

Draws the C3 soft-label UMAP as a SQUARE central panel (height = width), with the
representative Tier-1 tiles placed as thumbnails down the LEFT and RIGHT margins,
each linked to its position in the embedding by a connector line.

Same C3 computation as umap_tile_montage.py (Tier-1 pathology-review set,
prototype_2/patches images, soft-label logistic regression, UMAP random_state=42,
KMeans k=6 region centroids + 3 soft-label extremes), so the embedding and tile
selection match component3_umap_tilemontage_pathreview.pdf — only the LAYOUT
differs (square + side callouts instead of scatter-over-thumbnail-row).

NB (same caveat as umap_tile_montage.py): this uses the TIER-1 pathology-review
tile set (has images), a different/smaller set than the Tier-2 §2.4 latent space.

Output: results/component3_umap_tilemontage_square_trial.pdf
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
from matplotlib.patches import ConnectionPatch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import umap

warnings.filterwarnings("ignore")

BASE = os.path.join(os.environ.get("PHENO_MYCN_ROOT", "/path/to/pheno_mycn"), "olga_refactered")
IMG  = os.path.join(BASE, "results/prototype_analysis")
OUT  = os.path.join(os.path.dirname(__file__), "results")
PROTO      = 2          # C3 = prototype_2
NAN_THRESH = 0.50
N_REGIONS  = 6


def tile_image_path(idx):
    cls, rest = idx.split("|", 1)
    p = os.path.join(IMG, f"prototype_{PROTO}", "patches", cls, rest + ".png")
    return p if os.path.exists(p) else None


def parse_meta(s):
    sp = s.str.split("|", n=1, expand=True)
    return sp[0], sp[0] + "_" + sp[1].str.split("_x_", n=1).str[0]


# ── recompute C3 soft-label UMAP (identical to umap_tile_montage.py) ──────────
df = pd.read_csv(os.path.join(BASE, f"results/prototype_analysis/prototype_{PROTO}/features/cell_info_updated.csv"),
                 index_col=0)
class_str, _ = parse_meta(pd.Series(df.index))
y = (class_str.values == "class_1").astype(int)
feat = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
am, nm = y == 1, y == 0
feat = [c for c in feat
        if df.loc[am, c].isna().mean() <= NAN_THRESH
        and df.loc[nm, c].isna().mean() <= NAN_THRESH
        and df[c].std() > 1e-9]
X = StandardScaler().fit_transform(
    SimpleImputer(strategy="median").fit_transform(df[feat].values.astype(float)))
lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, solver="lbfgs").fit(X, y)
soft = lr.predict_proba(X)[:, 1]
emb = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                random_state=42, verbose=False).fit_transform(X)

# ── tile selection: 6 region centroids + 3 soft-label extremes ────────────────
km = KMeans(n_clusters=N_REGIONS, random_state=42, n_init=10).fit(emb)
region = []
for k in range(N_REGIONS):
    mem = np.where(km.labels_ == k)[0]
    d = np.linalg.norm(emb[mem] - km.cluster_centers_[k], axis=1)
    region.append(mem[d.argmin()])
extreme = [int(np.argmin(soft)), int(np.argmin(np.abs(soft - 0.5))), int(np.argmax(soft))]
sel = [("R%d" % (i + 1), p) for i, p in enumerate(region)] + \
      [("E-low", extreme[0]), ("E-mid", extreme[1]), ("E-high", extreme[2])]
rows = []
for lbl, p in sel:
    idx = df.index[p]
    rows.append(dict(label=lbl, tile=idx, cls=class_str.values[p], soft=float(soft[p]),
                     u1=float(emb[p, 0]), u2=float(emb[p, 1]), img=tile_image_path(idx)))
S = pd.DataFrame(rows)
S = S[S.img.notna()].reset_index(drop=True)
print(f"{len(S)} selected tiles with images")

# split L/R by umap-x (balanced), order each column top→bottom by umap-y
order = S.sort_values("u1").index.tolist()
nL = (len(order) + 1) // 2
left_ids = set(order[:nL])
S["side"] = ["L" if i in left_ids else "R" for i in S.index]

# ── figure: square UMAP in the middle, tile callouts left & right ─────────────
figW, figH = 14.0, 9.0
fig = plt.figure(figsize=(figW, figH))

side_in = 7.0                               # square side (inches) -> height==width
wf, hf  = side_in / figW, side_in / figH
ux0, uy0 = (1 - wf) / 2, (1 - hf) / 2
umap_ax = fig.add_axes([ux0, uy0, wf, hf])
sc = umap_ax.scatter(emb[:, 0], emb[:, 1], c=soft, cmap="RdBu_r", vmin=0, vmax=1,
                     s=10, alpha=0.55, linewidths=0, rasterized=True)
umap_ax.set_xticks([]); umap_ax.set_yticks([])
umap_ax.set_xlabel("UMAP 1", fontsize=8); umap_ax.set_ylabel("UMAP 2", fontsize=8)

# colourbar across the top of the square
cax = fig.add_axes([ux0, uy0 + hf + 0.015, wf, 0.018])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0 (non-amp)", "0.5", "1 (MYCN-amp)"], fontsize=8)
cb.ax.xaxis.set_ticks_position("top"); cb.ax.xaxis.set_label_position("top")
cb.set_label("P(MYCN-amp) soft label", fontsize=8)

th_in = 1.45                                # thumbnail side (inches, square)
twf, thf = th_in / figW, th_in / figH


def place_column(side, x_left):
    sub = S[S.side == side].sort_values("u2", ascending=False).reset_index(drop=True)
    n = len(sub)
    top, bot = 0.92 - thf, 0.06
    ys = np.linspace(top, bot, n) if n > 1 else [(top + bot) / 2]
    for k, (_, r) in enumerate(sub.iterrows()):
        ax = fig.add_axes([x_left, ys[k], twf, thf])
        ax.imshow(mpimg.imread(r["img"]))
        ax.set_xticks([]); ax.set_yticks([])
        col = plt.cm.RdBu_r(r["soft"])                 # border ties to soft label
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(3.0)
        clss = "amp" if r["cls"] == "class_1" else "non"
        ax.set_title(f"{r['label']} · {clss} · p={r['soft']:.2f}", fontsize=6.5)
        # marker + label on the UMAP
        umap_ax.scatter([r.u1], [r.u2], s=110, facecolors="none",
                        edgecolors="black", linewidths=1.4, zorder=6)
        umap_ax.annotate(r["label"], (r.u1, r.u2), fontsize=7, fontweight="bold",
                         xytext=(3, 3), textcoords="offset points", zorder=7)
        # connector from thumbnail inner edge to the UMAP point
        anchor = (1.0, 0.5) if side == "L" else (0.0, 0.5)
        con = ConnectionPatch(xyA=anchor, coordsA=ax.transAxes,
                              xyB=(r.u1, r.u2), coordsB=umap_ax.transData,
                              color="0.45", lw=0.8, alpha=0.85, zorder=2)
        con.set_clip_on(False)
        fig.add_artist(con)


place_column("L", 0.015)
place_column("R", 1 - 0.015 - twf)

fig.savefig(os.path.join(OUT, "component3_umap_tilemontage_square_trial.pdf"), format="pdf")
print("wrote results/component3_umap_tilemontage_square_trial.pdf")
