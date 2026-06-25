# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
umap_tile_montage_tier2.py — enlarged soft-label UMAP + real H&E tile callouts,
on the REAL Tier-2 embeddings, for all four MYCN-associated clusters.

  • C3 → main Fig 4 panel k (hero).
  • C2, C5, C6 → one supplementary UMAP figure each.

For each cluster this reproduces the EXACT same soft-label UMAP as
cell_latent_analysis.py / component{N}_umap_by_softlabel.pdf (Tier-2 set:
slide_inference/fold_9/images/prototype_N/features/cell_info_updated.csv;
soft-label logistic regression; UMAP random_state=42), enlarged, with real
Tier-2 plain-H&E tiles called out on the left and right margins + connector lines.

Tile selection per cluster: LEFT margin = lowest 30% UMAP-1 dots, RIGHT margin =
top 30% UMAP-1 dots; within each, KMeans(k=N_PER_SIDE) sub-regions and the most
informative tile near each centroid. Informativeness always penalises
out-of-focus tiles (image gradient energy); for the TUMOUR clusters (C3, C5) it
also favours nucleus-rich tiles (dark-pixel fraction), while for the ANUCLEAR
clusters (C2 necrotic, C6 haemorrhagic) the nuclei term is DROPPED — their
representative morphology is intrinsically necrotic / red-blood-cell, not
nucleated, so requiring nuclei would misrepresent them.

Tier-2 tiles DO have saved images at
slide_inference/fold_9/images/prototype_N/patches/<class>/<idx>.png.

Outputs (results/): component{2,3,5,6}_umap_tilemontage_tier2.pdf
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
OUT  = os.path.join(os.path.dirname(__file__), "results")
NAN_THRESH = 0.50
N_PER_SIDE = 10                 # tiles per margin
LEFT_Q, RIGHT_Q = 0.30, 0.70    # left margin = lowest 30% UMAP-1, right = top 30%

# cluster -> prototype dir + whether to weight nucleus content in tile selection
COMPONENTS = {
    "component2": dict(proto=1, use_nuclei=False),   # C2 necrotic     (anuclear)
    "component3": dict(proto=2, use_nuclei=True),    # C3 cellular tumour
    "component5": dict(proto=4, use_nuclei=True),    # C5 dense tumour
    "component6": dict(proto=5, use_nuclei=False),   # C6 haemorrhagic (anuclear)
}


def proto_dir(proto):
    return os.path.join(BASE, f"results/slide_inference/fold_9/images/prototype_{proto}")


def tile_image_path(proto, idx):
    cls, rest = idx.split("|", 1)
    p = os.path.join(proto_dir(proto), "patches", cls, rest + ".png")
    return p if os.path.exists(p) else None


def informativeness(path, use_nuclei):
    """High for in-focus tiles; for tumour clusters also for nucleus-rich tiles."""
    g = mpimg.imread(path)[..., :3].mean(axis=2).astype(float)
    if g.max() > 1.0:
        g /= 255.0
    gy, gx = np.gradient(g)
    sharp = float((gx ** 2 + gy ** 2).mean())          # focus / texture energy
    if use_nuclei:
        dark = float((g < 0.45).mean())                # haematoxylin nuclei present
        return sharp * (dark + 0.05)
    return sharp


def embed_cluster(proto):
    """Reproduce the Tier-2 soft-label UMAP (== component{N}_umap_by_softlabel)."""
    df = pd.read_csv(os.path.join(proto_dir(proto), "features/cell_info_updated.csv"), index_col=0)
    class_str = pd.Series(df.index).str.split("|", n=1, expand=True)[0].values
    y = (class_str == "class_1").astype(int)
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
    return df, class_str, soft, emb


def pick(side, emb, df, proto, use_nuclei):
    """N_PER_SIDE representative, in-focus tiles spread across one side's cluster."""
    u1 = emb[:, 0]
    lo, hi = np.quantile(u1, LEFT_Q), np.quantile(u1, RIGHT_Q)
    cand = np.where(u1 <= lo)[0] if side == "L" else np.where(u1 >= hi)[0]
    cand = np.array([p for p in cand if tile_image_path(proto, df.index[p]) is not None])
    k = min(N_PER_SIDE, len(cand))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(emb[cand])
    chosen = []
    for c in range(k):
        members = cand[km.labels_ == c]
        d = np.linalg.norm(emb[members] - km.cluster_centers_[c], axis=1)
        nearest = members[np.argsort(d)[:20]]
        best = max(nearest, key=lambda p: informativeness(tile_image_path(proto, df.index[p]), use_nuclei))
        chosen.append(int(best))
    return chosen


def render(comp, cfg):
    proto, use_nuclei = cfg["proto"], cfg["use_nuclei"]
    df, class_str, soft, emb = embed_cluster(proto)
    u1, u2 = emb[:, 0], emb[:, 1]
    sel = {"L": pick("L", emb, df, proto, use_nuclei),
           "R": pick("R", emb, df, proto, use_nuclei)}
    print(f"{comp} (prototype_{proto}): {len(df)} tiles; L={len(sel['L'])}, "
          f"R={len(sel['R'])}, use_nuclei={use_nuclei}")

    figW, figH = 15.0, 12.0
    fig = plt.figure(figsize=(figW, figH))
    cx0, cy0, cw, ch = 0.18, 0.06, 0.62, 0.84
    ax = fig.add_axes([cx0, cy0, cw, ch])
    sc = ax.scatter(u1, u2, c=soft, cmap="RdBu_r", vmin=0, vmax=1,
                    s=12, alpha=0.6, linewidths=0, rasterized=True)
    ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(labelsize=7)

    cax = fig.add_axes([cx0, cy0 + ch + 0.015, cw, 0.018])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0 (non-amp)", "0.5", "1 (MYCN-amp)"], fontsize=8)
    cb.ax.xaxis.set_ticks_position("top"); cb.ax.xaxis.set_label_position("top")
    cb.set_label("P(MYCN-amp) soft label", fontsize=8)

    th_in = 0.95
    twf, thf = th_in / figW, th_in / figH

    def place(side, x_left):
        pts = sorted(sel[side], key=lambda p: u2[p], reverse=True)    # top→bottom
        n = len(pts)
        top, bot = cy0 + ch - thf, cy0
        ys = np.linspace(top, bot, n) if n > 1 else [(top + bot) / 2]
        for kk, p in enumerate(pts):
            a = fig.add_axes([x_left, ys[kk], twf, thf])
            a.imshow(mpimg.imread(tile_image_path(proto, df.index[p])))
            a.set_xticks([]); a.set_yticks([])
            col = plt.cm.RdBu_r(float(soft[p]))
            for sp in a.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2.5)
            ax.scatter([u1[p]], [u2[p]], s=70, facecolors="none",
                       edgecolors="black", linewidths=1.2, zorder=6)
            anchor = (1.0, 0.5) if side == "L" else (0.0, 0.5)
            con = ConnectionPatch(xyA=anchor, coordsA=a.transAxes,
                                  xyB=(u1[p], u2[p]), coordsB=ax.transData,
                                  color="0.45", lw=0.7, alpha=0.8, zorder=2)
            con.set_clip_on(False)
            fig.add_artist(con)

    place("L", 0.012)
    place("R", 1 - 0.012 - twf)
    out = os.path.join(OUT, f"{comp}_umap_tilemontage_tier2.pdf")
    fig.savefig(out, format="pdf"); plt.close(fig)
    print(f"  wrote results/{comp}_umap_tilemontage_tier2.pdf")


for comp, cfg in COMPONENTS.items():
    render(comp, cfg)
print("Done.")
