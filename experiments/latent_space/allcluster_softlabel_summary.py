# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
allcluster_softlabel_summary.py — one compact panel summarising the soft-label
separation across ALL FOUR MYCN-associated clusters (C2,C3,C5,C6), so the
"every sub-population shifts with MYCN" heterogeneity message fits one A4 panel.

Reads the per-tile soft labels produced by cell_latent_analysis.py and renders
TWO companion panels for main Figure 4:

  • Panel 4a (allcluster_softlabel_summary.pdf) — SLIDE-LEVEL: per cluster, the
    per-slide median P(MYCN-amp) for non-amp vs MYCN-amp slides (strip + group
    median). Each point is one slide (tumour sample), the study's analysis unit
    — not one patient.

  • Panel 4b (allcluster_softlabel_violin.pdf) — TILE-LEVEL: per cluster, the
    full pooled per-tile P(MYCN-amp) distribution as a split violin, non-amp
    (left, blue) | MYCN-amp (right, red). A density/violin is appropriate here
    because each cluster pools thousands of tiles (NOT the 10 per-slide medians
    of 4a); it is deliberately tile-level so it complements the slide-level 4a.

Colours follow the canonical scheme (non-amp #4F74C8 blue, MYCN-amp #D94F3D red).

Outputs: results/allcluster_softlabel_summary.pdf
         results/allcluster_softlabel_violin.pdf
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CLUSTERS = [("component2", "C2\nnecrotic"), ("component3", "C3\ncellular tumour"),
            ("component5", "C5\ndense tumour"), ("component6", "C6\nhaemorrhagic")]
C_NON, C_AMP = "#4F74C8", "#D94F3D"

fig, ax = plt.subplots(figsize=(7.5, 4.5))
rng = np.random.default_rng(42)
xt, xtl = [], []
for i, (comp, label) in enumerate(CLUSTERS):
    df = pd.read_csv(os.path.join(RES, f"{comp}_soft_labels.csv"))
    per_slide = df.groupby(["slide", "true_label"])["soft_label_mycn_amp"].median().reset_index()
    for j, (lab, col) in enumerate([(0, C_NON), (1, C_AMP)]):
        vals = per_slide.loc[per_slide.true_label == lab, "soft_label_mycn_amp"].values
        x = i + (j - 0.5) * 0.42
        ax.scatter(x + rng.uniform(-0.07, 0.07, len(vals)), vals, c=col, s=26,
                   alpha=0.75, linewidths=0, zorder=3)
        ax.plot([x - 0.13, x + 0.13], [np.median(vals)] * 2, color="black", lw=2, zorder=4)
    xt.append(i); xtl.append(label)
ax.axhline(0.5, color="grey", ls=":", lw=0.8)
ax.set_xticks(xt); ax.set_xticklabels(xtl, fontsize=9)
ax.set_ylabel("Per-slide median P(MYCN-amp) soft label", fontsize=10)
ax.set_ylim(0, 1)
ax.legend(handles=[mpatches.Patch(color=C_NON, label="non-amplified"),
                   mpatches.Patch(color=C_AMP, label="MYCN-amplified")],
          fontsize=9, loc="upper left")
plt.tight_layout()
fig.savefig(os.path.join(RES, "allcluster_softlabel_summary.pdf"), format="pdf", bbox_inches="tight")
print("wrote results/allcluster_softlabel_summary.pdf")


# ── Panel 4b: all-four split-violins of pooled per-tile soft labels ───────────
# Tile-level companion to 4a: each cluster shows the full per-tile P(MYCN-amp)
# distribution, split non-amp (left, blue) | MYCN-amp (right, red).
def _half_violin(ax, data, pos, side, color):
    parts = ax.violinplot([data], positions=[pos], showextrema=False,
                          showmedians=False, widths=0.85)
    body  = parts["bodies"][0]
    verts = body.get_paths()[0].vertices
    # collapse the far side onto the centre line → a half violin
    verts[:, 0] = (np.clip(verts[:, 0], -np.inf, pos) if side == "left"
                   else np.clip(verts[:, 0], pos, np.inf))
    body.set_facecolor(color); body.set_alpha(0.55); body.set_edgecolor("none")
    med    = float(np.median(data))
    x0, x1 = (pos - 0.30, pos) if side == "left" else (pos, pos + 0.30)
    ax.plot([x0, x1], [med, med], color="black", lw=1.6, zorder=4)

fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
xt2, xtl2 = [], []
for i, (comp, label) in enumerate(CLUSTERS):
    df = pd.read_csv(os.path.join(RES, f"{comp}_soft_labels.csv"))
    non_vals = df.loc[df.true_label == 0, "soft_label_mycn_amp"].values
    amp_vals = df.loc[df.true_label == 1, "soft_label_mycn_amp"].values
    _half_violin(ax2, non_vals, i, "left",  C_NON)
    _half_violin(ax2, amp_vals, i, "right", C_AMP)
    xt2.append(i); xtl2.append(label)
ax2.axhline(0.5, color="grey", ls=":", lw=0.8)
ax2.set_xticks(xt2); ax2.set_xticklabels(xtl2, fontsize=9)
ax2.set_ylabel("Per-tile P(MYCN-amp) soft label", fontsize=10)
ax2.set_ylim(0, 1)
ax2.legend(handles=[mpatches.Patch(color=C_NON, alpha=0.55, label="non-amplified"),
                    mpatches.Patch(color=C_AMP, alpha=0.55, label="MYCN-amplified")],
           fontsize=9, loc="upper left")
plt.tight_layout()
fig2.savefig(os.path.join(RES, "allcluster_softlabel_violin.pdf"), format="pdf", bbox_inches="tight")
print("wrote results/allcluster_softlabel_violin.pdf")
