# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
allcluster_softlabel_summary.py — one compact panel summarising the soft-label
separation across ALL FOUR MYCN-associated clusters (C2,C3,C5,C6), so the
"every sub-population shifts with MYCN" heterogeneity message fits one A4 panel.

Reads the per-tile soft labels produced by cell_latent_analysis.py and plots,
per cluster, the per-slide median P(MYCN-amp) for non-amp vs MYCN-amp patients
(strip + group median).

Output: results/allcluster_softlabel_summary.pdf
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
