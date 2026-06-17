# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
lme_dotplot.py — compact LME effect dot-plot that replaces the four tall
per-cluster z-statistic bar charts (and the count-based modality summary) with a
single panel: curated cell features (rows, grouped by modality) x the four
MYCN-associated clusters (columns).

Dot colour = LME z-statistic (blue = lower in MYCN-amp, red = higher in
MYCN-amp); a black ring + asterisk marks FDR-significant effects; faint dots are
present-but-non-significant; an "x" means the feature was not in that cluster's
filtered feature set. Values read from component{N}_mixed_effects_stats.csv.

Output: results/lme_dotplot.pdf
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CLUSTERS = [("component2", "C2\nnecrotic"), ("component3", "C3\ncellular\ntumour"),
            ("component5", "C5\ndense\ntumour"), ("component6", "C6\nhaemorrhagic")]
# (modality, display label, exact csv feature name)
ROWS = [
    ("connectivity",   "Closeness centrality (mean)",   "mean of cells' closeness_centrality"),
    ("connectivity",   "Degree centrality (mean)",      "mean of cells' degree_centrality"),
    ("connectivity",   "Clustering coefficient (mean)", "mean of cells' clustering_coefficient"),
    ("connectivity",   "Degree (mean)",                 "mean of cells' degree"),
    ("diversity",      "Simpson index (global)",        "Gloabl Simpson index"),
    ("diversity",      "Shannon index (global)",        "Gloabl Shannon index"),
    ("diversity",      "Richness (global)",             "Gloabl Richness (number of cell-types present)"),
    ("morphology",     "Nuclear eccentricity (mean)",   "neuroblast cells: mean of their eccentricity"),
    ("morphology",     "Nuclear solidity (mean)",       "neuroblast cells: mean of their solidity"),
    ("morphology",     "Nuclear perimeter (std)",       "neuroblast cells: std of their perimeter"),
    ("colour/texture", "Hue (mean H)",                  "mean_h"),
    ("colour/texture", "Red (mean R)",                  "mean_r"),
    ("colour/texture", "Value (mean V)",                "mean_v"),
    ("counts",         "Necrosis-cell count",           "number of necrosis cells"),
    ("counts",         "Neuroblast-cell count",         "number of neuroblast cells"),
]
VMAX = 6.0
cmap = plt.cm.RdBu_r

# ── load z + fdr_sig per cluster ──────────────────────────────────────────────
data = {}
for comp, _ in CLUSTERS:
    df = pd.read_csv(os.path.join(RES, f"{comp}_mixed_effects_stats.csv")).set_index("feature")
    data[comp] = {"z": df["z_stat"].to_dict(), "sig": df["fdr_sig"].to_dict()}

# missing-feature report (so curated names can be corrected if needed)
for comp, lab in CLUSTERS:
    miss = [f for _, _, f in ROWS if f not in data[comp]["z"]]
    if miss:
        print(f"[{lab.splitlines()[0]}] curated features NOT found ({len(miss)}): {miss}")

nrows = len(ROWS)
mods  = [m for m, _, _ in ROWS]
fig, ax = plt.subplots(figsize=(6.4, 7.4))
for yi, (mod, disp, feat) in enumerate(ROWS):
    y = nrows - 1 - yi
    for xi, (comp, _) in enumerate(CLUSTERS):
        z = data[comp]["z"].get(feat, np.nan)
        sig = bool(data[comp]["sig"].get(feat, False))
        if np.isnan(z):
            ax.scatter(xi, y, marker="x", s=20, color="0.8", zorder=2)
            continue
        col = cmap(np.clip((z + VMAX) / (2 * VMAX), 0, 1))
        if sig:
            ax.scatter(xi, y, s=320, color=col, edgecolors="black", linewidths=2.0, zorder=3)
        else:
            ax.scatter(xi, y, s=150, color=col, edgecolors="0.6", linewidths=0.5, alpha=0.5, zorder=3)

ax.set_yticks(range(nrows)); ax.set_yticklabels([d for _, d, _ in ROWS][::-1], fontsize=8.5)
ax.set_xticks(range(len(CLUSTERS))); ax.set_xticklabels([c[1] for c in CLUSTERS], fontsize=8.5)
ax.set_xlim(-0.5, len(CLUSTERS) - 0.5); ax.set_ylim(-0.7, nrows - 0.3)
# modality group separators
for yi in range(1, nrows):
    if mods[yi] != mods[yi - 1]:
        ax.axhline(nrows - 0.5 - yi, color="0.85", lw=0.8, zorder=1)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-VMAX, VMAX)); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("LME z  (← lower in MYCN-amp   |   higher in MYCN-amp →)", fontsize=8)
ax.legend(handles=[
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.7", markeredgecolor="black",
           markeredgewidth=2.0, markersize=13, label="FDR-significant (black ring)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.85", markeredgecolor="0.6",
           markersize=9, alpha=0.6, label="not significant"),
    Line2D([0], [0], marker="x", color="0.8", linestyle="None", markersize=7, label="not in cluster"),
], fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False)
plt.tight_layout()
fig.savefig(os.path.join(RES, "lme_dotplot.pdf"), format="pdf", bbox_inches="tight")
print("wrote results/lme_dotplot.pdf")
