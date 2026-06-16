# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
allcluster_lme_modality_summary.py — one panel showing that different
sub-populations are distinguished by different FEATURE MODALITIES.

For each cluster (C2,C3,C5,C6) it counts BH-FDR-significant LME features,
split by direction (left = higher in non-amp / lower in MYCN-amp; right =
higher in MYCN-amp) and coloured by modality (connectivity, diversity,
morphology, colour/texture, counts). Diverging horizontal stacked bars.

Result (the heterogeneity story): C3/C5 are dominated by spatial-connectivity
features reduced in MYCN-amp; C2/C6 by colour/intensity features raised in
MYCN-amp (necrosis/haemorrhage are anuclear, colour-defined tissues).

Output: results/allcluster_lme_modality_summary.pdf
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CLUSTERS = [("component2", "C2 necrotic"), ("component3", "C3 cellular tumour"),
            ("component5", "C5 dense tumour"), ("component6", "C6 haemorrhagic")]
MOD_COLORS = {"connectivity": "#1f77b4", "diversity": "#2ca02c",
              "morphology": "#9467bd", "colour/texture": "#ff7f0e", "counts": "#8c564b"}

def modality(feat):
    f = feat.lower()
    if any(k in f for k in ["centrality", "clustering_coefficient", "degree"]):
        return "connectivity"
    if any(k in f for k in ["simpson", "shannon", "richness", "renyi", "entropy", "diversity"]):
        return "diversity"
    if any(k in f for k in ["mean_h", "mean_s", "mean_v", "mean_r", "mean_g", "mean_b",
                            "mean_gray", "std_b", "std_s", "std_h", "std_r", "std_g",
                            "glcm", "intensity"]):
        return "colour/texture"
    if any(k in f for k in ["number of", "count"]):
        return "counts"
    return "morphology"

fig, ax = plt.subplots(figsize=(8, 4.5))
mods = list(MOD_COLORS)
for i, (comp, label) in enumerate(CLUSTERS):
    df = pd.read_csv(os.path.join(RES, f"{comp}_mixed_effects_stats.csv"))
    sig = df[df["fdr_sig"] == True].copy()
    sig["mod"] = sig["feature"].apply(modality)
    for side, sign in [("neg", -1), ("pos", +1)]:
        sub = sig[(sig["z_stat"] < 0) if side == "neg" else (sig["z_stat"] > 0)]
        left = 0.0
        for m in mods:
            n = int((sub["mod"] == m).sum())
            if n:
                ax.barh(i, sign * n, left=left if sign > 0 else -left - n,
                        color=MOD_COLORS[m], edgecolor="white", height=0.6)
                left += n
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(CLUSTERS))); ax.set_yticklabels([c[1] for c in CLUSTERS], fontsize=9)
ax.set_xlabel("← higher in non-amp   |   # FDR-significant LME features   |   higher in MYCN-amp →",
              fontsize=9)
ax.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in MOD_COLORS.items()],
          fontsize=8, loc="lower right", ncol=2)
plt.tight_layout()
fig.savefig(os.path.join(RES, "allcluster_lme_modality_summary.pdf"), format="pdf", bbox_inches="tight")
print("wrote results/allcluster_lme_modality_summary.pdf")
