# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
07_progression_summary.py — one panel showing the survival "progression":
single clusters -> informative pairs {C3+C5},{C2+C6} -> all {C2+C3+C5+C6}.

Bars are -log10(descriptive log-rank p) for the dom (dominant-tile-fraction)
statistic, read from km_combined_summary.csv (produced by 06). The dashed line
marks p=0.05. Demonstrates: the dominant tumour axis (C3, C5, C3+C5) is null;
the MYCN-amp-enriched minor clusters (C6, then C2+C6) carry the signal; and
combining everything does not exceed C6 alone.

Output: results/km_curves/progression_summary.pdf
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "km_curves")
df = pd.read_csv(os.path.join(OUT, "km_combined_summary.csv"))
dom = df[df["stat"] == "dom"].set_index("set")

ORDER = [("c2", "C2"), ("c3", "C3"), ("c5", "C5"), ("c6", "C6"),
         ("c3_c5", "C3+C5"), ("c2_c6", "C2+C6"), ("c2_c3_c5_c6", "C2+C3+C5+C6")]
labels, vals, colors = [], [], []
for key, lab in ORDER:
    if key not in dom.index:
        continue
    p = float(dom.loc[key, "logrank_p"])
    labels.append(lab); vals.append(-np.log10(p))
    # colour: singles grey, pairs/all by membership
    if key in ("c3", "c5", "c3_c5"):
        colors.append("#4F74C8")          # dominant tumour axis (null)
    elif key in ("c2", "c6", "c2_c6"):
        colors.append("#D94F3D")          # MYCN-amp-enriched minor (signal)
    else:
        colors.append("#8c564b")          # all combined

fig, ax = plt.subplots(figsize=(7.5, 4.2))
bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="none")
ax.axhline(-np.log10(0.05), color="black", ls="--", lw=1)
ax.text(len(labels) - 0.5, -np.log10(0.05) + 0.05, "p = 0.05", ha="right", fontsize=8)
for i, (key, lab) in enumerate([o for o in ORDER if o[0] in dom.index]):
    p = float(dom.loc[key, "logrank_p"])
    ax.text(i, vals[i] + 0.03, f"{p:.3f}", ha="center", va="bottom", fontsize=7)
# group separators
ax.axvline(3.5, color="grey", lw=0.6, alpha=0.5)
ax.axvline(5.5, color="grey", lw=0.6, alpha=0.5)
ax.text(1.5, max(vals) * 1.05, "single clusters", ha="center", fontsize=8, color="grey")
ax.text(4.5, max(vals) * 1.05, "pairs", ha="center", fontsize=8, color="grey")
ax.text(6.0, max(vals) * 1.05, "all", ha="center", fontsize=8, color="grey")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel(r"$-\log_{10}$(log-rank $p$), dom statistic", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "progression_summary.pdf"), format="pdf", bbox_inches="tight")
print("wrote results/km_curves/progression_summary.pdf")
