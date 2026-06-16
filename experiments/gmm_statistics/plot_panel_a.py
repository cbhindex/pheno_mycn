# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
Figure 2 — Panel a: GMM phenotype-cluster-count selection (K=2..9).

Aggregates the per-fold cross-validation metrics for each candidate cluster
count K and plots mean ± SD across the 10 folds, one PDF per metric (Accuracy,
Precision, Recall, F1-score, AUC). No titles, no axis labels (added manually in
Illustrator); tick marks/values retained.

This is the sole producer of the K-selection summary statistics; the same
numbers populate Supplementary Table `tab:gmm_k_selection`. The aggregated
mean ± SD per K is therefore also written to CSV so the table has a single
canonical source (see manuscript_correction.md, canonical-stat-script rule).

Inputs (resolved under $PHENO_MYCN_ROOT; not shipped — patient-derived):
    olga_refactered/intermediate_outputs/training_checkpoints/
        pheno_mycn_gmm_k{K}/fold{0..9}/result.csv

Outputs (results/, git-ignored):
    panel_a{1..5}_{accuracy,precision,recall,f1,auc}.pdf
    panel_a_k_selection_metrics.csv   — mean ± SD per metric per K (canonical)

Run:
    export PHENO_MYCN_ROOT=/path/to/pheno_mycn
    python plot_panel_a.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = os.path.join(os.environ.get("PHENO_MYCN_ROOT", "/path/to/pheno_mycn"), "olga_refactered")
CKPT_DIR = os.path.join(BASE, "intermediate_outputs/training_checkpoints")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

K_VALUES = list(range(2, 10))

# (metric key, file slug, marker, colour)
METRICS = [
    ("accuracy",  "panel_a1_accuracy",  "s", "#1f77b4"),
    ("precision", "panel_a2_precision", "^", "#2ca02c"),
    ("recall",    "panel_a3_recall",    "v", "#d62728"),
    ("f1",        "panel_a4_f1",        "D", "#ff7f0e"),
    ("auc",       "panel_a5_auc",       "P", "#9467bd"),
]


def load_metrics():
    rows = []
    for k in K_VALUES:
        for fold in range(10):
            p = os.path.join(CKPT_DIR, f"pheno_mycn_gmm_k{k}", f"fold{fold}", "result.csv")
            if not os.path.isfile(p):
                continue
            df = pd.read_csv(p)
            r = df.iloc[0]
            rows.append({
                "k": k, "fold": fold,
                "accuracy":  float(r["test_accuracy"]),
                "precision": float(r["test_precision"]),
                "recall":    float(r["test_recall"]),
                "f1":        float(r["test_f1_score"]),
                "auc":       float(r["auc"]),
            })
    return pd.DataFrame(rows)


def main():
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    df = load_metrics()
    print(f"Loaded {len(df)} rows across K={sorted(df['k'].unique())}")
    # Aggregate metrics only (drop the fold index, which is not a metric).
    summary = df.drop(columns=["fold"]).groupby("k").agg(["mean", "std"])

    # Canonical CSV: mean ± SD per metric per K (source for tab:gmm_k_selection).
    csv_path = os.path.join(OUT_DIR, "panel_a_k_selection_metrics.csv")
    flat = summary.copy()
    flat.columns = [f"{metric}_{stat}" for metric, stat in flat.columns]
    flat.insert(0, "n_folds", df.groupby("k").size())
    flat.reset_index().to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")

    for metric, slug, marker, colour in METRICS:
        means = summary[metric]["mean"].reindex(K_VALUES).values
        stds  = summary[metric]["std"].reindex(K_VALUES).values

        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        ax.errorbar(
            K_VALUES, means, yerr=stds,
            fmt=marker, markersize=7, color=colour, ecolor=colour,
            capsize=4, linestyle="none", elinewidth=1.0, markeredgewidth=0.8,
        )
        # No axis labels; ticks retained
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0.4, 1.0)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        out = os.path.join(OUT_DIR, f"{slug}.pdf")
        fig.tight_layout()
        fig.savefig(out, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
