# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
06_km_combined_components.py — Kaplan–Meier on COMBINED phenotype-cluster scores.

Extends 04_km_gmm_components.py (single-cluster screen) to merged cluster sets.
"Merging clusters" = treating them as one cluster, which is exactly summing the
per-slide scores:
    dom score of {3,5}  = dom_c3 + dom_c5    (fraction of tiles whose top
                                              assignment is 3 OR 5)
    mean score of {3,5} = mean_c3 + mean_c5  (mean membership in 3 or 5)

Sets analysed (Comment 9):
    {2}, {3}, {5}, {6}            -- single clusters (for reference)
    {3,5}                         -- the two dominant tumour-morphology clusters
    {2,6}                         -- the two MYCN-amp-enriched minor clusters
    {2,3,5,6}                     -- all informative clusters

For each set and each statistic (dom, mean):
    - per-slide combined score, median split, KM + descriptive log-rank
    - within-MYCN-stratified log-rank (does the signal survive adjustment for MYCN?)

Outputs (results/km_curves/):
    km_combined_<set>_<stat>.{pdf,png}     -- KM per set/statistic
    km_combined_summary.csv                -- log-rank p (overall + within-MYCN)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style  # noqa: F401
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

BASE   = os.environ.get("PHENO_MYCN_ROOT", "/path/to/pheno_mycn")
SURV   = os.path.dirname(os.path.abspath(__file__))
COHORT = os.path.join(SURV, "data/survival_per_slide.csv")
GMM    = os.path.join(SURV, "..", "gmm_responsibility/results/per_slide_stats.csv")
OUT    = os.path.join(os.path.dirname(__file__), "results", "km_curves")
os.makedirs(OUT, exist_ok=True)

# cluster sets to merge
SETS = {
    "c2":          [2],
    "c3":          [3],
    "c5":          [5],
    "c6":          [6],
    "c3_c5":       [3, 5],
    "c2_c6":       [2, 6],
    "c2_c3_c5_c6": [2, 3, 5, 6],
}

surv = pd.read_csv(COHORT)
g    = pd.read_csv(GMM)
df = surv.merge(
    g[["slide"] + [f"mean_c{i}" for i in range(1, 7)]
                 + [f"dom_c{i}"  for i in range(1, 7)]],
    left_on="slide_name", right_on="slide",
).dropna(subset=["OS_time_days", "event"])
df["OS_time_months"] = df.OS_time_days.astype(float) / 30.44
df["event"] = df.event.astype(int)
print(f"N = {len(df)} slides")

records = []

def km_plot(score, label_set, stat, fname):
    med = score.median()
    low  = df[score <= med]
    high = df[score >  med]
    if low.empty or high.empty or score.nunique() < 4:
        print(f"  skip {fname}: degenerate split")
        return None
    lr = logrank_test(low.OS_time_months, high.OS_time_months,
                      event_observed_A=low.event, event_observed_B=high.event)

    # within-MYCN-stratified log-rank
    within = {}
    for mycn in (0, 1):
        s = df[df.mycn_perslide == mycn]
        sc = score[df.mycn_perslide == mycn]
        m = sc.median()
        lo, hi = s[sc <= m], s[sc > m]
        if len(lo) >= 5 and len(hi) >= 5:
            wlr = logrank_test(lo.OS_time_months, hi.OS_time_months,
                               event_observed_A=lo.event, event_observed_B=hi.event)
            within[mycn] = wlr.p_value
        else:
            within[mycn] = float("nan")

    fig, ax = plt.subplots(figsize=(6, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(low.OS_time_months, low.event,
            label=f"low (n={len(low)}, ev={int(low.event.sum())})")
    kmf.plot_survival_function(ax=ax, color="#268bd2", ci_show=True)
    kmf.fit(high.OS_time_months, high.event,
            label=f"high (n={len(high)}, ev={int(high.event.sum())})")
    kmf.plot_survival_function(ax=ax, color="#dc322f", ci_show=True)
    ax.set_xlabel("Overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname + ".pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT, fname + ".png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → {fname}.pdf  p={lr.p_value:.3g}  "
          f"(within MYCN: non-amp={within[0]:.3g}, amp={within[1]:.3g})")
    return {"set": label_set, "stat": stat,
            "low_n": len(low), "high_n": len(high),
            "low_ev": int(low.event.sum()), "high_ev": int(high.event.sum()),
            "logrank_p": lr.p_value,
            "within_nonamp_p": within[0], "within_amp_p": within[1]}

for stat in ("dom", "mean"):
    print(f"\n=== {stat} score ===")
    for set_name, comps in SETS.items():
        score = sum(df[f"{stat}_c{k}"] for k in comps)
        rec = km_plot(score, set_name, stat, f"km_combined_{set_name}_{stat}")
        if rec:
            records.append(rec)

out_csv = os.path.join(OUT, "km_combined_summary.csv")
pd.DataFrame(records).to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")
print("Done.")
