# Pheno-MYCN — survival analysis.
# 09 — (#1) Does the H&E-derived MYCN call recover MYCN's prognostic stratification?
#      (#3) Do the phenotype clusters add survival signal independent of MYCN? (Cox)
#
# #1  Out-of-fold predicted P(MYCN-amp) (from 08) vs survival: KM by the model's
#     predicted MYCN call, Cox on the continuous score, and a concordance/discordance
#     breakdown against the true FISH MYCN status. Asks whether morphology alone
#     recovers the prognostic split that the molecular assay provides.
# #3  Continuous, MYCN-adjusted Cox on per-slide phenotype scores for the single
#     clusters and the combined sets {C3+C5},{C2+C6},{all} (combine = SUM of per-slide
#     scores, per the locked decision), for both the mean and dom statistics — the
#     rigorous version of the median-split screen / within-MYCN test.
#
# Outputs (results/):
#   km_curves/km_by_predicted_mycn.pdf            — #1 KM panel (vector, no title, Arial)
#   predicted_mycn_survival_summary.txt           — #1 medians, log-rank, Cox HR, C-index, concordance
#   cox_cluster_scores_summary.csv                — #3 univariable + MYCN-adjusted Cox per set/stat
# License: GPL-3.0.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("PHENO_MYCN_ROOT", "/media/digitalpathology/6BE7-FCF8/phyno_mycn")
COHORT = os.path.join(BASE, "data/survival_per_slide.csv")
OOF    = os.path.join(BASE, "results/oof_mycn_predictions.csv")
GMM    = os.path.join(ROOT, "github_repo/experiments/gmm_responsibility/results/per_slide_stats.csv")
OUT    = os.path.join(BASE, "results")
KMOUT  = os.path.join(OUT, "km_curves")
os.makedirs(KMOUT, exist_ok=True)

C_AMP, C_NON = "#D94F3D", "#4F74C8"

# ── survival cohort ─────────────────────────────────────────────────────────
surv = pd.read_csv(COHORT).dropna(subset=["OS_time_days", "event"]).copy()
surv["OS_time_days"] = surv["OS_time_days"].astype(float)
surv = surv[surv["OS_time_days"] > 0].copy()
surv["event"] = surv["event"].astype(int)
surv["months"] = surv["OS_time_days"] / 30.44
print(f"survival cohort: {len(surv)} slides")

# ════════════════════════════════════════════════════════════════════════════
# #1 — predicted (H&E-derived) MYCN vs survival
# ════════════════════════════════════════════════════════════════════════════
oof = pd.read_csv(OOF)
# The published CV uses repeated 171/9/9 holdouts, so a slide can appear in more
# than one fold's test set (and many slides never appear). Collapse to ONE honest
# out-of-fold prediction per slide by averaging across the folds that held it out.
oof_s = (oof.groupby("slide")
            .agg(true_label=("true_label", "first"), mycn_prob=("mycn_prob", "mean"),
                 n_testfolds=("fold", "nunique")).reset_index())
oof_s["pred_label"] = (oof_s["mycn_prob"] >= 0.5).astype(int)
d = surv.merge(oof_s, left_on="slide_name", right_on="slide", how="inner")
print(f"#1 OOF coverage: {oof_s.slide.nunique()} unique held-out slides; "
      f"{len(d)} have survival data")

L = []
def km_median_logrank(frame, group_col, g1, g0, label):
    a = frame[frame[group_col] == g1]; b = frame[frame[group_col] == g0]
    lr = logrank_test(b.months, a.months, event_observed_A=b.event, event_observed_B=a.event)
    kmf = KaplanMeierFitter()
    kmf.fit(a.months, a.event); m1 = kmf.median_survival_time_
    kmf.fit(b.months, b.event); m0 = kmf.median_survival_time_
    L.append(f"{label}: median OS {g1}={m1:.1f} vs {g0}={m0:.1f} months; log-rank p={lr.p_value:.4g} (n={len(a)}/{len(b)})")
    return lr.p_value, m1, m0

# KM by the model's predicted MYCN call (argmax / threshold 0.5)
p_pred, m1p, m0p = km_median_logrank(d, "pred_label", 1, 0, "Predicted MYCN (H&E)")
# KM by true FISH MYCN (reference benchmark)
p_true, m1t, m0t = km_median_logrank(d, "mycn_perslide", 1, 0, "True MYCN (FISH)")

# Cox: continuous predicted prob, and true MYCN, with C-index
def cox_one(frame, covar):
    cph = CoxPHFitter().fit(frame[["months", "event", covar]], "months", "event")
    s = cph.summary.loc[covar]
    return float(np.exp(s["coef"])), float(s["p"]), float(cph.concordance_index_)
d["mycn_prob_z"] = (d["mycn_prob"] - d["mycn_prob"].mean()) / d["mycn_prob"].std()
hr_pred, pp_pred, c_pred = cox_one(d, "mycn_prob_z")
hr_true, pp_true, c_true = cox_one(d, "mycn_perslide")
L.append("")
L.append(f"Cox continuous predicted P(MYCN-amp) [per SD]: HR={hr_pred:.2f}, p={pp_pred:.4g}, C-index={c_pred:.3f}")
L.append(f"Cox true MYCN (FISH, binary):                   HR={hr_true:.2f}, p={pp_true:.4g}, C-index={c_true:.3f}")

# concordance / discordance of the H&E call vs FISH
ct = pd.crosstab(d["mycn_perslide"], d["pred_label"])
L.append(""); L.append("Concordance (rows=true FISH, cols=predicted):"); L.append(ct.to_string())
disc = d[d["mycn_perslide"] != d["pred_label"]]
conc = d[d["mycn_perslide"] == d["pred_label"]]
L.append(f"concordant n={len(conc)} (event rate {conc.event.mean():.2f}); "
         f"discordant n={len(disc)} (event rate {disc.event.mean():.2f})")
# discordant subgroups: H&E says amp / FISH non-amp, and vice versa
for (t, p, name) in [(0, 1, "FISH non-amp but H&E-amp"), (1, 0, "FISH amp but H&E-non-amp")]:
    sub = d[(d.mycn_perslide == t) & (d.pred_label == p)]
    if len(sub):
        L.append(f"  {name}: n={len(sub)}, events={int(sub.event.sum())}, "
                 f"median OS={KaplanMeierFitter().fit(sub.months, sub.event).median_survival_time_:.1f} mo")

# ── #1 KM panel (house KM style: grid, no in-figure p — matches 01/04/06; the
#    log-rank p is reported in the summary txt / caption / main text, not on the plot) ──
fig, ax = plt.subplots(figsize=(6, 5))
kmf = KaplanMeierFitter()
g0 = d[d.pred_label == 0]; g1 = d[d.pred_label == 1]
kmf.fit(g0.months, g0.event, label=f"Predicted non-amp (n={len(g0)})"); kmf.plot_survival_function(ax=ax, ci_show=True, color=C_NON)
kmf.fit(g1.months, g1.event, label=f"Predicted MYCN-amp (n={len(g1)})"); kmf.plot_survival_function(ax=ax, ci_show=True, color=C_AMP)
ax.set_xlabel("Overall survival (months)"); ax.set_ylabel("Survival probability"); ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.25)
ax.legend(loc="upper right", framealpha=0.95)
plt.tight_layout()
fig.savefig(os.path.join(KMOUT, "km_by_predicted_mycn.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(KMOUT, "km_by_predicted_mycn.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

with open(os.path.join(OUT, "predicted_mycn_survival_summary.txt"), "w") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))

# ════════════════════════════════════════════════════════════════════════════
# #3 — MYCN-adjusted Cox on phenotype-cluster scores (singles + combined SUMS)
# ════════════════════════════════════════════════════════════════════════════
g = pd.read_csv(GMM)
m = surv.merge(g, left_on="slide_name", right_on="slide", how="inner")
print(f"\n#3 merged with per-slide GMM scores: {len(m)} slides")
sets = {"C2": [2], "C3": [3], "C5": [5], "C6": [6],
        "C3+C5": [3, 5], "C2+C6": [2, 6], "C2+C3+C5+C6": [2, 3, 5, 6]}
rows = []
for stat in ("mean", "dom"):
    for name, ks in sets.items():
        score = m[[f"{stat}_c{k}" for k in ks]].sum(axis=1)
        z = (score - score.mean()) / score.std()
        dd = pd.DataFrame({"months": m.months.values, "event": m.event.values,
                           "score": z.values, "mycn": m.mycn_perslide.values})
        u = CoxPHFitter().fit(dd[["months", "event", "score"]], "months", "event").summary.loc["score"]
        a = CoxPHFitter().fit(dd[["months", "event", "score", "mycn"]], "months", "event").summary.loc["score"]
        rows.append(dict(set=name, stat=stat,
            uni_HR=round(float(u["exp(coef)"]), 3), uni_lo=round(float(u["exp(coef) lower 95%"]), 3),
            uni_hi=round(float(u["exp(coef) upper 95%"]), 3), uni_p=round(float(u["p"]), 4),
            adj_HR=round(float(a["exp(coef)"]), 3), adj_lo=round(float(a["exp(coef) lower 95%"]), 3),
            adj_hi=round(float(a["exp(coef) upper 95%"]), 3), adj_p=round(float(a["p"]), 4)))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "cox_cluster_scores_summary.csv"), index=False)
print("\n=== #3 Cox on cluster scores (HR per SD; uni = univariable, adj = +true MYCN) ===")
print(res.to_string(index=False))

# ── 6d forest: univariable vs MYCN-adjusted HR per SD (dom statistic) ──
from matplotlib.lines import Line2D
fr = res[res.stat == "dom"].set_index("set").loc[list(sets.keys())]
C_UNI, C_ADJ = "#555555", "#4F74C8"
yy = np.arange(len(fr))
fig, ax = plt.subplots(figsize=(6, 5))
for i, (_, r) in enumerate(fr.iterrows()):
    ax.plot([r.uni_lo, r.uni_hi], [yy[i] + 0.15] * 2, color=C_UNI, lw=1.6, zorder=2)
    ax.plot(r.uni_HR, yy[i] + 0.15, "o", color=C_UNI, ms=5, zorder=3)
    ax.plot([r.adj_lo, r.adj_hi], [yy[i] - 0.15] * 2, color=C_ADJ, lw=1.6, zorder=2)
    ax.plot(r.adj_HR, yy[i] - 0.15, "s", color=C_ADJ, ms=5, zorder=3)
ax.grid(axis="x", alpha=0.25)
ax.axvline(1.0, ls="--", color="0.4", lw=1, zorder=1)
ax.set_yticks(yy); ax.set_yticklabels(fr.index); ax.invert_yaxis()
ax.set_xscale("log"); ax.set_xticks([0.6, 0.8, 1, 1.25, 1.6])
ax.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.6"])
ax.set_xlabel("Hazard ratio per SD (95% CI)")
ax.legend([Line2D([0], [0], color=C_UNI, marker="o", ls="-"),
           Line2D([0], [0], color=C_ADJ, marker="s", ls="-")],
          ["univariable", "adjusted for MYCN"], loc="lower right", framealpha=0.95)
plt.tight_layout()
fig.savefig(os.path.join(KMOUT, "cox_cluster_forest.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(KMOUT, "cox_cluster_forest.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved cox_cluster_forest.pdf")
