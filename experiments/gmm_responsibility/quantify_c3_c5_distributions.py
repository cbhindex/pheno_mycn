# Pheno-MYCN — additional experiments.
# Quantify the C3 / C5 per-tile phenotype-score distribution difference between
# MYCN-amplified and non-amplified tumours, to replace the qualitative
# ("visibly different distributions") claim in Section 2.2.
#
# Two complementary, clustering-aware tests computed HERE:
#   (1) Slide-level summary stats (mean, std, P(score>0.5), skew) -> Mann-Whitney U
#       across slides (49 amp vs 140 non-amp). Unit = slide, no tile pseudo-replication.
#   (2) Patient-level version of (1) (majority MYCN label per patient).
# The distribution-SHAPE p-value (slide-label permutation KS) is NOT recomputed here:
# `ks_all_clusters.py` is the single canonical source for every KS D / perm_p, so this
# script READS the C3/C5 rows from its `ks_all_clusters.csv` (run that first). Keeping
# the KS in one place stops the Supp table and the Section 2.2 text from drifting.
#
# License: GPL-3.0.
import os
import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = os.environ.get("PHENO_MYCN_ROOT", "/media/digitalpathology/6BE7-FCF8/phyno_mycn")
BASE = os.path.join(ROOT, "olga_refactered")
PT   = os.path.join(BASE, "results/slide_inference/fold_9/pt_outputs")
CV   = os.path.join(BASE, "data/cv_splits/neuroblastoma/fold9.csv")
OUT  = os.path.join(os.path.dirname(__file__), "results")
KS   = os.path.join(OUT, "ks_all_clusters.csv")   # canonical KS source (ks_all_clusters.py)
os.makedirs(OUT, exist_ok=True)

C3, C5 = 2, 4            # python idx -> manuscript cluster 3 / 5

# ---- labels (combine all splits = "all") ----
df = pd.read_csv(CV)
slide_label = {}
for _, r in df.iterrows():
    slide_label[r["train"]] = int(r["train_label"])
    if pd.notna(r["val"]):  slide_label[r["val"]]  = int(r["val_label"])
    if pd.notna(r["test"]): slide_label[r["test"]] = int(r["test_label"])

rows = []
for slide, lab in slide_label.items():
    f = os.path.join(PT, slide + "_gmm.pt")
    if not os.path.exists(f):
        continue
    resp = torch.load(f, map_location="cpu", weights_only=False)[0].numpy()
    c3, c5 = resp[:, C3], resp[:, C5]
    rows.append(dict(
        slide=slide, patient=slide.split(".")[0], label=lab, n=len(c3),
        c3_mean=c3.mean(), c3_std=c3.std(), c3_hi=(c3 > 0.5).mean(), c3_skew=stats.skew(c3),
        c5_mean=c5.mean(), c5_std=c5.std(), c5_hi=(c5 > 0.5).mean(), c5_skew=stats.skew(c5),
    ))

sdf = pd.DataFrame(rows)
print(f"slides={len(sdf)} amp={(sdf.label==1).sum()} nonamp={(sdf.label==0).sum()} "
      f"total tiles={sdf.n.sum():,}")

def mwu(sub, col):
    a = sub.loc[sub.label == 1, col]; b = sub.loc[sub.label == 0, col]
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return a.mean(), b.mean(), p

# ---- (1) slide-level ----
out = []
for cl in (3, 5):
    for stat in ("mean", "std", "hi", "skew"):
        col = f"c{cl}_{stat}"
        amp, non, p = mwu(sdf, col)
        out.append(dict(level="slide", cluster=cl, stat=stat,
                        amp=round(amp, 4), nonamp=round(non, 4), p=round(p, 4)))

# ---- (2) patient-level (majority label) ----
pat = sdf.groupby("patient").agg(
    label=("label", lambda s: int(round(s.mean()))),
    **{c: (c, "mean") for c in sdf.columns if c.startswith(("c3_", "c5_"))}
).reset_index()
print(f"patients={len(pat)} amp={(pat.label==1).sum()} nonamp={(pat.label==0).sum()}")
for cl in (3, 5):
    for stat in ("mean", "std", "hi", "skew"):
        col = f"c{cl}_{stat}"
        amp, non, p = mwu(pat, col)
        out.append(dict(level="patient", cluster=cl, stat=stat,
                        amp=round(amp, 4), nonamp=round(non, 4), p=round(p, 4)))

# ---- (3) distribution-SHAPE p-value: read from the canonical KS table ----
# ks_all_clusters.py is the ONE script that computes the slide-label permutation KS
# (all six clusters); we just pull its C3/C5 rows so there is a single source of truth.
if os.path.exists(KS):
    ks = pd.read_csv(KS).set_index("cluster")
    for cl in (3, 5):
        out.append(dict(level="perm-KS(canonical: ks_all_clusters.py)", cluster=cl,
                        stat="KS_D", amp=round(float(ks.loc[cl, "KS_D"]), 4),
                        nonamp=np.nan, p=round(float(ks.loc[cl, "perm_p"]), 4)))
        print(f"cluster {cl}: KS D={ks.loc[cl, 'KS_D']:.4f}  perm p={ks.loc[cl, 'perm_p']:.4f}  (from ks_all_clusters.csv)")
else:
    print(f"WARNING: {KS} not found — run ks_all_clusters.py first for the C3/C5 shape p-values")

res = pd.DataFrame(out)
res.to_csv(os.path.join(OUT, "c3_c5_distribution_tests.csv"), index=False)
print(res.to_string(index=False))
