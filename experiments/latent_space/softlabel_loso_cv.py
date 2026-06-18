# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
softlabel_loso_cv.py — leave-one-slide-out cross-validation of the cell-level
soft-label classifier (the §2.4 / Figure 4 de-circularisation check).

The soft label in Fig 4 is the IN-SAMPLE predicted P(MYCN-amp) from a per-cluster
logistic regression (cell_latent_analysis.py). This script re-runs the same
classifier under leave-one-slide-out cross-validation (group = class+slide, so the
10 amplified and 10 non-amplified slides of each cluster give 20 disjoint groups;
the median imputer, standardiser and classifier are refit on the training slides of
each fold), and reports how well the held-out soft labels separate the subtypes —
to confirm the separation seen in Fig 4 is not an artefact of in-sample fitting.

Reported per cluster (and written to results/softlabel_loso_cv.csv):
  - in-sample per-class soft-label medians (reproduce Fig 4a) + OOF medians
  - per-slide out-of-fold AUC (20 slides; the headline value cited in §2.4)
  - per-tile out-of-fold AUC

Matches the preprocessing of cell_latent_analysis.py exactly.
"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("PHENO_MYCN_ROOT", "/media/digitalpathology/6BE7-FCF8/phyno_mycn")
IMG  = os.path.join(ROOT, "olga_refactered/results/slide_inference/fold_9/images")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

CL = [("C2 necrotic", "prototype_1"), ("C3 cellular tumour", "prototype_2"),
      ("C5 dense tumour", "prototype_4"), ("C6 haemorrhagic", "prototype_5")]
NAN_THRESH = 0.50


def load(proto):
    df = pd.read_csv(os.path.join(IMG, proto, "features", "cell_info_updated.csv"), index_col=0)
    parts = pd.Series(df.index).str.split("|", n=1, expand=True)
    klass = parts[0].values
    slide = parts[1].str.split("_x_", n=1).str[0].values
    y = (klass == "class_1").astype(int)
    feats = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    a, n = y == 1, y == 0
    feats = [c for c in feats
             if df.loc[a, c].isna().mean() <= NAN_THRESH
             and df.loc[n, c].isna().mean() <= NAN_THRESH
             and df[c].std() > 1e-9]
    X = df[feats].values.astype(float)
    grp = np.array([f"{k}|{s}" for k, s in zip(klass, slide)])   # class+slide => 20 groups
    return X, y, grp, len(feats)


def slide_auc(p, y, grp):
    d = pd.DataFrame({"p": p, "y": y, "g": grp}).groupby("g").agg(p=("p", "mean"), y=("y", "first"))
    return roc_auc_score(d["y"], d["p"])


rows = []
for lab, proto in CL:
    X, y, grp, nf = load(proto)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, class_weight="balanced",
                                            max_iter=1000, solver="lbfgs"))
    pipe.fit(X, y)
    p_in = pipe.predict_proba(X)[:, 1]
    p_oof = cross_val_predict(pipe, X, y, groups=grp, cv=LeaveOneGroupOut(),
                              method="predict_proba")[:, 1]
    rows.append(dict(
        cluster=lab, n_features=nf,
        in_median_nonamp=round(float(np.median(p_in[y == 0])), 3),
        in_median_amp=round(float(np.median(p_in[y == 1])), 3),
        oof_median_nonamp=round(float(np.median(p_oof[y == 0])), 3),
        oof_median_amp=round(float(np.median(p_oof[y == 1])), 3),
        in_slide_auc=round(float(slide_auc(p_in, y, grp)), 3),
        oof_slide_auc=round(float(slide_auc(p_oof, y, grp)), 3),
        oof_tile_auc=round(float(roc_auc_score(y, p_oof)), 3),
    ))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "softlabel_loso_cv.csv"), index=False)
pd.set_option("display.width", 160, "display.max_columns", 20)
print(res.to_string(index=False))
print("\nwrote results/softlabel_loso_cv.csv")
