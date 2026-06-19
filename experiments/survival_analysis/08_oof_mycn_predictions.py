# Pheno-MYCN — survival analysis.
# 08 — Out-of-fold (OOF) per-slide MYCN predictions.
#
# For each of the 10 cross-validation folds, load that fold's trained Pheno-MYCN
# checkpoint and run forward inference on the fold's TEST slides (their precomputed
# UNI tile embeddings), collating ONE OOF predicted P(MYCN-amp) per slide — each
# slide scored by the fold that held it out. This gives an honest (no-leakage)
# H&E-derived MYCN score per slide, used by 09 to ask whether the morphology-based
# MYCN call recovers MYCN's prognostic stratification.
#
# Pure inference (no training); CPU-only forward pass on precomputed embeddings.
# Validated by matching each fold's saved test AUC (fold*/result.csv).
#
# Output: results/oof_mycn_predictions.csv  (slide, fold, true_label, mycn_prob, pred_label)
# License: GPL-3.0.
import os
import sys
import glob

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("PHENO_MYCN_ROOT", "/media/digitalpathology/6BE7-FCF8/phyno_mycn")
REPO = os.path.join(ROOT, "github_repo")            # the pheno_mycn package lives here
sys.path.insert(0, REPO)
from pheno_mycn.inference import PhenoMYCNPredictor   # noqa: E402

BASE     = os.path.join(ROOT, "olga_refactered")
WEIGHTS  = os.path.join(REPO, "plug_and_play", "weights")   # per-fold k6 checkpoints (gitignored)
CKPT_DIR = os.path.join(BASE, "intermediate_outputs/training_checkpoints/pheno_mycn_gmm_k6")  # for result.csv validation
CV_DIR   = os.path.join(BASE, "data/cv_splits/neuroblastoma")
FEAT_DIR = os.path.join(BASE, "data/wsi_embeddings/uni_feats/pt_files")
OUT      = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

DEVICE  = "cpu"
N_FOLDS = int(os.environ.get("PHENO_OOF_NFOLDS", "10"))   # smoke-test override

rows, fold_auc = [], {}
for k in range(N_FOLDS):
    cv = pd.read_csv(os.path.join(CV_DIR, f"fold{k}.csv"))
    test = cv[["test", "test_label"]].dropna(subset=["test"])
    ckpt = os.path.join(WEIGHTS, f"pheno_mycn_k6_fold{k}.ckpt")
    assert os.path.exists(ckpt), f"fold{k}: checkpoint not found at {ckpt}"
    pred = PhenoMYCNPredictor.from_pretrained(ckpt_path=ckpt, device=DEVICE)

    probs, labels = [], []
    for _, r in test.iterrows():
        slide = r["test"]
        lab = int(r["test_label"])
        fpath = os.path.join(FEAT_DIR, slide + ".pt")
        if not os.path.exists(fpath):
            print(f"  [fold{k}] MISSING embedding: {slide}")
            continue
        feats = torch.load(fpath, map_location="cpu", weights_only=False)
        out = pred.predict(feats)
        rows.append(dict(slide=slide, fold=k, true_label=lab,
                         mycn_prob=out["mycn_probability"], pred_label=out["predicted_label"]))
        probs.append(out["mycn_probability"]); labels.append(lab)
    if len(set(labels)) > 1:
        fold_auc[k] = roc_auc_score(labels, probs)
    print(f"fold{k}: n_test={len(probs)}  AUC={fold_auc.get(k, float('nan')):.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "oof_mycn_predictions.csv"), index=False)
print(f"\nSaved {len(df)} OOF predictions ({df.slide.nunique()} unique slides) -> results/oof_mycn_predictions.csv")
if len(df):
    print(f"overall OOF AUC = {roc_auc_score(df.true_label, df.mycn_prob):.4f}  "
          f"(accuracy = {(df.true_label == df.pred_label).mean():.4f})")

# ---- validation: my per-fold AUC vs the saved result.csv test AUC ----
print("\n=== validation: recomputed per-fold test AUC vs saved result.csv ===")
for k in range(N_FOLDS):
    rc = os.path.join(CKPT_DIR, f"fold{k}", "result.csv")
    saved = float(pd.read_csv(rc).iloc[0]["auc"]) if os.path.exists(rc) else float("nan")
    print(f"fold{k}: recomputed={fold_auc.get(k, float('nan')):.4f}  saved={saved:.4f}")
