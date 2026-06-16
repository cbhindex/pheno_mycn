# Pheno-MYCN — additional experiments.
# Slide-label permutation KS test of the per-tile phenotype-score distribution
# (MYCN-amp vs non-amp) for ALL SIX phenotype clusters, to add a distribution-shape
# column to Supplementary Table tab:gmm_responsibility.
#
# Unit = slide (justified: MYCN status is a sample-level property; 6/86 patients
# have both amp and non-amp slides, plus primary vs relapse are distinct samples).
# Per-tile scores are pooled by slide; the null shuffles whole-slide labels, so the
# nested tile-in-slide structure is respected.  Tiles capped per slide for speed.
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
CACHE = os.path.join(OUT, "all_tiles_capped_cache.npz")
os.makedirs(OUT, exist_ok=True)

CAP, NPERM = 1500, 2000
rng = np.random.default_rng(0)

df = pd.read_csv(CV)
slide_label = {}
for _, r in df.iterrows():
    slide_label[r["train"]] = int(r["train_label"])
    if pd.notna(r["val"]):  slide_label[r["val"]]  = int(r["val_label"])
    if pd.notna(r["test"]): slide_label[r["test"]] = int(r["test_label"])

if os.path.exists(CACHE):
    z = np.load(CACHE)
    data, lengths, labels = z["data"], z["lengths"], z["labels"]
    print(f"loaded cache: {data.shape[0]:,} capped tiles, {len(labels)} slides")
else:
    arrs, lengths, labels = [], [], []
    for slide, lab in slide_label.items():
        f = os.path.join(PT, slide + "_gmm.pt")
        if not os.path.exists(f):
            continue
        resp = torch.load(f, map_location="cpu", weights_only=False)[0].numpy().astype(np.float32)
        if len(resp) > CAP:
            resp = resp[rng.choice(len(resp), CAP, replace=False)]
        arrs.append(resp); lengths.append(resp.shape[0]); labels.append(lab)
    data = np.concatenate(arrs, axis=0)
    lengths = np.array(lengths); labels = np.array(labels)
    np.savez_compressed(CACHE, data=data, lengths=lengths, labels=labels)
    print(f"built cache: {data.shape[0]:,} capped tiles, {len(labels)} slides")

# per-slide views (split the concatenated array back by slide)
bounds = np.concatenate([[0], np.cumsum(lengths)])
def slide_col(c):
    return [data[bounds[i]:bounds[i+1], c] for i in range(len(lengths))]

def pooled_ks(cols, lab):
    amp = np.concatenate([a for a, l in zip(cols, lab) if l == 1])
    non = np.concatenate([a for a, l in zip(cols, lab) if l == 0])
    return stats.ks_2samp(amp, non).statistic

rows = []
for c in range(6):
    cols = slide_col(c)
    obs = pooled_ks(cols, labels)
    null = np.array([pooled_ks(cols, rng.permutation(labels)) for _ in range(NPERM)])
    p = (np.sum(null >= obs) + 1) / (NPERM + 1)
    rows.append(dict(cluster=c + 1, KS_D=round(float(obs), 4), perm_p=round(float(p), 4)))
    print(f"cluster {c+1}: KS D={obs:.4f}  perm p={p:.4f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "ks_all_clusters.csv"), index=False)
print("saved ks_all_clusters.csv")
