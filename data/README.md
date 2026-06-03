# Data

**No patient data is distributed in this repository.** The neuroblastoma cohort
(whole-slide images, tile embeddings, cross-validation splits with real slide
identifiers, and the clinical/survival tables) is patient-derived and is not
shared here for data-governance reasons. This page documents the **file formats**
the code expects, and `example/` provides a tiny **synthetic** sample so you can
see the layout. Point the configs / `PHENO_MYCN_ROOT` at your own data.

## 1. Tile embeddings (model input)

One file per slide: `<slide_id>.pt` — a PyTorch tensor of shape
`[n_tiles, 1024]`, the **UNI** tile embeddings for that slide. Produced upstream
by tiling a colour-normalised H&E WSI and encoding each tile with the
[UNI encoder](https://github.com/mahmoodlab/UNI).

```python
import torch
feats = torch.load("SLIDE_0001.pt")   # torch.FloatTensor, shape [n_tiles, 1024]
```

`data_dir` in the YAML config points at the directory holding these `.pt` files.

## 2. Cross-validation splits

One CSV per fold: `fold{N}.csv` (N = 0…9). Read with `pandas.read_csv(..., index_col=0)`.
Columns:

| column | meaning |
|--------|---------|
| `train`       | slide id of a training slide |
| `train_label` | its MYCN label (0 = non-amplified, 1 = MYCN-amplified) |
| `val`         | slide id of a validation slide |
| `val_label`   | its MYCN label |
| `test`        | slide id of a test slide |
| `test_label`  | its MYCN label |

The four columns `val/val_label/test/test_label` are only populated for as many
rows as there are val/test slides; the rest are blank (`NaN`) and are dropped per
column at load time (see `pheno_mycn/datasets/camel_data.py`). `label_dir` in the
config points at the directory holding these CSVs. A **synthetic** example is in
[`example/cv_splits/fold0.csv`](example/cv_splits/fold0.csv).

## 3. Survival cohort table (additional experiments only)

The survival experiments expect a slide-level table
`survival_per_slide.csv` built by
`experiments/survival_analysis/00_build_cohort.py` from the clinical registry.
Schema (one row per slide):

```
slide_name, patient_id, mycn_perslide, timepoint, split,
OS_time_days, event, date_of_death,
date_of_biopsy_resection_or_surgery_from_which_ffpe_taken,
current_patient_status, patient_age_at_biopsy_months, gender,
detailed_diagnosis, disease_category
```

This table contains clinical fields and is **not** distributed. Regenerate it
from your own registry with the documented build script.

## 4. Per-tile model outputs (visualization / experiments input)

The visualization and some experiment scripts consume per-tile tensors exported
from the test loop (one set per slide): `<slide>_gmm.pt` (responsibilities,
`[n_tiles, K]`), `<slide>_gmm_feats.pt` (projected features), `<slide>_att.pt`
(attention), `<slide>_energy.pt` (free-energy). See the note in
`pheno_mycn/models/model_interface.py` for where to save these.
