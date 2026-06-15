# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
cohort_demographics.py — Table-1-style demographics for the study cohort.

Builds a patient-level and slide-level demographic summary from the survival
cohort table (which already merges Book 6 clinical fields onto the 189-slide MIL
cohort). Intended to support the Dataset description in the Abstract / Intro /
Methods (Comment 2 / EXP-0a).

NB: the formal cohort/trial name, INSS/INRG stage, institution names and year
range are deferred (EXP-0b) and are NOT produced here.

Source : experiments/survival_analysis/data/survival_per_slide.csv
Outputs (results/):
  cohort_demographics_patient.csv   — patient-level Table 1
  cohort_demographics_slide.csv     — slide-level counts
  cohort_demographics_summary.txt   — human-readable
"""

import os
import numpy as np
import pandas as pd

BASE = os.environ.get("PHENO_MYCN_ROOT", "/path/to/pheno_mycn")
SRC  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "survival_analysis", "data", "survival_per_slide.csv")
OUT  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(SRC)
print(f"Loaded {len(df)} slides, {df.patient_id.nunique()} patients from {SRC}")


def fmt_med(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return "n/a"
    return f"{x.median():.1f} (IQR {x.quantile(.25):.1f}–{x.quantile(.75):.1f}; range {x.min():.1f}–{x.max():.1f})"


# ── slide-level ────────────────────────────────────────────────────────────────
slide_rows = [
    ("Slides (total)",               len(df)),
    ("  MYCN-amplified",             int((df.mycn_perslide == 1).sum())),
    ("  MYCN non-amplified",         int((df.mycn_perslide == 0).sum())),
    ("  Primary diagnosis",          int((df.timepoint == "primary").sum())),
    ("  Current relapse",            int((df.timepoint == "relapse").sum())),
]
slide_tbl = pd.DataFrame(slide_rows, columns=["characteristic", "value"])
slide_tbl.to_csv(os.path.join(OUT, "cohort_demographics_slide.csv"), index=False)

# ── patient-level (dedup by patient_id) ─────────────────────────────────────────
pat = df.sort_values("timepoint").drop_duplicates("patient_id").copy()
pat["age_years"] = pd.to_numeric(pat.patient_age_at_biopsy_months, errors="coerce") / 12.0
# MYCN per patient: amplified if ANY slide amplified
mycn_pat = df.groupby("patient_id").mycn_perslide.max()
pat = pat.merge(mycn_pat.rename("mycn_any"), left_on="patient_id", right_index=True)

n_pat = len(pat)
gender_counts = pat.gender.fillna("unknown").value_counts().to_dict()
status_counts = pat.current_patient_status.fillna("unknown").value_counts().to_dict()
dx_counts     = pat.disease_category.fillna("unknown").value_counts().to_dict()

pat_rows = [
    ("Patients (total)",                          n_pat),
    ("Age at biopsy, years — median (IQR; range)", fmt_med(pat.age_years)),
    ("  Age < 18 months",                          int((pat.age_years < 1.5).sum())),
    ("  Age 18 months – 5 years",                  int(((pat.age_years >= 1.5) & (pat.age_years < 5)).sum())),
    ("  Age ≥ 5 years",                            int((pat.age_years >= 5).sum())),
    ("MYCN-amplified (any timepoint)",             int((pat.mycn_any == 1).sum())),
    ("MYCN non-amplified",                         int((pat.mycn_any == 0).sum())),
]
for k, v in gender_counts.items():
    pat_rows.append((f"Gender: {k}", v))
for k, v in dx_counts.items():
    pat_rows.append((f"Disease category: {k}", v))
for k, v in status_counts.items():
    pat_rows.append((f"Patient status: {k}", v))

pat_tbl = pd.DataFrame(pat_rows, columns=["characteristic", "value"])
pat_tbl.to_csv(os.path.join(OUT, "cohort_demographics_patient.csv"), index=False)

# ── summary ─────────────────────────────────────────────────────────────────────
lines = ["=" * 64, "COHORT DEMOGRAPHICS (EXP-0a)", "=" * 64,
         f"Source: {SRC}", "",
         "SLIDE-LEVEL", "-" * 40]
lines += [f"  {c:<42} {v}" for c, v in slide_rows]
lines += ["", "PATIENT-LEVEL (dedup by patient_id)", "-" * 40]
lines += [f"  {c:<42} {v}" for c, v in pat_rows]
lines += ["",
          "NOTE: cohort/trial name, INSS/INRG stage, institution and year range",
          "are deferred (EXP-0b). Demographics computed on slides with valid OS;",
          "the full MIL cohort is 189 slides / 86 patients (see cohort_flow.txt)."]
summary = "\n".join(lines)
print("\n" + summary)
with open(os.path.join(OUT, "cohort_demographics_summary.txt"), "w") as f:
    f.write(summary + "\n")
print(f"\nOutputs written to: {OUT}")
