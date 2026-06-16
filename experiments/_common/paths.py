# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""Shared path configuration for the additional experiments.

Every script resolves inputs/outputs relative to a project root set via the
``PHENO_MYCN_ROOT`` env var (``export PHENO_MYCN_ROOT=/path/to/pheno_mycn``). It
should point at the working tree holding the intermediate outputs the
experiments consume (patient-derived and/or too large to ship, so NOT in this
repo)::

    $PHENO_MYCN_ROOT/
        olga_refactered/
            results/slide_inference/fold_9/pt_outputs/   # per-tile GMM/attention .pt
            results/cell_analysis/cell_info.csv          # HoverNet cell features
            data/cv_splits/neuroblastoma/foldN.csv       # cross-validation splits
        pheno_mycn_paper/Book 6(Sheet1).csv              # clinical registry

Experiment-local intermediates live in git-ignored ``experiments/*/data/`` and
``experiments/*/results/``. If the var is unset, scripts fall back to
``/path/to/pheno_mycn`` and raise a clear file-not-found error.
"""

import os

#: Root of the working Pheno-MYCN project tree (override with PHENO_MYCN_ROOT).
PROJECT_ROOT = os.environ.get("PHENO_MYCN_ROOT", "/path/to/pheno_mycn")


def project_path(*parts):
    """Join ``parts`` onto the configured project root."""
    return os.path.join(PROJECT_ROOT, *parts)
