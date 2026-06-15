# Pheno-MYCN — additional experiments.
# Author:  Dr Binghao Chai  (https://bhchai.com/, https://github.com/cbhindex)
# License: GPL-3.0 (see the LICENSE file at the repository root).
#
"""
Shared path configuration for the additional experiments.

Every experiment script resolves its inputs/outputs relative to a single
project root, configurable via the ``PHENO_MYCN_ROOT`` environment variable::

    export PHENO_MYCN_ROOT=/path/to/pheno_mycn

``PHENO_MYCN_ROOT`` should point at the working project tree that contains the
intermediate outputs the experiments consume (these are patient-derived and/or
too large to ship, so they are NOT included in this repository):

    $PHENO_MYCN_ROOT/
        olga_refactered/
            results/slide_inference/fold_9/pt_outputs/   # per-tile GMM/attention .pt
            results/cell_analysis/cell_info.csv          # HoverNet cell features
            data/cv_splits/neuroblastoma/foldN.csv       # cross-validation splits
        pheno_mycn_paper/Book 6(Sheet1).csv              # clinical registry

Experiment-local intermediates (cohort tables, per-slide GMM stats, cached
embeddings) now live inside the repo, alongside the scripts that build/consume
them, and are git-ignored:
        github_repo/experiments/survival_analysis/data/
        github_repo/experiments/gmm_responsibility/results/

If the variable is unset, the scripts fall back to the placeholder
``/path/to/pheno_mycn`` and will raise a clear file-not-found error until you
set it. See each script's module docstring for the exact inputs it expects.
"""

import os

#: Root of the working Pheno-MYCN project tree (override with PHENO_MYCN_ROOT).
PROJECT_ROOT = os.environ.get("PHENO_MYCN_ROOT", "/path/to/pheno_mycn")


def project_path(*parts):
    """Join ``parts`` onto the configured project root."""
    return os.path.join(PROJECT_ROOT, *parts)
