"""
Backwards-compatible launcher for the Pheno-MYCN plug-and-play predictor.

The canonical command-line interface is now `pheno-mycn` (installed with the
package; see `pheno_mycn/cli.py`). This script remains so that

    python plug_and_play/predict.py --features SLIDE_uni.pt [--checkpoint ...] [--device ...] [--output ...]

keeps working; it simply forwards to `pheno-mycn predict`.

Author: Dr Olga Fourkioti. Refactoring: Dr Binghao Chai. License: GPL-3.0.
"""

import sys

from pheno_mycn.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["predict"] + sys.argv[1:]))
