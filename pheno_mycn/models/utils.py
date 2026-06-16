"""
Shared model utilities for Pheno-MYCN.

Currently this provides weight initialisation used by the attention-MIL
backbone. Adapted from CLAM (Mahmood Lab, GPL-3.0).

Author: Dr Olga Fourkioti. Refactoring: Dr Binghao Chai. License: GPL-3.0.
"""

import torch.nn as nn


def initialize_weights(module):
    """Xavier-initialise ``nn.Linear`` weights and reset batch-norm parameters."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
