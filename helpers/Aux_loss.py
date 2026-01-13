"""
Auxiliary loss helper.

This module provides a small wrapper that runs feature extraction on a
segmentation model and returns the main decode output plus an optional
auxiliary head output (if enabled). 
"""

import torch.nn as nn


class AuxModelWrapper(nn.Module):

    """
    The wrapped `model` is expected to expose `extract_feat`, `decode_head`,
    and `auxiliary_head` attributes, which should be checked.
    """

    def __init__(self, model, is_enabled=False):
        super().__init__()
        # underlying segmentation model
        self.model = model
        # flag to enable/disable auxiliary head computation
        self.is_enabled = is_enabled

    def forward(self, x):
        """
        Args:
            x: input tensor (images or batch) passed to the model.
            
        Returns:
            tuple: (out_main, out_aux) where `out_aux` is `None` when
            auxiliary computation is disabled.
        """

        # extract intermediate features from the backbone
        feat = self.model.extract_feat(x)

        # main decode head prediction
        out_main = self.model.decode_head.forward(feat)

        # optional auxiliary head prediction
        if self.is_enabled:
            out_aux = self.model.auxiliary_head.forward(feat)
        else:
            out_aux = None

        return out_main, out_aux
        
    



