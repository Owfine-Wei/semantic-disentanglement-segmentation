import torch.nn as nn


class AuxModelWrapper(nn.Module):
    def __init__(self, model, is_enabled = False):
        super().__init__()
        self.model = model
        self.is_enabled = is_enabled
    
    def forward(self, x):

        feat = self.model.extract_feat(x)

        out_main = self.model.decode_head.forward(feat)

        if self.is_enabled:
            out_aux = self.model.auxiliary_head.forward(feat)
        else:
            out_aux = None

        return out_main , out_aux
        
    



