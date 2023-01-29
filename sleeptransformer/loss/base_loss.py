import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
