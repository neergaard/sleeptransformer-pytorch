import torch
from einops import rearrange
from pytorch_lightning import LightningModule

from .base_transformer import BaseTransformer


class SleepTransformer(LightningModule):
    def __init__(self, epoch_transformer: BaseTransformer, sequence_transformer: BaseTransformer) -> None:
        super().__init__()
        self.epoch_transformer = epoch_transformer
        self.sequence_transformer = sequence_transformer

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        N, L, T, F = X.shape
        X = rearrange(X, "N L T F -> L N T F")

        # Pass input through Epoch Transformer
        output = []
        for x in X:
            output.append(self.epoch_transformer(x))
        z, alpha = map(torch.stack, zip(*output))
        z = rearrange(z, "L N F -> N L F")
        alpha = rearrange(alpha, "L N T -> N L T")

        # Pass output vectors through Sequence Transformer
        y = self.sequence_transformer(z)

        return y, alpha
