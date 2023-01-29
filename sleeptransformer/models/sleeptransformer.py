import torch
from einops import rearrange, reduce
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class

from sleeptransformer.loss.base_loss import BaseLoss
from sleeptransformer.models.base_transformer import BaseTransformer


class SleepTransformer(LightningModule):
    def __init__(
        self,
        epoch_transformer: BaseTransformer,
        sequence_transformer: BaseTransformer,
        loss_fn: BaseLoss,
        optimizer_params: dict,
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.epoch_transformer = epoch_transformer
        self.sequence_transformer = sequence_transformer
        self.optimizer_params = optimizer_params

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

    def compute_loss(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.tensor:
        y_pred = rearrange(y_pred, "N L K -> N K L")
        loss = self.loss_fn(y_pred, y_target)
        loss = reduce(loss, "N L -> N", reduction="mean")
        return loss.mean()  # or .sum()

    def configure_optimizers(self):
        optimizer = instantiate_class(filter(lambda p: p.requires_grad, self.parameters()), self.optimizer_params)
        return [optimizer]
