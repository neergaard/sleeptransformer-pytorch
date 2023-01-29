from typing import List, Optional

import torch
from einops import rearrange, reduce

from sleeptransformer.loss.base_loss import BaseLoss


class SequenceLoss(BaseLoss):
    def __init__(self, weight: Optional[List[float]] = None):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight) if weight else None, reduction="none")

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        y_pred = rearrange(y_pred, "N L K -> N K L")
        loss = self.loss_fn(y_pred, y_target)
        loss = reduce(loss, "N L -> N", reduction="mean")
        return loss
