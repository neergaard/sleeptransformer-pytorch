from typing import List, Optional

import torch

from sleeptransformer.loss.base_loss import BaseLoss


class SequenceLoss(BaseLoss):
    def __init__(self, weight: Optional[List[float]] = None):
        super().__init__()
        self.register_buffer("weight", torch.tensor(weight) if weight else None)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.weight, reduction="none")

    def forward(self, logits: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(logits, y_target)
        return loss
