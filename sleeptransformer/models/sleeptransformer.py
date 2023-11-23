import math
from typing import List, Tuple

import torch
import torchmetrics
from einops import rearrange, reduce
from pytorch_lightning.cli import instantiate_class

from sleeptransformer.models.base_model import BaseModel
from sleeptransformer.loss.base_loss import BaseLoss
from sleeptransformer.models.base_transformer import BaseTransformer
from sleeptransformer.utils.logger import get_logger


logger = get_logger()


class SleepTransformer(BaseModel):
    def __init__(
        self,
        epoch_transformer: BaseTransformer,
        sequence_transformer: BaseTransformer,
        loss_fn: BaseLoss,
        optimizer: dict,
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.epoch_transformer = epoch_transformer
        self.sequence_transformer = sequence_transformer
        self.optimizer = {
            "class_path": optimizer["name"],
            "init_args": {k: v for k, v in optimizer.items() if k != "name"},
        }
        if isinstance(self.optimizer["init_args"]["weight_decay"], str):
            self.optimizer["init_args"]["weight_decay"] = float(self.optimizer["init_args"]["weight_decay"])

        hparams = dict(
            optimizer=self.optimizer["class_path"],
            **self.optimizer["init_args"],
            **{".".join(["epoch_transformer", k]): v for k, v in epoch_transformer.hparams.items()},
            **{".".join(["sequence_transformer", k]): v for k, v in sequence_transformer.hparams.items()},
        )
        self.save_hyperparameters(
            hparams, ignore=["optimizer", "epoch_transformer", "sequence_transformer", "loss_fn"]
        )
        # fmt: off
        n_classes = self.hparams["sequence_transformer.n_classes"]
        self.train_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        ], prefix='train_')
        self.eval_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        ], prefix='eval_')
        self.test_metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="multiclass", num_classes=n_classes),
            torchmetrics.CohenKappa(task="multiclass", num_classes=n_classes),
            torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average='macro'),
        ], prefix='test_')
        self.test_class_metrics = torchmetrics.MetricCollection([
            torchmetrics.ConfusionMatrix(task="multiclass", num_classes=n_classes),
            torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average=None),
            torchmetrics.Recall(task='multiclass', num_classes=n_classes, average=None),
            torchmetrics.Precision(task='multiclass', num_classes=n_classes, average=None)
        ])
        # fmt: on

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        N, L, F, T = X.shape
        X = rearrange(X, "N L F T -> T (N L) F")

        # Pass input through Epoch Transformer
        z, epoch_att_weights, alpha = self.epoch_transformer(X)
        z = rearrange(z, "(N L) F -> L N F", N=N, L=L, F=F)

        # Pass output vectors through Sequence Transformer
        y, seq_att_weights = self.sequence_transformer(z)

        # Everything reshaped to batch-first
        # _, _, K = y.shape
        y = rearrange(y, "L N K -> N L K")
        seq_att_weights = reduce(seq_att_weights, "Ns N L1 L2 -> N L1 L2", "mean")
        epoch_att_weights = reduce(epoch_att_weights, "Ne (N L) T1 T2 -> N L T1 T2", "mean", N=N, L=L)
        alpha = rearrange(alpha, "(N L) T -> N L T", N=N, L=L)

        # Compute confidence
        confidence = self.compute_confidence(y)

        return y, epoch_att_weights, seq_att_weights, alpha, confidence

    def compute_loss(self, logits: torch.Tensor, y_target: torch.Tensor) -> torch.tensor:
        logits = rearrange(logits, "N L K -> N K L")
        loss = self.loss_fn(logits, y_target)
        return loss.mean()

    def compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        y_pred = logits.softmax(dim=-1)
        h_y = -y_pred * y_pred.log() / math.log(self.hparams["sequence_transformer.n_classes"])
        return 1 - h_y.sum(dim=-1)

    def shared_step(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z, epoch_att_weights, seq_att_weights, alpha, confidence = self(x)
        loss = self.compute_loss(z, t)
        return loss, z, t

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        N, *_ = batch["data"].shape
        loss, z, t = self.shared_step(batch["data"], batch["targets"])
        log_output = self.train_metrics(z.argmax(-1), t)
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=N)
        self.log_dict(
            log_output,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=N,
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        N, *_ = batch["data"].shape
        loss, z, t = self.shared_step(batch["data"], batch["targets"])
        log_output = self.eval_metrics(z.argmax(-1), t)
        self.log("loss/eval", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=N)
        self.log_dict(
            log_output,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=N,
        )
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        N, *_ = batch["data"].shape
        loss, z, t = self.shared_step(batch["data"], batch["targets"])
        log_output = self.test_metrics(z.argmax(-1), t)
        self.test_class_metrics(z.argmax(-1), t)
        self.log_dict(log_output, on_step=False, on_epoch=True, batch_size=N)

    def on_test_epoch_end(self) -> None:
        cm = self.test_class_metrics["MulticlassConfusionMatrix"].compute().cpu().numpy()
        f1 = self.test_class_metrics["MulticlassF1Score"].compute().cpu().numpy()
        recall = self.test_class_metrics["MulticlassRecall"].compute().cpu().numpy()
        precision = self.test_class_metrics["MulticlassPrecision"].compute().cpu().numpy()
        # f1 = self.tes
        logger.info("Overall confusion matrix (row - true W, N1, N2, N3, REM; col - pred W, N1, N2, N3, REM)")
        logger.info(cm)
        f1_str = ", ".join([f"{s1}: {s2:.3f}" for s1, s2 in zip(["W", "N1", "N2", "N3", "R"], f1)])
        logger.info(f"Class-specific F1 score: {f1_str}")
        recall_str = ", ".join([f"{s1}: {s2:.3f}" for s1, s2 in zip(["W", "N1", "N2", "N3", "R"], recall)])
        logger.info(f"Class-specific recall score: {recall_str}")
        precision_str = ", ".join([f"{s1}: {s2:.3f}" for s1, s2 in zip(["W", "N1", "N2", "N3", "R"], precision)])
        logger.info(f"Class-specific precision score: {precision_str}")
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        logger.info(
            f"Configuring optimizer `{self.optimizer['class_path']}` with params: {self.optimizer['init_args']}"
        )
        optimizer = instantiate_class(filter(lambda p: p.requires_grad, self.parameters()), self.optimizer)
        return [optimizer]
