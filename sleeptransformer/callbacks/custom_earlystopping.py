from typing import Any, Mapping
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch import Tensor


class CustomEarlyStopping(EarlyStopping):
    def __init__(
        self,
        monitor: str,
        begin_after: int,
        min_delta: float = 0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: float | None = None,
        divergence_threshold: float | None = None,
        check_on_train_epoch_end: bool | None = None,
        log_rank_zero_only: bool = False,
    ):
        super().__init__(
            monitor,
            min_delta,
            patience,
            verbose,
            mode,
            strict,
            check_finite,
            stopping_threshold,
            divergence_threshold,
            check_on_train_epoch_end,
            log_rank_zero_only,
        )
        self.begin_after = begin_after
        self.counter = 0
        self.run_early_stopping = False

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self.run_early_stopping and trainer.global_step > self.begin_after:
            self.run_early_stopping = True
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    # def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    # self.counter += 1
    # if self.counter > self.begin_after:
    #     return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.run_early_stopping:
            return super().on_validation_end(trainer, pl_module)
        # if trainer.sanity_checking:
        #     return
        # self.counter += 1
        # if self.counter > self.begin_after:
