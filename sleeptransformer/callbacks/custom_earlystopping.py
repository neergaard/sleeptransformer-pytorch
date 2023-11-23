import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


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

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.counter += 1
        if self.counter > self.begin_after:
            return super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self.counter += 1
        if self.counter > self.begin_after:
            return super().on_validation_end(trainer, pl_module)
