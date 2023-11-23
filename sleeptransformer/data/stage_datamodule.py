from pathlib import Path
from typing import Callable, List, Literal, Optional

import torch
from torch.utils.data import DataLoader

from sleeptransformer.data.base_datamodule import BaseDataModule
from sleeptransformer.data.stage_dataset import SleepStageDataset
from sleeptransformer.utils.collate_fn import collate
from sleeptransformer.utils.partitioning import get_train_validation_test
from sleeptransformer.utils.logger import get_logger


logger = get_logger()


class SleepStageDataModule(BaseDataModule):
    """SleepStageDataModule containing logic to contain and split a dataset.
    It also contains methods to return PyTorch DataLoaders for each split.

    Args:
        data_dir (str)                              : Directory to .h5 data files
        data_type (str)                             : Type of input data (wav, stft)
        batch_size (int)                            : Number of data windows to include in a batch.
        cache_data (bool)                           : Whether to cache data for fast loading (default True)
        fs (int)                                    : Sampling frequency, Hz.
        n_eval (int)                                : Number of validation subjects to include
        n_jobs (int)                                : Number of workers to spin out for data loading. -1 means all workers (default -1).
        n_test (int)                                : Number of test subjects to include
        n_records (int)                             : Total number of records to include (default None).
        num_workers (int)                           : Number of workers to use for dataloaders.
        percent_eval (float)                        : Percentage of data to use for validation (default None).
        percent_test (float)                        : Percentage of data to use for testing (default None).
        picks (List[str])                           : List of channel names to include
        scaling (str)                               : Type of scaling to use ('robust', None, default 'robust').
        seed (int)                                  : Random seed
        sequence_length (int)                       : Number of 30 s epochs to include in sequence.

    """

    def __init__(
        self,
        data_dir: str,
        data_type: Literal["wav", "stft"],
        n_test: int = None,
        n_eval: int = None,
        overfit: bool = False,
        percent_eval: float = None,
        percent_test: float = None,
        seed: int = 1337,
        cache_data: bool = False,
        fs: int = None,
        n_jobs: int = None,
        n_records: int = None,
        picks: Optional[List[str]] = None,
        scaling: str = None,
        sequence_length: int = None,
        transform: Callable = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(self.hparams.data_dir)
        partitions = get_train_validation_test(
            self.data_dir,
            n_records=self.hparams.n_records,
            number_test=self.hparams.n_test,
            number_validation=self.hparams.n_eval,
            percent_test=self.hparams.percent_test,
            percent_validation=self.hparams.percent_eval,
            seed=self.hparams.seed,
        )
        self.train_records = partitions["train"]
        self.eval_records = partitions["eval"] if not self.hparams.overfit else partitions["train"]
        self.test_records = partitions["test"]
        self.n_channels = len(self.hparams.picks)
        input_shape = (
            (self.n_channels, self.hparams.sequence_length * 30 * self.hparams.fs)
            if self.hparams.data_type == "wav"
            else (self.n_channels, self.hparams.sequence_length, 128, 28)
        )  # TODO: this shouldn't be hardcoded.
        self.example_input_array = torch.randn(self.hparams.batch_size, *input_shape)
        logger.info(f"Example input array shape: {self.example_input_array.shape}")

        self.dataset_kwargs = dict(
            cache_data=self.hparams.cache_data,
            data_type=self.hparams.data_type,
            fs=self.hparams.fs,
            n_jobs=self.hparams.n_jobs,
            n_records=self.hparams.n_records,
            picks=self.hparams.picks,
            scaling=self.hparams.scaling,
            sequence_length=self.hparams.sequence_length,
            transform=self.hparams.transform,
        )

    def setup(self, stage: Optional[str] = "fit") -> None:
        if stage == "fit":
            self.train = SleepStageDataset(self.train_records, **self.dataset_kwargs)
            self.eval = SleepStageDataset(self.eval_records, **self.dataset_kwargs)
        else:
            self.test = SleepStageDataset(self.test_records, **self.dataset_kwargs)
        self.output_dims = self.example_input_array.numpy().shape

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.eval,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )


if __name__ == "__main__":
    dm = SleepStageDataModule("data/mros/processed")
    print(repr(dm))
    print(dm)
