from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, List, Optional

import numpy as np
from h5py import File
from einops import rearrange
from joblib import Memory, delayed
from sklearn import preprocessing
from torch.utils.data import Dataset

from sleeptransformer.utils.h5_utils import load_waveforms
from sleeptransformer.utils.logger import get_logger
from sleeptransformer.utils.parallel_bar import ParallelExecutor

logger = get_logger()
SCALERS = {"robust": preprocessing.RobustScaler, "standard": preprocessing.StandardScaler}


@dataclass
class SleepStageDataset(Dataset):
    """

    Args:
        records (List[pathlib.Path])                : List of Path objects to .h5 files.
        cache_data (bool)                           : Whether to cache data for fast loading (default True)
        data_type (str)                             : Type of input data (wav, stft)
        fs (int)                                    : Sampling frequency, Hz.
        n_jobs (int)                                : Number of workers to spin out for data loading. -1 means all workers (default -1).
        n_records (int)                             : Total number of records to include (default None).
        picks (List[str])                           : List of channel names to include
        scaling (str)                               : Type of scaling to use ('robust', None, default 'robust').
        transform (Callable)                        : A Callable object to transform signal data by STFT, Morlet transforms or multitaper spectrograms.
                                                    : See the transforms/ directory for inspiration.
        sequence_length (int)                       : Number of 30 s epochs to include in sequence.

    """

    records: List[Path]
    sequence_length: int
    data_type: Literal["wav", "stft"]
    cache_data: bool = False
    fs: int = 128
    n_jobs: int = 1
    n_records: int = None
    picks: List[str] = None
    transform: Callable = None
    scaling: str = "robust"

    def __post_init__(self):
        self.cache_dir = Path("") / "data" / ".cache"
        self.n_records = len(self.records)
        self.n_channels = len(self.picks)
        if self.cache_data:
            logger.info(f"Using cache for data prep: {self.cache_dir.resolve()}")
            memory = Memory(self.cache_dir.resolve(), mmap_mode="r", verbose=0)
            get_metadata = memory.cache(get_record_metadata)
        else:
            get_metadata = get_record_metadata
        self.record_metadata = {}
        self.index_to_record = []
        self.scalers = []

        if self.records is not None:
            logger.info(f"Prefetching study metadata using {self.n_jobs} worker(s) ...")
            try:
                sorted_data = ParallelExecutor(n_jobs=self.n_jobs, prefer="threads")(total=len(self.records))(
                    delayed(get_metadata)(
                        filename=record,
                        data_type=self.data_type,
                        sequence_length=self.sequence_length,
                        scaling=self.scaling,
                    )
                    for record in set(self.records)
                )
            except Exception as e:
                logger.error(f"Prefetching failed with error: {e}")
                raise e
            logger.info("Prefetching finished")
        else:
            raise ValueError(f"Please specify a data directory, received: {self.records}")

        self.index_to_record = [sub for s in sorted_data for sub in s["index_to_record"][1]]
        self.metadata = dict([s["metadata"] for s in sorted_data])
        self.stages = dict([s["stages"] for s in sorted_data])
        self.scaler = dict([s["scaler"] for s in sorted_data])
        self.label_key = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    def __len__(self):
        return len(self.index_to_record)

    def __getitem__(self, idx):
        record = self.index_to_record[idx]["record"]
        window_index = self.index_to_record[idx]["window_idx"]
        # window_start =   # self.index_to_record[idx]["window_start"]

        # Load specific channels and location
        signal = load_waveforms(
            self.metadata[record]["filename"],
            picks=self.picks,
            window=window_index,
            scaled=False,
        )
        if self.data_type == "wav":
            N, C, T = signal.shape
        elif self.data_type == "stft":
            N, C, F, T = signal.shape

        # Potentially scale data
        if self.scaler[record]:
            if self.data_type == "wav":
                signal = rearrange(
                    self.scaler[record].transform(rearrange(signal, "N C T -> (T N) C")),
                    "(T N) C -> N C T",
                    N=N,
                    C=C,
                    T=T,
                )
            elif self.data_type == "stft":
                assert C == 1, "Scaling of STFT currently only works with 1 channel, received C={C}"
                signal = rearrange(
                    self.scaler[record]().fit_transform(rearrange(signal, "N C F T -> (C F T) N")),
                    "(C F T) N -> N C F T",
                    N=N,
                    C=C,
                    F=F,
                    T=T,
                )

        # Get valid stages
        stages = self.stages[record][window_index]

        # Optionally transform the signal
        if self.transform is not None:
            signal = self.transform(signal)

        if C == 1:
            signal = signal.squeeze(1)

        return {
            "signal": signal,
            "stages": np.array(stages),
            "record": f"{record}_{window_index.start:04d}-{window_index.stop-1:04d}",
        }


def get_record_metadata(filename: str, data_type: str, sequence_length: int, scaling: Optional[str] = None):
    # Get signal metadata
    with File(filename, "r") as h5:
        X = h5["data"]["unscaled"]

        # Get the waveforms and shape info
        if data_type == "wav":
            N, C, T = X.shape
        elif data_type == "stft":
            N, C, F, T = X.shape
        stages = h5["stages"][:]

        # Possibly get scaling object. If data_type == "wav", then we scale the data along the time axis for the entire recording.
        # If data_type == "stft", then we scale each STFT independently.
        if scaling:
            if data_type == "wav":
                scaler = SCALERS[scaling]().fit(rearrange(X[:], "N C T -> (T N) C", N=N, C=C, T=T))
            elif data_type == "stft":
                assert C == 1, "Scaling of STFT currently only works with 1 channel, received C={C}"
                scaler = SCALERS[scaling]
        else:
            scaler = None

        # Set metadata
        index_to_record = [
            {"record": filename.stem, "window_idx": slice(x, x + sequence_length)}
            for x in range(N - sequence_length + 1)
        ]
        metadata = {"n_channels": C, "length": T, "n_frequencies": F, "filename": filename}

    return dict(
        index_to_record=(filename.stem, index_to_record),
        metadata=(filename.stem, metadata),
        stages=(filename.stem, stages),
        scaler=(filename.stem, scaler),
    )
