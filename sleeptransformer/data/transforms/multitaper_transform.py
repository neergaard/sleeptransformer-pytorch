from functools import partial
from typing import Optional

import numpy as np
from mne.time_frequency import tfr_array_multitaper
from librosa import power_to_db


class MultitaperTransform:
    """
    MultitaperTransform:
    """

    def __init__(self, fs: int, fmin: int, fmax: int, tw: float, normalize: bool = True, *args, **kwargs) -> None:
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.tw = tw
        self.normalize = normalize
        # self.nfft = nfft
        # self.freqs = np.linspace(self.fmin, self.fmax, endpoint=False)
        self.freqs = np.arange(self.fmin, self.fmax + 0.5, 0.5)
        self.transform_fn = partial(
            tfr_array_multitaper,
            sfreq=self.fs,
            freqs=self.freqs,
            n_cycles=self.freqs / 2.0,
            output="power",
            # zero_mean=True,
        )

    def __call__(self, X: np.ndarray, annotations: np.ndarray) -> np.ndarray:
        C, T = X.shape
        Cxx = self.transform_fn(X[None, :]).squeeze()[0]
        Cxx = power_to_db(Cxx)
        if self.normalize:
            Cxx = (Cxx + 20.0) / 50.0  # Normalize so that output is relatively in range [-1, 1]

        # show a plot
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(X[0])
        for start, stop in annotations[:, :-1]:
            axs[0].hlines(y=6, xmin=start * T, xmax=stop * T)
        vmin, vmax = None, None
        axs[1].imshow(Cxx, aspect="auto", vmin=vmin, vmax=vmax)  # , vmin=vmin, vmax=vmax)
        axs[1].set(ylim=(self.freqs[0], self.freqs[-1]))

        return Cxx
