from functools import partial
from typing import Optional

import numpy as np
from mne.time_frequency import tfr_array_morlet


class MorletTransform:
    """
    MorletTransform:
    """

    def __init__(self, fs: int, fmin: int, fmax: int, nfft: int) -> None:
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.nfft = nfft
        # self.freqs = np.linspace(self.fmin, self.fmax, endpoint=False)
        self.freqs = np.arange(self.fmin, self.fmax + 0.5, 0.5)
        self.transform_fn = partial(
            tfr_array_morlet,
            sfreq=self.fs,
            freqs=self.freqs,
            n_cycles=self.freqs / 2.0,
            output="power",
            zero_mean=True,
        )

    # def __call__(
    #     self, X: np.ndarray, nperseg: Optional[int] = None, noverlap: Optional[int] = None, nfft: Optional[int] = None
    # ) -> np.ndarray:

    def __call__(self, X: np.ndarray, annotations: np.ndarray) -> np.ndarray:

        Cxx = self.transform_fn(X[None, :], output="power").squeeze()[0]
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(X[0])
        for start, stop in annotations[:, :-1]:
            axs[0].axhline(y=6, xmin=start, xmax=stop)
        vmin, vmax = -100, 10.0
        axs[1].imshow(10 * np.log(Cxx), aspect="auto")  # , vmin=vmin, vmax=vmax)

        return Cxx
