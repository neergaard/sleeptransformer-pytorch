from functools import partial
from typing import List, Optional

import numpy as np

from librosa import stft
import librosa


class STFTTransform:
    """
    STFTTransform: Short-time Fourier Transform

    Attributes:
        fs (int): sampling frequency.
        segment_size (int): window length in samples.
        step_size (int): step size between successive windows in samples.
        nfft (int): number of points for FFT.
        transform_fn (Callable): partial function containing the STFT operation.
    """

    def __init__(self, fs: int, segment_size: int, step_size: int, nfft: int, *args, **kwargs) -> None:
        """
        Args:
            fs (int): sampling frequency.
            segment_size (int): window length in samples.
            step_size (int): step size between successive windows in samples.
            nfft (int): number of points for FFT.
        """
        self.fs = fs
        self.segment_size = segment_size
        self.step_size = step_size
        self.nfft = nfft
        self.transform_fn = partial(stft, n_fft=self.nfft, hop_length=self.step_size, win_length=self.segment_size, window='hamming', center=False)

    def calculate_output_dims(self, window_size: int) -> List[int]:
        """calculates output dimensions after STFT operation.
        The STFT windows and optionally overlaps data, which causes reductions in the time vectors.

        Args:
            window_size (int): Size of windows in samples.

        Returns:
            List[int]: List containing the frequency and time dimensions.
        """
        T = librosa.samples_to_frames(window_size, hop_length=self.step_size) + 1
        F = self.nfft // 2 + 1
        return [F, T]

    def __call__(self, X: np.ndarray, annotations: Optional[np.ndarray] = None) -> np.ndarray:
        Zxx = np.abs(self.transform_fn(X)) ** 2
        Zxx = librosa.power_to_db(Zxx)

        return Zxx[:, :, 1:]  # Remove DC component
