import numpy as np
from scipy.signal import butter, sosfiltfilt


class ButterworthFilter(object):
    def __init__(self, order: int = 2, fc: float = 100.0, type: str = "lowpass"):
        self.fc = np.asarray(fc)
        self.order = order
        self.type = type
        self.sos = None
        self.Wn = None

    def __call__(self, x: np.ndarray, fs: int):
        self.Wn = 2 * self.fc / fs
        self.sos = butter((self.order), (self.Wn), btype=(self.type), output="sos")
        return sosfiltfilt(self.sos, x)
