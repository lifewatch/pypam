__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

from pypam.signal import Signal
import numpy as np


class Event(Signal):
    def __init__(self, total_signal, fs, start=None, end=None):
        """
        Definition of an acoustic event

        Parameters
        ----------
        total_signal : 1D numpy array
            Signal contaning the event
        start :
        fs : int
            Sample rate, in Hz
        """
        signal = total_signal[start:end]
        self._total_signal = total_signal
        self.start = start
        self.end = end
        super().__init__(signal, fs)

    def cut(self, start=0, end=None):
        """
        Cut the signal
        :param start:
        :param end:
        :return:
        """
        if end is None:
            end = len(self.signal)
        self.start += start
        self.end += end
        self.signal = self.signal[start:end]

    def analyze(self):
        """
        Perform all necessary calculations for a single event

        Returns
        -------
        rms, sel, peak
        """
        rms = self.rms()
        sel = self.sel()
        peak = self.peak()
        return rms, sel, peak

    def sel(self, high_noise=False):
        """
        Compute the SEL by finding the peak of the event and then finding the first argument where the signal drops
        10 or 5 db below the peak (set high_noise to True if 5 db is desired)

        Parameters
        ----------
        high_noise: bool
            Set to True if the environment is noisy (SEL will be considered until when the signal drops 5 db below
            the peak)
        """
        # Choose the difference in db
        diff_db = 10
        if high_noise:
            diff_db = 5
        # Find the peak
        if len(self.signal) == 0:
            raise UserWarning('This event is empty!')
        cut_start = np.argmax(self.signal)
        peak = self.signal[cut_start]

        # Compute the cut level
        cut_level = peak / (10 ** (diff_db / 10))
        cut_end = np.argwhere(self._total_signal[cut_start+self.start::] < cut_level)
        if len(cut_end) == 0:
            cut_end = self.end
        else:
            cut_end = cut_end[0][0] + self.start + cut_start
        # Reasign signal to the new part and compute SEL
        self.signal = self._total_signal[cut_start+self.start:cut_end]
        sel = super(Event, self).sel()

        # Go back to the previous signal
        self.signal = self._total_signal[self.start:self.end]
        return sel
