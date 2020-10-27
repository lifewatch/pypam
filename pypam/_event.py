"""
Module: _event.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

from pypam.signal import Signal


class Event(Signal):
    def __init__(self, signal, fs):
        """
        Definition of an acoustic event
        
        Parameters
        ----------
        signal : 1D numpy array
            Signal
        fs : int
            Sample rate, in Hz
        """
        super().__init__(signal, fs)

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
