"""
Module: impulse_detector.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

from pypam._event import Event

import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class ImpulseDetector:
    def __init__(self, min_separation, band, threshold=150, dt=None):
        """
        Impulse events detector

        Parameters
        ----------
        min_separation : float
            Minimum separation of the event, in seconds
        threshold : float
            Threshold above ref value which one it is considered piling, in db
        dt : float
            Window size in seconds for the analysis (time resolution). Has to be smaller han min_duration!
        """
        self.min_separation = min_separation
        self.threshold = threshold
        self.dt = dt
        self.band = band

    def detect_events(self, signal):
        """
        Detection of event times. Events are detected on the basis of the SPL time series (channel 1)
        The time resolution is dt

        Parameters
        ----------
        signal : Signal object
            Signal to analyze
        """
        levels = []
        blocksize = int(self.dt*signal.fs)
        for block in signal.Blocks(blocksize=blocksize):
            level = block.rms(db=True)
            levels.append(level)
        times = events_times(np.array(levels), self.dt, self.threshold, self.min_separation)

        events_df = pd.DataFrame(columns=['seconds', 'rms', 'sel', 'peak'])
        events_df = events_df.set_index('seconds')
        for t in times:
            event = self.load_event(s=signal, t=t, duration=self.dt)
            rms, sel, peak = event.analyze()
            events_df.at[t] = [rms, sel, peak]
        return events_df

    def load_event(self, s, t, duration, after=0, before=0):
        """
        Load the event at time t (in seconds), with supplied time before and after the event (in seconds)
        return an object event
        Parameters
        ----------
        s : numpy array
            Signal
        t : float
            Starting time of the event (in seconds)
        duration : float
            Duration of the event, in seconds
        before : float
            Time before the event to save, in seconds
        after : float
            Time after the event to save, in seconds
        """
        n1 = int((t - before) * s.fs)
        if n1 < 0:
            n1 = 0
        n2 = int((t + duration + after) * s.fs)
        if n2 > s.signal.shape[0]:
            n2 = s.signal.shape[0]
        event = Event(s.signal[n1:n2], s.fs)
        # event.set_band(self.band)
        return event


class PilingDetector(ImpulseDetector):
    def __init__(self, min_separation, threshold, dt):
        super().__init__(min_separation=min_separation, band=[20, 10000], threshold=threshold, dt=dt)


# @nb.jit
def events_times(levels, dt, threshold, min_separation):
    indices = np.where(levels >= threshold)[0]
    times_th = indices * dt
    times = []
    for i, t in enumerate(times_th):
        if i == 0:
            times.append(t)
        else:
            if (t - times[-1]) >= min_separation:
                times.append(t)
    return times
