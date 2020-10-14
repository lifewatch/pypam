"""
Module: loud_event_detector.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

from pypam._event import Event
from pypam import utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class LoudEventDetector:
    def __init__(self, min_duration, band, threshold=150):
        """
        Event detector

        Parameters
        ----------
        min_duration : float
            Minimum duration of the event, in seconds
        threshold : float
            Threshold above ref value which one it is considered piling, in db
        """
        self.min_duration = min_duration
        self.threshold = threshold
        self.band = band
        self._last_start = 0

    def reset(self):
        """
        Reset the last detection
        """
        self._last_start = 0

    def detect_events(self, signal, verbose=False):
        """
        Detection of event times. Events are detected on the basis of the SPL time series (channel 1)
        The time resolution is dt

        Parameters
        ----------
        signal : Signal object
            Signal to analyze
        verbose : bool
            Set to True to plot the detection signals
        """
        signal.set_band(band=self.band)
        window = int(self.min_duration * signal.fs / 20.0)
        times_envelope, envelope = signal.average_envelope(window=window)
        envelope = utils.to_db(envelope, ref=1.0, square=True)
        envelope_diff = np.diff(envelope)
        possible_points = np.zeros(envelope_diff.shape)
        possible_points[np.where(np.abs(envelope_diff) >= self.threshold)] = 1
        start_points = times_envelope[np.where(np.diff(possible_points) == 1)[0]]
        end_points = times_envelope[np.where(np.diff(possible_points) == -1)[0]]

        # Check if the file starts with an end
        if end_points.size > start_points.size:
            # If it starts with an end and last file ended with a start, append it
            start_points = np.insert(start_points, 0, self._last_start)
        if start_points.size > end_points.size:
            end_points = np.insert(end_points, -1, signal.duration)
            self._last_start = start_points[-1] - signal.duration
        else:
            self._last_start = 0

        events_df = pd.DataFrame(columns=['start_seconds', 'end_seconds', 'duration', 'rms', 'sel', 'peak'])
        for i, start_i in enumerate(start_points):
            duration = (end_points[i] - start_i)
            if duration >= self.min_duration:
                event = self.load_event(s=signal, n_start=int(start_i*signal.fs),
                                        duration_samples=int(duration*signal.fs))
                rms, sel, peak = event.analyze()
                events_df.at[i] = {'start_seconds': start_i, 'end_seconds': end_points[i],
                                   'duration': duration, 'rms': rms, 'sel': sel, 'peak': peak}

        if verbose:
            fbands, t, sxx = signal.spectrogram(nfft=512, scaling='spectrum', db=True, mode='fast')
            fig, ax = plt.subplots(4, 1, sharex=True)
            ax[0].pcolormesh(t, fbands, sxx)
            ax[0].set_title('Spectrogram')
            ax[0].set_ylabel('Frequency [Hz]')
            ax[0].set_yscale('log')
            ax[1].plot(np.arange(len(signal.signal)) / signal.fs, utils.to_db(signal.signal, ref=1.0,
                                                                              square=True), label='signal')
            ax[1].plot(times_envelope, envelope, label='Average envelope window %s' % window)
            ax[1].set_title('Signal')
            ax[1].set_ylabel('Lrms [dB]')
            ax[1].legend(loc='right')
            ax[2].plot(times_envelope[1:], possible_points, color='green', label='loud events')
            for index in events_df.index:
                row = events_df.loc[index]
                ax[2].axvline(x=row['start_seconds'], color='red')
                ax[2].axvline(x=row['end_seconds'], color='blue')
            ax[2].set_title('Detections')
            ax[2].legend(loc='right')
            ax[2].set_xlabel('Time [s]')
            if len(events_df) > 0:
                events_df[['rms', 'peak', 'sel']].plot(ax=ax[3])
            plt.tight_layout()
            plt.show()
            plt.close()

        return events_df

    @staticmethod
    def load_event(s, n_start, duration_samples):
        """
        Load the event at time t (in seconds), with supplied time before and after the event (in seconds)
        return an object event
        Parameters
        ----------
        s : numpy array
            Signal
        n_start : int
            Starting time of the event (in samples)
        duration_samples : int
            Duration of the event, in samples
        """
        if n_start < 0:
            n_start = 0
        n_end = n_start + duration_samples
        if n_end > s.signal.shape[0]:
            n_end = s.signal.shape[0]
        event = Event(s.signal[n_start:n_end], s.fs)
        return event


class ShipDetector(LoudEventDetector):
    def __init__(self, min_duration=100, threshold=10.0):
        super().__init__(min_duration=min_duration, band=[50, 500], threshold=threshold)

