"""
Module: piling_detector.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from pypam._event import Event

plt.style.use('ggplot')


class PilingDetector:
    def __init__(self, min_duration, ref=0, threshold=150, dt=None, continuous=True, p_ref=1.0):
        """
        Event detector

        Parameters
        ----------
        min_duration : float
            Minimum duration of the event, in seconds
        ref : float
            Noise reference value, in db 
        threshold : float
            Threshold above ref value which one it is considered piling, in dB
        dt : float
            Window size in seconds for the analysis (time resolution). Has to be smaller han min_duration!
        continuous : boolean
            Weather the file is continuous or not (TO BE IMPLEMENTED)
        """
        self.min_duration = min_duration
        self.reference_level(ref)
        self.threshold = threshold
        self.continuous = continuous
        self.dt = dt
        self.p_ref = 1.0
        self.before = 0.0
        self.after = dt
        # self.bands = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250',
        #              '315', '400', '500', '630', '800', '1000', '1250', '1600', '2000', '2500',
        #              '3150', '4000', '5000', '6300', '8000', '10000', '12500']

    def detect_events(self, x=None, fs=None, datetime_start=None):
        """
        Detection of event times. Events are detected on the basis of the SPL time series (channel 1)
        The time resolution is dt

        Parameters
        ----------
        x : numpy array
            Signal to analyze
        fs : float
            Sampling frequency of the signal
        """
        # calculation of level-vs-time
        levels = []
        for block in self.Blocks(x=x, fs=fs, dt=self.dt):
            level = 20 * np.log10(np.sqrt(block ** 2).mean() / self.p_ref)
            levels.append(level)

        # find event start times
        times = self.find_time_events(levels=levels)

        # print('number of events: %d', len(times))
        columns = ['datetime', 'duration', 'rms', 'sel', 'peak']
        events = pd.DataFrame(columns=columns)
        events = events.set_index('datetime')
        for t in times:
            # CHECK IF EVENT!
            event = self.load_event(x, fs, t, self.before, self.after)
            time = datetime_start + datetime.timedelta(seconds=t)
            events.loc[time] = [event.duration(), event.rms(), event.sel(), event.peak()]

        return events

    def find_time_events(self, levels):
        """
        Estimation of events that exceed the threshold,
        with at least the given min_duration (in seconds) in between events

        Parameters
        ----------
        levels : numpy array 
            Signal level in dB for each dt
        """
        indices = np.where(np.array(levels) >= self.threshold)[0]
        times = indices * self.dt

        return times

    def load_event(self, x, fs, t, before, after):
        """
        Load the event at time t (in seconds), with supplied time before and after the event (in seconds)
        return an object event

        Parameters
        ----------
        x : numpy array 
            Signal
        fs : float
            Sampling frequency of the signal x
        t : float
            Starting time of the event (in seconds)
        before : float
            Time before the event to save, in seconds
        after : float
            Time after the event to save, in seconds
        """
        n1 = int((t - before) * fs)
        if n1 < 0:
            n1 = 0
        n2 = int((t + after) * fs)
        if n2 > x.shape[0]:
            n2 = x.shape[0]

        event = Event(x[n1:n2], fs)

        return event

    def plot_events(self, levels, thresholds):
        """
        Function that plots the number of events for a range of thresholds
        Can be used to pick the best threshold value

        Parameters
        ----------
        levels : numpy array
            1-D array of sound levels
        thresholds : numpy array
            1-D array of sound level thresholds to be compared
        min_duration : float
            the minimum duration between events (in seconds)
        """
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(np.arange(len(levels)) * self.dt, levels)
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Sound pressure level [dB]')

        numbers = np.zeros(thresholds.size)
        for i, th in enumerate(thresholds):
            numbers[i] = len(self.find_time_events(levels, self.dt, th, self.min_duration))

        ax[1].plot(thresholds, numbers)
        ax[1].set_xlabel('Threshold [dB]')
        ax[1].set_ylabel('Number of events')

    def reference_level(self, iref):
        """
        Calculation of reference level

        Parameters
        ----------
        iref : number or file path
            Either a number (e.g. -6 dB rms), or the filename of a single-channel wavfile containing
            a calibration tone (mono, assumed to be clean, i.e. not too much background noise)
        """
        if os.path.exists(iref):
            print('calculating reference level...')
            signal = sf.read(iref)
            if (signal.shape[1] != 1):
                raise Exception('Reference wavfile must be mono')

            oref = 10 * np.log10((signal ** 2).sum() / signal.size)
        else:
            # input is already a level
            oref = iref
        print('-> reference level %.2f' % (oref))
        self.ref = oref

        return oref

    class Blocks:
        def __init__(self, x, fs, dt):
            """
            Init

            Parameters
            ----------
            x : numpy array 
                Signal
            fs : float
                Sampling frequency
            dt : float
                Window integration time, in seconds
            """
            self.blocksize = int(dt * fs)
            self.x = x
            self.nsamples = x.shape[0]

        def __iter__(self):
            """
            Iteration
            """
            self.n = 0

            return self

        def __next__(self):
            """
            Return next block
            """
            if (self.n * self.blocksize) < self.nsamples:
                block = self.x[self.n * self.blocksize: self.n * self.blocksize + self.blocksize]
                self.n += 1
                return block
            else:
                raise StopIteration
