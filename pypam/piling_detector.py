"""
Module: piling_detector.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

import os
import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt

from pypam._event import Event


plt.style.use('ggplot')


class PilingDetector:
    def __init__(self, min_duration, ref=-6, threshold=150, dt=None, continuous=True):
        """
        Event detector
        `min_duration`: minimum duration of the event, in seconds
        `ref`: noise reference value, in db 
        `threshold`: threshold above which one it is considered piling
        `dt`: window size in seconds for the analysis (time resolution)
        `continuous`: TO BE IMPLEMENTED
        """
        self.min_duration = min_duration 
        self.reference_level(ref)
        self.threshold = threshold
        self.continuous = continuous
        self.dt = dt
        # self.bands = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250',
        #              '315', '400', '500', '630', '800', '1000', '1250', '1600', '2000', '2500',
        #              '3150', '4000', '5000', '6300', '8000', '10000', '12500']


    def detect_events(self, x=None, fs=None):
        """
        Detection of event times. Events are detected on the basis of the SPL time series (channel 1)
        The time resolution is dt
        `x`: signal to analyze
        `fs`: sampling frequency of the signal
        """
        # calculation of level-vs-time
        levels = []
        for block in self.Blocks(x=x, fs=fs, dt=self.dt):
            level[i] = 10*np.log10(np.sqrt(block**2).mean())

        # find event start times
        times = self.find_time_events(levels=levels)
        print('number of events: %d', len(times))
        events = []
        for t in times: 
            event = self.load_event(x, fs, t, self.before, self.start)
            events.append(event)

        return events


    def find_time_events(self, levels):
        """
        Estimation of events that exceed the threshold,
        with at least the given min_duration (in seconds) in between events
        `levels`: signal level in dB for each dt
        """
        indices = np.argmax(levels >= self.threshold)
        temp = indices * self.dt
        time = []
        for t in temp:
            if len(time) == 0:
                time.append(t)
            else:
                if (t - time[-1]) >= self.min_duration:
                    time.append(t)

        return time


    def load_event(self, x, fs, t, before, after):
        """
        Load the event at time t (in seconds), with supplied time before and after the event (in seconds)
        return an object event
        `x`:
        `fs`:
        `t`:
        `before`: 
        `after`: 
        """
        n1 = int((t-before)*fs)
        if n1 < 0:
            n1 = 0
        n2 = int((t+after)*fs)
        if n2 > x.shape[0]:
            n2 = x.shape[0]

        event = event.Event(x[n1:n2], fs)
        
        return event


    def plot_events(self, levels, thresholds):
        """
        Function that plots the number of events for a range of thresholds
        Can be used to pick the best threshold value
        * levels: 1-D array of sound levels
        * dt: timestep (in seconds)
        * thresholds: 1-D array of sound level thresholds
        * min_duration: the minimum duration between events (in seconds)
        """
        fig, ax = plt.subplots(2,1)
        ax[0].plot(np.arange(len(levels))*self.dt, levels)
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Sound pressure level [dB]')
        
        numbers = np.zeros(thresholds.size)
        for i, th in enumerate(thresholds):
            numbers[i] = len(find_time_events(levels, self.dt, th, self.min_duration))
            
        ax[1].plot(thresholds, numbers)
        ax[1].set_xlabel('Threshold [dB]')
        ax[1].set_ylabel('Number of events')


    def reference_level(self, iref):
        """
        Calculation of reference level
        `iref`: either a number (e.g. -6 dB rms), or the filename of a single-channel wavfile containing
        a calibration tone (mono, assumed to be clean, i.e. not too much background noise)
        """
        if os.path.exists(iref):
            print('calculating reference level...')
            signal = sf.read(iref)
            if (signal.shape[1] != 1):
                raise Exception('Reference wavfile must be mono')

            oref = 10*np.log10((signal**2).sum()/signal.size)
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
            """
            self.blocksize = int(dt*fs)
            self.fs = fs
            self.x = x
            self.nsamples = x.shape[0]


        def __iter__(self, blocksize):
            """
            Iteration
            """
            self.n = 0
            

        def __next__(self):
            """
            Return next block
            """
            if (self.n * self.blocksize) < self.nsamples:
                block = x[self.n*self.blocksize : self.n*self.blocksize + self.blocksize]
                self.n += 1
                return block
            else:
                raise StopIteration

