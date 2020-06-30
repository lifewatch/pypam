"""
Module: _event.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

import os
import operator
import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt

plt.style.use('ggplot')

from pypam.utils import *


class Event:
    def __init__(self, x, fs):
        """
        Definition of an acoustic event
        
        Parameters
        ----------
        x : 2D numy array
            2-channel signal (sound pressure and norm of velocity components)
        fs : float
            Sample rate, in Hz
        """
        self.x = x
        self.fs = fs
    

    def duration(self):
        """
        Return the duration of the event 
        """
        time = len(self.x)/self.fs

        return time


    def rms(self):
        """
        Return the rms sound pressure level
        """
        rms = 20*np.log10(np.sqrt((self.x**2).mean()))

        return rms 


    def sel(self):
        """
        Calculate the sound exposure level of an event (pressure and velocity)
        Returns a 2-element array with SEL values
        """
        y = 10*np.log10((self.x**2).sum()/self.fs)

        return y


    def peak(self):
        """
        Calculate the peak sound exposure level of an event (pressure and velocity)
        Returns a 2-element array with peak values
        """
        y = 10*np.log10(np.abs(self.x).max()**2)

        return y


    def sel_spectrum(self, spg, dt):
        """
        Calculation of total spectrum (SEL) of the calibrated spectrogram
        Returns a numpy matrix with in each cell the spectrum of a single channel of the input signal
        
        Parameters
        ----------
        spg: numpy matrix
            Array with in each cell the spectrogram of a single channel of the input signal
        dt : float
            timestep of the spectrogram calculation, in seconds
        """
        y = []
        for spg_i in spg:
            y.append(10.0*np.log10(sum(10.0**(spg_i/10.0), 1)*dt))
        
        return y


    def average_spectrum(self, spg, dt):
        """
        Calculation of average spectrum (Leq) of the calibrated spectrogram
        Returns a numpy array with in each cell the spectrum of a single channel of the input signal
        
        Parameters
        ----------
        spg: numpy matrix
            Array with in each cell the spectrogram of a single channel of the input signal
        dt : float
            timestep of the spectrogram calculation, in seconds
        """
        y = []
        for spg_i in spg:
            y.append(10.0*np.log10(np.mean(10.0**(spg_i/10.0), 1)))

        return y


    def spectrogram(self, dt):
        """
        Calculation of calibrated 1/3-octave band spectrogram for 28 bands from 25 Hz to 12.5 kHz
        Parameters
        ----------
        dt: float
            Timestep for calculation of spectrogram, in seconds
        
        Returns
        -------
        t : numpy array 
            Array with the time values of the spectrogram, in seconds
        f : numpy array
            Array with the frequency values of the spectrogram
        spg : numpy matrix
            Array with in each cell the spectrogram of a single channel of the input signal
        """
        # resample signal to 48 kHz
        new_fs = 48000
        frames = self.x.shape[0]       # total number of samples
        channels = self.x.shape[1]      # number of channels

        new_lenght = int(frames / self.fs)
        x = sig.resample(self.x, new_lenght)

        n = np.floor(new_fs*dt)     # number of samples in one timestep
        nt = np.floor(frames/n) # number of timesteps to process

        # construct filterbank
        bands = np.arange(-16, 11)
        b, a, d, fsnew = oct3bankdsgn(new_fs, bands, 3)
        nx = d.max()            # number of downsampling steps

        # construct time and frequency arrays
        t = np.arange(0, nt-1)*dt
        f = 1000*((2**(1/3))**bands)

        # calculate 1/3 octave band levels vs time
        spg = {}
        newx = {} #CHANGE!!!
        for j in np.arange(channels):
            # perform downsampling of j'th channel
            newx[1] = x[:,j]
            for i in np.arange(2, nx):
                newx[i] = sig.decimate(newx[i-1], 2)
            
            # perform filtering for each frequency band
            for i in np.arange(len(d)):
                factor = 2**(d(i)-1)
                y = sig.sosfilt(b[i,:], a[i,:], newx[d(i)])  # Check the filter!
            # calculate level time series
            for k in np.arange(nt):
                startindex = (k - 1)*n/factor + 1
                endindex = (k*n)/factor
                z = y[startindex:endindex]
                spg[j][k,i] = 10*np.log10(np.sum(z**2)/len(z))

        return t, f, spg


    def analyze(self, dt):
        """
        Perform all necessary calculations for a single event

        Parameters
        ----------
        dt : float
            Integration time, in seconds

        Returns
        -------
        sel : float
            Sound exposure level 
        spec : numpy array
            Spectrum
        t : numpy array 
            Time array in seconds 
        spg : numpy matrix
            Spectrogram
        """
        sel = self.sel()
        peak = self.peak()
        t, f, spg = self.spectrogram(dt)
        spec = self.sel_spectrum(spg, dt)

        return sel, spec, t, f, spg, peak


    def plot(self, sel, spec, t, f, spg, interval):
        """
        Plot the event 

        Parameters
        ----------
        sel : float
            Sound exposure level 
        spec : numpy array
            Spectrum
        t : numpy array 
            Time array in seconds 
        spg : numpy matrix
            Spectrogram
        interval : tuple or list
            min and max values of the y axis
        """
        print('plotting event...')
        fig, ax = plt.subplots(4, 1)
        # plot input signal
        ax[0].plot(np.arange(self.x.shape[0])/self.fs, self.x[:,0])
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Pressure [Pa]')
        ax[0].set_xlim([0.0, 2*t(-1) - t(-2)])
        
        # plot total level over time
        ax[1].plot(t, 10.0*np.log10(sum(10.0**(spg[0]/10.0), 2)))
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Level [dB]')
        ax[1].set_ylim(interval)
        
        # plot spectrogram
        im = ax[2].pcolormesh(t, np.arange(len(f)), spg[10].T)
        # shading flat
        # caxis(interval)
        # c = colorbar
        # set(get(c, 'plt.set_ylabel('), 'String', 'Level [dB]')
        ax[2].set_xlabel('Time [s]')
        ax[2].set_yticks(np.arange(start=2,stop=26,step=3) + 0.5)
        ax[2].set_ytickslabels(['31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k'])
        ax[2].set_ylabel('Frequency band [Hz]')
        
        # plot total spectrum
        ax[3].bar(np.arange(1,28), spec[1], 0.8)
        ax[3].set_xlabel('Frequency band [Hz]')
        ax[3].set_xticks(np.arange(start=2,stop=26,step=3))
        ax[3].set_xtickslabels(['31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k'])
        ax[3].set_ylabel('Level [dB]')
        ax[3].set_ylim(interval)
        # write out calculated values
        print('SEL in time domain: %.4f' % (sel[0]))
        print('SEL in freq domain: %.4f' % (10.0*np.log10(sum(10.0**(spec[0]/10.0)))))


    def draw_box(self, ax):
        """
        Draw a box marking the event

        Parameters
        ----------
        ax: matplotlib axis
        """

        return 0

    
    def oct3bands(self, N=3):
        """
        Return the 1/3 octave band levels of the event

        Parameters
        ----------
        N: int
            Number of bands
        """

        return 0