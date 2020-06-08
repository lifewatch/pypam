"""
Module: event.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

import os
import sys
import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

plt.style.use('ggplot')


class Event:
    def __init__(self, x, fs):
        """
        * x: 2-channel signal (sound pressure and norm of velocity components)
        * fs: sample rate (in Hz)
        """
        self.x = x
        self.fs = fs


    def sel(self):
        """
        Calculate the sound exposure level of an event (pressure and velocity)
        output:
        * y: 2-element array with SEL values
        """
        y = 10*np.log10((self.x**2).sum()/self.fs)

        return y


    def peak(self):
        """Calculate the peak sound exposure level of an event (pressure and velocity)
        output:
        * y: 2-element array with peak values
        """
        y = 10*np.log10(self.x.abs.max()**2)

        return y


    def sel_spectrum(self, spg, dt):
        """
        Calculation of total spectrum (SEL) of the calibrated spectrogram
        * spg: cell array with in each cell the spectrogram of a single channel of the input signal
        * dt: timestep of the spectrogram calculation
        output:
        * y: cell array with in each cell the spectrum of a single channel of the input signal
        """
        y = []
        for spg_i in spg:
            y.append(10.0*np.log10(sum(10.0**(spg_i/10.0), 1)*dt))
        
        return y


    def average_spectrum(self, spg, dt):
        """
        Calculation of average spectrum (Leq) of the calibrated spectrogram
        * spg: cell array with in each cell the spectrogram of a single channel of the input signal
        * dt: timestep of the spectrogram calculation
        output:
        * y: cell array with in each cell the spectrum of a single channel of the input signal
        """
        y = []
        for spg_i in spg:
            y.append(10.0*np.log10(np.mean(10.0**(spg_i/10.0), 1)))

        return y


    def spectrogram(self, dt):
        """
        Calculation of calibrated 1/3-octave band spectrogram for 28 bands from 25 Hz to 12.5 kHz
        parameters:
        * dt: timestep (in seconds) for calculation of spectrogram
        output:
        * t: array with the time values of the spectrogram
        * f: array with the frequency values of the spectrogram
        * spg: cell array with in each cell the spectrogram of a single channel of the input signal
        """
        # resample signal to 48 kHz
        new_fs = 48000
        frames = self.x.shape[0]       # total number of samples
        channels = self.x.shape[1]      # number of channels

        new_lenght = int(frames * new_fs / self.fs)
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
        """
        sel = self.sel()
        peak = self.peak()
        t, f, spg = self.spectrogram(dt)
        spec = self.sel_spectrum(spg, dt)

        return sel, spec, t, f, spg, peak


    def plot(self, sel, spec, t, f, spg, interval):
        """
        Plot the event 
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
        """
        



# Utility functions -------------------------------------------------------------------------------%

def oct3dsgn(fc, fs, N=3):
    """
    Design of a 1/3-octave band filter with center frequency fc for sampling frequency fs.
    Default value for N is 3. For meaningful results, fc should be in range fs/200 < fc < fs/5.
    """
    if (fc > 0.88*(fs/2)):
        raise Exception('Design not possible - check frequencies')
    # design Butterworth 2N-th-order 1/3-octave band filter
    f1 = fc/(2**(1/6))
    f2 = fc*(2**(1/6))
    Qr = fc/(f2-f1)
    Qd = (np.pi/2/N)/(np.sin(np.pi/2/N))*Qr
    alpha = (1 + np.sqrt(1+4*Qd**2))/2/Qd
    W1 = fc/(fs/2)/alpha
    W2 = fc/(fs/2)*alpha
    b, a = sig.butter(N, [W1, W2])

    return b, a


def oct3bankdsgn(fs, bands, N):
    """
    Construction of a 1/3 octave band filterbank.
    parameters:
    * fs: samplefrequency (Hz), at least 2.3x the center frequency of the highest 1/3 octave band
    * bands: row vector with the desired band numbers (0 = band with center frequency of 1 kHz)
      e.g. [-16:11] gives all bands with center frequency between 25 Hz and 12.5 kHz
    * N: order specification of the filters, N = 2 gives 4th order, N = 3 gives 6th order
      Higher N can give rise to numerical instability problems, so only 2 or 3 should be used
    output:
    * b, a: matrices with filter coefficients, one row per filter.
    * d: column vector with downsampling factors for each filter 1 means no downsampling, 2 means
      downsampling with factor 2, 3 means downsampling with factor 4 and so on.
    * fsnew: column vector with new sample frequencies.
    """
    fc = (1000)*((2**(1/3))**bands)     # exact center frequencies
    fclimit = 1/200                     # limit for center frequency compared to sample frequency
    # calculate downsampling factors
    d = np.ones(len(fc))
    for i in np.arange(len(fc)):
        while fc(i) < (fclimit*(fs/2**(d(i)-1))):
            d[i] += 1
    # calculate new sample frequencies
    fsnew = fs/(2**(d-1))
    # construct filterbank
    a = []
    b = []
    for i in np.arange(len(fc)):
        # construct filter coefficients
        tb, ta = oct3dsgn(fc(i), fsnew(i), N)
        a = [a, ta]
        b = [b, tb]

    return b, a, d, fsnew



