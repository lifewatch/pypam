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

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


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
        f : numpy array
            Frequency array
        spg : numpy matrix
            Spectrogram
        interval : tuple or list
            min and max values of the y axis
        """
        print('plotting event...')
        fig, ax = plt.subplots(4, 1)
        # plot input signal
        ax[0].plot(np.arange(self.x.shape[0])/self.fs, self.x[:, 0])
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
        ax[2].set_yticks(np.arange(start=2, stop=26, step=3) + 0.5)
        ax[2].set_ytickslabels(['31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k'])
        ax[2].set_ylabel('Frequency band [Hz]')
        
        # plot total spectrum
        ax[3].bar(np.arange(1, 28), spec[1], 0.8)
        ax[3].set_xlabel('Frequency band [Hz]')
        ax[3].set_xticks(np.arange(start=2, stop=26, step=3))
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
