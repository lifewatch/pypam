"""
Module : signal.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Institute voor de Zee)
"""

__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

import operator

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import noisereduce as nr
import numpy as np
import scipy.signal as sig
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
import seaborn as sns

from pypam import acoustic_indices
from pypam import utils

# Apply the default theme
sns.set_theme()


class Signal:
    def __init__(self, signal, fs, channel=0):
        """
        Representation of a signal
        Parameters
        ----------
        signal : np.array
            Signal to process
        fs : int
            Sample rate
        channel : int
            Channel to perform the calculations in
        """
        # Original signal
        self._fs = fs
        if len(signal.shape) > 1:
            signal = signal[:, channel]
        self._signal = signal.copy()

        # Init processed signal
        self.fs = fs
        self.signal = signal.copy()

        # Reset params
        self.band_n = -1
        self.bands_list = {}
        self._processed = {}
        self._reset_spectro()

        self.set_band(None)

    def __getattr__(self, item):
        """
        Return the current band
        """
        if item == 'band':
            return self.bands_list[self.band_n]
        elif item == 'duration':
            return len(self.signal) / self.fs
        elif item == 'times':
            return np.arange(self.signal.shape[0]) / self.fs
        else:
            return self.__dict__[item]

    def _reset_spectro(self):
        """
        Reset the spectrogram parameters
        """
        self.sxx = None
        self.psd = None
        self.freq = None
        self.t = None

    def set_band(self, band):
        """
        Process the signal to be working on the specified band
        Parameters
        ----------
        band : list or tuple
            [low_freq, high_freq] of the desired band
        """
        self.band_n += 1
        self._processed[self.band_n] = []
        self.bands_list[self.band_n] = band

        # Restart the signal to the original one
        self.signal = self._signal.copy()
        self.fs = self._fs
        self._reset_spectro()
        self._filter_and_downsample()

    def reset_original(self):
        """
        Reset the signal to the original band and process
        """
        original_band = self.bands_list[0]
        self.set_band(original_band)

    def _fill_or_crop(self, n_samples):
        """
        Crop the signal to the number specified or fill it with 0 values in case it is too short

        Parameters
        ----------
        n_samples : int
            Number of desired samples
        """
        if self.signal.size >= n_samples:
            s = self.signal[0:n_samples]
            # self._processed[self.band_n].append('crop')
        else:
            nan_array = np.full((n_samples,), 0)
            nan_array[0:self.signal.size] = self.signal
            s = nan_array
            # self._processed[self.band_n].append('fill')
        return s

    def _downsample(self, new_fs):
        """
        Reduce the sampling frequency
        Parameters
        ----------
        new_fs : int
            New sampling rate
        """
        self.fs = new_fs
        if self.fs > self._fs:
            raise Exception('This is upsampling!')
        ratio = (self._fs / self.fs)
        if (ratio % 2) != 0:
            new_length = int(self.signal.size * (self.fs / self._fs))
            self.signal = sig.resample(self.signal, new_length)
        else:
            self.signal = sig.resample_poly(self.signal, up=1, down=int(ratio))
        self._processed[self.band_n].append('downsample')

    def _filter_and_downsample(self):
        """
        Filter and downsample the signal
        """
        if self.band is not None:
            # Filter the signal
            sosfilt = sig.butter(N=4, btype='bandpass', Wn=self.band, analog=False, output='sos', fs=self.fs)
            self.signal = sig.sosfilt(sosfilt, self.signal)
            self._processed[self.band_n].append('filter')

            # Downsample if frequency analysis to get better resolution
            if self.band[1] < self.fs / 2:
                self._downsample(self.band[1] * 2)

    def envelope(self):
        """
        Return the envelope of the signal
        """
        analytic_signal = sig.hilbert(self.signal)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope

    def average_envelope(self, window):
        """
        Return the average envelope for each window
        Parameters
        ----------
        window : int
            Number of samples for each window to average
        """
        result = []
        envelope = self.envelope()
        for block in Blocks(signal=envelope, fs=self.fs, blocksize=window):
            val = np.sqrt((block.envelope() ** 2).mean())
            result.append(val)
        result = np.array(result)
        times = np.arange(len(result)) * window / self.fs
        return times, result

    def window_method(self, method_name, window, **kwargs):
        """
        Return the average envelope for each window
        Parameters
        ----------
        method_name : string
            Name of the function to calculate in each window
        window : int
            Number of samples for each window to average
        """
        f = operator.methodcaller(method_name, **kwargs)
        result = []
        for block in self.blocks(blocksize=window):
            val = f(block)
            result.append(val)
        result = np.array(result)
        times = np.arange(len(result)) * window / self.fs
        return times, result

    def rms(self, db=True, **kwargs):
        """
        Calculation of root mean squared value (rms) of the signal in uPa

        Parameters
        ----------
        db : bool
            If set to True the result will be given in db, otherwise in uPa
        """
        rms_val = utils.rms(self.signal)
        # Convert it to db if applicable
        if db:
            rms_val = utils.to_db(rms_val, ref=1.0, square=True)
        return rms_val

    def dynamic_range(self, db=True, **kwargs):
        """
        Compute the dynamic range of each bin
        Returns a dataframe with datetime as index and dr as column

        Parameters
        ----------
        db : bool
            If set to True the result will be given in db, otherwise in uPa
        """
        dr = utils.dynamic_range(self.signal)
        # Convert it to db if applicable
        if db:
            dr = utils.to_db(dr, ref=1.0, square=True)
        return dr

    def sel(self, db=True, **kwargs):
        """
        Calculate the sound exposure level of an event
        """
        y = utils.sel(self.signal, self.fs)
        if db:
            y = utils.to_db(y, square=False)
        return y

    def peak(self, db=True, **kwargs):
        """
        Calculate the peak sound exposure level of an event (pressure and velocity)
        Returns a 2-element array with peak values
        """
        y = utils.peak(self.signal)
        if db:
            y = utils.to_db(y, square=True)
        return y

    def _spectrogram(self, nfft=512, scaling='density', db=True, mode='fast'):
        """
        Computes the spectrogram of the signal and saves it in the attributes

        Parameters
        ----------
        nfft : int
            Length of the fft window in samples. Power of 2.
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        db : bool
            If set to True the result will be given in db, otherwise in uPa^2
        mode : string
            If set to 'fast', the signal will be zero padded up to the closest power of 2

        Returns
        -------
        None
        """
        real_size = self.signal.size
        if self.signal.size < nfft:
            s = self._fill_or_crop(n_samples=nfft)
        else:
            if mode == 'fast':
                # Choose the closest power of 2 to clocksize for faster computing
                optim_len = int(2 ** np.ceil(np.log2(real_size)))
                # Fill the missing values with 0
                s = self._fill_or_crop(n_samples=optim_len)
            else:
                s = self.signal
        window = sig.get_window('hann', nfft)
        freq, t, sxx = sig.spectrogram(s, fs=self.fs, nfft=nfft,
                                       window=window, scaling=scaling)
        if self.band is not None:
            low_freq = np.argmax(freq >= self.band[0])
        else:
            low_freq = 0
        self.freq = freq[low_freq:]
        n_bins = int(np.floor(real_size / (nfft * 7 / 8)))
        self.sxx = sxx[low_freq:, 0:n_bins]
        self.t = t[0:n_bins]
        if db:
            self.sxx = utils.to_db(self.sxx, ref=1.0, square=False)

    def spectrogram(self, nfft=512, scaling='density', db=True, mode='fast', force_calc=False):
        """
        Return the spectrogram of the signal (entire file)

        Parameters
        ----------
        db : bool
            If set to True the result will be given in db, otherwise in uPa^2
        nfft : int
            Length of the fft window in samples. Power of 2.
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        mode : string
            If set to 'fast', the signal will be zero padded up to the closest power of 2
        force_calc : bool
            Set to True if the computation has to be forced

        Returns
        -------
        freq, t, sxx
        """
        if self.sxx is None or force_calc:
            self._spectrogram(nfft=nfft, scaling=scaling, db=db, mode=mode)

        return self.freq, self.t, self.sxx

    def _spectrum(self, scaling='density', nfft=512, db=True, mode='fast'):
        """
        Return the spectrum : frequency distribution of all the file (periodogram)
        Returns Dataframe with 'datetime' as index and a colum for each frequency and each percentile,
        and a frequency array

        Parameters
        ----------
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        nfft : int
            Lenght of the fft window in samples. Power of 2. If the signal is shorter it will be zero-padded
        db : bool
            If set to True the result will be given in db, otherwise in uPa^2
        mode : string
            If set to 'fast', the signal will be zero padded up to the closest power of 2
        """
        if nfft > self.signal.size:
            s = self._fill_or_crop(n_samples=nfft)
        else:
            if mode == 'fast':
                # Choose the closest power of 2 to clocksize for faster computing
                real_size = self.signal.size
                optim_len = int(2 ** np.ceil(np.log2(real_size)))
                # Fill the missing values with 0
                s = self._fill_or_crop(n_samples=optim_len)
            else:
                s = self.signal
        window = sig.get_window('boxcar', nfft)
        freq, psd = sig.periodogram(s, fs=self.fs, window=window, nfft=nfft, scaling=scaling)
        if self.band is not None:
            low_freq = np.argmax(self.freq >= self.band[0])
        else:
            low_freq = 0
        self.psd = psd[low_freq:]
        self.freq = freq[low_freq:]

        if db:
            self.psd = utils.to_db(self.psd, ref=1.0, square=False)

    def spectrum(self, scaling='density', nfft=512, db=True, percentiles=None, mode='fast', force_calc=False, **kwargs):
        """
        Return the spectrum : frequency distribution of all the file (periodogram)
        Returns Dataframe with 'datetime' as index and a column for each frequency and each percentile,
        and a frequency array

        Parameters
        ----------
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        nfft : int
            Length of the fft window in samples. Power of 2.
        db : bool
            If set to True the result will be given in db, otherwise in uPa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list, no percentiles is returned
        mode : string
            If set to 'fast', the signal will be zero padded up to the closest power of 2
        force_calc : bool
            Set to True if the computation has to be forced
        Returns
        -------
        Frequency array, psd values, percentiles values
        """
        if self.psd is None or force_calc:
            self._spectrum(scaling=scaling, nfft=nfft, db=db, mode=mode)

        if percentiles is not None:
            percentiles_val = np.percentile(self.psd, percentiles)
        else:
            percentiles_val = None

        return self.freq, self.psd, percentiles_val

    def spectrum_slope(self, scaling='density', nfft=512, db=True, percentiles=None, mode='fast', **kwargs):
        """
        Return the slope of the spectrum

        Parameters
        ----------
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        nfft : int
            Length of the fft window in samples. Power of 2.
        db : bool
            If set to True the result will be given in db, otherwise in uPa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list, no percentiles is returned
        mode : string
            If set to 'fast', the signal will be zero padded up to the closest power of 2

        Returns
        -------
        slope of the spectrum (float)
        """
        if self.psd is None:
            self._spectrum(scaling=scaling, nfft=nfft, db=db, mode=mode)
        regression = linear_model.LinearRegression().fit(np.log10(self.freq), np.log10(self.psd))
        slope = regression.coef_[0]
        y_pred = regression.predict(np.log10(self.freq))
        error = metrics.mean_squared_error(np.log10(self.psd), y_pred)
        return slope, error

    def aci(self, nfft, mode='fast', **kwargs):
        """
        Calculation of root mean squared value (rms) of the signal in uPa for each bin
        Returns Dataframe with 'datetime' as index and 'rms' value as a column

        Parameters
        ----------
        nfft : int
            Number of fft
        mode : string
            If set to 'fast', the signal will be zero padded up to the closest power of 2
        """
        if self.sxx is None:
            self._spectrogram(nfft=nfft, scaling='density', db=True, mode=mode)
        aci_val = acoustic_indices.calculate_aci(self.sxx)

        return aci_val

    def total_correlation(self, signal):
        """
        Compute the correlation with the signal

        Parameters
        ----------
        signal : numpy array or signal object
            Signal to be correlated with
        """
        if isinstance(signal, Signal):
            if signal.fs > self.fs:
                signal._downsample(self.fs)
            elif signal.fs < self.fs:
                self._downsample(signal.fs)
        coeff = np.corrcoef(self.signal, signal.signal)

        return coeff

    def blocks_correlation(self, signal):
        """
        Compute the correlation with the signal for each block of the same length than the signal

        Parameters
        ----------
        signal : numpy array or signal object
            Signal to be correlated with
        """
        coeff_evo = []
        for block in self.blocks(blocksize=signal.size):
            coeff_evo.append(np.corrcoef(block.signal, signal))
        return coeff_evo

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
            y.append(10.0 * np.log10(sum(10.0 ** (spg_i / 10.0), 1) * dt))
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
            y.append(10.0 * np.log10(np.mean(10.0 ** (spg_i / 10.0), 1)))
        return y

    def spectrogram_third_bands(self, dt):
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
        frames = self.signal.shape[0]  # total number of samples
        channels = self.signal.shape[1]  # number of channels

        new_lenght = int(frames / self.fs)
        x = sig.resample(self.signal, new_lenght)

        n = np.floor(new_fs * dt)  # number of samples in one timestep
        nt = np.floor(frames / n)  # number of timesteps to process

        # construct filterbank
        bands = np.arange(-16, 11)
        b, a, d, fsnew = utils.oct3bankdsgn(new_fs, bands, 3)
        nx = d.max()  # number of downsampling steps

        # construct time and frequency arrays
        t = np.arange(0, nt - 1) * dt
        f = 1000 * ((2 ** (1 / 3)) ** bands)

        # calculate 1/3 octave band levels vs time
        spg = {}
        newx = {}  # CHANGE!!!
        for j in np.arange(channels):
            # perform downsampling of j'th channel
            newx[1] = x[:, j]
            for i in np.arange(2, nx):
                newx[i] = sig.decimate(newx[i - 1], 2)

            # Perform filtering for each frequency band
            for i in np.arange(len(d)):
                factor = 2 ** (d(i) - 1)
                y = sig.sosfilt(b[i, :], a[i, :], newx[d(i)])  # Check the filter!
            # Calculate level time series
            for k in np.arange(nt):
                startindex = (k - 1) * n / factor + 1
                endindex = (k * n) / factor
                z = y[startindex:endindex]
                spg[j][k, i] = 10 * np.log10(np.sum(z ** 2) / len(z))

        return t, f, spg

    def acoustic_index(self, name, **kwargs):
        """
        Return the acoustic index

        Parameters
        ----------
        name : string
            Name of the Acoustic Index to compute
        """
        f = getattr(acoustic_indices, 'compute_' + name)
        return f(**kwargs)

    def reduce_noise(self, noise_clip, prop_decrease=1.0, nfft=512):
        """
        Remove the noise of the signal using the noise clip

        Parameters
        ----------
        noise_clip : np.array
            Signal representing the noise to be removed
        prop_decrease : float
            0 to 1 amout of noise to be removed (0 None, 1 All)
        nfft : int
            Window size to compute the spectrum
        """
        self.signal = nr.reduce_noise(audio_clip=self.signal, noise_clip=noise_clip, prop_decrease=prop_decrease,
                                      n_fft=nfft, win_length=nfft, verbose=False, n_grad_freq=1, n_grad_time=1,
                                      hop_length=int(nfft * 0.2))
        self._processed[self.band_n].append('noisereduction')

    def plot(self, nfft=512, scaling='density', db=True, force_calc=False):
        """
        Plot the signal and its spectrogram
        """
        self.spectrogram(nfft, scaling, db, mode=None, force_calc=force_calc)

        fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05]}, sharex='col')
        ax[0, 0].plot(self.times, self.signal)
        ax[0, 0].set_title('Signal')
        ax[0, 0].set_xlabel('Time [s]')
        ax[0, 0].set_ylabel(r'Amplitude [$\mu Pa$]')
        ax[0, 1].set_axis_off()
        im = ax[1, 0].pcolormesh(self.t, self.freq, self.sxx, vmin=60, vmax=150, shading='auto')
        plt.colorbar(im, cax=ax[1, 1], label=r'$L_{rms}$ [dB]')
        ax[1, 0].set_title('Spectrogram')
        ax[1, 0].set_xlabel('Time [s]')
        ax[1, 0].set_ylabel('Frequency [Hz]')
        plt.show()
        plt.close()

    def blocks(self, blocksize):
        """
        Wrappper for the Blocks class

        Parameters
        ----------
        blocksize : float
            Window integration time, in samples
        """
        return Blocks(self.signal, self.fs, blocksize)


class Blocks:
    def __init__(self, signal, fs, blocksize):
        """
        Init

        Parameters
        ----------
        blocksize : float
            Window integration time, in samples
        """
        self.blocksize = blocksize
        self.signal = signal
        self.fs = fs
        self.nsamples = self.signal.shape[0]
        self.n = 0

    def __getattr__(self, item):
        """
        Return the attribute
        """
        if item == 'time':
            return self.n * self.blocksize
        else:
            return self.__dict__[item]

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
            block = self.signal[self.n * self.blocksize: self.n * self.blocksize + self.blocksize]
            self.n += 1
            s = Signal(block, self.fs)
            return s
        else:
            raise StopIteration
