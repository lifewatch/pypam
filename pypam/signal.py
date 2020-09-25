"""
Module : signal.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Institute voor de Zee)
"""
from pypam import utils

import numpy as np
import scipy.signal as sig


class Signal:
    def __init__(self, signal, fs):
        # Original signal
        self._fs = fs
        self._signal = signal.copy()

        # Init processed signal
        self.fs = fs
        self.signal = signal.copy()

        # Reset params
        self.band_n = -1
        self.bands_list = {}
        self._processed = {}
        self._reset_spectro()

    def __getattr__(self, item):
        """
        Return the current band
        """
        if item == 'band':
            return self.bands_list[self.band_n]
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

    def _fill_or_crop(self, n_samples):
        """
        Crop the signal to the number specified or fill it with Nan values in case it is too short

        Parameters
        ----------
        n_samples : int
            Number of desired samples
        """
        if self.signal.size >= n_samples:
            self.signal = self.signal[0:n_samples]
            self._processed[self.band_n].append('crop')
        else:
            nan_array = np.full((n_samples,), np.nan)
            nan_array[0:self.signal.size] = self.signal
            self.signal = nan_array
            self._processed[self.band_n].append('fill')

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
            new_length = int(self.signal.size * (self._fs / self.fs))
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

    def rms(self, db=True, **kwargs):
        """
        Calculation of root mean squared value (rms) of the signal in uPa

        Parameters
        ----------
        db : bool
            If set to True the result will be given in db, otherwise in uPa
        """
        rms_val = np.sqrt((self.signal ** 2).mean())
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

        dr = self.signal.max() - self.signal.min()
        # Convert it to db if applicable
        if db:
            dr = utils.to_db(dr, ref=1.0, square=True)

        return dr

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

        Returns
        -------
        None
        """
        if mode == 'fast':
            # Choose the closest power of 2 to clocksize for faster computing
            real_size = self.signal.size
            optim_len = int(2 ** np.ceil(np.log2(real_size)))
            # Fill the missing values with 0
            self._fill_or_crop(n_samples=optim_len)
        window = sig.get_window('hann', nfft)
        freq, self.t, sxx = sig.spectrogram(self.signal, fs=self.fs, nfft=nfft,
                                            window=window, scaling=scaling)
        if self.band is not None:
            low_freq = np.argmax(freq >= self.band[0])
        else:
            low_freq = 0
        self.freq = freq[low_freq:]
        self.sxx = sxx[low_freq:, :]
        if db:
            self.sxx = utils.to_db(self.sxx, ref=1.0, square=False)

    def spectrogram(self, nfft=512, scaling='density', db=True, mode='fast'):
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

        Returns
        -------
        freq, t, sxx
        """
        if self.sxx is None:
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
            Lenght of the fft window in samples. Power of 2.
        db : bool
            If set to True the result will be given in db, otherwise in uPa^2
        """
        if mode == 'fast':
            # Choose the closest power of 2 to clocksize for faster computing
            real_size = self.signal.size
            optim_len = int(2 ** np.ceil(np.log2(real_size)))
            # Fill the missing values with 0
            self._fill_or_crop(n_samples=optim_len)
        window = sig.get_window('boxcar', nfft)
        freq, psd = sig.periodogram(self.signal, fs=self.fs, window=window, nfft=nfft, scaling=scaling)
        if self.band is not None:
            low_freq = np.argmax(self.freq >= self.band[0])
        else:
            low_freq = 0
        self.psd = psd[low_freq:]
        self.freq = freq[low_freq:]

        if db:
            self.psd = utils.to_db(self.psd, ref=1.0, square=False)

    def spectrum(self, scaling='density', nfft=512, db=True, percentiles=None, mode='fast', **kwargs):
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
        """
        if self.psd is None:
            self._spectrum(scaling=scaling, nfft=nfft, db=db, mode=mode)

        if percentiles is not None:
            percentiles_val = np.percentile(self.psd, percentiles)
        else:
            percentiles_val = None

        return self.freq, self.psd, percentiles_val

    def aci(self, nfft, mode='fast', **kwargs):
        """
        Calculation of root mean squared value (rms) of the signal in uPa for each bin
        Returns Dataframe with 'datetime' as index and 'rms' value as a column

        Parameters
        ----------
        nfft : int
            Number of fft
        """
        if self.sxx is None:
            self._spectrogram(nfft=nfft, scaling='density', db=True, mode=mode)
        aci_val = utils.calculate_aci(self.sxx)

        return aci_val

    def correlation(self, signal):
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
