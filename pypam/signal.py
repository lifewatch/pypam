__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

import operator

import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import scipy.signal as sig
import seaborn as sns
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

from pypam import acoustic_indices
from pypam import utils
from pypam import units as output_units

# Apply the default theme
plt.rcParams.update({'pcolor.shading': 'auto'})
sns.set_theme()


FILTER_ORDER = 4
MIN_FREQ = 1


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

        self.set_band()

    def __getattr__(self, item):
        """
        Return the current band
        """
        if item == 'band':
            if len(self.bands_list) == 0:
                band = -1
            else:
                band = self.bands_list[self.band_n]
            return band
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
    
    def _band_is_broadband(self, band):
        """
        Return True if the selected band is "broaband", return False otherwise

        Parameters
        ----------
        band : list or tuple
            [low_freq, high_freq] of the desired band
        """
        return (band is None) or (band[0] in [0, None] and band[1] in [self.fs/2, None])

    def set_band(self, band=None, downsample=True):
        """
        Process the signal to be working on the specified band.
        If the upper limit band is higher than the nyquist frequency (fs/2), the band limit is set to the
        nyquist frequency. In case downsample is set to True, the band is downsampled to the closest int to twice the
        upper limit frequency. If the lower limit band is not None, it is after filtered with a high pass filter.
        If downsample is set to False, the signal is filtered in the specified band. In case one of the boundaries is
        None, 0 or fs/2, high-pass or low-pass (respectively) filters are used. Otherwise band-pass filter.
        See self.filter for more information about the filtering process.
        Parameters
        ----------
        band : list or tuple
            [low_freq, high_freq] of the desired band
        downsample: bool
            Set to True if signal has to be downsampled for spectral resolution incrementation
        """
        if band is None:
            band = [0, self.fs / 2]
        if band != self.band:
            if self._band_is_broadband(band):
                self.signal = self._signal.copy()
                self.fs = self._fs
            else:
                if band[1] > self._fs / 2:
                    print('Band upper limit %s is too big, setting to maximum fs: new fs %s' % (band[1], self._fs/2))
                    band[1] = self._fs / 2
                if band[1] > self.fs / 2:
                    # Reset to the original data
                    self.signal = self._signal.copy()
                    self.fs = self._fs
                if (not downsample) or (band[1] * 2 == self.fs):
                    self.filter(band=band)
                else:
                    self.downsample2band(band)

            self.band_n += 1
            self._processed[self.band_n] = []
            self.bands_list[self.band_n] = band
        self._reset_spectro()

    def reset_original(self):
        """
        Reset the signal to the original band and process
        """
        original_band = self.bands_list[0]
        self.set_band(original_band)

    def fill_or_crop(self, n_samples):
        """
        Crop the signal to the number specified or fill it with 0 values in case it is too short

        Parameters
        ----------
        n_samples : int
            Number of desired samples
        """
        if self.signal.size >= n_samples:
            self.signal = self.signal[0:n_samples]
            self._processed[self.band_n].append('crop')
        else:
            one_array = np.full((n_samples, ), 0)
            one_array[0:self.signal.size] = self.signal
            self.signal = one_array
            self._processed[self.band_n].append('one-pad')

    def _create_filter(self, band, output='sos'):
        """
        Return the butterworth filter for the specified band. If the limits are set to None, 0 or the nyquist
        frequency, only high-pass or low-pass filters are applied. Otherwise, a band-pass filter.
        Parameters
        ----------
        band: tuple or list
            [low_freq, high_freq], band to be filtered
        """
        if band[0] is None or band[0] == 0:
            sosfilt = sig.butter(N=FILTER_ORDER, btype='lowpass', Wn=band[1], analog=False, output=output, fs=self.fs)
        elif band[1] is None or band[1] == self.fs / 2:
            sosfilt = sig.butter(N=FILTER_ORDER, btype='highpass', Wn=band[0], analog=False, output=output, fs=self.fs)
        else:
            sosfilt = sig.butter(N=FILTER_ORDER, btype='bandpass', Wn=band, analog=False, output=output, fs=self.fs)
        return sosfilt

    def downsample(self, new_fs, filt=None):
        """
        Downsamples the signal to the new fs. If the downsampling factor is an integer, performs resample_poly,
        which applies a 
        Parameters
        ----------
        new_fs: float
            New sampling frequency
        filt:
            filter output of _create_filter(band). If None it will be set to [0, new_fs/2]
        """
        lcm = np.lcm(int(self.fs), int(new_fs))
        ratio_up = int(lcm / self.fs)
        ratio_down = int(lcm / new_fs)
        self.signal = sig.sosfilt(filt, self.signal)
        self._processed[self.band_n].append('filtered')
        self.signal = sig.resample_poly(self.signal, up=ratio_up, down=ratio_down)
        self._processed[self.band_n].append('downsample')
        self.fs = new_fs

    def downsample2band(self, band):
        """
        Reduce the sampling frequency. It uses the decimate function of scipy.signal
        In case the ratio is not an int, the closest int is chosen.
        Parameters
        ----------
        band : tuple
            Band to downsample to (low_freq, high_freq)
        """
        new_fs = band[1] * 2
        if new_fs != self.fs:
            if new_fs > self.fs:
                raise Exception('This is upsampling, can not downsample %s to %s!' % (self.fs, new_fs))
            filt = self._create_filter(band)
            self.downsample(new_fs, filt)
        else:
            print('trying to downsample to the same fs, ignoring...')

    def filter(self, band):
        """
        Filter the signal
        Parameters
        ----------
        band: tuple or list
            [low_freq, high_freq], band to be filtered
        """
        if band[1] > self._fs / 2:
            raise ValueError('Frequency %s is higher than nyquist frequency %s, and can not be filtered' % 
                             (band[1], self.fs / 2))
        if not self._band_is_broadband(band):
            # Filter the signal
            sosfilt = self._create_filter(band)
            self.signal = sig.sosfilt(sosfilt, self.signal)
            self._processed[self.band_n].append('filter')

    def remove_dc(self):
        """
        Remove the dc component of the signal
        """
        dc = np.mean(self.signal)
        self.signal = self.signal - dc
        self._processed[self.band_n].append('dc_removal')

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
        time = []
        for block in self.blocks(blocksize=window):
            try:
                output = f(block)
            except Exception as e:
                print('There was an error in feature %s. Setting to None. '
                      'Error: %s' % (method_name, e))
                output = None
            result.append(output)
            time.append(block.time)
        return time, output

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

    def third_octave_levels(self, db=True, **kwargs):
        """
        Calculation of calibrated 1/3-octave band levels

        Returns
        -------
        f : numpy array
            Array with the center frequencies of the bands
        spg : numpy array
            Level of each band
        """
        return self.octave_levels(db, 3)

    def octave_levels(self, db=True, fraction=1, **kwargs):
        """
        Calculation of calibrated octave band levels

        Returns
        -------
        f : numpy array
            Array with the center frequencies of the bands
        db : boolean
            Set to True to get the result in db
        fraction : int
            fraction of an octave to compute the bands (i.e. fraction=3 leads to 1/3 octave bands)
        """
        bands, f = utils.oct_fbands(min_freq=MIN_FREQ, max_freq=self.fs/2, fraction=fraction)

        # construct filterbank
        filterbank, fsnew, d = utils.octbankdsgn(self.fs, bands, fraction, 2)

        nx = d.max()  # number of downsampling steps
        # calculate octave band levels
        spg = np.zeros(len(d))
        newx = {0: self.signal}
        for i in np.arange(1, nx + 1):
            newx[i] = sig.decimate(newx[i - 1], 2)

        # Perform filtering for each frequency band
        for i in np.arange(len(d)):
            y = sig.sosfilt(filterbank[i], newx[d[i]])  # Check the filter!
            # Calculate level time series
            if db:
                spg[i] = 10 * np.log10(np.sum(y ** 2) / len(y))
            else:
                spg[i] = y

        return f, spg

    def _spectrogram(self, nfft=512, scaling='density', overlap=0.2):
        """
        Computes the spectrogram of the signal and saves it in the attributes

        Parameters
        ----------
        nfft : int
            Length of the fft window in samples. Power of 2.
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        overlap : float [0, 1]
            Percentage (in 1) to overlap

        Returns
        -------
        None
        """
        real_size = self.signal.size
        if self.signal.size < nfft:
            self.fill_or_crop(n_samples=nfft)
        window = sig.get_window('hann', nfft)
        noverlap = overlap * nfft
        freq, t, sxx = sig.spectrogram(self.signal, fs=self.fs, nfft=nfft, window=window, scaling=scaling, noverlap=noverlap)
        if self.band is not None:
            if self.band[0] is None:
                low_freq = 0
            else:
                low_freq = np.argmax(freq >= self.band[0])
        else:
            low_freq = 0
        self.freq = freq[low_freq:]
        n_bins = int(np.floor(real_size / (nfft - noverlap)))
        self.sxx = sxx[low_freq:, 0:n_bins]
        self.t = t[0:n_bins]

    def spectrogram(self, nfft=512, scaling='density', overlap=0, db=True, force_calc=False):
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
        force_calc : bool
            Set to True if the computation has to be forced
        overlap : float [0, 1]
            Percentage (in 1) to overlap

        Returns
        -------
        freq, t, sxx
        """
        if self.sxx is None or force_calc:
            self._spectrogram(nfft=nfft, scaling=scaling, overlap=overlap)
        if db:
            sxx = utils.to_db(self.sxx, ref=1.0, square=False)
        else:
            sxx = self.sxx
        return self.freq, self.t, sxx

    def _spectrum(self, scaling='density', nfft=512, db=True, overlap=0, window_name='hann', **kwargs):
        """
        Return the spectrum : frequency distribution of all the file (periodogram)
        Returns Dataframe with 'datetime' as index and a colum for each frequency and each
        percentile, and a frequency array

        Parameters
        ----------
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        nfft : int
            Length of the fft window in samples. Power of 2. If the signal is shorter it will be
            zero-padded
        db : bool
            If set to True the result will be given in db, otherwise in uPa^2
        overlap : float [0, 1]
            Percentage (in 1) to overlap

        """
        noverlap = nfft * overlap
        if nfft > self.signal.size:
            self.fill_or_crop(n_samples=nfft)
        window = sig.get_window(window_name, nfft)
        freq, psd = sig.welch(self.signal, fs=self.fs, window=window, nfft=nfft, scaling=scaling, noverlap=noverlap,
                              detrend=False, **kwargs)
        if self.band is not None and self.band[0] is not None:
            low_freq = np.argmax(freq >= self.band[0])
        else:
            low_freq = 0
        self.psd = psd[low_freq:]
        self.freq = freq[low_freq:]

        if db:
            self.psd = utils.to_db(self.psd, ref=1.0, square=False)

    # TODO implement stft!

    def spectrum(self, scaling='density', nfft=512, db=True, overlap=0, force_calc=False, percentiles=None, **kwargs):
        """
        Return the spectrum : frequency distribution of all the file (periodogram)
        Returns Dataframe with 'datetime' as index and a column for each frequency and
        each percentile, and a frequency array

        Parameters
        ----------
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        nfft : int
            Length of the fft window in samples. Power of 2.
        overlap : float [0, 1]
            Percentage (in 1) to overlap
        db : bool
            If set to True the result will be given in db, otherwise in uPa^2
        force_calc : bool
            Set to True if the computation has to be forced
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned
        Returns
        -------
        Frequency array, psd values
        """
        if self.psd is None or force_calc:
            self._spectrum(scaling=scaling, nfft=nfft, db=db, overlap=overlap, **kwargs)
        if percentiles is not None:
            percentiles_val = np.percentile(self.psd, percentiles)
        else:
            percentiles_val = None

        return self.freq, self.psd, percentiles_val

    def spectrum_slope(self, scaling='density', nfft=512, db=True, overlap=0, **kwargs):
        """
        Return the slope of the spectrum

        Parameters
        ----------
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        nfft : int
            Length of the fft window in samples. Power of 2.
        overlap : float [0, 1]
            Percentage (in 1) to overlap
        db : bool
            If set to True the result will be given in db, otherwise in uPa^2

        Returns
        -------
        slope of the spectrum (float)
        """
        if self.psd is None:
            self._spectrum(scaling=scaling, nfft=nfft, db=db, overlap=overlap)
        regression = linear_model.LinearRegression().fit(np.log10(self.freq), np.log10(self.psd))
        slope = regression.coef_[0]
        y_pred = regression.predict(np.log10(self.freq))
        error = metrics.mean_squared_error(np.log10(self.psd), y_pred)
        return slope, error

    def aci(self, nfft=512, overlap=0, **kwargs):
        """
        Calculation of root mean squared value (rms) of the signal in uPa for each bin
        Returns Dataframe with 'datetime' as index and 'rms' value as a column

        Parameters
        ----------
        nfft : int
            Number of fft
        overlap : float [0, 1]
            Percentage (in 1) to overlap
        """
        _, _, sxx = self.spectrogram(nfft=nfft, scaling='density', overlap=overlap, db=False)
        aci_val = self.acoustic_index('aci', sxx=sxx)

        return aci_val

    def bi(self, min_freq=2000, max_freq=8000, nfft=512, overlap=0, **kwargs):
        """
        Calculate the Bioacoustic Index index
        Parameters
        ----------
        min_freq: int
            Minimum frequency (in Hertz)
        max_freq: int
            Maximum frequency (in Hertz)
        nfft: int
            FFT number
        overlap : float [0, 1]
            Percentage (in 1) to overlap

        Returns
        -------
        BI value
        """
        if self.band[1] < max_freq or self.band[0] > min_freq:
            print('The band %s does not include this band limits (%s, %s). '
                  'BI will be set to nan' % (self.band, min_freq, max_freq))
            return np.nan
        else:
            _, _, sxx = self.spectrogram(nfft=nfft, scaling='density', overlap=overlap, db=False)
            bi_val = self.acoustic_index('bi', sxx=sxx, frequencies=self.freq, min_freq=min_freq,
                                         max_freq=max_freq)
            return bi_val

    def sh(self, nfft=512, overlap=0, **kwargs):
        """
        Return the Spectral Entropy of Shannon
        Parameters
        ----------
        nfft: int
            FFT number
        overlap : float [0, 1]
            Percentage (in 1) to overlap

        Returns
        -------
        SH index
        """
        _, _, sxx = self.spectrogram(nfft=nfft, overlap=overlap, scaling='density', db=False)
        sh_val = self.acoustic_index('sh', sxx=sxx)
        return sh_val

    def th(self, **kwargs):
        """
        Compute Temporal Entropy of Shannon

        Returns
        -------
        TH value
        """
        th_val = self.acoustic_index('th', s=self.signal)
        return th_val

    def ndsi(self, nfft=512, overlap=0, anthrophony=(1000, 2000), biophony=(2000, 11000), **kwargs):
        """
        Compute the Normalized Difference Sound Index
        Parameters
        ----------
        nfft: int
            FFT number
        overlap : float [0, 1]
            Percentage (in 1) to overlap
        anthrophony: tuple
            Band to consider the anthrophony.
        biophony: tuple
            Band to consider the biophony.

        Returns
        -------
        NDSI value
        """
        if self.band[1] < anthrophony[1] or self.band[1] < biophony[1]:
            print('The band %s does not include anthrophony %s or biophony %s. '
                  'NDSI will be set to nan' % (self.band, anthrophony, biophony))
            return np.nan
        else:
            _, _, sxx = self.spectrogram(nfft=nfft, overlap=overlap, scaling='density', db=False)
            ndsi_val = self.acoustic_index('ndsi', sxx=sxx, frequencies=self.freq, anthrophony=anthrophony,
                                           biophony=biophony)
            return ndsi_val

    def aei(self, db_threshold=-50, freq_step=100, nfft=512, overlap=0, **kwargs):
        """
        Compute Acoustic Evenness Index
        Parameters
        ----------
        db_threshold: int or float
            The minimum db value to consider for the bins of the spectrogram
        freq_step: int
            Size of frequency bands to compute AEI (in Hertz)
        nfft: int
            FFT number
        overlap : float [0, 1]
            Percentage (in 1) to overlap

        Returns
        -------
        AEI value
        """
        _, _, sxx = self.spectrogram(nfft=nfft, scaling='density', overlap=overlap, db=False)
        aei_val = self.acoustic_index('aei', sxx=sxx, frequencies=self.freq, max_freq=self.band[1],
                                      min_freq=self.band[0], db_threshold=db_threshold, freq_step=freq_step)
        return aei_val

    def adi(self, db_threshold=-50, freq_step=100, nfft=512, overlap=0, **kwargs):
        """
        Compute Acoustic Diversity Index
        Parameters
        db_threshold: int or float
            The minimum db value to consider for the bins of the spectrogram
        freq_step: int
            Size of frequency bands to compute AEI (in Hertz)
        nfft: int
            FFT number
        overlap : float [0, 1]
            Percentage (in 1) to overlap

        Returns
        -------
        ADI value
        """
        _, _, sxx = self.spectrogram(nfft=nfft, scaling='density', overlap=overlap, db=False)
        adi_val = self.acoustic_index('adi', sxx=sxx, frequencies=self.freq, max_freq=self.band[1],
                                      min_freq=self.band[0], db_threshold=db_threshold, freq_step=freq_step)
        return adi_val

    def zcr(self, **kwargs):
        """
        Compute the Zero Crossing Rate

        Returns
        -------
        A list of values (number of zero crossing for each window)
        """
        zcr = self.acoustic_index('zcr', s=self.signal, fs=self.fs)
        return zcr

    def zcr_avg(self, window_length=512, window_hop=256, **kwargs):
        """
        Zero Crossing Rate average
        Parameters
        ----------
        window_length: int
            Size of the sliding window (samples)
        window_hop: int
            Size of the lag window (samples)

        Returns
        -------
        ZCR average
        """
        zcr = self.acoustic_index('zcr_avg', s=self.signal, fs=self.fs, window_length=window_length,
                                  window_hop=window_hop)
        return zcr

    def bn_peaks(self, freqband=200, normalization=True, slopes=(0.01, 0.01), nfft=512, overlap=0, **kwargs):
        """
        Counts the number of major frequency peaks obtained on a mean spectrum.
        Parameters
        ----------
        freqband: int or float
            frequency threshold parameter (in Hz). If the frequency difference of two successive peaks
            is less than this threshold, then the peak of highest amplitude will be kept only.
            normalization: if set at True, the mean spectrum is scaled between 0 and 1
        normalization : bool
            Set to true if normalization is desired
        slopes: tuple of length 2
            Amplitude slope parameter, a tuple of length 2. Refers to the amplitude slopes of the peak.
            The first value is the left slope and the second value is the right slope. Only peaks with
            higher slopes than threshold values will be kept. i.e (0.01, 0.01)
        nfft: int
            FFT number
        frequencies: np.array 1D
            List of the frequencies of the spectrogram
        freqband: int or float
            frequency threshold parameter (in Hz). If the frequency difference of two successive peaks
            is less than this threshold, then the peak of highest amplitude will be kept only.
            normalization: if set at True, the mean spectrum is scaled between 0 and 1
        normalization : bool
            Set to true if normalization is desired
        slopes: tuple of length 2
            Amplitude slope parameter, a tuple of length 2. Refers to the amplitude slopes of the peak.
            The first value is the left slope and the second value is the right slope. Only peaks with
            higher slopes than threshold values will be kept. i.e (0.01, 0.01)
        overlap : float [0, 1]
            Percentage (in 1) to overlap

        Returns
        -------
        Int, number of BN peaks
        """
        _, _, sxx = self.spectrogram(nfft=nfft, overlap=overlap, scaling='density', db=False)
        frequencies = self.freq
        meanspec = sxx.mean(axis=1)

        if normalization:
            meanspec = np.array(meanspec) / np.max(meanspec)

        if slopes is not None:
            # Find peaks (with slopes)
            peaks_indices = np.r_[False,
                                  meanspec[1:] > np.array([x + slopes[0] for x in meanspec[:-1]])] & np.r_[
                                meanspec[:-1] > np.array([y + slopes[1] for y in meanspec[1:]]), False]
            peaks_indices = peaks_indices.nonzero()[0].tolist()
        else:
            # scipy method (without slope)
            peaks_indices = sig.argrelextrema(np.array(meanspec), np.greater)[0].tolist()

        # Remove peaks with difference of frequency < freqband
        # number of consecutive index
        nb_bin = next(i for i, v in enumerate(frequencies) if v > freqband)
        for consecutiveIndices in [np.arange(i, i + nb_bin) for i in peaks_indices]:
            if len(np.intersect1d(consecutiveIndices, peaks_indices)) > 1:
                # close values has been found
                maxi, _, _ = np.intersect1d(consecutiveIndices, peaks_indices)
                maxi = maxi[np.argmax([meanspec[f] for f in np.intersect1d(consecutiveIndices, peaks_indices)])]
                peaks_indices = [x for x in peaks_indices if x not in consecutiveIndices]
                # remove all indices that are in consecutiveIndices
                # append the max
                peaks_indices.append(maxi)
        peaks_indices.sort()

        # Frequencies of the peaks
        peak_freqs = [frequencies[p] for p in peaks_indices]
        return len(peaks_indices), peak_freqs

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
                signal.downsample(self.fs)
            elif signal.fs < self.fs:
                self.downsample(signal.fs)
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
        Returns a numpy matrix with in each cell the spectrum of a single channel of
        the input signal

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

    def average_spectrum(self, spg):
        """
        Calculation of average spectrum (Leq) of the calibrated spectrogram
        Returns a numpy array with in each cell the spectrum of a single channel of
        the input signal

        Parameters
        ----------
        spg: numpy matrix
            Array with in each cell the spectrogram of a single channel of the input signal
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

        new_lenght = int(self.signal.size / self.fs)
        x = sig.resample(self.signal, new_lenght)

        n = np.floor(new_fs * dt)  # number of samples in one timestep
        nt = np.floor(self.signal.size / n)  # number of timesteps to process

        # construct filterbank
        bands = np.arange(-16, 11)
        filterbank, fsnew, d = utils.octbankdsgn(new_fs, bands, 3)
        nx = d.max()  # number of downsampling steps

        # construct time and frequency arrays
        t = np.arange(0, nt - 1) * dt
        f = 1000 * ((2 ** (1 / 3)) ** bands)

        # calculate 1/3 octave band levels vs time
        spg = np.zeros(nt, len(d))
        newx = {0: x}
        for i in np.arange(1, nx):
            newx[i] = sig.decimate(newx[i - 1], 2)

        # Perform filtering for each frequency band
        for j in np.arange(len(d)):
            factor = 2 ** (d(j) - 1)
            y = sig.sosfilt(filterbank[j], newx[d(j)])
            # Calculate level time series
            for k in np.arange(nt):
                startindex = (k - 1) * n / factor + 1
                endindex = (k * n) / factor
                z = y[startindex:endindex]
                spg[k, j] = 10 * np.log10(np.sum(z ** 2) / len(z))

        return t, f, spg

    @staticmethod
    def acoustic_index(name, **kwargs):
        """
        Return the acoustic index

        Parameters
        ----------
        name : string
            Name of the Acoustic Index to compute
        """
        f = getattr(acoustic_indices, 'compute_' + name)
        return f(**kwargs)

    def reduce_noise(self, nfft=512, verbose=False):
        """
        Remove the noise of the signal using the noise clip

        Parameters
        ----------
        nfft : int
            Window size to compute the spectrum
        verbose : boolean
            Set to True to plot the signal before and after the reduction
        """
        s = nr.reduce_noise(y=self.signal, sr=self.fs, n_fft=nfft, win_length=nfft)
        if verbose:
            db = True
            label, units = output_units.get_units('spectrogram', log=db, p_ref=1.0)
            label_a, units_a = output_units.get_units('amplitude', log=False)
            _, _, sxx0 = self.spectrogram(nfft, db=db, force_calc=True)
            fig, ax = plt.subplots(2, 3, gridspec_kw={'width_ratios': [1, 1, 0.05]}, sharex='col')
            ax[0, 0].plot(self.times, self.signal)
            ax[0, 0].set_title('Signal')
            ax[0, 0].set_ylabel(r'% [%s]' % (label_a, units_a))
            ax[0, 1].set_axis_off()
            im = ax[0, 1].pcolormesh(self.t, self.freq, sxx0, vmin=60, vmax=150, shading='auto', cmap='viridis')
            plt.colorbar(im, cax=ax[0, 2], label=r'%s [%s]' % (label, units))
            ax[0, 1].set_title('Spectrogram')

            self.signal = s
            _, _, sxx1 = self.spectrogram(nfft, db=True, force_calc=True)
            ax[1, 0].plot(self.times, s)
            ax[1, 0].set_ylabel(r'% [%s]' % (label_a, units_a))
            ax[1, 0].set_xlabel('Time [s]')
            ax[1, 1].set_axis_off()
            ax[1, 1].set_xlabel('Time [s]')
            im = ax[1, 1].pcolormesh(self.t, self.freq, sxx1, vmin=60, vmax=150, shading='auto', cmap='viridis')
            plt.colorbar(im, cax=ax[1, 2], label=r'%s [%s]' % (label, units))
            plt.show()
            plt.close()
        else:
            self.signal = s

        self._processed[self.band_n].append('noisereduction')

    def plot(self, nfft=512, overlap=0, scaling='density', db=True, force_calc=False, show=False, save_path=None,
             vmin=None, vmax=None, log=False):
        """
        Plot the signal and its spectrogram
        Parameters
        ----------
        nfft : int
            nfft value
        scaling : string
            'density' or 'spectrum'
        db : bool
            Set to True for dB output
        force_calc : bool
            Set to True to force the re-calulation of the spectrogram
        overlap : float [0, 1]
            Percentage to overlap in windows for the plot
        show: bool
            Set to True to show
        save_path: str or Path
            Where to save the output. Set to None to not save if (default)
        vmin : float
            minimum value to plot in the spectrogram
        vmax: float
            maximum value to plot in the spectrogram
        """
        plt.rcParams.update(plt.rcParamsDefault)
        _, _, sxx = self.spectrogram(nfft=nfft, scaling=scaling, overlap=overlap, db=db, force_calc=force_calc)

        label, units = output_units.get_units('spectrum_' + scaling, log=db, p_ref=1.0)
        label_a, units_a = output_units.get_units('amplitude', log=False)

        fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05]}, sharex='col')
        ax[0, 0].plot(self.times, self.signal, color='k', linewidth=1)
        ax[0, 0].set_title('Signal')
        ax[0, 0].set_ylabel(r'% [%s]' % (label_a, units_a))
        ax[0, 1].set_axis_off()
        if vmin is None:
            vmin = sxx.min()
        if vmax is None:
            vmax = sxx.max()
        im = ax[1, 0].pcolormesh(self.t, self.freq, sxx, vmin=vmin, vmax=vmax, shading='auto', cmap='magma')
        plt.colorbar(im, cax=ax[1, 1], label=label)
        if log:
            ax[1, 0].set_yscale('symlog')
        ax[1, 0].set_title('Spectrogram')
        ax[1, 0].set_xlabel('Time [s]')
        ax[1, 0].set_ylabel('Frequency [Hz]')
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    def blocks(self, blocksize):
        """
        Wrapper for the Blocks class

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
