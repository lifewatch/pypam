import operator
import pathlib

import matplotlib.pyplot as plt
import noisereduce as nr
import numpy as np
import scipy.signal as sig
import seaborn as sns
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics

from pypam import acoustic_indices
from pypam import units as output_units
from pypam import utils

# Apply the default theme
plt.rcParams.update({"pcolor.shading": "auto"})
sns.set_theme()

FILTER_ORDER = 4
MIN_FREQ = 1


class Signal:
    def __init__(self, signal: np.array, fs: int, channel: int = 0):
        """
        Representation of a signal

        Args:
            signal: Signal to process
            fs: Sample rate
            channel: Channel to perform the calculations in
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
        if item == "band":
            if len(self.bands_list) == 0:
                band = -1
            else:
                band = self.bands_list[self.band_n]
            return band
        elif item == "duration":
            return len(self.signal) / self.fs
        elif item == "times":
            return np.arange(self.signal.shape[0]) / self.fs
        else:
            return self.__dict__[item]

    def _reset_spectro(self) -> None:
        """
        Reset the spectrogram parameters
        """
        self.sxx = None
        self.psd = None
        self.freq = None
        self.t = None

    def _band_is_broadband(self, band: list or tuple) -> bool:
        """
        Return True if the selected band is "broadband", return False otherwise

        Args:
            band: [low_freq, high_freq] of the desired band, in Hz
        """
        return (band is None) or (
            band[0] in [0, None] and band[1] in [self.fs / 2, None]
        )

    def set_band(self, band: list or tuple = None, downsample: bool = True) -> None:
        """
        Process the signal to be working on the specified band.
        If the upper limit band is higher than the nyquist frequency (fs/2), the band limit is set to the
        nyquist frequency. In case downsample is set to True, the band is downsampled to the closest int to twice the
        upper limit frequency. If the lower limit band is not None, it is after filtered with a high pass filter.
        If downsample is set to False, the signal is filtered in the specified band. In case one of the boundaries is
        None, 0 or fs/2, high-pass or low-pass (respectively) filters are used. Otherwise band-pass filter.
        See self.filter for more information about the filtering process.

        Args:
            band: [low_freq, high_freq] of the desired band
            downsample: Set to True if signal has to be downsampled for spectral resolution incrementation
        """
        if band is None:
            band = [0, self.fs / 2]
        if band != self.band:
            if self._band_is_broadband(band):
                self.signal = self._signal.copy()
                self.fs = self._fs
            else:
                if band[1] > self._fs / 2:
                    print(
                        "Band upper limit %s is too big, setting to maximum fs: new fs %s"
                        % (band[1], self._fs / 2)
                    )
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

    def reset_original(self) -> None:
        """
        Reset the signal to the original band and process
        """
        original_band = self.bands_list[0]
        self.set_band(original_band)

    def fill_or_crop(self, n_samples: int) -> None:
        """
        Crop the signal to the number specified or fill it with 0 values in case it is too short

        Args:
            n_samples: Number of desired samples
        """
        if self.signal.size >= n_samples:
            self.signal = self.signal[0:n_samples]
            self._processed[self.band_n].append("crop")
        else:
            one_array = np.full((n_samples,), 0)
            one_array[0 : self.signal.size] = self.signal
            self.signal = one_array
            self._processed[self.band_n].append("one-pad")

    def _create_filter(self, band: list or tuple, output: str = "sos") -> tuple:
        """
        Return the butterworth filter for the specified band. If the limits are set to None, 0 or the nyquist
        frequency, only high-pass or low-pass filters are applied. Otherwise, a band-pass filter.
        Args:
            band: [low_freq, high_freq], band to be filtered
            output: filter style

        Returns:
            tuple representing the filter
        """
        if band[0] is None or band[0] == 0:
            sosfilt = sig.butter(
                N=FILTER_ORDER,
                btype="lowpass",
                Wn=band[1],
                analog=False,
                output=output,
                fs=self.fs,
            )
        elif band[1] is None or band[1] == self.fs / 2:
            sosfilt = sig.butter(
                N=FILTER_ORDER,
                btype="highpass",
                Wn=band[0],
                analog=False,
                output=output,
                fs=self.fs,
            )
        else:
            sosfilt = sig.butter(
                N=FILTER_ORDER,
                btype="bandpass",
                Wn=band,
                analog=False,
                output=output,
                fs=self.fs,
            )
        return sosfilt

    def downsample(self, new_fs: int, filt: tuple or list = None) -> None:
        """
        Downsample the signal to the new fs. If the downsampling factor is an integer, performs resample_poly.

        Args:
            new_fs: New sampling frequency
            filt: filter output from _create_filter(band). If None it will be set to [0, new_fs/2]
        """
        lcm = np.lcm(int(self.fs), int(new_fs))
        ratio_up = int(lcm / self.fs)
        ratio_down = int(lcm / new_fs)
        self.signal = sig.sosfilt(filt, self.signal)
        self._processed[self.band_n].append("filtered")
        self.signal = sig.resample_poly(self.signal, up=ratio_up, down=ratio_down)
        self._processed[self.band_n].append("downsample")
        self.fs = new_fs

    def downsample2band(self, band: list or tuple) -> None:
        """
        Reduce the sampling frequency. It uses the decimate function of scipy.signal
        In case the ratio is not an int, the closest int is chosen.

        Args:
            band: Band to downsample to (low_freq, high_freq)
        """
        new_fs = band[1] * 2
        if new_fs != self.fs:
            if new_fs > self.fs:
                raise Exception(
                    "This is upsampling, can not downsample %s to %s!"
                    % (self.fs, new_fs)
                )
            filt = self._create_filter(band)
            self.downsample(new_fs, filt)
        else:
            print("trying to downsample to the same fs, ignoring...")

    def filter(self, band: list or tuple) -> None:
        """
        Filter the signal

        Args:
            band: [low_freq, high_freq], band to be filtered
        """
        if band[1] > self._fs / 2:
            raise ValueError(
                "Frequency %s is higher than nyquist frequency %s, and can not be filtered"
                % (band[1], self.fs / 2)
            )
        if not self._band_is_broadband(band):
            # Filter the signal
            sosfilt = self._create_filter(band)
            self.signal = sig.sosfilt(sosfilt, self.signal)
            self._processed[self.band_n].append("filter")

    def remove_dc(self) -> None:
        """
        Remove the dc component of the signal
        """
        dc = np.mean(self.signal)
        self.signal = self.signal - dc
        self._processed[self.band_n].append("dc_removal")

    def envelope(self) -> None:
        """
        Return the envelope of the signal
        """
        analytic_signal = sig.hilbert(self.signal)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope

    def average_envelope(self, window: int) -> tuple:
        """
        Return the average envelope for each window

        Args:
            window: Number of samples for each window to average

        Returns:
            time, envelope
        """
        result = []
        envelope = self.envelope()
        for block in Blocks(signal=envelope, fs=self.fs, blocksize=window):
            val = np.sqrt((block.envelope() ** 2).mean())
            result.append(val)
        result = np.array(result)
        times = np.arange(len(result)) * window / self.fs
        return times, result

    def window_method(self, method_name: str, window: int, **kwargs) -> tuple:
        """
        Return the average envelope for each window

        Args:
            method_name: Name of the function to calculate in each window
            window: Number of samples for each window to average

        Returns:
            time, envelope
        """
        f = operator.methodcaller(method_name, **kwargs)
        result = []
        time = []
        for block in self.blocks(blocksize=window):
            try:
                output = f(block)
            except Exception as e:
                print(
                    "There was an error in feature %s. Setting to None. "
                    "Error: %s" % (method_name, e)
                )
                output = None
            result.append(output)
            time.append(block.time)
        return time, output

    def rms(self, db: bool = True, energy_window: float = None, **kwargs) -> float:
        """
        Calculation of root mean squared value (rms) of the signal in uPa

        Args:
            db : If set to True the result will be given in db, otherwise in uPa
            energy_window: If provided, calculate the rms over the given energy window (e.g. RMS_90 for energy_window= .9).

        Returns:
            rms value
        """
        if energy_window:
            [start, end] = utils.energy_window(self.signal, energy_window)
            rms_val = utils.rms(self.signal[start:end])
        else:
            rms_val = utils.rms(self.signal)
        # Convert it to db if applicable
        if db:
            rms_val = utils.to_db(rms_val, ref=1.0, square=True)
        return rms_val

    def pulse_width(self, energy_window, **kwargs) -> float:
        """
        Returns the pulse width of an impulsive signal
        according to a fractional energy window

        Args:
            energy_window : given energy window to calculate pulse width [0 to 1]

        Returns:
            tau: energy_window pulse width in seconds

        """
        [start, end] = utils.energy_window(self.signal, energy_window)

        return (end - start) / self.fs

    def dynamic_range(self, db: bool = True, **kwargs):
        """
        Compute the dynamic range of each bin
        Returns a dataframe with datetime as index and dr as column

        Args:
            db: If set to True the result will be given in db, otherwise in uPa
        """
        dr = utils.dynamic_range(self.signal)
        # Convert it to db if applicable
        if db:
            dr = utils.to_db(dr, ref=1.0, square=True)
        return dr

    def sel(self, db: bool = True, **kwargs) -> float:
        """
        Calculate the sound exposure level of an event

        Args:
            db: If set to True the result will be given in db, otherwise in uPa

        Returns:
            sel: sound exposure level
        """
        y = utils.sel(self.signal, self.fs)

        if db:
            y = utils.to_db(y, square=False)
        return y

    def peak(self, db: bool = True, **kwargs) -> float:
        """
        Calculate the peak sound exposure level of the signal
        Returns a 2-element array with peak values

        Args:
            db: If set to True the result will be given in db, otherwise in uPa
        """
        y = utils.peak(self.signal)
        if db:
            y = utils.to_db(y, square=True)
        return y

    def kurtosis(self, **kwargs):
        """
        Calculation of kurtosis of the signal
        """
        return utils.kurtosis(self.signal)

    def third_octave_levels(self, db: bool = True, **kwargs) -> tuple:
        """
        Calculation of calibrated 1/3-octave band levels

        Args:
            db: If set to True the result will be given in db, otherwise in uPa

        Returns:
            f: Array with the center frequencies of the bands
            spg: Level of each band
        """
        return self.octave_levels(db, 3)

    def octave_levels(self, db: bool = True, fraction: int = 1, **kwargs) -> np.array:
        """
        Calculation of calibrated octave band levels

        Args:
            db: If set to True the result will be given in db, otherwise in uPa
            fraction: fraction of an octave to compute the bands (i.e. fraction=3 leads to 1/3 octave bands)


        Returns:
            f : Array with the center frequencies of the bands
        """
        bands, f = utils.oct_fbands(
            min_freq=MIN_FREQ, max_freq=self.fs / 2, fraction=fraction
        )

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
                spg[i] = 10 * np.log10(np.sum(y**2) / len(y))
            else:
                spg[i] = y

        return f, spg

    def _spectrogram(
        self, nfft: int = 512, scaling: str = "density", overlap: float = 0.2
    ) -> None:
        """
        Computes the spectrogram of the signal and saves it in the attributes

        Args:
            nfft: Length of the fft window in samples. Power of 2.
            scaling: Can be set to 'spectrum' or 'density' depending on the desired output
            overlap: Percentage to overlap [0 to 1]
        """
        real_size = self.signal.size
        if self.signal.size < nfft:
            self.fill_or_crop(n_samples=nfft)
        window = sig.get_window("hann", nfft)
        noverlap = overlap * nfft
        freq, t, sxx = sig.spectrogram(
            self.signal,
            fs=self.fs,
            nfft=nfft,
            window=window,
            scaling=scaling,
            noverlap=noverlap,
        )
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

    def spectrogram(
        self,
        nfft: int = 512,
        scaling: str = "density",
        overlap: float = 0,
        db: bool = True,
        force_calc: bool = False,
    ) -> tuple:
        """
        Return the spectrogram of the signal (entire file)

        Args:
        db: If set to True the result will be given in db, otherwise in uPa^2
        nfft: Length of the fft window in samples. Power of 2.
        scaling: Can be set to 'spectrum' or 'density' depending on the desired output
        force_calc: Set to True if the computation has to be forced
        overlap: Percentage to overlap [0 to 1]

        Returns:
            freq, t, sxx
        """
        if self.sxx is None or force_calc:
            self._spectrogram(nfft=nfft, scaling=scaling, overlap=overlap)
        if db:
            sxx = utils.to_db(self.sxx, ref=1.0, square=False)
        else:
            sxx = self.sxx
        return self.freq, self.t, sxx

    def _spectrum(
        self,
        scaling: str = "density",
        nfft: int = 512,
        db: bool = True,
        overlap: float = 0,
        window_name: str = "hann",
        **kwargs,
    ):
        """
        Return the spectrum : frequency distribution of all the file (periodogram)
        Returns Dataframe with 'datetime' as index and a colum for each frequency and each
        percentile, and a frequency array

        Args:
            scaling: Can be set to 'spectrum' or 'density' depending on the desired output
            nfft: Length of the fft window in samples. Power of 2. If the signal is shorter it will be
                zero-padded
            db: If set to True the result will be given in db, otherwise in uPa^2
            overlap: Percentage to overlap [0 to 1]
        """
        noverlap = nfft * overlap
        if nfft > self.signal.size:
            self.fill_or_crop(n_samples=nfft)
        window = sig.get_window(window_name, nfft)
        freq, psd = sig.welch(
            self.signal,
            fs=self.fs,
            window=window,
            nfft=nfft,
            scaling=scaling,
            noverlap=noverlap,
            detrend=False,
            **kwargs,
        )
        if self.band is not None and self.band[0] is not None:
            low_freq = np.argmax(freq >= self.band[0])
        else:
            low_freq = 0
        self.psd = psd[low_freq:]
        self.freq = freq[low_freq:]

        if db:
            self.psd = utils.to_db(self.psd, ref=1.0, square=False)

    # TODO implement stft!

    def spectrum(
        self,
        scaling: str = "density",
        nfft: int = 512,
        db: bool = True,
        overlap: float = 0,
        force_calc: float = False,
        percentiles: list = None,
        **kwargs,
    ) -> tuple:
        """
        Return the spectrum : frequency distribution of all the file (periodogram)
        Returns Dataframe with 'datetime' as index and a column for each frequency and
        each percentile, and a frequency array

        Args:
            scaling: Can be set to 'spectrum' or 'density' depending on the desired output
            nfft: Length of the fft window in samples. Power of 2.
            overlap : Percentage to overlap [0 to 1]
            db: If set to True the result will be given in db, otherwise in uPa^2
            force_calc: Set to True if the computation has to be forced
            percentiles: List of all the percentiles that have to be returned. If set to empty list,
                no percentiles is returned

        Returns:
            Frequency array, psd values
        """
        if self.psd is None or force_calc:
            self._spectrum(scaling=scaling, nfft=nfft, db=db, overlap=overlap, **kwargs)
        if percentiles is not None:
            percentiles_val = np.percentile(self.psd, percentiles)
        else:
            percentiles_val = None

        return self.freq, self.psd, percentiles_val

    def spectrum_slope(
        self,
        scaling: str = "density",
        nfft: int = 512,
        db: bool = True,
        overlap: float = 0,
        **kwargs,
    ) -> float:
        """
        Return the slope of the spectrum

        Args:
            scaling: Can be set to 'spectrum' or 'density' depending on the desired output
            nfft : Length of the fft window in samples. Power of 2.
            overlap : Percentage to overlap [0 to 1]
            db: If set to True the result will be given in db, otherwise in uPa^2

        Returns:
            slope of the spectrum (float)
        """
        if self.psd is None:
            self._spectrum(scaling=scaling, nfft=nfft, db=db, overlap=overlap)
        regression = linear_model.LinearRegression().fit(
            np.log10(self.freq), np.log10(self.psd)
        )
        slope = regression.coef_[0]
        y_pred = regression.predict(np.log10(self.freq))
        error = metrics.mean_squared_error(np.log10(self.psd), y_pred)
        return slope, error

    def aci(self, nfft: int = 512, overlap: float = 0, **kwargs) -> float:
        """
        Calculation of root mean squared value (rms) of the signal in uPa for each bin
        Returns Dataframe with 'datetime' as index and 'rms' value as a column

        Args:
            nfft: Number of fft
            overlap: Percentage (in 1) to overlap

        Returns:
            ACI value
        """
        _, _, sxx = self.spectrogram(
            nfft=nfft, scaling="density", overlap=overlap, db=False
        )
        aci_val = self.acoustic_index("aci", sxx=sxx)

        return aci_val

    def bi(
        self,
        min_freq: int = 2000,
        max_freq: int = 8000,
        nfft: int = 512,
        overlap: float = 0,
        **kwargs,
    ) -> float:
        """
        Calculate the Bioacoustic Index index

        Args:
            min_freq: Minimum frequency (in Hertz)
            max_freq: Maximum frequency (in Hertz)
            nfft: FFT number
            overlap : Percentage to overlap [0 to 1]

        Returns:
            BI value
        """
        if self.band[1] < max_freq or self.band[0] > min_freq:
            print(
                "The band %s does not include this band limits (%s, %s). "
                "BI will be set to nan" % (self.band, min_freq, max_freq)
            )
            return np.nan
        else:
            _, _, sxx = self.spectrogram(
                nfft=nfft, scaling="density", overlap=overlap, db=False
            )
            bi_val = self.acoustic_index(
                "bi",
                sxx=sxx,
                frequencies=self.freq,
                min_freq=min_freq,
                max_freq=max_freq,
            )
            return bi_val

    def sh(self, nfft: int = 512, overlap: float = 0, **kwargs) -> float:
        """
        Return the Spectral Entropy of Shannon

        Args:
            nfft: FFT number
            overlap : Percentage to overlap [0 to 1]

        Returns:
            SH index
        """
        _, _, sxx = self.spectrogram(
            nfft=nfft, overlap=overlap, scaling="density", db=False
        )
        sh_val = self.acoustic_index("sh", sxx=sxx)
        return sh_val

    def th(self, **kwargs) -> float:
        """
        Compute Temporal Entropy of Shannon

        Returns:
            TH value
        """
        th_val = self.acoustic_index("th", s=self.signal)
        return th_val

    def ndsi(
        self,
        nfft: int = 512,
        overlap: float = 0,
        anthrophony: tuple = (1000, 2000),
        biophony: tuple = (2000, 11000),
        **kwargs,
    ) -> float:
        """
        Compute the Normalized Difference Sound Index

        Args:
            nfft: FFT number
            overlap : Percentage to overlap [0 to 1]
            anthrophony: Band to consider the anthrophony.
            biophony: Band to consider the biophony.

        Returns:
            NDSI value
        """
        if self.band[1] < anthrophony[1] or self.band[1] < biophony[1]:
            print(
                "The band %s does not include anthrophony %s or biophony %s. "
                "NDSI will be set to nan" % (self.band, anthrophony, biophony)
            )
            return np.nan
        else:
            _, _, sxx = self.spectrogram(
                nfft=nfft, overlap=overlap, scaling="density", db=False
            )
            ndsi_val = self.acoustic_index(
                "ndsi",
                sxx=sxx,
                frequencies=self.freq,
                anthrophony=anthrophony,
                biophony=biophony,
            )
            return ndsi_val

    def aei(
        self,
        db_threshold: int or float = -50,
        freq_step: int = 100,
        nfft: int = 512,
        overlap: float = 0,
        **kwargs,
    ) -> float:
        """
        Compute Acoustic Evenness Index

        Args:
            db_threshold: The minimum db value to consider for the bins of the spectrogram
            freq_step: Size of frequency bands to compute AEI (in Hertz)
            nfft:  FFT number
            overlap : Percentage to overlap [0 to 1]

        Returns:
            AEI value
        """
        _, _, sxx = self.spectrogram(
            nfft=nfft, scaling="density", overlap=overlap, db=False
        )
        aei_val = self.acoustic_index(
            "aei",
            sxx=sxx,
            frequencies=self.freq,
            max_freq=self.band[1],
            min_freq=self.band[0],
            db_threshold=db_threshold,
            freq_step=freq_step,
        )
        return aei_val

    def adi(
        self,
        db_threshold: int or float = -50,
        freq_step: int = 100,
        nfft: int = 512,
        overlap: float = 0,
        **kwargs,
    ) -> float:
        """
        Compute Acoustic Diversity Index

        Args:
            db_threshold: The minimum db value to consider for the bins of the spectrogram
            freq_step: Size of frequency bands to compute AEI (in Hertz)
            nfft: FFT number
            overlap: Percentage to overlap [0 to 1]

        Returns:
            ADI value
        """
        _, _, sxx = self.spectrogram(
            nfft=nfft, scaling="density", overlap=overlap, db=False
        )
        adi_val = self.acoustic_index(
            "adi",
            sxx=sxx,
            frequencies=self.freq,
            max_freq=self.band[1],
            min_freq=self.band[0],
            db_threshold=db_threshold,
            freq_step=freq_step,
        )
        return adi_val

    def zcr(self, **kwargs) -> float:
        """
        Compute the Zero Crossing Rate

        Returns:
            A list of values (number of zero crossing for each window)
        """
        zcr = self.acoustic_index("zcr", s=self.signal, fs=self.fs)
        return zcr

    def zcr_avg(
        self, window_length: int = 512, window_hop: int = 256, **kwargs
    ) -> list or np.ndarray:
        """
        Zero Crossing Rate average

        Args:
            window_length: Size of the sliding window (samples)
            window_hop: Size of the lag window (samples)

        Returns:
            ZCR average (list)
        """
        zcr = self.acoustic_index(
            "zcr_avg",
            s=self.signal,
            fs=self.fs,
            window_length=window_length,
            window_hop=window_hop,
        )
        return zcr

    def bn_peaks(
        self,
        freqband: int or float = 200,
        normalization: bool = True,
        slopes: tuple = (0.01, 0.01),
        nfft: int = 512,
        overlap: float = 0,
        **kwargs,
    ) -> int:
        """
        Counts the number of major frequency peaks obtained on a mean spectrum.

        Args:
            freqband: frequency threshold parameter (in Hz). If the frequency difference of two successive peaks
                is less than this threshold, then the peak of highest amplitude will be kept only.
                normalization: if set at True, the mean spectrum is scaled between 0 and 1
            normalization: Set to true if normalization is desired
            slopes: Amplitude slope parameter, a tuple of length 2. Refers to the amplitude slopes of the peak.
                The first value is the left slope and the second value is the right slope. Only peaks with
                higher slopes than threshold values will be kept. i.e (0.01, 0.01)
            nfft: FFT number
            overlap : Percentage to overlap [0 to 1]

        Returns:
            Number of BN peaks
        """
        _, _, sxx = self.spectrogram(
            nfft=nfft, overlap=overlap, scaling="density", db=False
        )
        frequencies = self.freq
        meanspec = sxx.mean(axis=1)

        if normalization:
            meanspec = np.array(meanspec) / np.max(meanspec)

        if slopes is not None:
            # Find peaks (with slopes)
            peaks_indices = (
                np.r_[
                    False,
                    meanspec[1:] > np.array([x + slopes[0] for x in meanspec[:-1]]),
                ]
                & np.r_[
                    meanspec[:-1] > np.array([y + slopes[1] for y in meanspec[1:]]),
                    False,
                ]
            )
            peaks_indices = peaks_indices.nonzero()[0].tolist()
        else:
            # scipy method (without slope)
            peaks_indices = sig.argrelextrema(np.array(meanspec), np.greater)[
                0
            ].tolist()

        # Remove peaks with difference of frequency < freqband
        # number of consecutive index
        nb_bin = next(i for i, v in enumerate(frequencies) if v > freqband)
        for consecutiveIndices in [np.arange(i, i + nb_bin) for i in peaks_indices]:
            if len(np.intersect1d(consecutiveIndices, peaks_indices)) > 1:
                # close values has been found
                maxi, _, _ = np.intersect1d(consecutiveIndices, peaks_indices)
                maxi = maxi[
                    np.argmax(
                        [
                            meanspec[f]
                            for f in np.intersect1d(consecutiveIndices, peaks_indices)
                        ]
                    )
                ]
                peaks_indices = [
                    x for x in peaks_indices if x not in consecutiveIndices
                ]
                # remove all indices that are in consecutiveIndices
                # append the max
                peaks_indices.append(maxi)
        peaks_indices.sort()

        # Frequencies of the peaks
        peak_freqs = [frequencies[p] for p in peaks_indices]
        return len(peaks_indices), peak_freqs

    def total_correlation(self, signal: np.array) -> float:
        """
        Compute the correlation with the signal

        Args:
            signal: Signal to be correlated with

        Returns:
            correlation coefficient
        """
        if isinstance(signal, Signal):
            if signal.fs > self.fs:
                signal.downsample(self.fs)
            elif signal.fs < self.fs:
                self.downsample(signal.fs)
        coeff = np.corrcoef(self.signal, signal.signal)

        return coeff

    def blocks_correlation(self, signal: np.array) -> float:
        """
        Compute the correlation with the signal for each block of the same length than the signal

        Args:
            signal: Signal to be correlated with

        Returns:
            correlation coefficient list per each block
        """
        coeff_evo = []
        for block in self.blocks(blocksize=signal.size):
            coeff_evo.append(np.corrcoef(block.signal, signal))
        return coeff_evo

    def sel_spectrum(self, spg: np.array, dt: float) -> np.array:
        """
        Calculation of total spectrum (SEL) of the calibrated spectrogram

        Args:
            spg: Array with in each cell the spectrogram of a single channel of the input signal
            dt: timestep of the spectrogram calculation, in seconds

        Returns:
            numpy matrix with in each cell the spectrum of a single channel of the input signal
        """
        y = []
        for spg_i in spg:
            y.append(10.0 * np.log10(sum(10.0 ** (spg_i / 10.0), 1) * dt))
        return y

    def average_spectrum(self, spg: np.array) -> np.array:
        """
        Calculation of average spectrum (Leq) of the calibrated spectrogram


        Args:
            spg: Array with in each cell the spectrogram of a single channel of the input signal

        Returns:
             numpy array with in each cell the spectrum of a single channel of the input signal
        """
        y = []
        for spg_i in spg:
            y.append(10.0 * np.log10(np.mean(10.0 ** (spg_i / 10.0), 1)))
        return y

    def spectrogram_third_bands(self, dt: float) -> tuple:
        """
        Calculation of calibrated 1/3-octave band spectrogram for 28 bands from 25 Hz to 12.5 kHz

        Args:
            dt: Timestep for calculation of spectrogram, in seconds

        Returns:
            t: Array with the time values of the spectrogram, in seconds
            f: Array with the frequency values of the spectrogram
            spg: Array with in each cell the spectrogram of a single channel of the input signal
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
                spg[k, j] = 10 * np.log10(np.sum(z**2) / len(z))

        return t, f, spg

    @staticmethod
    def acoustic_index(name: str, **kwargs):
        """
        Return the acoustic index

        Args:
            name: Name of the Acoustic Index to compute
        """
        f = getattr(acoustic_indices, "compute_" + name)
        return f(**kwargs)

    def reduce_noise(self, nfft: int = 512, verbose: bool = False) -> None:
        """
        Remove the noise of the signal using the noise clip

        Args:
            nfft: Window size to compute the spectrum
            verbose: Set to True to plot the signal before and after the reduction
        """
        s = nr.reduce_noise(y=self.signal, sr=self.fs, n_fft=nfft, win_length=nfft)
        if verbose:
            db = True
            label, units = output_units.get_units("spectrogram", log=db, p_ref=1.0)
            label_a, units_a = output_units.get_units("amplitude", log=False)
            _, _, sxx0 = self.spectrogram(nfft, db=db, force_calc=True)
            fig, ax = plt.subplots(
                2, 3, gridspec_kw={"width_ratios": [1, 1, 0.05]}, sharex="col"
            )
            ax[0, 0].plot(self.times, self.signal)
            ax[0, 0].set_title("Signal")
            ax[0, 0].set_ylabel(r"% [%s]" % (label_a, units_a))
            ax[0, 1].set_axis_off()
            im = ax[0, 1].pcolormesh(
                self.t,
                self.freq,
                sxx0,
                vmin=60,
                vmax=150,
                shading="auto",
                cmap="viridis",
            )
            plt.colorbar(im, cax=ax[0, 2], label=r"%s [%s]" % (label, units))
            ax[0, 1].set_title("Spectrogram")

            self.signal = s
            _, _, sxx1 = self.spectrogram(nfft, db=True, force_calc=True)
            ax[1, 0].plot(self.times, s)
            ax[1, 0].set_ylabel(r"% [%s]" % (label_a, units_a))
            ax[1, 0].set_xlabel("Time [s]")
            ax[1, 1].set_axis_off()
            ax[1, 1].set_xlabel("Time [s]")
            im = ax[1, 1].pcolormesh(
                self.t,
                self.freq,
                sxx1,
                vmin=60,
                vmax=150,
                shading="auto",
                cmap="viridis",
            )
            plt.colorbar(im, cax=ax[1, 2], label=r"%s [%s]" % (label, units))
            plt.show()
            plt.close()
        else:
            self.signal = s

        self._processed[self.band_n].append("noisereduction")

    def plot(
        self,
        nfft: int = 512,
        overlap: float = 0,
        scaling: str = "density",
        db: bool = True,
        force_calc: bool = False,
        show: bool = False,
        save_path: str or pathlib.Path = None,
        vmin: float = None,
        vmax: float = None,
        log: bool = False,
    ) -> None:
        """
        Plot the signal and its spectrogram

        Args:
            nfft: nfft value
            scaling: 'density' or 'spectrum'
            db: Set to True for dB output
            force_calc: Set to True to force the re-calulation of the spectrogram
            overlap: Percentage to overlap in windows for the plot [0 to 1]
            show: Set to True to show
            save_path: Where to save the output. Set to None to not save if (default)
            vmin: minimum value to plot in the spectrogram
            vmax: maximum value to plot in the spectrogram
        """
        plt.rcParams.update(plt.rcParamsDefault)
        _, _, sxx = self.spectrogram(
            nfft=nfft, scaling=scaling, overlap=overlap, db=db, force_calc=force_calc
        )

        label, units = output_units.get_units("spectrum_" + scaling, log=db, p_ref=1.0)
        label_a, units_a = output_units.get_units("amplitude", log=False)

        fig, ax = plt.subplots(
            2, 2, gridspec_kw={"width_ratios": [1, 0.05]}, sharex="col"
        )
        ax[0, 0].plot(self.times, self.signal, color="k", linewidth=1)
        ax[0, 0].set_title("Signal")
        ax[0, 0].set_ylabel(r"% [%s]" % (label_a, units_a))
        ax[0, 1].set_axis_off()
        if vmin is None:
            vmin = sxx.min()
        if vmax is None:
            vmax = sxx.max()
        im = ax[1, 0].pcolormesh(
            self.t, self.freq, sxx, vmin=vmin, vmax=vmax, shading="auto", cmap="magma"
        )
        plt.colorbar(im, cax=ax[1, 1], label=label)
        if log:
            ax[1, 0].set_yscale("symlog")
        ax[1, 0].set_title("Spectrogram")
        ax[1, 0].set_xlabel("Time [s]")
        ax[1, 0].set_ylabel("Frequency [Hz]")
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    def blocks(self, blocksize: int):
        """
        Wrapper for the Blocks class

        Args:
            blocksize:  Window integration time, in samples
        """
        return Blocks(self.signal, self.fs, blocksize)


class Blocks:
    def __init__(self, signal: np.array, fs: int, blocksize: int):
        """
        Args:
            signal: Signal to process
            fs: Sample rate
            blocksize: Window integration time, in samples
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
        if item == "time":
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
            block = self.signal[
                self.n * self.blocksize : self.n * self.blocksize + self.blocksize
            ]
            self.n += 1
            s = Signal(block, self.fs)
            return s
        else:
            raise StopIteration
