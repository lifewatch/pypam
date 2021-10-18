"""
Module : acoustic_file.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Institute voor de Zee)
"""
__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

import datetime
import operator
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sci
import scipy.integrate as integrate
import seaborn as sns
import soundfile as sf
import xarray
from tqdm.auto import tqdm

from pypam import impulse_detector
from pypam import loud_event_detector
from pypam import nmf
from pypam import utils
from pypam.signal import Signal

pd.plotting.register_matplotlib_converters()

# Apply the default theme
sns.set_theme()


class AcuFile:
    def __init__(self, sfile, hydrophone, p_ref, band=None, utc=True, channel=0,
                 calibration_time=0.0, max_cal_duration=60.0, cal_freq=250, dc_subtract=False):
        """
        Data recorded in a wav file.

        Parameters
        ----------
        sfile : Sound file
            Can be a path or an file object
        hydrophone : Object for the class hydrophone
        p_ref : Float
            Reference pressure in upa
        band : list
            Band to filter
        utc : boolean
            Set to True if working on UTC and not localtime
        channel : int
            Channel to perform the calculations in
        calibration_time: float or str
            If a float, the amount of seconds that are ignored at the beggning of the file. If 'auto' then
            before the analysis, find_calibration_tone will be performed
        max_cal_duration: float
            Maximum time in seconds for the calibration tone (only applies if calibration_time is 'auto')
        cal_freq: float
            Frequency of the calibration tone (only applies if calibration_time is 'auto')
        dc_subtract: bool
            Set to True to subtract the dc noise (root mean squared value
        """
        # Save hydrophone model
        self.hydrophone = hydrophone

        # Get the date from the name
        if type(sfile) == str:
            file_name = os.path.split(sfile)[-1]
        elif issubclass(sfile.__class__, pathlib.Path):
            file_name = sfile.name
        else:
            raise Exception('The filename has to be either a Path object or a string')

        try:
            self.date = hydrophone.get_name_datetime(file_name, utc=utc)
        except ValueError:
            self.date = datetime.datetime.now()
            print('Filename %s does not match the %s file structure. Setting time to now...' %
                  (file_name, self.hydrophone.name))

        # Signal
        self.file_path = sfile
        self.file = sf.SoundFile(self.file_path)
        self.fs = self.file.samplerate

        # Reference pressure in upa
        self.p_ref = p_ref

        # Band selected to study
        self.band = band

        # Work on local time or UTC time
        self.utc = utc

        # Select channel
        self.channel = channel

        # Set an empty wav array
        self.wav = None
        self.time = None

        # Set a starting frame for the file
        self.cal_freq = cal_freq
        self.max_cal_duration = max_cal_duration
        if calibration_time == 'auto':
            start_cal, self._start_frame, calibration_signal = self.find_calibration_tone()
            if self._start_frame is None:
                self._start_frame = 0
            else:
                self.hydrophone.update_calibration(calibration_signal)
        else:
            self._start_frame = int(calibration_time * self.fs)

        self.dc_subtract = dc_subtract

    def __getattr__(self, name):
        """
        Specific methods to make it easier to access attributes
        """
        if name == 'signal':
            return self.signal('upa')
        elif name == 'time':
            if self.__dict__[name] is None:
                return self._time_array(blocksize=1)[0]
            else:
                return self.__dict__[name]
        else:
            return self.__dict__[name]

    def _bins(self, blocksize=None):
        """
        Yields the bins each blocksize
        Parameters
        ----------
        blocksize: int or None
            Number of frames per bin to yield

        Returns
        -------
        Iterates through all the bins, yields i, time_bin, signal
        Where i is the index, time_bin is the datetime of the beggning of the block and signal is the signal object
        of the bin
        """
        if blocksize is None:
            n_blocks = 1
        else:
            n_blocks = self._n_blocks(blocksize)
        for i, block in tqdm(enumerate(sf.blocks(self.file_path, blocksize=blocksize, start=self._start_frame)),
                             total=n_blocks, leave=False, position=0):
            time_bin = self.time_bin_i(blocksize, i)
            # Read the signal and prepare it for analysis
            signal_upa = self.wav2upa(wav=block)
            signal = Signal(signal=signal_upa, fs=self.fs, channel=self.channel)
            if self.dc_subtract:
                signal.remove_dc()
            yield i, time_bin, signal
        self.file.seek(0)

    def _get_fbands(self, band, nfft, downsample=True):
        if band is not None:
            if downsample:
                fs = band[1] * 2
            else:
                fs = self.fs
        else:
            fs = self.fs
        return sci.fft.rfftfreq(nfft) * fs

    def _n_blocks(self, blocksize):
        return int(np.ceil((sf.SoundFile(self.file_path).frames - self._start_frame) / blocksize))

    def samples(self, bintime):
        """
        Return the samples according to the fs

        Parameters
        ----------
        bintime : Float
            Seconds of bintime desired to convert to samples
        """
        return int(bintime * self.fs)

    def time_bin_i(self, blocksize, i):
        """
        Return the datetime of the bin i with a bin size of blocksize samples

        Parameters
        ----------
        blocksize : int
            Number of samples in each bin
        i : int
            Index of the bin to get the time of

        Returns
        -------
        datetime object
        """
        return self.date + datetime.timedelta(seconds=(((blocksize * i) + self._start_frame) / self.fs))

    def set_calibration_time(self, calibration_time):
        """
        Set a calibration time in seconds. This time will be ignored in the processing

        Parameters
        ----------
        calibration_time : float
            Seconds to ignore at the beggining of the file
        """
        self._start_frame = int(calibration_time * self.fs)

    def instrument(self):
        """
        Instrument will be the name of the hydrophone
        """
        return self.hydrophone.name

    def total_time(self):
        """
        Return the total time in seconds of the file
        """
        return self.samples2time(self.file.frames)

    def samples2time(self, samples):
        """
        Return the samples according to the fs

        Parameters
        ----------
        samples : Int
            Number of samples to convert to seconds
        """
        return float(samples) / self.fs

    def is_in_period(self, period):
        """
        Return True if the WHOLE file is included in the specified period

        Parameters
        ----------
        period : list or a tuple with (start, end).
            Values have to be a datetime object
        """
        if period is None:
            return True
        else:
            return (self.date >= period[0]) & (self.date <= period[1])

    def contains_date(self, date):
        """
        Return True if data is included in the file

        Parameters
        ----------
        date : datetime object
            Datetime to check
        """
        end = self.date + datetime.timedelta(seconds=self.file.frames / self.fs)
        return (self.date < date) & (end > date)

    def split(self, date):
        """
        Save two different files out of one splitting on the specified date

        Parameters
        ----------
        date : datetime object
            Datetime where to split the file
        """
        if not self.contains_date(date):
            raise Exception('This date is not included in the file!')
        else:
            self.file.seek(0)
        seconds = (date - self.date).seconds
        frames = self.samples(seconds)
        first_file = self.file.read(frames=frames)
        second_file = self.file.read()
        self.file.close()

        new_file_name = self.hydrophone.get_new_name(filename=self.file_path.name, new_date=date)
        new_file_path = self.file_path.parent.joinpath(new_file_name)
        sf.write(self.file_path, first_file, samplerate=self.fs)
        sf.write(new_file_path, second_file, samplerate=self.fs)

        return self.file_path, new_file_path

    def freq_resolution_window(self, freq_resolution):
        """
        Given the frequency resolution, window length needed to obtain it
        Returns window lenght in samples

        Parameters
        ----------
        freq_resolution : int
            Must be a power of 2, in Hz
        """
        n = np.log2(self.fs / freq_resolution)
        nfft = 2 ** n
        if nfft > self.file.frames:
            raise Exception('This is not achievable with this sampling rate, '
                            'it must be downsampled!')
        return nfft

    def signal(self, units='wav'):
        """
        Returns the signal in the specified units

        Parameters
        ----------
        units : string
            Units in which to return the signal. Can be 'wav', 'db', 'upa', 'Pa' or 'acc'.
        """
        # First time, read the file and store it to not read it over and over
        if self.wav is None:
            self.wav = self.file.read()
            self.file.seek(0)
        if units == 'wav':
            signal = self.wav
        elif units == 'db':
            signal = self.wav2db()
        elif units == 'upa':
            signal = self.wav2upa()
        elif units == 'Pa':
            signal = self.wav2upa() / 1e6
        elif units == 'acc':
            signal = self.wav2acc()
        else:
            raise Exception('%s is not implemented as an outcome unit' % units)

        return signal

    def _time_array(self, blocksize=1):
        """
        Return a time array for each point of the signal
        """
        blocks_samples = np.arange(start=self._start_frame, stop=self.file.frames, step=blocksize)
        start_samples = blocks_samples * blocksize
        end_samples = start_samples + blocksize
        incr = pd.to_timedelta(blocks_samples * blocksize / self.fs, unit='seconds')
        self.time = self.date + datetime.timedelta(seconds=self._start_frame / self.fs) + incr
        return self.time, start_samples.astype(int), end_samples.astype(int)

    def wav2upa(self, wav=None):
        """
        Compute the pressure from the wav signal

        Parameters
        ----------
        wav : ndarray
            Signal in wav (-1 to 1)
        """
        # Read if no signal is passed
        if wav is None:
            wav = self.signal('wav')
        # First convert it to Volts and then to Pascals according to sensitivity
        mv = 10 ** (self.hydrophone.sensitivity / 20.0) * self.p_ref
        ma = 10 ** (self.hydrophone.preamp_gain / 20.0) * self.p_ref
        gain_upa = (self.hydrophone.Vpp / 2.0) / (mv * ma)

        return utils.set_gain(wave=wav, gain=gain_upa)

    def wav2db(self, wav=None):
        """
        Compute the db from the wav signal. Consider the hydrophone sensitivity in db.
        If wav is None, it will read the whole file.

        Parameters
        ----------
        wav : ndarray
            Signal in wav (-1 to 1)
        """
        # Read if no signal is passed
        if wav is None:
            wav = self.signal('wav')
        upa = self.wav2upa(wav)
        return utils.to_db(wave=upa, ref=self.p_ref, square=True)

    def db2upa(self, db=None):
        """
        Compute the upa from the db signals. If db is None, it will read the whole file.

        Parameters
        ----------
        db : ndarray
            Signal in db
        """
        if db is None:
            db = self.signal('db')
        # return np.power(10, db / 20.0 - np.log10(self.p_ref))
        return utils.to_mag(wave=db, ref=self.p_ref)

    def upa2db(self, upa=None):
        """
        Compute the db from the upa signal. If upa is None, it will read the whole file.

        Parameters
        ----------
        upa : ndarray
            Signal in upa
        """
        if upa is None:
            upa = self.signal('upa')
        return utils.to_db(upa, ref=self.p_ref, square=True)

    def wav2acc(self, wav=None):
        """
        Convert the wav file to acceleration. If wav is None, it will read the whole file.

        Parameters
        ----------
        wav : ndarray
            Signal in wav (-1 to 1)
        """
        if wav is None:
            wav = self.file.read()
        mv = 10 ** (self.hydrophone.mems_sensitivity / 20.0)
        return wav / mv

    def _get_metadata_attrs(self):
        metadata_keys = ['hydrophone.name',
                         'hydrophone.model',
                         'hydrophone.sensitivity',
                         'hydrophone.preamp_gain',
                         'hydrophone.Vpp',
                         'date',
                         'file_path',
                         'band',
                         'utc',
                         'cal_freq',
                         'p_ref',
                         'channel',
                         '_start_frame',
                         'max_cal_duration',
                         'dc_subtract'
                         ]
        metadata_attrs = {}
        for k in metadata_keys:
            d = self
            for sub_k in k.split('.'):
                d = d.__dict__[sub_k]
            metadata_attrs[k] = d

        return metadata_attrs

    def _create_dataset(self, data, freq, blocksize, extra_coords=None, extra_attrs=None):
        time_array, start_sample, end_sample = self._time_array(blocksize)
        coords = {'datetime': time_array,
                  'bands': np.arange(len(freq)),
                  'start_sample': ('datetime', start_sample),
                  'end_sample': ('datetime', end_sample)
                  }
        attrs = self._get_metadata_attrs()
        if extra_coords is not None:
            coords.update(extra_coords)
        if extra_attrs is not None:
            attrs.update(extra_attrs)
        ds = xarray.Dataset(data_vars=data,
                            coords=coords,
                            attrs=attrs)
        return ds

    def _apply_multiple(self, method_list, band_list=None, binsize=None, **kwargs):
        """
        Apply multiple methods per bin to save computational time

        Parameters
        ----------
        method_list: list of strings
            List of all the methods to apply
        band_list: list of tuples (or lists)
            List of all the bands to analyze. If set to None, broadband is analyzed
        binsize: float
            Lenght in seconds of the bins to analyze
        kwargs: any parameters that have to be passed to the methods

        Returns
        -------
        DataFrame with time as index and a multiindex column with band, method as levels.
        """
        # TODO decide if it is downsampled or not
        downsample = False
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)

        if band_list is None:
            band_list = [self.band]
        # Sort bands to diminish downsampling efforts!
        sorted_bands = []
        low_freqs = []
        high_freqs = []
        for band in band_list:
            low_freqs.append(band[0])
            high_freqs.append(band[1])
            if len(sorted_bands) == 0:
                sorted_bands = [band]
            else:
                if band[1] >= sorted_bands[-1][1]:
                    sorted_bands = [band] + sorted_bands
                else:
                    sorted_bands = sorted_bands + [band]

        # Define the array dimensions
        methods_output = {}
        n_blocks = self._n_blocks(blocksize)
        for m in method_list:
            methods_output[m] = (['datetime', 'bands'], np.zeros((n_blocks, len(sorted_bands))))

        for i, time_bin, signal in self._bins(blocksize):
            for j, band in enumerate(sorted_bands):
                signal.set_band(band, downsample=downsample)
                for method_name in method_list:
                    f = operator.methodcaller(method_name, **kwargs)
                    try:
                        output = f(signal)
                    except Exception as e:
                        print('There was an error in band %s, feature %s. Setting to None. '
                              'Error: %s' % (band, method_name, e))
                        output = None
                    methods_output[method_name][1][i, j] = output

        ds = self._create_dataset(data=methods_output, freq=np.arange(len(sorted_bands)),
                                  extra_coords={'band_lowfreq': ('bands', low_freqs),
                                                'band_highfreq': ('bands', high_freqs)},
                                  blocksize=blocksize, extra_attrs={'downsample': downsample})
        return ds

    def _apply(self, method_name, binsize=None, db=True, band_list=None, **kwargs):
        """
        Apply one single method

        Parameters
        ----------
        method_name : string
            Name of the method to apply
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        db : bool
            If set to True the result will be given in db, otherwise in upa
        """
        return self._apply_multiple(method_list=[method_name], binsize=binsize,
                                    db=db, band_list=band_list, **kwargs)

    def rms(self, binsize=None, db=True):
        """
        Calculation of root mean squared value (rms) of the signal in upa for each bin
        Returns Dataframe with 'datetime' as index and 'rms' value as a column

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        db : bool
            If set to True the result will be given in db, otherwise in upa
        """
        rms_ds = self._apply(method_name='rms', binsize=binsize, db=db)
        return rms_ds

    def aci(self, binsize=None, nfft=1024):
        """
        Calculation of root mean squared value (rms) of the signal in upa for each bin
        Returns Dataframe with 'datetime' as index and 'rms' value as a column

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        nfft : int
            Window size for processing
        """
        aci_ds = self._apply(method_name='aci', binsize=binsize, nfft=nfft)
        return aci_ds

    def dynamic_range(self, binsize=None, db=True):
        """
        Compute the dynamic range of each bin
        Returns a dataframe with datetime as index and dr as column

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        db : bool
            If set to True the result will be given in db, otherwise in upa
        """
        dr_ds = self._apply(method_name='dynamic_range', binsize=binsize, db=db)
        return dr_ds

    def cumulative_dynamic_range(self, binsize=None, db=True):
        """
        Compute the cumulative dynamic range for each bin

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        db : bool
            If set to True the result will be given in db, otherwise in upa^2

        Returns
        -------
        DataFrame with an extra column with the cumulative sum of dynamic range of each bin
        """
        cumdr = self.dynamic_range(binsize=binsize, db=db)
        cumdr['cumsum_dr'] = cumdr.dr.cumsum()
        return cumdr

    def octaves_levels(self, binsize=None, db=True, band=None, **kwargs):
        """
        Return the octave levels
        Parameters
        ----------
        binsize: float
            Lenght in seconds of the bin to analyze
        db: boolean
            Set to True if the result should be in decibels
        band: list or tuple
            List or tuple of [low_frequency, high_frequency]

        Returns
        -------
        DataFrame with multiindex columns with levels method and band. The method is '3-oct'

        """
        return self._octaves_levels(fraction=1, binsize=binsize, db=db, band=band)

    def third_octaves_levels(self, binsize=None, db=True, band=None, **kwargs):
        """
        Return the octave levels
        Parameters
        ----------
        binsize: float
            Lenght in seconds of the bin to analyze
        db: boolean
            Set to True if the result should be in decibels
        band: list or tuple
            List or tuple of [low_frequency, high_frequency]

        Returns
        -------
        DataFrame with multiindex columns with levels method and band. The method is '3-oct'

        """
        return self._octaves_levels(fraction=3, binsize=binsize, db=db, band=band)

    def _octaves_levels(self, fraction=1, binsize=None, db=True, band=None):
        """
        Return the octave levels
        Parameters
        ----------
        fraction: int
            Fraction of the desired octave. Set to 1 for octave bands, set to 3 for 1/3-octave bands
        binsize: float
            Lenght in seconds of the bin to analyze
        db: boolean
            Set to True if the result should be in decibels
        band: list or tuple
            List or tuple of [low_frequency, high_frequency]

        Returns
        -------
        DataFrame with multiindex columns with levels method and band. The method is '3-oct'

        """
        downsample = True
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)

        bands = np.arange(-16, 11)
        bands = 1000 * ((2 ** (1 / 3)) ** bands)
        n_blocks = self._n_blocks(blocksize)
        oct_str = 'oct%s' % fraction
        # Define the array dimensions
        methods_output = {oct_str: (['datetime', 'bands'], np.zeros((n_blocks, len(bands))))}
        for i, time_bin, signal in self._bins(blocksize):
            signal.set_band(band, downsample=True)
            _, levels = signal.octave_levels(db, fraction)
            methods_output[oct_str][1][i, :] = levels

        ds = self._create_dataset(data=methods_output, freq=bands, blocksize=blocksize,
                                  extra_attrs={'downsample': downsample})
        return ds

    def spectrogram(self, binsize=None, nfft=512, scaling='density', db=True, mode='fast'):
        """
        Return the spectrogram of the signal (entire file)

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        nfft : int
            Length of the fft window in samples. Power of 2.
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        mode : string
            If set to 'fast', the signal will be zero padded up to the closest power of 2

        Returns
        -------
        time : ndarray
            Array with the starting time of each bin
        freq : ndarray
            Array with all the frequencies
        t : ndarray
            Time array in seconds of the windows of the spectrogram
        sxx_list : list
            Spectrogram list, one for each bin
        """
        downsample = True
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = self.samples(binsize)

        methods_output = {'spectrogram': (['datetime', 'bands'], [])}
        for i, time_bin, signal in self._bins(blocksize):
            signal.set_band(self.band, downsample=downsample)
            freq, t, sxx = signal.spectrogram(nfft=nfft, scaling=scaling, db=db, mode=mode)
            methods_output['spectrogram'][1].append = sxx

        methods_output['spectrogram'][1] = np.array(methods_output['spectrogram'][1])
        ds = self._create_dataset(data=methods_output, freq=freq,
                                  extra_coords={'bin_time': t},
                                  blocksize=blocksize,
                                  extra_attrs={'downsample': downsample})
        return ds

    def _spectrum(self, scaling='density', binsize=None, nfft=512, db=True, percentiles=None):
        """
        Return the spectrum : frequency distribution of all the file (periodogram)
        Returns Dataframe with 'datetime' as index and a colum for each frequency and each
        percentile, and a frequency array

        Parameters
        ----------
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        nfft : int
            Lenght of the fft window in samples. Power of 2.
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned
        """
        downsample = True
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)
        if percentiles is None:
            percentiles = []

        fbands = self._get_fbands(self.band, nfft, downsample=True)
        spectrum_str = 'band_' + scaling
        n_blocks = self._n_blocks(blocksize)
        spectrum_ds = {spectrum_str: (['datetime', 'bands'], np.zeros((n_blocks, len(fbands))))}
        for i, time_bin, signal in self._bins(blocksize):
            signal.set_band(band=self.band, downsample=downsample)
            fbands, spectra = signal.spectrum(scaling=scaling, nfft=nfft, db=db,
                                              percentiles=percentiles)
            spectrum_ds[spectrum_str][1][i, :] = spectra

        ds = self._create_dataset(data=spectrum_ds, freq=fbands, blocksize=blocksize,
                                  extra_attrs={'downsample': downsample})

        # TODO Add percentiles!
        return ds

    def psd(self, binsize=None, nfft=512, db=True, percentiles=None):
        """
        Return the power spectrogram density (PSD) of all the file (units^2 / Hz) re 1 V 1 upa
        Returns a Dataframe with 'datetime' as index and a colum for each frequency and each
        percentile

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        nfft : int
            Lenght of the fft window in samples. Power of 2.
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned
        """
        psd_df = self._spectrum(scaling='density', binsize=binsize, nfft=nfft, db=db,
                                percentiles=percentiles)
        return psd_df

    def power_spectrum(self, binsize=None, nfft=512, db=True, percentiles=None):
        """
        Return the power spectrogram density of all the file (units^2 / Hz) re 1 V 1 upa
        Returns a Dataframe with 'datetime' as index and a colum for each frequency and
        each percentile

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        nfft : int
            Lenght of the fft window in samples. Power of 2.
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned
        """

        spectrum_df = self._spectrum(scaling='spectrum', binsize=binsize, nfft=nfft, db=db,
                                     percentiles=percentiles)
        return spectrum_df

    def spd(self, binsize=None, h=0.1, nfft=512, db=True, percentiles=None):
        """
        Return the spectral probability density.

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        h : float
            Histogram bin width (in the correspondent units, upa or db)
        nfft : int
            Lenght of the fft window in samples. Power of 2.
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : array_like
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned

        Returns
        -------
        time : ndarray
            list with the starting point of each spd df
        fbands : ndarray
            list of all the frequencies
        percentiles : list of float
            Percentiles to compute
        edges_list : list of float
            list of the psd values of the distribution
        spd_list : list of ndarray
            list of dataframes with 'frequency' as index and a column for each psd bin and
            for each percentile (one df per bin)
        p_list : list of 2d ndarray
            list of matrices with all the probabilities
        """
        time, fbands, t, sxx_list = self.spectrogram(binsize=binsize, nfft=nfft, db=db,
                                                     scaling='density')
        spd_list = []
        p_list = []
        edges_list = []
        if percentiles is None:
            percentiles = []
        for sxx in sxx_list:
            # Calculate the bins of the psd values and compute spd using numba
            bin_edges = np.arange(start=sxx.min(), stop=sxx.max(), step=h)
            spd, p = utils.sxx2spd(sxx=sxx, h=h, percentiles=np.array(percentiles) / 100.0,
                                   bin_edges=bin_edges)
            spd_list.append(spd)
            p_list.append(p)
            edges_list.append(bin_edges)

        return time, fbands, percentiles, edges_list, spd_list, p_list

    def correlation(self, signal, fs_signal, binsize=1.0):
        """
        Compute the correlation with the signal

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        signal : ndarray
            Signal to be correlated with
        fs_signal : int
            Sampling frequency of the signal. It will be down/up sampled in case it does not match
            with the file sampling frequency
        """
        # TODO
        fs = 1
        return fs

    def detect_piling_events(self, min_separation, max_duration, threshold, dt, binsize=None,
                             verbose=False, save_path=None, band=None, method=None):
        """
        Detect piling events

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        min_separation : float
            Minimum separation of the event, in seconds
        max_duration : float
            Maximum duration of the event, in seconds
        threshold : float
            Threshold above ref value which one it is considered piling, in db
        dt : float
            Window size in seconds for the analysis (time resolution). Has to be smaller
            than min_duration!
        verbose : bool
            Set to True to get plots of the detections
        save_path: Path or str
            Path where to save the images of the detections
        band : list or tuple
            Band to analyze [low_freq, high_freq]
        method : str
            Method to use for the detection. Can be 'snr', 'dt' or 'envelope'
        """
        if type(save_path) == str:
            save_path = pathlib.Path(save_path)
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)

        detector = impulse_detector.PilingDetector(min_separation=min_separation,
                                                   max_duration=max_duration,
                                                   threshold=threshold, dt=dt, detection_band=band,
                                                   analysis_band=self.band)
        total_events = pd.DataFrame()
        for _, time_bin, signal in self._bins(blocksize):
            signal.set_band(band=self.band, downsample=False)
            if save_path is not None:
                file_path = save_path.joinpath('%s.png' % datetime.datetime.strftime(time_bin, "%y%m%d_%H%M%S"))
            else:
                file_path = None
            events_df = detector.detect_events(signal, method=method, verbose=verbose,
                                               save_path=file_path)
            events_df['datetime'] = pd.to_timedelta(events_df[('temporal', 'start_seconds')],
                                                    unit='seconds') + time_bin
            events_df = events_df.set_index('datetime')
            total_events = total_events.append(events_df)
        if save_path is not None:
            csv_path = save_path.joinpath('%s.csv' %datetime.datetime.strftime(time_bin, "%y%m%d_%H%M%S"))
            total_events.to_csv(csv_path)
        return total_events

    def detect_ship_events(self, binsize=None, threshold=160.0, min_duration=10.0, detector=None,
                           verbose=False):
        """
        Find the loud events of the file
        Parameters
        ----------
        binsize : float
            Time window considered, in seconds. If set to None, only one value is returned
        threshold : float
            Threshold above which it is considered loud
        min_duration : float
            Minimum duration of the event, in seconds
        detector : loud_event_detector object
            The detector to be used
        verbose : boolean
            Set to True to see the spectrograms of the detections
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)

        if detector is None:
            detector = loud_event_detector.ShipDetector(min_duration=min_duration,
                                                        threshold=threshold)
        total_events = pd.DataFrame()
        for _, time_bin, signal in self._bins(blocksize):
            events_df = detector.detect_events(signal, verbose=True)
            events_df['start_datetime'] = pd.to_timedelta(events_df.duration, unit='seconds') + self.date
            events_df = events_df.set_index('start_datetime')
            total_events = total_events.append(events_df)

        if verbose:
            # _, fbands, t, sxx_list = self.spectrogram(nfft=4096*4, scaling='spectrum',
            # db=True, mode='fast')
            # sxx = sxx_list[0]
            fig, ax = plt.subplots(2, 1, sharex='all')
            # im = ax[0].pcolormesh(t, fbands, sxx, shading='auto')
            # cbar = plt.colorbar(im)
            # cbar.set_label('SPLrms [dB re 1 uPa]', rotation=90)
            ax[0].set_title('Spectrogram')
            ax[0].set_xlabel('Time [s]')
            ax[0].set_ylabel('Frequency [Hz]')
            ax[0].set_yscale('log')
            for index in total_events.index:
                row = total_events.loc[index]
                start_x = (row['start_datetime'] - self.date).total_seconds()
                end_x = start_x + row['duration']
                ax[0].axvline(x=start_x, color='red', label='detected start')
                ax[0].axvline(x=end_x, color='blue', label='detected stop')
            if len(total_events) > 0:
                total_events[['rms', 'sel', 'peak']].plot(ax=ax[2])
            plt.tight_layout()
            plt.show()
            plt.close()
        return total_events

    def source_separation(self, window_time=1.0, n_sources=15, save_path=None, verbose=False):
        """
        Perform non-negative Matrix Factorization to separate sources

        Parameters
        ----------
        window_time: float
            window time to consider in seconds
        n_sources : int
            Number of sources
        """
        separator = nmf.NMF(window_time=window_time, rank=n_sources)
        for i, time_bin, signal in self._bins(self.file.frames):
            signal.set_band(self.band)
            W, H, WH_prod, G_tf, C_tf, c_tf = separator(signal, verbose=verbose)
        return W, H, WH_prod, G_tf, C_tf, c_tf

    def find_calibration_tone(self, min_duration=10.0):
        """
        Find the beggining and ending sample of the calibration tone
        Returns start and end points, in seconds

        Parameters
        ----------
        min_duration : float
            Minimum duration of the calibration tone, in sec
        """
        high_freq = self.cal_freq * 1.05
        low_freq = self.cal_freq * 0.95
        tone_samples = self.samples(self.max_cal_duration)
        self.file.seek(0)
        first_part = self.file.read(frames=tone_samples)
        signal = Signal(first_part, self.fs, channel=self.channel)
        signal.set_band(band=[low_freq, high_freq], downsample=False)
        amplitude_envelope = signal.envelope()
        possible_points = np.zeros(amplitude_envelope.shape)
        possible_points[np.where(amplitude_envelope >= 0.05)] = 1
        start_points = np.where(np.diff(possible_points) == 1)[0]
        end_points = np.where(np.diff(possible_points) == -1)[0]
        if start_points.size == 0:
            return None, None, []
        if end_points[0] < start_points[0]:
            end_points = end_points[1:]
        if start_points.size != end_points.size:
            start_points = start_points[0:end_points.size]
        select_idx = np.argmax(end_points - start_points)
        # Round to a second
        start = int(start_points[select_idx])
        end = int(end_points[select_idx])

        if (end - start) / self.fs < min_duration:
            return None, None, []

        self.file.seek(0)
        return start, end, signal.signal[start_points[select_idx]:end_points[select_idx]]

    def cut_calibration_tone(self, min_duration=10.0, save_path=None):
        """
        Cut the calibration tone from the file
        Returns a numpy array with only calibration signal and signal without the
        calibration signal
        None if no reference signal is found

        Parameters
        ----------
        min_duration : float
            Minimum duration of the calibration tone, in sec
        save_path : string or Path
            Path where to save the calibration tone
        """
        start, stop, calibration_signal = self.find_calibration_tone(min_duration=min_duration)
        if start is not None:
            new_datetime = self.date + datetime.timedelta(seconds=self.samples2time(stop))
            calibration_signal, _ = sf.read(self.file_path, start=start, stop=stop)
            signal, _ = sf.read(self.file_path, start=stop + 1, stop=-1)
            if save_path is not None:
                if save_path == 'auto':
                    new_folder = self.file_path.parent.joinpath('formatted')
                    ref_folder = new_folder.joinpath('ref')
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    if not os.path.exists(ref_folder):
                        os.makedirs(ref_folder)
                    ref_name = self.file_path.name.replace('.wav', '_ref.wav')
                    save_path = ref_folder.joinpath(ref_name)
                    new_file_name = self.hydrophone.get_new_name(filename=self.file_path.name,
                                                                 new_date=new_datetime)
                    new_file_path = new_folder.joinpath(new_file_name)
                else:
                    new_file_path = self.file_path.parent.joinpath(
                        self.file_path.name.replace('.wav', '_cut.wav'))
                sf.write(file=save_path, data=calibration_signal, samplerate=self.fs)
                # Update datetime
                sf.write(file=new_file_path, data=signal, samplerate=self.fs)
            self.file.seek(0)
            return calibration_signal, signal

        else:
            return None

    def plot_psd(self, db=True, log=True, save_path=None, **kwargs):
        """
        Plot the power spectrogram density of all the file (units^2 / Hz) re 1 V 1 upa

        Parameters
        ----------
        db : boolean
            If set to True the result will be given in db. Otherwise in upa^2/Hz
        log : boolean
            If set to True the scale of the y axis is set to logarithmic
        save_path : string or Path
            Where to save the images
        **kwargs : any attribute valid on psd() function
        """
        psd = self.psd(db=db, **kwargs)
        if db:
            units = 'db %s upa^2/Hz' % self.p_ref
        else:
            units = 'upa^2/Hz'
        self._plot_spectrum(ds=psd, col_name='density', output_name='PSD', units=units, log=log,
                            save_path=save_path)

    def plot_power_spectrum(self, db=True, log=True, save_path=None, **kwargs):
        """
        Plot the power spectrogram of all the file (units^2) re 1 V 1 upa

        Parameters
        ----------
        db : boolean
            If set to True the result will be given in db. Otherwise in upa^2/Hz
        log : boolean
            If set to True the scale of the y axis is set to logarithmic
        save_path : string or Path
            Where to save the images
        **kwargs : any attribute valid on power_spectrum() function
        """
        power = self.power_spectrum(db=db, **kwargs)
        if db:
            units = 'db %s upa^2' % self.p_ref
        else:
            units = 'upa^2'
        self._plot_spectrum(df=power, col_name='spectrum', output_name='SPLrms', units=units,
                            log=log, save_path=save_path)

    @staticmethod
    def _plot_spectrum(ds, col_name, output_name, units, log=True, save_path=None):

        """
        Plot the spectrums contained on the dataset

        Parameters
        ----------
        ds : xarray Dataset
            Dataset resultant from psd or power spectrum calculation
        col_name : string
            Name of the column where the data is (scaling type) 'spectrum' or 'density'
        units : string
            Units of the data
        save_path: string or Path
            Where to save the image
        """
        for i in ds['datetime']:
            plt.figure()
            plt.plot(ds.coords.bands, ds.sel(datetime=i))
            plt.title('%s of bin %s' % (col_name.capitalize(), i.strftime("%Y-%m-%d %H:%M")))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('%s [%s]' % (output_name, units))

            plt.hlines(y=ds.loc[i, 'percentiles'].values, xmin=ds.coords.bands.min(), xmax=ds.coords.bands.max(),
                       label=ds['percentiles'].columns)
            if log:
                plt.xscale('log')
            plt.tight_layout()
            plt.show()
            if save_path is not None:
                plt.savefig(save_path)
            plt.close()

    def plot_spectrogram(self, db=True, log=True, save_path=None, **kwargs):
        """
        Return the spectrogram of the signal (entire file)

        Parameters
        ----------
        db : boolean
            If set to True the result will be given in db. Otherwise in upa^2/Hz
        log : boolean
            If set to True the scale of the y axis is set to logarithmic
        save_path : string or Path
            Where to save the images
        **kwargs : any attribute valid on spectrogram() function
        """
        time, fbands, t, sxx_list = self.spectrogram(db=db, **kwargs)
        for i, sxx in enumerate(sxx_list):
            # Plot the patterns
            plt.figure()
            im = plt.pcolormesh(t, fbands, sxx, shading='auto')
            plt.title('Spectrogram of bin %s' % (time[i].strftime("%Y-%m-%d %H:%M")))
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            if log:
                plt.yscale('log')
            if db:
                units = r'dB %s $\mu Pa$' % self.p_ref
            else:
                units = r'$\mu Pa$'
            cbar = plt.colorbar(im)
            cbar.set_label(r'$L_{rms}$ [%s]' % units, rotation=90)
            plt.tight_layout()
            plt.show()
            if save_path is not None:
                plt.savefig(save_path + time[i].strftime("%Y-%m-%d %H:%M"))
            plt.close()

    def plot_spd(self, db=True, log=True, save_path=None, **kwargs):
        """
        Plot the the SPD graph of the bin

        Parameters
        ----------
        db : boolean
            If set to True the result will be given in db. Otherwise in upa^2/Hz
        log : boolean
            If set to True the scale of the y axis is set to logarithmic
        save_path : string or Path
            Where to save the images
        **kwargs : any attribute valid on spd() function
        """
        time, fbands, percentiles, edges_list, spd_list, p_list = self.spd(db=db, **kwargs)
        if db:
            units = r'dB %s $\mu Pa^2/Hz$' % self.p_ref
        else:
            units = r'$\mu Pa^2/Hz$'
        for i, spd in enumerate(spd_list):
            # Plot the EPD
            fig = plt.figure()
            im = plt.pcolormesh(fbands, edges_list[i], spd.T, cmap='BuPu', shading='auto')
            if log:
                plt.xscale('log')
            plt.title('Spectral probability density at bin %s' %
                      time[i].strftime("%Y-%m-%d %H:%M"))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('PSD [%s]' % units)
            cbar = fig.colorbar(im)
            cbar.set_label('Empirical Probability Density', rotation=90)

            # Plot the lines of the percentiles
            plt.plot(fbands, p_list[i], label=percentiles)

            plt.tight_layout()
            if save_path is not None:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()


class HydroFile(AcuFile):
    def __init__(self, **kwargs):
        """
        Sound data recorded in a wav file with a hydrophone.

        Parameters
        ----------
        sfile : Sound file
            Can be a path or an file object
        hydrophone : Object for the class hydrophone
            Hydrophone used to record
        band: tuple or list
            Lowcut, Highcut. Frequency band to analyze
        channel : int
            Channel to perform the calculations in
        calibration_time : float
            Time to ignore at the beggining of the file
        dc_subtract: bool
            Set to True to subtract the dc noise (root mean squared value
        """
        super().__init__(**kwargs)


class MEMSFile(AcuFile):
    def __init__(self, sfile, hydrophone, acc_ref=1.0, band=None, utc=True, channel=0):
        """
        Acceleration data recorded in a wav file.

        Parameters
        ----------
        sfile : Sound file
            Can be a path or an file object
        hydrophone : Object for the class hydrophone
        acc_ref : float
            Reference acceleration in um/s
        band: tuple or list
            Lowcut, Highcut. Frequency band to analyze
        """
        super().__init__(sfile, hydrophone, acc_ref, band, utc, channel)

    def integrate_acceleration(self):
        """
        Integrate the acceleration to get the velocity of the particle.
        The constant is NOT resolved
        """
        if self.instant:
            raise Exception('This can only be implemented in average mode!')

        velocity = integrate.cumtrapz(self.signal('acc'), dx=1 / self.fs)

        return velocity


class MEMS3axFile:
    def __init__(self, hydrophone, xfile_path, yfile_path, zfile_path, acc_ref=1.0):
        """
        Class to treat the 3 axes together

        Parameters
        ----------
        hydrophone : Object for the class hydrophone
        xfile_path : string or Path
            File where the X accelerometer data is
        yfile_path : string or Path
            File where the Y accelerometer data is
        zfile_path : string or Path
            File where the Z accelerometer data is
        acc_ref : float
            Reference acceleration in um/s
        """
        self.x_path = xfile_path
        self.y_path = yfile_path
        self.z_path = zfile_path

        self.x = MEMSFile(hydrophone, xfile_path, acc_ref)
        self.y = MEMSFile(hydrophone, yfile_path, acc_ref)
        self.z = MEMSFile(hydrophone, zfile_path, acc_ref)

    def acceleration_magnitude(self):
        """
        Calculate the magniture acceleration signal
        """
        x_acc = self.x.signal('acc')
        y_acc = self.y.signal('acc')
        z_acc = self.z.signal('acc')

        return np.sqrt(x_acc ** 2 + y_acc ** 2 + z_acc ** 2)

    def velocity_magnitude(self):
        """
        Get the start and the end velocity
        """
        vx = self.x.integrate_acceleration()
        vy = self.y.integrate_acceleration()
        vz = self.z.integrate_acceleration()

        v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

        return v

    def mean_velocity_increment(self):
        """
        Get the mean increment of the velocity
        """
        mean_acc = self.acceleration_magnitude().mean()
        time = self.x._time_array()
        t_inc = (time()[-1] - time()[0]).total_seconds()

        return mean_acc * t_inc
