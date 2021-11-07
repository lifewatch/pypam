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
import seaborn as sns
import soundfile as sf
import xarray
from tqdm.auto import tqdm

from pypam import impulse_detector
from pypam import loud_event_detector
from pypam import nmf
from pypam import plots
from pypam import signal as sig
from pypam import utils

pd.plotting.register_matplotlib_converters()
plt.rcParams.update({'pcolor.shading': 'auto'})

# Apply the default theme
sns.set_theme()


class AcuFile:
    """
    Data recorded in a wav file.

    Parameters
    ----------
    sfile : Sound file
        Can be a path or an file object
    hydrophone : Object for the class hydrophone
    p_ref : Float
        Reference pressure in upa
    utc : boolean
        Set to True if working on UTC and not localtime
    channel : int
        Channel to perform the calculations in
    calibration_time: float or str
        If a float, the amount of seconds that are ignored at the beginning of the file. If 'auto' then
        before the analysis, find_calibration_tone will be performed
    max_cal_duration: float
        Maximum time in seconds for the calibration tone (only applies if calibration_time is 'auto')
    cal_freq: float
        Frequency of the calibration tone (only applies if calibration_time is 'auto')
    dc_subtract: bool
        Set to True to subtract the dc noise (root mean squared value
    """
    def __init__(self, sfile, hydrophone, p_ref, utc=True, channel=0,
                 calibration_time=0.0, max_cal_duration=60.0, cal_freq=250, dc_subtract=False):
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
                return self._time_array(blocksize=1/self.fs)[0]
            else:
                return self.__dict__[name]
        else:
            return self.__dict__[name]

    def _bins(self, binsize=None):
        """
        Yields the bins each binsize
        Parameters
        ----------
        binsize: float or None
            Number of seconds per bin to yield. If set to None, a single bin is yield for the entire file

        Returns
        -------
        Iterates through all the bins, yields i, time_bin, signal
        Where i is the index, time_bin is the datetime of the beginning of the block and signal is the signal object
        of the bin
        """
        if binsize is None:
            blocksize = self.file.frames - self._start_frame
        else:
            blocksize = self.samples(binsize)
        n_blocks = self._n_blocks(blocksize)
        time_array, _, _ = self._time_array(binsize)
        for i, block in tqdm(enumerate(sf.blocks(self.file_path, blocksize=blocksize, start=self._start_frame,
                                                 always_2d=True)),
                             total=n_blocks, leave=False, position=0):
            # Select the desired channel
            block = block[:, self.channel]
            if len(block) == blocksize:
                time_bin = time_array[i]
                # Read the signal and prepare it for analysis
                signal_upa = self.wav2upa(wav=block)
                signal = sig.Signal(signal=signal_upa, fs=self.fs, channel=self.channel)
                if self.dc_subtract:
                    signal.remove_dc()
                yield i, time_bin, signal
        self.file.seek(0)

    def _n_blocks(self, blocksize):
        return int(np.floor((sf.SoundFile(self.file_path).frames - self._start_frame) / blocksize))

    def samples(self, bintime):
        """
        Return the samples according to the fs

        Parameters
        ----------
        bintime : Float
            Seconds of bintime desired to convert to samples
        """
        return int(bintime * self.fs)

    def set_calibration_time(self, calibration_time):
        """
        Set a calibration time in seconds. This time will be ignored in the processing

        Parameters
        ----------
        calibration_time : float
            Seconds to ignore at the beginning of the file
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
        Returns window length in samples

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

    def _time_array(self, binsize=None):
        """
        Return a time array for each point of the signal
        """
        if binsize is None:
            blocksize = self.file.frames - self._start_frame
        else:
            blocksize = self.samples(binsize)
        blocks_samples = np.arange(start=self._start_frame, stop=self.file.frames - blocksize + 1, step=blocksize)
        end_samples = blocks_samples + blocksize
        incr = pd.to_timedelta(blocks_samples / self.fs, unit='seconds')
        self.time = self.date + datetime.timedelta(seconds=self._start_frame / self.fs) + incr
        return self.time, blocks_samples.astype(int), end_samples.astype(int)

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

    def _empty_dataset(self, binsize, var_name, extra_coords=None, extra_dims=None):
        time_array, start_sample, end_sample = self._time_array(binsize)
        coords = {'datetime': time_array,
                  'start_sample': ('datetime', start_sample),
                  'end_sample': ('datetime', end_sample)
                  }
        dims = ['datetime']
        if extra_coords is not None:
            coords.update(extra_coords)
        if extra_dims is not None:
            dims = dims + extra_dims
        attrs = self._get_metadata_attrs()
        empty_arr = xarray.DataArray(coords=coords, dims=dims)
        ds = xarray.Dataset(data_vars={var_name: empty_arr}, attrs=attrs)
        return ds

    def _apply_multiple(self, method_list, binsize=None, band_list=None, **kwargs):
        """
        Apply multiple methods per bin to save computational time

        Parameters
        ----------
        method_list: list of strings
            List of all the methods to apply
        band_list: list of tuples, tuple or None
            Bands to filter. Can be multiple bands (all of them will be analyzed) or only one band. A band is
            represented with a tuple as (low_freq, high_freq). If set to None, the broadband up to the Nyquist
            frequency will be analyzed
        binsize: float
            Length in seconds of the bins to analyze
        kwargs: any parameters that have to be passed to the methods

        Returns
        -------
        DataFrame with time as index and a multiindex column with band, method as levels.
        """
        # TODO decide if it is downsampled or not
        downsample = False

        # Bands selected to study
        if band_list is None:
            band_list = [[None, self.fs / 2]]

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

        # Define an empty dataset
        ds = xarray.Dataset()
        band_coords = {'bands': np.arange(len(sorted_bands)),
                       'band_lowfreq': ('bands', low_freqs),
                       'band_highfreq': ('bands', high_freqs),
                       }
        for f in method_list:
            ds = ds.merge(self._empty_dataset(binsize=binsize, var_name=f, extra_coords=band_coords, extra_dims=['bands']))

        for _, time_bin, signal in self._bins(binsize):
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
                    ds[method_name].loc[time_bin, j] = output
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
            Length in seconds of the bin to analyze
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
            Length in seconds of the bin to analyze
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
            Length in seconds of the bin to analyze
        db: boolean
            Set to True if the result should be in decibels

        Returns
        -------
        DataFrame with multiindex columns with levels method and band. The method is '3-oct'

        """
        downsample = True

        if band is None:
            band = [None, self.fs / 2]
        oct_str = 'oct%s' % fraction

        # Create an empty dataset
        ds = self._empty_dataset(binsize=binsize, var_name=oct_str)
        for _, time_bin, signal in self._bins(binsize):
            signal.set_band(band, downsample=downsample)
            fbands, levels = signal.octave_levels(db, fraction)
            if len(ds.dims) == 1:
                ds = self._empty_dataset(binsize=binsize, var_name=oct_str,
                                         extra_coords={'frequency': fbands},
                                         extra_dims=['frequency'])
            ds[oct_str].loc[time_bin, :] = levels

        return ds

    def spectrogram(self, binsize=None, nfft=512, scaling='density', db=True, band=None):
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
        band : tuple or None
            Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
            (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed

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
        if band is None:
            band = [None, self.fs / 2]

        ds = self._empty_dataset(binsize=binsize, var_name='spectrogram')
        for i, time_bin, signal in self._bins(binsize):
            signal.set_band(band, downsample=downsample)
            freq, t, sxx = signal.spectrogram(nfft=nfft, scaling=scaling, db=db)
            if len(ds.dims) == 1:
                ds = self._empty_dataset(binsize=binsize, var_name='spectrogram',
                                         extra_coords={'frequency': freq, 'time': t},
                                         extra_dims=['frequency', 'time'])
            ds['spectrogram'].loc[time_bin, :, :] = sxx
        return ds

    def _spectrum(self, scaling='density', binsize=None, nfft=512, db=True, percentiles=None, band=None):
        """
        Return the spectrum : frequency distribution of all the file (periodogram)
        Returns Dataframe with 'datetime' as index and a column for each frequency and each
        percentile, and a frequency array

        Parameters
        ----------
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        nfft : int
            Length of the fft window in samples. Power of 2.
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned
        """
        downsample = True
        if percentiles is None:
            percentiles = []
        if band is None:
            band = [None, self.fs / 2]

        spectrum_str = 'band_' + scaling
        ds = self._empty_dataset(binsize=binsize, var_name=spectrum_str)
        for _, time_bin, signal in self._bins(binsize):
            signal.set_band(band, downsample=downsample)
            fbands, spectra, percentiles_val = signal.spectrum(scaling=scaling, nfft=nfft, db=db,
                                                               percentiles=percentiles)
            if len(ds.dims) == 1:
                ds = self._empty_dataset(binsize=binsize, var_name=spectrum_str, extra_coords={'frequency': fbands},
                                         extra_dims=['frequency'])
                ds_percentiles = self._empty_dataset(binsize=binsize, var_name='value_percentiles',
                                                     extra_coords={'frequency': fbands, 'percentiles': percentiles},
                                                     extra_dims=['frequency', 'percentiles'])
                ds = ds.merge(ds_percentiles)
            ds[spectrum_str].loc[time_bin, fbands] = spectra
            ds['value_percentiles'].loc[time_bin] = percentiles_val

        return ds

    def psd(self, binsize=None, nfft=512, db=True, percentiles=None):
        """
        Return the power spectrum density (PSD) of all the file (units^2 / Hz) re 1 V 1 upa
        Returns a Dataframe with 'datetime' as index and a column for each frequency and each
        percentile

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        nfft : int
            Length of the fft window in samples. Power of 2.
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned
        """
        psd_ds = self._spectrum(scaling='density', binsize=binsize, nfft=nfft, db=db,
                                percentiles=percentiles)
        return psd_ds

    def power_spectrum(self, binsize=None, nfft=512, db=True, percentiles=None):
        """
        Return the power spectrum of all the file (units^2 / Hz) re 1 V 1 upa
        Returns a Dataframe with 'datetime' as index and a column for each frequency and
        each percentile

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        nfft : int
            Length of the fft window in samples. Power of 2.
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned
        """

        spectrum_ds = self._spectrum(scaling='spectrum', binsize=binsize, nfft=nfft, db=db,
                                     percentiles=percentiles)
        return spectrum_ds

    def spd(self, binsize=None, h=0.1, nfft=512, db=True, percentiles=None, min_val=None, max_val=None):
        """
        Return the spectral probability density.

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        h : float
            Histogram bin width (in the correspondent units, upa or db)
        nfft : int
            Length of the fft window in samples. Power of 2.
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        min_val : float
            Minimum value to compute the SPD histogram
        max_val : float
            Maximum value to compute the SPD histogram

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
        psd_evolution = self.psd(binsize=binsize, nfft=nfft, db=db, percentiles=percentiles)
        return compute_spd(psd_evolution, h=h, percentiles=percentiles, max_val=max_val, min_val=min_val)

    def detect_piling_events(self, min_separation, max_duration, threshold, dt, binsize=None,
                             verbose=False, save_path=None, detection_band=None, analysis_band=None, method=None):
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
        detection_band : list or tuple
            Band used to detect the pulses [low_freq, high_freq]
        analysis_band : list or tuple
            Band used to analyze the pulses [low_freq, high_freq]
        method : str
            Method to use for the detection. Can be 'snr', 'dt' or 'envelope'
        """
        if type(save_path) == str:
            save_path = pathlib.Path(save_path)

        detector = impulse_detector.PilingDetector(min_separation=min_separation,
                                                   max_duration=max_duration,
                                                   threshold=threshold, dt=dt, detection_band=detection_band,
                                                   analysis_band=analysis_band)
        total_events = pd.DataFrame()
        for _, time_bin, signal in self._bins(binsize):
            signal.set_band(band=analysis_band, downsample=False)
            if save_path is not None:
                file_path = save_path.joinpath('%s.png' % datetime.datetime.strftime(time_bin, "%y%m%d_%H%M%S"))
            else:
                file_path = None
            events_df = detector.detect_events(signal, method=method, verbose=verbose, save_path=file_path)
            events_df['datetime'] = pd.to_timedelta(events_df[('temporal', 'start_seconds')],
                                                    unit='seconds') + time_bin
            events_df = events_df.set_index('datetime')
            total_events = total_events.append(events_df)
        if save_path is not None:
            csv_path = save_path.joinpath('%s.csv' % datetime.datetime.strftime(self.date, "%y%m%d_%H%M%S"))
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
        if detector is None:
            detector = loud_event_detector.ShipDetector(min_duration=min_duration,
                                                        threshold=threshold)

        total_events = pd.DataFrame()
        for i, time_bin, signal in self._bins(binsize):
            events_df = detector.detect_events(signal, verbose=verbose)
            events_df['start_datetime'] = pd.to_timedelta(events_df.start_seconds, unit='seconds') + time_bin
            seconds_start = binsize * i
            events_df['start_seconds'] = events_df['start_seconds'] + seconds_start
            events_df['end_seconds'] = events_df['end_seconds'] + seconds_start
            total_events = total_events.append(events_df)

        return total_events

    def source_separation(self, window_time=1.0, n_sources=15, save_path=None, verbose=False, band=None):
        """
        Perform non-negative Matrix Factorization to separate sources

        Parameters
        ----------
        window_time: float
            window time to consider in seconds
        n_sources : int
            Number of sources
        save_path: str or Path
            Where to save the output
        verbose: bool
            Set to True to make plots of the process
        band : tuple or None
            Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
            (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed
        """
        if band is None:
            band = [None, self.fs / 2]
        separator = nmf.NMF(window_time=window_time, rank=n_sources)
        for i, time_bin, signal in self._bins(self.file.frames):
            signal.set_band(band)
            w, h, wh_prod, g_tf, c_tf, c_tf_i = separator(signal, verbose=verbose)
            # TODO decide how to return this output!
            yield w, h, wh_prod, g_tf, c_tf, c_tf_i

    def find_calibration_tone(self, min_duration=10.0):
        """
        Find the beginning and ending sample of the calibration tone
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
        signal = sig.Signal(first_part, self.fs, channel=self.channel)
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
            units = r'$SPL_rms [dB %s \mu Pa^2/Hz]$' % self.p_ref
        else:
            units = r'$SPL_rms [\mu Pa^2/Hz]$'
        self._plot_spectrum(ds=psd, col_name='density', units=units, log=log,
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
            units = r'$SPL_rms [dB %s \mu Pa^2]$' % self.p_ref
        else:
            units = r'$SPL_rms [\mu Pa^2]$'
        self._plot_spectrum(ds=power, col_name='spectrum', units=units,
                            log=log, save_path=save_path)

    @staticmethod
    def _plot_spectrum(ds, col_name, units, log=False, save_path=None):
        plots.plot_spectrum(ds, 'band_'+col_name, log=log, ylabel=units, save_path=save_path)

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
        ds_spectrogram = self.spectrogram(db=db, **kwargs)
        plots.plot_spectrograms(ds_spectrogram, log, db, self.p_ref, save_path)

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
        spd_ds = self.spd(db=db, **kwargs)
        plots.plot_spd(spd_ds, db=db, log=log, p_ref=self.p_ref, save_path=save_path)


def compute_spd(psd_evolution, h=1.0, percentiles=None, max_val=None, min_val=None):
    pxx = psd_evolution['band_density'].to_numpy().T
    if percentiles is None:
        percentiles = []
    if min_val is None:
        min_val = pxx.min()
    if max_val is None:
        max_val = pxx.max()
    # Calculate the bins of the psd values and compute spd using numba
    bin_edges = np.arange(start=max(0, min_val), stop=max_val, step=h)
    spd, p = utils.sxx2spd(sxx=pxx, h=h, percentiles=np.array(percentiles) / 100.0, bin_edges=bin_edges)
    spd_arr = xarray.DataArray(data=spd,
                               coords={'frequency': psd_evolution.frequency, 'spl': bin_edges[:-1]},
                               dims=['frequency', 'spl'])
    p_arr = xarray.DataArray(data=p,
                             coords={'frequency': psd_evolution.frequency, 'percentiles': percentiles},
                             dims=['frequency', 'percentiles'])
    spd_ds = xarray.Dataset(data_vars={'spd': spd_arr, 'value_percentiles': p_arr})

    return spd_ds
