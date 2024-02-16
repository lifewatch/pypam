__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

import datetime
import operator
import os
import pathlib
import zipfile

import dateutil.parser as parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray
from tqdm import tqdm

from pypam import acoustic_file
from pypam import plots
from pypam import utils

# Apply the default theme
sns.set_theme()


class ASA:
    """
    Init a AcousticSurveyAnalysis (ASA)

    Parameters
    ----------
    hydrophone : Hydrophone class from pyhydrophone
    folder_path : string or Path
        Where all the sound files are
    zipped : boolean
        Set to True if the directory is zipped
    extension: str
        Default to .wav, extension of the sound files to process
    include_dirs : boolean
        Set to True if the folder contains other folders with sound files
    p_ref : float
        Reference pressure in uPa
    binsize : float
        Time window considered, in seconds. If set to None, only one value is returned
    nfft : int
        Samples of the fft bin used for the spectral analysis
    bin_overlap : float [0 to 1]
        Percentage to overlap the bin windows
    period : tuple or list
        Tuple or list with two elements: start and stop. Has to be a string in the
        format YYYY-MM-DD HH:MM:SS
    calibration: float, -1 or None
        If it is a float, it is the time ignored at the beginning of the file. If None, nothing is done. If negative,
        the function calibrate from the hydrophone is performed, and the first samples ignored (and hydrophone updated)
    dc_subtract: bool
        Set to True to subtract the dc noise (root mean squared value)
    timezone: datetime.tzinfo, pytz.tzinfo.BaseTZInfo, dateutil.tz.tz.tzfile, str or None
        Timezone where the data was recorded in
    """

    def __init__(self,
                 hydrophone: object,
                 folder_path,
                 zipped=False,
                 extension='.wav',
                 include_dirs=False,
                 p_ref=1.0,
                 binsize=None,
                 bin_overlap=0,
                 nfft=1.0,
                 fft_overlap=0.5,
                 period=None,
                 timezone='UTC',
                 channel=0,
                 calibration=None,
                 dc_subtract=False,
                 extra_attrs=None):

        self.hydrophone = hydrophone
        self.acu_files = AcousticFolder(folder_path=folder_path, zipped=zipped,
                                        include_dirs=include_dirs, extension=extension)
        self.p_ref = p_ref
        self.binsize = binsize
        self.nfft = nfft
        self.bin_overlap = bin_overlap
        self.fft_overlap = fft_overlap

        if period is not None:
            if not isinstance(period[0], datetime.datetime):
                start = parser.parse(period[0])
                end = parser.parse(period[1])
                period = [start, end]
        self.period = period

        self.timezone = timezone
        self.datetime_timezone = 'UTC'
        self.channel = channel
        self.calibration = calibration
        self.dc_subtract = dc_subtract

        if extra_attrs is None:
            self.extra_attrs = {}
        else:
            self.extra_attrs = extra_attrs

        self.file_dependent_attrs = ['file_path', '_start_frame', 'end_to_end_calibration']

    def _files(self):
        """
        Iterator that returns AcuFile for each wav file in the folder
        """
        for file_list in tqdm(self.acu_files):
            wav_file = file_list[0]
            print(wav_file)
            sound_file = self._hydro_file(wav_file)
            if sound_file.is_in_period(self.period) and sound_file.file.frames > 0:
                yield sound_file

    def _hydro_file(self, wav_file):
        """
        Return the AcuFile object from the wav_file
        Parameters
        ----------
        wav_file : str or Path
            Sound file

        Returns
        -------
        Object AcuFile
        """
        hydro_file = acoustic_file.AcuFile(sfile=wav_file, hydrophone=self.hydrophone, p_ref=self.p_ref,
                                           timezone=self.timezone, channel=self.channel, calibration=self.calibration,
                                           dc_subtract=self.dc_subtract)
        return hydro_file
    
    def _get_metadata_attrs(self):
        metadata_keys = [
            'binsize',
            'nfft',
            'bin_overlap',
            'fft_overlap',
            'timezone',
            'datetime_timezone',
            'p_ref',
            'channel',
            'dc_subtract',
            'hydrophone.name',
            'hydrophone.model',
            'hydrophone.sensitivity',
            'hydrophone.preamp_gain',
            'hydrophone.Vpp',
        ]
        metadata_attrs = self.extra_attrs.copy()
        for k in metadata_keys:
            d = self
            for sub_k in k.split('.'):
                d = d.__dict__[sub_k]
            if isinstance(d, pathlib.Path):
                d = str(d)
            metadata_attrs[k.replace('.', '_')] = d

        return metadata_attrs

    def evolution_multiple(self, method_list: list, band_list=None, **kwargs):
        """
        Compute the method in each file and output the evolution
        Returns a xarray DataSet with datetime as index and one row for each bin of each file

        Parameters
        ----------
        method_list : string
            Method name present in AcuFile
        band_list: list of tuples, tuple or None
            Bands to filter. Can be multiple bands (all of them will be analyzed) or only one band. A band is
            represented with a tuple as (low_freq, high_freq). If set to None, the broadband up to the Nyquist
            frequency will be analyzed
        **kwargs :
            Any accepted parameter for the method_name
        """
        ds = xarray.Dataset(attrs=self._get_metadata_attrs())
        f = operator.methodcaller('_apply_multiple', method_list=method_list, binsize=self.binsize,
                                  nfft=self.nfft, fft_overlap=self.fft_overlap, bin_overlap=self.bin_overlap,
                                  band_list=band_list, **kwargs)
        for sound_file in self._files():
            ds_output = f(sound_file)
            ds = utils.merge_ds(ds, ds_output, self.file_dependent_attrs)
        return ds

    def evolution(self, method_name, band_list=None, **kwargs):
        """
        Evolution of only one param name

        Parameters
        ----------
        method_name : string
            Method to compute the evolution of
        band_list: list of tuples, tuple or None
            Bands to filter. Can be multiple bands (all of them will be analyzed) or only one band. A band is
            represented with a tuple as (low_freq, high_freq). If set to None, the broadband up to the Nyquist
            frequency will be analyzed
        **kwargs : any arguments to be passed to the method
        """
        return self.evolution_multiple(method_list=[method_name], band_list=band_list, **kwargs)

    def evolution_freq_dom(self, method_name, **kwargs):
        """
        Returns the evolution of frequency domain parameters
        Parameters
        ----------
        method_name : str
            Name of the method of the acoustic_file class to compute
        Returns
        -------
        A xarray DataSet with a row per bin with the method name output
        """
        ds = xarray.Dataset(attrs=self._get_metadata_attrs())
        f = operator.methodcaller(method_name, binsize=self.binsize, nfft=self.nfft, fft_overlap=self.fft_overlap,
                                  bin_overlap=self.bin_overlap, **kwargs)
        for sound_file in self._files():
            ds_output = f(sound_file)
            ds = utils.merge_ds(ds, ds_output, self.file_dependent_attrs)
        return ds

    def timestamps_array(self):
        """
        Return a xarray DataSet with the timestamps of each bin.
        """
        ds = xarray.Dataset(attrs=self._get_metadata_attrs())
        f = operator.methodcaller('timestamp_da', binsize=self.binsize, bin_overlap=self.bin_overlap)
        for sound_file in self._files():
            ds_output = f(sound_file)
            ds = utils.merge_ds(ds, ds_output, self.file_dependent_attrs)
        return ds

    def start_end_timestamp(self):
        """
        Return the start and the end timestamps
        """
        wav_file = self.acu_files[0][0]
        print(wav_file)

        sound_file = self._hydro_file(wav_file)
        start_datetime = sound_file.date

        file_list = self.acu_files[-1]
        wav_file = file_list[0]
        print(wav_file)
        sound_file = self._hydro_file(wav_file)
        end_datetime = sound_file.date + datetime.timedelta(seconds=sound_file.total_time())

        return start_datetime, end_datetime

    def apply_to_all(self, method_name, **kwargs):
        """
        Apply the method to all the files

        Parameters
        ----------
        method_name : string
            Method name present in AcuFile
        **kwargs :
            Any accepted parameter for the method_name

        """
        f = operator.methodcaller(method_name, binsize=self.binsize, nfft=self.nfft, fft_overlap=self.fft_overlap,
                                  bin_overlap=self.bin_overlap, **kwargs)
        for sound_file in self._files():
            f(sound_file)

    def duration(self):
        """
        Return the duration in seconds of all the survey
        """
        total_time = 0
        for sound_file in self._files():
            total_time += sound_file.total_time()

        return total_time

    def mean_rms(self, **kwargs):
        """
        Return the mean root mean squared value of the survey
        Accepts any other input than the correspondent method in the acoustic file.
        Returns the rms value of the whole survey

        Parameters
        ----------
        **kwargs :
            Any accepted arguments for the rms function of the AcuFile
        """
        rms_evolution = self.evolution('rms', **kwargs)
        return rms_evolution['rms'].mean()

    def spd(self, db=True, h=0.1, percentiles=None, min_val=None, max_val=None):
        """
        Return the empirical power density.

        Parameters
        ----------
        db : boolean
            If set to True the result will be given in db. Otherwise, in uPa^2
        h : float
            Histogram bin (in the correspondent units, uPa or db)
        percentiles : list or None
            All the percentiles that have to be returned. If set to None, no percentiles
            is returned (in 100 per cent)
        min_val : float
            Minimum value to compute the SPD histogram
        max_val : float
            Maximum value to compute the SPD histogram

        Returns
        -------
        percentiles : array like
            List of the percentiles calculated
        p : np.array
            Matrix with all the probabilities
        """
        psd_evolution = self.evolution_freq_dom('psd', db=db, percentiles=percentiles)
        return utils.compute_spd(psd_evolution, h=h, percentiles=percentiles, min_val=min_val, max_val=max_val)

    def hybrid_millidecade_bands(self, db=True, method='spectrum', band=None, percentiles=None, average='mean'):
        """

        Parameters
        ----------
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        method: string
            Can be 'spectrum' or 'density'
        average: string
            Can be 'mean' or 'median'
        band : tuple or None
            Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
            (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed
        percentiles : list or None
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned

        Returns
        -------
        An xarray dataset with the band_density (or band_spectrum) and the millidecade_bands variables
        """
        spectra_ds = self.evolution_freq_dom('_spectrum', band=band, db=False, percentiles=percentiles, scaling=method,
                                             avreage=average)
        bands_limits, bands_c = utils.get_hybrid_millidecade_limits(band=band, nfft=self.nfft)
        fft_bin_width = band[1] * 2 / self.nfft # Signal has been downsampled
        milli_spectra = utils.spectra_ds_to_bands(spectra_ds['band_%s' % method],
                                                  bands_limits, bands_c, fft_bin_width=fft_bin_width, db=db)

        # Add the millidecade
        spectra_ds['millidecade_bands'] = milli_spectra
        return spectra_ds

    def source_separation(self, window_time=1.0, n_sources=15, save_path=None, verbose=False, band=None):
        """
        Separate the signal in n_sources sources, using non-negative matrix factorization
        Parameters
        ----------
        window_time: float
            Duration of the window in seconds
        n_sources: int
            Number of sources to separate the sound in
        save_path: str or Path
            Where to save the output
        verbose: bool
            Set to True to make plots of the process
        band : tuple or list
            Tuple or list with two elements: low-cut and high-cut of the band to analyze
        """
        ds = xarray.Dataset(attrs=self._get_metadata_attrs())
        for sound_file in self._files():
            nmf_ds = sound_file.source_separation(window_time, n_sources, binsize=self.binsize, band=band,
                                                  save_path=save_path, verbose=verbose)
            ds = utils.merge_ds(ds, nmf_ds, self.file_dependent_attrs)

        return ds

    def plot_rms_evolution(self, db=True, save_path=None):
        """
        Plot the rms evolution

        Parameters
        ----------
        db : boolean
            If set to True, output in db
        save_path : string or Path
            Where to save the output graph. If None, it is not saved
        """
        rms_evolution = self.evolution('rms', db=db)
        plots.plot_rms_evolution(ds=rms_evolution, save_path=save_path)

    def plot_rms_daily_patterns(self, db=True, save_path=None):
        """
        Plot the daily rms patterns

        Parameters
        ----------
        db : boolean
            If set to True, the output is in db and will be show in the units output
        save_path : string or Path
            Where to save the output graph. If None, it is not saved
        """
        rms_evolution = self.evolution('rms', db=db).sel(band=0)
        daily_xr = rms_evolution.swap_dims(id='datetime')
        daily_xr = daily_xr.sortby('datetime')
        plots.plot_daily_patterns_from_ds(ds=daily_xr, data_var='rms', save_path=save_path, datetime_coord='datetime')

    def plot_median_power_spectrum(self, db=True, save_path=None, log=True, **kwargs):
        """
        Plot the resulting mean power spectrum

        Parameters
        ----------
        db : boolean
            If set to True, output in db
        log : boolean
            If set to True, y axis in logarithmic scale
        save_path : string or Path
            Where to save the output graph. If None, it is not saved
        **kwargs : Any accepted for the power_spectrum method
        """
        power = self.evolution_freq_dom(method_name='power_spectrum', db=db, **kwargs)

        return plots.plot_spectrum_median(ds=power, data_var='band_spectrum', log=log, save_path=save_path)

    def plot_median_psd(self, db=True, save_path=None, log=True, **kwargs):
        """
        Plot the resulting mean psd

        Parameters
        ----------
        db : boolean
            If set to True, output in db
        log : boolean
            If set to True, y axis in logarithmic scale
        save_path : string or Path
            Where to save the output graph. If None, it is not saved
        **kwargs : Any accepted for the psd method
        """
        psd = self.evolution_freq_dom(method_name='psd', db=db, **kwargs)

        return plots.plot_spectrum_median(ds=psd, data_var='band_density', log=log, save_path=save_path)

    def plot_power_ltsa(self, db=True, save_path=None, **kwargs):
        """
        Plot the evolution of the power frequency distribution (Long Term Spectrogram Analysis)

        Parameters
        ----------
        db : boolean
            If set to True, output in db
        save_path : string or Path
            Where to save the output graph. If None, it is not saved
        **kwargs : Any accepted for the power spectrum method
        """
        power_evolution = self.evolution_freq_dom(method_name='power_spectrum', db=db, **kwargs)
        plots.plot_ltsa(ds=power_evolution, data_var='band_spectrum', save_path=save_path)

        return power_evolution

    def plot_psd_ltsa(self, db=True, save_path=None, **kwargs):
        """
        Plot the evolution of the psd power spectrum density (Long Term Spectrogram Analysis)

        Parameters
        ----------
        db : boolean
            If set to True, output in db
        save_path : string or Path
            Where to save the output graph. If None, it is not saved
        **kwargs : Any accepted for the psd method
        """
        psd_evolution = self.evolution_freq_dom(method_name='psd', db=db, **kwargs)
        plots.plot_ltsa(ds=psd_evolution, data_var='band_density', save_path=save_path)

        return psd_evolution

    def plot_spd(self, db=True, log=True, save_path=None, **kwargs):
        """
        Plot the the SPD graph

        Parameters
        ----------
        db : boolean
            If set to True, output in db
        log : boolean
            If set to True, y-axis in logarithmic scale
        save_path : string or Path
            Where to save the output graph. If None, it is not saved
        **kwargs : Any accepted for the spd method
        """
        spd_ds = self.spd(db=db, **kwargs)
        plots.plot_spd(spd_ds, log=log, save_path=save_path)

    def save(self, file_path):
        """
        Save the ASA with all the computed values
        Returns
        -------

        """

    def update_freq_cal(self, ds, data_var, **kwargs):
        return utils.update_freq_cal(hydrophone=self.hydrophone, ds=ds, data_var=data_var, **kwargs)


class AcousticFolder:
    """
    Class to help through the iterations of the acoustic folder.
    """

    def __init__(self, folder_path, zipped=False, include_dirs=False, extension='.wav', extra_extensions=None):
        """
        Store the information about the folder.
        It will create an iterator that returns all the pairs of extensions having the same name than the wav file

        Parameters
        ----------
        folder_path : string or pathlib.Path
            Path to the folder containing the acoustic files
        zipped : boolean
            Set to True if the subfolders are zipped
        include_dirs : boolean
            Set to True if the subfolders are included in the study
        extension : str
            Default to .wav, sound file extension
        extra_extensions : list
            List of strings with all the extra extensions that will be returned
            i.e. extensions=['.xml', '.bcl'] will return [wav, xml and bcl] files
        """
        self.folder_path = pathlib.Path(folder_path)
        self.extension = extension
        if not self.folder_path.exists():
            raise FileNotFoundError('The path %s does not exist. Please choose another one.' % folder_path)
        if len(list(self.folder_path.glob('**/*%s' % self.extension))) == 0:
            raise ValueError('The directory %s is empty. Please select another directory with *.%s files' %
                             (folder_path, self.extension))
        self.zipped = zipped
        self.recursive = include_dirs
        if extra_extensions is None:
            extra_extensions = []
        self.extra_extensions = extra_extensions

    def __getitem__(self, n):
        """
        Get n wav file
        """
        self.__iter__()
        self.n = n
        return self.__next__()

    def __iter__(self):
        """
        Iteration
        """
        self.n = 0
        if not self.zipped:
            if self.recursive:
                self.files_list = sorted(self.folder_path.glob('**/*%s' % self.extension))
            else:
                self.files_list = sorted(self.folder_path.glob('*%s' % self.extension))
        else:
            if self.recursive:
                self.folder_list = sorted(self.folder_path.iterdir())
                self.zipped_subfolder = AcousticFolder(self.folder_list[self.n],
                                                       extra_extensions=self.extra_extensions,
                                                       zipped=self.zipped,
                                                       include_dirs=self.recursive)
            else:
                zipped_folder = zipfile.ZipFile(self.folder_path, 'r', allowZip64=True)
                self.files_list = []
                total_files_list = zipped_folder.namelist()
                for f in total_files_list: 
                    extension = f.split(".")[-1]
                    if extension == 'wav':
                        self.files_list.append(f)
        return self

    def __next__(self):
        """
        Next wav file
        """
        if self.n < len(self.files_list):
            files_list = []
            if self.zipped:
                if self.recursive:
                    try:
                        self.files_list = self.zipped_subfolder.__next__()
                    except StopIteration:
                        self.n += 1
                        self.zipped_subfolder = AcousticFolder(self.folder_list[self.n],
                                                               extra_extensions=self.extra_extensions,
                                                               zipped=self.zipped,
                                                               include_dirs=self.recursive)
                else:
                    file_name = self.files_list[self.n]
                    zipped_folder = zipfile.ZipFile(self.folder_path, 'r', allowZip64=True)
                    wav_file = zipped_folder.open(file_name)
                    files_list.append(wav_file)
                    for extension in self.extra_extensions:
                        ext_file_name = file_name.parent.joinpath(
                            file_name.name.replace(self.extension, extension))
                        files_list.append(zipped_folder.open(ext_file_name))
                    self.n += 1
                    return files_list
            else:
                wav_path = self.files_list[self.n]
                files_list.append(wav_path)
                for extension in self.extra_extensions:
                    files_list.append(pathlib.Path(str(wav_path).replace(self.extension, extension)))

                self.n += 1
                return files_list
        else:
            raise StopIteration

    def __len__(self):
        if not self.zipped:
            if self.recursive:
                n_files = len(list(self.folder_path.glob('**/*%s' % self.extension)))
            else:
                n_files = len(list(self.folder_path.glob('*%s' % self.extension)))
        else:
            if self.recursive:
                n_files = len(list(self.folder_path.iterdir()))
            else:
                zipped_folder = zipfile.ZipFile(self.folder_path, 'r', allowZip64=True)
                n_files = len(zipped_folder.namelist())
        return n_files


def move_file(file_path, new_folder_path):
    """
    Move the file to the new folder

    Parameters
    ----------
    file_path : string or Path
        Original file path
    new_folder_path : string or Path
        New folder destination (without the file name)
    """
    if not isinstance(file_path, pathlib.Path):
        file_path = pathlib.Path(file_path)
    if not isinstance(new_folder_path, pathlib.Path):
        new_folder_path = pathlib.Path(new_folder_path)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    new_path = new_folder_path.joinpath(file_path.name)
    os.rename(file_path, new_path)
