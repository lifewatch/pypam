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
    AcousticSurveyAnalysis (ASA) is a class that represents a collection of audio files from one deployment
    """

    def __init__(
        self,
        hydrophone: object,
        folder_path: str or pathlib.Path,
        zipped: bool = False,
        extension: str = ".wav",
        include_dirs: bool = False,
        gridded_data: bool = True,
        p_ref: float = 1.0,
        binsize: float = None,
        bin_overlap: float = 0,
        nfft: int = 1.0,
        fft_overlap: float = 0.5,
        period: tuple or list = None,
        timezone: datetime.tzinfo
        or pytz.tzinfo.BaseTZInfo
        or dateutil.tz.tz.tzfile
        or str = "UTC",
        channel: int = 0,
        calibration: float or int or None = None,
        dc_subtract: bool = False,
        extra_attrs: dict or None = None,
    ):
        """
        Args:
            hydrophone: Hydrophone object from pyhydrophone
            folder_path: Where all the sound files are
            zipped: Set to True if the directory is zipped
            extension: Default to .wav, extension of the sound files to process
            include_dirs: Set to True if the folder contains other folders with sound files
            gridded_data: Set to True to start the processing at all the minutes with 0 seconds
            p_ref: Reference pressure in uPa
            binsize: Time window considered, in seconds. If set to None, only one value is returned
            bin_overlap: Percentage to overlap the bin windows [0 to 1]
            nfft: Samples of the fft bin used for the spectral analysis
            fft_overlap: Percentage of overlap between fft bins [0 to 1]
            period: Tuple or list with two elements: start and stop. Has to be a string in the format YYYY-MM-DD HH:MM:SS
            timezone: Timezone where the data was recorded in
            channel: Channel to analyze
            calibration: If it is a float, it is the time ignored at the beginning of the file. If None, nothing is done. If negative,
                the function calibrate from the hydrophone is performed, and the first samples ignored (and hydrophone updated)
            dc_subtract: Set to True to subtract the dc noise (root mean squared value)
            extra_attrs: Extra attributes to store in the attrs variable of the output xarray Dataset
        """
        self.hydrophone = hydrophone
        self.acu_files = AcousticFolder(
            folder_path=folder_path,
            zipped=zipped,
            include_dirs=include_dirs,
            extension=extension,
        )
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
        self.datetime_timezone = "UTC"
        self.channel = channel
        self.calibration = calibration
        self.dc_subtract = dc_subtract

        if extra_attrs is None:
            self.extra_attrs = {}
        else:
            self.extra_attrs = extra_attrs

        self.file_dependent_attrs = [
            "file_path",
            "_start_frame",
            "end_to_end_calibration",
        ]
        self.file_dependent_attrs = [
            "file_path",
            "_start_frame",
            "end_to_end_calibration",
        ]
        self.current_chunk_id = 0

        self.gridded = gridded_data

    def _files(self) -> acoustic_file.AcuFile:
        """
        Iterator that returns AcuFile for each wav file in the folder
        """
        self.current_chunk_id = 0
        for file_i, file_list in tqdm(
            enumerate(self.acu_files), total=len(self.acu_files)
        ):
            wav_file = file_list[0]
            if file_i >= (len(self.acu_files) - 1):
                next_file = None
            else:
                next_file = self.acu_files[file_i + 1][0]
            sound_file = self._hydro_file(
                wav_file, wav_file_next=next_file, chunk_id_start=self.current_chunk_id
            )
            if sound_file.is_in_period(self.period) and sound_file.file.frames > 0:
                yield sound_file

    def _hydro_file(
        self,
        wav_file: str or pathlib.Path,
        wav_file_next: str or pathlib.Path = None,
        chunk_id_start: int = 0,
    ) -> acoustic_file.AcuFile:
        """
        Return the AcuFile object from the wav_file

        Args:
            wav_file: Sound file
            wav_file_next: Next Sound File
            chunk_id_start: id of the chunk to start from

        Returns:
            Object AcuFile
        """
        hydro_file = acoustic_file.AcuFile(
            sfile=wav_file,
            sfile_next=wav_file_next,
            hydrophone=self.hydrophone,
            p_ref=self.p_ref,
            timezone=self.timezone,
            channel=self.channel,
            calibration=self.calibration,
            dc_subtract=self.dc_subtract,
            chunk_id_start=chunk_id_start,
            gridded=self.gridded,
        )
        return hydro_file

    def _get_metadata_attrs(self) -> dict:
        metadata_keys = [
            "binsize",
            "nfft",
            "bin_overlap",
            "fft_overlap",
            "timezone",
            "datetime_timezone",
            "p_ref",
            "channel",
            "dc_subtract",
            "hydrophone.name",
            "hydrophone.model",
            "hydrophone.sensitivity",
            "hydrophone.preamp_gain",
            "hydrophone.Vpp",
        ]
        metadata_attrs = self.extra_attrs.copy()
        for k in metadata_keys:
            d = self
            for sub_k in k.split("."):
                d = d.__dict__[sub_k]
            if isinstance(d, pathlib.Path):
                d = str(d)
            if isinstance(d, bool):
                d = int(d)
            metadata_attrs[k.replace(".", "_")] = d

        return metadata_attrs

    def evolution_multiple(
        self,
        method_list: list,
        band_list: list = None,
        save_daily: bool = False,
        output_folder: str or pathlib.Path = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Compute the method in each file and output the evolution
        Returns a xarray DataSet with datetime as index and one row for each bin of each file

        Args:
            method_list: List of method names present in AcuFile
            band_list: list of tuples, tuple or None. Bands to filter. Can be multiple bands (all of them will be analyzed) or only one band. A band is
                represented with a tuple as (low_freq, high_freq). If set to None, the broadband up to the Nyquist
                frequency will be analyzed
            save_daily: Set to True to save daily netcdf files instead of returning a huge big file (useful for long deployments)
            output_folder: Directory to save the netcdf files. Only works with save_daily
            **kwargs:  Any accepted parameter for the method_name
        """
        if save_daily and output_folder is None:
            raise ValueError(
                "output_folder must not be none to save daily netcdf files"
            )
        if isinstance(output_folder, str):
            output_folder = pathlib.Path(output_folder)
        ds = xarray.Dataset(attrs=self._get_metadata_attrs())
        f = operator.methodcaller(
            "_apply_multiple",
            method_list=method_list,
            binsize=self.binsize,
            nfft=self.nfft,
            fft_overlap=self.fft_overlap,
            bin_overlap=self.bin_overlap,
            band_list=band_list,
            **kwargs,
        )
        start_date, end_date = self.start_end_timestamp()
        current_date = start_date.date()
        for sound_file in self._files():
            if save_daily and (sound_file.date.date() > current_date):
                ds.to_netcdf(output_folder.joinpath("%s.nc" % current_date))
                ds = xarray.Dataset(attrs=self._get_metadata_attrs())
                current_date = sound_file.date.date()
            ds_output = f(sound_file)
            ds = utils.merge_ds(ds, ds_output, self.file_dependent_attrs)
            self.current_chunk_id += ds.id.max()
        return ds

    def evolution(
        self, method_name: str, band_list: list = None, **kwargs
    ) -> xarray.Dataset:
        """
        Evolution of only one param name

        Args:
            method_name: Method to compute the evolution of
            band_list: list of tuples, tuple or None. Bands to filter.
                Can be multiple bands (all of them will be analyzed) or only one band. A band is represented with a
                tuple as (low_freq, high_freq). If set to None, the broadband up to the Nyquist
                frequency will be analyzed
            **kwargs : any arguments to be passed to the method
        """
        return self.evolution_multiple(
            method_list=[method_name], band_list=band_list, **kwargs
        )

    def evolution_freq_dom(
        self,
        method_name: str,
        save_daily: bool = False,
        output_folder: str or pathlib.Path = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Returns the evolution of frequency domain parameters

        Args:
            method_name: Name of the method of the acoustic_file class to compute
            save_daily: Set to True to save daily netcdf files instead of returning a huge big file (useful for long deployments)
            output_folder: Directory to save the netcdf files. Only works with save_daily

        Returns:
            A xarray DataSet with a row per bin with the method name output
        """
        if save_daily and output_folder is None:
            raise ValueError(
                "output_folder must not be none to save daily netcdf files"
            )
        if isinstance(output_folder, str):
            output_folder = pathlib.Path(output_folder)
        ds = xarray.Dataset(attrs=self._get_metadata_attrs())
        f = operator.methodcaller(
            method_name,
            binsize=self.binsize,
            nfft=self.nfft,
            fft_overlap=self.fft_overlap,
            bin_overlap=self.bin_overlap,
            **kwargs,
        )
        start_date, end_date = self.start_end_timestamp()
        current_date = start_date.date()
        for sound_file in self._files():
            if save_daily and (sound_file.date.date() > current_date):
                ds.to_netcdf(output_folder.joinpath("%s.nc" % current_date))
                ds = xarray.Dataset(attrs=self._get_metadata_attrs())
                current_date = sound_file.date.date()
            ds_output = f(sound_file)
            ds = utils.merge_ds(ds, ds_output, self.file_dependent_attrs)
            self.current_chunk_id += ds.id.max()
        if save_daily:
            ds.to_netcdf(output_folder.joinpath("%s.nc" % current_date))
        return ds

    def timestamps_array(self) -> xarray.Dataset:
        """
        Returns a xarray DataSet with the timestamps of each bin.
        """
        ds = xarray.Dataset(attrs=self._get_metadata_attrs())
        f = operator.methodcaller(
            "timestamp_da", binsize=self.binsize, bin_overlap=self.bin_overlap
        )
        for sound_file in self._files():
            ds_output = f(sound_file)
            ds = utils.merge_ds(ds, ds_output, self.file_dependent_attrs)
            self.current_chunk_id += ds.id.max()
        return ds

    def start_end_timestamp(self) -> tuple:
        """
        Returns the start and the end timestamps
        """
        wav_file = self.acu_files[0][0]

        sound_file = self._hydro_file(wav_file)
        start_datetime = sound_file.date

        file_list = self.acu_files[-1]
        wav_file = file_list[0]
        sound_file = self._hydro_file(wav_file)
        end_datetime = sound_file.date + datetime.timedelta(
            seconds=sound_file.total_time()
        )

        return start_datetime, end_datetime

    def apply_to_all(self, method_name: str, **kwargs):
        """
        Apply the method to all the files

        Args:
            method_name: Method name present in AcuFile
            **kwargs: Any accepted parameter for the method_name
        """
        f = operator.methodcaller(
            method_name,
            binsize=self.binsize,
            nfft=self.nfft,
            fft_overlap=self.fft_overlap,
            bin_overlap=self.bin_overlap,
            **kwargs,
        )
        for sound_file in self._files():
            f(sound_file)

    def duration(self) -> float:
        """
        Return the duration in seconds of all the survey
        """
        total_time = 0
        for sound_file in self._files():
            total_time += sound_file.total_time()

        return total_time

    def mean_rms(self, **kwargs) -> xarray.DataArray:
        """
        Return the mean root mean squared value of the survey
        Accepts any other input than the correspondent method in the acoustic file.
        Returns the rms value of the whole survey

        Args:
            **kwargs: Any accepted arguments for the rms function of the AcuFile
        """
        rms_evolution = self.evolution("rms", **kwargs)
        return rms_evolution["rms"].mean()

    def spd(
        self,
        db: bool = True,
        h: float = 0.1,
        percentiles: list = None,
        min_val: float = None,
        max_val: float = None,
    ) -> xarray.Dataset:
        """
        Return the empirical power density.

        Args:
            db: If set to True the result will be given in db. Otherwise, in uPa^2
            h: Histogram bin (in the correspondent units, uPa or db)
            percentiles: All the percentiles that have to be returned. If set to None, no percentiles
                is returned (in 100 per cent)
            min_val: Minimum value to compute the SPD histogram
            max_val: Maximum value to compute the SPD histogram

        Returns:
            xarray Dataset
        """
        psd_evolution = self.evolution_freq_dom("psd", db=db, percentiles=percentiles)
        return utils.compute_spd(
            psd_evolution,
            h=h,
            percentiles=percentiles,
            min_val=min_val,
            max_val=max_val,
        )

    def hybrid_millidecade_bands(
        self,
        db: bool = True,
        method: str = "spectrum",
        band: list or tuple = None,
        percentiles: list = None,
    ) -> xarray.Dataset:
        """
        Args:
            db: If set to True the result will be given in db, otherwise in upa^2
            method: Can be 'spectrum' or 'density'
            band: Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
                (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed
            percentiles:  List of all the percentiles that have to be returned. If set to empty list,
                no percentiles is returned

        Returns:
            An xarray dataset with the band_density (or band_spectrum) and the millidecade_bands variables
        """
        spectra_ds = self.evolution_freq_dom(
            "_spectrum", band=band, db=False, percentiles=percentiles, scaling=method
        )
        bands_limits, bands_c = utils.get_hybrid_millidecade_limits(
            band=band, nfft=self.nfft
        )
        fft_bin_width = band[1] * 2 / self.nfft  # Signal has been downsampled
        milli_spectra = utils.spectra_ds_to_bands(
            spectra_ds["band_%s" % method],
            bands_limits,
            bands_c,
            fft_bin_width=fft_bin_width,
            db=db,
        )

        # Add the millidecade
        spectra_ds["millidecade_bands"] = milli_spectra
        return spectra_ds

    def plot_rms_evolution(
        self, db: bool = True, save_path: str or pathlib.Path = None
    ) -> None:
        """
        Plot the rms evolution

        Args:
            db: If set to True, output in db
            save_path: Where to save the output graph. If None, it is not saved
        """
        rms_evolution = self.evolution("rms", db=db)
        plots.plot_rms_evolution(ds=rms_evolution, save_path=save_path)

    def plot_rms_daily_patterns(
        self, db: bool = True, save_path: str or pathlib.Path = None
    ):
        """
        Plot the daily rms patterns

        Args:
            db: If set to True, output in db
            save_path: Where to save the output graph. If None, it is not saved
        """
        rms_evolution = self.evolution("rms", db=db).sel(band=0)
        daily_xr = rms_evolution.swap_dims(id="datetime")
        daily_xr = daily_xr.sortby("datetime")
        plots.plot_daily_patterns_from_ds(
            ds=daily_xr, data_var="rms", save_path=save_path, datetime_coord="datetime"
        )

    def plot_median_power_spectrum(
        self,
        db: bool = True,
        save_path: str or pathlib.Path = None,
        log: bool = True,
        **kwargs,
    ):
        """
        Plot the resulting mean power spectrum

        Args:
            db: If set to True, output in db
            save_path: Where to save the output graph. If None, it is not saved
            log: If set to True, y axis in logarithmic scale
            **kwargs : Any accepted for the power_spectrum method
        """
        power = self.evolution_freq_dom(method_name="power_spectrum", db=db, **kwargs)

        return plots.plot_spectrum_median(
            ds=power, data_var="band_spectrum", log=log, save_path=save_path
        )

    def plot_median_psd(
        self,
        db: bool = True,
        save_path: str or pathlib.Path = None,
        log: bool = True,
        **kwargs,
    ):
        """
        Plot the resulting mean psd

        Args:
            db: If set to True, output in db
            save_path: Where to save the output graph. If None, it is not saved
            log: If set to True, y axis in logarithmic scale
            **kwargs : Any accepted for the power_spectrum method
        """
        psd = self.evolution_freq_dom(method_name="psd", db=db, **kwargs)

        return plots.plot_spectrum_median(
            ds=psd, data_var="band_density", log=log, save_path=save_path
        )

    def plot_power_ltsa(
        self, db: bool = True, save_path: str or pathlib.Path = None, **kwargs
    ):
        """
        Plot the evolution of the power frequency distribution (Long Term Spectrogram Analysis)

        Args:
            db: If set to True, output in db
            save_path: Where to save the output graph. If None, it is not saved
            **kwargs : Any accepted for the power_spectrum method
        """
        power_evolution = self.evolution_freq_dom(
            method_name="power_spectrum", db=db, **kwargs
        )
        plots.plot_ltsa(
            ds=power_evolution, data_var="band_spectrum", save_path=save_path
        )

        return power_evolution

    def plot_psd_ltsa(
        self, db: bool = True, save_path: str or pathlib.Path = None, **kwargs
    ):
        """
        Plot the evolution of the psd power spectrum density (Long Term Spectrogram Analysis)

        Args:
            db: If set to True, output in db
            save_path: Where to save the output graph. If None, it is not saved
            **kwargs : Any accepted for the psd method
        """
        psd_evolution = self.evolution_freq_dom(method_name="psd", db=db, **kwargs)
        plots.plot_ltsa(ds=psd_evolution, data_var="band_density", save_path=save_path)

        return psd_evolution

    def plot_spd(
        self,
        db: bool = True,
        save_path: str or pathlib.Path = None,
        log: bool = True,
        **kwargs,
    ):
        """
        Plot the SPD graph

        Args:
            db: If set to True, output in db
            save_path: Where to save the output graph. If None, it is not saved
            log: If set to True, y axis in logarithmic scale
            **kwargs : Any accepted for the pd method
        """
        spd_ds = self.spd(db=db, **kwargs)
        plots.plot_spd(spd_ds, log=log, save_path=save_path)

    def update_freq_cal(self, ds, data_var, **kwargs):
        return utils.update_freq_cal(
            hydrophone=self.hydrophone, ds=ds, data_var=data_var, **kwargs
        )


class AcousticFolder:
    """
    Class to help through the iterations of the acoustic folder.
    """

    def __init__(
        self,
        folder_path: str or pathlib.Path,
        zipped: bool = False,
        include_dirs: bool = False,
        extension: str = ".wav",
        extra_extensions: list = None,
    ):
        """
        Store the information about the folder.
        It will create an iterator that returns all the pairs of extensions having the same name than the wav file

        Args:
            folder_path: Path to the folder containing the acoustic files
            zipped: Set to True if the subfolders are zipped
            include_dirs: Set to True if the subfolders are included in the study
            extension: Default to .wav, sound file extension
            extra_extensions: List of strings with all the extra extensions that will be returned
                i.e. extensions=['.xml', '.bcl'] will return [wav, xml and bcl] files
        """
        self.folder_path = pathlib.Path(folder_path)
        self.extension = extension
        if not self.folder_path.exists():
            raise FileNotFoundError(
                "The path %s does not exist. Please choose another one." % folder_path
            )
        if len(list(self.folder_path.glob("**/*%s" % self.extension))) == 0:
            raise ValueError(
                "The directory %s is empty. Please select another directory with *.%s files"
                % (folder_path, self.extension)
            )
        self.zipped = zipped
        self.recursive = include_dirs
        if extra_extensions is None:
            extra_extensions = []
        self.extra_extensions = extra_extensions

        if not self.zipped:
            if self.recursive:
                self.files_list = sorted(
                    self.folder_path.glob("**/*%s" % self.extension)
                )
            else:
                self.files_list = sorted(self.folder_path.glob("*%s" % self.extension))
        else:
            if self.recursive:
                self.folder_list = sorted(self.folder_path.iterdir())
                self.files_list = []
                for fol in self.folder_list:
                    self.zipped_subfolder = AcousticFolder(
                        fol,
                        extra_extensions=self.extra_extensions,
                        zipped=self.zipped,
                        include_dirs=self.recursive,
                    )
                    np.concatenate((self.files_list, self.zipped_subfolder.files_list))
            else:
                zipped_folder = zipfile.ZipFile(self.folder_path, "r", allowZip64=True)
                self.files_list = []
                total_files_list = zipped_folder.namelist()
                for f in total_files_list:
                    extension = f.split(".")[-1]
                    if extension == "wav":
                        self.files_list.append(f)

    def __getitem__(self, index):
        """
        Get n wav file
        """
        if index < len(self.files_list):
            files_list = []
            if self.zipped:
                file_name = self.files_list[index]
                zipped_folder = zipfile.ZipFile(self.folder_path, "r", allowZip64=True)
                wav_file = zipped_folder.open(file_name)
                files_list.append(wav_file)
                for extension in self.extra_extensions:
                    ext_file_name = file_name.parent.joinpath(
                        file_name.name.replace(self.extension, extension)
                    )
                    files_list.append(zipped_folder.open(ext_file_name))
                return files_list
            else:
                wav_path = self.files_list[index]
                files_list.append(wav_path)
                for extension in self.extra_extensions:
                    files_list.append(
                        pathlib.Path(str(wav_path).replace(self.extension, extension))
                    )

                return files_list
        else:
            raise IndexError

    def __len__(self):
        if not self.zipped:
            if self.recursive:
                n_files = len(list(self.folder_path.glob("**/*%s" % self.extension)))
            else:
                n_files = len(list(self.folder_path.glob("*%s" % self.extension)))
        else:
            if self.recursive:
                n_files = len(list(self.folder_path.iterdir()))
            else:
                zipped_folder = zipfile.ZipFile(self.folder_path, "r", allowZip64=True)
                n_files = len(zipped_folder.namelist())
        return n_files


def move_file(file_path: str or pathlib.Path, new_folder_path: str or pathlib.Path):
    """
    Move the file to the new folder

    Args:
        file_path: Original file path
        new_folder_path: New folder destination (without the file name)
    """
    if not isinstance(file_path, pathlib.Path):
        file_path = pathlib.Path(file_path)
    if not isinstance(new_folder_path, pathlib.Path):
        new_folder_path = pathlib.Path(new_folder_path)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    new_path = new_folder_path.joinpath(file_path.name)
    os.rename(file_path, new_path)
