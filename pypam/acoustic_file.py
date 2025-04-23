import datetime
import operator
import os
import pathlib
import zipfile

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhydrophone as pyhy
import pytz
import seaborn as sns
import soundfile as sf
import xarray
from tqdm.auto import tqdm

from pypam import plots
from pypam import signal as sig
from pypam import units as output_units
from pypam import utils

pd.plotting.register_matplotlib_converters()
plt.rcParams.update({"pcolor.shading": "auto"})

# Apply the default theme
sns.set_theme()


class AcuFile:
    """
    Data recorded in a wav file.
    """

    def __init__(
        self,
        sfile: sf.SoundFile or pathlib.Path,
        hydrophone: pyhy.Hydrophone,
        p_ref: float,
        timezone: datetime.tzinfo
        or pytz.tzinfo.BaseTZInfo
        or dateutil.tz.tz.tzfile
        or str
        or None = "UTC",
        channel: int = 0,
        calibration: str or None or int or float = None,
        dc_subtract: bool = False,
    ):
        """

        Args:
            sfile: Can be a path or a file object, referring to the audio file to analyze
            hydrophone: Object for the class hydrophone
            p_ref: Reference pressure in upa
            timezone: Timezone where the data was recorded in
            channel: Channel to perform the calculations in
            calibration: If it is a float, it is the time ignored at the beginning of the file. If None, nothing is
            done. If negative, the function calibrate from the hydrophone is performed, and the first samples ignored
            (and hydrophone object's metadata updated)
            dc_subtract: Set to True to subtract the dc noise (root mean squared value
        """
        # Save hydrophone model
        self.hydrophone = hydrophone

        # Get the date from the name
        if type(sfile) == str:
            file_name = os.path.split(sfile)[-1]
        elif issubclass(sfile.__class__, pathlib.Path):
            file_name = sfile.name
        elif issubclass(sfile.__class__, zipfile.ZipExtFile):
            file_name = sfile.name
        else:
            raise Exception("The filename has to be either a Path object or a string")

        try:
            self.date = hydrophone.get_name_datetime(file_name)
        except ValueError:
            self.date = datetime.datetime.now()
            print(
                "Filename %s does not match the %s file structure. Setting time to now..."
                % (file_name, self.hydrophone.name)
            )

        # Signal
        self.file_path = sfile
        self.file = sf.SoundFile(self.file_path, "r")
        self.fs = self.file.samplerate

        # Reference pressure in upa
        self.p_ref = p_ref

        # Work on local time or UTC time
        self.timezone = timezone

        # datetime will always be in UTC
        self.datetime_timezone = "UTC"

        # Select channel
        self.channel = channel

        # Set an empty wav array
        self.wav = None
        self.time = None

        # Set a starting frame for the file
        if calibration is None:
            self._start_frame = 0
        elif calibration < 0:
            self._start_frame = self.hydrophone.calibrate(self.file_path)
            if self._start_frame is None:
                self._start_frame = 0
        else:
            self._start_frame = int(calibration * self.fs)
        self.calibration = calibration

        self.dc_subtract = dc_subtract

    def __getattr__(self, name: str):
        """
        Specific methods to make it easier to access attributes
        """
        if name == "signal":
            return self.signal("upa")
        elif name == "time":
            if self.__dict__[name] is None:
                return self._time_array(binsize=1 / self.fs)[0]
            else:
                return self.__dict__[name]
        else:
            return self.__dict__[name]

    def _bins(self, binsize: float = None, bin_overlap: float = 0) -> tuple:
        """
        Iterates through all the bins, yields (i, time_bin, signal, start_sample, end_sample), where i is the index,
        time_bin is the datetime of the beginning of the block and signal is the signal object of the bin

        Args:
            binsize: Number of seconds per bin to yield. If set to None, a single bin is yield for the entire file
            bin_overlap: Percentage to overlap the bin windows

        Returns:
             i: index
             time_bin: datetime of the beginning of the block
             signal: signal object of the time bin
             start_sample: number sample to start
             end_sample: number sample to end
        """
        if bin_overlap > 1:
            raise ValueError(f"bin_overlap must be fractional.")
        if binsize is None:
            blocksize = self.file.frames - self._start_frame
        else:
            blocksize = self.samples(binsize)
        noverlap = int(bin_overlap * blocksize)
        n_blocks = self._n_blocks(blocksize, noverlap=noverlap)
        time_array, _, _ = self._time_array(binsize, bin_overlap=bin_overlap)
        for i, block in tqdm(
            enumerate(
                sf.blocks(
                    self.file_path,
                    blocksize=blocksize,
                    start=self._start_frame,
                    overlap=noverlap,
                    always_2d=True,
                    fill_value=0.0,
                )
            ),
            total=n_blocks,
            leave=False,
            position=0,
        ):
            # Select the desired channel
            block = block[:, self.channel]
            time_bin = time_array[i]
            # Read the signal and prepare it for analysis
            signal_upa = self.wav2upa(wav=block)
            signal = sig.Signal(signal=signal_upa, fs=self.fs, channel=self.channel)
            if self.dc_subtract:
                signal.remove_dc()
            step = blocksize - noverlap
            start_sample = i * step + self._start_frame
            end_sample = start_sample + blocksize
            yield i, time_bin, signal, start_sample, end_sample
        self.file.seek(0)

    def _n_blocks(self, blocksize: int, noverlap: int):
        return int(
            np.floor(self.file.frames - self._start_frame) / (blocksize - noverlap)
        )

    def samples(self, bintime: float):
        """
        Return the samples according to the fs

        Args:
            bintime: Seconds of bintime desired to convert2324.sh to samples
        """
        return int(bintime * self.fs)

    def set_calibration_time(self, calibration_time: float):
        """
        Set a calibration time in seconds. This time will be ignored in the processing

        Args:
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

    def samples2time(self, samples: int):
        """
        Return the samples according to the fs

        Args:
            samples: Number of samples to convert to seconds
        """
        return float(samples) / self.fs

    def is_in_period(self, period: list or tuple):
        """
        Return True if the WHOLE file is included in the specified period

        Args:
            period: list or a tuple with (start, end). Values have to be a datetime object
        """
        if period is None:
            return True
        else:
            return (self.date >= period[0]) & (self.date <= period[1])

    def contains_date(self, date: datetime.datetime):
        """
        Return True if data is included in the file

        Args:
            date: Datetime to check
        """
        end = self.date + datetime.timedelta(seconds=self.file.frames / self.fs)
        return (self.date < date) & (end > date)

    def split(self, date: datetime.date):
        """
        Save two different files out of one splitting on the specified date

        Args:
            date: Datetime where to split the file
        """
        if issubclass(self.file_path, zipfile.ZipExtFile):
            raise Exception("The split method is not implemented for zipped files")
        if not self.contains_date(date):
            raise Exception("This date is not included in the file!")
        else:
            self.file.seek(0)
        seconds = (date - self.date).seconds
        frames = self.samples(seconds)
        first_file = self.file.read(frames=frames)
        second_file = self.file.read()
        self.file.close()

        new_file_name = self.hydrophone.get_new_name(
            filename=self.file_path.name, new_date=date
        )
        new_file_path = self.file_path.parent.joinpath(new_file_name)
        sf.write(self.file_path, first_file, samplerate=self.fs)
        sf.write(new_file_path, second_file, samplerate=self.fs)

        return self.file_path, new_file_path

    def freq_resolution_window(self, freq_resolution: int) -> int:
        """
        Given the frequency resolution, window length needed to obtain it
        Returns window length in samples

        Args:
        freq_resolution: Must be a power of 2, in Hz
        """
        n = np.log2(self.fs / freq_resolution)
        nfft = 2**n
        if nfft > self.file.frames:
            raise Exception(
                "This is not achievable with this sampling rate, "
                "it must be downsampled!"
            )
        return nfft

    def signal(self, units: str = "wav") -> np.array:
        """
        Returns the signal in the specified units

        Args:
            units: Units in which to return the signal. Can be 'wav', 'db', 'upa', 'Pa' or 'acc'.
        """
        # First time, read the file and store it to not read it over and over
        if self.wav is None:
            self.wav = self.file.read()
            self.file.seek(0)
        if units == "wav":
            signal = self.wav
        elif units == "db":
            signal = self.wav2db()
        elif units == "upa":
            signal = self.wav2upa()
        elif units == "Pa":
            signal = self.wav2upa() / 1e6
        elif units == "acc":
            signal = self.wav2acc()
        else:
            raise Exception("%s is not implemented as an outcome unit" % units)

        return signal

    def _time_array(self, binsize: float = None, bin_overlap: float = 0) -> tuple:
        """
        Return a time array for each point of the signal
        """
        if binsize is None:
            total_block = self.file.frames - self._start_frame
        else:
            total_block = self.samples(binsize)
        blocksize = total_block - int(total_block) * bin_overlap
        blocks_samples = np.arange(
            start=self._start_frame, stop=self.file.frames - 1, step=blocksize
        )
        end_samples = blocks_samples + blocksize
        incr = pd.to_timedelta(blocks_samples / self.fs, unit="seconds")
        self.time = (
            self.date + datetime.timedelta(seconds=self._start_frame / self.fs) + incr
        )
        if self.timezone != "UTC":
            self.time = (
                pd.to_datetime(self.time)
                .tz_localize(self.timezone)
                .tz_convert("UTC")
                .tz_convert(None)
            )
        return self.time, blocks_samples.astype(int), end_samples.astype(int)

    def timestamp_da(
        self, binsize: float = None, bin_overlap: float = 0
    ) -> xarray.Dataset:
        time_array, start_samples, end_samples = self._time_array(binsize, bin_overlap)
        ds = xarray.Dataset(
            coords={
                "id": np.arange(len(time_array)),
                "datetime": ("id", time_array.values),
                "start_sample": ("id", start_samples),
                "end_sample": ("id", end_samples),
            }
        )
        return ds

    def wav2upa(self, wav: np.array = None) -> np.array:
        """
        Compute the pressure from the wav signal

        Args:
            wav: Signal in wav (-1 to 1)
        """
        # Read if no signal is passed
        if wav is None:
            wav = self.signal("wav")
        # First convert2324.sh it to Volts and then to Pascals according to sensitivity
        mv = 10 ** (self.hydrophone.sensitivity / 20.0) * self.p_ref
        ma = 10 ** (self.hydrophone.preamp_gain / 20.0) * self.p_ref
        gain_upa = (self.hydrophone.Vpp / 2.0) / (mv * ma)

        return utils.set_gain(wave=wav, gain=gain_upa)

    def wav2db(self, wav: np.array = None) -> np.array:
        """
        Compute the db from the wav signal. Consider the hydrophone sensitivity in db.
        If wav is None, it will read the whole file.

        Args:
            wav: Signal in wav (-1 to 1)
        """
        # Read if no signal is passed
        if wav is None:
            wav = self.signal("wav")
        upa = self.wav2upa(wav)
        return utils.to_db(wave=upa, ref=self.p_ref, square=True)

    def db2upa(self, db: np.array = None) -> np.array:
        """
        Compute the upa from the db signals. If db is None, it will read the whole file.

        Args:
            db: Signal in db
        """
        if db is None:
            db = self.signal("db")
        # return np.power(10, db / 20.0 - np.log10(self.p_ref))
        return utils.to_mag(wave=db, ref=self.p_ref)

    def upa2db(self, upa: np.array = None) -> np.array:
        """
        Compute the db from the upa signal. If upa is None, it will read the whole file.

        Args:
            upa: Signal in upa
        """
        if upa is None:
            upa = self.signal("upa")
        return utils.to_db(upa, ref=self.p_ref, square=True)

    def wav2acc(self, wav: np.array = None) -> np.array:
        """
        Convert the wav file to acceleration. If wav is None, it will read the whole file.

        Args:
            wav: Signal in wav (-1 to 1)
        """
        if wav is None:
            wav = self.file.read()
        mv = 10 ** (self.hydrophone.mems_sensitivity / 20.0)
        return wav / mv

    def _get_metadata_attrs(self):
        metadata_keys = [
            "file_path",
            "timezone",
            "datetime_timezone",
            "p_ref",
            "channel",
            "_start_frame",
            "calibration",
            "dc_subtract",
            "fs",
            "hydrophone.name",
            "hydrophone.model",
            "hydrophone.sensitivity",
            "hydrophone.preamp_gain",
            "hydrophone.Vpp",
        ]

        metadata_attrs = {}
        for k in metadata_keys:
            d = self
            for sub_k in k.split("."):
                d = d.__dict__[sub_k]
            if isinstance(d, pathlib.Path):
                d = str(d)
            if d is None:
                d = 0
            metadata_attrs[k.replace(".", "_")] = d

        return metadata_attrs

    def _apply_multiple(
        self,
        method_list: list,
        binsize: float = None,
        band_list: list = None,
        bin_overlap: float = 0,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Apply multiple methods per bin to save computational time

        Args:
            method_list: List of all the methods to apply as strings
            band_list: list of tuples, tuple or None of the bands to filter. Can be multiple bands (all of them will be analyzed) or only one band. A band is
                represented with a tuple as (low_freq, high_freq). If set to None, the broadband up to the Nyquist
                frequency will be analyzed
            binsize: Length in seconds of the bins to analyze
            bin_overlap: Percentage to overlap the bin windows [0 to 1]
            kwargs: any parameters that have to be passed to the methods

        Returns:
            DataFrame with time as index and a multiindex column with band, method as levels.
        """
        # TODO decide if it is downsampled or not
        downsample = False

        # Bands selected to study
        if band_list is None:
            band_list = [[0, self.fs / 2]]

        # Sort bands to diminish downsampling efforts!
        sorted_bands = []
        for band in band_list:
            if len(sorted_bands) == 0:
                sorted_bands = [band]
            else:
                if band[1] >= sorted_bands[-1][1]:
                    sorted_bands = [band] + sorted_bands
                else:
                    sorted_bands = sorted_bands + [band]
        log = True
        if "db" in kwargs.keys():
            if not kwargs["db"]:
                log = False
        # Define an empty dataset
        ds = xarray.Dataset()
        for i, time_bin, signal, start_sample, end_sample in self._bins(
            binsize, bin_overlap=bin_overlap
        ):
            ds_bands = xarray.Dataset()
            for j, band in enumerate(sorted_bands):
                signal.set_band(band, downsample=downsample)
                methods_output = xarray.Dataset()
                for method_name in method_list:
                    f = operator.methodcaller(method_name, **kwargs)
                    try:
                        output = f(signal)
                    except Exception as e:
                        print(
                            "There was an error in band %s, feature %s. Setting to None. "
                            "Error: %s" % (band, method_name, e)
                        )
                        output = None

                    units_attrs = output_units.get_units_attrs(
                        method_name=method_name, log=log, p_ref=self.p_ref, **kwargs
                    )
                    methods_output[method_name] = xarray.DataArray(
                        [[output]],
                        coords={
                            "id": [i],
                            "datetime": ("id", [time_bin]),
                            "start_sample": ("id", [start_sample]),
                            "end_sample": ("id", [end_sample]),
                            "band": [j],
                            "low_freq": ("band", [band[0]]),
                            "high_freq": ("band", [band[1]]),
                        },
                        dims=["id", "band"],
                        attrs=units_attrs,
                    )
                if j == 0:
                    ds_bands = methods_output
                else:
                    ds_bands = xarray.concat((ds_bands, methods_output), "band")
            if i == 0:
                ds = ds_bands
            else:
                ds = xarray.concat((ds, ds_bands), "id")
        ds.attrs = self._get_metadata_attrs()
        return ds

    def _apply(
        self,
        method_name: str,
        binsize: float = None,
        db: bool = True,
        band_list: list = None,
        bin_overlap: float = 0,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Apply one single method

        Args:
            method_name: Name of the method to apply
            binsize: Time window considered in seconds. If set to None, only one value is returned
            overlap: Percentage to overlap the bin windows [0 to 1]
            db: If set to True the result will be given in db, otherwise in upa
        """
        return self._apply_multiple(
            method_list=[method_name],
            binsize=binsize,
            bin_overlap=bin_overlap,
            db=db,
            band_list=band_list,
            **kwargs,
        )

    def rms(
        self, binsize: float = None, bin_overlap: float = 0, db: bool = True
    ) -> xarray.Dataset:
        """
        Calculation of root mean squared value (rms) of the signal in upa for each bin
        Returns Dataset with 'datetime' as coordinate and 'rms' value as a variable

        Args:
            binsize: Time window considered in seconds. If set to None, only one value is returned
            bin_overlap:  Percentage to overlap the bin windows [0 to 1]
            db: If set to True the result will be given in db, otherwise in upa
        """
        rms_ds = self._apply(
            method_name="rms", binsize=binsize, bin_overlap=bin_overlap, db=db
        )
        return rms_ds

    def kurtosis(self, binsize: float = None, bin_overlap: float = 0) -> xarray.Dataset:
        """
        Calculation of kurtosis value of the signal for each bin
        Returns Dataset with 'datetime' as coordinate and 'kurtosis' value as a variable

        Args:
            binsize: Time window considered in seconds. If set to None, only one value is returned
            bin_overlap:  Percentage to overlap the bin windows [0 to 1]
        """
        kurtosis_ds = self._apply(
            method_name="kurtosis", binsize=binsize, bin_overlap=bin_overlap, db=False
        )
        return kurtosis_ds

    def aci(
        self,
        binsize: float = None,
        bin_overlap: float = 0,
        nfft: int = 1024,
        fft_overlap: float = 0.5,
    ) -> xarray.Dataset:
        """
        Calculation of root mean squared value (rms) of the signal in upa for each bin
        Returns Dataset with 'datetime' as coordinate and 'aci' value as a variable

        Args:
            binsize: Time window considered in seconds. If set to None, only one value is returned
            bin_overlap: Percentage to overlap the bin windows [0 to 1]
            nfft: Window size for processing
            fft_overlap : Percentage to overlap the bin windows [0 to 1]
        """
        aci_ds = self._apply(
            method_name="aci",
            binsize=binsize,
            bin_overlap=bin_overlap,
            nfft=nfft,
            fft_overlap=fft_overlap,
        )
        return aci_ds

    def dynamic_range(
        self, binsize: float = None, bin_overlap: float = 0, db: bool = True
    ) -> xarray.Dataset:
        """
        Compute the dynamic range of each bin
        Returns a Dataset with 'datetime' as coordinate and 'dr' as variable

        Args:
            binsize: Time window considered in seconds. If set to None, only one value is returned
            bin_overlap:  Percentage to overlap the bin windows [0 to 1]
            db: If set to True the result will be given in db, otherwise in upa
        """
        dr_ds = self._apply(
            method_name="dynamic_range", binsize=binsize, bin_overlap=bin_overlap, db=db
        )
        return dr_ds

    def cumulative_dynamic_range(
        self, binsize: float = None, bin_overlap: float = 0, db: bool = True
    ) -> xarray.Dataset:
        """
        Compute the cumulative dynamic range for each bin

        Args:
            binsize: Time window considered in seconds. If set to None, only one value is returned
            bin_overlap:  Percentage to overlap the bin windows [0 to 1]
            db: If set to True the result will be given in db, otherwise in upa

        Returns:
              Dataset with an extra column with the cumulative sum of dynamic range of each bin
        """
        cumdr = self.dynamic_range(binsize=binsize, bin_overlap=bin_overlap, db=db)
        cumdr["cumsum_dr"] = cumdr.dr.cumsum()
        return cumdr

    def octaves_levels(
        self,
        binsize: float = None,
        bin_overlap: float = 0,
        db: bool = True,
        band: list or tuple = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Returns the octave levels

        Args:
        binsize: Length in seconds of the bin to analyze
        bin_overlap: Percentage to overlap the bin windows [0 to 1]
        db: Set to True if the result should be in decibels
        band: List or tuple of [low_frequency, high_frequency]

        Returns:
            Dataset with multiindex columns with levels method and band. The method is '3-oct'

        """
        return self._octaves_levels(
            fraction=1, binsize=binsize, bin_overlap=bin_overlap, db=db, band=band
        )

    def third_octaves_levels(
        self,
        binsize: float = None,
        bin_overlap: float = 0,
        db: bool = True,
        band: list or tuple = None,
        **kwargs,
    ) -> xarray.Dataset:
        """
        Returns the octave levels

        Args:
            binsize: Length in seconds of the bin to analyze
            bin_overlap : Percentage to overlap the bin windows [0 to 1]
            db: Set to True if the result should be in decibels
            band: List or tuple of [low_frequency, high_frequency]

        Returns:
            Dataset with multiindex columns with levels method and band. The method is '3-oct'
        """
        return self._octaves_levels(
            fraction=3, binsize=binsize, bin_overlap=bin_overlap, db=db, band=band
        )

    def _octaves_levels(
        self,
        fraction: int = 1,
        binsize: float = None,
        bin_overlap: float = 0,
        db: bool = True,
        band: list or tuple = None,
    ) -> xarray.Dataset:
        """
        Returns the octave levels

        Args:
            fraction: Fraction of the desired octave. Set to 1 for octave bands, set to 3 for 1/3-octave bands
            binsize: Length in seconds of the bin to analyze
            bin_overlap : Percentage to overlap the bin windows [0 to 1]
            db: Set to True if the result should be in decibels
            band: List or tuple of [low_frequency, high_frequency]

        Returns:
            Dataset with multiindex columns with levels method and band. The method is '3-oct'
        """
        downsample = True

        if band is None:
            band = [None, self.fs / 2]
        oct_str = "oct%s" % fraction

        # Create an empty dataset
        da = xarray.DataArray()
        units_attrs = output_units.get_units_attrs(
            method_name="octave_levels", p_ref=self.p_ref, log=db
        )
        for i, time_bin, signal, start_sample, end_sample in self._bins(
            binsize, bin_overlap=bin_overlap
        ):
            signal.set_band(band, downsample=downsample)
            fbands, levels = signal.octave_levels(db, fraction)
            da_levels = xarray.DataArray(
                data=[levels],
                coords={
                    "id": [i],
                    "start_sample": ("id", [start_sample]),
                    "end_sample": ("id", [end_sample]),
                    "datetime": ("id", [time_bin]),
                    "frequency": fbands,
                },
                dims=["id", "frequency"],
            )
            if i == 0:
                da = da_levels
            else:
                da = xarray.concat((da, da_levels), "id")
        da.attrs.update(units_attrs)
        ds = xarray.Dataset(data_vars={oct_str: da}, attrs=self._get_metadata_attrs())
        return ds

    def hybrid_millidecade_bands(
        self,
        nfft: int,
        fft_overlap: float = 0.5,
        binsize: float = None,
        bin_overlap: float = 0,
        db: bool = True,
        method: str = "density",
        band: list or tuple = None,
        percentiles: list or np.array = None,
    ) -> xarray.Dataset:
        """
        Returns the hybrid millidecade bands

        Args:
            binsize: Time window considered in seconds. If set to None, only one value is returned
            bin_overlap: Percentage to overlap the bin windows [0 to 1]
            nfft: Length of the fft window in samples. Power of 2.
            fft_overlap: Percentage to overlap the bin windows [0 to 1]
            db: If set to True the result will be given in db, otherwise in upa^2
            method: Can be 'spectrum' or 'density'
            band: Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
                (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed
            percentiles : List of all the percentiles that have to be returned. If set to empty list,
                no percentiles is returned

        Returns:
            xarray Dataset
        """

        if band is None:
            band = [0, self.fs / 2]
        spectra_ds = self._spectrum(
            scaling=method,
            binsize=binsize,
            nfft=nfft,
            fft_overlap=fft_overlap,
            db=False,
            bin_overlap=bin_overlap,
            percentiles=percentiles,
            band=band,
        )
        (
            millidecade_bands_limits,
            millidecade_bands_c,
        ) = utils.get_hybrid_millidecade_limits(band, nfft)
        fft_bin_width = band[1] * 2 / nfft
        hybrid_millidecade_ds = utils.spectra_ds_to_bands(
            spectra_ds["band_%s" % method],
            millidecade_bands_limits,
            millidecade_bands_c,
            fft_bin_width=fft_bin_width,
            db=db,
        )
        spectra_ds["millidecade_bands"] = hybrid_millidecade_ds
        return spectra_ds

    def spectrogram(
        self,
        binsize: float = None,
        bin_overlap: float = 0,
        nfft: int = 512,
        fft_overlap: float = 0.5,
        scaling: str = "density",
        db: bool = True,
        band: list or tuple = None,
    ) -> xarray.Dataset:
        """
        Return the spectrogram of the signal (entire file)

        Args:
            binsize: Time window considered. If set to None, only one value is returned
            bin_overlap: ercentage to overlap the bin windows [0 to 1]
            db: If set to True the result will be given in db, otherwise in upa^2
            nfft: Length of the fft window in samples. Power of 2.
            fft_overlap: Percentage to overlap the bin windows [0 to 1]
            scaling: Can be set to 'spectrum' or 'density' depending on the desired output
            band: Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
                (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed
        """
        downsample = True
        if band is None:
            band = [None, self.fs / 2]

        da = xarray.DataArray()
        for i, time_bin, signal, start_sample, end_sample in self._bins(
            binsize, bin_overlap=bin_overlap
        ):
            signal.set_band(band, downsample=downsample)
            freq, t, sxx = signal.spectrogram(
                nfft=nfft, overlap=fft_overlap, scaling=scaling, db=db
            )
            da_sxx = xarray.DataArray(
                [sxx],
                coords={
                    "id": [i],
                    "start_sample": ("id", [start_sample]),
                    "end_sample": ("id", [end_sample]),
                    "datetime": ("id", [time_bin]),
                    "frequency": freq,
                    "time": t,
                },
                dims=["id", "frequency", "time"],
            )
            if i == 0:
                da = da_sxx
            else:
                da = xarray.concat((da, da_sxx), "id")
        units_attrs = output_units.get_units_attrs(
            method_name="spectrogram_" + scaling, p_ref=self.p_ref, log=db
        )
        da.attrs.update(units_attrs)
        ds = xarray.Dataset(
            data_vars={"spectrogram": da}, attrs=self._get_metadata_attrs()
        )
        return ds

    def _spectrum(
        self,
        scaling: str = "density",
        binsize: float = None,
        bin_overlap: float = 0,
        nfft: int = 512,
        fft_overlap: float = 0.5,
        db: bool = True,
        percentiles: list or np.array = None,
        band: list or tuple = None,
    ):
        """
        Return the spectrum (frequency distribution of every bin (periodogram))


        Args:
            scaling:  Can be set to 'spectrum' or 'density' depending on the desired output
            binsize: Time window considered in seconds. If set to None, only one value is returned
            bin_overlap: Percentage to overlap the bin windows [0 to 1]
            nfft: Length of the fft window in samples. Power of 2.
            fft_overlap: Percentage to overlap the bin windows [0 to 1]
            db: If set to True the result will be given in db, otherwise in upa^2
            percentiles: List of all the percentiles that have to be returned. If set to empty list,
                no percentiles is returned
            band: Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
                (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed

        Returns:
             xarray Dataset
        """
        downsample = True
        if percentiles is None:
            percentiles = []
        if band is None:
            band = [None, self.fs / 2]

        spectrum_str = "band_" + scaling
        ds = xarray.DataArray()
        for i, time_bin, signal, start_sample, end_sample in self._bins(
            binsize, bin_overlap=bin_overlap
        ):
            signal.set_band(band, downsample=downsample)
            fbands, spectra, percentiles_val = signal.spectrum(
                scaling=scaling,
                nfft=nfft,
                db=db,
                percentiles=percentiles,
                overlap=fft_overlap,
            )

            spectra_da = xarray.DataArray(
                [spectra],
                coords={
                    "id": [i],
                    "datetime": ("id", [time_bin]),
                    "frequency": fbands,
                    "start_sample": ("id", [start_sample]),
                    "end_sample": ("id", [end_sample]),
                },
                dims=["id", "frequency"],
            )
            percentiles_da = xarray.DataArray(
                [percentiles_val],
                coords={
                    "id": [i],
                    "datetime": ("id", [time_bin]),
                    "percentiles": percentiles,
                },
                dims=["id", "percentiles"],
            )

            ds_bin = xarray.Dataset(
                {spectrum_str: spectra_da, "value_percentiles": percentiles_da}
            )
            if i == 0:
                ds = ds_bin
            else:
                ds = xarray.concat((ds, ds_bin), "id")
        units_attrs = output_units.get_units_attrs(
            method_name="spectrum_" + scaling, log=db, p_ref=self.p_ref
        )
        ds[spectrum_str].attrs.update(units_attrs)
        ds["value_percentiles"].attrs.update(
            {"units": "%", "standard_name": "percentiles"}
        )
        ds.attrs = self._get_metadata_attrs()
        return ds

    def psd(
        self,
        binsize: float = None,
        bin_overlap: float = 0,
        nfft: int = 512,
        fft_overlap: float = 0.5,
        db: bool = True,
        percentiles: list or np.array = None,
        band: list or tuple = None,
    ):
        """
        Return the power spectrum density (PSD) of all the file (units^2 / Hz) re 1 V 1 upa

        Args:
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        bin_overlap : float [0 to 1]
            Percentage to overlap the bin windows
        nfft : int
            Length of the fft window in samples. Recommended power of 2.
        fft_overlap : float [0 to 1]
            Percentage to overlap the bin windows
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : list or None
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned
        band : tuple or None
            Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
            (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed

        Returns:
            xarray Dataset
        """
        psd_ds = self._spectrum(
            scaling="density",
            binsize=binsize,
            nfft=nfft,
            fft_overlap=fft_overlap,
            db=db,
            bin_overlap=bin_overlap,
            percentiles=percentiles,
            band=band,
        )
        return psd_ds

    def power_spectrum(
        self,
        binsize: float = None,
        bin_overlap: float = 0,
        nfft: int = 512,
        fft_overlap: float = 0.5,
        db: bool = True,
        percentiles: list or np.array = None,
        band: list or tuple = None,
    ):
        """
        Return the power spectrum of all the file (units^2 / Hz) re 1 V 1 upa
        Returns a Dataframe with 'datetime' as index and a column for each frequency and
        each percentile

        Args:
            binsize: Time window considered in seconds. If set to None, only one value is returned
            bin_overlap: Percentage to overlap the bin windows
            nfft: Length of the fft window in samples. Power of 2.
            fft_overlap: Percentage to overlap the windows in the fft [0 to 1]
            db: If set to True the result will be given in db, otherwise in upa^2
            percentiles: List of all the percentiles that have to be returned. If set to empty list,
                no percentiles is returned
            band: Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
                (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed
        """

        spectrum_ds = self._spectrum(
            scaling="spectrum",
            binsize=binsize,
            nfft=nfft,
            fft_overlap=fft_overlap,
            db=db,
            bin_overlap=bin_overlap,
            percentiles=percentiles,
            band=band,
        )
        return spectrum_ds

    def spd(
        self,
        binsize: float = None,
        bin_overlap: float = 0,
        h: float = 0.1,
        nfft: int = 512,
        fft_overlap: float = 0.5,
        db: bool = True,
        percentiles: list or np.array = None,
        min_val: float = None,
        max_val: float = None,
        band: list or tuple = None,
    ):
        """
        Return the spectral probability density.

        Args:
            binsize: Time window considered in seconds. If set to None, only one value is returned
            bin_overlap:  Percentage to overlap the bin windows [0 to 1]
            h: Histogram bin width (in the correspondent units, upa or db)
            nfft: Length of the fft window in samples. Power of 2.
            fft_overlap: Percentage to overlap the bin windows [0 to 1]
            db: If set to True the result will be given in db, otherwise in upa^2
            min_val: Minimum value to compute the SPD histogram
            max_val: Maximum value to compute the SPD histogram
            percentiles: List of all the percentiles that have to be returned. If set to empty list,
                no percentiles is returned
            band: Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
                (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed
        """
        psd_evolution = self.psd(
            binsize=binsize,
            nfft=nfft,
            fft_overlap=fft_overlap,
            db=db,
            percentiles=percentiles,
            bin_overlap=bin_overlap,
            band=band,
        )
        return utils.compute_spd(
            psd_evolution,
            h=h,
            percentiles=percentiles,
            max_val=max_val,
            min_val=min_val,
        )

    def plot_spectrum_median(
        self,
        scaling: str = "density",
        db: bool = True,
        log: bool = True,
        save_path: str or pathlib.Path = None,
        **kwargs,
    ):
        """
        Plot the power spectrogram density of all the file (units^2 / Hz) re 1 V 1 upa

        Args:
            scaling: 'density' or 'spectrum'
            db: If set to True the result will be given in db. Otherwise in upa^2/Hz
            log: If set to True the scale of the y axis is set to logarithmic
            save_path: Where to save the images
            **kwargs: any attribute valid on psd() function
        """
        psd = self._spectrum(db=db, scaling=scaling, **kwargs)
        plots.plot_spectrum_median(
            ds=psd, data_var="band_" + scaling, log=log, save_path=save_path
        )

    def plot_spectrum_per_chunk(
        self,
        scaling: str = "density",
        db: bool = True,
        log: bool = True,
        save_path: str or pathlib.Path = None,
        **kwargs,
    ):
        """
        Plot the power spectrogram density of all the file (units^2 / Hz) re 1 V 1 upa

        Args:
            scaling: 'density' or 'spectrum'
            db: If set to True the result will be given in db. Otherwise in upa^2/Hz
            log: If set to True the scale of the y axis is set to logarithmic
            save_path:  Where to save the images
            **kwargs : any attribute valid on psd() function
        """
        psd = self._spectrum(db=db, scaling=scaling, **kwargs)
        plots.plot_spectrum_per_chunk(
            ds=psd, data_var="band_" + scaling, log=log, save_path=save_path
        )

    def plot_spectrogram(
        self,
        db: bool = True,
        log: bool = True,
        save_path: str or pathlib.Path = None,
        **kwargs,
    ):
        """
        Return the spectrogram of the signal (entire file)

        Args:
            db: If set to True the result will be given in db. Otherwise in upa^2/Hz
            log: If set to True the scale of the y axis is set to logarithmic
            save_path: Where to save the images
            **kwargs: any attribute valid on spectrogram() function
        """
        ds_spectrogram = self.spectrogram(db=db, **kwargs)
        plots.plot_spectrogram_per_chunk(ds_spectrogram, log, save_path)

    def plot_spd(
        self,
        db: bool = True,
        log: bool = True,
        save_path: str or pathlib.Path = None,
        **kwargs,
    ):
        """
        Plot the SPD graph of the bin

        Args:
            db: If set to True the result will be given in db. Otherwise, in upa^2/Hz
            log: If set to True the scale of the y-axis is set to logarithmic
            save_path: Path where to save the images
            **kwargs : any attribute valid on spd() function
        """
        spd_ds = self.spd(db=db, **kwargs)
        plots.plot_spd(spd_ds, log=log, save_path=save_path)

    def update_freq_cal(self, ds: xarray.Dataset, data_var: str, **kwargs):
        return utils.update_freq_cal(
            hydrophone=self.hydrophone, ds=ds, data_var=data_var, **kwargs
        )
