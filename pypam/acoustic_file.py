import datetime
import pathlib
import zipfile

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
import soundfile as sf
import xarray

from pypam import plots
from pypam import utils
from pypam import units as output_units
from pypam import acoustic_chunk

pd.plotting.register_matplotlib_converters()
plt.rcParams.update({"pcolor.shading": "auto"})

# Apply the default theme
sns.set_theme()


class AcuFile(acoustic_chunk.GenericAcuFile):
    """
    Data recorded in a wav file.
    """

    def __init__(
        self,
        sfile: sf.SoundFile or pathlib.Path,
        hydrophone: object,
        p_ref: float,
        timezone: datetime.tzinfo
        or pytz.tzinfo.BaseTZInfo
        or dateutil.tz.tz.tzfile
        or str
        or None = "UTC",
        channel: int = 0,
        calibration: str or None or int or float = None,
        dc_subtract: bool = False,
        sfile_next=None,
        gridded: bool = True,
        chunk_id_start: int = 0,
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
        super().__init__(
            sfile=sfile,
            hydrophone=hydrophone,
            p_ref=p_ref,
            timezone=timezone,
            channel=channel,
            dc_subtract=dc_subtract,
        )

        # Check if next file is continuous
        self.date_end = self.date + datetime.timedelta(seconds=self.total_time())
        self.file_path_next = None
        if sfile_next is not None:
            next_file_name = utils.parse_file_name(sfile_next)
            next_date = utils.get_file_datetime(next_file_name, self.hydrophone)

            if (next_date - self.date_end) < datetime.timedelta(seconds=1):
                self.file_path_next = sfile_next
                self.file_next = sf.SoundFile(self.file_path_next, "r")

        if gridded and calibration is not None:
            raise AttributeError(
                "The options gridded and calibration are not compatible"
            )
        self.calibration = calibration
        self.gridded = gridded

        if gridded:
            self._start_frame = int(
                (pd.to_datetime(self.date).ceil("min") - self.date).total_seconds()
                * self.fs
            )
        else:
            # Set a starting frame for the file
            if calibration is None:
                self._start_frame = 0
            elif calibration < 0:
                self._start_frame = self.hydrophone.calibrate(self.file_path)
                if self._start_frame is None:
                    self._start_frame = 0
            else:
                self._start_frame = int(calibration * self.fs)

        # Start to count chunk id's
        self.chunk_id_start = chunk_id_start

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
        if self.file_path_next is not None:
            is_there_next = True
        else:
            is_there_next = False
        n_blocks = self._n_blocks(blocksize, noverlap=noverlap, add_last=is_there_next)
        time_array, _, _ = self._time_array(binsize, bin_overlap=bin_overlap)
        chunk_id = self.chunk_id_start
        file_end = self.file_path
        for i in np.arange(n_blocks):
            time_bin = time_array[i]
            start_sample = i * (blocksize - noverlap) + self._start_frame
            if i == (n_blocks - 1):
                if is_there_next:
                    end_sample = blocksize - (self.file.frames - start_sample)
                    file_end = self.file_path_next
                else:
                    end_sample = self.file.frames
            else:
                end_sample = start_sample + blocksize
            chunk = acoustic_chunk.AcuChunk(
                sfile_start=self.file_path,
                sfile_end=file_end,
                start_frame=start_sample,
                end_frame=end_sample,
                hydrophone=self.hydrophone,
                p_ref=self.p_ref,
                chunk_id=chunk_id,
                chunk_file_id=i,
                time_bin=time_bin,
            )

            chunk_id += 1

            yield i, chunk

        self.file.seek(0)

    def _n_blocks(self, blocksize: int, noverlap: int, add_last: bool) -> int:
        if add_last:
            n_blocks = int(
                np.ceil((self.file.frames - self._start_frame) / (blocksize - noverlap))
            )
        else:
            n_blocks = int(
                np.floor(
                    (self.file.frames - self._start_frame) / (blocksize - noverlap)
                )
            )
        return n_blocks

    def set_calibration_time(self, calibration_time: float):
        """
        Set a calibration time in seconds. This time will be ignored in the processing

        Args:
        calibration_time : float
            Seconds to ignore at the beginning of the file
        """
        self._start_frame = int(calibration_time * self.fs)

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
        # Define an empty dataset
        ds = xarray.Dataset()
        for i, chunk in self._bins(binsize, bin_overlap=bin_overlap):
            ds_bands = chunk.apply_multiple(
                method_list=method_list, band_list=band_list, **kwargs
            )
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
        for i, chunk in self._bins(binsize, bin_overlap=bin_overlap):
            da_levels = chunk.octaves_levels(fraction=fraction, db=db, band=band)
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
        for i, chunk in self._bins(binsize, bin_overlap=bin_overlap):
            da_sxx = chunk.spectrogram(
                nfft=nfft, fft_overlap=fft_overlap, scaling=scaling, db=db
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
        if percentiles is None:
            percentiles = []
        if band is None:
            band = [None, self.fs / 2]

        spectrum_str = "band_" + scaling
        ds = xarray.DataArray()
        for i, chunk in self._bins(binsize, bin_overlap=bin_overlap):
            ds_bin = chunk.spectrum(
                scaling=scaling,
                nfft=nfft,
                fft_overlap=fft_overlap,
                db=db,
                percentiles=percentiles,
                band=band,
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
