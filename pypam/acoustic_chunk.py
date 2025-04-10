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
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import xarray
from tqdm.auto import tqdm

from pypam import plots
from pypam import signal as sig
from pypam import utils
from pypam import units as output_units

pd.plotting.register_matplotlib_converters()
plt.rcParams.update({"pcolor.shading": "auto"})

# Apply the default theme
sns.set_theme()


class AcuChunk:
    """
    Chunk recorded in a wav file.

    Parameters
    ----------
    sfile_start : Sound file start
        Can be a path or a file object
    sfile_end : Sound file end
        Can be a path or a file object
    start_frame : int
        Start frame on the start file
    end_frame : int
        End frame on the end file
    hydrophone : Object for the class hydrophone
    p_ref : Float
        Reference pressure in upa

    channel : int
        Channel to perform the calculations in
    dc_subtract: bool
        Set to True to subtract the dc noise (root mean squared value
    """

    def __init__(
        self,
        sfile_start,
        sfile_end,
        hydrophone,
        p_ref,
        time_bin,
        chunk_id,
        chunk_file_id,
        start_frame=0,
        end_frame=-1,
        channel=0,
        dc_subtract=False,
    ):
        # Save hydrophone model
        self.hydrophone = hydrophone

        # Get the date from the name
        # Signal
        self.file_start_path = sfile_start
        self.file_end_path = sfile_end
        self.file_start = sf.SoundFile(self.file_start_path, "r")
        self.file_end = sf.SoundFile(self.file_end_path, mode="r")
        self.fs = (
            self.file_start.samplerate
        )  # Should be the same for both files, only reading one!

        # Time and id
        self.time_bin = time_bin
        self.chunk_id = chunk_id
        self.chunk_file_id = chunk_file_id

        # Reference pressure in upa
        self.p_ref = p_ref

        # Select channel
        self.channel = channel
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.dc_subtract = dc_subtract

        self._signal = None

    def __getattr__(self, name):
        """
        Specific methods to make it easier to access attributes
        """
        if name == "signal":
            return self.read()
        elif name == "time":
            if self.__dict__[name] is None:
                return self._time_array(binsize=1 / self.fs)[0]
            else:
                return self.__dict__[name]
        else:
            return self.__dict__[name]

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
        nfft = 2**n
        if nfft > self.file.frames:
            raise Exception(
                "This is not achievable with this sampling rate, "
                "it must be downsampled!"
            )
        return nfft

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
            wav = self._signal.signal
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
            wav = self._signal.signal
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
            wav = self._signal.signal
            db = 10 * np.log10(wav**2)
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
            wav = self._signal.signal
            upa = self.wav2upa(wav)
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
            wav = self._signal.signal
        mv = 10 ** (self.hydrophone.mems_sensitivity / 20.0)
        return wav / mv

    def read(self):
        if self._signal is None:
            self.file_start.seek(self.start_frame)
            if self.file_start_path == self.file_end_path:
                signal = self.file_start.read(
                    self.end_frame - self.start_frame, always_2d=True
                )[:, self.channel]
            else:
                first_chunk = self.file_start.read(frames=-1, always_2d=True)[
                    :, self.channel
                ]
                self.file_end.seek(0)
                second_chunk = self.file_end.read(
                    frames=self.end_frame, always_2d=True
                )[:, self.channel]
                signal = np.concatenate((first_chunk, second_chunk))

            # Read the signal and prepare it for analysis
            signal_upa = self.wav2upa(wav=signal)
            signal = sig.Signal(signal=signal_upa, fs=self.fs, channel=self.channel)
            if self.dc_subtract:
                signal.remove_dc()
            self._signal = signal
        else:
            return self._signal

        return signal

    def apply_multiple(self, method_list, band_list=None, **kwargs):
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
        kwargs: any parameters that have to be passed to the methods

        Returns
        -------
        DataFrame with time as index and a multiindex column with band, method as levels.
        """
        signal = self.signal
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
                        "id": [self.chunk_id],
                        "file_id": ("id", [self.chunk_file_id]),
                        "datetime": ("id", [self.time_bin]),
                        "start_sample": ("id", [self.start_frame]),
                        "end_sample": ("id", [self.end_frame]),
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

        return ds_bands

    def octaves_levels(self, fraction=1, db=True, band=None):
        """
        Return the octave levels
        Parameters
        ----------
        fraction: int
            Fraction of the desired octave. Set to 1 for octave bands, set to 3 for 1/3-octave bands
        db: boolean
            Set to True if the result should be in decibels

        Returns
        -------
        DataFrame with multiindex columns with levels method and band. The method is '3-oct'

        """
        downsample = True

        if band is None:
            band = [None, self.fs / 2]

        # Create an empty dataset
        signal = self.signal
        signal.set_band(band, downsample=downsample)
        fbands, levels = signal.octave_levels(db, fraction)
        da_levels = xarray.DataArray(
            data=[levels],
            coords={
                "id": [self.chunk_id],
                "file_id": ("id", [self.chunk_file_id]),
                "start_sample": ("id", [self.start_frame]),
                "end_sample": ("id", [self.end_frame]),
                "datetime": ("id", [self.time_bin]),
                "frequency": fbands,
            },
            dims=["id", "frequency"],
        )
        return da_levels

    def hybrid_millidecade_bands(
        self,
        nfft,
        fft_overlap=0.5,
        db=True,
        method="density",
        band=None,
        percentiles=None,
    ):
        """

        Parameters
        ----------
        nfft : int
            Length of the fft window in samples. Power of 2.
        fft_overlap : float [0 to 1]
            Percentage to overlap the bin windows
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        method: string
            Can be 'spectrum' or 'density'
        band : tuple or None
            Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
            (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed
        percentiles : list or None
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned

        Returns
        -------

        """

        if band is None:
            band = [0, self.fs / 2]
        spectra_ds = self.spectrum(
            scaling=method,
            nfft=nfft,
            fft_overlap=fft_overlap,
            db=False,
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
        self, nfft=512, fft_overlap=0.5, scaling="density", db=True, band=None
    ):
        """
        Return the spectrogram of the signal (entire file)

        Parameters
        ----------
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        nfft : int
            Length of the fft window in samples. Power of 2.
        fft_overlap : float [0 to 1]
            Percentage to overlap the bin windows
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

        signal = self.signal
        signal.set_band(band, downsample=downsample)
        freq, t, sxx = signal.spectrogram(
            nfft=nfft, overlap=fft_overlap, scaling=scaling, db=db
        )
        da_sxx = xarray.DataArray(
            [sxx],
            coords={
                "id": [self.chunk_id],
                "file_id": ("id", [self.chunk_file_id]),
                "start_sample": ("id", [self.start_frame]),
                "end_sample": ("id", [self.end_frame]),
                "datetime": ("id", [self.time_bin]),
                "frequency": freq,
                "time": t,
            },
            dims=["id", "frequency", "time"],
        )
        return da_sxx

    def spectrum(
        self,
        scaling="density",
        nfft=512,
        fft_overlap=0.5,
        db=True,
        percentiles=None,
        band=None,
    ):
        """
        Return the spectrum : frequency distribution of every bin (periodogram)
        Returns Dataframe with 'datetime' as index and a column for each frequency and each
        percentile, and a frequency array

        Parameters
        ----------
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        nfft : int
            Length of the fft window in samples. Power of 2.
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
        """
        downsample = True
        if percentiles is None:
            percentiles = []
        if band is None:
            band = [None, self.fs / 2]

        spectrum_str = "band_" + scaling
        signal = self.signal
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
                "id": [self.chunk_id],
                "file_id": ("id", [self.chunk_file_id]),
                "datetime": ("id", [self.time_bin]),
                "frequency": fbands,
                "start_sample": ("id", [self.start_frame]),
                "end_sample": ("id", [self.end_frame]),
            },
            dims=["id", "frequency"],
        )
        percentiles_da = xarray.DataArray(
            [percentiles_val],
            coords={
                "id": [self.chunk_id],
                "file_id": ("id", [self.chunk_file_id]),
                "datetime": ("id", [self.time_bin]),
                "percentiles": percentiles,
            },
            dims=["id", "percentiles"],
        )

        ds_bin = xarray.Dataset(
            {spectrum_str: spectra_da, "value_percentiles": percentiles_da}
        )

        return ds_bin

    def spd(
        self,
        h=0.1,
        nfft=512,
        fft_overlap=0.5,
        db=True,
        percentiles=None,
        min_val=None,
        max_val=None,
        band=None,
    ):
        """
        Return the spectral probability density.

        Parameters
        ----------
        h : float
            Histogram bin width (in the correspondent units, upa or db)
        nfft : int
            Length of the fft window in samples. Power of 2.
        fft_overlap : float [0 to 1]
            Percentage to overlap the bin windows
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        min_val : float
            Minimum value to compute the SPD histogram
        max_val : float
            Maximum value to compute the SPD histogram
        percentiles : array_like
            List of all the percentiles that have to be returned. If set to empty list,
            no percentiles is returned
        band : tuple or None
            Band to filter the spectrogram in. A band is represented with a tuple - or a list - as
            (low_freq, high_freq). If set to None, the broadband up to the Nyquist frequency will be analyzed

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
        psd_evolution = self.psd(
            nfft=nfft,
            fft_overlap=fft_overlap,
            db=db,
            percentiles=percentiles,
            band=band,
        )
        return utils.compute_spd(
            psd_evolution,
            h=h,
            percentiles=percentiles,
            max_val=max_val,
            min_val=min_val,
        )

    def source_separation(
        self,
        window_time=1.0,
        n_sources=15,
        binsize=None,
        save_path=None,
        verbose=False,
        band=None,
    ):
        """
        Perform non-negative Matrix Factorization to separate sources

        Parameters
        ----------
        window_time: float
            window time to consider in seconds
        n_sources : int
            Number of sources
        binsize : float
            Time window considered, in seconds. If set to None, only one value is returned
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
        separator = nmf.NMF(
            window_time=window_time, rank=n_sources, save_path=save_path
        )
        ds = xarray.Dataset()
        for i, time_bin, signal, start_sample, end_sample in self._bins(
            binsize, bin_overlap=0.0
        ):
            signal.set_band(band)
            separation_ds = separator(signal, verbose=verbose)
            separation_ds = separation_ds.assign_coords(
                {"id": [i], "datetime": ("id", [time_bin])}
            )
            if i == 0:
                ds = separation_ds
            else:
                ds = xarray.concat((ds, separation_ds), "id")
        return ds

    def plot_spectrum_median(
        self, scaling="density", db=True, log=True, save_path=None, **kwargs
    ):
        """
        Plot the power spectrogram density of all the file (units^2 / Hz) re 1 V 1 upa

        Parameters
        ----------
        scaling : str
            'density' or 'spectrum'
        db : boolean
            If set to True the result will be given in db. Otherwise in upa^2/Hz
        log : boolean
            If set to True the scale of the y axis is set to logarithmic
        save_path : string or Path
            Where to save the images
        **kwargs : any attribute valid on psd() function
        """
        psd = self.spectrum(db=db, scaling=scaling, **kwargs)
        plots.plot_spectrum_median(
            ds=psd, data_var="band_" + scaling, log=log, save_path=save_path
        )

    def plot_spectrum_per_chunk(
        self, scaling="density", db=True, log=True, save_path=None, **kwargs
    ):
        """
        Plot the power spectrogram density of all the file (units^2 / Hz) re 1 V 1 upa

        Parameters
        ----------
        scaling : str
            'density' or 'spectrum'
        db : boolean
            If set to True the result will be given in db. Otherwise in upa^2/Hz
        log : boolean
            If set to True the scale of the y axis is set to logarithmic
        save_path : string or Path
            Where to save the images
        **kwargs : any attribute valid on psd() function
        """
        psd = self.spectrum(db=db, scaling=scaling, **kwargs)
        plots.plot_spectrum_per_chunk(
            ds=psd, data_var="band_" + scaling, log=log, save_path=save_path
        )

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
        if "scaling" not in kwargs.keys():
            scaling = "density"
        ds_spectrogram = self.spectrogram(db=db, **kwargs)
        ds_spectrogram = ds_spectrogram.squeeze(dim="id")
        units_attrs = output_units.get_units_attrs(
            method_name=f"spectrogram_{scaling}", p_ref=self.p_ref, log=db
        )
        ds_spectrogram.attrs.update(units_attrs)
        plots.plot_2d(
            ds=ds_spectrogram,
            x="time",
            y="frequency",
            xlabel="Time [s]",
            ylabel="Frequency [Hz]",
            cbar_label=r"%s [$%s$]"
            % (
                re.sub("_", " ", ds_spectrogram.standard_name).title(),
                ds_spectrogram.units,
            ),
            ylog=log,
            title=f"Chunk {ds_spectrogram.id.values}",
        )

    def plot_spd(self, db=True, log=True, save_path=None, **kwargs):
        """
        Plot the SPD graph of the bin

        Parameters
        ----------
        db : boolean
            If set to True the result will be given in db. Otherwise, in upa^2/Hz
        log : boolean
            If set to True the scale of the y-axis is set to logarithmic
        save_path : string or Path
            Where to save the images
        **kwargs : any attribute valid on spd() function
        """
        spd_ds = self.spd(db=db, **kwargs)
        plots.plot_spd(spd_ds, log=log, save_path=save_path)

    def update_freq_cal(self, ds, data_var, **kwargs):
        return utils.update_freq_cal(
            hydrophone=self.hydrophone, ds=ds, data_var=data_var, **kwargs
        )
