"""
Utils
=====

The module ``utils`` is an ensemble of functions to re-process some of the outputs generated by `pypam`


To merge or re-index output
---------------------------

    reindexing_datetime
    join_all_ds_output_station
    join_all_ds_output_deployment
    select_datetime_range
    select_frequency_range
    merge_ds


To join frequency bands
-----------------------
    get_bands_limits
    get_hybrid_millidecade_limits
    spectra_ds_to_bands


SPD
---
    compute_spd

"""
import datetime
import pathlib
from functools import partial

import numba as nb
import numpy as np
import pandas as pd
import scipy.signal as sig
import xarray
from tqdm import tqdm

from functools import partial
import zipfile
import os
import datetime

try:
    import dask
    from dask.diagnostics import ProgressBar

except ModuleNotFoundError:
    dask = None

from pypam import units as output_units

G = 10.0 ** (3.0 / 10.0)
f_ref = 1000


@nb.njit
def sxx2spd(sxx: np.ndarray, h: float, bin_edges: np.ndarray):
    """
    Return spd from the spectrogram

    Args:
        sxx: Spectrogram
        h: Histogram bin width
        bin_edges: limits of the histogram bins
    """
    spd = np.zeros((sxx.shape[0], bin_edges.size - 1), dtype=np.float64)
    for i in nb.prange(sxx.shape[0]):
        spd[i, :] = np.histogram(sxx[i, :], bin_edges)[0] / ((sxx.shape[1]) * h)

    return spd


@nb.njit
def rms(signal: np.array) -> float:
    """
    Return the rms value of the signal

    Args:
        signal: Signal to compute the rms value

    Returns:
        RMS value of the signal
    """
    return np.sqrt(np.mean(signal**2))


@nb.njit
def dynamic_range(signal: np.array) -> float:
    """
    Return the dynamic range of the signal

    Args:
        signal: Signal to compute the dynamic range

    Returns:
        dynamic range value of the signal
    """
    return np.max(signal) - np.min(signal)


@nb.njit
def sel(signal: np.array, fs: int) -> float:
    """
    Return the Sound Exposure Level

    Args:
        signal: Signal to compute the SEL
        fs: Sampling frequency

    Returns:
        SEL
    """
    return np.sum(signal**2) / fs


@nb.njit
def peak(signal: np.array) -> float:
    """
    Return the peak value

    Args:
        signal: Signal to compute the peak value

    Returns:
        Peak value
    """
    return np.max(np.abs(signal))


@nb.njit
def kurtosis(signal: np.array):
    """
    Return the kurtosis of the signal according to Muller et al. 2020

    Args:
        signal: Signal to compute the kurtosis

    Returns:
        Kurtosis value
    """
    n = len(signal)
    var = (signal - np.mean(signal)) ** 2
    mu4 = np.sum(var**2) / n
    mu2 = np.sum(var) / (n - 1)
    return mu4 / mu2**2


@nb.njit
def energy_window(signal: np.array, percentage: float) -> list:
    """
    Return sample window [start, end] which contains a given percentage of the
    signals total energy. See Madsen 2005 for more details.

    Args:
        signal: Signal to compute the sel
        percentage : percentage of total energy contained in output window [0 to 1]

    Returns:
        Sample window [start, end] which contains a given percentage of the signals total energy
    """
    # calculate beginning and ending percentage window (e.g., for x=90%, window = [5%,95%])
    percent_start = 0.50 - percentage / 2
    percent_end = 0.50 + percentage / 2

    # calculate normalized cumulative energy distribution
    Ep_cs = np.cumsum(signal**2)
    Ep_cs_norm = Ep_cs / np.max(Ep_cs)

    # find corresponding indices
    iStartPercent = np.argmin(np.abs(Ep_cs_norm - percent_start))
    iEndPercent = np.argmin(np.abs(Ep_cs_norm - percent_end))

    window = [iStartPercent, iEndPercent]
    return window


@nb.njit
def set_gain(wave: np.array, gain: float) -> np.array:
    """
    Apply the gain in the same magnitude

    Args:
        wave: Signal in upa
        gain: Gain to apply, in uPa

    Returns:
        signal with the applied gain
    """
    return wave * gain


@nb.njit
def set_gain_db(wave: np.array, gain: float) -> np.array:
    """
    Apply the gain in db

    Args:
        wave: Signal in db
        gain: Gain to apply, in db

    Returns:
        signal with the applied gain, in db
    """
    return wave + gain


@nb.njit
def set_gain_upa_db(wave: np.array, gain: float) -> np.array:
    """
    Apply the gain in db to the signal in upa

    Args:
        wave: Signal in upla
        gain: Gain to apply, in db

    Returns:
        signal with the applied gain, in upa
    """
    gain = np.pow(10, gain / 20.0)
    return gain(wave, gain)


@nb.njit
def to_mag(wave: np.array, ref: float) -> np.array:
    """
    Compute the upa from the db signals

    Args:
        wave: Signal in db
        ref  Reference pressure

    Returns:
        signal in upa
    """
    return np.power(10, wave / 20.0 - np.log10(ref))


@nb.njit
def to_db(wave: np.array, ref: float = 1.0, square: bool = False) -> np.array:
    """
    Compute the db from the upa signal

    Args:
        wave: Signal in upa
        ref: Reference pressure
        square: Set to True if the signal has to be squared

    Returns:
        signal in db
    """
    if square:
        db = 10 * np.log10(wave**2 / ref**2)
    else:
        db = 10 * np.log10(wave / ref**2)
    return db


# @nb.jit
def oct_fbands(min_freq: int, max_freq: int, fraction: int) -> tuple:
    """

    Args:
        min_freq: minimum frequency to compute the octave bands
        max_freq: maximum frequency to compute the octave bands
        fraction: fraction of the bands (3 for 1/3 octave bands)

    Returns:
        bands: band limits
        f: band centers
    """
    min_band_n = 0
    max_band_n = 0
    while 1000 * 2 ** (min_band_n / fraction) > min_freq:
        min_band_n = min_band_n - 1
    while 1000 * 2 ** (max_band_n / fraction) < max_freq:
        max_band_n += 1
    bands = np.arange(min_band_n, max_band_n)

    # construct frequency arrays
    f = 1000 * (2 ** (bands / fraction))

    return bands, f


def octdsgn(fc: float, fs: float, fraction: int = 1, n: int = 2) -> tuple:
    """
    Design of an octave band filter with center frequency fc for sampling frequency fs.
    Default value for N is 3. For meaningful results, fc should be in range fs/200 < fc < fs/5.

    Args:
        fc: Center frequency, in Hz
        fs: Sample frequency at least 2.3x the center frequency of the highest octave band, in Hz
        fraction: fraction of the octave band (3 for 1/3-octave bands and 1 for octave bands)
        n: Order specification of the filters, N = 2 gives 4th order, N = 3 gives 6th order
          Higher N can give rise to numerical instability problems, so only 2 or 3 should be used

    Returns:
        sos filter
    """
    if fc > 0.88 * fs / 2:
        raise Exception("Design not possible - check frequencies")
    # design Butterworth 2N-th-order
    f1 = fc * G ** (-1.0 / (2.0 * fraction))
    f2 = fc * G ** (+1.0 / (2.0 * fraction))
    qr = fc / (f2 - f1)
    qd = (np.pi / 2 / n) / (np.sin(np.pi / 2 / n)) * qr
    alpha = (1 + np.sqrt(1 + 4 * qd**2)) / 2 / qd
    w1 = fc / (fs / 2) / alpha
    w2 = fc / (fs / 2) * alpha
    sos = sig.butter(n, [w1, w2], btype="bandpass", output="sos")

    return sos


def octbankdsgn(fs: int, bands: np.array, fraction: int = 1, n: int = 2) -> tuple:
    """
    Construction of an octave band filterbank.

    Args:
        fs: Sample frequency, at least 2.3x the center frequency of the highest 1/3 octave band, in Hz
        bands : row vector with the desired band numbers (0 = band with center frequency of 1 kHz)
          e.g. [-16:11] gives all bands with center frequency between 25 Hz and 12.5 kHz
        fraction: 1 or 3 to get 1-octave or 1/3-octave bands
        n: Order specification of the filters, N = 2 gives 4th order, N = 3 gives 6th order
          Higher N can give rise to numerical instability problems, so only 2 or 3 should be used

    Returns:
        b, a : Matrices with filter coefficients, one row per filter.
        d : Downsampling factors for each filter 1 means no downsampling, 2 means downsampling with factor 2,
            3 means downsampling with factor 4 and so on.
        fsnew : New sample frequencies.
    """
    uneven = fraction % 2 != 0
    fc = f_ref * G ** ((2.0 * bands + 1.0) / (2.0 * fraction)) * np.logical_not(
        uneven
    ) + uneven * f_ref * G ** (bands / fraction)

    # limit for center frequency compared to sample frequency
    fclimit = 1 / 200
    # calculate downsampling factors
    d = np.ones(len(fc))
    for i in np.arange(len(fc)):
        while fc[i] < (fclimit * (fs / 2 ** (d[i] - 1))):
            d[i] += 1
    # calculate new sample frequencies
    fsnew = fs / (2 ** (d - 1))
    # construct filterbank
    filterbank = []
    for i in np.arange(len(fc)):
        # construct filter coefficients
        sos = octdsgn(fc[i], fsnew[i], fraction, n)
        filterbank.append(sos)

    return filterbank, fsnew, d


def get_bands_limits(
    band: list or tuple,
    nfft: int,
    base: int,
    bands_per_division: int,
    hybrid_mode: bool,
    fs: int = None,
) -> tuple:
    """
    Get the limits of the frequency bands between min and max frequency bands specified in band.
    As specified in Martin et al. (2021) https://doi.org/10.1121/10.0003324

    Args:
        band: tuple of [min, max] frequency in Hz
        nfft: number of FFT to use
        base: 2 or 10, for logarithmic base
        bands_per_division: number of bands per division
        hybrid_mode: set to True to get a hybrid mode (bands smaller than 1 Hz are not split)
        fs: if not provided, it will be assumed to be double the highest frequency band limit

    Returns:
        bands_limits: limits of the bands
        bands_c: bands centers
    """
    first_bin_centre = 0
    low_side_multiplier = base ** (-1 / (2 * bands_per_division))
    high_side_multiplier = base ** (1 / (2 * bands_per_division))

    if fs is None:
        fs = band[1] * 2
    fft_bin_width = fs / nfft

    # Start the frequencies list
    bands_limits = []
    bands_c = []

    # count the number of bands:
    band_count = 0
    center_freq = 0
    if hybrid_mode:
        bin_width = 0
        while bin_width < fft_bin_width:
            band_count = band_count + 1
            center_freq = get_center_freq(base, bands_per_division, band_count, band[0])
            bin_width = (
                high_side_multiplier * center_freq - low_side_multiplier * center_freq
            )

        # now keep counting until the difference between the log spaced
        # center frequency and new frequency is greater than .025
        center_freq = get_center_freq(base, bands_per_division, band_count, band[0])
        linear_bin_count = round(center_freq / fft_bin_width - first_bin_centre)
        dc = abs(linear_bin_count * fft_bin_width - center_freq) + 0.1
        while abs(linear_bin_count * fft_bin_width - center_freq) < dc:
            # Compute next one
            dc = abs(linear_bin_count * fft_bin_width - center_freq)
            band_count = band_count + 1
            linear_bin_count = linear_bin_count + 1
            center_freq = get_center_freq(base, bands_per_division, band_count, band[0])

        linear_bin_count = linear_bin_count - 1
        band_count = band_count - 1

        if (fft_bin_width * linear_bin_count) > band[1]:
            linear_bin_count = fs / 2 / fft_bin_width + 1

        for i in np.arange(linear_bin_count):
            # Add the frequencies
            fc = first_bin_centre + i * fft_bin_width
            if fc >= band[0]:
                bands_c.append(fc)
                bands_limits.append(fc - fft_bin_width / 2)

    # count the log space frequencies
    ls_freq = center_freq * high_side_multiplier
    while ls_freq < band[1]:
        fc = get_center_freq(base, bands_per_division, band_count, band[0])
        ls_freq = fc * high_side_multiplier
        if fc >= band[0]:
            bands_c.append(fc)
            bands_limits.append(fc * low_side_multiplier)
        band_count += 1
    # Add the upper limit (bands_limits's length will be +1 compared to bands_c)
    if ls_freq > band[1]:
        ls_freq = band[1]
        if fc > band[1]:
            bands_c[-1] = band[1]
    bands_limits.append(ls_freq)
    return bands_limits, bands_c


def get_center_freq(
    base: int, bands_per_division: int, n: int, first_out_band_centre_freq: float
):
    """
    Get the center frequency

    Args:
        base: 2 or 10, for logarithmic base
        bands_per_division: number of bands per division
        n:
        first_out_band_centre_freq:

    Returns:

    """
    if (bands_per_division == 10) or ((bands_per_division % 2) == 1):
        center_freq = first_out_band_centre_freq * base ** (
            (n - 1) / bands_per_division
        )
    else:
        b = bands_per_division * 0.3
        center_freq = base * G ** ((2 * (n - 1) + 1) / (2 * b))

    return center_freq


def get_hybrid_millidecade_limits(band: list or tuple, nfft: int, fs: int = None):
    """
    Get band limits in hybrid mode for millidecade bands

    Args:
        band: band to get the limits of [min_freq, max_freq]
        nfft: number of fft
        fs: sampling rate

    Returns:

    """
    if fs is None:
        fs = band[1] * 2
    return get_bands_limits(
        band, nfft, base=10, bands_per_division=1000, hybrid_mode=True, fs=fs
    )


def get_decidecade_limits(band: list or tuple, nfft: int, fs: int = None):
    """

    Args:
        band: band to get the limits of [min_freq, max_freq]
        nfft: number of fft
        fs: sampling rate

    Returns:

    """
    if fs is None:
        fs = band[1] * 2
    return get_bands_limits(
        band, nfft, base=10, bands_per_division=10, hybrid_mode=False, fs=fs
    )


def spectra_ds_to_bands(
    psd: np.array,
    bands_limits: list or np.array,
    bands_c: list or np.array,
    fft_bin_width: float,
    freq_coord: str = "frequency",
    db: bool = True,
) -> xarray.DataArray:
    """
    Group the psd according to the limits band_limits given. If a limit is not aligned with the limits in the psd
    frequency axis then that psd frequency bin is divided in proportion to each of the adjacent bands. For more details
    see publication Ocean Sound Analysis Software for Making Ambient Noise Trends Accessible (MANTA)
    (https://doi.org/10.3389/fmars.2021.703650)

    Args:
        psd: Output of pypam spectrum function. It should not be directly the dataset
        bands_limits: Limits of the desired bands
        bands_c: Centre of the bands (used only of the output frequency axis naming)
        fft_bin_width: fft bin width in seconds
        freq_coord : Name of the frequency coordinate
        db: Set to True to return db instead of linear units


    Returns:
        xarray DataArray with frequency_bins instead of frequency as a dimension.

    """
    fft_freq_indices = (
        np.floor((np.array(bands_limits) + (fft_bin_width / 2)) / fft_bin_width)
    ).astype(int)
    original_first_fft_index = int(psd[freq_coord].values[0] / fft_bin_width)
    fft_freq_indices -= original_first_fft_index

    if fft_freq_indices[-1] > (len(psd[freq_coord]) - 1):
        fft_freq_indices[-1] = len(psd[freq_coord]) - 1
    limits_df = pd.DataFrame(
        data={
            "lower_indexes": fft_freq_indices[:-1],
            "upper_indexes": fft_freq_indices[1:],
            "lower_freq": bands_limits[:-1],
            "upper_freq": bands_limits[1:],
        }
    )
    limits_df["lower_factor"] = (
        limits_df["lower_indexes"] * fft_bin_width
        + fft_bin_width / 2
        - limits_df["lower_freq"]
        + psd[freq_coord].values[0]
    )
    limits_df["upper_factor"] = (
        limits_df["upper_freq"]
        - (limits_df["upper_indexes"] * fft_bin_width - fft_bin_width / 2)
        - psd[freq_coord].values[0]
    )

    psd_limits_lower = (
        psd.isel(**{freq_coord: limits_df["lower_indexes"].values})
        * [limits_df["lower_factor"]]
        / fft_bin_width
    )
    psd_limits_upper = (
        psd.isel(**{freq_coord: limits_df["upper_indexes"].values})
        * [limits_df["upper_factor"]]
        / fft_bin_width
    )
    # Bin the bands and add the borders
    psd_without_borders = psd.drop_isel(**{freq_coord: fft_freq_indices})
    new_coord_name = freq_coord + "_bins"
    if len(psd_without_borders[freq_coord]) == 0:
        psd_bands = xarray.zeros_like(psd)
        psd_bands = psd_bands.assign_coords({new_coord_name: (freq_coord, bands_c)})
        psd_bands = psd_bands.swap_dims({freq_coord: new_coord_name}).drop_vars(
            freq_coord
        )
    else:
        psd_bands = psd_without_borders.groupby_bins(
            freq_coord, bins=bands_limits, labels=bands_c, right=False
        ).sum()
        psd_bands = psd_bands.fillna(0)
    psd_bands = psd_bands + psd_limits_lower.values + psd_limits_upper.values
    psd_bands = psd_bands.assign_coords(
        {"lower_frequency": (new_coord_name, limits_df["lower_freq"])}
    )
    psd_bands = psd_bands.assign_coords(
        {"upper_frequency": (new_coord_name, limits_df["upper_freq"])}
    )

    bandwidths = psd_bands.upper_frequency - psd_bands.lower_frequency
    psd_bands = psd_bands / bandwidths

    if db:
        psd_bands = 10 * np.log10(psd_bands)

    psd_bands.attrs.update(psd.attrs)
    return psd_bands


def pcm2float(s: np.array, dtype: str = "float64") -> np.array:
    """
    Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.

    Args:
        s: Input array, must have integral type.
        dtype: Desired (floating point) data type.

    Returns:
        Normalized floating point data.
    """
    s = np.asarray(s)
    if s.dtype.kind not in "iu":
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(s.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (s.astype(dtype) - offset) / abs_max


def merge_ds(ds: xarray.Dataset, new_ds: xarray.Dataset, attrs_to_vars: list):
    """
    Merges de new_ds into the ds, the attributes that are file depending are converted to another coordinate
    depending on datetime.

    Args:
        ds: Already existing dataset
        new_ds : New dataset to merge
        attrs_to_vars: List of all the attributes to convert to coordinates (not dimensions)

    Returns:
        ds : merged dataset
    """
    new_coords = {}
    for attr in attrs_to_vars:
        if attr in new_ds.attrs.keys():
            new_coords[attr] = ("id", [new_ds.attrs[attr]] * new_ds.dims["id"])
    new_ds = new_ds.assign_coords(new_coords)
    if len(ds.dims) == 0:
        ds = ds.merge(new_ds)
    else:
        ds = xarray.concat((ds, new_ds), "id", combine_attrs="drop_conflicts")
    ds.attrs.update(new_ds.attrs)
    return ds


def compute_spd(
    psd_evolution: xarray.Dataset,
    data_var: str = "band_density",
    h: float = 1.0,
    percentiles: list or np.array = None,
    max_val: float = None,
    min_val: float = None,
) -> xarray.Dataset:
    """

    Args:
        psd_evolution: xarray dataset with the psd evolution
        data_var: name of the data var which contains the psd
        h: bin for the histogram
        percentiles: list of the percentiles to compute
        max_val: maximum value for the histogram of the spd
        min_val: minimum value for the histogram of the spd

    Returns:
        Dataset with spd and value_percentiles as data vars
    """
    pxx = psd_evolution[data_var].to_numpy().T
    freq_axis = psd_evolution[data_var].dims[1]
    if percentiles is None:
        percentiles = []
    if min_val is None:
        min_val = pxx.min()
    if max_val is None:
        max_val = pxx.max()
    # Calculate the bins of the psd values and compute spd using numba
    bin_edges = np.arange(start=max(0, min_val), stop=max_val, step=h)
    spd = sxx2spd(sxx=pxx, h=h, bin_edges=bin_edges)

    p = np.nanpercentile(pxx, np.array(percentiles), axis=1)
    percentiles_names = []
    for level_p in percentiles:
        percentiles_names.append("L%s" % str(100 - level_p))

    spd_arr = xarray.DataArray(
        data=spd,
        coords={freq_axis: psd_evolution[freq_axis].values, "spl": bin_edges[:-1]},
        dims=[freq_axis, "spl"],
    )
    p_arr = xarray.DataArray(
        data=p.T,
        coords={
            freq_axis: psd_evolution[freq_axis].values,
            "percentiles": percentiles_names,
        },
        dims=[freq_axis, "percentiles"],
    )
    spd_ds = xarray.Dataset(data_vars={"spd": spd_arr, "value_percentiles": p_arr})
    units_attrs = output_units.get_units_attrs(method_name="spd", log=False)
    spd_ds["spd"].attrs.update(units_attrs)
    spd_ds["spl"].attrs.update(psd_evolution[data_var].attrs)
    return spd_ds


def _swap_dimensions_if_not_dim(
    ds: xarray.Dataset, datetime_coord: str
) -> xarray.Dataset:
    """
    Swap the coordinates between ds and datetime_coord

    Args:
        ds: Dataset to swap dimensions of
        datetime_coord: Name of the datetime dimension


    Returns:
        xarray.Dataset with swapped coordinates
    """
    if datetime_coord not in ds.dims:
        ds = ds.swap_dims({"id": datetime_coord})
    return ds


def _selection_when_joining(
    ds: xarray.Dataset,
    datetime_coord: str,
    data_vars: str or list = None,
    time_resample: str = None,
    freq_band: list or tuple = None,
    freq_coord: str = "frequency",
):
    """

    Args:
        ds: xarray dataset to perform selection on
        datetime_coord: name of the datetime dimension
        time_resample: String indicating the unit to resample to in time
        freq_band: tuple or list with (min_freq, max_freq) to include
        freq_coord: Name of the frequency coordinate

    Returns:
        xarray Dataset only with the selected frequency band and variables

    """
    ds = _swap_dimensions_if_not_dim(ds, datetime_coord)
    if freq_band is not None:
        ds = select_frequency_range(
            ds, freq_band[0], freq_band[1], freq_coord=freq_coord
        )
    if time_resample is not None:
        ds = ds.resample({datetime_coord: time_resample}).median()

    if data_vars is not None:
        ds = ds[data_vars]
    return ds


def join_all_ds_output_deployment(
    deployment_path: str or pathlib.Path,
    data_vars: list = None,
    datetime_coord: str = "datetime",
    join_only_if_contains: str = None,
    load: bool = False,
    parallel: bool = True,
    time_resample: str = None,
    freq_band: list or tuple = None,
    freq_coord: str = "frequency",
    **kwargs,
) -> xarray.Dataset:
    """
    Return a DataArray by joining the data you selected from all the output ds for one deployment

    Args:
    deployment_path: Where all the netCDF files of a deployment are stored
    data_vars: Name of the data that you want to keep for joining ds. If None, all the data vars will be joined
    datetime_coord: Name of the time coordinate to join the datasets along
    load: Set to True to load the entire dataset in memory. Otherwise it will return a dask xarray
    join_only_if_contains: String which needs to be contained in the path name to be joined. If set to None (default),
        all the files are joined
    time_resample: String indicating the unit to resample to in time
    freq_band: tuple or list with (min_freq, max_freq) to include
    freq_coord: Name of the frequency coordinate
    parallel: Set to True to speed up loading
    **kwargs: any args which can be passed to open_mfdataset

    Returns:
    ds_tot: Data joined of one deployment, if load=False, returns a xarray dask dataset. Otherwise it loads into memory.
        To load the full dataset into memory, use afterwards ds_tot.load()
    """
    if dask is None:
        raise ModuleNotFoundError("This function requires dask to be installed.")

    deployment_path = pathlib.Path(deployment_path)
    list_path = list(deployment_path.glob("*.nc"))
    # Remove files not matching the pattern
    if join_only_if_contains is not None:
        clean_list_path = []
        for path in list_path:
            if str(join_only_if_contains) in str(path):
                clean_list_path.append(path)
        list_path = clean_list_path

    partial_func = partial(
        _selection_when_joining,
        data_vars=data_vars,
        datetime_coord=datetime_coord,
        freq_band=freq_band,
        time_resample=time_resample,
        freq_coord=freq_coord,
    )
    ds_tot = xarray.open_mfdataset(
        list_path,
        parallel=parallel,
        preprocess=partial_func,
        data_vars=data_vars,
        **kwargs,
    )

    if load:
        with ProgressBar():
            ds_tot = ds_tot.compute()

    return ds_tot


def select_datetime_range(
    da_sxx: xarray.DataArray,
    start_datetime: datetime.datetime,
    end_datetime: datetime.datetime,
) -> tuple:
    """
    Args:
        da_sxx: Data in which we want to select only a certain range of datetime
        start_datetime: Lower limit of datetime that you want to plot
        end_datetime: Upper limit of datetime that you want to plot

    Returns:
        da_sxx: Data with the new limits
        old_start_datetime: Old lower datetime limit of the data
        old_end_datetime: Old upper datetime limit of the data
    """

    old_start_datetime = np.asarray(da_sxx.datetime)[0]
    old_end_datetime = np.asarray(da_sxx.datetime)[-1]

    da_sxx = da_sxx.where(da_sxx.datetime >= start_datetime, drop=True)
    da_sxx = da_sxx.where(da_sxx.datetime <= end_datetime, drop=True)

    return da_sxx, old_start_datetime, old_end_datetime


def select_frequency_range(
    ds: xarray.Dataset or xarray.DataArray,
    min_freq: int,
    max_freq: int,
    freq_coord: str = "frequency",
) -> xarray.Dataset or xarray.DataArray:
    """
    Crop the dataset to the specified band between min freq and max freq.

    Args:
        ds: Data to crop
        min_freq: Minimum frequency in Hz
        max_freq: Maximum frequency in Hz
        freq_coord: Name of the frequency coordinate

    Returns:
        The dataset cropped
    """
    ds_cropped = ds.sel(
        frequency=ds[freq_coord][
            (ds[freq_coord] > min_freq) & (ds[freq_coord] < max_freq)
        ]
    )
    return ds_cropped


def join_all_ds_output_station(
    directory_path: str or pathlib.Path, station: str, data_var_name: str
) -> xarray.DataArray:
    """
    Return a DataArray by joining the data you selected from all the output ds for one station

    Args:
        directory_path:  Where all the deployments folders are
        station: Name of the station to compute the spectrogram
        data_var_name: Name of the data that you want to keep for joining ds

    Returns:
        da_tot: Data joined of one deployment
    """

    list_path_deployment = list(directory_path.iterdir())
    list_path_deployment_station = []

    for deployment_path in list_path_deployment:
        if deployment_path.parts[-1].split("_")[0] == station:
            list_path_deployment_station.append(deployment_path)

    for deployment_station_path in tqdm(list_path_deployment_station):
        da_deployment = join_all_ds_output_deployment(
            deployment_station_path, data_vars=data_var_name
        )

        if deployment_station_path == list_path_deployment_station[0]:
            da_tot = da_deployment.copy()

        else:
            da_tot = xarray.concat([da_tot, da_deployment], "datetime")

    datetime = np.asarray(da_tot.datetime)
    errors = []
    for i in range(1, len(datetime)):
        if datetime[i] < datetime[i - 1]:
            errors.append(datetime[i])
    da_tot = da_tot.drop_sel(datetime=errors)

    return da_tot


def reindexing_datetime(
    da: xarray.DataArray,
    first_datetime: datetime.datetime,
    last_datetime: datetime.datetime,
    freq: str = "10T",
    tolerance: str = "1D",
    fill_value=np.nan,
) -> xarray.DataArray:
    """
    Reindex the datetime of your data and fill missing values

    Args:
        da: Data you want to reindex
        first_datetime: Lower limit of the new datetime index
        last_datetime: Upper limit of the new datetime index
        freq: Frequency of values in the new datetime index
        tolerance: Maximum distance between original and new datetimes for inexact matches
        fill_value: Value to use for newly missing values

    Returns:
        da_reindex: Data after reindexing
    """
    index = pd.date_range(start=first_datetime, end=last_datetime, freq=freq).round("T")
    da_reindex = da.reindex(
        datetime=index, tolerance=tolerance, method="nearest", fill_value=fill_value
    )
    return da_reindex


def freq_band_aggregation(
    ds: xarray.Dataset,
    data_var: str,
    aggregation_freq_band: list or tuple = None,
    freq_coord: str = None,
) -> xarray.Dataset:
    """
    It will compute the median of all the values included in the frequency band specified in 'aggregation_freq_band'.

    Args:
        ds: Dataset to process, has to have datetime as coords, not id
        data_var: Name of the data variable to select datetime
        freq_coord: Name of the frequency coordinate
        aggregation_freq_band : If a float is given, this function compute aggregation for the frequency which is selected
            If a tuple is given, this function will compute aggregation for the average of all frequencies which are
            selected
            If None is given, this function will compute aggregation for the data_var given, assuming that there is no
            frequency dependence

    Returns:
        ds_new : Same Dataset but the frequency axi is replaced by the median value
    """
    ds_copy = ds.copy()
    if freq_coord is None:
        freq_coord = ds_copy[data_var].dims[1]
    if aggregation_freq_band is not None:
        if isinstance(aggregation_freq_band, tuple):
            ds_copy = ds_copy.where(
                (ds_copy[freq_coord] >= aggregation_freq_band[0])
                & (ds_copy[freq_coord] <= aggregation_freq_band[1]),
                drop=True,
            )
        if isinstance(aggregation_freq_band, int):
            aggregation_freq_band = float(aggregation_freq_band)
        if isinstance(aggregation_freq_band, float):
            aggregation_freq_band = ds_copy[freq_coord].values[
                min(
                    range(len(ds_copy[freq_coord].values)),
                    key=lambda i: abs(
                        ds_copy[freq_coord].values[i] - aggregation_freq_band
                    ),
                )
            ]
            ds_copy = ds_copy.where(
                ds_copy[freq_coord].isin(aggregation_freq_band), drop=True
            )

    ds_copy = ds_copy.median(dim=freq_coord)
    ds_copy[data_var].attrs = ds[data_var].attrs

    return ds_copy


def update_freq_cal(
    hydrophone: object, ds: xarray.Dataset, data_var: str, **kwargs
) -> xarray.Dataset:
    """
    Update the dataset with the difference between flat calibration and frequency-dependent calibration

    Args:
        hydrophone: hydrophone object with the frequency-dependent calibration information
        ds: dataset to apply the update to
        data_var: data variable to apply the update to
        **kwargs:

    Returns:
        updated xarray dataset
    """
    index_coord = ds[data_var].dims[0]
    freq_coord = ds[data_var].dims[1]
    frequencies = ds[freq_coord].values

    if hydrophone.freq_cal is None:
        hydrophone.get_freq_cal(**kwargs)
    df = hydrophone.freq_cal_inc(frequencies=frequencies)
    ds_copy = ds.copy(deep=True)

    for i in range(ds[index_coord].size):
        ds_copy[data_var][i] = ds[data_var][i] + df["inc_value"].values

    return ds_copy


def parse_file_name(sfile):
    if type(sfile) == str:
        file_name = os.path.split(sfile)[-1]
    elif issubclass(sfile.__class__, pathlib.Path):
        file_name = sfile.name
    elif issubclass(sfile.__class__, zipfile.ZipExtFile):
        file_name = sfile.name
    else:
        raise Exception("The filename has to be either a Path object or a string")

    return file_name


def hmb_to_decidecade(
    ds: xarray.Dataset, data_var: str, freq_coord: str, fs: int = None
) -> xarray.Dataset:
    """
    Aggregate the hybrid millidecade bands to decidecade bands

    Args:
        ds: xarray dataset containing the hybrid millidecade bands
        data_var: name of the variable where the hybrid millidecade bands are stored
        freq_coord: name of the frequency coordinate
        fs: sampling frequency. If set to none, the sampling frequency will be considered twice the maximum frequency value

    Returns:
        xarray dataset with decidecade bands
    """
    # Convert back to upa for the sum operations
    ds_data_var = np.power(10, ds[data_var].copy() / 10.0 - np.log10(1))
    fft_bin_width = 1.0
    changing_frequency = 434
    if fs is None:
        if "fs" not in ds.attrs.keys():
            max_freq = ds[freq_coord].values.max()
        else:
            max_freq = ds.attrs["fs"] / 2
    else:
        max_freq = fs / 2

    ds[freq_coord] = ds[freq_coord].values.astype(float).round(decimals=2)
    # Add the frequency limits if they are not in the ds cordinates
    if "upper_frequency" not in ds_data_var.coords:
        hmb_limits, hmb_c = get_hybrid_millidecade_limits(
            band=[0, max_freq], nfft=max_freq * 2, fs=max_freq * 2
        )
        hmb_limits = np.around(hmb_limits, decimals=2).tolist()
        hmb_c = np.around(hmb_c, decimals=2).tolist()
        rounded_freq = ds_data_var[freq_coord].values.astype(float).round(decimals=2)
        hmb_limits = hmb_limits[
            hmb_c.index(rounded_freq.min()) : hmb_c.index(rounded_freq.max()) + 2
        ]
        ds_data_var = ds_data_var.assign_coords(
            upper_frequency=(freq_coord, hmb_limits[1:]),
            lower_frequency=(freq_coord, hmb_limits[:-1]),
        )

    bands_limits, bands_c = get_decidecade_limits(
        band=[10, max_freq], nfft=max_freq * 2, fs=max_freq * 2
    )
    bands_limits = np.array(bands_limits).round(decimals=2)
    bands_c = np.array(bands_c).round(decimals=2)
    maximum_band = np.where(
        np.array(bands_limits) > ds_data_var.upper_frequency.values.max()
    )[0][0]
    bands_limits = bands_limits[: maximum_band + 1]
    bands_c = bands_c[:maximum_band]

    changing_band = np.where(np.array(bands_limits) < changing_frequency)[0][-1]
    # We need to split the dataset in two, the part which is below the changing frequency and the part which is above
    low_psd = ds_data_var.where(
        ds_data_var.upper_frequency <= bands_limits[changing_band], drop=True
    )
    low_decidecade = spectra_ds_to_bands(
        low_psd,
        bands_limits[: changing_band + 1],
        bands_c[:changing_band],
        fft_bin_width,
        freq_coord=freq_coord,
        db=False,
    )

    # Compute the decidecades on the non-linear part
    high_psd = ds_data_var.where(
        ds_data_var.upper_frequency >= bands_limits[changing_band], drop=True
    )
    high_pwd = high_psd * (high_psd.upper_frequency - high_psd.lower_frequency)
    high_decidecade = high_pwd.groupby_bins(
        freq_coord,
        bins=bands_limits[changing_band:],
        labels=bands_c[changing_band:],
        right=True,
    ).sum()
    high_decidecade = high_decidecade.assign_coords(
        {"lower_frequency": (freq_coord + "_bins", bands_limits[changing_band:-1])}
    )
    high_decidecade = high_decidecade.assign_coords(
        {"upper_frequency": (freq_coord + "_bins", bands_limits[changing_band + 1 :])}
    )
    bandwidths = high_decidecade.upper_frequency - high_decidecade.lower_frequency
    high_decidecade = high_decidecade / bandwidths

    # Merge the low and the high decidecade psd
    decidecade_psd = xarray.merge(
        [{data_var: low_decidecade}, {data_var: high_decidecade}]
    )

    # change the name of the frequency coord
    decidecade_psd = decidecade_psd.rename({freq_coord + "_bins": freq_coord})

    # Convert back to db
    decidecade_psd = 10 * np.log10(decidecade_psd)

    return decidecade_psd


def get_file_datetime(file_name, hydrophone):
    try:
        date = hydrophone.get_name_datetime(file_name)
    except ValueError:
        date = datetime.datetime.now()
        print(
            "Filename %s does not match the %s file structure. Setting time to now..."
            % (file_name, hydrophone.name)
        )
    return date
