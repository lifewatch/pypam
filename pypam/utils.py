__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

import numba as nb
import numpy as np
import scipy.signal as sig
import xarray
import pandas as pd
from tqdm import tqdm

G = 10.0 ** (3.0 / 10.0)
f_ref = 1000


@nb.jit
def sxx2spd(sxx: np.ndarray, h: float, percentiles: np.ndarray, bin_edges: np.ndarray):
    """
    Return spd from the spectrogram

    Parameters
    ----------
    sxx : numpy matrix
        Spectrogram
    h : float
        Histogram bin width
    percentiles : list or None
        List of floats with all the percentiles to be computed
    bin_edges : numpy array
        Limits of the histogram bins
    """
    spd = np.zeros((sxx.shape[0], bin_edges.size - 1), dtype=np.float64)
    p = np.zeros((sxx.shape[0], percentiles.size), dtype=np.float64)
    for i in nb.prange(sxx.shape[0]):
        spd[i, :] = np.histogram(sxx[i, :], bin_edges)[0] / ((sxx.shape[1]) * h)
        cumsum = np.cumsum(spd[i, :])
        for j in nb.prange(percentiles.size):
            p[i, j] = bin_edges[np.argmax(cumsum > percentiles[j] * cumsum[-1])]

    return spd, p


@nb.njit
def rms(signal):
    """
    Return the rms value of the signal

    Parameters
    ----------
    signal : numpy array
        Signal to compute the rms value
    """
    return np.sqrt(np.mean(signal ** 2))


@nb.njit
def dynamic_range(signal):
    """
    Return the dynamic range of the signal

    Parameters
    ----------
    signal : numpy array
        Signal to compute the dynamic range
    """
    return np.max(signal) - np.min(signal)


@nb.njit
def sel(signal, fs):
    """
    Return the Sound Exposure Level

    Parameters
    ----------
    signal : numpy array
        Signal to compute the dynamic range
    fs : int
        Sampling frequency
    """
    return np.sum(signal ** 2) / fs


@nb.jit
def peak(signal):
    """
    Return the peak value

    Parameters
    ----------
    signal : numpy array
        Signal to compute the dynamic range
    """
    return np.max(np.abs(signal))


@nb.njit
def set_gain(wave, gain):
    """
    Apply the gain in the same magnitude

    Parameters
    ----------
    wave : numpy array
        Signal in upa
    gain :
        Gain to apply, in uPa
    """
    return wave * gain


@nb.njit
def set_gain_db(wave, gain):
    """
    Apply the gain in db

    Parameters
    ----------
    wave : numpy array
        Signal in db
    gain :
        Gain to apply, in db
    """
    return wave + gain


@nb.njit
def set_gain_upa_db(wave, gain):
    """
    Apply the gain in db to the signal in upa
    """
    gain = np.pow(10, gain / 20.0)
    return gain(wave, gain)


@nb.njit
def to_mag(wave, ref):
    """
    Compute the upa from the db signals

    Parameters
    ----------
    wave : numpy array
        Signal in db
    ref : float
        Reference pressure
    """
    return np.power(10, wave / 20.0 - np.log10(ref))


@nb.njit
def to_db(wave, ref=1.0, square=False):
    """
    Compute the db from the upa signal

    Parameters
    ----------
    wave : numpy array
        Signal in upa
    ref : float
        Reference pressure
    square : boolean
        Set to True if the signal has to be squared
    """
    if square:
        db = 10 * np.log10(wave ** 2 / ref ** 2)
    else:
        db = 10 * np.log10(wave / ref ** 2)
    return db


# @nb.jit
def oct_fbands(min_freq, max_freq, fraction):
    min_band_n = 0
    max_band_n = 0
    while 1000 * 2 ** (min_band_n / fraction) > min_freq:
        min_band_n = min_band_n - 1
    while 1000 * 2 ** (max_band_n / fraction) < max_freq:
        max_band_n += 1
    bands = np.arange(min_band_n, max_band_n)

    # construct requency arrays
    f = 1000 * (2 ** (bands / fraction))

    return bands, f


def octdsgn(fc, fs, fraction=1, n=2):
    """
    Design of an octave band filter with center frequency fc for sampling frequency fs.
    Default value for N is 3. For meaningful results, fc should be in range fs/200 < fc < fs/5.

    Parameters
    ----------
    fc : float
      Center frequency, in Hz
    fs : float
      Sample frequency at least 2.3x the center frequency of the highest octave band, in Hz
    fraction : int
        fraction of the octave band (3 for 1/3-octave bands and 1 for octave bands)
    n : int
      Order specification of the filters, N = 2 gives 4th order, N = 3 gives 6th order
      Higher N can give rise to numerical instability problems, so only 2 or 3 should be used
    """
    if fc > 0.88 * fs / 2:
        raise Exception('Design not possible - check frequencies')
    # design Butterworth 2N-th-order
    f1 = fc * G ** (-1.0 / (2.0 * fraction))
    f2 = fc * G ** (+1.0 / (2.0 * fraction))
    qr = fc / (f2 - f1)
    qd = (np.pi / 2 / n) / (np.sin(np.pi / 2 / n)) * qr
    alpha = (1 + np.sqrt(1 + 4 * qd ** 2)) / 2 / qd
    w1 = fc / (fs / 2) / alpha
    w2 = fc / (fs / 2) * alpha
    sos = sig.butter(n, [w1, w2], btype='bandpass', output='sos')

    return sos


def octbankdsgn(fs, bands, fraction=1, n=2):
    """
    Construction of an octave band filterbank.

    Parameters
    ----------
    fs : float
      Sample frequency, at least 2.3x the center frequency of the highest 1/3 octave band, in Hz
    bands : numpy array
      row vector with the desired band numbers (0 = band with center frequency of 1 kHz)
      e.g. [-16:11] gives all bands with center frequency between 25 Hz and 12.5 kHz
    fraction : int
        1 or 3 to get 1-octave or 1/3-octave bands
    n : int
      Order specification of the filters, N = 2 gives 4th order, N = 3 gives 6th order
      Higher N can give rise to numerical instability problems, so only 2 or 3 should be used

    Returns
    -------
    b, a : numpy matrix
      Matrices with filter coefficients, one row per filter.
    d : numpy array
      Downsampling factors for each filter 1 means no downsampling, 2 means
      downsampling with factor 2, 3 means downsampling with factor 4 and so on.
    fsnew : numpy array
      New sample frequencies.
    """
    uneven = (fraction % 2 != 0)
    fc = f_ref * G ** ((2.0 * bands + 1.0) / (2.0 * fraction)) * np.logical_not(uneven) + uneven * f_ref * G ** (
            bands / fraction)

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


def get_bands_limits(band, nfft, base, bands_per_division, hybrid_mode):
    """

    Parameters
    ----------
    band
    nfft
    base
    bands_per_division
    hybrid_mode

    Returns
    -------

    """
    first_bin_centre = 0
    low_side_multiplier = base ** (-1 / (2 * bands_per_division))
    high_side_multiplier = base ** (1 / (2 * bands_per_division))

    fft_bin_width = band[1] * 2 / nfft

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
            bin_width = high_side_multiplier * center_freq - low_side_multiplier * center_freq

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
            linear_bin_count = band[1] / fft_bin_width + 1

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


def get_center_freq(base, bands_per_division, n, first_out_band_centre_freq):
    if (bands_per_division == 10) or ((bands_per_division % 2) == 1):
        center_freq = first_out_band_centre_freq * base ** ((n - 1) / bands_per_division)
    else:
        b = bands_per_division * 0.3
        center_freq = base * G ** ((2 * (n - 1) + 1) / (2 * b))

    return center_freq


def get_hybrid_millidecade_limits(band, nfft):
    """

    Parameters
    ----------
    band
    nfft

    Returns
    -------

    """
    return get_bands_limits(band, nfft, base=10, bands_per_division=1000, hybrid_mode=True)


def get_decidecade_limits(band, nfft):
    """

    Parameters
    ----------
    band
    nfft

    Returns
    -------

    """
    return get_bands_limits(band, nfft, base=10, bands_per_division=10, hybrid_mode=False)


def spectra_ds_to_bands(psd, bands_limits, bands_c, fft_bin_width, db=True):
    """
    Group the psd according to the limits band_limits given. If a limit is not aligned with the limits in the psd
    frequency axis then that psd frequency bin is divided in proportion to each of the adjacent bands. For more details
    see publication Ocean Sound Analysis Software for Making Ambient Noise Trends Accessible (MANTA)
    (https://doi.org/10.3389/fmars.2021.703650)

    Parameters
    ----------
    psd: xarray DataArray
        Output of pypam spectrum function. It should not be directly the dataset
    bands_limits: list or array
        Limits of the desired bands
    bands_c: list or array
        Centre of the bands (used only of the output frequency axis naming)
    fft_bin_width: float
        fft bin width in seconds
    db: bool
        Set to True to return db instead of linear units


    Returns
    -------
    xarray DataArray with frequency_bins instead of frequency as a dimension.

    """
    fft_freq_indices = (np.floor((np.array(bands_limits) + (fft_bin_width / 2)) / fft_bin_width)).astype(int)
    original_first_fft_index = int(psd.frequency.values[0] / fft_bin_width)
    fft_freq_indices -= original_first_fft_index

    if fft_freq_indices[-1] > (len(psd.frequency) - 1):
        fft_freq_indices[-1] = len(psd.frequency) - 1
    limits_df = pd.DataFrame(data={'lower_indexes': fft_freq_indices[:-1], 'upper_indexes': fft_freq_indices[1:],
                                   'lower_freq': bands_limits[:-1], 'upper_freq': bands_limits[1:]})
    limits_df['lower_factor'] = limits_df['lower_indexes'] * fft_bin_width + fft_bin_width / 2 - \
                                limits_df['lower_freq'] + psd.frequency.values[0]
    limits_df['upper_factor'] = limits_df['upper_freq'] - \
                                (limits_df['upper_indexes'] * fft_bin_width - fft_bin_width / 2) - psd.frequency.values[
                                    0]

    psd_limits_lower = psd.isel(frequency=limits_df['lower_indexes'].values) * [
        limits_df['lower_factor']] / fft_bin_width
    psd_limits_upper = psd.isel(frequency=limits_df['upper_indexes'].values) * [
        limits_df['upper_factor']] / fft_bin_width
    # Bin the bands and add the borders
    psd_without_borders = psd.drop_isel(frequency=fft_freq_indices)
    if len(psd_without_borders.frequency) == 0:
        psd_bands = xarray.zeros_like(psd)
        psd_bands = psd_bands.assign_coords({'frequency_bins': ('frequency', bands_c)})
        psd_bands = psd_bands.swap_dims({'frequency': 'frequency_bins'}).drop('frequency')
    else:
        psd_bands = psd_without_borders.groupby_bins('frequency', bins=bands_limits, labels=bands_c, right=False).sum()
        psd_bands = psd_bands.fillna(0)
    psd_bands = psd_bands + psd_limits_lower.values + psd_limits_upper.values
    psd_bands = psd_bands.assign_coords({'lower_frequency': ('frequency_bins', limits_df['lower_freq'])})
    psd_bands = psd_bands.assign_coords({'upper_frequency': ('frequency_bins', limits_df['upper_freq'])})

    bandwidths = psd_bands.upper_frequency - psd_bands.lower_frequency
    psd_bands = psd_bands / bandwidths

    if db:
        psd_bands = 10 * np.log10(psd_bands)

    return psd_bands


def pcm2float(s, dtype='float64'):
    """
    Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    s : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    """
    s = np.asarray(s)
    if s.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(s.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (s.astype(dtype) - offset) / abs_max


def merge_ds(ds, new_ds, attrs_to_vars):
    """
    Merges de new_ds into the ds, the attributes that are file depending are converted to another coordinate
    depending on datetime.

    Parameters
    ----------
    ds: xarray Dataset
        Already existing dataset
    new_ds : xarray Dataset
        New dataset to merge
    attrs_to_vars: list or None
        List of all the attributes to convert to coordinates (not dimensions)

    Returns
    -------
    ds : merged dataset
    """
    new_coords = {}
    for attr in attrs_to_vars:
        if attr in new_ds.attrs.keys():
            new_coords[attr] = ('id', [new_ds.attrs[attr]] * new_ds.dims['id'])
    if len(ds.dims) != 0:
        start_value = ds['id'][-1].values + 1
    else:
        start_value = 0
    new_ids = np.arange(start_value, start_value + new_ds.dims['id'])
    new_ds = new_ds.reset_index('id')
    new_coords['id'] = new_ids
    new_ds = new_ds.assign_coords(new_coords)
    if len(ds.dims) == 0:
        ds = ds.merge(new_ds)
    else:
        ds = xarray.concat((ds, new_ds), 'id', combine_attrs="drop_conflicts")
    ds.attrs.update(new_ds.attrs)
    return ds


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
    spd, p = sxx2spd(sxx=pxx, h=h, percentiles=np.array(percentiles) / 100.0, bin_edges=bin_edges)
    spd_arr = xarray.DataArray(data=spd,
                               coords={'frequency': psd_evolution.frequency, 'spl': bin_edges[:-1]},
                               dims=['frequency', 'spl'])
    p_arr = xarray.DataArray(data=p,
                             coords={'frequency': psd_evolution.frequency, 'percentiles': percentiles},
                             dims=['frequency', 'percentiles'])
    spd_ds = xarray.Dataset(data_vars={'spd': spd_arr, 'value_percentiles': p_arr})

    return spd_ds


def join_all_ds_output_deployment(deployment_path, data_var_name):
    """
    Return the long-term spectrogram for one deployment in hybrid millidecade bands by joining spectrograms stored
    in the files
    Parameters
    ----------
    deployment_path : str or Path
        Where all the netCDF files of a deployment are stored
    data_var_name : str
        Name of the data that you want to keep for joining ds

    Returns
    -------
    da_tot : DataArray
        The spectrogram of one deployment
    """

    list_path = list(deployment_path.glob('*.nc'))

    for path in tqdm(list_path):
        ds = xarray.open_dataset(path)
        ds = ds.swap_dims({'id': 'datetime'})
        da = ds[data_var_name]

        coords_to_drop = list(da.coords)
        for dims in list(da.dims):
            coords_to_drop.remove(dims)
        da = da.drop_vars(coords_to_drop)

        if path == list_path[0]:
            da_tot = da.copy()
        else:
            da_tot = xarray.concat([da_tot, da], 'datetime')

    return da_tot


def select_datetime_range(da_sxx, start_datetime, end_datetime):
    """
    Parameters
    ----------
    da_sxx : xarray DataArray
        Spectrogram in which we want to select only a certain range of datetime
    start_datetime : datetime64
        Lower limit of datetime that you want to plot
    end_datetime : datetime64
        Upper limit of datetime that you want to plot

    Returns
    -------
    da_sxx : xarray DataArray
        Spectrogram with the new limits
    old_start_datetime : datetime64
        Old lower datetime limit of the spectrogram
    old_end_datetime : datetime64
        Old upper datetime limit of the spectrogram
    """

    old_start_datetime = np.asarray(da_sxx.datetime)[0]
    old_end_datetime = np.asarray(da_sxx.datetime)[-1]

    da_sxx = da_sxx.where(da_sxx.datetime >= start_datetime, drop=True)
    da_sxx = da_sxx.where(da_sxx.datetime <= end_datetime, drop=True)

    return da_sxx, old_start_datetime, old_end_datetime
