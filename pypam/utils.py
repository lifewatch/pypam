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
        spd[i, :] = np.histogram(sxx[i, :], bin_edges)[0] / ((sxx.shape[0]) * h)
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


def get_bands_limits(band, nfft, base, bands_per_division, hybrid_mode, first_bin_centre=0):
    """

    Parameters
    ----------
    band
    nfft
    base
    bands_per_division
    hybrid_mode
    first_bin_centre

    Returns
    -------

    """
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
        linear_bin_count = round(center_freq / fft_bin_width)
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


def psd_ds_to_bands(psd, bands_limits, bands_c, fft_bin_width, method='spectrum', db=True):
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
    method: string
        Should be 'density' or 'spectrum'
    db: bool
        Set to True to return db instead of linear units


    Returns
    -------
    xarray DataArray with frequency_bins instead of frequency as a dimension.

    """
    fft_freq_indices = np.floor(np.array(bands_limits) / fft_bin_width + fft_bin_width / 2).astype(int)
    if fft_freq_indices[-1] > len(psd.frequency):
        fft_freq_indices[-1] = len(psd.frequency) - 1
    limits_df = pd.DataFrame(data={'lower_indexes': fft_freq_indices[:-1], 'upper_indexes': fft_freq_indices[1:],
                                   'lower_freq': bands_limits[:-1], 'upper_freq': bands_limits[1:]})
    limits_df['lower_factor'] = limits_df['lower_indexes'] * fft_bin_width + fft_bin_width/2 - limits_df['lower_freq']
    limits_df['upper_factor'] = limits_df['upper_freq'] - (limits_df['upper_indexes'] * fft_bin_width - fft_bin_width/2)
    psd_limits_lower = psd.isel(frequency=limits_df['lower_indexes']) * [limits_df['lower_factor']]
    psd_limits_upper = psd.isel(frequency=limits_df['upper_indexes']) * [limits_df['upper_factor']]
    # Bin the bands and add the borders
    psd_without_borders = psd.drop_isel(frequency=fft_freq_indices)
    psd_bands = psd_without_borders.groupby_bins('frequency', bins=bands_limits, labels=bands_c, right=False).sum()
    psd_bands = psd_bands.fillna(0)
    psd_bands = psd_bands + psd_limits_lower.values + psd_limits_upper.values
    psd_bands = psd_bands.assign_coords({'lower_frequency': ('frequency_bins', limits_df['lower_freq'])})
    psd_bands = psd_bands.assign_coords({'upper_frequency': ('frequency_bins', limits_df['upper_freq'])})

    if method == 'density':
        bandwidths = psd_bands.frequency_bins.diff('frequency_bins')
        psd_bands = psd_bands / bandwidths

    if db:
        psd_bands = 10 * np.log10(psd_bands)

    return psd_bands

# Original MANTA code
# def psd_to_bands(psd, bands_limits, bands_c):
#     nRows = len(psd)
#     bandsOut = np.zeros(nRows, lastBand-firstBand+1)
#     step = fftBinSize / 2
#     nFFTBins = size(psd, 2)
#     startOffset = floor(bin1CenterFrequency / fftBinSize)
#
#     for row in np.arange(nRows):
#         for j in np.arange(firstBand, lastBand):
#             minFFTBin = np.floor((freqTable[j,1] / fftBinSize) + step) + 1 - startOffset
#             maxFFTBin = np.floor((freqTable[j,3] / fftBinSize) + step) + 1 - startOffset
#             if maxFFTBin > nFFTBins:
#                 maxFFTBin = nFFTBins
#
#             if minFFTBin < 1:
#                 minFFTBin = 1
#
#             if minFFTBin == maxFFTBin:
#                 bandsOut[row, j] = psd[row, minFFTBin] *((freqTable[j, 3]- freqTable[j, 1])/ fftBinSize)
#             else:
#                 # Add the first partial FFT bin - take the top of the bin and
#                 # subtract the lower freq to get the amount we will use:
#                 # the top freq of a bin is bin# * step size - binSize/2 since bin
#                 #
#                 lowerFactor = ((minFFTBin - step) * fftBinSize - freqTable[j, 1])
#                 bandsOut[row, j] = psd(row, minFFTBin) * lowerFactor
#
#                 # Add the last partial FFT bin.
#                 upperFactor = freqTable[j, 3] - (maxFFTBin - 1.5*fftBinSize) * fftBinSize
#                 bandsOut[row, j] = bandsOut[row, j] + psd(row, maxFFTBin)* upperFactor
#
#                 # Add any FFT bins in between min and max.
#                 if (maxFFTBin - minFFTBin) > 1:
#                     bandsOut[row, j] = bandsOut[row, j] + sum(psd[row, minFFTBin+1:maxFFTBin-1])
#     return bandsOut
# In python:
# psd_bands = xarray.DataArray(np.zeros((psd.dims['id'], len(bands_c))), coords={'frequency': bands_c, 'id': psd.id},
#                              dims=['id', 'frequency'])
# step = fft_bin_width / 2
# n_fft_bins = psd.dims['frequency']
# start_offset = 0
# for id_n, id_name in enumerate(psd_bands.id):
#     for frequency_n, frequency_val in enumerate(psd_bands.frequency):
#         min_fft_bin = int(np.floor((bands_limits[frequency_n] / fft_bin_width) + step) - start_offset)
#         max_fft_bin = int(np.floor((bands_limits[frequency_n+1] / fft_bin_width) + step) - start_offset)
#         if max_fft_bin > n_fft_bins:
#             max_fft_bin = n_fft_bins
#
#         if min_fft_bin < 0:
#             min_fft_bin = 0
#
#         if min_fft_bin == max_fft_bin:
#             psd_bands.iloc[id_n, frequency_n] = psd['band_spectrum'].isel(id=id_n, frequency=min_fft_bin) * \
#                                                     ((bands_limits[frequency_n+1] -
#                                                       bands_limits[frequency_n]) / fft_bin_width)
#         else:
#             # Add the first partial FFT bin - take the top of the bin and
#             # subtract the lower freq to get the amount we will use:
#             # the top freq of a bin is bin# * step size - binSize/2 since bin
#
#             lower_factor = (min_fft_bin - step + 1) * fft_bin_width - bands_limits[frequency_n]
#             psd_lower = psd['band_spectrum'].isel(id=id_n, frequency=min_fft_bin) * lower_factor
#
#             # Add the last partial FFT bin.
#             upper_factor = bands_limits[frequency_n+1] - (max_fft_bin - fft_bin_width/2) * fft_bin_width
#             psd_borders = psd_lower + psd['band_spectrum'].isel(id=id_n, frequency=max_fft_bin) * upper_factor
#
#             # Add any FFT bins in between min and max.
#             if (max_fft_bin - min_fft_bin) > 1:
#                 psd_bands.loc[id_name, frequency_val] = psd_borders + \
#                                                     sum(psd['band_spectrum'].isel(id=id_n,
#                                                                                   frequency=np.arange(min_fft_bin,
#                                                                                                       max_fft_bin-1)))
#             else:
#                 psd_band = psd_borders
#
#             psd_bands.loc[id_name, frequency_val] = psd_band
#             print(id_n, frequency_n, psd_band.values)


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
