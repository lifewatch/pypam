"""
Module : utils.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Institute voor de Zee)
"""
__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

import numba as nb
import numpy as np
import scipy.signal as sig

BANDS = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250',
         '315', '400', '500', '630', '800', '1000', '1250', '1600', '2000', '2500',
         '3150', '4000', '5000', '6300', '8000', '10000', '12500']

NOMINAL_OCTAVE_CENTER_FREQUENCIES = np.array([31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
"""Nominal octave center frequencies.
"""

NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES = np.array([
    25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0,
    1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0
])
"""Nominal third-octave center frequencies in the audio range.
"""

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
    percentiles : list
        List of floats with all the percentiles to be computed
    bin_edges : numpy array
        Limits of the histogram bins
    """
    spd = np.zeros((sxx.shape[0], bin_edges.size - 1), dtype=np.float64)
    p = np.zeros((sxx.shape[0], percentiles.size), dtype=np.float64)
    for i in nb.prange(sxx.shape[0]):
        spd[i, :] = np.histogram(sxx[i, :], bin_edges)[0] / ((bin_edges.size - 1) * h)
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
def calculate_aci(sxx):
    """
    Return the aci of the signal
    """
    aci_val = 0
    for j in np.arange(sxx.shape[1]):
        d = 0
        i = 0
        for k in np.arange(1, sxx.shape[0]):
            dk = np.abs(sxx[k][j] - sxx[k - 1][j])
            d = d + dk
            i = i + sxx[k][j]
        aci_val = aci_val + d / i

    return aci_val


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


def octdsgn(fc, fs, fraction=1, n=2):
    """
    Design of a octave band filter with center frequency fc for sampling frequency fs.
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
    # f1 = fc / (2 ** (1 / 6))
    # f2 = fc * (2 ** (1 / 6))
    qr = fc / (f2 - f1)
    qd = (np.pi / 2 / n) / (np.sin(np.pi / 2 / n)) * qr
    alpha = (1 + np.sqrt(1 + 4 * qd ** 2)) / 2 / qd
    w1 = fc / (fs / 2) / alpha
    w2 = fc / (fs / 2) * alpha
    sos = sig.butter(n, [w1, w2], btype='bandpass', output='sos')

    return sos


def octbankdsgn(fs, bands, fraction=1, n=2):
    """
    Construction of a octave band filterbank.

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
    ref = 1000
    G = 10.0**(3.0 / 10.0)
    uneven = (fraction % 2 != 0)
    fc = ref * G**((2.0 * bands + 1.0) / (2.0 * fraction)) * np.logical_not(uneven) + uneven * ref * G**(bands / fraction)
    # fc = 1000 * ((2 ** (1 / fraction)) ** bands)  # exact center frequencies

    fclimit = 1 / 200  # limit for center frequency compared to sample frequency
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
