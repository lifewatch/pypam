#!/usr/bin/env python
"""
Adapted version of acoustic_indexes.py (https://github.com/patriceguyot/Acoustic_Indices)

Set of functions to compute acoustic indices in the framework of Soundscape Ecology.
Some features are inspired or ported from those proposed in:
    - seewave R package (http://rug.mnhn.fr/seewave/) / Jerome Sueur, Thierry Aubin and
    Caroline Simonis
    - soundecology R package (http://cran.r-project.org/web/packages/soundecology/index.html) /
    Luis J. Villanueva-Rivera and Bryan C. Pijanowski
This file use an object oriented type for audio files described in the file "acoustic_index.py".
"""

__author__ = "Patrice Guyot"
__version__ = "0.4"
__credits__ = ["Patrice Guyot", "Alice Eldridge", "Mika Peck"]
__email__ = ["guyot.patrice@gmail.com", "alicee@sussex.ac.uk", "m.r.peck@sussex.ac.uk"]
__status__ = "Development"

import numba as nb
import numpy as np
from scipy import fftpack
from scipy import signal as sig


# def compute_aci(sxx, j_bin):
#     """
#     Compute the Acoustic Complexity Index from the spectrogram of an audio signal.
#     Reference: Pieretti N, Farina A, Morri FD (2011) A new methodology to infer the singing activity of an avian
#     community: the Acoustic Complexity Index (ACI). Ecological Indicators, 11, 868-873.
#     Ported from the soundecology R package.
#
#     Parameters
#     ----------
#     sxx:
#         the spectrogram of the audio signal
#     j_bin:
#         temporal size of the frame (in samples)
#     """
#     # relevant time indices
#     # times = range(0, sxx.shape[1], j_bin)
#     # alternative time indices to follow the R code
#     times = range(0, sxx.shape[1] - 10, j_bin)
#
#     # sub-spectros of temporal size j
#     jspecs = [np.array(sxx[:, i:i + j_bin]) for i in times]
#     aci = [sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1))) for jspec in jspecs]
#     # list of ACI values on each jspecs
#     main_value = sum(aci)
#     temporal_values = aci
#
#     return main_value, temporal_values


@nb.njit
def compute_aci(sxx: np.ndarray):
    """
    Return the aci of the signal

    Parameters
    ----------
    sxx : np.array 2D
        Spectrogram of the signal

    Returns
    -------
    ACI value
    """
    aci_evo = np.zeros(sxx.shape[1], dtype=np.float64)
    for j in np.arange(sxx.shape[1]):
        d = 0
        i = 0
        for k in np.arange(1, sxx.shape[0]):
            dk = np.abs(sxx[k][j] - sxx[k - 1][j])
            d = d + dk
            i = i + sxx[k][j]
        aci_evo[j] = d / i

    aci_val = np.sum(aci_evo)
    return aci_val


@nb.njit
def compute_bi(sxx, frequencies, min_freq=2000, max_freq=8000):
    """
    Compute the Bioacoustic Index from the spectrogram of an audio signal.
    In this code, the Bioacoustic Index correspond to the area under the mean spectre (in db)
    minus the minimum frequency value of this mean spectre.
    Reference: Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic invasion resistance
    in Hawaii: bioacoustics, field surveys, and airborne remote sensing.
    Ecological Applications 17: 2137-2144.
    Ported from the soundecology R package.

    Parameters
    ----------
    sxx: np.array 2D
        The spectrogram of the audio signal
    frequencies: np.array 1D
        List of the frequencies of the spectrogram
    min_freq: int
        Minimum frequency (in Hertz)
    max_freq: int
        Maximum frequency (in Hertz)

    Returns
    -------
    Bioacoustic Index (BI) value
    """
    # min freq in samples (or bin)
    min_freq_bin = int(np.argmin(np.abs(frequencies - min_freq)))
    # max freq in samples (or bin)
    max_freq_bin = int(np.ceil(np.argmin(np.abs(frequencies - max_freq))))

    # alternative value to follow the R code
    # min_freq_bin = min_freq_bin - 1

    #  Use of decibel values. Equivalent in the R code to:
    #  spec_left <- spectro(left, f = samplingrate, wl = fft_w, plot = FALSE, db = "max0")$amp
    spectro_bi = 10 * np.log10(sxx ** 2 / np.max(sxx ** 2))

    # Compute the mean for each frequency (the output is a spectre).
    # This is not exactly the mean, but it is equivalent to the R code to: return(a*log10(mean(10^(x/a))))
    spectre_bi_mean = np.zeros(spectro_bi.shape[0])
    for k in np.arange(spectro_bi.shape[0]):
        spectre_bi_mean[k] = 10 * np.log10(np.mean(10 ** (spectro_bi[k] / 10)))

    # Segment between min_freq and max_freq
    spectre_bi_mean_segment = spectre_bi_mean[min_freq_bin:max_freq_bin]

    # Normalization: set the minimum value of the frequencies to zero.
    spectre_bi_mean_segment_normalized = spectre_bi_mean_segment - np.min(spectre_bi_mean_segment)

    # Compute the area under the spectre curve.
    # Equivalent in the R code to: left_area <- sum(specA_left_segment_normalized * rows_width)
    area = np.sum(spectre_bi_mean_segment_normalized / (frequencies[1] - frequencies[0]))
    return area


@nb.njit
def compute_sh(sxx):
    """
    Compute Spectral Entropy of Shannon from the spectrogram of an audio signal.
    Ported from the seewave R package.

    Parameters
    ----------
    sxx: np.array 2D
     The spectrogram of the audio signal

    Returns
    -------
    Spectral Entropy (SH)
    """
    n = sxx.shape[0]
    spec = np.sum(sxx, axis=1)
    spec = spec / np.sum(spec)  # Normalization by the sum of the values
    main_value = 0
    for y in spec:
        main_value += y * np.log2(y)
    # Equivalent in the R code to: z <- -sum(spec*log(spec))/log(n)
    # temporal_values = [- sum([y * np.log2(y) for y in frame]) / (np.sum(frame) * np.log2(n)) for frame in sxx.T]
    return - main_value / np.log2(n)


def compute_th(s):
    """
    Compute Temporal Entropy of Shannon from an audio signal.
    Ported from the seewave R package.

    Parameters
    ----------
    s: np.array
        Signal
    """
    # Modulo of the Hilbert Envelope, computed with the next fast length window
    env = np.abs(sig.hilbert(s, fftpack.helper.next_fast_len(len(s))))

    # Normalization
    env = env / np.sum(env)
    n = len(env)
    th = 0
    for y in env:
        th += y * np.log2(y)
    return - th / np.log2(n)


def compute_ndsi(s, fs, window_length=1024, anthrophony=None, biophony=None):
    """
    Compute Normalized Difference Sound Index from an audio signal.
    This function computes an estimate power spectral density using Welch's method.
    Reference: Kasten, Eric P., Stuart H. Gage, Jordan Fox, and Wooyeong Joo. 2012.
    The Remote Environ- mental Assessment Laboratory's Acoustic Library: An Archive for Studying
    Soundscape Ecology.
    Ecological Informatics 12: 50-67.
    Inspired by the seewave R package, the soundecology R package and the original matlab code from the authors.

    Parameters
    ----------
    s: np.array
        Signal
    fs : int
        Sampling rate
    window_length: int
        the length of the window for the Welch's method.
    anthrophony: list of ints
        list of two values containing the minimum and maximum frequencies (in Hertz) for antrophony.
    biophony: list of ints
        list of two values containing the minimum and maximum frequencies (in Hertz) for biophony.
    """
    # frequencies, pxx = signal.welch(file.sig_float, fs=file.sr, window='hamming',
    # nperseg=window_length,
    # noverlap=window_length/2, nfft=window_length, detrend=False, return_onesided=True,
    # scaling='density', axis=-1)
    # Estimate power spectral density using Welch's method
    # TODO change of detrend for apollo
    # Estimate power spectral density using Welch's method
    if biophony is None:
        biophony = [2000, 11000]
    if anthrophony is None:
        anthrophony = [1000, 2000]
    frequencies, pxx = sig.welch(s, fs=fs, window='hamming', nperseg=window_length,
                                 noverlap=int(window_length / 2), nfft=window_length, detrend='constant',
                                 return_onesided=True, scaling='density', axis=-1)
    avgpow = pxx * frequencies[1]
    # use a rectangle approximation of the integral of the signal's power spectral density (PSD)
    # avgpow = avgpow / np.linalg.norm(avgpow, ord=2)
    # Normalization (doesn't change the NDSI values. Slightly differ from the matlab code).

    # min freq of anthrophony in samples (or bin) (closest bin)
    min_anthro_bin = np.argmin(np.abs(frequencies - anthrophony[0]))

    # max freq of anthrophony in samples (or bin)
    max_anthro_bin = np.argmin(np.abs(frequencies - anthrophony[1]))

    # min freq of biophony in samples (or bin)
    min_bio_bin = np.argmin(np.abs(frequencies - biophony[0]))

    # max freq of biophony in samples (or bin)
    max_bio_bin = np.argmin(np.abs(frequencies - biophony[1]))

    min_anthro_bin = np.argmin(min_anthro_bin)
    anthro = np.sum(avgpow[min_anthro_bin:max_anthro_bin])
    bio = np.sum(avgpow[min_bio_bin:max_bio_bin])

    ndsi = (bio - anthro) / (bio + anthro)
    return ndsi


@nb.njit
def gini(values):
    """
    Compute the Gini index of values.
    Inspired by http://mathworld.wolfram.com/GiniCoefficient.html and
    http://en.wikipedia.org/wiki/Gini_coefficient

    Parameters
    ----------
    values: np.array or list
        A list of values

    """
    y = np.sort(values)
    n = len(y)
    g = 0
    for i, j in zip(y, np.arange(1, n + 1)):
        g += i * j
    g = 2 * g / np.sum(y) - (n + 1)
    return g / n


@nb.njit
def compute_aei(sxx, frequencies, max_freq=10000, min_freq=0, db_threshold=-50, freq_step=1000):
    """
    Compute Acoustic Evenness Index of an audio signal.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011.
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    Ported from the soundecology R package.

    Parameters
    ----------
    sxx: 2d np array
        Spectrogram of the audio signal
    frequencies: list of ints
        Frequencies list of the spectrogram
    max_freq: int
        The maximum frequency to consider to compute AEI (in Hertz)
    min_freq: int
        The minimum frequency to consider to compute AEI (in Hertz)
    db_threshold: int or float
        The minimum db value to consider for the bins of the spectrogram
    freq_step: int
        Size of frequency bands to compute AEI (in Hertz)
    """
    bands_hz = np.arange(min_freq, max_freq, freq_step)

    spec_aei = 10 * np.log10(sxx ** 2 / np.max(sxx) ** 2)
    values = np.zeros(bands_hz.size)
    for k in np.arange(bands_hz.size - 1):
        spec_aei_band = spec_aei[np.where((frequencies > bands_hz[k]) & (frequencies < bands_hz[k + 1]))]
        values[k] = np.sum(spec_aei_band > db_threshold) / float(spec_aei_band.size)

    return gini(values)


@nb.njit
def compute_adi(sxx, frequencies, max_freq=10000, min_freq=0, db_threshold=-50, freq_step=1000):
    """
    Compute Acoustic Diversity Index.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011.
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    Ported from the soundecology R package.

    Parameters
    ----------
    sxx:
        Spectrogram of the audio signal
    frequencies: list of ints
        Frequencies list of the spectrogram
    max_freq: int
        The maximum frequency to consider to compute AEI (in Hertz)
    min_freq: int
        The minimum frequency to consider to compute AEI (in Hertz)
    db_threshold: int or float
        The minimum db value to consider for the bins of the spectrogram
    freq_step: int
        Size of frequency bands to compute AEI (in Hertz)
    """
    bands_hz = np.arange(min_freq, max_freq, freq_step)

    spec_adi = 10 * np.log10(sxx ** 2 / np.max(sxx ** 2))
    values = np.zeros(bands_hz.size)
    for k in range(bands_hz.size - 1):
        spec_adi_band = spec_adi[np.where((frequencies > bands_hz[k]) & (frequencies < bands_hz[k + 1]))]
        values[k] = np.sum(spec_adi_band > db_threshold) / float(spec_adi_band.size)

    # Shannon Entropy of the values
    # shannon = - sum([y * np.log(y) for y in values]) / len(values)  # Follows the R code.
    # But log is generally log2 for Shannon entropy. Equivalent to shannon = False in soundecology.

    # The following is equivalent to shannon = True (default) in soundecology.
    # Compute the Shannon diversity index from the R function diversity {vegan}.
    # v = [x/np.sum(values) for x in values]
    # v2 = [-i * j  for i,j in zip(v, np.log(v))]
    # return np.sum(v2)

    # Remove zero values (Jan 2016)
    # values = [value for value in values if value != 0]
    values = values[np.nonzero(values)]

    # replace zero values by 1e-07 (closer to R code, but results quite similars)
    # values = [x if x != 0 else 1e-07 for x in values]
    adi = 0
    for i in values:
        adi += -i / np.sum(values) * np.log(i / np.sum(values))
    # adi = np.sum([-i / np.sum(values) * np.log(i / np.sum(values)) for i in values])
    return adi


@nb.njit
def compute_zcr_avg(s, window_length=512, window_hop=256):
    """
    Compute the Zero Crossing Rate of an audio signal.

    Parameters
    ----------
    s: np.array
        Signal to process
    window_length: int
        Size of the sliding window (samples)
    window_hop: int
        Size of the lag window (samples)

    Returns
    -------
    A list of values (number of zero crossing for each window)
    """
    times = np.arange(0, len(s) - window_length + window_hop)
    zcr_bins = np.zeros(times.size)
    for k, i in enumerate(times):
        x = s[i: i + window_length]
        zcr_bins[k] = len(np.where(np.diff(np.signbit(x)))[0]) / float(window_length)

    return np.mean(zcr_bins)


@nb.njit
def compute_zcr(s):
    """
    Compute the Zero Crossing Rate of an audio signal.

    Parameters
    ----------
    s: np.array

    Returns
    -------
    A list of values (number of zero crossing for each window)
    """
    return len(np.where(np.diff(np.signbit(s)))[0]) / float(len(s))
