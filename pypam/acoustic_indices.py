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
import maad.features


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
    bi = maad.features.alpha_indices.bioacoustics_index(Sxx=sxx, fn=frequencies, flim=(min_freq, max_freq),
                                                        R_compatible='soundecology')
    return bi


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
    sh, _ = maad.features.alpha_indices.frequency_entropy(X=sxx)
    return sh


def compute_th(s):
    """
    Compute Temporal Entropy of Shannon from an audio signal.
    Ported from the seewave R package.

    Parameters
    ----------
    s: np.array
        Signal
    """
    th = maad.features.alpha_indices.temporal_entropy(s=s)
    return th


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
    aei = maad.features.alpha_indices.acoustic_eveness_index(Sxx=sxx, fn=frequencies, fmin=min_freq, fmax=max_freq,
                                                             bin_step=freq_step, dB_threshold=db_threshold)
    return aei


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
    adi = maad.features.alpha_indices.acoustic_diversity_index(Sxx=sxx, fn=frequencies, fmin=min_freq, fmax=max_freq,
                                                               bin_step=freq_step, dB_threshold=db_threshold,
                                                               index='shannon')
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
