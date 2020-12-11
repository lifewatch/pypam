#!/usr/bin/env python
"""
Adapted version of acoustic_indexes.py (https://github.com/patriceguyot/Acoustic_Indices)

Set of functions to compute acoustic indices in the framework of Soundscape Ecology.
Some features are inspired or ported from those proposed in:
    - seewave R package (http://rug.mnhn.fr/seewave/) / Jerome Sueur, Thierry Aubin and  Caroline Simonis
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


def compute_aci(sxx, j_bin):
    """
    Compute the Acoustic Complexity Index from the spectrogram of an audio signal.
    Reference: Pieretti N, Farina A, Morri FD (2011) A new methodology to infer the singing activity of an avian
    community: the Acoustic Complexity Index (ACI). Ecological Indicators, 11, 868-873.
    Ported from the soundecology R package.

    Parameters
    ----------
    sxx:
        the spectrogram of the audio signal
    j_bin:
        temporal size of the frame (in samples)
    """
    # relevant time indices
    # times = range(0, sxx.shape[1], j_bin)
    # alternative time indices to follow the R code
    times = range(0, sxx.shape[1] - 10, j_bin)

    # sub-spectros of temporal size j
    jspecs = [np.array(sxx[:, i:i + j_bin]) for i in times]
    aci = [sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1))) for jspec in jspecs]
    # list of ACI values on each jspecs
    main_value = sum(aci)
    temporal_values = aci

    return main_value, temporal_values


@nb.jit
def calculate_aci(sxx: np.ndarray):
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
        d = np.float64(0)
        i = np.float64(0)
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
    In this code, the Bioacoustic Index correspond to the area under the mean spectre (in db) minus the minimum
    frequency value of this mean spectre.
    Reference: Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic invasion resistance in Hawaii:
    bioacoustics, field surveys, and airborne remote sensing. Ecological Applications 17: 2137-2144.
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
    min_freq_bin = int(np.argmin([abs(e - min_freq) for e in frequencies]))
    # max freq in samples (or bin)
    max_freq_bin = int(np.ceil(np.argmin([abs(e - max_freq) for e in frequencies])))

    # alternative value to follow the R code
    min_freq_bin = min_freq_bin - 1

    #  Use of decibel values. Equivalent in the R code to:
    #  spec_left <- spectro(left, f = samplingrate, wl = fft_w, plot = FALSE, db = "max0")$amp
    spectro_bi = 20 * np.log10(sxx / np.max(sxx))

    # Compute the mean for each frequency (the output is a spectre).
    # This is not exactly the mean, but it is equivalent to the R code to: return(a*log10(mean(10^(x/a))))
    spectre_bi_mean = 10 * np.log10(np.mean(10 ** (spectro_bi / 10), axis=1))

    # Segment between min_freq and max_freq
    spectre_bi_mean_segment = spectre_bi_mean[min_freq_bin:max_freq_bin]

    # Normalization: set the minimum value of the frequencies to zero.
    spectre_bi_mean_segment_normalized = spectre_bi_mean_segment - min(spectre_bi_mean_segment)

    # Compute the area under the spectre curve.
    # Equivalent in the R code to: left_area <- sum(specA_left_segment_normalized * rows_width)
    area = np.sum(spectre_bi_mean_segment_normalized / (frequencies[1] - frequencies[0]))
    return area


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
    main_value = - sum([y * np.log2(y) for y in spec]) / np.log2(n)
    # Equivalent in the R code to: z <- -sum(spec*log(spec))/log(n)
    # temporal_values = [- sum([y * np.log2(y) for y in frame]) / (np.sum(frame) * np.log2(n)) for frame in sxx.T]
    return main_value


def compute_th(s, integer=True):
    """
    Compute Temporal Entropy of Shannon from an audio signal.
    Ported from the seewave R package.

    Parameters
    ----------
    s: np.array
        Signal
    integer: bool
        If set as True, the Temporal Entropy will be compute on the Integer values of the signal.
        If not, the signal will be set between -1 and 1.
    """
    if integer:
        s = s.astype(np.int)
    # env = abs(sig.hilbert(sig)) # Modulo of the Hilbert Envelope
    env = abs(sig.hilbert(s, fftpack.helper.next_fast_len(len(s))))
    # Modulo of the Hilbert Envelope, computed with the next fast length window

    # Normalization
    env = env / np.sum(env)
    n = len(env)
    return - sum([y * np.log2(y) for y in env]) / np.log2(n)


def compute_ndsi(file, window_length=1024, anthrophony=[1000, 2000], biophony=[2000, 11000]):
    """
    Compute Normalized Difference Sound Index from an audio signal.
    This function compute an estimate power spectral density using Welch's method.
    Reference: Kasten, Eric P., Stuart H. Gage, Jordan Fox, and Wooyeong Joo. 2012.
    The Remote Environ- mental Assessment Laboratory's Acoustic Library: An Archive for Studying Soundscape Ecology.
    Ecological Informatics 12: 50-67.
    window_length: the length of the window for the Welch's method.
    anthrophony: list of two values containing the minimum and maximum frequencies (in Hertz) for antrophony.
    biophony: list of two values containing the minimum and maximum frequencies (in Hertz) for biophony.
    Inspired by the seewave R package, the soundecology R package and the original matlab code from the authors.
    """
    # frequencies, pxx = signal.welch(file.sig_float, fs=file.sr, window='hamming', nperseg=window_length,
    # noverlap=window_length/2, nfft=window_length, detrend=False, return_onesided=True, scaling='density', axis=-1)
    # Estimate power spectral density using Welch's method
    # TODO change of detrend for apollo
    # Estimate power spectral density using Welch's method
    frequencies, pxx = sig.welch(file.sig_float, fs=file.sr, window='hamming', nperseg=window_length,
                                 noverlap=window_length / 2, nfft=window_length, detrend='constant',
                                 return_onesided=True, scaling='density', axis=-1)

    avgpow = pxx * frequencies[1]
    # use a rectangle approximation of the integral of the signal's power spectral density (PSD)
    # avgpow = avgpow / np.linalg.norm(avgpow, ord=2)
    # Normalization (doesn't change the NDSI values. Slightly differ from the matlab code).

    # min freq of anthrophony in samples (or bin) (closest bin)
    min_anthro_bin = np.argmin([abs(e - anthrophony[0]) for e in frequencies])

    # max freq of anthrophony in samples (or bin)
    max_anthro_bin = np.argmin([abs(e - anthrophony[1]) for e in frequencies])

    # min freq of biophony in samples (or bin)
    min_bio_bin = np.argmin([abs(e - biophony[0]) for e in frequencies])

    # max freq of biophony in samples (or bin)
    max_bio_bin = np.argmin([abs(e - biophony[1]) for e in frequencies])

    anthro = np.sum(avgpow[min_anthro_bin:max_anthro_bin])
    bio = np.sum(avgpow[min_bio_bin:max_bio_bin])

    ndsi = (bio - anthro) / (bio + anthro)
    return ndsi


def gini(values):
    """
    Compute the Gini index of values.
    Inspired by http://mathworld.wolfram.com/GiniCoefficient.html and http://en.wikipedia.org/wiki/Gini_coefficient

    Parameters
    ----------
    values: np.array or list
        A list of values

    """
    y = sorted(values)
    n = len(y)
    g = np.sum([i * j for i, j in zip(y, range(1, n + 1))])
    g = 2 * g / np.sum(y) - (n + 1)
    return g / n


def compute_aei(sxx, freq_band_hz, max_freq=10000, db_threshold=-50, freq_step=1000):
    """
    Compute Acoustic Evenness Index of an audio signal.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011.
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    Ported from the soundecology R package.

    Parameters
    ----------
    sxx:
        Spectrogram of the audio signal
    freq_band_hz: int
        Frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: int
        The maximum frequency to consider to compute AEI (in Hertz)
    db_threshold: int or float
        The minimum db value to consider for the bins of the spectrogram
    freq_step: int
        Size of frequency bands to compute AEI (in Hertz)
    """
    bands_hz = range(0, max_freq, freq_step)
    bands_bin = [f / freq_band_hz for f in bands_hz]

    spec_aei = 20 * np.log10(sxx / np.max(sxx))
    spec_aei_bands = [spec_aei[int(bands_bin[k]):int(bands_bin[k] + bands_bin[1]), ] for k in range(len(bands_bin))]

    values = [np.sum(spec_aei_bands[k] > db_threshold) / float(spec_aei_bands[k].size) for k in range(len(bands_bin))]

    return gini(values)


def compute_adi(sxx, freq_band_hz, max_freq=10000, db_threshold=-50, freq_step=1000):
    """
    Compute Acoustic Diversity Index.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011.
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    Ported from the soundecology R package.

    Parameters
    ----------
    sxx:
        Spectrogram of the audio signal
    freq_band_hz: int
        Frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: int
        The maximum frequency to consider to compute AEI (in Hertz)
    db_threshold: int or float
        The minimum db value to consider for the bins of the spectrogram
    freq_step: int
        Size of frequency bands to compute AEI (in Hertz)
    """
    bands_hz = range(0, max_freq, freq_step)
    bands_bin = [f / freq_band_hz for f in bands_hz]

    spec_adi = 20 * np.log10(sxx / np.max(sxx))
    spec_adi_bands = [spec_adi[int(bands_bin[k]):int(bands_bin[k] + bands_bin[1]), ] for k in range(len(bands_bin))]

    values = [np.sum(spec_adi_bands[k] > db_threshold) / float(spec_adi_bands[k].size) for k in range(len(bands_bin))]

    # Shannon Entropy of the values
    # shannon = - sum([y * np.log(y) for y in values]) / len(values)  # Follows the R code.
    # But log is generally log2 for Shannon entropy. Equivalent to shannon = False in soundecology.

    # The following is equivalent to shannon = True (default) in soundecology.
    # Compute the Shannon diversity index from the R function diversity {vegan}.
    # v = [x/np.sum(values) for x in values]
    # v2 = [-i * j  for i,j in zip(v, np.log(v))]
    # return np.sum(v2)

    # Remove zero values (Jan 2016)
    values = [value for value in values if value != 0]

    # replace zero values by 1e-07 (closer to R code, but results quite similars)
    # values = [x if x != 0 else 1e-07 for x in values]

    return np.sum([-i / np.sum(values) * np.log(i / np.sum(values)) for i in values])


def compute_zcr(s, window_length=512, window_hop=256):
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
    times = range(0, len(s) - window_length + window_hop)
    frames = [s[i: i + window_length] for i in times]
    return [len(np.where(np.diff(np.signbit(x)))[0]) / float(window_length) for x in frames]


def compute_bn_peaks(sxx, frequencies, freqband=200, normalization=True, slopes=(0.01, 0.01)):
    """
    Counts the number of major frequency peaks obtained on a mean spectrum.
    Ref: Gasc, A., Sueur, J., Pavoine, S., Pellens, R., & Grandcolas, P. (2013). Biodiversity sampling using a global
    acoustic approach: contrasting sites with microendemics in New Caledonia. PloS one, 8(5), e65311.

    Parameters
    ----------
    sxx: np.array 2D
        Spectrogram of the audio signal
    frequencies: np.array 1D
        List of the frequencies of the spectrogram
    freqband: int or float
        frequency threshold parameter (in Hz). If the frequency difference of two successive peaks is
        less than this threshold, then the peak of highest amplitude will be kept only.
        normalization: if set at True, the mean spectrum is scaled between 0 and 1
    slopes: tuple of length 2
        Amplitude slope parameter, a tuple of length 2. Refers to the amplitude slopes of the peak.
        The first value is the left slope and the second value is the right slope. Only peaks with higher
        slopes than threshold values will be kept.
    """
    meanspec = np.array([np.mean(row) for row in sxx])

    if normalization:
        meanspec = meanspec / np.max(meanspec)

    # Find peaks (with slopes)
    peaks_indices = np.r_[False, meanspec[1:] > np.array([x + slopes[0] for x in meanspec[:-1]])] & np.r_[
        meanspec[:-1] > np.array([y + slopes[1] for y in meanspec[1:]]), False]
    peaks_indices = peaks_indices.nonzero()[0].tolist()

    peaks_indices = sig.argrelextrema(np.array(meanspec), np.greater)[0].tolist()  # scipy method (without slope)

    # Remove peaks with difference of frequency < freqband
    # number of consecutive index
    nb_bin = next(i for i, v in enumerate(frequencies) if v > freqband)
    for consecutiveIndices in [np.arange(i, i + nb_bin) for i in peaks_indices]:
        if len(np.intersect1d(consecutiveIndices, peaks_indices)) > 1:
            # close values has been found
            maxi = np.intersect1d(consecutiveIndices, peaks_indices)[
                np.argmax([meanspec[f] for f in np.intersect1d(consecutiveIndices, peaks_indices)])]
            peaks_indices = [x for x in peaks_indices if x not in consecutiveIndices]
            # remove all indices that are in consecutiveIndices
            # append the max
            peaks_indices.append(maxi)
    peaks_indices.sort()

    # Frequencies of the peaks
    peak_freqs = [frequencies[p] for p in peaks_indices]
    return len(peaks_indices), peak_freqs
