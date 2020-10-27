#!/usr/bin/env python

"""
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


import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy import signal, fftpack, stats

from pypam import utils


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
    times = range(0, sxx.shape[1]-10, j_bin)

    # sub-spectros of temporal size j
    jspecs = [np.array(sxx[:, i:i+j_bin]) for i in times]
    aci = [sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1))) for jspec in jspecs]
    # list of ACI values on each jspecs
    main_value = sum(aci)
    temporal_values = aci

    return main_value, temporal_values


@nb.jit
def calculate_aci(sxx: np.ndarray):
    """
    Return the aci of the signal
    """
    aci_evo = np.zeros(sxx.shape[1], dtype=np.float64)
    for j in np.arange(sxx.shape[1]):
        d = np.float64(0)
        i = np.float64(0)
        for k in np.arange(1, sxx.shape[0]):
            dk = np.abs(sxx[k][j] - sxx[k - 1][j])
            d = d + dk
            i = i + sxx[k][j]
        aci_evo[j] = d/i

    aci_val = np.sum(aci_evo)
    return aci_val


def compute_bi(sxx, frequencies, min_freq=2000, max_freq=8000):
    """
    Compute the Bioacoustic Index from the spectrogram of an audio signal.
    In this code, the Bioacoustic Index correspond to the area under the mean spectre (in db) minus the minimum
    frequency value of this mean spectre.
    Reference: Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic invasion resistance in Hawaii:
    bioacoustics, field surveys, and airborne remote sensing. Ecological Applications 17: 2137-2144.
    spectro: the spectrogram of the audio signal
    frequencies: list of the frequencies of the spectrogram
    min_freq: minimum frequency (in Hertz)
    max_freq: maximum frequency (in Hertz)
    Ported from the soundecology R package.
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
    spectro: the spectrogram of the audio signal
    Ported from the seewave R package.
    """
    n = sxx.shape[0]
    spec = np.sum(sxx, axis=1)
    spec = spec / np.sum(spec)  # Normalization by the sum of the values
    main_value = - sum([y * np.log2(y) for y in spec]) / np.log2(n)
    # Equivalent in the R code to: z <- -sum(spec*log(spec))/log(n)
    # temporal_values = [- sum([y * np.log2(y) for y in frame]) / (np.sum(frame) * np.log2(n)) for frame in sxx.T]
    return main_value


def compute_th(file, integer=True):
    """
    Compute Temporal Entropy of Shannon from an audio signal.
    file: an instance of the AudioFile class.
    integer: if set as True, the Temporal Entropy will be compute on the Integer values of the signal.
    If not, the signal will be set between -1 and 1.
    Ported from the seewave R package.
    """
    if integer:
        sig = file.sig_int
    else:
        sig = file.sig_float

    # env = abs(signal.hilbert(sig)) # Modulo of the Hilbert Envelope
    env = abs(signal.hilbert(sig, fftpack.helper.next_fast_len(len(sig))))
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
    frequencies, pxx = signal.welch(file.sig_float, fs=file.sr, window='hamming', nperseg=window_length,
                                    noverlap=window_length/2, nfft=window_length, detrend='constant',
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
    values: a list of values
    Inspired by http://mathworld.wolfram.com/GiniCoefficient.html and http://en.wikipedia.org/wiki/Gini_coefficient
    """
    y = sorted(values)
    n = len(y)
    g = np.sum([i*j for i, j in zip(y, range(1, n+1))])
    g = 2 * g / np.sum(y) - (n+1)
    return g/n


def compute_aei(sxx, freq_band_hz, max_freq=10000, db_threshold=-50, freq_step=1000):
    """
    Compute Acoustic Evenness Index of an audio signal.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011.
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    spectro: spectrogram of the audio signal
    freq_band_Hz: frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: the maximum frequency to consider to compute AEI (in Hertz)
    db_threshold: the minimum db value to consider for the bins of the spectrogram
    freq_step: size of frequency bands to compute AEI (in Hertz)
    Ported from the soundecology R package.
    """
    bands_hz = range(0, max_freq, freq_step)
    bands_bin = [f / freq_band_hz for f in bands_hz]

    spec_aei = 20*np.log10(sxx/np.max(sxx))
    spec_aei_bands = [spec_aei[int(bands_bin[k]):int(bands_bin[k]+bands_bin[1]), ] for k in range(len(bands_bin))]

    values = [np.sum(spec_aei_bands[k] > db_threshold)/float(spec_aei_bands[k].size) for k in range(len(bands_bin))]

    return gini(values)


def compute_adi(sxx, freq_band_hz,  max_freq=10000, db_threshold=-50, freq_step=1000):
    """
    Compute Acoustic Diversity Index.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011.
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    spectro: spectrogram of the audio signal
    freq_band_Hz: frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: the maximum frequency to consider to compute ADI (in Hertz)
    db_threshold: the minimum db value to consider for the bins of the spectrogram
    freq_step: size of frequency bands to compute ADI (in Hertz)
    Ported from the soundecology R package.
    """
    bands_hz = range(0, max_freq, freq_step)
    bands_bin = [f / freq_band_hz for f in bands_hz]

    spec_adi = 20*np.log10(sxx/np.max(sxx))
    spec_adi_bands = [spec_adi[int(bands_bin[k]):int(bands_bin[k]+bands_bin[1]), ] for k in range(len(bands_bin))]

    values = [np.sum(spec_adi_bands[k] > db_threshold)/float(spec_adi_bands[k].size) for k in range(len(bands_bin))]

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


def compute_zcr(file, window_length=512, window_hop=256):
    """
    Compute the Zero Crossing Rate of an audio signal.
    file: an instance of the AudioFile class.
    window_length: size of the sliding window (samples)
    window_hop: size of the lag window (samples)
    return: a list of values (number of zero crossing for each window)
    """
    # Signal on integer values
    sig = file.sig_int

    times = range(0, len(sig) - window_length + window_hop)
    frames = [sig[i: i+window_length] for i in times]
    return [len(np.where(np.diff(np.signbit(x)))[0])/float(window_length) for x in frames]


def compute_rms_energy(file, window_length=512, window_hop=256, integer=False):
    """
    Compute the RMS short time energy.
    file: an instance of the AudioFile class.
    window_length: size of the sliding window (samples)
    window_hop: size of the lag window (samples)
    integer: if set as True, the Temporal Entropy will be compute on the Integer values of the signal.
    If not, the signal will be set between -1 and 1.
    return: a list of values (rms energy for each window)
    """
    if integer:
        sig = file.sig_int
    else:
        sig = file.sig_float

    times = range(0, len(sig) - window_length+1, window_hop)
    frames = [sig[i:i + window_length] for i in times]
    return [np.sqrt(sum([x**2 for x in frame]) / window_length) for frame in frames]


def compute_spectral_centroid(sxx, frequencies):
    """
    Compute the spectral centroid of an audio signal from its spectrogram.
    spectro: spectrogram of the audio signal
    frequencies: list of the frequencies of the spectrogram
    """
    centroid = [np.sum(magnitudes*frequencies) / np.sum(magnitudes) for magnitudes in sxx.T]
    return centroid


def compute_wave_snr(file, frame_length_e=512, min_db=-60, window_smoothing_e=5, activity_threshold_db=3,
                     hist_number_bins=100, db_range=10, n=0):
    """
    Computes indices from the Signal to Noise Ratio of a waveform.
    file: an instance of the AudioFile class.
    window_smoothing_e: odd number for sliding mean smoothing of the histogram (can be 3, 5 or 7)
    hist_number_bins - Number of columns in the histogram
    db_range - db range to consider in the histogram
    N: The decibel threshold for the waveform is given by the modal intensity plus N times the standard deviation.
    Higher values of N will remove more energy from the waveform.
    Output:
        Signal-to-noise ratio (snr): the decibel difference between the maximum envelope amplitude
        in any minute segment and the background noise.
        Acoustic activity: the fraction of frames within a one minute segment where the signal envelope is more than
        3 db above the level of background noise
        Count of acoustic events: the number of times that the signal envelope crosses the 3 db threshold
        Average duration of acoustic events: an acoustic event is a portion of recordingwhich startswhen
        the signal envelope crosses above the 3 db threshold and ends when it crosses belowthe 3 db threshold.
    Ref: Towsey, Michael W. (2013) Noise removal from wave-forms and spectro- grams derived from natural
    recordings of the environment.
    Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms
    Derived from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    """
    half_window_smoothing = int(window_smoothing_e/2)

    times = range(0, len(file.sig_int)-frame_length_e+1, frame_length_e)
    wave_env = 20*np.log10([np.max(abs(file.sig_float[i: i + frame_length_e])) for i in times])

    # If the minimum value is less than -60db, the minimum is set to -60db
    minimum = np.max((np.min(wave_env), min_db))

    hist, bin_edges = np.histogram(wave_env, range=(minimum, minimum + db_range), bins=hist_number_bins, density=False)

    hist_smooth = ([np.mean(hist[i - half_window_smoothing: i + half_window_smoothing]) 
                    for i in range(half_window_smoothing, len(hist) - half_window_smoothing)])
    hist_smooth = np.concatenate((np.zeros(half_window_smoothing), hist_smooth, np.zeros(half_window_smoothing)))
    modal_intensity = np.argmax(hist_smooth)

    if n > 0:
        count_thresh = 68 * sum(hist_smooth) / 100
        count = hist_smooth[modal_intensity]
        index_bin = 1
        while count < count_thresh:
            if modal_intensity + index_bin <= len(hist_smooth):
                count = count + hist_smooth[modal_intensity + index_bin]
            if modal_intensity - index_bin >= 0:
                count = count + hist_smooth[modal_intensity - index_bin]
            index_bin += 1
        thresh = np.min((hist_number_bins, modal_intensity + n * index_bin))
        background_noise = bin_edges[thresh]
    elif n == 0:
        background_noise = bin_edges[modal_intensity]

    snr = np.max(wave_env) - background_noise
    sn = np.array([frame-background_noise-activity_threshold_db for frame in wave_env])
    acoustic_activity = np.sum([i > 0 for i in sn])/float(len(sn))

    # Compute acoustic events
    start_event = [n[0] for n in np.argwhere((sn[:-1] < 0) & (sn[1:] > 0))]
    end_event = [n[0] for n in np.argwhere((sn[:-1] > 0) & (sn[1:] < 0))]
    if len(start_event) != 0 and len(end_event) != 0:
        if start_event[0] < end_event[0]:
            events = list(zip(start_event, end_event))
        else:
            events = list(zip(end_event, start_event))
        count_acoustic_events = len(events)
        average_duration_e = np.mean([end - begin for begin, end in events])
        average_duration_s = average_duration_e * file.duration / float(len(sn))
    else:
        count_acoustic_events = 0
        average_duration_s = 0

    d = {'snr': snr, 'Acoustic_activity': acoustic_activity, 'Count_acoustic_events': count_acoustic_events,
         'Average_duration': average_duration_s}
    return d


def remove_noise_spectro(sxx, histo_relative_size=8, window_smoothing=5, n=0.1, db=False, plot=False):
    """
    Compute a new spectrogram which is "Noise Removed".
    spectro: spectrogram of the audio signal
    histo_relative_size: ration between the size of the spectrogram and the size of the histogram
    window_smoothing: number of points to apply a mean filtering on the histogram and on the background noise curve
    n: Parameter to set the threshold around the modal intensity
    db: If set at True, the spectrogram is converted in decibels
    plot: if set at True, the function plot the orginal and noise removed spectrograms
    Output:
        Noise removed spectrogram
    Ref: Towsey, Michael W. (2013) Noise removal from wave-forms and spectrograms derived from natural
    recordings of the environment.
    Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived from Natural Recordings of the
    Environment. Queensland University of Technology, Brisbane.
    """
    # Minimum value for the new spectrogram (preferably slightly higher than 0)
    low_value = 1.e-07
    half_window_smoothing = int(window_smoothing / 2)

    if db:
        sxx = 20 * np.log10(sxx)

    len_spectro_e = len(sxx[0])
    histo_size = int(len_spectro_e/histo_relative_size)

    background_noise = []
    for row in sxx:
        hist, bin_edges = np.histogram(row, bins=histo_size, density=False)

        hist_smooth = ([np.mean(hist[i - half_window_smoothing: i + half_window_smoothing])
                        for i in range(half_window_smoothing, len(hist) - half_window_smoothing)])
        hist_smooth = np.concatenate((np.zeros(half_window_smoothing), hist_smooth, np.zeros(half_window_smoothing)))

        modal_intensity = int(np.min([np.argmax(hist_smooth), 95 * histo_size / 100]))
        # test if modal intensity value is in the top 5%
        if n > 0:
            count_thresh = 68 * sum(hist_smooth) / 100
            count = hist_smooth[modal_intensity]
            index_bin = 1
            while count < count_thresh:
                if modal_intensity + index_bin <= len(hist_smooth):
                    count = count + hist_smooth[modal_intensity + index_bin]
                if modal_intensity - index_bin >= 0:
                    count = count + hist_smooth[modal_intensity - index_bin]
                index_bin += 1
            thresh = int(np.min((histo_size, modal_intensity + n * index_bin)))
            background_noise.append(bin_edges[thresh])
        elif n == 0:
            background_noise.append(bin_edges[modal_intensity])

    background_noise_smooth = ([np.mean(background_noise[i - half_window_smoothing: i + half_window_smoothing])
                                for i in range(half_window_smoothing, len(background_noise) - half_window_smoothing)])
    # keep background noise at the end to avoid last row problem (last bin with old microphones)
    background_noise_smooth = np.concatenate((background_noise[0:half_window_smoothing], background_noise_smooth,
                                              background_noise[-half_window_smoothing:]))

    new_spec = np.array([col - background_noise_smooth for col in sxx.T]).T
    # replace negative values by value close to zero
    new_spec = new_spec.clip(min=low_value)

    # Figure
    if plot:
        colormap = "jet"
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        if db:
            plt.imshow(new_spec, origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        else:
            plt.imshow(20*np.log10(new_spec), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        a = fig.add_subplot(1, 2, 2)
        if db:
            plt.imshow(new_spec, origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        else:
            plt.imshow(20*np.log10(sxx), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        plt.show()
    return new_spec


def compute_bn_peaks(sxx, frequencies, freqband=200, normalization=True, slopes=(0.01, 0.01)):
    """
    Counts the number of major frequency peaks obtained on a mean spectrum.
    spectro: spectrogram of the audio signal
    frequencies: list of the frequencies of the spectrogram
    freqband: frequency threshold parameter (in Hz). If the frequency difference of two successive peaks is
    less than this threshold, then the peak of highest amplitude will be kept only.
    normalization: if set at True, the mean spectrum is scaled between 0 and 1
    slopes: amplitude slope parameter, a tuple of length 2. Refers to the amplitude slopes of the peak.
    The first value is the left slope and the second value is the right slope. Only peaks with higher
    slopes than threshold values will be kept.
    Ref: Gasc, A., Sueur, J., Pavoine, S., Pellens, R., & Grandcolas, P. (2013). Biodiversity sampling using a global
    acoustic approach: contrasting sites with microendemics in New Caledonia. PloS one, 8(5), e65311.
    """
    meanspec = np.array([np.mean(row) for row in sxx])

    if normalization:
        meanspec = meanspec/np.max(meanspec)

    # Find peaks (with slopes)
    peaks_indices = np.r_[False, meanspec[1:] > np.array([x + slopes[0] for x in meanspec[:-1]])] & np.r_[meanspec[:-1] > np.array([y + slopes[1] for y in meanspec[1:]]), False]
    peaks_indices = peaks_indices.nonzero()[0].tolist()

    # peaks_indices = signal.argrelextrema(np.array(meanspec), np.greater)[0].tolist() # scipy method (without slope)

    # Remove peaks with difference of frequency < freqband
    # number of consecutive index
    nb_bin = next(i for i, v in enumerate(frequencies) if v > freqband)
    for consecutiveIndices in [np.arange(i, i+nb_bin) for i in peaks_indices]:
        if len(np.intersect1d(consecutiveIndices, peaks_indices)) > 1:
            # close values has been found
            maxi = np.intersect1d(consecutiveIndices, peaks_indices)[np.argmax([meanspec[f] for f in np.intersect1d(consecutiveIndices, peaks_indices)])]
            peaks_indices = [x for x in peaks_indices if x not in consecutiveIndices]
            # remove all indices that are in consecutiveIndices
            # append the max
            peaks_indices.append(maxi)
    peaks_indices.sort()

    peak_freqs = [frequencies[p] for p in peaks_indices]
    # Frequencies of the peaks

    return len(peaks_indices)


# Clausius Duque G. Reis (UFV/USP-SC) - clausiusreis@gmail.com
# Thalisson Nobre Santos (USP-SC)
# Maria Cristina Ferreira de Oliveira (USP-SC)

def gini(values):
    """
    Compute the Gini index of values.
    values: a list of values
    Inspired by http://mathworld.wolfram.com/GiniCoefficient.html and http://en.wikipedia.org/wiki/Gini_coefficient
    """
    y = sorted(values)
    n = len(y)
    G = np.sum([i * j for i, j in zip(y, range(1, n + 1))])
    G = 2 * G / np.sum(y) - (n + 1)
    return G / n


def pcm2float(sig, dtype='float64'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def compute_ACI(sample, rate, timeWindow, min_freq, max_freq, j_bin=5, fft_w=512, db_threshold=50):
    """
    Compute Acoustic Evenness Index of an audio signal.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    spectro: spectrogram of the audio signal
    freq_band_Hz: frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: the maximum frequency to consider to compute AEI (in Hertz)
    db_threshold: the minimum dB value to consider for the bins of the spectrogram
    freq_step: size of frequency bands to compute AEI (in Hertz)
    Ported from the soundecology R package.
    """

    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    j_bin = j_bin + 1

    spectro = sample[min_freqi:max_freqi, :]

    # alternative time indices to follow the R code
    times = range(0, spectro.shape[1] - 10, j_bin)

    # sub-spectros of temporal size j
    jspecs = [np.array(spectro[:, j:j + j_bin]) for j in times]

    # list of ACI values on each jspecs
    aci = [np.sum((np.sum(abs(np.diff(jspecs[k])), axis=1) / np.sum(jspecs[k], axis=1))) for k in range(len(jspecs))]

    result = np.sum(aci)

    return result


def compute_ADI(sample, rate, timeWindow, max_freq=10000, freq_step=1000, fft_w=512, db_threshold=50):
    """
    Compute Acoustic Diversity Index.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    spectro: spectrogram of the audio signal
    freq_band_Hz: frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: the maximum frequency to consider to compute ADI (in Hertz)
    db_threshold: the minimum dB value to consider for the bins of the spectrogram
    freq_step: size of frequency bands to compute ADI (in Hertz)
    Ported from the soundecology R package.
    """

    # freq_band_Hz = (max_freq / freq_step)

    bandSize = (rate / 2) / (fft_w / 2)
    max_freqi = (max_freq / bandSize)
    freq_stepi = freq_step / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = (fft_w / 2) + 1

    bands_Hz = range(0, max_freqi, freq_stepi)
    # bands_bin = [f / freq_band_Hz for f in bands_Hz]

    spec_ADI = sample[0:max_freqi, :]

    spec_ADI_bands = [spec_ADI[bands_Hz[k]:bands_Hz[k + 1], ] for k in range(len(bands_Hz) - 1)]

    values = [np.sum(spec_ADI_bands[k] > -db_threshold) / float(spec_ADI_bands[k].size) for k in
              range(len(bands_Hz) - 1)]
    values = [value for value in values if value != 0]

    result = np.sum([-j / np.sum(values) * np.log(j / np.sum(values)) for j in values])

    return result


def compute_AEI(sample, rate, timeWindow, max_freq=10000, freq_step=1000, db_threshold=50, fft_w=512):
    """
    Compute Acoustic Evenness Index of an audio signal.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    spectro: spectrogram of the audio signal
    freq_band_Hz: frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: the maximum frequency to consider to compute AEI (in Hertz)
    db_threshold: the minimum dB value to consider for the bins of the spectrogram
    freq_step: size of frequency bands to compute AEI (in Hertz)
    Ported from the soundecology R package.
    """

    # freq_band_Hz = (max_freq / freq_step)

    bandSize = (rate / 2) / (fft_w / 2)
    max_freqi = max_freq / bandSize
    freq_stepi = freq_step / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = (fft_w / 2) + 1

    bands_Hz = range(0, max_freqi, freq_stepi)
    # bands_bin = [f / freq_band_Hz for f in bands_Hz]

    spectro = sample[0:max_freqi, :]

    spec_AEI = spectro
    spec_AEI_bands = [spec_AEI[bands_Hz[k]:bands_Hz[k] + bands_Hz[1], ] for k in range(len(bands_Hz) - 1)]

    values = [np.sum(spec_AEI_bands[k] > -db_threshold) / float(spec_AEI_bands[k].size) for k in
              range(len(bands_Hz) - 1)]
    result = gini(values)

    return result


def compute_BIO(sample, rate, timeWindow, min_freq=2000, max_freq=8000, fft_w=512, db_threshold=50):
    """
    Compute the Bioacoustic Index from the spectrogram of an audio signal.
    In this code, the Bioacoustic Index correspond to the area under the mean spectre (in dB) minus the minimum frequency value of this mean spectre.
    Reference: Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic invasion resistance in Hawaii: bioacoustics, field surveys, and airborne remote sensing. Ecological Applications 17: 2137-2144.
    spectro: the spectrogram of the audio signal
    min_freq: minimum frequency (in Hertz)
    max_freq: maximum frequency (in Hertz)
    Ported from the soundecology R package.
    """

    # list of the frequencies of the spectrogram
    frequencies = librosa.fft_frequencies(rate, fft_w)

    # freq_band_Hz=(max_freq / freq_step)

    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    # min freq in samples (or bin)
    min_freq_bin = np.argmin([abs(e - min_freq) for e in frequencies])
    max_freq_bin = int(np.ceil(np.argmin([abs(e - max_freq) for e in frequencies])))  # max freq in samples (or bin)

    min_freq_bin = min_freq_bin - 1  # alternative value to follow the R code

    # Use of decibel values. Equivalent in the R code to:
    # spec_left <- spectro(left, f = samplingrate, wl = fft_w, plot = FALSE, dB = "max0")$amp
    # spectro_BI = 20 * np.log10(spectro/np.max(spectro))
    spectro_BI = spectro

    # Compute the mean for each frequency (the output is a spectre).
    # This is not exactly the mean, but it is equivalent to the R code to: return(a*log10(mean(10^(x/a))))
    spectre_BI_mean = 10 * np.log10(np.mean(10 ** (spectro_BI / 10), axis=1))

    # Segment between min_freq and max_freq
    spectre_BI_mean_segment = spectre_BI_mean[min_freq_bin:max_freq_bin]

    # Normalization: set the minimum value of the frequencies to zero.
    spectre_BI_mean_segment_normalized = spectre_BI_mean_segment - min(spectre_BI_mean_segment)

    # Compute the area under the spectre curve. Equivalent in the R code to: left_area <- sum(specA_left_segment_normalized * rows_width)
    area = np.sum(spectre_BI_mean_segment_normalized / (frequencies[1] - frequencies[0]))
    result = area

    return result


def compute_MFCC(soundMono, rate, timeWindow=1, numcep=12, min_freq=0, max_freq=8000, useMel=False):
    sample = soundMono

    if useMel:
        melSpec = librosa.feature.melspectrogram(y=sample, sr=rate, n_mels=numcep, fmin=min_freq, fmax=max_freq)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(melSpec), sr=rate, n_mfcc=numcep)
    else:
        mfccs = librosa.feature.mfcc(y=sample, sr=rate, n_mfcc=numcep)

    result = np.mean(mfccs, axis=1)

    return result


def compute_NDSI(soundMono, rate, timeWindow, fft_w=512, anthrophony=[1000, 2000], biophony=[2000, 11000]):
    """
    Compute Normalized Difference Sound Index from an audio signal.
    This function compute an estimate power spectral density using Welch's method.
    Reference: Kasten, Eric P., Stuart H. Gage, Jordan Fox, and Wooyeong Joo. 2012. The Remote Environ- mental Assessment Laboratory's Acoustic Library: An Archive for Studying Soundscape Ecology. Ecological Informatics 12: 50-67.
    windowLength: the length of the window for the Welch's method.
    anthrophony: list of two values containing the minimum and maximum frequencies (in Hertz) for antrophony.
    biophony: list of two values containing the minimum and maximum frequencies (in Hertz) for biophony.
    Inspired by the seewave R package, the soundecology R package and the original matlab code from the authors.
    """
    sample = soundMono

    frequencies, pxx = signal.welch(
        sample,  # pcm2float(sample,dtype='float64'),
        fs=rate,
        window='hamming',
        nperseg=fft_w,
        noverlap=fft_w / 2,
        nfft=fft_w,
        detrend='constant',
        return_onesided=True,
        scaling='density',
        axis=-1)  # Estimate power spectral density using Welch's method
    avgpow = pxx * frequencies[
        1]  # use a rectangle approximation of the integral of the signal's power spectral density (PSD)
    # avgpow = avgpow / np.linalg.norm(avgpow, ord=2) # Normalization (doesn't change the NDSI values. Slightly differ from the matlab code).

    min_anthro_bin = np.argmin(
        [abs(e - anthrophony[0]) for e in frequencies])  # min freq of anthrophony in samples (or bin) (closest bin)
    max_anthro_bin = np.argmin(
        [abs(e - anthrophony[1]) for e in frequencies])  # max freq of anthrophony in samples (or bin)
    min_bio_bin = np.argmin([abs(e - biophony[0]) for e in frequencies])  # min freq of biophony in samples (or bin)
    max_bio_bin = np.argmin([abs(e - biophony[1]) for e in frequencies])  # max freq of biophony in samples (or bin)

    anthro = np.sum(avgpow[min_anthro_bin:max_anthro_bin])
    bio = np.sum(avgpow[min_bio_bin:max_bio_bin])

    ndsi = (bio - anthro) / (bio + anthro)

    return ndsi, anthro, bio


def compute_RMS_Energy(soundMono, rate, timeWindow):
    """
    Compute the RMS short time energy.
    file: an instance of the AudioFile class.
    windowLength: size of the sliding window (samples)
    windowHop: size of the lag window (samples)
    integer: if set as True, the Temporal Entropy will be compute on the Integer values of the signal. If not, the signal will be set between -1 and 1.
    return: a list of values (rms energy for each window)
    """

    sample = soundMono

    result = librosa.feature.rmse(S=sample)

    return result


def compute_TH(soundMono, rate, timeWindow):
    """
    Compute Temporal Entropy of Shannon from an audio signal.
    file: an instance of the AudioFile class.
    integer: if set as True, the Temporal Entropy will be compute on the Integer values of the signal. If not, the signal will be set between -1 and 1.
    Ported from the seewave R package.
    """

    sample = soundMono

    env = abs(signal.hilbert(sample))  # Modulo of the Hilbert Envelope

    env = env / np.sum(env)  # Normalization
    N = len(env)
    result = -sum([y * np.log2(y) for y in env]) / np.log2(N)

    return result


def compute_ZCR(soundMono, rate, timeWindow):
    """
    Compute the Zero Crossing Rate of an audio signal.
    file: an instance of the AudioFile class.
    windowLength: size of the sliding window (samples)
    windowHop: size of the lag window (samples)
    return: a list of values (number of zero crossing for each window)
    """

    sample = soundMono

    result = np.mean(librosa.feature.zero_crossing_rate(sample))

    return result


def compute_SpecProp_Mean(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    # #FFT's means
    mn_ffts = np.mean(spectro, axis=1)

    # Mean
    mn_spec = np.mean(mn_ffts, axis=0)

    return mn_spec


def compute_SpecProp_SD(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    # #FFT's means'
    mn_ffts = np.mean(spectro, axis=1)

    # Mean
    std_spec = np.std(mn_ffts)

    return std_spec


def compute_SpecProp_SEM(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    mn_ffts = np.mean(spectro, axis=1)

    std_spec = np.std(mn_ffts)

    # Sem: standard error of the mean. *
    sqt_L = np.sqrt(np.shape(spectro)[0])
    sem_spec = std_spec / sqt_L

    return sem_spec


def compute_SpecProp_Median(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    mn_ffts = np.mean(spectro, axis=1)

    # Median
    mdn_spec = np.median(mn_ffts, axis=0)

    return mdn_spec


def compute_SpecProp_Mode(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    mn_ffts = np.mean(spectro, axis=1)

    mod_spec = stats.mode(mn_ffts, axis=0)[0][0]

    return mod_spec


def compute_SpecProp_Quartiles(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    mn_ffts = np.mean(spectro, axis=1)

    q25_spec = np.percentile(mn_ffts, 25, axis=0)
    q50_spec = np.percentile(mn_ffts, 50, axis=0)
    q75_spec = np.percentile(mn_ffts, 75, axis=0)
    IQR_spec = q75_spec - q25_spec

    return q25_spec, q50_spec, q75_spec, IQR_spec


def compute_SpecProp_Skewness(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50, power=True):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    mn_ffts = np.mean(spectro, axis=1)

    skw_spec = np.mean([stats.skew(abs(mn_ffts))])

    return skw_spec


def compute_SpecProp_Kurtosis(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50, power=True):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    mn_ffts = np.mean(spectro, axis=1)

    kts_spec = np.mean(stats.kurtosis(mn_ffts))

    return kts_spec


def compute_SpecProp_Entropy(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50, power=True):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    mn_ffts = np.mean(spectro, axis=1)

    etp_spec = stats.entropy(mn_ffts)

    return etp_spec


def compute_SpecProp_Variance(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50, power=True):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    mn_ffts = np.mean(spectro, axis=1)

    result = np.var(abs(mn_ffts))

    return result


def compute_SpecProp_Chroma_STFT(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50, power=True):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    result = np.mean(np.abs(librosa.feature.chroma_stft(sr=rate, S=spectro)))

    return result


def compute_SpecProp_Chroma_CQT(soundMono, rate, timeWindow):
    sample = np.array(soundMono).astype(np.float)

    result = np.mean(np.abs(librosa.feature.chroma_cqt(y=sample, sr=rate)))

    return result


def compute_SpecProp_Centroid(soundMono, rate, timeWindow, fft_w=512):
    sample = np.array(soundMono).astype(np.float)

    result = np.mean(librosa.feature.spectral_centroid(y=sample, sr=rate, n_fft=fft_w))

    return result


def compute_SpecProp_Bandwidth(soundMono, rate, timeWindow, fft_w=512):
    sample = np.array(soundMono).astype(np.float)

    result = np.mean(librosa.feature.spectral_bandwidth(y=sample, sr=rate, n_fft=fft_w))

    return result


def compute_SpecProp_Contrast(soundMono, rate, timeWindow, fft_w=512):
    sample = np.array(soundMono).astype(np.float)

    result = np.mean(librosa.feature.spectral_contrast(sr=rate, y=np.abs(sample), n_fft=fft_w))

    return result


def compute_SpecProp_Rolloff(soundMono, rate, timeWindow, fft_w=512):
    sample = np.array(soundMono).astype(np.float)

    result = np.mean(librosa.feature.spectral_rolloff(y=sample, sr=rate))

    return result


def compute_SpecProp_PolyFeatures(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50, power=True):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    result_zero = np.mean(librosa.feature.poly_features(S=np.abs(spectro), order=0, n_fft=fft_w))

    # Poly_features_linear
    result_linear = np.mean(librosa.feature.poly_features(S=np.abs(spectro), order=1, n_fft=fft_w))

    # Poly_features_quadratic
    result_quadratic = np.mean(librosa.feature.poly_features(S=np.abs(spectro), order=2, n_fft=fft_w))

    return result_zero, result_linear, result_quadratic


def compute_SpecSpread(soundMono, sample, rate, timeWindow, fft_w=512, db_threshold=50):
    """
    Compute the spectral spread (basically a variance of the spectrum around the spectral centroid)
    """

    centroid = np.mean(librosa.feature.spectral_centroid(y=soundMono, sr=rate))

    spectro = sample

    binNumber = 0
    numerator = 0
    denominator = 0

    for bin in spectro:
        # Compute center frequency
        f = (rate / 2.0) / len(spectro)
        f = f * binNumber

        numerator = numerator + (((f - centroid) ** 2) * abs(bin))
        denominator = denominator + abs(bin)

        binNumber = binNumber + 1

    result = np.sqrt(np.mean(numerator * 1.0) / np.mean(denominator))

    return result


def compute_SPL(sample, rate, timeWindow, min_freq, max_freq, fft_w=512, db_threshold=50):
    # Max and Min frequency
    bandSize = (rate / 2) / (fft_w / 2)
    min_freqi = min_freq / bandSize
    max_freqi = max_freq / bandSize
    if max_freqi > (fft_w / 2):
        max_freqi = fft_w / 2

    spectro = sample[min_freqi:max_freqi, :]

    result = np.mean(spectro)

    return result
