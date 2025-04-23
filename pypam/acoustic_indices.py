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

import maad.features
import numpy as np


def compute_aci(sxx: np.ndarray) -> float:
    """
    Return the aci of the signal

    Args:
        sxx: Spectrogram of the signal in linear units

    Returns:
        ACI value
    """
    _, _, aci_val = maad.features.alpha_indices.acoustic_complexity_index(Sxx=sxx)
    return aci_val


def compute_bi(
    sxx: np.array, frequencies: np.array, min_freq: int = 2000, max_freq: int = 8000
) -> float:
    """
    Compute the Bioacoustic Index from the spectrogram of an audio signal.
    In this code, the Bioacoustic Index correspond to the area under the mean spectre (in db)
    minus the minimum frequency value of this mean spectre.
    Reference: Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic invasion resistance
    in Hawaii: bioacoustics, field surveys, and airborne remote sensing.
    Ecological Applications 17: 2137-2144.
    Ported from the soundecology R package.

    Args:
        sxx: The spectrogram of the audio signal in linear units
        frequencies: List of the frequencies of the spectrogram
        min_freq:  Minimum frequency (in Hertz)
        max_freq: Maximum frequency (in Hertz)

    Returns:
        Bioacoustic Index (BI) value
    """
    bi = maad.features.alpha_indices.bioacoustics_index(
        Sxx=sxx, fn=frequencies, flim=(min_freq, max_freq), R_compatible="soundecology"
    )
    return bi


def compute_sh(sxx: np.array) -> float:
    """
    Compute Spectral Entropy of Shannon from the spectrogram of an audio signal.
    Ported from the seewave R package.

    Args:
        sxx: The spectrogram of the audio signal in linear units

    Returns:
        Spectral Entropy (SH)
    """
    sh, _ = maad.features.alpha_indices.frequency_entropy(X=sxx)
    return sh


def compute_th(s: np.array) -> float:
    """
    Compute Temporal Entropy of Shannon from an audio signal.
    Ported from the seewave R package.

    Args:
    s: Signal
    """
    th = maad.features.alpha_indices.temporal_entropy(s=s)
    return th


def compute_ndsi(
    sxx: np.array,
    frequencies: np.array,
    anthrophony: tuple = (1000, 2000),
    biophony: tuple = (2000, 11000),
) -> float:
    """
    Compute Normalized Difference Sound Index from power spectrogram.
    Reference: Kasten, Eric P., Stuart H. Gage, Jordan Fox, and Wooyeong Joo. 2012.
    The Remote Environmental Assessment Laboratory's Acoustic Library: An Archive for Studying
    Soundscape Ecology.
    Ecological Informatics 12: 50-67.
    Inspired by the seewave R package, the soundecology R package and the original matlab code from the authors.

    Args:
        sxx: The spectrogram of the audio signal in linear units
        frequencies: List of the frequencies of the spectrogram
        anthrophony: Tuple of two int values containing the minimum and maximum frequencies (in Hertz) for antrophony.
        biophony: Tuple of two int values containing the minimum and maximum frequencies (in Hertz) for biophony.
    """
    ndsi, _, _, _ = maad.features.alpha_indices.soundscape_index(
        Sxx_power=sxx,
        fn=frequencies,
        flim_bioPh=biophony,
        flim_antroPh=anthrophony,
        R_compatible="soundecology",
    )
    return ndsi


def compute_aei(
    sxx: np.array,
    frequencies: list,
    max_freq: int = 10000,
    min_freq: int = 0,
    db_threshold: int or float = -50,
    freq_step: int = 1000,
):
    """
    Compute Acoustic Evenness Index of an audio signal.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011.
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    Ported from the soundecology R package.

    Args:
        sxx: Spectrogram of the audio signal in linear units
        frequencies: Frequencies list of the spectrogram (in Hz)
        max_freq: The maximum frequency to consider to compute AEI (in Hz)
        min_freq: The minimum frequency to consider to compute AEI (in Hz)
        db_threshold: The minimum db value to consider for the bins of the spectrogram
        freq_step: Size of frequency bands to compute AEI (in Hz)
    """
    aei = maad.features.alpha_indices.acoustic_eveness_index(
        Sxx=sxx,
        fn=frequencies,
        fmin=min_freq,
        fmax=max_freq,
        bin_step=freq_step,
        dB_threshold=db_threshold,
    )
    return aei


def compute_adi(
    sxx,
    frequencies: list,
    max_freq: int = 10000,
    min_freq: int = 0,
    db_threshold: int or float = -50,
    freq_step: int = 1000,
):
    """
    Compute Acoustic Diversity Index.
    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011.
    A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.
    Ported from the soundecology R package.

    Args:
        sxx: Spectrogram of the audio signal in linear units
        frequencies: Frequencies list of the spectrogram
        max_freq: The maximum frequency to consider to compute AEI (in Hz)
        min_freq: The minimum frequency to consider to compute AEI (in Hz)
        db_threshold: The minimum db value to consider for the bins of the spectrogram
        freq_step: Size of frequency bands to compute AEI (in Hz)
    """
    adi = maad.features.alpha_indices.acoustic_diversity_index(
        Sxx=sxx,
        fn=frequencies,
        fmin=min_freq,
        fmax=max_freq,
        bin_step=freq_step,
        dB_threshold=db_threshold,
        index="shannon",
    )
    return adi


def compute_zcr(s: np.array, fs: int):
    """
    Compute the Zero Crossing Rate of an audio signal.

    Args:
        s: Signal
        fs : Sampling frequency in Hz

    Returns:
        A list of values (number of zero crossing for each window)
    """
    zcr = maad.features.temporal.zero_crossing_rate(s=s, fs=fs) / fs
    return zcr


def compute_zcr_avg(
    s: np.array, fs: int, window_length: int = 512, window_hop: int = 256
):
    """
    Compute the Zero Crossing Rate of an audio signal.

    Args:
        s: Signal to process
        fs: Sampling frequency in Hz
        window_length: Size of the sliding window (samples)
        window_hop: Size of the lag window (samples)

    Returns:
        A list of values (number of zero crossing for each window)
    """
    times = np.arange(0, len(s) - window_length + window_hop)
    zcr_bins = np.zeros(times.size)
    for k, i in enumerate(times):
        x = s[i : i + window_length]
        zcr_bins[k] = compute_zcr(s=x, fs=fs)

    return np.mean(zcr_bins)
