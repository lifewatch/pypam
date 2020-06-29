"""
Module: utils.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

import os
import operator
import numpy as np
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt



def oct3dsgn(fc, fs, N=3):
    """
    Design of a 1/3-octave band filter with center frequency fc for sampling frequency fs.
    Default value for N is 3. For meaningful results, fc should be in range fs/200 < fc < fs/5.
    """
    if (fc > 0.88*(fs/2)):
        raise Exception('Design not possible - check frequencies')
    # design Butterworth 2N-th-order 1/3-octave band filter
    f1 = fc/(2**(1/6))
    f2 = fc*(2**(1/6))
    Qr = fc/(f2-f1)
    Qd = (np.pi/2/N)/(np.sin(np.pi/2/N))*Qr
    alpha = (1 + np.sqrt(1+4*Qd**2))/2/Qd
    W1 = fc/(fs/2)/alpha
    W2 = fc/(fs/2)*alpha
    b, a = sig.butter(N, [W1, W2])

    return b, a


def oct3bankdsgn(fs, bands, N):
    """
    Construction of a 1/3 octave band filterbank.
    parameters:
    * fs: samplefrequency (Hz), at least 2.3x the center frequency of the highest 1/3 octave band
    * bands: row vector with the desired band numbers (0 = band with center frequency of 1 kHz)
      e.g. [-16:11] gives all bands with center frequency between 25 Hz and 12.5 kHz
    * N: order specification of the filters, N = 2 gives 4th order, N = 3 gives 6th order
      Higher N can give rise to numerical instability problems, so only 2 or 3 should be used
    output:
    * b, a: matrices with filter coefficients, one row per filter.
    * d: column vector with downsampling factors for each filter 1 means no downsampling, 2 means
      downsampling with factor 2, 3 means downsampling with factor 4 and so on.
    * fsnew: column vector with new sample frequencies.
    """
    fc = (1000)*((2**(1/3))**bands)     # exact center frequencies
    fclimit = 1/200                     # limit for center frequency compared to sample frequency
    # calculate downsampling factors
    d = np.ones(len(fc))
    for i in np.arange(len(fc)):
        while fc(i) < (fclimit*(fs/2**(d(i)-1))):
            d[i] += 1
    # calculate new sample frequencies
    fsnew = fs/(2**(d-1))
    # construct filterbank
    a = []
    b = []
    for i in np.arange(len(fc)):
        # construct filter coefficients
        tb, ta = oct3dsgn(fc(i), fsnew(i), N)
        a = [a, ta]
        b = [b, tb]

    return b, a, d, fsnew

