"""
Module : acoustic_file.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Institute voor de Zee)
"""

import numpy as np
import numba as nb
import scipy.signal as sig


@nb.jit
def fill_or_crop(signal, n_samples):
    """ 
    Crop the signal to the number specified or fill it with Nan values in case it is too short 

    Parameters
    ----------
    n_samples : int
        Number of desired samples
    signal : numpy array 
        Signal to modify
    """
    if signal.size >= n_samples: 
        return signal[0:n_samples]
    else:
        nan_array = np.full((n_samples,), np.nan)
        nan_array[0:signal.size] = signal
        return nan_array
    
    
def downsample(signal, old_fs, new_fs):
    """
    Reduce the sampling frequency

    Parameters
    ----------
    signal : numpy array
        Original signal
    old_fs : int
        Original sampling frequency
    new_fs : int
        New sampling frequency
    """
    if new_fs > old_fs:
        raise Exception('This is upsampling!')
    ratio = (old_fs / new_fs)
    if (ratio % 2) != 0:
        new_lenght = int(signal.size * (new_fs / old_fs))
        new_signal = sig.resample(signal, new_lenght)
    else:
        new_signal = sig.resample_poly(signal, up=1, down=int(ratio))
    return new_signal


def filter_and_downsample(signal, band, fs):
    """
    Filter and downsample the signal
    """
    if band is not None:
        # Filter the signal
        sosfilt = sig.butter(N=4, btype='bandpass', Wn=band, analog=False, output='sos', fs=fs)
        signal = sig.sosfilt(sosfilt, signal)

        # Downsample if frequency analysis to get better resolution
        if band[1] < fs / 2: 
            new_fs = band[1] * 2
            signal = downsample(signal, fs, new_fs)
        else:
            new_fs = fs
    else:
        new_fs = fs
    
    return new_fs, signal


@nb.jit
def rms(signal, db=True):
    """
    Calculation of root mean squared value (rms) of the signal in uPa

    Parameters
    ----------
    signal : numpy array
        Signal to be computed
    db : bool
        If set to True the result will be given in db, otherwise in uPa
    """
    rms_val = np.sqrt((signal**2).mean())
    # Convert it to db if applicatble
    if db:
        rms_val = 10 * np.log10(rms_val**2)

    return rms_val


def aci(signal, new_fs, nfft, window):
    """
    Calculation of root mean squared value (rms) of the signal in uPa for each bin
    Returns Dataframe with 'datetime' as index and 'rms' value as a column

    Parameters
    ----------
    signal : numpy array
        Signal to be computed
    new_fs : int
        New sampling frequency
    nfft : int
        Number of fft
    window : int
        window to use to calculate the spectrogram
    """
    _, _, sxx = sig.spectrogram(signal, fs=new_fs, nfft=nfft, window=window, scaling='spectrum')
    aci_val = calculate_aci(sxx)
        
    return aci_val


@nb.jit
def dynamic_range(signal, db=True):
    """
    Compute the dynamic range of each bin
    Returns a dataframe with datetime as index and dr as column

    Parameters
    ----------
    signal : numpy array
        Signal to be computed
    db : bool
        If set to True the result will be given in db, otherwise in uPa
    """

    dr = signal.max() - signal.min()
    # Convert it to db if applicable
    if db:
        dr = 10 * np.log10(dr**2)
        
    return dr


# def spectrogram(binsize=None, nfft=512, scaling='density', db=True):
#     """
#     Return the spectrogram of the signal (entire file)
    
#     Parameters
#     ----------
#     binsize : float, in sec
#         Time window considered. If set to None, only one value is returned
#     db : bool
#         If set to True the result will be given in db, otherwise in uPa^2
#     nfft : int
#         Lenght of the fft window in samples. Power of 2. 
#     scaling : string
#         Can be set to 'spectrum' or 'density' depending on the desired output
    
#     Returns
#     -------
    
#     """
#     freq, t, Sxx = sig.spectrogram(signal, fs=new_fs, nfft=nfft, window=window, scaling=scaling)
#     if db:
#         Sxx = 10 * np.log10(Sxx)

#     return freq, t, Sxx


# def _spectrum(scaling='density', binsize=None, bands='all', nfft=512, db=True, percentiles=[]):
#     """
#     Return the spectrum : frequency distribution of all the file (periodogram)
#     Returns Dataframe with 'datetime' as index and a colum for each frequency and each percentile,
#     and a frequency array

#     Parameters
#     ----------
#     scaling : string
#         Can be set to 'spectrum' or 'density' depending on the desired output       
#     binsize : float, in sec
#         Time window considered. If set to None, only one value is returned
#     bands : string
#         Can be set to 'octaves', 'third_octaves' or 'all'. 
#     nfft : int
#         Lenght of the fft window in samples. Power of 2. 
#     db : bool
#         If set to True the result will be given in db, otherwise in uPa^2
#     percentiles : list
#         List of all the percentiles that have to be returned. If set to empty list, no percentiles is returned
#     """

#     fbands, spectra = sig.periodogram(signal, fs=new_fs, window=window, nfft=nfft, scaling=scaling)
    
#     return fbands, spectra


# def correlation(signal, fs_signal):
#     """
#     Compute the correlation with the signal
#
#     Parameters
#     ----------
#     signal : numpy array
#         Signal to be correlated with
#     fs_signal : int
#         Sampling frequency of the signal. It will be down/up sampled in case it does not match with the file
#         samplig frequency
#     """
#     return 0


def oct3dsgn(fc, fs, n=3):
    """
    Design of a 1/3-octave band filter with center frequency fc for sampling frequency fs.
    Default value for N is 3. For meaningful results, fc should be in range fs/200 < fc < fs/5.

    Parameters
    ----------
    fc : float
      Center frequency, in Hz
    fs : float
      Sample frequency at least 2.3x the center frequency of the highest 1/3 octave band, in Hz
    n : int
      Order specification of the filters, N = 2 gives 4th order, N = 3 gives 6th order
      Higher N can give rise to numerical instability problems, so only 2 or 3 should be used
    """
    if fc > 0.88*fs/2:
        raise Exception('Design not possible - check frequencies')
    # design Butterworth 2N-th-order 1/3-octave band filter
    f1 = fc/(2**(1/6))
    f2 = fc*(2**(1/6))
    qr = fc/(f2-f1)
    qd = (np.pi/2/n)/(np.sin(np.pi/2/n))*qr
    alpha = (1 + np.sqrt(1+4*qd**2))/2/qd
    w1 = fc/(fs/2)/alpha
    w2 = fc/(fs/2)*alpha
    sos = sig.butter(n, [w1, w2], output='sos')

    return sos


def oct3bankdsgn(fs, bands, n):
    """
    Construction of a 1/3 octave band filterbank.
    
    Parameters
    ----------
    fs : float
      Sample frequency, at least 2.3x the center frequency of the highest 1/3 octave band, in Hz
    bands : numpy array
      row vector with the desired band numbers (0 = band with center frequency of 1 kHz)
      e.g. [-16:11] gives all bands with center frequency between 25 Hz and 12.5 kHz
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
    fc = 1000*((2**(1/3))**bands)     # exact center frequencies
    fclimit = 1/200                     # limit for center frequency compared to sample frequency
    # calculate downsampling factors
    d = np.ones(len(fc))
    for i in np.arange(len(fc)):
        while fc(i) < (fclimit*(fs/2**(d[i]-1))):
            d[i] += 1
    # calculate new sample frequencies
    fsnew = fs/(2**(d-1))
    # construct filterbank
    filterbank = []
    for i in np.arange(len(fc)):
        # construct filter coefficients
        sos = oct3dsgn(fc(i), fsnew(i), n)
        filterbank.append(sos)

    return filterbank, fsnew


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
    spd = np.zeros((sxx.shape[0], bin_edges.size-1), dtype=np.float64)
    p = np.zeros((sxx.shape[0], percentiles.size), dtype=np.float64)
    for i in nb.prange(sxx.shape[0]):
        spd[i, :] = np.histogram(sxx[i, :], bin_edges)[0] / ((bin_edges.size - 1) * h)
        cumsum = np.cumsum(spd[i, :])
        for j in nb.prange(percentiles.size):
            p[i, j] = bin_edges[np.argmax(cumsum > percentiles[j]*cumsum[-1])]

    return spd, p


@nb.jit
def calculate_aci(sxx):
    """
    Return the aci of the signal
    """
    aci_val = 0
    for j in np.arange(sxx.shape[1]):
        d = 0
        i = 0
        for k in np.arange(1, sxx.shape[0]):
            dk = np.abs(sxx[k][j] - sxx[k-1][j])
            d += dk
            i += sxx[k][j]
        aci_val += d/i
    
    return aci_val
