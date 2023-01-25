import numpy as np
import xarray
import matplotlib.pyplot as plt

import pypam.utils as utils
import pypam.signal as sig

N_CHUNKS = 5

# Create artificial data of 1 second
fs = 512000
test_freqs = [400, fs / 4]
samples = fs * N_CHUNKS
noise_amp = 100
signal_amp = 100
data = np.random.random(samples)
t = np.linspace(0, 1 - 1 / fs, samples)
phase = 2 * np.pi * t
for test_freq in test_freqs:
    data = data + signal_amp * np.sin(test_freq * phase)
# Set the nfft to 1 second
nfft = fs

# Loop through all your chunks
list_spectras = []
for i in np.arange(N_CHUNKS):
    s = sig.Signal(data, fs=fs)
    s.set_band(None)
    fbands, spectra, _ = s.spectrum(scaling='spectrum', nfft=fs, db=False, overlap=0, force_calc=True)
    list_spectras.append(spectra)

# Convert the spectra to a datarray
psd_da = xarray.DataArray(list_spectras, coords={'id': np.arange(N_CHUNKS), 'frequency': fbands}, dims=['id', 'frequency'])

# Get the millidecade bands
bands_limits, bands_c = utils.get_hybrid_millidecade_limits(band=[0, fs/2], nfft=nfft)
milli_psd = utils.psd_ds_to_bands(psd_da, bands_limits, bands_c, fft_bin_width=fs / nfft, db=False)

milli_psd.mean('id').plot()
plt.show()
