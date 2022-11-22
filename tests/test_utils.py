import unittest
import numpy as np
import pandas as pd
import xarray
import matplotlib.pyplot as plt

import pypam.utils as utils
import pypam.signal as sig

# Create artificial data of 1 second
fs = 512000
test_freqs = [400, fs / 4]
samples = fs
noise_amp = 100
signal_amp = 100
data = pd.read_csv('./test_data/signal_data.csv', header=None)[0].values
t = np.linspace(0, 1 - 1 / fs, samples)
phase = 2 * np.pi * t
for test_freq in test_freqs:
    data = data + signal_amp * np.sin(test_freq * phase)

# Set the nfft to 1 second
nfft = fs


class TestMillidecades(unittest.TestCase):
    def test_get_millidecade_bands(self):
        bands_limits, bands_c = utils.get_hybrid_millidecade_limits(band=[0, fs/2], nfft=nfft)
        mdec_bands_test = pd.read_csv('./test_data/mdec_bands_test.csv', header=None)
        assert ((mdec_bands_test.iloc[:, 0] - bands_limits[:-1]) > 5e-5).sum() == 0
        assert ((mdec_bands_test.iloc[:, 2] - bands_limits[1:]) > 5e-5).sum() == 0
        assert ((mdec_bands_test.iloc[:, 1] - bands_c) > 5e-5).sum() == 0

    def test_psd_to_millidecades(self):
        bands_limits, bands_c = utils.get_hybrid_millidecade_limits(band=[0, fs/2], nfft=nfft)
        s = sig.Signal(data, fs=fs)
        s.set_band(None)
        fbands, spectra, _ = s.spectrum(scaling='spectrum', nfft=fs, db=False, overlap=0, force_calc=True)
        psd_da = xarray.DataArray([spectra], coords={'id': [0], 'frequency': fbands}, dims=['id', 'frequency'])

        psd_ds = xarray.Dataset({'band_spectrum': psd_da})
        milli_psd = utils.psd_ds_to_bands(psd_ds, bands_limits, bands_c, fft_bin_width=fs/nfft)

        # Read MANTA's output
        mdec_power_test = pd.read_csv('./test_data/mdec_power_test.csv')

        # Plot the two outputs for comparison
        fig, ax = plt.subplots()
        ax.plot(milli_psd.frequency_bins, mdec_power_test['sum'], label='MANTA')
        plt.legend()
        plt.show()

        fig, ax = plt.subplots()
        milli_psd['band_spectrum'].plot(ax=ax, label='pypam')
        plt.legend()
        plt.show()
        assert ((mdec_power_test['sum'] - milli_psd['band_spectrum'].sel(id=0).values) > 1e-5).sum() == 0


if __name__ == '__main__':
    unittest.main()
