import unittest
import numpy as np
import pandas as pd
import xarray
import matplotlib.pyplot as plt
import scipy
from tests import with_plots

import pypam.utils as utils

# Create artificial data of 1 second
fs = 512000
test_freqs = [400, fs / 4]
samples = fs
noise_amp = 100
signal_amp = 100
data = pd.read_csv('tests/test_data/signal_data.csv', header=None)[0].values
t = np.linspace(0, 1 - 1 / fs, samples)
phase = 2 * np.pi * t
for test_freq in test_freqs:
    data = data + signal_amp * np.sin(test_freq * phase)

# Set the nfft to 1 second
nfft = fs


class TestMillidecades(unittest.TestCase):
    def test_get_millidecade_bands(self):
        bands_limits, bands_c = utils.get_hybrid_millidecade_limits(band=[0, fs/2], nfft=nfft)
        mdec_bands_test = pd.read_csv('tests/test_data/mdec_bands_test.csv', header=None)
        assert ((mdec_bands_test.iloc[:, 0] - bands_limits[:-1]) > 5e-5).sum() == 0
        assert ((mdec_bands_test.iloc[:, 2] - bands_limits[1:]) > 5e-5).sum() == 0
        assert ((mdec_bands_test.iloc[:, 1] - bands_c) > 5e-5).sum() == 0

    def test_psd_to_millidecades(self):
        bands_limits, bands_c = utils.get_hybrid_millidecade_limits(band=[0, fs/2], nfft=nfft)

        # Compute the spectrum manually
        ny_freq = int(int(nfft / 2))
        c_spec = np.fft.fft(data) / fs
        spectra_ds = abs(c_spec * c_spec)
        spectra = 2 * spectra_ds[:ny_freq + 1]
        spectra[0] = spectra_ds[0]
        spectra[ny_freq] = spectra_ds[ny_freq]

        fbands = scipy.fft.rfftfreq(nfft, 1/fs)

        # Load the spectrum used for MANTA
        spectra_manta = pd.read_csv('tests/test_data/spectra.csv', header=None)

        # Check if they are the same
        assert (abs(spectra_manta[0].values - spectra) > 1e-5).sum() == 0

        # Convert the spectra to a datarray
        psd_da = xarray.DataArray([spectra], coords={'id': [0], 'frequency': fbands}, dims=['id', 'frequency'])

        milli_psd = utils.psd_ds_to_bands(psd_da, bands_limits, bands_c, fft_bin_width=fs/nfft, db=False)

        # Read MANTA's output
        mdec_power_test = pd.read_csv('tests/test_data/mdec_power_test.csv')

        if with_plots():
            # Plot the two outputs for comparison
            fig, ax = plt.subplots()
            ax.plot(milli_psd.frequency_bins, mdec_power_test['sum'], label='MANTA')
            plt.legend()
            plt.show()

            fig, ax = plt.subplots()
            milli_psd.plot(ax=ax, label='pypam')
            plt.legend()
            plt.show()

        # Check if the results are the same
        assert ((mdec_power_test['sum'] - milli_psd.sel(id=0).values).abs() > 1e-5).sum() == 0


if __name__ == '__main__':
    unittest.main()
