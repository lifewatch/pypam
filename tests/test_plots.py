import unittest

import xarray

import pathlib
import pypam.plots
import pypam.utils
from pypam.acoustic_file import AcuFile
from pypam.acoustic_survey import ASA
import pyhydrophone as pyhy
from tests import skip_unless_with_plots, with_plots
import matplotlib.pyplot as plt

plt.rcParams.update(plt.rcParamsDefault)

# Data information
folder_path = pathlib.Path('tests/test_data')

# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

# SURVEY PARAMETERS
nfft = 8000
binsize = 60.0
dc_subtract = True
include_dirs = True
zipped_files = False


class TestPlots(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = xarray.open_dataset('tests/test_data/test_day.nc')
        self.ds = self.ds.rename({'millidecade_bands': 'band_density', 'frequency_bins': 'frequency'})
        self.acu_file = AcuFile('tests/test_data/67416073.210610033655.wav', soundtrap, 1)
        self.asa = ASA(hydrophone=soundtrap, folder_path=folder_path, binsize=binsize, nfft=nfft, timezone='UTC',
                       include_dirs=include_dirs, zipped=zipped_files, dc_subtract=dc_subtract)

    @skip_unless_with_plots()
    def test_plot_spd(self):
        ds_spd = pypam.utils.compute_spd(self.ds)
        pypam.plots.plot_spd(spd=ds_spd)

    @skip_unless_with_plots()
    def test_plot_spectrogram_per_chunk(self):
        ds_spectrogram = self.acu_file.spectrogram()
        pypam.plots.plot_spectrograms_per_chunk(ds_spectrogram=ds_spectrogram)

    @skip_unless_with_plots()
    def plot_spectrum_per_chunk(self):
        psd = self.acu_file.psd()
        pypam.plots.plot_spectrum(ds=psd, col_name='band_density')

    @skip_unless_with_plots()
    def test_plot_spectrum_mean(self):
        psd = self.asa.evolution_freq_dom('psd')
        pypam.plots.plot_spectrum_mean(ds=psd, col_name='band_density', output_name='PSD', show=True)

    @skip_unless_with_plots()
    def test_plot_hmb_ltsa(self):
        ds = self.ds.copy()
        ds = ds.rename({'band_density': 'millidecade_bands', 'frequency': 'frequency_bins'})
        ds = ds.swap_dims({'id': 'datetime'})
        da = ds['millidecade_bands']
        pypam.plots.plot_hmb_ltsa(da)

    @skip_unless_with_plots()
    def test_summary_plot(self):
        # Only necessary while compute_spd not updated
        pctlev = [1, 10, 25, 50, 75, 90, 99]
        pypam.plots.plot_summary_dataset(ds=self.ds, percentiles=pctlev,
                                         min_val=40, max_val=130, location=[112.186, 36.713])
        pypam.plots.plot_summary_dataset(ds=self.ds, percentiles=pctlev,
                                         min_val=40, max_val=130, location=None)
