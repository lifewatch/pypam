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

import pypam.units as output_units

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
        self.ds['band_density'].attrs.update(output_units.get_units_attrs('spectrum_density', log=True, p_ref=1.0))
        self.ds = self.ds.where(self.ds.frequency > 10, drop=True)
        self.acu_file = AcuFile('tests/test_data/67416073.210610033655.wav', soundtrap, 1)
        self.asa = ASA(hydrophone=soundtrap, folder_path=folder_path, binsize=binsize, nfft=nfft, timezone='UTC',
                       include_dirs=include_dirs, zipped=zipped_files, dc_subtract=dc_subtract)

    @skip_unless_with_plots()
    def test_plot_spd(self):
        ds_spd = pypam.utils.compute_spd(self.ds, percentiles=[1, 10, 50, 90, 99])
        pypam.plots.plot_spd(spd=ds_spd, show=True)

    @skip_unless_with_plots()
    def test_plot_spectrogram_per_chunk(self):
        ds_spectrogram = self.acu_file.spectrogram()
        pypam.plots.plot_spectrogram_per_chunk(ds_spectrogram=ds_spectrogram, show=True)

    @skip_unless_with_plots()
    def test_plot_spectrum_per_chunk(self):
        psd = self.acu_file.psd()
        pypam.plots.plot_spectrum_per_chunk(ds=psd, data_var='band_density')

    @skip_unless_with_plots()
    def test_plot_spectrum_mean(self):
        psd = self.asa.evolution_freq_dom('psd')
        pypam.plots.plot_spectrum_mean(ds=psd, data_var='band_density', show=True)

    @skip_unless_with_plots()
    def test_plot_ltsa(self):
        pypam.plots.plot_ltsa(ds=self.ds, data_var='band_density', show=True)

    @skip_unless_with_plots()
    def test_summary_plot(self):
        # Only necessary while compute_spd not updated
        pctlev = [1, 10, 25, 50, 75, 90, 99]
        pypam.plots.plot_summary_dataset(ds=self.ds, percentiles=pctlev,
                                         min_val=40, max_val=130, location=[112.186, 36.713], show=True,
                                         save_path='tests/test_data/data_exploration/img/data_overview/'
                                                   'summary_plot_test1.png')
        pypam.plots.plot_summary_dataset(ds=self.ds, percentiles=pctlev,
                                         min_val=40, max_val=130, location=None, show=True,
                                         save_path='tests/test_data/data_exploration/img/data_overview/'
                                                   'summary_plot_test2.png')

    @skip_unless_with_plots()
    def test_plot_daily_patterns_from_ds(self):
        ds = self.ds.mean(dim='frequency', keep_attrs=True)
        pypam.plots.plot_daily_patterns_from_ds(ds=ds, data_var='band_density', show=True)

    @skip_unless_with_plots()
    def test_plot_rms_evolution(self):
        rms_evolution = self.asa.evolution('rms', db=True)
        pypam.plots.plot_rms_evolution(ds=rms_evolution, show=True)

    @skip_unless_with_plots()
    def test_plot_aggregation_evolution(self):
        self.ds = self.ds.swap_dims({'id': 'datetime'})
        ds_one_freq = pypam.utils.bin_aggregation(ds=self.ds, data_var='band_density', band=1000, freq='H')
        ds_band = pypam.utils.bin_aggregation(ds=self.ds, data_var='band_density', band=(100, 10000), freq='H')
        pypam.plots.plot_aggregation_evolution(ds=ds_one_freq, data_var='band_density', mode='violin', show=True)
        pypam.plots.plot_aggregation_evolution(ds=ds_band, data_var='band_density', mode='violin', show=True)

