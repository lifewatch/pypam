import unittest

import xarray

import pypam.plots
import pypam.signal as sig
import pypam.nmf as nmf
import numpy as np
from tests import skip_unless_with_plots, with_plots
import matplotlib.pyplot as plt


class TestSignal(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = xarray.open_dataset('tests/test_data/test_day.nc')
        self.ds = self.ds.rename({'millidecade_bands': 'band_density', 'frequency_bins': 'frequency'})

    @skip_unless_with_plots()
    def test_plot_spd(self):
        pypam.plots.plot_spd(self.ds)

    @skip_unless_with_plots()
    def test_plot_spectrograms(self):
        pypam.plots.plot_spectrograms(self.ds)

    @skip_unless_with_plots()
    def test_plot_spectrum_mean(self):
        pypam.plots.plot_spectrum_mean(self.ds)

    @skip_unless_with_plots()
    def test_plot_hmb_ltsa(self):
        pypam.plots.plot_hmb_ltsa(self.ds)

    @skip_unless_with_plots()
    def test_summary_plot(self):
        # Only necessary while compute_spd not updated
        pctlev = [1, 10, 25, 50, 75, 90, 99]
        pypam.plots.plot_summary_dataset(ds=self.ds, percentiles=pctlev,
                                         min_val=40, max_val=130, location=[112.186, 36.713])
        pypam.plots.plot_summary_dataset(ds=self.ds, percentiles=pctlev,
                                         min_val=40, max_val=130, location=None)
