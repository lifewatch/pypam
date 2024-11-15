import unittest
from pypam.acoustic_file import AcuFile
import pyhydrophone as pyhy
from tests import skip_unless_with_plots
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams.update(plt.rcParamsDefault)
# get relative path
dir = os.path.dirname(__file__)
# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
calibration_file = pathlib.Path(f"{dir}/test_data/calibration_data.xlsx")
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number,
                                     calibration_file=calibration_file, val='sensitivity', freq_col_id=1,
                                     val_col_id=29, start_data_id=6)


class TestAcuFile(unittest.TestCase):
    def setUp(self) -> None:
        self.acu_file = AcuFile(f'{dir}/test_data/67416073.210610033655.wav', soundtrap, 1)

    @skip_unless_with_plots()
    def test_plots(self):
        self.acu_file.plot_spectrum_per_chunk(scaling='density', db=True)
        self.acu_file.plot_spectrum_median(scaling='spectrum', db=True, binsize=10)

    def test_millidecade_bands(self):
        nfft = 8000
        self.acu_file.hybrid_millidecade_bands(nfft, fft_overlap=0.5, binsize=None, bin_overlap=0, db=True,
                                               method='spectrum', band=None)
        self.acu_file.hybrid_millidecade_bands(nfft, fft_overlap=0.5, binsize=None, bin_overlap=0, db=True,
                                               method='density', band=None)

    def test_update_freq_cal(self):
        ds_psd = self.acu_file.psd()
        ds_psd_updated = self.acu_file.update_freq_cal(ds=ds_psd, data_var='band_density')
        print(ds_psd['band_density'].values)
        print(ds_psd_updated['band_density'].values)

    @skip_unless_with_plots()
    def test_overlapping_bins(self):
        binsize, bin_overlap = 1, 0.2
        ds_rms_overlap = self.acu_file.rms(binsize, bin_overlap)
        ds_rms = self.acu_file.rms(binsize, 0)
        fig,ax = plt.subplots()
        ax.plot(ds_rms_overlap.datetime[0:100], ds_rms_overlap.rms[0:100], label='.50 overlap')
        ax.plot(ds_rms.datetime[0:100], ds_rms.rms[0:100], label='no overlap')
        ax.legend()
        fig.show()
        # compare output time step (dt) to expected time step
        step = np.mean(np.diff(ds_rms_overlap.start_sample))
        dt = step/self.acu_file.fs
        expected_dt = binsize*(1-bin_overlap)
        error = np.abs(dt-expected_dt)
        # error should be less than soundfile dt (if rounded)
        assert error<=(1/self.acu_file.fs)


