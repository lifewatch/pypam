import unittest
from pypam.acoustic_file import AcuFile
import pyhydrophone as pyhy
from tests import skip_unless_with_plots
import pathlib
import matplotlib.pyplot as plt
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
