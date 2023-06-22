import unittest
from pypam.acoustic_file import AcuFile
import pyhydrophone as pyhy
from tests import skip_unless_with_plots


# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)


class TestAcuFile(unittest.TestCase):
    def setUp(self) -> None:
        self.acu_file = AcuFile('tests/test_data/67416073.210610033655.wav', soundtrap, 1)

    @skip_unless_with_plots()
    def test_plots(self):
        self.acu_file.plot_spectrum_per_chunk(scaling='density', db=True)
        self.acu_file.plot_spectrum_mean(scaling='spectrum', db=True)

    def test_millidecade_bands(self):
        nfft = 8000
        self.acu_file.hybrid_millidecade_bands(nfft, fft_overlap=0.5, binsize=None, bin_overlap=0, db=True,
                                               method='spectrum', band=None)
        self.acu_file.hybrid_millidecade_bands(nfft, fft_overlap=0.5, binsize=None, bin_overlap=0, db=True,
                                               method='density', band=None)

    def test_nmf(self):
        self.acu_file.source_separation(window_time=1.0, n_sources=15,
                                        binsize=None, save_path=None, verbose=False, band=None)
