import unittest
from pypam.acoustic_file import AcuFile
from pypam.signal import Signal
import pypam.acoustic_indices
import pyhydrophone as pyhy


# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)
acu_file = AcuFile(sfile='tests/test_data/67416073.210610033655.wav', hydrophone=soundtrap, p_ref=1)


class TestAcousticIndices(unittest.TestCase):
    def setUp(self) -> None:
        self.signal = Signal(signal=acu_file.signal(units='upa'), fs=acu_file.fs, channel=acu_file.channel)
        self.frequencies, _, self.spectrogram = self.signal.spectrogram()
        self.spectrogram = 10**(self.spectrogram/20)

    def test_compute_aci(self):
        pypam.acoustic_indices.compute_aci(sxx=self.spectrogram)

    def test_compute_bi(self):
        pypam.acoustic_indices.compute_bi(sxx=self.spectrogram, frequencies=self.frequencies)

    def test_compute_sh(self):
        pypam.acoustic_indices.compute_sh(sxx=self.spectrogram)

    def test_compute_th(self):
        pypam.acoustic_indices.compute_th(s=self.signal.signal)

    def test_compute_ndsi(self):
        pypam.acoustic_indices.compute_ndsi(sxx=self.spectrogram, frequencies=self.frequencies)

    def test_compute_aei(self):
        pypam.acoustic_indices.compute_aei(sxx=self.spectrogram, frequencies=self.frequencies)

    def test_compute_adi(self):
        pypam.acoustic_indices.compute_adi(sxx=self.spectrogram, frequencies=self.frequencies)

    def test_compute_zcr(self):
        pypam.acoustic_indices.compute_zcr(s=self.signal.signal, fs=self.signal.fs)

    def test_compute_zcr_avg(self):
        pypam.acoustic_indices.compute_zcr_avg(s=self.signal.signal, fs=self.signal.fs)
