import unittest
from pypam.acoustic_file import AcuFile
from pypam.signal import Signal
import pypam.acoustic_indices
import pyhydrophone as pyhy
import numpy as np


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
        self.frequencies, _, self.spectrogram = self.signal.spectrogram(db=False)

    def test_compute_aci(self):
        aci = pypam.acoustic_indices.compute_aci(sxx=self.spectrogram)
        assert aci >= 0

    def test_compute_bi(self):
        bi = pypam.acoustic_indices.compute_bi(sxx=self.spectrogram, frequencies=self.frequencies)
        assert bi >= 0

    def test_compute_sh(self):
        sh = pypam.acoustic_indices.compute_sh(sxx=self.spectrogram)
        assert np.logical_and(sh >= 0, sh <= 1)

    def test_compute_th(self):
        th = pypam.acoustic_indices.compute_th(s=self.signal.signal)
        assert np.logical_and(th >= 0, th <= 1)

    def test_compute_ndsi(self):
        ndsi = pypam.acoustic_indices.compute_ndsi(sxx=self.spectrogram, frequencies=self.frequencies)
        assert np.logical_and(ndsi >= -1, ndsi <= 1)

    def test_compute_aei(self):
        aei = pypam.acoustic_indices.compute_aei(sxx=self.spectrogram, frequencies=self.frequencies)
        assert np.logical_and(aei >= 0, aei <= 1)

    def test_compute_adi(self):
        adi = pypam.acoustic_indices.compute_adi(sxx=self.spectrogram, frequencies=self.frequencies)
        assert adi >= 0

    def test_compute_zcr(self):
        zcr = pypam.acoustic_indices.compute_zcr(s=self.signal.signal, fs=self.signal.fs)
        assert np.logical_and(zcr >= 0, zcr <= 1)

    def test_compute_zcr_avg(self):
        zcr_avg = pypam.acoustic_indices.compute_zcr_avg(s=self.signal.signal, fs=self.signal.fs)
        assert np.logical_and(zcr_avg >= 0, zcr_avg <= 1)
