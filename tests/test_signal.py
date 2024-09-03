import unittest
import pypam.signal as sig
import pypam.nmf as nmf
import numpy as np
from tests import skip_unless_with_plots, with_plots
import matplotlib.pyplot as plt


# Create artificial data of 1 second
fs = 48000
test_freqs = [10000, 20000]
seconds_signal = 30
samples = fs * seconds_signal
noise_amp = 1
signal_amp = 100
data = np.random.random(samples)
t = np.linspace(0, seconds_signal, samples)
phase = 2 * np.pi * t
for test_freq in test_freqs:
    data = data + signal_amp * np.sin(test_freq * phase)

# Set the nfft to 1 second
nfft = fs


class TestSignal(unittest.TestCase):
    def setUp(self) -> None:
        self.data = data

    @skip_unless_with_plots()
    def test_spectrum(self):
        s = sig.Signal(self.data, fs=fs)
        s.set_band(None)
        fbands, spectra, _ = s.spectrum(scaling='spectrum', nfft=fs, db=False, overlap=0, force_calc=True)
        plt.plot(fbands, spectra)
        plt.show()

    def test_acoustic_indices(self):
        s = sig.Signal(self.data, fs=fs)
        aci = s.aci()
        assert aci >= 0
        bi = s.bi()
        assert bi >= 0
        sh = s.sh()
        assert np.logical_and(sh >= 0, sh <= 1)
        th = s.th()
        assert np.logical_and(th >= 0, th <= 1)
        ndsi = s.ndsi()
        assert np.logical_and(ndsi >= -1, ndsi <= 1)
        aei = s.aei()
        assert np.logical_and(aei >= 0, aei <= 1)
        adi = s.adi()
        assert adi >= 0
        zcr = s.zcr()
        assert np.logical_and(zcr >= 0, zcr <= 1)
        zcr_avg = s.zcr_avg()
        assert np.logical_and(zcr_avg >= 0, zcr_avg <= 1)


