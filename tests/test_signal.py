import unittest
import pypam.signal as sig
import pypam.nmf as nmf
import numpy as np
from tests import skip_unless_with_plots
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

    def test_source_separation(self):
        separator = nmf.NMF(window_time=0.1, rank=15)
        s = sig.Signal(self.data, fs=fs)
        s.set_band(None)
        separation_ds = separator(s, verbose=True)
        reconstructed_sources = separator.reconstruct_sources(separation_ds)
        separator.return_filtered_signal(s, reconstructed_sources['C_tf'])
