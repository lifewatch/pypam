import unittest
import pypam
import numpy as np
from tests import skip_unless_with_plots, with_plots
import matplotlib.pyplot as plt
import pathlib
import pyhydrophone as pyhy

# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073

soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

class TestSignal(unittest.TestCase):
    def setUp(self) -> None:
        self.file = 'tests/test_data/67416073.210610033655.wav'
        self.start_sample = 10 * 8000
        self.end_sample = 70 * 8000

    @skip_unless_with_plots()
    def test_spectrogram_plot(self):
        chunk = pypam.acoustic_chunk.AcuChunk(
            sfile_start=self.file,
            sfile_end=self.file,
            start_frame=self.start_sample, end_frame=self.end_sample, hydrophone=soundtrap,
            p_ref=1, chunk_id=0, chunk_file_id=0, time_bin='210610033655')

        chunk.plot_spectrogram()
        plt.show()

