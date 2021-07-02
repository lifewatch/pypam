import unittest
import requests
import pathlib

from pypam.acoustic_survey import ASA
import pyhydrophone as pyhy


# Data information
data_path = pathlib.Path('./../data')
data_url = 'https://www.lifewatch.be/code/pypam-files/tests/'

# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 4096
binsize = 60.0
band_lf = [50, 500]
band_hf = [500, 2000]
band_list = [band_lf, band_hf]
features = ['rms', 'peak', 'sel', 'dynamic_range', 'aci', 'bi', 'sh', 'th', 'ndsi', 'aei', 'adi', 'zcr', 'zcr_avg',
            'bn_peaks']
third_octaves = True

include_dirs = False
zipped_files = False


class TestASA(unittest.TestCase):
    def setUp(self) -> None:
        if not any(data_path.iterdir()):
            requests.get(data_url)
        self.asa = ASA(hydrophone=soundtrap, folder_path='./../data', binsize=binsize, nfft=nfft, utc=True,
                       include_dirs=include_dirs, zipped=zipped_files)

    def test_features(self):
        self.asa.evolution_multiple(method_list=features, band_list=band_list)

    def test_third_oct(self):
        self.asa.evolution_freq_dom('third_octaves_levels', band=third_octaves, db=True)

    def test_detect_piling_events(self):
        min_separation = 1
        max_duration = 0.2
        threshold = 20
        dt = 2.0
        detection_band = [500, 1000]

        self.asa.detect_piling_events(max_duration=max_duration, min_separation=min_separation,
                                      threshold=threshold, dt=dt, verbose=True, band=detection_band, method='snr',
                                      save_path=None)

    def test_detect_ship_events(self):
        # just a smoke test to check if the function can run without errors
        self.asa.detect_ship_events(0.1, 0.5)

    def test_nmf(self):
        self.asa.source_separation(1.0, 15)


if __name__ == '__main__':
    unittest.main()
