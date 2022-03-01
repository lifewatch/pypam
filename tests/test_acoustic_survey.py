import unittest

from pypam.acoustic_survey import ASA
import pyhydrophone as pyhy


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
band_hf = [500, 4000]
band_list = [band_lf, band_hf]
# features = ['rms', 'peak', 'sel', 'dynamic_range', 'aci', 'bi', 'sh', 'th', 'ndsi', 'aei', 'adi', 'zcr', 'zcr_avg']
fast_features = ['rms', 'peak', 'sel']
third_octaves = None
dc_subtract = True

include_dirs = False
zipped_files = False


class TestASA(unittest.TestCase):
    def setUp(self) -> None:
        self.asa = ASA(hydrophone=soundtrap, folder_path='./test_data', binsize=binsize, nfft=nfft, timezone='UTC',
                       include_dirs=include_dirs, zipped=zipped_files, dc_subtract=dc_subtract)

    def test_timestamp_array(self):
        self.asa.timestamps_array()

    def test_nmf(self):
        ds = self.asa.source_separation(window_time=1.0, n_sources=15, save_path=None, verbose=False)

    def test_features(self):
        self.asa.evolution_multiple(method_list=fast_features, band_list=band_list)

    def test_third_oct(self):
        ds = self.asa.evolution_freq_dom('spectrogram', band=third_octaves, db=True)
        print(ds)

    def test_spectrogram(self):
        self.asa.apply_to_all('spectrogram')

    def test_plots(self):
        self.asa.apply_to_all('plot_spectrogram')
        self.asa.apply_to_all('plot_psd')
        self.asa.plot_mean_psd(percentiles=[10, 50, 90])

    def test_spd(self):
        h_db = 1
        percentiles = [1, 10, 50, 90, 95]
        min_val = 60
        max_val = 140
        self.asa.plot_spd(db=True, h=h_db, percentiles=percentiles, min_val=min_val, max_val=max_val)

    def test_detect_piling_events(self):
        min_separation = 1
        max_duration = 0.2
        threshold = 20
        dt = 2.0
        detection_band = [500, 1000]

        self.asa.detect_piling_events(max_duration=max_duration, min_separation=min_separation,
                                      threshold=threshold, dt=dt, verbose=True, method='snr',
                                      save_path=None, detection_band=detection_band, analysis_band=None)

    def test_detect_ship_events(self):
        # just a smoke test to check if the function can run without errors
        self.asa.detect_ship_events(0.1, 0.5)



if __name__ == '__main__':
    unittest.main()
