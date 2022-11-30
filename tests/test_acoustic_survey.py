import unittest
import pathlib
import matplotlib.pyplot as plt

from pypam.acoustic_survey import ASA
import pyhydrophone as pyhy


# Data information
data_path = pathlib.Path('./../data')

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

include_dirs = True
zipped_files = False


class TestASA(unittest.TestCase):
    def setUp(self) -> None:
        self.asa = ASA(hydrophone=soundtrap, folder_path='./test_data', binsize=binsize, nfft=nfft, timezone='UTC',
                       include_dirs=include_dirs, zipped=zipped_files, dc_subtract=dc_subtract)

    def test_timestamp_array(self):
        self.asa.timestamps_array()

    def test_nmf(self):
        ds = self.asa.source_separation(window_time=1.0, n_sources=15, save_path=None, verbose=False)
        self.asa = ASA(hydrophone=soundtrap, folder_path='./../data', binsize=binsize, nfft=nfft, utc=True,
                       include_dirs=include_dirs, zipped=zipped_files)

    def test_features(self):
        self.asa.evolution_multiple(method_list=fast_features, band_list=band_list)

    def test_third_oct(self):
        ds = self.asa.evolution_freq_dom('spectrogram', band=third_octaves, db=True)
        print(ds)

    def test_spectrogram(self):
        self.asa.apply_to_all('spectrogram')

    def test_apply_to_all(self):
        self.asa.apply_to_all('plot_spectrogram')
        self.asa.apply_to_all('plot_psd')

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

    def test_plot_spd(self):
        self.asa.plot_spd(percentiles=[1, 5, 10, 50, 90, 95, 99])

    def test_plot_mean_spectrum(self):
        self.asa.plot_mean_power_spectrum()
        self.asa.plot_mean_psd(percentiles=[10, 50, 90])

    def test_plot_ltsa(self):
        self.asa.plot_power_ltsa(percentiles=[1, 5, 10, 50, 90, 95, 99])
        self.asa.plot_psd_ltsa()

    def test_plot_rms_evolution(self):
        self.asa.plot_rms_evolution()

    def test_plot_daily_patterns(self):
        self.asa.plot_rms_daily_patterns()

    def test_millidecade_bands(self):
        # Set the frequency resolution to 1 Hz and the duration of 1 second
        milli_psd = self.asa.hybrid_millidecade_bands(db=True, method='spectrum', band=[0, 4000], percentiles=None)
        milli_psd['millidecade_bands'].plot()
        plt.show()

        milli_psd = self.asa.hybrid_millidecade_bands(db=True, method='density', band=[0, 4000], percentiles=None)
        milli_psd['millidecade_bands'].plot()
        plt.show()


if __name__ == '__main__':
    unittest.main()
