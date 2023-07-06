import unittest
import pathlib
import matplotlib.pyplot as plt

from pypam.acoustic_survey import ASA
import pyhydrophone as pyhy
from tests import skip_unless_with_plots, with_plots


# Data information
folder_path = pathlib.Path('tests/test_data')

# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 8000
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

# Don't plot if it is running on CI
verbose = with_plots()


class TestASA(unittest.TestCase):
    def setUp(self) -> None:
        self.asa = ASA(hydrophone=soundtrap, folder_path=folder_path, binsize=binsize, nfft=nfft, timezone='UTC',
                       include_dirs=include_dirs, zipped=zipped_files, dc_subtract=dc_subtract)

    def test_empty_directory(self):
        with self.assertRaises(ValueError) as context:
            ASA(hydrophone=soundtrap, folder_path=folder_path.joinpath('empty_folder'), binsize=binsize,
                nfft=nfft, timezone='UTC', include_dirs=include_dirs, zipped=zipped_files, dc_subtract=dc_subtract)

    def test_path_not_exists(self):
        with self.assertRaises(FileNotFoundError) as context:
            ASA(hydrophone=soundtrap, folder_path='non_existing_folder', binsize=binsize,
                nfft=nfft, timezone='UTC', include_dirs=include_dirs, zipped=zipped_files, dc_subtract=dc_subtract)

    def test_timestamp_array(self):
        self.asa.timestamps_array()

    def test_nmf(self):
        ds = self.asa.source_separation(window_time=1.0, n_sources=15, save_path=None, verbose=verbose)

    def test_features(self):
        self.asa.evolution_multiple(method_list=fast_features, band_list=band_list)

    def test_third_oct(self):
        ds = self.asa.evolution_freq_dom('spectrogram', band=third_octaves, db=True)
        print(ds)

    def test_millidecade_bands(self):
        # Set the frequency resolution to 1 Hz
        # Check for the complete broadband and a band filtered in the lower frequencies
        milli_psd = self.asa.hybrid_millidecade_bands(db=True, method='density', band=[0, 4000], percentiles=None)
        assert (milli_psd.frequency_bins[0] == 0) & (milli_psd.frequency_bins[-1] == 4000)

        milli_psd_filtered = self.asa.hybrid_millidecade_bands(db=True, method='density', band=[10, 4000], percentiles=None)
        assert (milli_psd_filtered.frequency_bins[0] == 10) & (milli_psd_filtered.frequency_bins[-1] == 4000)

        # Same check than above, but for spectrum
        milli_psd_spectrum = self.asa.hybrid_millidecade_bands(db=True, method='spectrum', band=[10, 4000], percentiles=None)
        assert (milli_psd_spectrum.frequency_bins[0] == 10) & (milli_psd_spectrum.frequency_bins[-1] == 4000)

        # Now check for a band with a higher limit < nyq
        # Frequency resolution is then not 1 Hz because the signal is downsampled but the nfft is still 8000
        milli_psd_halfhz = self.asa.hybrid_millidecade_bands(db=True, method='density', band=[0, 2000], percentiles=None)
        assert (milli_psd_halfhz.frequency_bins[0] == 0) & (milli_psd_halfhz.frequency_bins[-1] == 2000)
        assert ((milli_psd_halfhz.frequency[2] - milli_psd_halfhz.frequency[1]) == 0.5)

        # Change the nfft so it is 2 Hz resolution
        self.asa.nfft = 4000
        milli_psd_2hz = self.asa.hybrid_millidecade_bands(db=True, method='density', band=[0, 4000], percentiles=None)
        assert (milli_psd_2hz.frequency_bins[0] == 0) & (milli_psd_2hz.frequency_bins[-1] == 4000)
        assert ((milli_psd_2hz.frequency[2] - milli_psd_2hz.frequency[1]) == 2)

        # Check with a multiple of 2 to the power -> it is faster so some people might want to use it
        self.asa.nfft = 512
        milli_psd_512 = self.asa.hybrid_millidecade_bands(db=True, method='density', band=[50, 1000], percentiles=None)
        assert (milli_psd_512.frequency[0] == 50.78125) & (milli_psd_512.frequency[-1] == 1000)

        if verbose:
            fig, ax = plt.subplots()
            milli_psd_filtered['millidecade_bands'].mean(dim='id').plot(ax=ax, label='filtered')
            milli_psd['millidecade_bands'].mean(dim='id').plot(ax=ax, label='not filtered')
            milli_psd_halfhz['millidecade_bands'].mean(dim='id').plot(ax=ax, label='half_hz')
            milli_psd_2hz['millidecade_bands'].mean(dim='id').plot(ax=ax, label='2Hz')
            milli_psd_512['millidecade_bands'].mean(dim='id').plot(ax=ax, label='512')
            plt.legend()
            plt.show()

    def test_spectrogram(self):
        self.asa.apply_to_all('spectrogram')

    @skip_unless_with_plots()
    def test_apply_to_all(self):
        self.asa.apply_to_all('plot_spectrogram')
        self.asa.apply_to_all('plot_spectrum_mean', scaling='density')

    # def test_detect_piling_events(self):
    #     min_separation = 1
    #     max_duration = 0.2
    #     threshold = 20
    #     dt = 2.0
    #     detection_band = [500, 1000]
    #
    #     self.asa.detect_piling_events(max_duration=max_duration, min_separation=min_separation,
    #                                   threshold=threshold, dt=dt, verbose=verbose, method='snr',
    #                                   save_path=None, detection_band=detection_band, analysis_band=None)
    #
    # def test_detect_ship_events(self):
    #     # just a smoke test to check if the function can run without errors
    #     self.asa.detect_ship_events(0.1, 0.5, verbose=verbose)

    @skip_unless_with_plots()
    def test_plot_spd(self):
        h_db = 1
        percentiles = [1, 10, 50, 90, 95]
        min_val = 60
        max_val = 140
        self.asa.plot_spd(db=True, h=h_db, percentiles=percentiles, min_val=min_val, max_val=max_val)

    @skip_unless_with_plots()
    def test_plot_mean_spectrum(self):
        self.asa.plot_mean_power_spectrum()
        self.asa.plot_mean_psd(percentiles=[10, 50, 90])

    @skip_unless_with_plots()
    def test_plot_ltsa(self):
        self.asa.plot_power_ltsa(percentiles=[1, 5, 10, 50, 90, 95, 99])
        self.asa.plot_psd_ltsa()

    @skip_unless_with_plots()
    def test_plot_rms_evolution(self):
        self.asa.plot_rms_evolution()

    @skip_unless_with_plots()
    def test_plot_millidecade_bands(self):
        # Set the frequency resolution to 1 Hz and the duration of 1 second
        milli_psd = self.asa.hybrid_millidecade_bands(db=True, method='spectrum', band=[0, 4000], percentiles=None)
        milli_psd['millidecade_bands'].plot()
        plt.show()

        milli_psd = self.asa.hybrid_millidecade_bands(db=True, method='density', band=[0, 4000], percentiles=None)
        milli_psd['millidecade_bands'].plot()
        plt.show()


if __name__ == '__main__':
    unittest.main()
