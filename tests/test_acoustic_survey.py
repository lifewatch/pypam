import unittest
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os

from pypam.acoustic_survey import ASA
import pyhydrophone as pyhy
from tests import skip_unless_with_plots, with_plots

plt.rcParams.update(plt.rcParamsDefault)

# get relative path
test_dir = os.path.dirname(__file__)

# Data information
folder_path = pathlib.Path(f"{test_dir}/test_data")

# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
model = "ST300HF"
name = "SoundTrap"
serial_number = 67416073
calibration_file = pathlib.Path(f"{test_dir}/test_data/calibration_data.xlsx")
soundtrap = pyhy.soundtrap.SoundTrap(
    name=name,
    model=model,
    serial_number=serial_number,
    calibration_file=calibration_file,
    val="sensitivity",
    freq_col_id=1,
    val_col_id=29,
    start_data_id=6,
)

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 8000
binsize = 60.0
band_lf = [50, 500]
band_hf = [500, 4000]
band_list = [band_lf, band_hf]
# features = ['rms', 'peak', 'sel', 'dynamic_range', 'aci', 'bi', 'sh', 'th', 'ndsi', 'aei', 'adi', 'zcr', 'zcr_avg']
acoustic_indices_features = [
    "aci",
    "bi",
    "sh",
    "th",
    "ndsi",
    "aei",
    "adi",
    "zcr",
    "zcr_avg",
]
fast_features = ["rms", "peak", "sel"]
third_octaves = None
dc_subtract = True

include_dirs = False
zipped_files = False

# Don't plot if it is running on CI
verbose = with_plots()


class TestASA(unittest.TestCase):
    def setUp(self) -> None:
        self.asa = ASA(
            hydrophone=soundtrap,
            folder_path=folder_path,
            binsize=binsize,
            nfft=nfft,
            timezone="UTC",
            include_dirs=include_dirs,
            zipped=zipped_files,
            dc_subtract=dc_subtract,
        )
        self.asa_flac = ASA(
            hydrophone=soundtrap,
            folder_path=folder_path.joinpath("flac_data"),
            binsize=binsize,
            nfft=nfft,
            timezone="UTC",
            include_dirs=include_dirs,
            zipped=zipped_files,
            dc_subtract=dc_subtract,
            extension=".flac",
        )
        self.asa_gridded = ASA(
            hydrophone=soundtrap,
            folder_path=folder_path,
            binsize=binsize,
            nfft=nfft,
            timezone="UTC",
            include_dirs=include_dirs,
            zipped=zipped_files,
            dc_subtract=dc_subtract,
            gridded_data=True,
        )

    def test_empty_directory(self):
        with self.assertRaises(ValueError) as context:
            ASA(
                hydrophone=soundtrap,
                folder_path=folder_path.joinpath("empty_folder"),
                binsize=binsize,
                nfft=nfft,
                timezone="UTC",
                include_dirs=include_dirs,
                zipped=zipped_files,
                dc_subtract=dc_subtract,
            )

    def test_path_not_exists(self):
        with self.assertRaises(FileNotFoundError) as context:
            ASA(
                hydrophone=soundtrap,
                folder_path="non_existing_folder",
                binsize=binsize,
                nfft=nfft,
                timezone="UTC",
                include_dirs=include_dirs,
                zipped=zipped_files,
                dc_subtract=dc_subtract,
            )

    def test_timestamp_array(self):
        self.asa.timestamps_array()

    def test_features(self):
        self.asa.evolution_multiple(method_list=fast_features, band_list=band_list)
        ds = self.asa.evolution_multiple(
            method_list=acoustic_indices_features,
            min_freq=0,
            max_freq=4000,
            anthrophony=(1000, 2000),
            biophony=(2000, 4000),
        )
        assert (ds.aci.values >= 0).all()
        assert (ds.bi.values >= 0).all()
        assert np.logical_and(ds.sh.values >= 0, ds.sh.values <= 1).all()
        assert np.logical_and(ds.th.values >= 0, ds.th.values <= 1).all()
        assert np.logical_and(ds.ndsi.values >= -1, ds.ndsi.values <= 1).all()
        assert np.logical_and(ds.aei.values >= 0, ds.aei.values <= 1).all()
        assert (ds.adi.values >= 0).all()
        assert np.logical_and(ds.zcr.values >= 0, ds.zcr.values <= 1).all()
        assert np.logical_and(ds.zcr_avg.values >= 0, ds.zcr_avg.values <= 1).all()

    def test_third_oct_flac(self):
        ds = self.asa_flac.evolution_freq_dom(
            "spectrogram", band=third_octaves, db=True
        )
        print(ds)

    def test_third_oct(self):
        ds = self.asa.evolution_freq_dom("spectrogram", band=third_octaves, db=True)
        print(ds)

    def test_millidecade_bands(self):
        # Set the frequency resolution to 1 Hz
        # Check for the complete broadband and a band filtered in the lower frequencies
        milli_psd = self.asa.hybrid_millidecade_bands(
            db=True, method="density", band=[0, 4000], percentiles=None
        )
        assert (milli_psd.frequency_bins[0] == 0) & (
            milli_psd.frequency_bins[-1] == 4000
        )

        milli_psd_filtered = self.asa.hybrid_millidecade_bands(
            db=True, method="density", band=[10, 4000], percentiles=None
        )
        assert (milli_psd_filtered.frequency_bins[0] == 10) & (
            milli_psd_filtered.frequency_bins[-1] == 4000
        )

        # Same check than above, but for spectrum
        milli_psd_spectrum = self.asa.hybrid_millidecade_bands(
            db=True, method="spectrum", band=[10, 4000], percentiles=None
        )
        assert (milli_psd_spectrum.frequency_bins[0] == 10) & (
            milli_psd_spectrum.frequency_bins[-1] == 4000
        )

        # Now check for a band with a higher limit < nyq
        # Frequency resolution is then not 1 Hz because the signal is downsampled but the nfft is still 8000
        milli_psd_halfhz = self.asa.hybrid_millidecade_bands(
            db=True, method="density", band=[0, 2000], percentiles=None
        )
        assert (milli_psd_halfhz.frequency_bins[0] == 0) & (
            milli_psd_halfhz.frequency_bins[-1] == 2000
        )
        assert (milli_psd_halfhz.frequency[2] - milli_psd_halfhz.frequency[1]) == 0.5

        # Change the nfft so it is 2 Hz resolution
        self.asa.nfft = 4000
        milli_psd_2hz = self.asa.hybrid_millidecade_bands(
            db=True, method="density", band=[0, 4000], percentiles=None
        )
        assert (milli_psd_2hz.frequency_bins[0] == 0) & (
            milli_psd_2hz.frequency_bins[-1] == 4000
        )
        assert (milli_psd_2hz.frequency[2] - milli_psd_2hz.frequency[1]) == 2

        # Check with a multiple of 2 to the power -> it is faster so some people might want to use it
        self.asa.nfft = 512
        milli_psd_512 = self.asa.hybrid_millidecade_bands(
            db=True, method="density", band=[50, 1000], percentiles=None
        )
        assert (milli_psd_512.frequency[0] == 50.78125) & (
            milli_psd_512.frequency[-1] == 1000
        )

        if verbose:
            fig, ax = plt.subplots()
            milli_psd_filtered["millidecade_bands"].mean(dim="id").plot(
                ax=ax, label="filtered"
            )
            milli_psd["millidecade_bands"].mean(dim="id").plot(
                ax=ax, label="not filtered"
            )
            milli_psd_halfhz["millidecade_bands"].mean(dim="id").plot(
                ax=ax, label="half_hz"
            )
            milli_psd_2hz["millidecade_bands"].mean(dim="id").plot(ax=ax, label="2Hz")
            milli_psd_512["millidecade_bands"].mean(dim="id").plot(ax=ax, label="512")
            plt.legend()
            plt.show()

    def test_millidecade_bands_gridded(self):
        # Set the frequency resolution to 1 Hz
        # Check for the complete broadband and a band filtered in the lower frequencies
        milli_psd = self.asa_gridded.hybrid_millidecade_bands(
            db=True, method="density", band=[0, 4000], percentiles=None
        )
        assert milli_psd.datetime.dt.time.values[0].second == 0

    def test_spectrogram(self):
        self.asa.apply_to_all("spectrogram")

    @skip_unless_with_plots()
    def test_apply_to_all(self):
        self.asa.apply_to_all("plot_spectrogram")
        self.asa.apply_to_all("plot_spectrum_median", scaling="density")

    @skip_unless_with_plots()
    def test_plot_spd(self):
        h_db = 1
        percentiles = [1, 10, 50, 90, 95]
        min_val = 60
        max_val = 140
        self.asa.plot_spd(
            db=True, h=h_db, percentiles=percentiles, min_val=min_val, max_val=max_val
        )

    @skip_unless_with_plots()
    def test_plot_median_spectrum(self):
        self.asa.plot_median_power_spectrum()
        self.asa.plot_median_psd(percentiles=[10, 50, 90])

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
        milli_psd = self.asa.hybrid_millidecade_bands(
            db=True, method="spectrum", band=[0, 4000], percentiles=None
        )
        milli_psd["millidecade_bands"].plot()
        plt.show()

        milli_psd = self.asa.hybrid_millidecade_bands(
            db=True, method="density", band=[0, 4000], percentiles=None
        )
        milli_psd["millidecade_bands"].plot()
        plt.show()

    def test_update_freq_cal(self):
        ds_psd = self.asa.evolution_freq_dom("psd")
        ds_psd_updated = self.asa.update_freq_cal(ds=ds_psd, data_var="band_density")
        print(ds_psd["band_density"].values)
        print(ds_psd_updated["band_density"].values)


if __name__ == "__main__":
    unittest.main()
