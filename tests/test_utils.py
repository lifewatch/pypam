import pytest
import os
import numpy as np
import pandas as pd
import xarray
import matplotlib.pyplot as plt
import scipy
from tests import with_plots
from pypam import utils

plt.rcParams.update(plt.rcParamsDefault)
# get relative path
test_dir = os.path.dirname(__file__)


# Create artificial data of 1 second
@pytest.fixture
def artificial_data():
    fs = 512000
    test_freqs = [400, fs / 4]
    samples = fs
    noise_amp = 100
    signal_amp = 100
    data = pd.read_csv(f"{test_dir}/test_data/signal_data.csv", header=None)[
        0
    ].values
    t = np.linspace(0, 1 - 1 / fs, samples)
    phase = 2 * np.pi * t
    for test_freq in test_freqs:
        data = data + signal_amp * np.sin(test_freq * phase)

    # Set the nfft to 1 second
    nfft = fs
    return data, nfft, fs


def test_get_millidecade_bands(artificial_data):
    _, nfft, fs = artificial_data
    bands_limits, bands_c = utils.get_hybrid_millidecade_limits(
        band=[0, fs / 2], nfft=nfft
    )
    mdec_bands_test = pd.read_csv(
        f"{test_dir}//test_data/mdec_bands_test.csv", header=None
    )
    assert ((mdec_bands_test.iloc[:, 0] - bands_limits[:-1]) > 5e-5).sum() == 0
    assert ((mdec_bands_test.iloc[:, 2] - bands_limits[1:]) > 5e-5).sum() == 0
    assert ((mdec_bands_test.iloc[:, 1] - bands_c) > 5e-5).sum() == 0


def test_psd_to_millidecades(artificial_data):
    data, nfft, fs = artificial_data
    bands_limits, bands_c = utils.get_hybrid_millidecade_limits(
        band=[0, fs / 2], nfft=nfft
    )

    # Compute the spectrum manually
    ny_freq = int(int(nfft / 2))
    c_spec = np.fft.fft(data) / fs
    spectra_ds = abs(c_spec * c_spec)
    spectra = 2 * spectra_ds[: ny_freq + 1]
    spectra[0] = spectra_ds[0]
    spectra[ny_freq] = spectra_ds[ny_freq]

    fbands = scipy.fft.rfftfreq(nfft, 1 / fs)

    # Load the spectrum used for MANTA
    spectra_manta = pd.read_csv(
        f"{test_dir}//test_data/spectra.csv", header=None
    )

    # Check if they are the same
    assert (abs(spectra_manta[0].values - spectra) > 1e-5).sum() == 0

    # Convert the spectra to a datarray
    psd_da = xarray.DataArray(
        [spectra],
        coords={"id": [0], "frequency": fbands},
        dims=["id", "frequency"],
    )

    milli_psd = utils.spectra_ds_to_bands(
        psd_da, bands_limits, bands_c, fft_bin_width=fs / nfft, db=False
    )

    bandwidths = milli_psd.upper_frequency - milli_psd.lower_frequency
    milli_psd_power = milli_psd * bandwidths
    # Read MANTA's output
    mdec_power_test = pd.read_csv(f"{test_dir}//test_data/mdec_power_test.csv")

    if with_plots():
        # Plot the two outputs for comparison

        # generate figure with the two subplots
        fig, ax = plt.subplots()
        ax.plot(
            milli_psd.frequency_bins, mdec_power_test["sum"], label="MANTA"
        )
        plt.legend()
        plt.show()

        fig, ax = plt.subplots()
        milli_psd.plot(ax=ax, label="pypam")
        plt.legend()
        plt.show()

    # Check if the results are the same
    assert (
        (mdec_power_test["sum"] - milli_psd_power.sel(id=0).values).abs()
        > 1e-5
    ).sum() == 0


def test_hmb_to_decidecade():
    daily_ds = xarray.load_dataset(f"{test_dir}//test_data/test_day.nc")
    daily_ds_deci = utils.hmb_to_decidecade(
        daily_ds, "millidecade_bands", freq_coord="frequency_bins"
    )

    if with_plots():
        daily_ds_example_deci = daily_ds_deci.isel(id=0)
        daily_ds_example = daily_ds.isel(id=0)
        # Plot the two outputs for comparison
        fig, ax = plt.subplots()
        ax.plot(
            daily_ds_example_deci.frequency_bins,
            daily_ds_example_deci["millidecade_bands"],
            label="decidecades",
        )
        ax.plot(
            daily_ds_example.frequency_bins,
            daily_ds_example["millidecade_bands"],
            label="HMB",
        )
        plt.xscale("symlog")
        plt.xlim(left=10)
        plt.legend()
        plt.show()


def test_kurtosis():
    # build test data and calculate kurtosis
    n = 1000
    normally_distributed_data = scipy.stats.norm.rvs(size=n)

    result_utils = utils.kurtosis(normally_distributed_data)
    result_scipy = scipy.stats.kurtosis(
        normally_distributed_data, fisher=False
    )

    # account for slight difference in the method here
    expected_ratio = ((n - 1) / n) ** 2
    result_scipy = result_scipy * expected_ratio

    assert np.isclose(result_scipy, result_utils)
    # kurtosis should be near 3 for random gaussian distribution
    assert np.round(result_utils) == 3


def test_energy_window():
    """Check energy window method against examples in Madsen 2005."""
    Madsen_tau_1 = {"90": 81, "97": 125}  # fig 1
    Madsen_tau_2 = {"90": 9, "97": 10}  # fig 3a
    tol = 7  # us
    plot = True

    def load_and_window_data(filename, plot=False):
        raw_data = np.loadtxt(
            f"{test_dir}\\test_data\\impulsive_data\\{filename}", delimiter=","
        )
        t, P = raw_data[:, 0], raw_data[:, 1]
        t = t + np.abs(np.min(t))
        t_interp = np.linspace(0, np.max(t), num=500)
        P_interp = np.interp(t_interp, t, P)
        start, end = utils.energy_window(P_interp, 0.97)
        tau_97 = t_interp[end] - t_interp[start]
        start, end = utils.energy_window(P_interp, 0.90)
        tau_90 = t_interp[end] - t_interp[start]
        if plot:
            fig, ax = plt.subplots()
            ax.plot(t_interp, P_interp)
            ax.plot(
                t_interp[start:end],
                P_interp[start:end],
                color="r",
                linestyle="--",
            )
            fig.suptitle(
                f"energy window test, Madsen 2005, tau_90={tau_90:.2f} us"
            )
            plt.show()
        return tau_90, tau_97

    tau_90_1, tau_97_1 = load_and_window_data("madsen_digitized_signal.csv")
    tau_90_2, tau_97_2 = load_and_window_data("madsen_digitized_signal_2.csv")
    err90_1 = np.abs(Madsen_tau_1["90"] - tau_90_1) / Madsen_tau_1["90"]
    err97_1 = np.abs(Madsen_tau_1["97"] - tau_97_1) / Madsen_tau_1["97"]
    err90_2 = np.abs(Madsen_tau_2["90"] - tau_90_2) / Madsen_tau_2["90"]
    err97_2 = np.abs(Madsen_tau_2["97"] - tau_97_2) / Madsen_tau_2["97"]
    mean_error = np.mean([err90_1, err97_2, err90_2, err97_1])
    assert mean_error < 0.05  # accept 5% digitization error


def test_third_octave_band_limits():
    center, high, low = utils.decidecade_bands(16, 125, bounded=True)
    expected = np.array(
        [
            19.953,
            25.119,
            31.623,
            39.811,
            50.119,
            63.096,
            79.433,
            100,
        ]
    )
    center = np.round(center, decimals=3)
    assert np.allclose(center, expected)
