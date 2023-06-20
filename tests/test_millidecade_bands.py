import pyhydrophone as pyhy
from pypam.acoustic_file import AcuFile
import pytest

# Adapted from examples/millidecade_bands.py to use pytest and snapshots.

@pytest.fixture
def millidecade_bands():
    # If the model is not implemented yet in pyhydrophone, a general Hydrophone can be defined
    model = 'ST300HF'
    name = 'SoundTrap'
    serial_number = 67416073
    soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

    p_ref = 1.0
    fs = 8000
    nfft = fs
    fft_overlap = 0.5
    binsize = 60.0
    bin_overlap = 0
    method = 'density'
    band = [0, 4000]

    wav_path = 'tests/test_data/67416073.210610033655.wav'
    acu_file = AcuFile(sfile=wav_path, hydrophone=soundtrap,
                       p_ref=p_ref, timezone='UTC', channel=0, calibration=None,
                       dc_subtract=False)

    millis = acu_file.hybrid_millidecade_bands(nfft=nfft, fft_overlap=fft_overlap, binsize=binsize,
                                             bin_overlap=bin_overlap,
                                             db=True, method=method, band=band)

    return millis['millidecade_bands']


def test_millidecade_bands(millidecade_bands, snapshot):
    # check data as pandas in dict format:
    data_frame = millidecade_bands.to_pandas()
    data_frame.columns = data_frame.columns.to_numpy().round(6)
    out = data_frame.round(6).to_dict()
    assert out == snapshot(name="data-as-pandas-dict")

    # # check data as list:  (probably unneeded per the above)
    # assert millidecade_bands.data.tolist() == snapshot(name="data-as-list")

    # check coords:
    assert millidecade_bands.coords.to_dataset().to_dataframe().round(6).to_dict() == snapshot(name="coords")
