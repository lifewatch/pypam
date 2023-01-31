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


def test_millidecade_bands_dict(millidecade_bands, snapshot):
    pd = millidecade_bands.to_pandas()
    out = pd.round(6).to_dict()
    assert out == snapshot


def test_millidecade_bands_csv(millidecade_bands, snapshot):
    pd = millidecade_bands.to_pandas()
    out = pd.round(6).to_csv(index=False)
    assert out == snapshot
