from pypam import acoustic_survey

import pyhydrophone as pyhy


# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
st_model = 'ST300HF'
st_name = 'SoundTrap'
st_serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=st_name, model=st_model, serial_number=st_serial_number)

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 4096
binsize = None

# Piling params
min_separation = 1
max_duration = 0.2
threshold = 20
dt = 2.0
detection_band = [500, 1000]

if __name__ == "__main__":
    asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path='./../tests/test_data', zipped=False,
                              include_dirs=False, binsize=binsize, nfft=nfft)
    df = asa.detect_ship_events(min_duration=5.0, threshold=120.0)
    df.to_csv('ship_detections.csv')

    df = asa.detect_piling_events(max_duration=max_duration, min_separation=min_separation,
                                  threshold=threshold, dt=dt, verbose=True, method='snr',
                                  save_path=None, detection_band=detection_band, analysis_band=None)
    df.to_csv('piling_detected.csv')
