from pypam import acoustic_survey, geolocation


import pathlib
import geopandas
import pandas as pd
import pyhydrophone as pyhy


folder_path = '//archive/other_platforms/soundtrap/2020/COVID-19/200427 Zeekat/Renilde'
zipped = False
include_dirs = True

# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
st_model = 'ST300HF'
st_name = 'SoundTrap'
st_serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=st_name, model=st_model, serial_number=st_serial_number)

bk_model = 'Nexus'
bk_name = 'B&K'
amplif0 = 10e-3
bk = pyhy.BruelKjaer(name=bk_name, model=bk_model, amplif=amplif0, serial_number=1)

upam_model = 'uPam'
upam_name = 'Seiche'
upam_serial_number = 'SM7213'
upam_sensitivity = -196.0
upam_preamp_gain = 0.0
upam_Vpp = 20.0
upam = pyhy.Seiche(name=upam_name, model=upam_name, serial_number=upam_serial_number, sensitivity=upam_sensitivity,
                   preamp_gain=upam_preamp_gain, Vpp=upam_Vpp)


# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 4096
binsize = None

if __name__ == "__main__":
    asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path=folder_path, zipped=zipped, include_dirs=include_dirs,
                              binsize=binsize, nfft=nfft)
    df = asa.detect_ship_events(min_duration=10.0, threshold=120.0)
    df.to_csv('ship_detections.csv')
