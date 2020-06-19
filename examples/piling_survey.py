import os
import pyhydrophone as pyhy


from pypam import acoustic_survey
from pypam import piling_detector



# Recordings information 
st_folder = 'C:/Users/cleap/Documents/Data/Sound Data/SoundTrap/COVID-19/'
zipped = False
include_dirs = False

# Hydrophone Setup
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)


# Detector information
min_duration = 300
ref = -6
threshold = 150
dt = 1
continous = False


# Survey parameters
# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6 

# SURVEY PARAMETERS
binsize = 120.0


if __name__ == "__main__":
    """
    Cut all the files according to the periods
    """
    piling_d = piling_detector.PilingDetector(min_duration=min_duration, ref=ref, threshold=threshold, dt=dt, continous=continous)
    asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path=st_folder, zipped=zipped, include_dirs=include_dirs, binsize=binsize)
