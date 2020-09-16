import os
import pandas as pd
import pyhydrophone as pyhy


from pypam import acoustic_survey, geolocation



# Sound Analysis
st_folder = 'C:/Users/cleap/Documents/Data/Sound Data/SoundTrap/COVID-19/'
bk_folder = 'C:/Users/cleap/Documents/Data/Sound Data/B&K/COVID-19/'
zipped = False
include_dirs = False

# GPS Location data
gps = "C:/Users/cleap/Documents/Data/Tracks/COVID-19/Track_2020-05-18 092212.gpx"


# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V              
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

bk_model = 'Nexus'
bk_name = 'B&K'
amplif0 = 10e-3
bk = pyhy.BruelKjaer(name=bk_name, model=bk_model, amplif=amplif0, serial_number=1)


# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6 

# SURVEY PARAMETERS
nfft = 1024
binsize = 120.0
h = 1.0
percentiles = []
period = None
band = [10, 48000]
log = False


save_folder = 'C:/Users/cleap/Documents/PhD/Projects/COVID-19/SPD/'



if __name__ == "__main__":
  