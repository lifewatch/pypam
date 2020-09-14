import os
import pathlib
import pandas as pd
import pyhydrophone as pyhy


from pypam import acoustic_survey, geolocation





#---------------------------------------------------------------------------------------------------------
# Hydrophone Setup
#---------------------------------------------------------------------------------------------------------
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V              
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrapHF(name=name, model=model, serial_number=serial_number)

# Sound Files
st_folder_test = "C:/Users/cleap/Documents/Data/Sound Data/SoundTrap/BelwindTest/Test"
# st_folder = "//archive/other_platforms/soundtrap/2017/najaar2017_reefballs_Belwind"
st_folder = "//archive/other_platforms/soundtrap/2020/COVID-19 Westhinder"
zipped = False
include_dirs = True


#---------------------------------------------------------------------------------------------------------
# Acoustic Analysis
#---------------------------------------------------------------------------------------------------------
REF_PRESSURE = 1e-6 

# SURVEY PARAMETERS
nfft = 1024
binsize = 120.0
h = 0.1
percentiles = [10, 50, 90]
period = None
band = None


if __name__ == "__main__":
    """
    Main function
    """
    # belwind = acoustic_survey.ASA(hydrophone=soundtrap, folder_path=st_folder, zipped=zipped, \
    #             include_dirs=include_dirs, p_ref=REF_PRESSURE, binsize=binsize, nfft=nfft, period=period, band=band)

    for folder in os.listdir(st_folder):
        folder_path = os.path.join(st_folder, folder)
        clicks_df = soundtrap.read_HFfolder(st_folder, zip_mode=zipped)
        print('hello!')
