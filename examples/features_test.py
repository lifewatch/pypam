import os
import pathlib
import pandas as pd
import pyhydrophone as pyhy
import matplotlib.pyplot as plt


from pypam import acoustic_survey, geolocation



# Sound Analysis
st_folder = pathlib.Path('//archive/other_platforms/soundtrap/2020/COVID-19/200505 Tripode/Westhinder/')
zipped = False
include_dirs = True

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
nfft = 512
binsize = None
h = 1.0
band_lf = [100, 500]
band_mf = [500, 2000]
band_hf = [2000, 20000]
bands_list = [band_lf, band_mf, band_hf]


if __name__ == "__main__":
    # asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path=st_folder, binsize=binsize,
    #                           nfft=nfft, include_dirs=include_dirs)
    # evo = asa.evolution_multiple(method_list=['rms', 'dynamic_range', 'aci', 'sel', 'peak'], band_list=bands_list)

    evo = pd.read_pickle('df_test.pkl', compression='gzip')
    for method in evo.columns.get_level_values('method'):
        evo[method].plot(linewidth=1.0)
        plt.title(method)
        plt.show()
        plt.close()
    print('hello')


