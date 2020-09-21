import os
import pathlib
import pandas as pd
import pyhydrophone as pyhy
import matplotlib.pyplot as plt


from pypam import acoustic_survey, geolocation



# Sound Analysis
st_folder = pathlib.Path('//archive/other_platforms/soundtrap/2020/COVID-19/200505 Tripode/Westhinder')
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
binsize = 10.0
h = 1.0
band_lf = [100, 500]
band_mf = [500, 2000]
band_hf = [2000, 20000]


if __name__ == "__main__":
  # asa_lf = acoustic_survey.ASA(hydrophone=soundtrap, folder_path=st_folder, binsize=binsize, nfft=nfft, band=band_lf, include_dirs=include_dirs)
  # evo_lf = asa_lf.evolution_multiple(method_list=['rms', 'dynamic_range'])

  # evo_lf['cumsum'] = evo_lf.dynamic_range.cumsum()
  # evo_lf.plot(subplots=True)
  # plt.show()

  asa_mf = acoustic_survey.ASA(hydrophone=soundtrap, folder_path=st_folder, binsize=binsize, nfft=nfft, band=band_mf, include_dirs=include_dirs)
  evo_mf = asa_mf.evolution_multiple(method_list=['rms', 'dynamic_range'])

  evo_mf['cumsum'] = evo_mf.dynamic_range.cumsum()
  evo_mf.plot(subplots=True)
  plt.show()

  asa_hf = acoustic_survey.ASA(hydrophone=soundtrap, folder_path=st_folder, binsize=binsize, nfft=nfft, band=band_hf, include_dirs=include_dirs)
  evo_hf = asa_hf.evolution_multiple(method_list=['rms', 'dynamic_range'])

  evo_hf['cumsum'] = evo_hf.dynamic_range.cumsum()
  evo_hf.plot(subplots=True)
  plt.show()

