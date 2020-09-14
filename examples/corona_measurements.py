import os
import glob
import pathlib
import geopandas
import pandas as pd
import pandas as pd
from pandas.io.pickle import read_pickle
import pyhydrophone as pyhy
import matplotlib.pyplot as plt


from pypam import acoustic_survey, geolocation



# Sound Analysis
summary_path = pathlib.Path('C:/Users/cleap/Documents/PhD/Projects/COVID-19/summary_recordings.csv')
save_path = pathlib.Path('C:/Users/cleap/Documents/PhD/Projects/COVID-19/data_overview.pkl')
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
upam = pyhy.Seiche(name=upam_name, model=upam_name, serial_number=upam_serial_number, sensitivity=upam_sensitivity, preamp_gain=upam_preamp_gain, Vpp=upam_Vpp)


# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6 

# SURVEY PARAMETERS
nfft = 512
binsize = 60.0
# band_lf = [100, 500]
# band_mf = [500, 2000]
# band_hf = [2000, 20000]
band = None


def get_data_overview(hydrophone, folder_path, gps_path, datetime_col, lat_col, lon_col, method):
    """
    Return a db with a data overview of the folder 
    """
    asa = acoustic_survey.ASA(hydrophone=hydrophone, folder_path=folder_path, binsize=binsize, nfft=nfft, band=band, include_dirs=include_dirs)
    asa_evo = asa.timestamps_df()
    duration = asa.duration()

    geoloc = geolocation.SurveyLocation(gps_path, datetime_col=datetime_col, lat_col=lat_col, lon_col=lon_col)

    asa_points = geoloc.add_survey_location(asa_evo)
    
    asa_points['instrument'] = hydrophone.name
    asa_points['method'] = method

    return duration, asa_points[['datetime', 'instrument', 'method', 'geometry']]


def plot_data_overview(save_path):
    """
    Plot the overview in a map
    """
    geoloc = geolocation.SurveyLocation(save_path)
    geoloc.plot_survey_color(column='method', df=geoloc, units='Method')

    



if __name__ == "__main__":
    metadata = pd.read_csv(summary_path)
    metadata['duration'] = 0
    overview = pd.DataFrame(columns=['datetime', 'instrument', 'method', 'geometry'])
    for index in metadata.index:
        row = metadata.iloc[index] 
        if row['instrument'] == 'SoundTrap':
            hydrophone = soundtrap
        elif row['instrument'] == 'B&K':
            hydrophone = bk
        elif row['instrument'] == 'uPam':
            hydrophone = upam
        else:
            raise Exception('Hydrophone %s is not defined!' % (row['instrument']))
        duration, points = get_data_overview(hydrophone=hydrophone, 
                                            folder_path=row['data_folder'], 
                                            gps_path=row['gps_folder'],
                                            datetime_col=row['datetime_col'],
                                            lat_col=row['lat_col'],
                                            lon_col=row['lon_col'],
                                            method=row['method'])
        overview = overview.append(points)
        metadata[index, 'duration'] = duration
    metadata.to_csv(summary_path)
    overview.to_pickle(save_path)

    plot_data_overview(save_path)
