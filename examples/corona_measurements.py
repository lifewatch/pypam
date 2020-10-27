from pypam import acoustic_survey, geolocation


import pathlib
import geopandas
import pandas as pd
import pyhydrophone as pyhy


# Sound Analysis
summary_path = pathlib.Path('S:/onderzoek/6. Marine Observation Center/Projects/PhD_Clea/Topics/'
                            'SoundscapesMapping/dataset/data_summary.csv')
# save_data_path = pathlib.Path('C:/Users/cleap/Documents/PhD/Projects/COVID-19/locations.pkl')
zipped = False
include_dirs = False

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
nfft = 512
binsize = 120.0
# band_lf = [100, 500]
# band_mf = [500, 2000]
# band_hf = [2000, 20000]


def get_data_locations(hydrophone, location, folder_path, gps_path, datetime_col, lat_col, lon_col, method):
    """
    Return a db with a data overview of the folder 
    """
    geoloc = geolocation.SurveyLocation(gps_path, datetime_col=datetime_col, lat_col=lat_col, lon_col=lon_col)

    asa = acoustic_survey.ASA(hydrophone=hydrophone, folder_path=folder_path, binsize=binsize,
                              nfft=nfft, include_dirs=include_dirs)
    if method == 'Moored':
        start, _ = asa.start_end_timestamp()
        asa_evo = pd.DataFrame({'datetime': start}, index=[0])
    else:
        asa_evo = asa.timestamps_df()
    
    asa_points = geoloc.add_survey_location(asa_evo)

    asa_points['instrument'] = hydrophone.name
    asa_points['location'] = location
    asa_points['method'] = method

    return asa_points[['datetime', 'instrument', 'location', 'method', 'geometry']]


def get_data_overview(hydrophone, folder_path, gps_path, datetime_col, lat_col, lon_col, method):
    """
    Return a db with a data overview of the folder 
    """
    overview_df = pd.DataFrame(columns=['start_datetime', 'end_datetime', 'location', 'instrument',
                                        'method', 'data_folder', 'gps_folder', 'datetime_col',
                                        'lat_col', 'lon_col', 'duration'])
    start, end, duration = get_location_metadata(hydrophone, folder_path)
    shipwreck_name = folder_path.parts[-1]
    shipwreck_row = pd.DataFrame({'start_datetime': start,
                                  'end_datetime': end,
                                  'location': shipwreck_name,
                                  'instrument': hydrophone.name,
                                  'method': method,
                                  'data_folder': folder_path,
                                  'gps_folder': gps_path,
                                  'datetime_col': datetime_col,
                                  'lat_col': lat_col,
                                  'lon_col': lon_col,
                                  'duration': duration})
    overview_df = overview_df.append(shipwreck_row)
    return overview_df


def get_location_metadata(hydrophone, folder_path):
    """
    Get all the metadata of the location
    """
    asa = acoustic_survey.ASA(hydrophone=hydrophone, folder_path=folder_path, binsize=binsize,
                              nfft=nfft, include_dirs=include_dirs)
    start, end = asa.start_end_timestamp()
    duration = asa.duration()

    return start, end, duration


def plot_data_location_overview(save_path):
    """
    Plot the overview in a map
    """
    geoloc = geolocation.SurveyLocation()
    df = pd.read_pickle(save_path)
    geodf = geopandas.GeoDataFrame(df, crs='EPSG:4326', geometry='geometry')
    geoloc.plot_survey_color(column='method', df=geodf, units='Method')


if __name__ == "__main__":
    metadata = pd.read_csv(summary_path)
    metadata['duration'] = 0
    overview = pd.DataFrame()
    for index in metadata.index:
        row = metadata.iloc[index] 
        if row['instrument'] == 'SoundTrap':
            inst = soundtrap
        elif row['instrument'] == 'B&K':
            inst = bk
        elif row['instrument'] == 'uPam':
            inst = upam
        else:
            raise Exception('Hydrophone %s is not defined!' % (row['instrument']))
        overview_location = get_data_locations(hydrophone=inst,
                                               location=row['location'],
                                               folder_path=row['data_folder'],
                                               gps_path=row['gps_folder'],
                                               datetime_col=row['datetime_col'],
                                               lat_col=row['lat_col'],
                                               lon_col=row['lon_col'],
                                               method=row['method'])
        overview = overview.append(overview_location)
        # start_datetime, end_datetime, duration_s = get_location_metadata(hydrophone=inst,
        #                                                                  folder_path=row['data_folder'])
        # metadata.at[index, ['start_datetime', 'end_datetime', 'duration']] = [start_datetime, end_datetime, duration_s]
    # overview.to_pickle(save_data_path)
    # metadata.to_csv('summary_recordings.csv', index=False)

    geoloc = geolocation.SurveyLocation()
    # geodf = geopandas.GeoDataFrame(overview, crs='EPSG:4326', geometry='geometry')
    geoloc.plot_survey_color(column='method', df=overview, units='Method')
