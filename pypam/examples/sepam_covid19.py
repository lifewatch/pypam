import os
import sys
import pandas as pd
import pyhydrophone as pyhy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from acousticsurvey import acoustic_survey, geolocation



# Sound Analysis
# st_folder27 = 'C:/Users/cleap/Documents/Data/Sound Data/SoundTrap/COVID-19/67416073.200427'
# st_folder29 = 'C:/Users/cleap/Documents/Data/Sound Data/SoundTrap/COVID-19/67416073.200429'
st_folder = 'C:/Users/cleap/Documents/Data/Sound Data/SoundTrap/COVID-19/'
# bk_folder29 = 'C:/Users/cleap/Documents/Data/Sound Data/B&K/COVID-19/Zeekat 200429'
# bk_folder04 = 'C:/Users/cleap/Documents/Data/Sound Data/B&K/COVID-19/Simon Stevin 200504'
# bk_folder06 = 'C:/Users/cleap/Documents/Data/Sound Data/B&K/COVID-19/Simon Stevin 200506'
# bk_folder07 = 'C:/Users/cleap/Documents/Data/Sound Data/B&K/COVID-19/Simon Stevin 200507'
# bk_folder08 = 'C:/Users/cleap/Documents/Data/Sound Data/B&K/COVID-19/Simon Stevin 200508'
bk_folder = 'C:/Users/cleap/Documents/Data/Sound Data/B&K/COVID-19/'
zipped = False
include_dirs = False

# GPS Location data
# gps_29 = "C:/Users/cleap/Documents/Data/Tracks/COVID-19/Navionics_archive_export_Zeekat_29042020.gpx"
# gps_rest = "C:/Users/cleap/Documents/Data/Tracks/COVID-19/Track_2020-05-06 171724.gpx"
# gps_rest2 = "C:/Users/cleap/Documents/Data/Tracks/COVID-19/Track_2020-05-18 092212.gpx"
gps = "C:/Users/cleap/Documents/Data/Tracks/COVID-19/gps_sepam_covid.pkl"
map_file = 'C:/Users/cleap/Documents/Data/Maps/BPNS_wrecks.tif'

# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V              
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

bk_model = 'Nexus'
bk_name = 'B&K'
amplif0 = 10e-3
bk = pyhy.BruelKjaer(name=bk_name, model=bk_model, amplif=amplif0)


#---------------------------------------------------------------------------------------------------------
# Cut the recording periods and store them in separate folders 
#---------------------------------------------------------------------------------------------------------
# periods27 = [ 
#             (['27/04/2020 09:47:14', '27/04/2020 10:32:54'], 'Nautica Ena'),
#             (['27/04/2020 10:42:49', '27/04/2020 10:53:49'], 'Loreley'),
#             (['27/04/2020 10:56:19', '27/04/2020 11:35:04'], 'Loreley'),
#             (['27/04/2020 11:45:04', '27/04/2020 11:52:54'], 'Loreley'),
#             (['27/04/2020 12:02:14', '27/04/2020 12:14:24'], 'Renilde'),
#             (['27/04/2020 12:18:24', '27/04/2020 12:50:24'], 'Renilde'),
#             (['27/04/2020 12:56:04', '27/04/2020 13:08:34'], 'Noordster'),
#             (['27/04/2020 13:12:04', '27/04/2020 13:44:59'], 'Noordster'),
#             (['27/04/2020 13:50:04', '27/04/2020 14:00:14'], 'Paragon'),
#             (['27/04/2020 14:02:54', '27/04/2020 14:33:59'], 'Paragon'),
#             (['27/04/2020 14:38:49', '27/04/2020 15:10:44'], 'Coast')
#           ]

# periods29 = [ 
#             (['29/04/2020 11:48:28', '29/04/2020 11:51:13'], 'Heinkel 111'),
#             (['29/04/2020 11:53:28', '29/04/2020 11:54:58'], 'Heinkel 111'),
#             (['29/04/2020 11:58:33', '29/04/2020 12:47:28'], 'Heinkel 111'),
#             (['29/04/2020 13:09:03', '29/04/2020 13:47:38'], 'HMS Colsay'),
#             (['29/04/2020 14:05:58', '29/04/2020 14:34:48'], 'Lola'),
#             (['29/04/2020 14:38:18', '29/04/2020 14:49:08'], 'Lola'),
#             (['29/04/2020 15:09:18', '29/04/2020 15:42:23'], 'Westerbroek')
#           ]

# Order the SoundTrap files in different folders
# hydrophone = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)
# asa = acoustic_survey.ASA(hydrophone=hydrophone, folder_path=folder_path29, zipped=zipped, include_dirs=include_dirs)
# asa.cut_and_place_files_periods(periods29, extensions=['.accel.csv', '.temp.csv', '.log.xml'])

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

# stations27 = ['Nautica Ena', 'Loreley', 'Renilde', 'Noordster', 'Paragon', 'Coast']
# stations29 = ['Heinkel 111', 'HMS Colsay', 'Lola', 'Westerbroek']
# stations_bk29 = ['Heinkel 111', 'HMS Colsay', 'Lola']
# stations_bk04 = ['Buitenratel Panton', 'Killmore', 'Westhinder']
# stations_bk06 = ['Bellwind reefballs', 'CPower']
# stations_bk07 = ['Garden City']
# stations_bk08 = ['G-88', 'Loreley', 'Nautica Ena', 'Senator']

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6 

# SURVEY PARAMETERS
nfft = 1024
binsize = 60.0
h = 0.1
percentiles = [10, 50, 90]
period = None
band = [10, 1600]


save_folder_st = 'C:/Users/cleap/Documents/PhD/COVID-19/Filtered ST'
save_folder_bk = 'C:/Users/cleap/Documents/PhD/COVID-19/B&K'



if __name__ == "__main__":
    """
    Cut all the files according to the periods
    """
    # df_rms = pd.DataFrame(columns=['station', 'rms', 'low_freq', 'high_freq'])
    
    # # Day 29/05/2020
    # for station in stations29: 
    #   station_path = os.path.join(st_folder29, station)
    #   asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path=station_path, zipped=zipped, include_dirs=include_dirs, \
    #                             binsize=binsize, nfft=nfft, period=period, band=band)
    #   df = asa.evolution('rms', dB=True)
    #   df['station'] = station
    #   df['low_freq'] = band[0]
    #   df['high_freq'] = band[1]
    #   df_rms = df_rms.append(df, sort=False)
    
    # for i, station in enumerate(stations_bk04): 
    #   station_path = os.path.join(bk_folder04, station)
    #   ref_file = os.path.join(bk_folder04, station + '_ref.wav')
    #   amplif = bk_amplif[i]
    #   bk.amplif = amplif
    #   bk.update_calibration(ref_file)
    #   asa = acoustic_survey.ASA(hydrophone=bk, folder_path=station_path, zipped=zipped, include_dirs=include_dirs, \
    #                             binsize=binsize, nfft=nfft, period=period, band=band)
    #   df = asa.evolution('rms', dB=True)
    #   df['station'] = station
    #   df['low_freq'] = band[0]
    #   df['high_freq'] = band[1]
    #   df_rms = df_rms.append(df, sort=False)

    
    for i, station in enumerate(stations_bk06): 
      station_path = os.path.join(bk_folder06, station)
      ref_file = os.path.join(bk_folder06, station + '_ref.wav')
      amplif = bk_amplif[i]
      bk.amplif = amplif
      bk.update_calibration(ref_file)
      asa = acoustic_survey.ASA(hydrophone=bk, folder_path=station_path, zipped=zipped, include_dirs=include_dirs, \
                                binsize=binsize, nfft=nfft, period=period, band=band)
      df = asa.evolution('rms', dB=True)
      df['station'] = station
      df['low_freq'] = band[0]
      df['high_freq'] = band[1]
      df_rms = df_rms.append(df, sort=False)

    for i, station in enumerate(stations_bk07): 
      station_path = os.path.join(bk_folder07, station)
      ref_file = os.path.join(bk_folder07, station + '_ref.wav')
      amplif = bk_amplif[i]
      bk.amplif = amplif
      bk.update_calibration(ref_file)
      asa = acoustic_survey.ASA(hydrophone=bk, folder_path=station_path, zipped=zipped, include_dirs=include_dirs, \
                                binsize=binsize, nfft=nfft, period=period, band=band)
      df = asa.evolution('rms', dB=True)
      df['station'] = station
      df['low_freq'] = band[0]
      df['high_freq'] = band[1]
      df_rms = df_rms.append(df, sort=False)

    for i, station in enumerate(stations_bk08): 
      station_path = os.path.join(bk_folder08, station)
      ref_file = os.path.join(bk_folder08, station + '_ref.wav')
      amplif = bk_amplif[i]
      bk.amplif = amplif
      bk.update_calibration(ref_file)
      asa = acoustic_survey.ASA(hydrophone=bk, folder_path=station_path, zipped=zipped, include_dirs=include_dirs, \
                                binsize=binsize, nfft=nfft, period=period, band=band)
      df = asa.evolution('rms', dB=True)
      df['station'] = station
      df['low_freq'] = band[0]
      df['high_freq'] = band[1]
      df_rms = df_rms.append(df, sort=False)
    
    # geoloc = geolocation.SurveyLocation(gps)
    # geoloc.plot_survey_color(column='rms', units='dB', df=df_station, map_file=map_file)
  
    # for station in stations29: 
    #   station_path = os.path.join(folder_path29, station)
    #   save_path = os.path.join(save_folder_st, station + '_ST_10highpass_SPD')
    #   asa = acoustic_survey.ASA(hydrophone=hydrophone, folder_path=station_path, zipped=zipped, include_dirs=include_dirs, \
    #                             binsize=binsize, nfft=nfft, period=period, band=band)
    #   # asa.plot_all_files('plot_spectrogram')
    #   # asa.plot_rms_evolution()
    #   asa.plot_spd(h=h, percentiles=percentiles, save_path=save_path)
    
    
    # for i, station in enumerate(stations_bk07):
    #   amplif = bk_amplif[i]
    #   station_path = os.path.join(bk_folder, station)
    #   ref_file = os.path.join(bk_folder, station + '_ref.wav')
    #   save_path = os.path.join(save_folder_bk, station + '_BK_SPD')
    #   hydrophone = pyhy.BruelKjaer(name=bk_name, model=bk_model, amplif=amplif)
      # hydrophone.update_calibration(ref_file)
      # asa = acoustic_survey.ASA(hydrophone=hydrophone, folder_path=station_path, zipped=zipped, include_dirs=include_dirs, \
      #                         binsize=binsize, nfft=nfft, period=None, band=None)
      # asa.plot_spd(h=h, percentiles=percentiles, save_path=save_path)
      # asa.apply_to_all('find_calibration_tone', max_duration=120, freq=159, min_duration=5.0)
      # asa.apply_to_all('cut_calibration_tone', max_duration=100, freq=159, save_path=ref_file)

