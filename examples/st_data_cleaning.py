import os
import pathlib
import pandas as pd
import pyhydrophone as pyhy


from pypam import acoustic_survey, geolocation



# Sound Data
st_folder27 = pathlib.Path('//archive/other_platforms/soundtrap/2020/COVID-19/200427 Zeekat/67416073.200427')
st_folder29 = pathlib.Path('//archive/other_platforms/soundtrap/2020/COVID-19/200429 Zeekat/67416073.200429')

zipped = False
include_dirs = False


# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V              
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)



def cut_and_separate_files(folder_path, hydrophone):
    hydrophone = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)
    asa = acoustic_survey.ASA(hydrophone=hydrophone, folder_path=folder_path, zipped=zipped, include_dirs=include_dirs)
    metadata = pd.read_csv(folder_path.joinpath('metadata.csv'))
    for index in metadata.index:
        row = metadata.iloc[index]
        folder_name = row['Location']
        period = (row['start'], row['stop'])
        asa.cut_and_place_files_period(period=period, folder_name=folder_name, extensions=['.accel.csv', '.temp.csv', '.log.xml'])



if __name__ == "__main__":
    """
    Order the SoundTrap files in different folders
    """
    cut_and_separate_files(st_folder27, soundtrap)
    # cut_and_separate_files(st_folder29, soundtrap)

