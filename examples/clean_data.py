import pandas as pd
import pyhydrophone as pyhy

from pypam import acoustic_survey
import argparse


parser = argparse.ArgumentParser(description='Clean the calibration tones')
parser.add_argument('folder_path', metavar='N', type=str, nargs='+', help='folder where the wav files are')
parser.add_argument('hydrophone', metavar='N', type=str, nargs='+', help='Name of the hydrophone')
parser.add_argument('--includedirs', metavar='N', type=int, nargs='+', help='Add if the subfloders have to be added')

args = parser.parse_args()

folder_path = args.folder_path
if args.includedirs:
    include_dirs = True
else:
    include_dirs = False
zipped = False

# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V              
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)


# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
bk_model = 'Nexus'
bk_name = 'B&K'
amplif0 = 10e-3
bk = pyhy.BruelKjaer(name=bk_name, model=bk_model, amplif=amplif0, serial_number=1)


def cut_and_separate_files(folder, hydrophone):
    asa = acoustic_survey.ASA(hydrophone=hydrophone, folder_path=folder, zipped=zipped,
                              include_dirs=include_dirs, utc=False)
    metadata = pd.read_csv(folder.joinpath('metadata.csv'))
    for index in metadata.index:
        row = metadata.iloc[index]
        folder_name = row['Location']
        period = (row['start'], row['stop'])
        asa.cut_and_place_files_period(period=period, folder_name=folder_name,
                                       extensions=['.accel.csv', '.temp.csv', '.log.xml'])


def separate_ref_signals(folder):
    """
    Run through all the data and cut and store the ref signals
    """
    metadata = pd.read_csv(folder.joinpath('metadata.csv'))
    metadata = metadata.set_index('Location')
    for shipwreck_path in folder.glob('**/*/'):
        if shipwreck_path.is_dir():
            shipwreck_name = shipwreck_path.parts[-1]
            shipwreck_metadata = metadata.loc[shipwreck_name]
            if shipwreck_metadata['Instrument'] != 'SoundTrap':
                asa = acoustic_survey.ASA(bk, folder_path=shipwreck_path)
                asa.apply_to_all('cut_calibration_tone', max_duration=120, freq=159.0, min_duration=10.0,
                                 save_path=folder.joinpath(shipwreck_name + '_ref.wav'))


if __name__ == "__main__":
    """
    Order the SoundTrap files in different folders
    """
    cut_and_separate_files(folder_path, soundtrap)
