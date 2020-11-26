from pypam import acoustic_survey

import pyhydrophone as pyhy

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('folder_path', metavar='N', type=str, nargs='+',
                    help='folder where the wav files are')
parser.add_argument('hydrophone', metavar='N', type=str, nargs='+',
                    help='Name of the hydrophone')
parser.add_argument('--includedirs', metavar='N', type=int, nargs='+',
                    help='Add if the subfloders have to be added')

args = parser.parse_args()

folder_path = args['folder_path']
if args['includedirs']:
    include_dirs = True
else:
    include_dirs = False
zipped = False

# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
st_model = 'ST300HF'
st_name = 'SoundTrap'
st_serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=st_name, model=st_model, serial_number=st_serial_number)

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 4096
binsize = None


if __name__ == "__main__":
    asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path=folder_path, zipped=zipped, include_dirs=include_dirs,
                              binsize=binsize, nfft=nfft)
    df = asa.detect_ship_events(min_duration=10.0, threshold=120.0)
    df.to_csv('ship_detections.csv')
