import pathlib
import pandas as pd
import pyhydrophone as pyhy

from pypam import acoustic_survey


# Data Information
data_folder = pathlib.Path('C:/Users/cleap/Documents/PhD/Classifiers/sounnscapes/Data/recordings')
zipped = False
include_dirs = False


# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V              
bk_model = 'Nexus'
bk_name = 'B&K'
amplif0 = 10e-3
bk = pyhy.BruelKjaer(name=bk_name, model=bk_model, amplif=amplif0, serial_number=1)


# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6 

# SURVEY PARAMETERS
nfft = 1024
binsize = 10.0
# h = 0.1
# percentiles = [10, 50, 90]
# period = None
band = None


def separate_ref_signals():
    """
    Run through all the data and cut and store the ref signals
    """
    metadata = pd.read_csv(data_folder.joinpath('metadata.csv'))
    metadata = metadata.set_index('Location')
    for shipwreck_path in data_folder.glob('**/*/'):
        if shipwreck_path.is_dir():
            shipwreck_name = shipwreck_path.parts[-1]
            shipwreck_metadata = metadata.loc[shipwreck_name]
            if shipwreck_metadata['Instrument'] != 'SoundTrap':
                asa = acoustic_survey.ASA(bk, folder_path=shipwreck_path)
                asa.apply_to_all('cut_calibration_tone', max_duration=120, freq=159.0, min_duration=10.0,
                                 save_path=data_folder.joinpath(shipwreck_name+'_ref.wav'))
    

if __name__ == "__main__":
    """
    Clean the B&K data
    """
    separate_ref_signals()
