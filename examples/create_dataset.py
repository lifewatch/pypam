import pathlib

import pyhydrophone as pyhy
import pypam

# Acoustic Data
summary_path = pathlib.Path('./../tests/test_data/data_summary.csv')

# Output folder
output_folder = summary_path.parent.joinpath('data_exploration')

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

upam_model = 'uPam'
upam_name = 'Seiche'
upam_serial_number = 'SM7213'
upam_sensitivity = -196.0
upam_preamp_gain = 0.0
upam_Vpp = 20.0
upam = pyhy.Seiche(name=upam_name, model=upam_name, serial_number=upam_serial_number, sensitivity=upam_sensitivity,
                   preamp_gain=upam_preamp_gain, Vpp=upam_Vpp)


instruments = {'SoundTrap': soundtrap, 'uPam': upam, 'B&K': bk}

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 4096
binsize = 60.0
band_lf = [50, 500]
band_mf = [500, 2000]
band_hf = [2000, 20000]
band_list = [band_lf]
features = ['rms', 'sel', 'aci']
third_octaves = None

if __name__ == "__main__":
    # Create the dataset object
    ds = pypam.DataSet(summary_path, output_folder, instruments, features, third_octaves, band_list, binsize,
                       nfft)
    # Call the dataset creation. Will create the files in the corresponding folder
    ds()
