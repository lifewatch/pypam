from pypam.acoustic_survey import ASA
import pyhydrophone as pyhy


# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 4096
binsize = 60.0
band_lf = [50, 500]
band_mf = [500, 2000]
band_hf = [2000, 20000]
band_list = [band_lf, band_mf, band_hf]
features = ['rms', 'sel', 'aci', 'sh']
third_octaves = True

include_dirs = False
zipped_files = False


if __name__ == "__main__":
    asa = ASA(hydrophone=soundtrap, folder_path='./../data', binsize=binsize, nfft=nfft, utc=True,
              include_dirs=include_dirs, zipped=zipped_files)
