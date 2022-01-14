import pyhydrophone as pyhy
import pypam


# Soundtrap
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, sensitivity=-172.8, serial_number=serial_number)

# SURVEY PARAMETERS
nfft = 4096
binsize = 30.0
band_lf = [50, 500]
band_hf = [500, 4000]
band_list = [band_lf, band_hf]
features = ['rms', 'peak', 'sel']
# features bi, nsdi ignored because of too low sampling frequency to compute them
third_octaves = None
dc_subtract = True

include_dirs = False
zipped_files = False

min_separation = 1
max_duration = 0.2
threshold = 20
dt = 2.0
detection_band = [500, 1000]


if __name__ == "__main__":
    asa = pypam.ASA(hydrophone=soundtrap, folder_path='./../tests/test_data', binsize=binsize, nfft=nfft,
                    timezone='UTC', include_dirs=include_dirs, zipped=zipped_files, dc_subtract=dc_subtract)
    features_ds = asa.evolution_multiple(method_list=features, band_list=band_list)
    oct_ds = asa.evolution_freq_dom('third_octaves_levels', band=third_octaves, db=True)
