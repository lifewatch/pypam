import pyhydrophone as pyhy
from pypam import acoustic_survey

# Soundtrap
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

# Analysis parameters
features = ['rms', 'sel', 'peak', 'aci']
band_list = [[10, 100], [500, 1000], [500, 100000]]
third_octaves = None  # Calculate third octaves for the entire freq range

if __name__ == "__main__":
    asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path='./../tests/test_data', binsize=60.0)
    features_ds = asa.evolution_multiple(method_list=features, band_list=band_list)
    oct_ds = asa.evolution_freq_dom('third_octaves_levels', band=third_octaves, db=True)
