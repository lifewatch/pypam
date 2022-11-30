import pyhydrophone as pyhy
import matplotlib.pyplot as plt
import time
import numpy as np

from pypam.acoustic_file import AcuFile
import pypam.utils as utils

"""
This script is meant to reproduce the standards described in 
Ocean Sound Analysis Software for Making Ambient Noise Trends Accessible (MANTA)
(https://doi.org/10.3389/fmars.2021.703650 , https://doi.org/10.1121/10.0003324)
for one wav file. 
"""

# If the model is not implemented yet in pyhydrophone, a general Hydrophone can be defined
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

p_ref = 1.0
fs = 8000
nfft = fs
fft_overlap = 0.5
binsize = 60.0
bin_overlap = 0
method = 'density'
band = [0, 4000]

wav_path = '../tests/test_data/67416073.210610033655.wav'
acu_file = AcuFile(sfile=wav_path, hydrophone=soundtrap, p_ref=p_ref, timezone='UTC', channel=0, calibration=None,
                   dc_subtract=False)

# record start time
start = time.time()
millis = acu_file.hybrid_millidecade_bands(nfft=nfft, fft_overlap=fft_overlap, binsize=binsize, bin_overlap=bin_overlap,
                                           db=False, method=method, band=band)
# record end time
end = time.time()

# print the difference between start and end time in milli secs
print("The time of execution of the millidecades is :", (end - start), "s")

fig, ax = plt.subplots()
ax.plot(10*np.log10(millis['band_density'].mean('id')), label='pypam')
plt.legend()
plt.show()

# Convert it to pandas to save to csv
millis_pd = millis['band_density'].to_pandas()
millis_pd_db = 10 * np.log10(millis_pd)


# Save to csv
millis_pd.to_csv('../tests/test_data/data_exploration/1_min_1_sec_0.5_overlap.csv')
millis_pd_db.to_csv('../tests/test_data/data_exploration/1_min_1_sec_0.5_overlap_db.csv')

# Save to netcdf
millis.to_netcdf('../tests/test_data/data_exploration/1_min_1_sec_0.5_overlap.nc')
