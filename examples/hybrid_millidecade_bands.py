#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Millidecade bands
========================

This script is an example to compute hybrid millidecade bands, save them as netCDF files and use them to plot
"""

# %%
# Create the hydrophone object
import pyhydrophone as pyhy
import pypam

# Soundtrap
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, sensitivity=-172.8, serial_number=serial_number)

# %%
# Set the study parameters

# First, decide band to study. The top frequency should not be higher than the nyquist frequency (sampling rate/2)
band = [0, 4000]

# Then, set the nfft to double the sampling rate. If you want to pass None to your band, that is also an option, but
# then you need to know the sampling frequency to choose the nfft.
nfft = band[1] * 2  # or nfft = 8000

# Set the band to 1 minute
binsize = 60.0


# %%
# Declare the Acoustic Survey
include_dirs = False
zipped_files = False
dc_subtract = True
asa = pypam.ASA(hydrophone=soundtrap, folder_path='./tests/test_data', binsize=binsize, nfft=nfft,
                timezone='UTC', include_dirs=include_dirs, zipped=zipped_files, dc_subtract=dc_subtract)

# Compute the hybrid millidecade bands
# You can choose 'density' or 'spectrum' as a method
milli_psd = asa.hybrid_millidecade_bands(db=True, method='density',
                                         band=band,
                                         percentiles=None)
print(milli_psd['millidecade_bands'])

# Now, with the obtained hybrid millidecade bands, make some plots
milli_psd = milli_psd.drop_vars(['band_density', 'frequency'])
milli_psd = milli_psd.rename({'frequency_bins': 'frequency'})
pypam.plots.plot_spectrum_mean(milli_psd, data_var='millidecade_bands')
