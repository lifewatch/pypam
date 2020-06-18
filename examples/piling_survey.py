import os
import sys
import pyhydrophone as pyhy


from pypam import acoustic_survey


# Analysis of pile stroke measurement data
# Bert De Coensel, INTEC, Ghent University, Belgium (April 2013)
# Update v2 - May 2013
# Update v3 - July 2013 (corrected version)
# Update v4 - August 2013 (continuous recording added)


# Information about the wavfile to process
# -----------------------------------------

# clean 4-channel file that contains only the peaks, not the calibration tone
filename = 'excerpt.wav'

# use 1 for a continuous measurement, and 0 for a pilestroke measurement
continuous = 0

# calibration value of pressure and velocity
cal_hydro = [200.0, 200.0]
cal_acc = [200.0, 200.0]

# reference values for pressure and velocity
# alternatively, a mono wavfile with calibration tone can be supplied to reference_level
ref = [reference_level('cal1.wav'), 
       reference_level('cal2.wav'),
       reference_level('cal3.wav'),
       reference_level('cal4.wav')]

# file with results
outfile = 'results.txt'

# time-step for calculation of level-vs-time
dt = 0.01

# minimum time in between two events
min_duration = 0.5

# a good threshold for detection of events
threshold = 150.0


events = detect_events()


# Step 2: analysis of the individual events
#------------------------------------------

# a small window [t-before, t+after] around each event t is selected
before = 0.20 # time before the start of the event
after = 0.50 # time after the start of the event

# time-step for calculation of spectrogram for individual events
dt = 0.001

# time-step for calculation of spectrogram of full recording
dtfull = 1.0

# finally, analyze full recording and all events, and save results
leqf, specf, tf, ff, spgf = analyze_recording(filename, cal, ref, events, before, after, dt, dtfull, outfile, continuous)


# Step 3 (optional): visualization of results
#--------------------------------------------

# the following function can be used to estimate the best threshold value
# change 0 to 1 in order to plot the number of events for a range of thresholds
if 0:
  plot_events(y, newdt, np.arange(100,220), duration)


# the following functions can be used to visualize the calculation for a single event
# change 0 to 1 in order to plot results for the first event
if 0:
  y, fs = load_event(filename, cal, ref, events[1], before, after)
  sel, spec, t, f, levels, peak = analyze_event(y, fs, dt)
  plot_event(y, fs, sel, spec, t, f, levels, [100.0, 200.0])


# the following functions can be used to visualize the calculation for the full recording
# change 0 to 1 in order to plot spectrogram and spectra
if 0:
  plot_full(leqf, specf, tf, ff, spgf, [100.0, 200.0])




