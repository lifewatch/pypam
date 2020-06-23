import os
import pyhydrophone as pyhy
import matplotlib.pyplot as plt


from pypam import acoustic_survey
from pypam import piling_detector
from pypam import geolocation



# Recordings information 
upam_folder = 'C:/Users/cleap/Documents/Data/Sound Data/uPam/AMUC'
zipped = False
include_dirs = False

# Hydrophone Setup
model = 'uPam'
name = 'Seiche'
serial_number = 'SM7213' # Should it be added?
sensitivity = -196.0
preamp_gain = 40
Vpp = 12.0
upam = pyhy.Seiche(name=name, model=model, sensitivity=sensitivity, preamp_gain=preamp_gain, Vpp=Vpp)


# Detector information
min_duration = 0.1
ref = -6
threshold = 140
dt = 0.05
continuous = False


# Survey parameters
# SURVEY PARAMETERS
binsize = 120.0


# GPS information 
gps = "C:/Users/cleap/Documents/Data/Tracks/COVID-19/Track_2020-05-18 092212.gpx"
geoloc = geolocation.SurveyLocation(gps)

lat = 51.0
lon = 2.0


if __name__ == "__main__":
    """
    Cut all the files according to the periods
    """
    piling_d = piling_detector.PilingDetector(min_duration=min_duration, ref=ref, threshold=threshold, dt=dt, continuous=continuous)
    asa = acoustic_survey.ASA(hydrophone=upam, folder_path=upam_folder, zipped=zipped, include_dirs=include_dirs, binsize=binsize)
    df = asa.evolution('detect_events', detector=piling_d)

    df[['rms', 'sel', 'peak']].plot(subplots=True, marker='.', linestyle='')

    plt.show()

