import os
import pyhydrophone as pyhy


from pypam import acoustic_survey


# Sound Analysis
folder_path = "C:/Users/cleap/Documents/Data/Sound Data/Seiche/AutonautTest"
zipped = False
include_dirs = False


# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
sensitivity = -199.841
preamp_gain = 0.0
Vpp = 20.0
model = 'AMARG3'
name = 'JASCO'

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6 

# Decide bin so the nsamples is a power of two
fft_bintime = 1.0


if __name__ == "__main__":
    """
    Run the survey
    """
    hydrophone = pyhy.seiche.Seiche(name, model, sensitivity, preamp_gain, Vpp)
    # hydrophone = pyhy.AmarG3(name, model, sensitivity, preamp_gain, Vpp)
    asa = acoustic_survey.ASA(hydrophone=hydrophone, folder_path=folder_path, zipped=zipped, include_dirs=include_dirs, fft_bintime=fft_bintime)
    asa.plot_rms_evolution()
    asa.plot_power_spectrum_evolution(percentiles=[10,50,90])
    # asa.plot_psd_evolution(percentiles=[10,50,90])
    # asa.plot_all_files('plot_spd', percentiles=[10,50,90])
    # asa.plot_all_files('plot_psd', dB=True, percentiles=[10,50,90])
