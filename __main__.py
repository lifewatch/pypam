import pyhydrophone

import acoustic_survey


# Sound Analysis
# folder_path = '//fs/shared/transfert/MRC/PAM 20190719'
folder_path = "C:/Users/cleap/Documents/Data/Sound Data/Seiche/AutonautTest"

# Hydrophone Setup
sensitiviy = -196.0
preamp_gain = 0
model = 'uPam'
name = 'Seiche'

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6 


if __name__ == "__main__":
    """
    Run 
    """
    hydrophone = pyhydrophone.seiche.Seiche(name, model, sensitiviy, preamp_gain)
    asa = acoustic_survey.ASA(folder_path=folder_path, hydrophone=hydrophone, ref_pressure=REF_PRESSURE)
    # spl = asa.octavebands_spl_evolution()
    # fdistr = asa.freq_distr_evolution(verbose=True)
    ldistr = asa.level_distr_evolution(verbose=True)

    print(fdistr)
