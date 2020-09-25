import pathlib
import pyhydrophone as pyhy

from pypam import acoustic_survey


# Sound Analysis
data_folder = pathlib.Path('//archive/other_platforms/b&k/COVID-19/20200504_Simon Stevin/Buitenratel')
zipped = False
include_dirs = False


# Hydrophone Setup
bk_model = 'Nexus'
bk_name = 'B&K'
amplif0 = 10e-3
bk = pyhy.BruelKjaer(name=bk_name, model=bk_model, amplif=amplif0, serial_number=1)


# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 4096
binsize = 60.0
# h = 0.1
# percentiles = [10, 50, 90]
# period = None
band_lf = [100, 500]
band_mf = [500, 2000]
band_hf = [2000, 20000]


if __name__ == "__main__":
    """
    Compute the spl of each band for each bin
    """
    asa = acoustic_survey.ASA(hydrophone=bk, folder_path=data_folder, zipped=zipped,
                              include_dirs=include_dirs, binsize=binsize, nfft=nfft)
    # evo = asa.evolution_multiple(method_list=['rms', 'dynamic_range'], band_list=[band_lf, band_mf, band_hf])
    asa.apply_to_all('plot_spectrogram', mode='fast')
