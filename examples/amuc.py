import pathlib
import sqlite3
import datetime
import pandas as pd
import pyhydrophone as pyhy
import matplotlib.pyplot as plt


from pypam import acoustic_survey
from pypam import acoustic_file


# Recordings information 
upam_folder = pathlib.Path('//fs/shared/mrc/E-Equipment/E02-AutonautAdhemar/08 - Missions/20200608-PR-E02-AMUC-M002/Data/uPAM/20200610')
zipped = False
include_dirs = False


# # Autonaut information
# db_path = '//fs/SHARED/transfert/MRC-MOC/uPAM/AMUC002.sqlite3'
# conn = sqlite3.connect(db_path).cursor()
# query = conn.execute('SELECT * FROM gpsData')
# cols = [column[0] for column in query.description]
# gps_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
# gps_df['UTC'] = pd.to_datetime(gps_df.UTC)

# Specific files
blast1_file = 'C:/Users/cleap/Documents/Data/Sound Data/uPam/AMUC/Selected sounds/blastAMUC002_20200608_122933_792.wav'
noblast1_file = 'C:/Users/cleap/Documents/Data/Sound Data/uPam/AMUC/Selected sounds/' \
                'noblastAMUC002_20200608_122933_792.wav'

blast3_file = 'C:/Users/cleap/Documents/Data/Sound Data/uPam/AMUC/Selected sounds/blastAMUC002_20200610_095601_535.wav'
noblast3_file = 'C:/Users/cleap/Documents/Data/Sound Data/uPam/AMUC/Selected sounds/' \
                'noblastAMUC002_20200610_095601_535.wav'

usv_example1 = '//fs/SHARED/transfert/MRC-MOC/uPAM/20200608/AMUC002_20200608_121222_520.wav'
usv_example2 = '//fs/SHARED/transfert/MRC-MOC/uPAM/20200609/AMUC002_20200609_090042_341.wav'
usv_example3 = '//fs/SHARED/transfert/MRC-MOC/uPAM/20200610/AMUC002_20200610_134710_036.wav'
usv_examples = [usv_example1, usv_example2, usv_example3]

usv_air = '//fs/SHARED/transfert/MRC-MOC/AMUC/Auxiliar engine in air MRC 20200708.wav'


# Hydrophone Settings
model = 'uPam'
name = 'Seiche'
serial_number = 'SM7213'
sensitivity = -196.0
preamp_gain = 0.0
Vpp = 20.0
upam = pyhy.Seiche(name=name, model=model, sensitivity=sensitivity, serial_number=serial_number,
                   preamp_gain=preamp_gain, Vpp=Vpp)

# Soundtrap
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)


# Detector information
min_duration = 0.1
ref = -6
threshold = 140
dt = 0.05
continuous = False


# ANALYSIS PARAMETERS
binsize = None
nfft = 2048
band = [20, 10000]
percentiles = [10, 50, 90]
period = ['2020-06-10 09:45:00', '2020-06-10 11:00:00']


if __name__ == "__main__":
    """
    Perform the AMUC acoustic survey
    """
    # -------------------------------------------------------------------------------------------
    # Event detection and analysis
    # -------------------------------------------------------------------------------------------
    asa = acoustic_survey.ASA(hydrophone=upam, folder_path=upam_folder, zipped=zipped, include_dirs=include_dirs,
                              binsize=60.0, period=period)
    df = asa.detect_piling_events(max_duration=0.3, min_separation=1.0, threshold=20, dt=0.5)
    df[['rms', 'sel', 'peak']].plot(subplots=True, marker='.', linestyle='')
    plt.show()
    print('hello!')

    # for folder in upam_folder.glob('**/*/'):
    #     asa = acoustic_survey.ASA(hydrophone=upam, folder_path=folder, zipped=zipped, include_dirs=include_dirs,
    #                               binsize=60.0)
    #     df = asa.detect_ship_events(min_duration=0.1, threshold=160)
    #     df[['rms', 'sel', 'peak']].plot(subplots=True, marker='.', linestyle='')
    #     plt.show()
    #     print('hello!')

    # -------------------------------------------------------------------------------------------
    # Blast vs No blast sound analysis
    # -------------------------------------------------------------------------------------------
    # blast1 = acoustic_file.HydroFile(sfile=blast1_file, hydrophone=upam, p_ref=1.0, band=band)
    # noblast1 = acoustic_file.HydroFile(sfile=noblast1_file, hydrophone=upam, p_ref=1.0, band=band)
    #
    # blast1_ps = blast1.power_spectrum(db=True, nfft=nfft, percentiles=percentiles)
    # noblast1_ps = noblast1.power_spectrum(db=True, nfft=nfft, percentiles=percentiles)
    #
    # fbands = blast1_ps['band_spectrum'].columns
    #
    # plt.figure()
    # plt.plot(fbands, blast1_ps['band_spectrum'][fbands].values[0], label='Blast')
    # plt.plot(fbands, noblast1_ps['band_spectrum'][fbands].values[0], label='No blast')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('SPLrms [db]')
    # plt.title('Day 1 examples Power Spectrum')
    # plt.legend()
    # plt.xscale('log')
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    #
    # diff1 = blast1_ps['band_spectrum'][fbands].values[0] - noblast1_ps['band_spectrum'][fbands].values[0]
    #
    # blast3 = acoustic_file.HydroFile(sfile=blast3_file, hydrophone=upam, p_ref=1.0, band=band)
    # noblast3 = acoustic_file.HydroFile(sfile=noblast3_file, hydrophone=upam, p_ref=1.0, band=band)
    #
    # blast3_ps = blast3.power_spectrum(db=True, nfft=nfft, percentiles=percentiles)
    # noblast3_ps = noblast3.power_spectrum(db=True, nfft=nfft, percentiles=percentiles)
    #
    # fbands = blast3_ps['band_spectrum'].columns
    #
    # plt.figure()
    # plt.plot(fbands, blast3_ps['band_spectrum'][fbands].values[0], label='Blast')
    # plt.plot(fbands, noblast3_ps['band_spectrum'][fbands].values[0], label='No blast')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('SPLrms [db]')
    # plt.title('Day 3 examples Power Spectrum')
    # plt.legend()
    # plt.xscale('log')
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    #
    # diff3 = blast3_ps['band_spectrum'][fbands].values[0] - noblast3_ps['band_spectrum'][fbands].values[0]
    #
    # fig, ax = plt.subplots(2, 1, sharex='all')
    # ax[0].plot(fbands, diff1, label='Day 1')
    # # ax[0].set_xlabel('Frequency [Hz]')
    # ax[0].set_title('SNR Blast')
    # ax[0].set_ylabel('SNR [db]')
    # ax[0].hlines(y=6, xmin=fbands.min(), xmax=fbands.max(), label='6 db')
    # ax[0].legend()
    # ax[1].plot(fbands, diff3, label='Day 3')
    # ax[1].set_xlabel('Frequency [Hz]')
    # ax[1].set_ylabel('SNR [db]')
    # ax[1].hlines(y=6, xmin=fbands.min(), xmax=fbands.max(), label='6 db')
    # ax[1].legend()
    # # ax[1].set_title('SNR Blast Day 3')
    # plt.xscale('log')
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    #
    # plt.figure()
    # plt.plot(fbands, diff1, label='Day 1')
    # plt.hlines(y=6, xmin=fbands.min(), xmax=fbands.max(), label='6 db')
    # plt.title('SNR Blast Day 1')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('SNR [db]')
    # plt.xscale('log')
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
    # plt.close()
    #
    # plt.figure()
    # plt.plot(fbands, diff3, label='Day 3')
    # plt.hlines(y=6, xmin=fbands.min(), xmax=fbands.max(), label='6 db')
    # plt.title('SNR Blast Day 3')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('SNR [db]')
    # plt.xscale('log')
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
    # plt.close()
    #
    # # -------------------------------------------------------------------------------------------
    # # USV noise correlation
    # # -------------------------------------------------------------------------------------------
    # usv_sound = acoustic_file.HydroFile(sfile=usv_air, hydrophone=soundtrap, p_ref=1.0, band=band)
    # _, _, t_air, Sxx_air = usv_sound.spectrogram(db=True, nfft=nfft, binsize=binsize)
    # for i, usv_i in enumerate(usv_examples):
    #     usv = acoustic_file.HydroFile(sfile=usv_i, hydrophone=upam, p_ref=1.0, band=band)
    #     start_time, fbands, t, Sxx = usv.spectrogram(db=True, nfft=nfft, binsize=binsize)
    #     end_time = start_time[0] + datetime.timedelta(seconds=t[-1])
    #
    #     gps_dfi = gps_df[(gps_df.UTC >= start_time[0]) & (gps_df.UTC <= end_time)]
    #
    #     fig, ax = plt.subplots(1, 2, figsize=[15, 5], gridspec_kw={'width_ratios': [2, 1]})
    #     im = ax[0].pcolormesh(t, fbands, Sxx[0])
    #     ax[0].set_title('Spectrogram and USV Speed example of Day %s' % (i+1))
    #     ax[0].set_xlabel('Time [s]')
    #     ax[0].set_ylabel('Frequency [Hz]')
    #     # ax[0].set_yscale('log')
    #     # cbar = fig.colorbar(im, cax=ax[0], orientation='vertical')
    #     # cbar.set_label('SPLrms [db re 1.0 uPa]', rotation=90)
    #
    #     ax2 = ax[0].twinx()
    #     ax2.plot((gps_dfi.UTC - gps_dfi.UTC.iloc[0]).dt.total_seconds(), gps_dfi.Speed, label='USV speed')
    #     ax2.legend()
    #     ax2.set_ylabel('Speed [m/s]')
    #
    #     ax[1].pcolormesh(t_air, fbands, Sxx_air[0])
    #     ax[1].set_title('Spectrogram of USV engine in air')
    #     ax[1].set_xlabel('Time [s]')
    #     ax[1].set_ylabel('Frequency [Hz]')
    #     # ax[1].set_yscale('log')
    #
    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()
