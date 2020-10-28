import pathlib
import sqlite3
import datetime
import numpy as np
import pandas as pd
import pyhydrophone as pyhy
import matplotlib.pyplot as plt


from pypam import acoustic_survey
from pypam import acoustic_file

# -------------------------------------------------------------------------------------------
# Recordings information
# -------------------------------------------------------------------------------------------
upam_folder = pathlib.Path('//fs/shared/mrc/E-Equipment/E02-AutonautAdhemar/08 - Missions/'
                           '20200608-PR-E02-AMUC-M002/Data/uPam')
zipped = False
include_dirs = True

# Piling source LOG
# 08/06/2020: 2000 J interval 1 seconds (unsure),
# 09/06/2020: 2000 J interval 5 seconds (unsure)
# 10/06/2020: 2000 J
# Period	            Shot interval (s)
# [08:18:08 08:22:04]	        2
# [08:22:22	08:55:04]	        2
# [09:46:00	09:50:52]	        4
# [09:51:04	10:58:08]         	4
# [10:58:40	13:10:32]       	4
# [13:21:29	13:22:29]       	4
# [13:23:02	13:23:12]       	2
# [13:23:24	13:48:18]       	2


# -------------------------------------------------------------------------------------------
# Autonaut data
# -------------------------------------------------------------------------------------------
db_path = pathlib.Path('//fs/shared/mrc/E-Equipment/E02-AutonautAdhemar/08 - Missions/20200608-PR-E02-AMUC-M002/'
                       'Data/uPAM/AMUC002.sqlite3')
conn = sqlite3.connect(db_path).cursor()
query = conn.execute('SELECT * FROM gpsData')
cols = [column[0] for column in query.description]
gps_df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
gps_df['UTC'] = pd.to_datetime(gps_df.UTC)
conn.close()

# Pamguard detections
pamguard_output = pathlib.Path("//fs/shared/mrc/P-Projects/02 PC-Commercial/PC1902-AMUC/05 projectverloop/AMUC M002/"
                               "Acoustic Measurements/PAMGuard/PAMdb/amuc_piling_detection_20201027_2.sqlite3")
conn = sqlite3.connect(pamguard_output).cursor()
query = conn.execute('SELECT * FROM Filtered_Noise_measurement_pulses')
cols = [column[0] for column in query.description]
df_pamguard_pulses = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
df_pamguard_pulses.UTC = pd.to_datetime(df_pamguard_pulses.UTC)
query = conn.execute('SELECT * FROM Seismic_Veto')
cols = [column[0] for column in query.description]
df_pamguard_veto = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
df_pamguard_veto.UTC = pd.to_datetime(df_pamguard_veto.UTC)


# -------------------------------------------------------------------------------------------
# Manual selected files
# -------------------------------------------------------------------------------------------
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


# -------------------------------------------------------------------------------------------
# Output files
# -------------------------------------------------------------------------------------------
output_folder = pathlib.Path("//fs/shared/mrc/P-Projects/02 PC-Commercial/PC1902-AMUC/05 projectverloop/AMUC M002/"
                             "Acoustic Measurements/pypam")

# -------------------------------------------------------------------------------------------
# Hydrophone Settings
# -------------------------------------------------------------------------------------------
# uPam
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


# -------------------------------------------------------------------------------------------
# Detector information
# -------------------------------------------------------------------------------------------
min_separation = 1
max_duration = 0.2
threshold = 15
dt = 0.5


# -------------------------------------------------------------------------------------------
# Acoustic ANALYSIS PARAMETERS
# -------------------------------------------------------------------------------------------
binsize = None
nfft = 2048
band = [20, 10000]
percentiles = [10, 50, 90]
period = None


if __name__ == "__main__":
    """
    Perform the AMUC acoustic survey
    """
    # -------------------------------------------------------------------------------------------
    # Event detection and analysis
    # -------------------------------------------------------------------------------------------
    asa = acoustic_survey.ASA(hydrophone=upam, folder_path=upam_folder, zipped=zipped, include_dirs=include_dirs,
                              binsize=60.0, period=period)
    df = asa.detect_piling_events(max_duration=max_duration, min_separation=min_separation, threshold=threshold, dt=dt)
    df[['rms', 'sel', 'peak']].plot(subplots=True, marker='.', linestyle='')
    plt.show()
    detection_output = output_folder.joinpath('piling_detection_th%s_sep%s_dt%s_dur%s.csv' %
                                              (threshold, min_separation, dt, max_duration))
    df.to_csv(detection_output)
    print('Piling detected!')

    # -------------------------------------------------------------------------------------------
    # Detection analysis
    # -------------------------------------------------------------------------------------------
    df_pypam = pd.read_csv(detection_output)
    df_pypam.datetime = pd.to_datetime(df_pypam.datetime)
    df_pypam_days = [group[1] for group in df_pypam.groupby(df_pypam.datetime.dt.date)]
    df_pamguard_pulses_days = [group[1] for group in df_pamguard_pulses.groupby(df_pamguard_pulses.UTC.dt.date)]
    df_pamguard_veto_days = [group[1] for group in df_pamguard_veto.groupby(df_pamguard_veto.UTC.dt.date)]

    for i in np.arange(len(df_pypam_days)):
        day = df_pypam_days[i].head(1).datetime.dt.date.values[0]
        df_pypam_day = df_pypam_days[i].assign(label=1)
        df_pg_pulses = df_pamguard_pulses_days[i].assign(label=2)
        df_pg_veto = df_pamguard_veto_days[i].assign(label=3)
        print('Number of sparks detected in day %s' % day)
        print('pypam: %s' % len(df_pypam_day))
        print('PAMGuard Pulses: %s' % len(df_pg_pulses))
        print('PAMGuard Veto: %s' % len(df_pg_veto))
        ax = df_pypam_day.plot(x='datetime', y='label', marker='.', linestyle='', label='pypam')
        df_pg_pulses.plot(x='UTC', y='label', marker='.', linestyle='', ax=ax, label='PAMGuard pulses')
        df_pg_veto.plot(x='UTC', y='label', marker='.', linestyle='', ax=ax, label='PAMGuard veto')
        plt.title('Detections of %s' % day)
        plt.gcf().autofmt_xdate()
        plt.xlabel('Time')
        plt.legend()
        plt.show()

    print('hello!')
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
