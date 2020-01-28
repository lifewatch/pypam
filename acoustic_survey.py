import math 
import numpy as np
import matplotlib.pyplot as plt 
import os
import configparser
import pandas as pd
from datetime import datetime, timedelta
import acoustics
import soundfile as sf
import scipy.signal as sig

import pyhydrophone.surveyfile as sfile


class ASA:
    def __init__(self, folder_path, hydrophone, ref_pressure):
        """ 
        Init a AcousticSurveyAnalysis (ASA)
        """
        self.folder_path = folder_path
        self.hydrophone = hydrophone
        self.ref_pressure = ref_pressure


    def octavebands_spl_evolution(self):
        """ 
        Read the octave-band level evolution.
        """
        # Create a df with all the results
        bands = acoustics.standards.iec_61672_1_2013.NOMINAL_OCTAVE_CENTER_FREQUENCIES
        bands_evolution = pd.DataFrame(columns=np.concatenate((['datetime'], bands)))
        bands_evolution = bands_evolution.set_index('datetime')
        
        for file_name in os.listdir(self.folder_path):
            extension = file_name.split(".")[-1]
            if extension == 'wav':
                print(file_name)
                file_path = os.path.join(self.folder_path, file_name)
                # Read the wav file and compute the periodogram 
                sound_file = sfile.AudioFile(wavfile_path=file_path, hydrophone=self.hydrophone, p_ref=self.ref_pressure/1e6)
                # freq, bands_level = sound_file.signal_uPa.octaves()

                # Start analyzing the data
                bands_evolution.loc[sound_file.date] = bands_level.mean()

        # Plot the evolution 
        plt.figure()    
        bands_evolution.plot(str(bands))
        plt.title('Noise levels (rms) evolution for band frequency')
        plt.ylabel('rms')
        plt.xlabel('Time')
        plt.legend()
        plt.show()
        plt.close()

    
    def freq_distr_evolution(self, nfft=512, verbose=False):
        """
        Read the frequency distribution amongst time 
        """
        start = True

        for file_name in os.listdir(self.folder_path):
            extension = file_name.split(".")[-1]
            if extension == 'wav':
                print(file_name)
                file_path = os.path.join(self.folder_path, file_name)
                # Read the wav file and compute the periodogram 
                sound_file = sfile.AudioFile(wavfile_path=file_path, hydrophone=self.hydrophone, p_ref=self.ref_pressure/1e6)
                freqs, bands_level_db = sound_file.freq_distr()
                if start:
                    freq_distr_evolution = pd.DataFrame(columns=np.concatenate((['datetime'], freqs[1::])))
                    freq_distr_evolution = freq_distr_evolution.set_index('datetime')
                    freq_distr_evolution.loc[sound_file.date] = bands_level_db[1::]
                    start = False
                else:
                    freq_distr_evolution.loc[sound_file.date] = bands_level_db[1::]

        if verbose:
            # Plot the evolution and the frequency distribution
            fig = plt.figure()
            im = plt.pcolormesh(freq_distr_evolution.index, freqs[1::], np.transpose(freq_distr_evolution[freqs[1::].astype(np.unicode)]))
            plt.title('Frequency distribution evolution')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time')
            plt.yscale('log')
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('Pressure Level [dB]')
            plt.show()
            plt.close()

            fig = plt.figure()
            freq_distr_evolution.boxplot()
            plt.title('Frequency distribution boxplot')
            plt.show()
            plt.close()

        return freq_distr_evolution

    
    def level_distr_evolution(self, verbose=False):
        """
        Return a df with the evolution of the min and the max pressure level of every file
        """
        percentages = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        level_distr_evolution = pd.DataFrame(columns=np.concatenate((['datetime', 'min', 'max'], percentages*100)))
        level_distr_evolution = level_distr_evolution.set_index('datetime')
        for file_name in os.listdir(self.folder_path):
            extension = file_name.split(".")[-1]
            if extension == 'wav':
                print(file_name)
                file_path = os.path.join(self.folder_path, file_name)
                # Read the wav file and compute the periodogram 
                sound_file = sfile.AudioFile(wavfile_path=file_path, hydrophone=self.hydrophone, p_ref=self.ref_pressure/1e6)
                percentages_level = sound_file.perc_distr(percentages)
                level_distr_evolution.loc[sound_file.date] = percentages_level

        if verbose:
            # Plot the evolution   
            level_distr_evolution.plot()
            plt.title('Sound level distribution evolution')
            plt.ylabel('Sound pressure level (SPL) [dB]')
            plt.xlabel('Time')
            plt.show()
            plt.close()
        
        return level_distr_evolution   


    def total_analysis(self, verbose=False):     
        """
        Return a df with the evolution of the min and the max pressure level of every file, and the evolution of frequency distribution
        """
        start = True
        percentages = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        level_distr_evolution = pd.DataFrame(columns=np.concatenate((['datetime', 'min', 'max'], percentages*100)))
        level_distr_evolution = level_distr_evolution.set_index('datetime')
        for file_name in os.listdir(self.folder_path):
            extension = file_name.split(".")[-1]
            if extension == 'wav':
                print(file_name)
                file_path = os.path.join(self.folder_path, file_name)
                # Read the wav file and compute the periodogram 
                sound_file = sfile.AudioFile(wavfile_path=file_path, hydrophone=self.hydrophone, p_ref=self.ref_pressure/1e6)
                percentages_level = sound_file.perc_distr(percentages)
                freqs, bands_level_db = sound_file.freq_distr()
                
                level_distr_evolution.loc[sound_file.date] = percentages_level
                if start:
                    freq_distr_evolution = pd.DataFrame(columns=np.concatenate((['datetime'], freqs[1::])))
                    freq_distr_evolution = freq_distr_evolution.set_index('datetime')
                    freq_distr_evolution.loc[sound_file.date] = bands_level_db[1::]
                    start = False
                else:
                    freq_distr_evolution.loc[sound_file.date] = bands_level_db[1::]
        
        
        if verbose:
            # Plot the evolution   
            level_distr_evolution.plot()
            plt.title('Sound level distribution evolution')
            plt.ylabel('Sound pressure level (SPL) [dB]')
            plt.xlabel('Time')
            plt.show()
            plt.close()

            # Plot the evolution and the frequency distribution
            fig = plt.figure()
            im = plt.pcolormesh(freq_distr_evolution.index, freqs[1::], np.transpose(freq_distr_evolution[freqs[1::].astype(np.unicode)]))
            plt.title('Frequency distribution evolution')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time')
            plt.yscale('log')
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('Pressure Level [dB]')
            plt.show()
            plt.close()

            fig = plt.figure()
            freq_distr_evolution.boxplot()
            plt.title('Frequency distribution boxplot')
            plt.show()
            plt.close()     


        return freq_distr_evolution, level_distr_evolution   