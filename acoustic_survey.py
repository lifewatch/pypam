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
                sound_file = SurveyFile(wavfile_path=file_path, hydrophone=self.hydrophone, p_ref=self.ref_pressure/1e6)
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
                sound_file = SurveyFile(wavfile_path=file_path, hydrophone=self.hydrophone, p_ref=self.ref_pressure/1e6)
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
            fig, ax = plt.subplots(2,1)  
            im = ax[0].pcolormesh(freq_distr_evolution.index, freqs, np.transpose(freq_distr_evolution[freqs[1::].astype(np.unicode)]))
            ax[0].set_title('Frequency distribution evolution')
            ax[0].set_ylabel('Frequency [Hz]')
            ax[0].set_xlabel('Time')
            plt.colorbar(im, ax=ax[0])

            freq_distr_evolution.boxplot(ax=ax[1])
            ax[1].set_title('Frequency distribution boxplot')

            plt.show()
            plt.close()

        return freq_distr_evolution

    
    def level_distr_evolution(self, verbose=False):
        """
        Return a df with the evolution of the min and the max pressure level of every file
        """
        percentages = [0.1, 0.25, 0.5, 0.75, 0.9]
        level_distr_evolution = pd.DataFrame(columns=np.concatenate((['datetime', 'min', 'max'], percentages*100)))
        level_distr_evolution = level_distr_evolution.set_index('datetime')
        for file_name in os.listdir(self.folder_path):
            extension = file_name.split(".")[-1]
            if extension == 'wav':
                print(file_name)
                file_path = os.path.join(self.folder_path, file_name)
                # Read the wav file and compute the periodogram 
                sound_file = SurveyFile(wavfile_path=file_path, hydrophone=self.hydrophone, p_ref=self.ref_pressure/1e6)
                percentages_level = sound_file.perc_distr(percentages)
                level_distr_evolution.loc[sound_file.date] = percentages_level

        if verbose:
            # Plot the evolution 
            plt.figure()    
            freq_distr_evolution.plot()
            plt.title('Sound level distribution evolution')
            plt.ylabel('Sound pressure level (SPL) [dB]')
            plt.xlabel('Time')
            plt.colorbar()
            plt.show()
            plt.close()
        
        return freq_distr_evolution        



class SurveyFile:
    def __init__(self, wavfile_path, hydrophone, p_ref=1.0):
        """
        Data recorded in a wav file.
        Hydrophone has to be from the class hydrphone 
        p_ref in uPa
        """
        # Save hydrophone model 
        self.hydrophone = hydrophone 

        # Get the date from the name
        file_name = os.path.split(wavfile_path)[-1]
        self.date = hydrophone.get_name_date(file_name)

        # Signal
        self.wavfile_path = wavfile_path
        signal, self.fs = sf.read(self.wavfile_path)
        
        # Calculate the mean of the two channels to do the analysis
        self.signal = signal.mean(axis=-1)
        self.signal_uPa = self.wav2uPa(self.signal)

        self.p_ref = 1.0


    def wav2uPa(self, wav):
        """ 
        Compute the pressure from the wav signal 
        """
        Mv = 10 ** (self.hydrophone.sensitivity / 20.0)
        return wav / Mv


    def wav2dB(self, wav):
        """ 
        Compute the dB SPL from the wav signal
        """
        return 20*np.log10(wav) - self.hydrophone.sensitivity

    
    def dB2uPa(self, signal_dB):
        """
        Compute the uPa from the dB signals
        """
        return np.power(10, signal_dB / 20.0)
    

    def uPa2dB(self):
        """ 
        Compute the dB from the uPa signal
        """
        return 20*np.log10(self.signal_uPa / self.p_ref)

    
    def freq_distr(self, nfft=512):
        """
        Read the frequency distribution
        """
        window = sig.get_window('hann', nfft)

        # PS is the freq power spectrum in [uPa**2]
        freq, ps = sig.periodogram(self.signal_uPa, nfft=nfft, fs=self.fs, scaling='spectrum')

        ps_db = 10*np.log10(ps/self.p_ref) 

        return freq, ps_db

    
    def rms_db(self):
        """
        Return the mean squared value of the signal in db
        """
        return 10*np.log10((self.signal**2).mean()/self.p_ref)

    
    def perc_distr(self, percentages):
        """
        Return the sound level where a percentage of the samples is 
        """
        min_lev = self.signal_uPa.min()
        max_lev = self.signal_uPa.max()

        quantiles = np.quantile(self.signal_uPa, percentages)

        lims = np.concatenate(([min_lev, max_lev], quantiles))
        lims_db = 20*np.log10(lims / self.p_ref)
        
        return lims_db


    def plot_sound_spectrogram(self, ax=None):
        """ 
        Plot the spectrogram of the sound signal
        """
        show = False
        if ax is None:
            fig, ax = plt.subplots(1,1)
            show = True
        
        # Compute the spectrogram
        frequencies, time, Sxx = signal.spectrogram(self.signal, fs=self.fs, window = ('hann'))
        
        # Plot the spectrogram
        ax.pcolormesh(time, frequencies, self.uPa2dB(Sxx)) 

        if show: 
            plt.show()
            plt.close()    