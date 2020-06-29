"""
Module: acoustic_file.py
Authors: Clea Parcerisas
Institution: VLIZ (Vlaams Institute voor de Zee)
"""

import os
import datetime
import acoustics
import numpy as np
import numba as nb
import pandas as pd
import soundfile as sf
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.integrate as integrate

pd.plotting.register_matplotlib_converters()
plt.style.use('ggplot')

from pypam import _event



class AcuFile:
    def __init__(self, sfile, hydrophone, ref, band=None):
        """
        Data recorded in a wav file.

        Parameters
        ----------
        sfile: the sound file. Can be a path or an file object 
        hydrophone: is an object for the class hydrophone
        ref: is the reference pressure or acceleration in uPa or um/s
        """
        # Save hydrophone model 
        self.hydrophone = hydrophone 

        # Get the date from the name
        try:
            file_name = sfile.name
        except:
            file_name = os.path.split(sfile)[-1]
        self.date = hydrophone.get_name_date(file_name)

        # Signal
        self.file_path = sfile
        self.file = sf.SoundFile(self.file_path)
        self.fs = self.file.samplerate

        # Reference pressure or acceleration in uPa or um/s
        self.ref = ref

        # Band selected to study
        self.band = band
    

    def __getattr__(self, name):
        """
        Specific methods to make it easier
        """
        if name == 'signal':
            return self.signal('uPa')
        elif name == 'time':
            return self.time()
        else:
            return self.__dict__[name]


    def samples(self, bintime):
        """
        Return the samples according to the fs

        Parameters
        ----------
        bintime: bintime in seconds
        """
        return int(bintime * self.fs)

    
    def is_in_period(self, period):
        """
        Return True if the WHOLE file is included in the specified period
        
        Parameters
        ----------
        period: list or a tuple with (start, end). The values have to be a datetime object
        """
        if period is None: 
            return True
        else: 
            end = self.date + datetime.timedelta(seconds=self.file.frames/self.fs)
            return (self.date >= period[0]) & (self.date <= period[1])
        
    
    def contains_date(self, date):
        """
        Return True if data is included in the file

        Parameters
        ----------
        date:  
        """
        end = self.date + datetime.timedelta(seconds=self.file.frames/self.fs)
        return (self.date < date) & (end > date)

    
    def split(self, date):
        """
        Save two different files out of one splitting on the specified date

        Parameters
        ----------
        date: date where to split the file
        """
        if not self.contains_date(date):
            raise Exception('This date is not included in the file!')
        else:
            self.file.seek(0)
        seconds = (date - self.date).seconds
        frames = self.samples(seconds)
        first_file = self.file.read(frames=frames)
        second_file = self.file.read()
        # THIS HAS TO BE CHANGED FOR OTHER HYDROPHONES
        old_date_string = self.date.strftime("%y%m%d%H%M%S")
        new_date_string = date.strftime("%y%m%d%H%M%S")
        new_file_path = str(self.file_path).replace(old_date_string, new_date_string)
        sf.write(self.file_path, first_file, samplerate=self.fs)
        sf.write(new_file_path, second_file, samplerate=self.fs)

        self.file.close()

        return self.file_path, new_file_path
    

    def freq_resolution_window(self, freq_resolution):
        """
        Given the frequency resolution, window length needed to obtain it (only 2 to the power)

        Parameters
        ----------
        freq_resolution: XXXXXXXXXXXX
        """
        n = np.log2(self.fs / freq_resolution)
        nfft = 2**n
        if nfft > self.file.frames: 
            raise Exception('This is not achievable with this sampling rate, it must be downsampled!')
        return nfft


    def signal(self, units='wav'):
        """
        Get the signal in the specified units

        Parameters
        ----------
        units: XXXXXXXXXX
        """
        # First time, read the file and store it to not read it over and over
        if 'wav' not in self.__dict__.keys():
            self.wav = self.file.read()
        if units == 'wav':
            signal = self.wav
        elif units == 'dB':
            signal = self.wav2dB()
        elif units == 'uPa':
            signal = self.wav2uPa()
        elif units == 'Pa':
            signal = self.wav2uPa() / 1e6
        elif units == 'acc':
            signal = self.wav2acc()
        else:
            raise Exception('%s is not implemented as an outcome unit' % (units))
        
        return signal

    
    def downsample(self, signal, new_fs):
        """
        Reduce the sampling frequency

        Parameters
        ----------
        signal: XXXXXXXXXXXX
        new_fs: XXXXXXXXXXXXXXX
        """
        if new_fs > self.fs: 
            raise Exception('This is upsampling!')
        new_lenght = int(signal.size * (new_fs / self.fs))
        new_signal = sig.resample(signal, new_lenght)
        return new_signal
    

    def time(self):
        """
        Return a time array for each point of the signal 
        """
        # First time, read the file and store it to not read it over and over
        if 'time' not in self.__dict__.keys():
            self.wav = self.file.read()
        incr = datetime.timedelta(seconds=(np.linspace(start=0, num=self.file.frames))) # Change, it is not correct!
        self.time = self.date + incr

        return self.time


    def wav2uPa(self, wav=None):
        """ 
        Compute the pressure from the wav signal 
        
        Parameters
        ----------
        wav: XXXXXXXXXXXXXX
        """
        # Read if no signal is passed
        if wav is None:
            wav = self.signal('wav')

        # First convert it to Volts and then to dB according to sensitivity
        Mv = 10 ** (self.hydrophone.sensitivity / 20.0) * self.ref
        Ma = 10 ** (self.hydrophone.preamp_gain / 20.0) * self.ref
        return (wav * self.hydrophone.Vpp/2.0) / (Mv * Ma)


    def wav2dB(self, wav=None):
        """ 
        Compute the dB from the wav signal. Consider the hydrophone sensitivity in dB

        Parameters
        ----------
        wav: XXXXXXXXX
        """
        # Read if no signal is passed
        if wav is None:
            wav = self.signal('wav')
        signal_db = 10*np.log10(wav**2) - self.hydrophone.sensitivity
        return signal_db

    
    def dB2uPa(self, dB=None):
        """
        Compute the uPa from the dB signals

        Parameters
        ----------
        dB: signal in dB
        """
        if dB is None:
            dB = self.signal('dB')
        return np.power(10, dB / 20.0 - np.log10(self.ref))
    

    def uPa2dB(self, uPa=None):
        """ 
        Compute the dB from the uPa signal

        Parameters
        ----------
        uPa: signal in uPa
        """
        if uPa is None:
            uPa = self.signal('uPa')
        return 10*np.log10(uPa**2 / self.ref**2)


    def wav2acc(self, wav=None):
        """
        Convert the wav file to acceleration 

        Parameters
        ----------
        wav: wav file
        """
        if wav is None: 
            wav = self.file.read()
        Mv = 10 ** (self.hydrophone.mems_sensitivity / 20.0)
        return wav / Mv


    def fill_or_crop(self, n_samples, signal):
        """ 
        Crop the signal to the number specified or fill it with Nan values in case it is too short 

        Parameters
        ----------
        n_samples: XXXXXX
        signal: XXXXXXX
        """
        if signal.size >= n_samples: 
            return signal[0:n_samples]
        else:
            nan_array = np.full((n_samples,), np.nan)
            nan_array[0:signal.size] = signal
            return nan_array


    def rms(self, binsize=None, dB=True):
        """
        Return the root mean squared value of the signal in uPa

        Parameters
        ----------
        binsize: is the time window considered. If set to None, only one value is returned (in sec)
        dB: if set to True the result will be given in dB. Otherwise in uPa
        
        Returns
        -------
        The output is a dataframe with 'datetime' as index and 'rms' value as a column
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)
        rms_df = pd.DataFrame(columns=['datetime', 'rms'])
        rms_df = rms_df.set_index('datetime')
        for i, block in enumerate(self.file.blocks(blocksize=blocksize)): 
            signal = self.wav2uPa(wav=block)
            if self.band is not None: 
                # Filter the signal
                sosfilt = sig.butter(N=4, btype='bandpass', Wn=self.band, analog=False, output='sos', fs=self.fs)
                signal = sig.sosfilt(sosfilt, signal)
            time = self.date + datetime.timedelta(seconds=(blocksize * i)/self.fs)
            rms = np.sqrt((signal**2).mean())
            # Convert it to dB if applicatble
            if dB:
                rms = 10 * np.log10(rms**2)
            rms_df.loc[time] = rms
            
        return rms_df

    
    def get_timestamps_bins(self, binsize=None, nfft=None, dB=None):
        """
        Return a df with the timestamps of each bin 

        Parameters
        ----------
        binsize: is the time window considered. If set to None, only one value is returned (in sec)
        
        Returns
        -------
        The output is a dataframe with 'datetime'
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)
        df = pd.DataFrame(columns=['datetime', 'instrument'])
        time_list = []
        for i, block in enumerate(self.file.blocks(blocksize=blocksize)): 
            time = self.date + datetime.timedelta(seconds=(blocksize * i)/self.fs)
            time_list.append(time)
        df['datetime'] = time_list
        df['instrument'] = self.hydrophone.name
            
        return df


    def spectrogram(self, binsize=None, nfft=512, scaling='density', dB=True):
        """
        Return the spectrogram of the signal (entire file)
        `binsize` is the time window considered. If set to None, only one value is returned (in sec)
        `nfft` lenght in samples of the window of the fast fourier transform to be computed
        `scaling` can be set to 'spectrum' or 'density' depending on the desired output
        `dB` if set to True the result will be given in dB. Otherwise in uPa^2
        ---------------
        The output is 
        `time` array with the starting time of each bin
        `freq` frequency array 
        `t` seconds of the spectrogram
        `Sxx_list` spectrogram list, one for each bin
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = self.samples(binsize)
        Sxx_list = []
        time = []
        # Window to use for the spectrogram 
        window = sig.get_window('boxcar', nfft)
        for i, block in enumerate(self.file.blocks(blocksize=blocksize)): 
            signal = self.wav2uPa(wav=block)
            if signal.size != blocksize:
                signal = self.fill_or_crop(n_samples=blocksize, signal=signal)
            if self.band is not None: 
                # Filter the signal
                sosfilt = sig.butter(N=4, btype='bandpass', Wn=self.band, analog=False, output='sos', fs=self.fs)
                signal = sig.sosfilt(sosfilt, signal)
                # If the max frequency is lower than the nyquist freq, downsample the signal 
                # This will give better frequency resolution without loosing time resolution
                if self.band[1] < self.fs / 2: 
                    new_fs = self.band[1] * 2
                    signal = self.downsample(signal, new_fs)
                else:
                    new_fs = self.fs
            else:
                new_fs = self.fs
            
            freq, t, Sxx = sig.spectrogram(signal, fs=new_fs, nfft=nfft, window=window, scaling=scaling)
            if dB:
                Sxx = 10 * np.log10(Sxx)
            if self.band is not None: 
                low_freq = np.argmax(freq >= self.band[0])
            else:
                low_freq = 0
            Sxx_list.append(Sxx[low_freq:,:])
            time.append(self.date + datetime.timedelta(seconds=(blocksize/self.fs * i)))
        
        return time, freq[low_freq:], t, Sxx_list


    def _spectrum(self, scaling='density', binsize=None, bands='all', nfft=512, dB=True, percentiles=[]):
        """
        Return the spectrum: frequency distribution of all the file (periodogram)
        `scaling` can be 'density' for psd or 'spectrum' for power output
        `binsize` is the time window considered. If set to None, only one value is returned (in sec)
        `bands` can be set to octaves, third_octaves or all. 
        `nfft` lenght in samples of the window of the fast fourier transform to be computed
        `dB` if set to True the result will be given in dB. Otherwise in uPa^2
        `percentiles` is a list of all the percentiles that have to be returned. If set to None, no percentiles is returned
        ---
        The output is a dataframe with 'datetime' as index and a colum for each frequency and each percentile, and a frequency array
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)
        
        columns_df = pd.DataFrame({'variable': 'percentiles', 'value': percentiles})
        for i, block in enumerate(self.file.blocks(blocksize=blocksize)): 
            signal = self.wav2uPa(wav=block)
            if self.band is not None: 
                # If the max frequency is lower than the nyquist freq, downsample the signal 
                # This will give better frequency resolution without loosing time resolution
                if self.band[1] < self.fs / 2: 
                    new_fs = self.band[1] * 2
                    signal = self.downsample(signal, new_fs)
                else:
                    new_fs = self.fs
                
                if self.band[0] != 0:
                    # Filter the signal
                    sosfilt = sig.butter(N=2, btype='highpass', Wn=self.band[0], analog=False, output='sos', fs=new_fs)
                    signal = sig.sosfilt(sosfilt, signal)
            else:
                new_fs = self.fs
            if bands == 'third_octaves':
                # Return the power for each third octave band (log output)
                fbands, spectra = acoustics.signal.third_octaves(signal, new_fs)
            elif bands == 'octaves':
                # Return the power for each octave band (log output)
                fbands, spectra = acoustics.signal.octaves(signal, new_fs)
            elif bands == 'all':
                window = sig.get_window('boxcar', nfft)
                noverlap = int(nfft * 0.5)
                if signal.size < nfft: 
                    continue
                fbands, spectra = sig.periodogram(signal, fs=new_fs, window=window, nfft=nfft, scaling=scaling)
                # fbands, spectra = sig.welch(signal, fs=new_fs, window=window, nperseg=nfft, nfft=nfft, noverlap=noverlap, scaling=scaling)
            
            if dB:
                spectra = 10 * np.log10(spectra)
            
            # Add the spectra of the bin and the correspondent time step
            time = self.date + datetime.timedelta(seconds=(blocksize/self.fs * i))
            if self.band is not None:
                low_freq = np.argmax(fbands >= self.band[0])
            else:
                low_freq = 1
            try:
                spectra_df.loc[time, ('band_'+scaling, fbands[low_freq:])] = spectra[low_freq:]
            except: 
                columns_df = pd.concat([columns_df, pd.DataFrame({'variable':'band_'+scaling, 'value':fbands[low_freq:]})])
                columns = pd.MultiIndex.from_frame(columns_df)
                spectra_df = pd.DataFrame(columns=columns)
                spectra_df.loc[time, ('band_'+scaling, fbands[low_freq:])] = spectra[low_freq:]

            # Calculate the percentiles
            spectra_df.loc[time, ('percentiles', percentiles)] = np.percentile(spectra, percentiles)

        return spectra_df


    def psd(self, binsize=None, bands='all', nfft=512, dB=True, percentiles=[]):
        """
        Return the power spectrogram density of all the file (units^2 / Hz) re 1 V 1 uPa
        `binsize` is the time window considered. If set to None, only one value is returned (in sec)
        `bands` can be set to octaves, third_octaves or all. 
        `nfft` lenght in samples of the window of the fast fourier transform to be computed
        `dB` if set to True the result will be given in dB. Otherwise in uPa^2
        `percentiles` is a list of all the percentiles that have to be returned. If set to None, no percentiles is returned
        ---
        The output is a dataframe with 'datetime' as index and a colum for each frequency and each percentile
        """
        psd_df = self._spectrum(scaling='density', binsize=binsize, bands=bands, nfft=nfft, dB=dB, percentiles=percentiles)
        
        return psd_df
    

    def power_spectrum(self, binsize=None, bands='all', nfft=512, dB=True, percentiles=[]):
        """
        Return the power spectrogram density of all the file (units^2 / Hz) re 1 V 1 uPa
        `binsize` is the time window considered. If set to None, only one value is returned (in sec)
        `bands` can be set to octaves, third_octaves or all. 
        `nfft` lenght in samples of the window of the fast fourier transform to be computed
        `dB` if set to True the result will be given in dB. Otherwise in uPa^2
        `percentiles` is a list of all the percentiles that have to be returned. If set to None, no percentiles is returned
        ---
        The output is a dataframe with 'datetime' as index and a colum for each frequency and each percentile,
        """
        spectrum_df = self._spectrum(scaling='spectrum', binsize=binsize, bands=bands, nfft=nfft, dB=dB, percentiles=percentiles)
        
        return spectrum_df


    def spd(self, binsize=None, h=0.1, nfft=512, dB=True, percentiles=[]):
        """
        Return the empirical power density. 
        `binsize` is the time window considered. If set to None, only one value is returned (in sec)
        `h` histogram bin (in the correspondent units, uPa or dB)
        `nfft` lenght in samples of the window of the fast fourier transform to be computed
        `dB` if set to True the result will be given in dB. Otherwise in uPa^2
        `percentiles` is a list of all the percentiles that have to be returned. If set to None, no percentiles is returned
        ---
        The output is 
        `time` is a list with the starting point of each spd df
        `fbands` is a list of all the frequencies
        `bin_edges` is a list of the psd values of the distribution
        `spd_list` list of dataframes with 'frequency' as index and a colum for each psd bin and for each percentile (one df per bin)
        `p_list` is a list of matrices with all the probabilities
        """
        time, fbands, t, Sxx_list = self.spectrogram(binsize=binsize, nfft=nfft, dB=dB, scaling='density')
        spd_list = []
        p_list = []
        edges_list = []
        for Sxx in Sxx_list:
            # Calculate the bins of the psd values and compute spd using numba
            bin_edges = np.arange(start=Sxx.min(), stop=Sxx.max(), step=h)
            spd, p = Sxx2spd(Sxx=Sxx, h=h, percentiles=np.array(percentiles)/100.0, bin_edges=bin_edges)
            spd_list.append(spd)
            p_list.append(p)
            edges_list.append(bin_edges)
        
        return time, fbands, percentiles, edges_list, spd_list, p_list

    
    def correlation(self, signal, fs_signal):
        """
        Compute the correlation with the signal 
        `signal` signal to be correlated with 
        `fs_signal` sampling frequency of the signal. It will be down/up sampled in case it does not match with the file
        """
        return 0

    
    def detect_events(self, detector, params=[], binsize=None, nfft=None):
        """
        Detect events
        `detector` object with a detect_events() function that returns a list of Event objects
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)
        
        events_df = pd.DataFrame()
        for i, block in enumerate(self.file.blocks(blocksize=blocksize)): 
            signal = self.wav2uPa(wav=block)
            # TBI: Process the signal 
            start_time =  self.date + datetime.timedelta(seconds=(blocksize/self.fs * i))
            events = detector.detect_events(signal, self.fs, datetime_start=start_time)
            events_df = events_df.append(events)
        
        return events_df
    

    # def level_vs_time(self, binsize=None, cal=0, ref=0):
    #     """
    #     Calculation of sound pressure level over time for whole file, with timestep dt (in seconds)
    #     Assumes first channel in wavfile is sound pressure
    #     """
    #     if binsize is None:
    #         blocksize = self.file.frames
    #     else:
    #         blocksize = int(binsize * self.fs)

    #     y = []
    #     N = int(self.file.frames / blocksize)
    #     for i, block in enumerate(self.file.blocks(blocksize=blocksize)): 
    #         signal = self.wav2uPa(wav=block)
    #         if self.band is not None: 
    #             # Filter the signal
    #             sosfilt = sig.butter(N=4, btype='bandpass', Wn=self.band, analog=False, output='sos', fs=self.fs)
    #             signal = sig.sosfilt(sosfilt, signal)
    #             # If the max frequency is lower than the nyquist freq, downsample the signal 
    #             # This will give better frequency resolution without loosing time resolution
    #             if self.band[1] < self.fs / 2: 
    #                 new_fs = self.band[1] * 2
    #                 signal = self.downsample(signal, new_fs)
    #             else:
    #                 new_fs = self.fs
    #             # calculate level of first channel
    #             z = z[:,1]
    #             y[i] = 10*np.log10((signal**2).sum()/N) - ref[0] + cal[0]

    #     return y


    def find_calibration_tone(self, max_duration, freq, min_duration=10.0):
        """
        Find the beggining and ending sample of the calibration tone
        """
        high_freq = freq * 1.05
        low_freq = freq * 0.95
        tone_samples = self.samples(max_duration)
        self.file.seek(0)
        first_part = self.file.read(frames=tone_samples)
        sosfilt = sig.butter(N=1, btype='bandpass', Wn=[low_freq, high_freq], analog=False, output='sos', fs=self.fs)
        filtered_signal = sig.sosfilt(sosfilt, first_part)
        analytic_signal = sig.hilbert(filtered_signal)
        amplitude_envelope = np.abs(analytic_signal)
        possible_points = np.zeros(amplitude_envelope.shape)
        possible_points[np.where(amplitude_envelope >= 0.05)] = 1
        start_points = np.where(np.diff(possible_points) == 1)[0]
        end_points = np.where(np.diff(possible_points) == -1)[0]
        if start_points.size == 0: 
            return None, None
        if end_points[0] < start_points[0]:
            end_points = end_points[1:]
        if start_points.size != end_points.size: 
            start_points = start_points[0:end_points.size]
        select_idx = np.argmax(end_points - start_points)
        start = start_points[select_idx]
        end = end_points[select_idx]

        if (end - start)/self.fs < min_duration: 
            return None, None

        plt.figure()
        plt.plot(filtered_signal, label='filtered_signal')
        plt.plot(amplitude_envelope, label='envelope')
        plt.axvline(x=start, color='red')
        plt.axvline(x=end, color='blue')
        plt.tight_layout()
        plt.show()
        plt.close()

        return start, end

    
    def cut_calibration_tone(self, max_duration, freq, min_duration=10.0, save_path=None):
        """
        Cut the calibration tone from the file
        """
        start, stop = self.find_calibration_tone(max_duration=max_duration, freq=freq, min_duration=min_duration)
        if start is not None: 
            calibration_signal, _ = sf.read(self.file_path, start=start, stop=stop)
            signal, _ = sf.read(self.file_path, start=stop+1, stop=-1)
            if save_path is not None: 
                sf.write(file=save_path, data=calibration_signal, samplerate=self.fs)
                sf.write(file=str(self.file_path), data=signal, samplerate=self.fs)
            return calibration_signal, signal
        
        else:
            return None


    def plot_psd(self, dB=True, log=True, save_path=None, **kwargs):
        """
        Plot the power spectrogram density of all the file (units^2 / Hz) re 1 V 1 uPa
        `dB` if set to True the result will be given in dB. Otherwise in uPa^2/Hz
        `save_path` is where to save the images
        The kwargs valid are the ones to compute the psd
        """
        psd = self.psd(dB=dB, **kwargs)
        if dB: 
            units = 'dB re 1V %s uPa^2/Hz' % (self.ref)
        else:
            units = 'uPa^2/Hz' 
        self._plot_spectrum(df=psd, col_name='density', output_name='PSD', units=units, dB=dB, log=log, save_path=save_path)

    
    def plot_power_spectrum(self, dB=True, log=True, save_path=None, **kwargs):
        """
        Plot the power spectrogram of all the file (units^2) re 1 V 1 uPa
        `dB` if set to True the result will be given in dB. Otherwise in uPa^2
        `save_path` is where to save the images
        The kwargs valid are the ones to compute the power spectrum
        """
        power = self.power_spectrum(dB=dB, **kwargs)
        if dB: 
            units = 'dB re 1V %s uPa^2' % (self.ref)
        else:
            units = 'uPa^2' 
        self._plot_spectrum(df=power, col_name='spectrum', output_name='SPL', units=units, dB=dB, log=log, save_path=save_path)


    def _plot_spectrum(self, df, col_name, output_name, units, dB=True, log=True, save_path=None):
        """
        Plot the spectrums contained on the df
        `df` must be a dataframe resultant from psd or power spectrum calculation 
        `col_name` the name of the column where the data is (scaling type)
        `units` the units of the data
        `dB` if set to True, sata plot in dB
        `save_path` where to save the image
        """
        fbands = df['band_'+col_name].columns
        for i in df.index:
            fig = plt.figure()
            plt.plot(fbands, df.loc[i, 'band_'+col_name][fbands])
            plt.title(col_name.capitalize())
            plt.xlabel('Frequency [Hz')
            plt.ylabel('%s [%s]' % (output_name, units))

            plt.hlines(y=df.loc[i, 'percentiles'].values, xmin=fbands.min(), xmax=fbands.max(), label=df['percentiles'].columns)
            if log: 
                plt.xscale('log')
            plt.tight_layout()
            plt.show()
            if save_path is not None: 
                plt.savefig(save_path)
            plt.close()   

    
    def plot_spectrogram(self, dB=True, log=True, save_path=None, **kwargs):
        """
        Return the spectrogram of the signal (entire file)
        `dB` if set to True the result will be given in dB. Otherwise in uPa^2
        `log` set to True if the y scale must be in a logarithmic scale
        `save_path` is the path where the images want to be stored. In case there are more than one, a index will be added
        The valid kwargs are the ones from spectrogram
        """
        time, fbands, t, Sxx_list = self.spectrogram(dB=dB, **kwargs)
        for i, Sxx in enumerate(Sxx_list): 
            # Plot the patterns
            plt.figure()
            im = plt.pcolormesh(t, fbands, Sxx)
            plt.title('Spectrogram of bin %s' % (time[i])) 
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            if log: 
                plt.yscale('log')
            if dB: 
                units = 'dB re 1V %s uPa' % (self.ref)
            else:
                units = 'uPa'
            cbar = plt.colorbar(im)
            cbar.set_label('SPL [%s]' % (units), rotation=90)
            plt.tight_layout()
            plt.show()
            if save_path is not None: 
                plt.savefig(save_path + str(time[i]))
            plt.close()   
        

    def plot_spd(self, dB=True, log=True, save_path=None, **kwargs):
        """
        Plot the the SPD graph of the bin
        `dB` if set to True the result will be given in dB. Otherwise in uPa^2
        `log` set to True if the y scale must be in a logarithmic scale
        `save_path` is the path where the images want to be stored. In case there are more than one, a index will be added
        The valid kwargs are the ones from spd
        """
        time, fbands, percentiles, edges_list, spd_list, p_list = self.spd(dB=dB, **kwargs)
        if dB: 
            units = 'dB re 1V %s uPa^2/Hz' % (self.ref)
        else:
            units = 'uPa^2/Hz'
        for i, spd in enumerate(spd_list): 
            # Plot the EPD
            fig = plt.figure()
            im = plt.pcolormesh(fbands, edges_list[i], spd.T, cmap='BuPu')
            if log: 
                plt.xscale('log')
            plt.title('Spectral probability density at bin %s' % time[i])
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('PSD [%s]' % (units))
            cbar = fig.colorbar(im)
            cbar.set_label('Empirical Probability Density', rotation=90)
            
            # Plot the lines of the percentiles
            plt.plot(fbands, p_list[i], label=percentiles)

            plt.tight_layout()
            plt.show()
            if save_path is not None: 
                plt.savefig(save_path)
            plt.close()  


@nb.jit
def Sxx2spd(Sxx, h, percentiles, bin_edges):
    """
    Return spd from the spectrogram
    """
    spd = np.zeros((Sxx.shape[0], bin_edges.size-1), dtype=np.float64)
    p = np.zeros((Sxx.shape[0], percentiles.size), dtype=np.float64)
    for i in nb.prange(Sxx.shape[0]):
        spd[i,:] = np.histogram(Sxx[i,:], bin_edges)[0] / ((bin_edges.size - 1) * h)
        cumsum = np.cumsum(spd[i,:])
        for j in nb.prange(percentiles.size):
            p[i,j] = bin_edges[np.argmax(cumsum > percentiles[j]*cumsum[-1])]

    return spd, p



class HydroFile(AcuFile):
    def __init__(self, sfile, hydrophone, p_ref=1.0, band=None):
        """
        Sound data recorded in a wav file.
        `sfile` the sound file. Can be a path or an file object 
        `hydrophone` is an object for the class hydrophone
        `ref` is the reference pressure or acceleration in uPa
        """
        super().__init__(sfile, hydrophone, p_ref, band)


    def click_detector(self, binsize=None):
        """
        Check for click detections in each bin
        """
        return 0




class MEMSFile(AcuFile):
    def __init__(self, sfile, hydrophone, acc_ref=1.0, band=None):
        """
        Acceleration data recorded in a wav file.
        `sfile` the sound file. Can be a path or an file object 
        `hydrophone` is an object for the class hydrophone
        `acc_ref` is the reference pressure or acceleration in um/s
        """
        super().__init__(sfile, hydrophone, acc_ref, band)
    
    
    def integrate_acceleration(self):
        """ 
        Integrate the acceleration to get the velocity of the particle. The constant is NOT resolved
        """
        if self.instant:
            raise Exception('This can only be implemented in average mode!')

        velocity = integrate.cumtrapz(self.signal('acc'), dx=1/self.fs)

        return velocity



class MEMS3axFile:
    def __init__(self, hydrophone, xfile_path, yfile_path, zfile_path, acc_ref=1.0):
        """
        Class to treat the 3 axes together
        """
        self.x_path = xfile_path
        self.y_path = yfile_path
        self.z_path = zfile_path

        self.x = MEMSFile(hydrophone, xfile_path, acc_ref)
        self.y = MEMSFile(hydrophone, yfile_path, acc_ref)
        self.z = MEMSFile(hydrophone, zfile_path, acc_ref)


    def acceleration_magnitude(self):
        """ 
        Calculate the magniture acceleration signal 
        """ 
        x_acc = self.x.signal('acc')
        y_acc = self.y.signal('acc')
        z_acc = self.z.signal('acc')

        return np.sqrt(x_acc**2 + y_acc**2 + z_acc**2)


    def velocity_magnitude(self):
        """
        Get the start and the end velocity 
        """
        vx = self.x.integrate_acceleration()
        vy = self.y.integrate_acceleration()
        vz = self.z.integrate_acceleration()

        v = np.sqrt(vx**2 + vy**2 + vz**2)
        
        return v


    def mean_velocity_increment(self):
        """
        Get the mean increment of the velocity
        """
        mean_acc = self.acceleration_magnitude.mean()
        time = self.time()
        t_inc = (time()[-1] - time()[0]).total_seconds()

        return mean_acc * t_inc


    # def plot_particle_velocity(self, ax=None):
    #     """
    #     Plot the particle velocity 
    #     """
    #     # Compute the particle velocity
    #     v = self.integrate_acceleration()
    #     show = False
    #     if ax is None:
    #         fig, ax = plt.subplots(1,1)
    #         show = True
    #     # Plot 
    #     ax.plot(self.x.time(), vx, label='x')
    #     ax.plot(self.measurements.index[0:-1], vy, label='y')
    #     ax.plot(self.measurements.index[0:-1], vz, label='z')
    #     ax.plot(self.measurements.index[0:-1], v_mag, label='magnitude')

    #     ax.set_title('Particle velocity')
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('um/s')
    #     ax.legend()

    #     if show: 
    #         plt.show()
    #         plt.close()