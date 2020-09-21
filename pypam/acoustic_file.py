"""
Module : acoustic_file.py
Authors : Clea Parcerisas
Institution : VLIZ (Vlaams Institute voor de Zee)
"""
import datetime
import os
import pathlib

import acoustics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.signal as sig
import soundfile as sf

from pypam import utils

pd.plotting.register_matplotlib_converters()
plt.style.use('ggplot')


class AcuFile:
    def __init__(self, sfile, hydrophone, ref, band=None, utc=True):
        """
        Data recorded in a wav file.

        Parameters
        ----------
        sfile : Sound file 
            Can be a path or an file object 
        hydrophone : Object for the class hydrophone
        ref : Float
            Reference pressure or acceleration in upa or um/s
        band : list
            Band to filter 
        utc : boolean
            Set to True if working on UTC and not localtime
        """
        # Save hydrophone model 
        self.hydrophone = hydrophone

        # Get the date from the name
        if type(sfile) == str:
            file_name = os.path.split(sfile)[-1]
        elif issubclass(sfile.__class__, pathlib.Path):
            file_name = sfile.name
        else:
            raise Exception('The filename has to be either a Path object or a string')

        try:
            self.date = hydrophone.get_name_datetime(file_name, utc=utc)
        except ValueError:
            self.date = datetime.datetime.now()
            raise Warning('Filename %s does not match the %s file structure. Setting time to now...' %
                          (file_name, self.hydrophone.name))

        # Signal
        self.file_path = sfile
        self.file = sf.SoundFile(self.file_path)
        self.fs = self.file.samplerate

        # Reference pressure or acceleration in upa or um/s
        self.ref = ref

        # Band selected to study
        self.band = band

        # Work on local time or UTC time
        self.utc = utc

        # Set an empty wav array
        self.wav = None
        self.time = None

    def __getattr__(self, name):
        """
        Specific methods to make it easier to access attributes
        """
        if name == 'signal':
            return self.signal('upa')
        elif name == 'time':
            return self.get_time()
        else:
            return self.__dict__[name]

    def samples(self, bintime):
        """
        Return the samples according to the fs

        Parameters
        ----------
        bintime : Float
            Seconds of bintime desired to convert to samples
        """
        return int(bintime * self.fs)

    def instrument(self):
        """
        Instrument will be the name of the hydrophone
        """
        return self.hydrophone.name

    def total_time(self):
        """
        Return the total time in seconds of the file 
        """
        return self.samples2time(self.file.frames)

    def samples2time(self, samples):
        """
        Return the samples according to the fs

        Parameters
        ----------
        samples : Int
            Number of samples to convert to seconds
        """
        return float(samples) / self.fs

    def is_in_period(self, period):
        """
        Return True if the WHOLE file is included in the specified period
        
        Parameters
        ----------
        period : list or a tuple with (start, end).
            Values have to be a datetime object
        """
        if period is None:
            return True
        else:
            # end = self.date + datetime.timedelta(seconds=self.file.frames / self.fs)
            return (self.date >= period[0]) & (self.date <= period[1])

    def contains_date(self, date):
        """
        Return True if data is included in the file

        Parameters
        ----------
        date : datetime object
            Datetime to check
        """
        end = self.date + datetime.timedelta(seconds=self.file.frames / self.fs)
        return (self.date < date) & (end > date)

    def split(self, date):
        """
        Save two different files out of one splitting on the specified date

        Parameters
        ----------
        date : datetime object
            Datetime where to split the file
        """
        if not self.contains_date(date):
            raise Exception('This date is not included in the file!')
        else:
            self.file.seek(0)
        seconds = (date - self.date).seconds
        frames = self.samples(seconds)
        first_file = self.file.read(frames=frames)
        second_file = self.file.read()
        self.file.close()

        new_file_name = self.hydrophone.get_new_name(filename=self.file_path.name, new_date=date)
        new_file_path = self.file_path.parent.joinpath(new_file_name)
        sf.write(self.file_path, first_file, samplerate=self.fs)
        sf.write(new_file_path, second_file, samplerate=self.fs)

        return self.file_path, new_file_path

    def freq_resolution_window(self, freq_resolution):
        """
        Given the frequency resolution, window length needed to obtain it
        Returns window lenght in samples

        Parameters
        ----------
        freq_resolution : int
            Must be a power of 2, in Hz
        """
        n = np.log2(self.fs / freq_resolution)
        nfft = 2 ** n
        if nfft > self.file.frames:
            raise Exception('This is not achievable with this sampling rate, it must be downsampled!')
        return nfft

    def signal(self, units='wav'):
        """
        Returns the signal in the specified units

        Parameters
        ----------
        units : string
            Units in which to return the signal. Can be 'wav', 'db', 'upa', 'Pa' or 'acc'.
        """
        # First time, read the file and store it to not read it over and over
        if self.wav is None:
            self.wav = self.file.read()
        if units == 'wav':
            signal = self.wav
        elif units == 'db':
            signal = self.wav2db()
        elif units == 'upa':
            signal = self.wav2upa()
        elif units == 'Pa':
            signal = self.wav2upa() / 1e6
        elif units == 'acc':
            signal = self.wav2acc()
        else:
            raise Exception('%s is not implemented as an outcome unit' % units)

        return signal

    def get_time(self):
        """
        Return a time array for each point of the signal 
        """
        # First time, read the file and store it to not read it over and over
        if 'time' not in self.__dict__.keys():
            self.wav = self.file.read()
        incr = pd.to_timedelta(np.linspace(start=0, stop=self.total_time(), num=self.file.frames), unit='seconds')
        self.time = self.date + incr

        return self.time

    def wav2upa(self, wav=None):
        """ 
        Compute the pressure from the wav signal 
        
        Parameters
        ----------
        wav : numpy array
            Signal in wav (-1 to 1)
        """
        # Read if no signal is passed
        if wav is None:
            wav = self.signal('wav')

        # First convert it to Volts and then to db according to sensitivity
        mv = 10 ** (self.hydrophone.sensitivity / 20.0) * self.ref
        ma = 10 ** (self.hydrophone.preamp_gain / 20.0) * self.ref
        return (wav * self.hydrophone.Vpp / 2.0) / (mv * ma)

    def wav2db(self, wav=None):
        """ 
        Compute the db from the wav signal. Consider the hydrophone sensitivity in db. 
        If wav is None, it will read the whole file. 

        Parameters
        ----------
        wav : numpy array
            Signal in wav (-1 to 1)
        """
        # Read if no signal is passed
        if wav is None:
            wav = self.signal('wav')
        upa = self.wav2upa(wav)
        signal_db = 10 * np.log10(upa ** 2)
        return signal_db

    def db2upa(self, db=None):
        """
        Compute the upa from the db signals. If db is None, it will read the whole file. 

        Parameters
        ----------
        db : numpy array
            Signal in db
        """
        if db is None:
            db = self.signal('db')
        return np.power(10, db / 20.0 - np.log10(self.ref))

    def upa2db(self, upa=None):
        """ 
        Compute the db from the upa signal. If upa is None, it will read the whole file. 

        Parameters
        ----------
        upa : numpy array
            Signal in upa
        """
        if upa is None:
            upa = self.signal('upa')
        return 10 * np.log10(upa ** 2 / self.ref ** 2)

    def wav2acc(self, wav=None):
        """
        Convert the wav file to acceleration. If wav is None, it will read the whole file. 

        Parameters
        ----------
        wav : numpy array 
            Signal in wav (-1 to 1)
        """
        if wav is None:
            wav = self.file.read()
        mv = 10 ** (self.hydrophone.mems_sensitivity / 20.0)
        return wav / mv

    def timestamps_df(self, binsize=None, nfft=None, db=None):
        """
        Return a pandas dataframe with the timestamps of each bin.
        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        db : None
            Does not apply. It is ignored
        nfft : None
            Does not apply. It is ignored
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)
        df = pd.DataFrame(columns=['datetime'])
        time_list = []
        i = 0
        while i < self.file.frames:
            time = self.date + datetime.timedelta(seconds=i / self.fs)
            time_list.append(time)
            i += blocksize
        df['datetime'] = time_list

        return df

    def _apply_multiple(self, method_list, binsize=None, db=True, **kwargs):
        """
        Apply the method name
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)

        df = pd.DataFrame(columns=['datetime'] + method_list)
        df = df.set_index('datetime')

        for i, block in enumerate(self.file.blocks(blocksize=blocksize)):
            # Read the signal and prepare it for analysis
            signal = self.wav2upa(wav=block)

            # If it is the last bit, fill or crop to get same size
            if signal.size != blocksize:
                signal = utils.fill_or_crop(n_samples=blocksize, signal=signal)
            _, signal = utils.filter_and_downsample(signal=signal, band=self.band, fs=self.fs)

            time = self.date + datetime.timedelta(seconds=(blocksize * i) / self.fs)
            for method_name in method_list:
                # f = operator.methodcaller(method_name, signal=signal, **kwargs)
                f = getattr(utils, method_name)
                output = f(signal=signal, **kwargs)
                # Convert it to db if applicatble
                # if db:
                #     output = 10 * np.log10(output**2)
                df.at[time, method_name] = output

        return df

    def _apply(self, method_name, binsize=None, db=True, **kwargs):
        """
        Apply one single method
        """
        return self._apply_multiple(self, method_list=[method_name], binsize=binsize, db=db, **kwargs)

    def rms(self, binsize=None, db=True):
        """
        Calculation of root mean squared value (rms) of the signal in upa for each bin
        Returns Dataframe with 'datetime' as index and 'rms' value as a column

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        db : bool
            If set to True the result will be given in db, otherwise in upa
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)
        rms_df = pd.DataFrame(columns=['datetime', 'rms'])
        rms_df = rms_df.set_index('datetime')
        for i, block in enumerate(self.file.blocks(blocksize=blocksize)):
            signal = self.wav2upa(wav=block)
            if self.band is not None:
                # Filter the signal
                sosfilt = sig.butter(N=4, btype='bandpass', Wn=self.band, analog=False, output='sos', fs=self.fs)
                signal = sig.sosfilt(sosfilt, signal)
            time = self.date + datetime.timedelta(seconds=(blocksize * i) / self.fs)
            rms = utils.rms(signal=signal, db=db)
            rms_df.loc[time] = rms

        return rms_df

    def aci(self, binsize=None, nfft=1024):
        """
        Calculation of root mean squared value (rms) of the signal in upa for each bin
        Returns Dataframe with 'datetime' as index and 'rms' value as a column

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        nfft : int
            Window size for processing
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)
        window = sig.get_window('hann', nfft)
        aci_df = pd.DataFrame(columns=['datetime', 'aci'])
        aci_df = aci_df.set_index('datetime')
        for i, block in enumerate(self.file.blocks(blocksize=blocksize)):
            signal = self.wav2upa(wav=block)
            if signal.size != blocksize:
                signal = utils.fill_or_crop(n_samples=blocksize, signal=signal)
            new_fs, signal = utils.filter_and_downsample(signal=signal, band=self.band, fs=self.fs)
            time = self.date + datetime.timedelta(seconds=(blocksize * i) / self.fs)
            _, _, sxx = sig.spectrogram(signal, fs=new_fs, nfft=nfft, window=window, scaling='spectrum')
            aci = utils.calculate_aci(sxx)
            aci_df.loc[time] = aci

        return aci_df

    def dynamic_range(self, binsize=None, db=True):
        """
        Compute the dynamic range of each bin
        Returns a dataframe with datetime as index and dr as column

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        db : bool
            If set to True the result will be given in db, otherwise in upa
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)
        dr_df = pd.DataFrame(columns=['datetime', 'dr'])
        dr_df = dr_df.set_index('datetime')
        for i, block in enumerate(self.file.blocks(blocksize=blocksize)):
            signal = self.wav2upa(wav=block)
            if self.band is not None:
                # Filter the signal
                sosfilt = sig.butter(N=4, btype='bandpass', Wn=self.band, analog=False, output='sos', fs=self.fs)
                signal = sig.sosfilt(sosfilt, signal)
            time = self.date + datetime.timedelta(seconds=(blocksize * i) / self.fs)
            dr = utils.dynamic_range(signal=signal, db=db)
            dr_df.loc[time] = dr

        return dr_df

    def cumulative_dynamic_range(self, binsize=None, db=True):
        """
        Compute the cumulative dynamic range for each bin
        
        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        db : bool
            If set to True the result will be given in db, otherwise in upa^2

        Returns
        -------
        cumsum : 
            Cumulative sum of dynamic range of each bin
        """
        cumdr = self.dynamic_range(binsize=binsize, db=db)
        cumdr['cumsum_dr'] = cumdr.dr.cumsum()
        return cumdr

    def spectrogram(self, binsize=None, nfft=512, scaling='density', db=True):
        """
        Return the spectrogram of the signal (entire file)
        
        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        nfft : int
            Lenght of the fft window in samples. Power of 2. 
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output
        
        Returns
        -------
        time : numpy array
            Array with the starting time of each bin
        freq : numpy array 
            Array with all the frequencies 
        t : numpy array
            Time array in seconds of the windows of the spectrogram
        sxx_list : list
            Spectrogram list, one for each bin
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = self.samples(binsize)
        sxx_list = []
        time = []
        # Window to use for the spectrogram 
        window = sig.get_window('boxcar', nfft)
        freq, t, low_freq = None, None, None
        for i, block in enumerate(self.file.blocks(blocksize=blocksize)):
            signal = self.wav2upa(wav=block)
            if signal.size != blocksize:
                signal = utils.fill_or_crop(n_samples=blocksize, signal=signal)
            new_fs, signal = utils.filter_and_downsample(signal=signal, band=self.band, fs=self.fs)

            freq, t, sxx = sig.spectrogram(signal, fs=new_fs, nfft=nfft, window=window, scaling=scaling)
            if db:
                sxx = 10 * np.log10(sxx)
            if self.band is not None:
                low_freq = np.argmax(freq >= self.band[0])
            else:
                low_freq = 0
            sxx_list.append(sxx[low_freq:, :])
            time.append(self.date + datetime.timedelta(seconds=(blocksize / self.fs * i)))

        return time, freq[low_freq:], t, sxx_list

    def _spectrum(self, scaling='density', binsize=None, bands='all', nfft=512, db=True, percentiles=None):
        """
        Return the spectrum : frequency distribution of all the file (periodogram)
        Returns Dataframe with 'datetime' as index and a colum for each frequency and each percentile,
        and a frequency array

        Parameters
        ----------
        scaling : string
            Can be set to 'spectrum' or 'density' depending on the desired output       
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        bands : string
            Can be set to 'octaves', 'third_octaves' or 'all'. 
        nfft : int
            Lenght of the fft window in samples. Power of 2. 
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list, no percentiles is returned
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)

        if percentiles is not None:
            columns_df = pd.DataFrame({'variable': 'percentiles', 'value': percentiles})
        else:
            columns_df = pd.DataFrame()

        spectra_df = pd.DataFrame(columns=columns_df)
        for i, block in enumerate(self.file.blocks(blocksize=blocksize)):
            signal = self.wav2upa(wav=block)
            new_fs, signal = utils.filter_and_downsample(signal=signal, band=self.band, fs=self.fs)
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
            else:
                raise Exception('%s is not accepted as bands!' % bands)
            if db:
                spectra = 10 * np.log10(spectra)

            # Add the spectra of the bin and the correspondent time step
            time = self.date + datetime.timedelta(seconds=(blocksize / self.fs * i))
            if self.band is not None:
                low_freq = np.argmax(fbands >= self.band[0])
            else:
                low_freq = 1
            if i == 0:
                columns_df = pd.concat(
                    [columns_df, pd.DataFrame({'variable': 'band_' + scaling, 'value': fbands[low_freq:]})])
                columns = pd.MultiIndex.from_frame(columns_df)
                spectra_df = pd.DataFrame(columns=columns)
                spectra_df.loc[time, ('band_' + scaling, fbands[low_freq:])] = spectra[low_freq:]
            else:
                spectra_df.loc[time, ('band_' + scaling, fbands[low_freq:])] = spectra[low_freq:]
            # Calculate the percentiles
            if len(percentiles) != 0:
                spectra_df.loc[time, ('percentiles', percentiles)] = np.percentile(spectra, percentiles)

        return spectra_df

    def psd(self, binsize=None, bands='all', nfft=512, db=True, percentiles=None):
        """
        Return the power spectrogram density (PSD) of all the file (units^2 / Hz) re 1 V 1 upa
        Returns a Dataframe with 'datetime' as index and a colum for each frequency and each percentile

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        bands : string
            Can be set to 'octaves', 'third_octaves' or 'all'. 
        nfft : int
            Lenght of the fft window in samples. Power of 2. 
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list, no percentiles is returned
        """
        psd_df = self._spectrum(scaling='density', binsize=binsize, bands=bands, nfft=nfft, db=db,
                                percentiles=percentiles)

        return psd_df

    def power_spectrum(self, binsize=None, bands='all', nfft=512, db=True, percentiles=None):
        """
        Return the power spectrogram density of all the file (units^2 / Hz) re 1 V 1 upa
        Returns a Dataframe with 'datetime' as index and a colum for each frequency and each percentile

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        bands : string
            Can be set to 'octaves', 'third_octaves' or 'all'. 
        nfft : int
            Lenght of the fft window in samples. Power of 2. 
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list, no percentiles is returned
        """
        spectrum_df = self._spectrum(scaling='spectrum', binsize=binsize, bands=bands, nfft=nfft, db=db,
                                     percentiles=percentiles)

        return spectrum_df

    def spd(self, binsize=None, h=0.1, nfft=512, db=True, percentiles=None):
        """
        Return the spectral probability density. 

        Parameters
        ----------
        binsize : float, in sec
            Time window considered. If set to None, only one value is returned
        h : float
            Histogram bin width (in the correspondent units, upa or db)
        nfft : int
            Lenght of the fft window in samples. Power of 2. 
        db : bool
            If set to True the result will be given in db, otherwise in upa^2
        percentiles : list
            List of all the percentiles that have to be returned. If set to empty list, no percentiles is returned
        
        Returns
        -------
        time : list
            list with the starting point of each spd df
        fbands : list
            list of all the frequencies
        bin_edges : list
            list of the psd values of the distribution
        spd_list : list
            list of dataframes with 'frequency' as index and a colum for each psd bin and
            for each percentile (one df per bin)
        p_list : list of numpy matrices
            list of matrices with all the probabilities
        """
        time, fbands, t, sxx_list = self.spectrogram(binsize=binsize, nfft=nfft, db=db, scaling='density')
        spd_list = []
        p_list = []
        edges_list = []
        if percentiles is None:
            percentiles = []
        for sxx in sxx_list:
            # Calculate the bins of the psd values and compute spd using numba
            bin_edges = np.arange(start=sxx.min(), stop=sxx.max(), step=h)
            spd, p = utils.sxx2spd(sxx=sxx, h=h, percentiles=np.array(percentiles) / 100.0, bin_edges=bin_edges)
            spd_list.append(spd)
            p_list.append(p)
            edges_list.append(bin_edges)

        return time, fbands, percentiles, edges_list, spd_list, p_list

    def correlation(self, signal, fs_signal):
        """
        Compute the correlation with the signal 

        Parameters
        ----------
        signal : numpy array 
            Signal to be correlated with 
        fs_signal : int
            Sampling frequency of the signal. It will be down/up sampled in case it does not match with the file
            samplig frequency
        """
        return 0

    def detect_events(self, detector, binsize=None, nfft=None):
        """
        Detect events. Returns a DataFrame with all the events and their information

        Parameters
        ----------
        detector : object 
            The detector must have a detect_events() function that returns a DataFrame with events information
        binsize : float, in sec
            Time window considered for detections. If set to None, all the file will be processed in one
        nfft : int
            Lenght of the fft window in samples. Power of 2.
        """
        if binsize is None:
            blocksize = self.file.frames
        else:
            blocksize = int(binsize * self.fs)

        events_df = pd.DataFrame()
        for i, block in enumerate(self.file.blocks(blocksize=blocksize)):
            signal = self.wav2upa(wav=block)
            # TBI : Process the signal 
            start_time = self.date + datetime.timedelta(seconds=(blocksize / self.fs * i))
            events = detector.detect_events(signal, self.fs, datetime_start=start_time)
            events_df = events_df.append(events)

        return events_df

    def find_calibration_tone(self, max_duration, freq, min_duration=10.0):
        """
        Find the beggining and ending sample of the calibration tone
        Returns start and end points, in seconds

        Parameters
        ----------
        max_duration : float
            Maximum duration of the calibration tone, in sec
        freq : float
            Expected frequency of the calibration tone, in Hz
        min_duration : float
            Minimum duration of the calibration tone, in sec
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

        if (end - start) / self.fs < min_duration:
            return None, None

        # plt.figure()
        # plt.plot(filtered_signal, label='filtered_signal')
        # plt.plot(amplitude_envelope, label='envelope')
        # plt.axvline(x=start, color='red')
        # plt.axvline(x=end, color='blue')
        # plt.tight_layout()
        # plt.show()
        # plt.close()

        return start, end

    def cut_calibration_tone(self, max_duration, freq, min_duration=10.0, save_path=None):
        """
        Cut the calibration tone from the file
        Returns a numpy array with only calibration signal and signal without the calibration signal
        None if no reference signal is found

        Parameters
        ----------
        max_duration : float
            Maximum duration of the calibration tone, in sec
        freq : float
            Expected frequency of the calibration tone, in Hz
        min_duration : float
            Minimum duration of the calibration tone, in sec
        save_path : string or Path
            Path where to save the calibration tone 
        """
        start, stop = self.find_calibration_tone(max_duration=max_duration, freq=freq, min_duration=min_duration)
        if start is not None:
            new_datetime = self.date + datetime.timedelta(seconds=self.samples2time(stop))
            calibration_signal, _ = sf.read(self.file_path, start=start, stop=stop)
            signal, _ = sf.read(self.file_path, start=stop + 1, stop=-1)
            if save_path is not None:
                if save_path == 'auto':
                    new_folder = self.file_path.parent.joinpath('formatted')
                    ref_folder = new_folder.joinpath('ref')
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    if not os.path.exists(ref_folder):
                        os.makedirs(ref_folder)
                    ref_name = self.file_path.name.replace('.wav', '_ref.wav')
                    save_path = ref_folder.joinpath(ref_name)
                    new_file_name = self.hydrophone.get_new_name(filename=self.file_path.name, new_date=new_datetime)
                    new_file_path = new_folder.joinpath(new_file_name)
                else:
                    new_file_path = self.file_path.parent.joinpath(self.file_path.name.replace('.wav', '_cut.wav'))
                sf.write(file=save_path, data=calibration_signal, samplerate=self.fs)
                # Update datetime
                sf.write(file=new_file_path, data=signal, samplerate=self.fs)
            return calibration_signal, signal

        else:
            return None

    def plot_psd(self, db=True, log=True, save_path=None, **kwargs):
        """
        Plot the power spectrogram density of all the file (units^2 / Hz) re 1 V 1 upa

        Parameters
        ----------
        db : boolean
            If set to True the result will be given in db. Otherwise in upa^2/Hz
        log : boolean
            If set to True the scale of the y axis is set to logarithmic
        save_path : string or Path 
            Where to save the images
        **kwargs : any attribute valid on psd() function
        """
        psd = self.psd(db=db, **kwargs)
        if db:
            units = 'db %s upa^2/Hz' % self.ref
        else:
            units = 'upa^2/Hz'
        self._plot_spectrum(df=psd, col_name='density', output_name='PSD', units=units, db=db, log=log,
                            save_path=save_path)

    def plot_power_spectrum(self, db=True, log=True, save_path=None, **kwargs):
        """
        Plot the power spectrogram of all the file (units^2) re 1 V 1 upa
        
        Parameters
        ----------
        db : boolean
            If set to True the result will be given in db. Otherwise in upa^2/Hz
        log : boolean
            If set to True the scale of the y axis is set to logarithmic
        save_path : string or Path 
            Where to save the images
        **kwargs : any attribute valid on power_spectrum() function
        """
        power = self.power_spectrum(db=db, **kwargs)
        if db:
            units = 'db %s upa^2' % self.ref
        else:
            units = 'upa^2'
        self._plot_spectrum(df=power, col_name='spectrum', output_name='SPLrms', units=units, db=db, log=log,
                            save_path=save_path)

    def _plot_spectrum(self, df, col_name, output_name, units, db=True, log=True, save_path=None):
        """
        Plot the spectrums contained on the df

        Parameters
        ----------
        df : pandas DataFrame
            Dataframe resultant from psd or power spectrum calculation 
        col_name : string
            Name of the column where the data is (scaling type) 'spectrum' or 'density'
        units : string
            Units of the data
        db : boolean
            If set to True, sata plot in db
        save_path: string or Path
            Where to save the image
        """
        fbands = df['band_' + col_name].columns
        for i in df.index:
            plt.figure()
            plt.plot(fbands, df.loc[i, 'band_' + col_name][fbands])
            plt.title('%s of bin %s' % (col_name.capitalize(), i.strftime("%Y-%m-%d %H:%M")))
            plt.xlabel('Frequency [Hz')
            plt.ylabel('%s [%s]' % (output_name, units))

            plt.hlines(y=df.loc[i, 'percentiles'].values, xmin=fbands.min(), xmax=fbands.max(),
                       label=df['percentiles'].columns)
            if log:
                plt.xscale('log')
            plt.tight_layout()
            plt.show()
            if save_path is not None:
                plt.savefig(save_path)
            plt.close()

    def plot_spectrogram(self, db=True, log=True, save_path=None, **kwargs):
        """
        Return the spectrogram of the signal (entire file)

        Parameters
        ----------
        db : boolean
            If set to True the result will be given in db. Otherwise in upa^2/Hz
        log : boolean
            If set to True the scale of the y axis is set to logarithmic
        save_path : string or Path 
            Where to save the images
        **kwargs : any attribute valid on spectrogram() function
        """
        time, fbands, t, sxx_list = self.spectrogram(db=db, **kwargs)
        for i, sxx in enumerate(sxx_list):
            # Plot the patterns
            plt.figure()
            im = plt.pcolormesh(t, fbands, sxx)
            plt.title('Spectrogram of bin %s' % (time[i].strftime("%Y-%m-%d %H:%M")))
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            if log:
                plt.yscale('log')
            if db:
                units = 'db %s upa' % self.ref
            else:
                units = 'upa'
            cbar = plt.colorbar(im)
            cbar.set_label('SPLrms [%s]' % units, rotation=90)
            plt.tight_layout()
            plt.show()
            if save_path is not None:
                plt.savefig(save_path + time[i].strftime("%Y-%m-%d %H:%M"))
            plt.close()

    def plot_spd(self, db=True, log=True, save_path=None, **kwargs):
        """
        Plot the the SPD graph of the bin

        Parameters
        ----------
        db : boolean
            If set to True the result will be given in db. Otherwise in upa^2/Hz
        log : boolean
            If set to True the scale of the y axis is set to logarithmic
        save_path : string or Path 
            Where to save the images
        **kwargs : any attribute valid on spd() function
        """
        time, fbands, percentiles, edges_list, spd_list, p_list = self.spd(db=db, **kwargs)
        if db:
            units = 'db %s upa^2/Hz' % self.ref
        else:
            units = 'upa^2/Hz'
        for i, spd in enumerate(spd_list):
            # Plot the EPD
            fig = plt.figure()
            im = plt.pcolormesh(fbands, edges_list[i], spd.T, cmap='BuPu')
            if log:
                plt.xscale('log')
            plt.title('Spectral probability density at bin %s' % time[i].strftime("%Y-%m-%d %H:%M"))
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('PSD [%s]' % units)
            cbar = fig.colorbar(im)
            cbar.set_label('Empirical Probability Density', rotation=90)

            # Plot the lines of the percentiles
            plt.plot(fbands, p_list[i], label=percentiles)

            plt.tight_layout()
            if save_path is not None:
                plt.savefig(save_path)
            else:
                plt.show()
            plt.close()


class HydroFile(AcuFile):
    def __init__(self, sfile, hydrophone, p_ref=1.0, band=None, utc=True):
        """
        Sound data recorded in a wav file with a hydrophone.

        Parameters
        ----------
        sfile : Sound file 
            Can be a path or an file object 
        hydrophone : Object for the class hydrophone
            Hydrophone used to record
        p_ref : Float
            Reference pressure in upa
        band: tuple or list
            Lowcut, Highcut. Frequency band to analyze
        """
        super().__init__(sfile, hydrophone, p_ref, band, utc)


class MEMSFile(AcuFile):
    def __init__(self, sfile, hydrophone, acc_ref=1.0, band=None, utc=True):
        """
        Acceleration data recorded in a wav file.

        Parameters
        ----------
        sfile : Sound file 
            Can be a path or an file object 
        hydrophone : Object for the class hydrophone
        acc_ref : float
            Reference acceleration in um/s
        band: tuple or list
            Lowcut, Highcut. Frequency band to analyze        
        """
        super().__init__(sfile, hydrophone, acc_ref, band, utc)

    def integrate_acceleration(self):
        """ 
        Integrate the acceleration to get the velocity of the particle. The constant is NOT resolved
        """
        if self.instant:
            raise Exception('This can only be implemented in average mode!')

        velocity = integrate.cumtrapz(self.signal('acc'), dx=1 / self.fs)

        return velocity


class MEMS3axFile:
    def __init__(self, hydrophone, xfile_path, yfile_path, zfile_path, acc_ref=1.0):
        """
        Class to treat the 3 axes together

        Parameters
        ----------
        hydrophone : Object for the class hydrophone
        xfile_path : string or Path
            File where the X accelerometer data is
        yfile_path : string or Path
            File where the Y accelerometer data is
        zfile_path : string or Path
            File where the Z accelerometer data is
        acc_ref : float
            Reference acceleration in um/s
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

        return np.sqrt(x_acc ** 2 + y_acc ** 2 + z_acc ** 2)

    def velocity_magnitude(self):
        """
        Get the start and the end velocity 
        """
        vx = self.x.integrate_acceleration()
        vy = self.y.integrate_acceleration()
        vz = self.z.integrate_acceleration()

        v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

        return v

    def mean_velocity_increment(self):
        """
        Get the mean increment of the velocity
        """
        mean_acc = self.acceleration_magnitude().mean()
        time = self.x.get_time()
        t_inc = (time()[-1] - time()[0]).total_seconds()

        return mean_acc * t_inc

    # def plot_particle_velocity(self, ax=None):
    #     """
    #     Plot the particle velocity 
    #     """
    #     # Compute the particle velocity
    #     v = self.integrate_acceleration()
    #     show = False
    #     if ax is None :
    #         fig, ax = plt.subplots(1,1)
    #         show = True
    #     # Plot 
    #     ax.plot(self.x.time(), vx, label='x')
    #     ax.plot(self.measurements.index[0 :-1], vy, label='y')
    #     ax.plot(self.measurements.index[0 :-1], vz, label='z')
    #     ax.plot(self.measurements.index[0 :-1], v_mag, label='magnitude')

    #     ax.set_title('Particle velocity')
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('um/s')
    #     ax.legend()

    #     if show : 
    #         plt.show()
    #         plt.close()
