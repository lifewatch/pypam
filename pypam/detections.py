__author__ = "Clea Parcerisas"
__version__ = "0.1"
__credits__ = "Clea Parcerisas"
__email__ = "clea.parcerisas@vliz.be"
__status__ = "Development"

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import soundfile as sf
try:
    from oceansoundscape import raven
except ModuleNotFoundError:
    raven = None
from tqdm import tqdm

from pypam import signal as sig
from pypam import acoustic_file

pd.plotting.register_matplotlib_converters()
plt.rcParams.update({'pcolor.shading': 'auto'})

# Apply the default theme
sns.set_theme()


class Detection(sig.Signal):
    """
    Detection recorded in a wav file, with start and end

    Parameters
    ----------
    sfile : Sound file
        Can be a path or a file object
    hydrophone : Object for the class hydrophone
    p_ref : Float
        Reference pressure in upa
    timezone: datetime.tzinfo, pytz.tzinfo.BaseTZInfo, dateutil.tz.tz.tzfile, str or None
        Timezone where the data was recorded in
    channel : int
        Channel to perform the calculations in
    calibration: float, -1 or None
        If it is a float, it is the time ignored at the beginning of the file. If None, nothing is done. If negative,
        the function calibrate from the hydrophone is performed, and the first samples ignored (and hydrophone updated)
    dc_subtract: bool
        Set to True to subtract the dc noise (root mean squared value
    start_seconds: float
        Seconds from start where the detection starts
    end_seconds: float
        Seconds from start where the detection ends
    """

    def __init__(self, start_seconds, end_seconds, sfile, hydrophone, p_ref, timezone='UTC', channel=0,
                 calibration=None, dc_subtract=False):

        self.acu_file = acoustic_file.AcuFile(sfile, hydrophone, p_ref, timezone=timezone, channel=channel,
                                              calibration=calibration, dc_subtract=dc_subtract)
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds

        self.frame_init = int(self.acu_file.fs * self.start_seconds)
        self.frame_end = int(self.acu_file.fs * self.end_seconds)

        self.frames = self.frame_end - self.frame_init

        self.duration = self.end_seconds - self.start_seconds

        wav_sig, fs = sf.read(self.acu_file.file_path, start=self.frame_init, stop=min(self.frame_end,
                                                                                       self.acu_file.file.frames))

        self.orig_wav = wav_sig
        self.orig_fs = fs
        # Read the signal and prepare it for analysis
        signal_upa = self.acu_file.wav2upa(wav=wav_sig)
        super().__init__(signal=signal_upa, fs=self.acu_file.fs, channel=self.acu_file.channel)

    def save_clip(self, clip_path):
        """
        Save the snippet into a file (will keep original sampling rate and no filtering)

        Parameters
        ----------
        clip_path: str or Path
            path to save the clip to (.wav)

        """
        sf.write(clip_path, self.orig_wav, self.orig_fs)


class DetectionsDF:
    """
    DataFrame containing a list of detections
    """
    def __init__(self):
        self.df = pd.DataFrame()

    def load_csv(self, csv_path):
        new_csv = pd.read_csv(csv_path)
        self.df = pd.concat([self.df, new_csv])

    def load_from_raven(self, bled_file: pathlib.Path, call_conf: dict, max_samples: int, sampling_rate: int,
                        exclude_unlabeled: bool = True):
        if raven is None:
            raise Exception('You need to install oceansoundscape to be able to use the Raven functionality')
        new_df = raven.BLEDParser(bled_file, call_conf, max_samples, sampling_rate, exclude_unlabeled)
        self.df = pd.concat([self.df, new_df])

    def detections(self, hydrophone, p_ref=1.0, timezone='UTC', channel=0, calibration=None, dc_subtract=False):
        for i, detection_row in tqdm(self.df.iterrows(), total=len(self.df)):
            folder_path = pathlib.Path(detection_row['folder_path'])
            file_path = folder_path.joinpath(detection_row['file_name'])

            start_seconds = detection_row['start_time'].total_seconds()
            end_seconds = detection_row['end_time'].total_seconds()

            detection = Detection(start_seconds, end_seconds, file_path, hydrophone,
                                  p_ref, timezone=timezone, channel=channel, calibration=calibration,
                                  dc_subtract=dc_subtract)
            yield i, detection

    def plot_spectrograms_detections(self, band=None, downsample=True, max_duration_time=None, output_folder=None,
                                     save_clips=False):
        for i, d in self.detections():
            try:
                duration = d.duration
                if max_duration_time is not None:
                    if duration > max_duration_time:
                        end_seconds = d.start_seconds + max_duration_time
                    elif duration < 1:
                        extra = 1 - duration
                        end_seconds += extra  # add only at the end because apparently we are biased when annotating
                d.set_band(band, downsample=downsample)
                if output_folder is not None:
                    spectrogram_path = output_folder.joinpath('%s_%s.png' % (i, d.label))
                else:
                    spectrogram_path = None
                d.plot(save_path=spectrogram_path, show=False, nfft=128, overlap=0.7, log=False)

                if save_clips:
                    clip_path = output_folder.joinpath('%s_%s.wav' % (i, d.label))
                    d.save_clip(clip_path)

            except FileNotFoundError:
                print('Detection %s was not produced because file was not found' % i)
