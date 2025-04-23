import datetime
import pathlib

import dateutil
import matplotlib.pyplot as plt
import pandas as pd
import pyhydrophone as pyhy
import pytz
import seaborn as sns
import soundfile as sf

from pypam import acoustic_file
from pypam import signal as sig

pd.plotting.register_matplotlib_converters()
plt.rcParams.update({"pcolor.shading": "auto"})

# Apply the default theme
sns.set_theme()


class Detection(sig.Signal):
    """
    Detection recorded in a wav file, with start and end
    """

    def __init__(
        self,
        start_seconds: float,
        end_seconds: float,
        sfile: pathlib.Path or sf.SoundFile,
        hydrophone: pyhy.Hydrophone,
        p_ref: float,
        timezone: datetime.tzinfo
        or pytz.tzinfo.BaseTZInfo
        or dateutil.tz.tz.tzfile
        or str = "UTC",
        channel: int = 0,
        calibration: int or float or None = None,
        dc_subtract: bool = False,
    ):
        """
        Args:
            start_seconds: Seconds from start where the detection starts
            end_seconds: Seconds from start where the detection ends
            sfile: Can be a path or a file object. File to analyze
            hydrophone: Object for the class hydrophone
            p_ref: Reference pressure in upa
            timezone: Timezone where the data was recorded in
            channel: Channel to perform the calculations in
            calibration: float, -1 or None. If it is a float, it is the time ignored at the beginning of the file.
                If None, nothing is done. If negative, the function calibrate from the hydrophone is performed,
                and the first samples ignored (and hydrophone metadata attrs updated)
            dc_subtract: Set to True to subtract the dc noise (root mean squared value
        """
        self.acu_file = acoustic_file.AcuFile(
            sfile,
            hydrophone,
            p_ref,
            timezone=timezone,
            channel=channel,
            calibration=calibration,
            dc_subtract=dc_subtract,
        )
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds

        self.frame_init = int(self.acu_file.fs * self.start_seconds)
        self.frame_end = int(self.acu_file.fs * self.end_seconds)

        self.frames = self.frame_end - self.frame_init

        self.duration = self.end_seconds - self.start_seconds

        wav_sig, fs = sf.read(
            self.acu_file.file_path,
            start=self.frame_init,
            stop=min(self.frame_end, self.acu_file.file.frames),
        )

        self.orig_wav = wav_sig
        self.orig_fs = fs
        # Read the signal and prepare it for analysis
        signal_upa = self.acu_file.wav2upa(wav=wav_sig)
        super().__init__(
            signal=signal_upa, fs=self.acu_file.fs, channel=self.acu_file.channel
        )

    def save_clip(self, clip_path: str or pathlib.Path):
        """
        Save the snippet into a file (will keep original sampling rate and no filtering)

        Args:
            clip_path: path to save the clip to (.wav)

        """
        sf.write(clip_path, self.orig_wav, self.orig_fs)
