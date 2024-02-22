"""
Main PyPAM module.
"""
__version__ = "0.3.0"

from pypam.dataset import DataSet
from pypam.acoustic_survey import ASA
from pypam.acoustic_file import AcuFile
from pypam.signal import Signal
from pypam.detection import Detection


def get_pypam_version():
    """
    Get the version of the pypam package.
    """
    return __version__
