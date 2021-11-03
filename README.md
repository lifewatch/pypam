# PyPam
pypam is a python package to analyze underwater sound. 
It is made to make easier the processing of underwater data stored in *.wav files. 
The main classes are AcousticFile, AcousticSurvey and DataSet. The first one is a representation of a wav file together 
with all the metadata needed to process the data (such as hydrophone used). The second one is the representation of a
folder where all the files are stored. The Dataset is a combination of different AcousticSurveys in one dataset.
Then pypam allows to go through all the wav files from the deployments only with one line of code. 

## Installation
# using pip distribution 
```
    pip install lifewatch-pypam
```

### Using cit clone
This installation process assumes you have ssh keys on github setup correctly. In case you don't know how to do it, 
here is a nice tutorial:  https://www.inmotionhosting.com/support/server/ssh/how-to-add-ssh-keys-to-your-github-account/

We strongly advise to use a proper python virtualenv environment to keep package dependencies nicely non conflicting

If your ssh keys are set up correctly, you should be able to install the requirements by: 
```
    pip install -r requirements.txt 
```
(pyhydrophone will be automatically downloaded and installed using your ssh key)

In case you can't set up your ssh keys working, don't panic! Try this way:
1. Install FIRST pyhydrophone (follow the instructions): https://github.com/lifewatch/pyhydrophone

2. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the remaining dependencies
    ```bash
    pip install -r requirements.txt 
    ```
3. Build the project
    ```bash
    python setup.py install
    ```

## Usage
pypam can be used to analyze a single file, a folder with files or a group of different deployments.
The available methods and features are: 
- Events detection: 
  - Ship detection 
  - Pile driving detection 
- Acoustic Indices: 
  - ACI 
  - BI 
  - SH 
  - TH 
  - NDSI
  - AEI 
  - ADI 
  - Zero crossing (average)
  - BN peaks 
- Features: 
  - rms 
  - dynamic_range
  - sel
  - peak 
  - rms_envelope
  - spectrum_slope
  - correlation coefficient
- Frequency domain 
  - spectrogram (also octave bands spectrogram)
  - spectrum
  - spectral probability density (SPD)
  - 1/3-octave bands 
  - octave bands
- Plots
  - Signal and spectrogram 
  - Spectrum evolution 
  - SPD 
- Signal operations
   - Noise reduction 
   - Downsample 
   - Band filter 
   - Envelope
   - DC noise removal
- Other 
    - Calibration signal detection (and recalibration of the signal)
   
pypam allows the user to choose a window bin size (in seconds), so all the features / methodologies are applied to that
window. If set to None, the operations are performed along an entire file.

### Acoustic analysis
```python
import pyhydrophone as pyhy
from pypam import acoustic_survey

# SoundTrap
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

# Analysis parameters
features = ['rms', 'sel', 'peak', 'aci']
band_list = [[10, 100], [500, 1000], [500, 100000]]
third_octaves = None  # Calculate third octaves for the entire freq range

asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path='./../tests/test_data', binsize=60.0)
features_ds = asa.evolution_multiple(method_list=features, band_list=band_list)
oct_ds = asa.evolution_freq_dom('third_octaves_levels', band=third_octaves, db=True)
```

### Acoustic Dataset
To create an acoustic dataset made out of several deployments (with different metadata), first it is necessary to 
create a csv file where each row is a deployment. You can find an example in docs/data_summary_example.csv. There is 
also a test file in tests/test_data/data_summary.csv. 
This metadata information will be at one point linked with the download output of ETN Underwater Acoustics 
(https://www.lifewatch.be/etn/), but now the csv has to be manually created.

So far, all the fields up to dc_subtract (see example) have to be present in the csv file (even if they are left blank). 
If some extra metadata should be added per deployment, then columns can be added (in the example, etn_id, 
instrument_depth and method).

A Dataset is a conjunction of AcousticSurveys to be studied together. The output is always in a structured folder.
* output_folder/
    * deployments/: contains one netcdf file per deployment processed
    * detections/: contains the output of the detectors (if applicable)
    * img/: contains graphs created (if applicable)
        * data_overview/: general plots 
        * features_analysis/: stats from the features
        * temporal_features/: graphs in time domain of the features 
        * spatial_features/: spatial plots (in a map) of the features

```python 
import pathlib

import pyhydrophone as pyhy
import pypam

# Acoustic Data
summary_path = pathlib.Path('./../tests/test_data/data_summary.csv')

# Output folder
output_folder = summary_path.parent.joinpath('data_exploration')

# Hydrophone Setup
# If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

bk_model = 'Nexus'
bk_name = 'B&K'
amplif0 = 10e-3
bk = pyhy.BruelKjaer(name=bk_name, model=bk_model, amplif=amplif0, serial_number=1)

upam_model = 'uPam'
upam_name = 'Seiche'
upam_serial_number = 'SM7213'
upam_sensitivity = -196.0
upam_preamp_gain = 0.0
upam_Vpp = 20.0
upam = pyhy.Seiche(name=upam_name, model=upam_name, serial_number=upam_serial_number, sensitivity=upam_sensitivity,
                   preamp_gain=upam_preamp_gain, Vpp=upam_Vpp)


instruments = {'SoundTrap': soundtrap, 'uPam': upam, 'B&K': bk}

# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 4096
binsize = 60.0
band_lf = [50, 500]
band_mf = [500, 2000]
band_hf = [2000, 20000]
band_list = [band_lf]
features = ['rms', 'sel', 'aci']
third_octaves = None

# Create the dataset object
ds = pypam.DataSet(summary_path, output_folder, instruments, features, third_octaves, band_list, binsize,
                   nfft)
# Call the dataset creation. Will create the files in the corresponding folder
ds()
```
   

## Cite
DOI: 10.5281/zenodo.5031690