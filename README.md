# PyPam
pypam is a python package to analyze underwater sound. 
It is made to make easier the processing of underwater data stored in *.wav files. 
The main classes are AcousticFile, AcousticSurvey and DataSet. The first one is a representation of a wav file together 
with all the metadata needed to process the data (such as hydrophone used). The second one is the representation of a
folder where all the files are stored for one deployment. Here we consider a deployment as a measurement interval 
corresponding to the time when a hydrophone was in the water, without changing any recording parameters.
The Dataset is a combination of different AcousticSurveys in one dataset. This is to be used if the user has made 
several deployments and wants to process them with the same parameters.

Then pypam allows to go through all the wav files from the deployments only with one line of code. 

All the documentation can be found on [readthedocs](https://lifewatch-pypam.readthedocs.io)

## Installation
### Using pip distribution 
```bash
pip install lifewatch-pypam
```

### Using git clone

1. Clone the package
    ```bash
    git clone https://github.com/lifewatch/pypam.git
    ```
2. Use poetry to install the project dependencies
    ```bash
    poetry install
    ```
3. Build the project
    ```bash
    poetry build
    ```

## News from version 0.2.0
In version 0.2.0 we removed the detectors, because there are better maintained packages for these purposes. 

## Usage
pypam can be used to analyze a single file, a folder with files or a group of different deployments.
The available methods and features are: 
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
- time-domain features: 
  - rms 
  - dynamic_range
  - sel
  - peak 
  - rms_envelope
  - spectrum_slope
  - correlation coefficient
- frequency-domain 
  - spectrogram (also octave bands spectrogram)
  - spectrum (density or power)
  - 1/n-octave bands
  - hybrid millidecade bands
  - long-term spectrogram
- time and frequency 
  - SPD

   
pypam allows the user to choose a window chunk size (parameter binsize, in seconds), so all the features / methods 
are applied to that window. If set to None, the operations are performed along an entire file.

Futheremore, there are several plotting functions available and some signal-based operations:
- Signal operations
   - Noise reduction 
   - Downsample 
   - Band filter 
   - Envelope
   - DC noise removal

pypam deals with the calibration directly, so the output obtained is already in uPa or db! 

### General workflow 
First, we need to define our metadata by defining the hydrophone used, using 
[pyhydrophone](https://github.com/lifewatch/pyhydrophone). Check the docs of pyhydrophone to know the specific parameters needed for your hydrophone.

```python
import pyhydrophone as pyhy

# SoundTrap
model = 'SoundTrap 300 STD'
name = 'SoundTrap'
serial_number = 67416073
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)
```

Then, we need to process either a Signal, AcuFile or ASA: 

#### Acoustic File (acoustic_file.AcuFile)
```python
from pypam import acoustic_file

acu_file = acoustic_file.AcuFile('tests/test_data/67416073.210610033655.wav', soundtrap, 1)
acu_file.plot_power_spectrum()

nfft = 8000  # Set to the same as sampling rate (or higher band limit, when downsampling) for 1s time resolution 
acu_file.hybrid_millidecade_bands(nfft=nfft, fft_overlap=0.5, binsize=None, bin_overlap=0, db=True,
                                               method='density', band=None)
```

#### Acoustic Survey (acoustic_survey.ASA) 
For example, to obtain several features on a certain binsize, at three different frequency bands:
```python
from pypam import acoustic_survey

# Analysis parameters
features = ['rms', 'sel', 'peak', 'aci']
band_list = [[10, 100], [500, 1000], [500, 100000]]
binsize = 60.0

asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path='/tests/test_data', binsize=binsize)
features_ds = asa.evolution_multiple(method_list=features, band_list=band_list)
```

Another example would be to obtain the third octave bands:
```python
from pypam import acoustic_survey

# Analysis parameters
binsize = 60.0
third_octaves = None  # Calculate third octaves for the entire freq range

asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path='/tests/test_data', binsize=binsize)
oct_ds = asa.evolution_freq_dom('third_octaves_levels', band=third_octaves, db=True)
```

#### Save the output 
Finally, the output can be saved as a netCDF file. The output should contain all the metadata necessary to reproduce 
the results, such as all the metadata passed to the functions. 

The saved files can afterwards be loaded in pypam to produce plots.

```python
oct_ds.to_netcdf('path_to_the_file.nc')
```

### Acoustic Dataset (dataset.Dataset)
Alternatively, we can process several deployments at once using the Dataset class.
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
summary_path = pathlib.Path('./data_summary_example.csv')
include_dirs = False

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
preamp_gain = -170
bk_Vpp = 2.0
bk = pyhy.BruelKjaer(name=bk_name, model=bk_model, preamp_gain=preamp_gain, Vpp=bk_Vpp, serial_number=1,
                     type_signal='ref')

upam_model = 'uPam'
upam_name = 'Seiche'
upam_serial_number = 'SM7213'
upam_sensitivity = -196.0
upam_preamp_gain = 0.0
upam_Vpp = 20.0
upam = pyhy.uPam(name=upam_name, model=upam_name, serial_number=upam_serial_number, sensitivity=upam_sensitivity,
                 preamp_gain=upam_preamp_gain, Vpp=upam_Vpp)


instruments = {'SoundTrap': soundtrap, 'uPam': upam, 'B&K': bk}

# SURVEY PARAMETERS
nfft = 4096
fft_overlap = 0.5
binsize = 60.0
bin_overlap = 0.0
overlap = 0.5
dc_subtract = False
band_lf = [50, 500]
band_list = [band_lf]
temporal_features = ['rms', 'sel', 'aci']
frequency_features = ['third_octaves_levels']

# Create the dataset object
ds = pypam.dataset.DataSet(summary_path, output_folder, instruments, temporal_features=temporal_features,
                           frequency_features=frequency_features, bands_list=band_list, binsize=binsize,
                           bin_overlap=bin_overlap, nfft=nfft, fft_overlap=fft_overlap, dc_subtract=dc_subtract)

# Call the dataset creation. Will create the files in the corresponding folder
ds()
```

## Plots 
There are functions to plot compute and plot in one-go, such as: 

```python
h_db = 1
percentiles = [1, 10, 50, 90, 95]
min_val = 60
max_val = 140

# ASA defined before
asa.plot_spd(db=True, h=h_db, percentiles=percentiles, min_val=min_val, max_val=max_val)
```

But there is also the module plots, which allows to pass the saved *.nc files to produce plots. 


# Under development 
Planned:
- Add function to generate files per included folder (too big deployments)
- Add options for the user to choose what to do when the blocksize is not multiple of the frames, 
and to deal with time keeping
- Add a logger that logs the code that was run and the warnings together with the output
- Add deep learning features (vggish and compatibility with koogu and AVES)
- Add parallel processing options 
- Add support for frequency calibration
- Support for reading detections 
   

## Cite
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6044593.svg)](https://doi.org/10.5281/zenodo.6044593)
