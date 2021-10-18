# PyPam
pypam is a python package to analyze underwater sound. 
It is made to make easier the processing of underwater data stored in *.wav files. 
The main classes are AcusticFile, AcousticSurvey and DataSet. The first one is a representation of a wav file together 
with all the metadata needed to process the data (such as hydrophone used). The second one is the representation of a
folder where all the files are stored. Then pypam allows to go through all the wav files from the deployments only with
one line of code. 

## Installation
1. Install pyhydrophone (follow the instructions): https://github.com/lifewatch/pyhydrophone
2. Install geopandas (only necessary if you will work with geospatial data). If you are working on Windows, it can be
 tricky to install it. We recommend to install FIRST the following packages (in this order) by downloading the 
 wheels of each package. You can follow this tutorial if you're not familiar with wheels and/or pip: 
 https://geoffboeing.com/2014/09/using-geopandas-windows/: 
    * GDAL
    * rasterio
    * Fiona

3. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the remaining dependencies
    ```bash
    pip install -r requirements.txt 
    ```
4. Build the project
    ```bash
    python setup.py install
    ```


## Usage
pypam can be used to analyze a single file, a folder with files or a group of different deployments (coming soon).

```python
import pyhydrophone as pyhy
from pypam import acoustic_survey

# Soundtrap
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 0000
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

# Analysis parameters
features = ['rms', 'sel', 'peak', 'aci']
band_list = [[10,100], [500, 1000], [500, 100000]]
third_octaves = None  # Calculate third octaves for the entire freq range

asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path='', binsize=60.0)
df_features = asa.evolution_multiple(method_list=features, band_list=band_list)

df_3oct = asa.evolution_freq_dom('third_octaves_levels', band=third_octaves, db=True)
```

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
- Operations
   - Noise reduction 
   - Downsample, filtering and cropping 
   - Envelope
   - Calibration signal detection
   
## Output
The output of the data generation is a pandas DataFrame with datetime as index and a multiindex column 
with all the acoustic features as a multiindex column.

## Generation of datasets
pypam can be called by using the function generate_deployment_data from the Deployment class to generate the data from one 
deployment, or generate_entire_dataset when multiple deployments have to be processed. 
When multiple deployments have to be processed, the metadata has to be summarized in a csv file. Each row is a 
deployment, and an example can be found at docs/data_summary_example.csv.
This metadata information will be at one point linked with the download output of ETN Underwater Acoustics 
(https://www.lifewatch.be/etn/).

The idea is to be able to process several deployments only with one command. 
All the deployments have to be listed in a csv file with all the metadata, an example
is provided in docs/data_summary_example.csv.
With this objective there are the classes Dataset and Deployment. 
A deployment represents a period of data acquisition with constant metadata (same instrument and 
instrument settings). A Dataset is a conjunction of deployments to be studied together.
The output is always in a structured folder.
* output_folder/
    * deployments/: contains one pickle file per deployment processed
    * detections/: contains the output of the detectors (if applicable)
    * img/: contains graphs created (if applicable)
        * data_overview/: general plots 
        * features_analysis/: stats from the features
        * temporal_features/: graphs in time domain of the features 
        * spatial_features/: spatial plots (in a map) of the features
    * dataset.pkl : dataset containing all the deployments from deployments/
   
 In the next versions, the output will be changed to hdf5 for better interchangeability with other programming 
 languagues

# Use example  
```python  
    # Acoustic Data
    summary_path = pathlib.Path('docs/data_summary_example.csv')
    include_dirs = False
    
    # Output folder
    output_folder = summary_path.parent.joinpath('data_exploration')
    
    # Hydrophone Setup
    # If Vpp is 2.0 then it means the wav is -1 to 1 directly related to V
    model = 'ST300HF'
    name = 'SoundTrap'
    serial_number = 67416073
    soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)
    
    # All the instruments from the dataset can be added in the dictionary
    instruments = {'SoundTrap': soundtrap}
    
    # Acoustic params. Reference pressure 1 uPa
    REF_PRESSURE = 1e-6
    
    # SURVEY PARAMETERS. Look into pypam documentation for further information
    nfft = 4096
    binsize = 5.0
    band_lf = [50, 500]
    band_mf = [500, 2000]
    band_hf = [2000, 20000]
    band_list = [band_lf, band_mf, band_hf]
    
    # Features can be any of the features that can be passed to pypam
    features = ['rms', 'sel', 'aci']
    
    # Third octaves can be None (for broadband analysis), a specific band [low_freq, high_freq], for 
    only certain band analysis or False if no computation is wanted
    third_octaves = False
    
    env_vars = ['spatial_data', 'sea_state', 'time_data', 'sea_bottom', 'shipping', 'shipwreck']
    
    # Generate the dataset
    dataset = dataset.DataSet(summary_path, output_folder, instruments, features, third_octaves, band_list, binsize, nfft)
    dataset.generate_entire_dataset(env_vars=env_vars)
   ```
## Cite
DOI: 10.5281/zenodo.5031690