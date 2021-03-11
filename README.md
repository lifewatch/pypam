# PyPam
pypam is a python package to analyze underwater sound. 
It is made to make easier the processing of underwater data stored in *.wav files. 
The main classes are AcusticFile and AcousticSurvey. The first one is a representation of a wav file together 
with all the metadata needed to process the data (such as hydrophone used). The second one is the representation of a
folder where all the files are stored. Then pypam allows to go through all the wav files from the deployments only with
one line of code. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install 
the dependencies 

```bash
pip install -r requirements.txt 
```

Build the project

```bash
python setup.py install
```

## Usage

```bash
import pyhydrophone as pyhy

# Soundtrap
model = 'ST300HF'
name = 'SoundTrap'
serial_number = 0000
soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)

# Analysis parameters
features = ['rms', 'sel', 'peak', 'aci']
band_list = [[10,100], [500, 1000], [500, 100000]]

asa = acoustic_survey.ASA(hydrophone=soundtrap, folder_path='', binsize=60.0)
df_output = asa.evolution_multiple(method_list=features, band_list=band_list)
```
