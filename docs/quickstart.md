# Introduction to *pypam*

## Concepts

The idea of *pypam* is to process all the acoustic data resulting from an
underwater acoustic deployment. For a deployment, we understand a single
instrument from the moment it gets into the water until the moment it is
taken out. This data is usually stored in different files (and folders), and usually these files have a
continuity in time, and very often it is interesting to extract time
series.

*pypam* allows to choose a time window (currently named binsize) for processing. 
The selected binsize will be the time resolution of the computed output. Each feature will be
computed on that time window (independently of the file duration and sampling frequency). 

The output will be in the form of an [xarray](https://xarray.dev/) dataset. The dimensions of these arrays are always:

- *id*: id of the bin for the total deployment (increasing integer)

Extra coordinates (metadata) on the same dimension that *id*:

- *datetime*: bin start timestamp of the "time window"
- *start_sample*: start sample of the bin respect to the file
- *end_sample*: end sample of the bin respect to the file
- *file_id*: id with respect to the file (changes when multiple files per
  deployment)

For frequency dependent features, there coordinates are also present: 

- *frequency*: center frequency of the band 
- *frequency_bins* (only for hybrid millidecade bands): center frequency of joined bands
- *upper_frequency*: upper limit of the frequency band
- *lower_frequency*: lower limit of the frequency band

The processed acoustic features are then stored in one data variable of the xarray dataset per feature. 


## File format 
Pypam allows for any file format supported by [SoundFile](https://python-soundfile.readthedocs.io). 


Sometimes the collected underwater acoustics data is zipped to save storage space. Don't
panic, you can still work with zipped folders directly from *pypam*
without having to unzip the whole deployment. 

## General workflow

First, we need to define our metadata by defining the hydrophone used,
using [pyhydrophone](<https://github.com/lifewatch/pyhydrophone>), which deals with the complexities of specific brands
of hydrophones. Check the docs of pyhydrophone to know the specific parameters needed for your hydrophone, 
and to know if the brand is supported (if not, you can always raise an issue asking for it!).

A typical example would be as follows:
```python
    import pyhydrophone as pyhy

    # SoundTrap
    model = 'SoundTrap 300 STD'
    name = 'SoundTrap'
    serial_number = 67416073
    soundtrap = pyhy.soundtrap.SoundTrap(name=name, model=model, serial_number=serial_number)
```
Once the hydrophone is declared, we can process the data. 
This can be done using one of the four main classes: 

| Class Equivalent | Class Object Name                       | Description                                                                                                                                                                                             |
|------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Signal           | [Signal](signal.md)                     | The signal extracted from one (or a part of) recording                                                                                                                                                  |
| One sound file   | [AcuFile](acufile.md)  (Acoustic File)  | One single audio file, with its associated metadata                                                                                                                                                     | 
| One Deployment   | [ASA](asa.md) (Acoustic Survey Analyis) | All the files contained in one deployment (continuous or not in time), stored in one folder (can contain multiple subfolders)                                                                           |
| A full dataset   | [Dataset](dataset.md)                   | A conjunction of deployments which need to be processed the same way. For this, a csv with metadata needs to be created first specifying the location of each deployment and its corresponding metadata |

## Acoustic File ([acoustic_file.AcuFile](acufile.md))

To create or process an acoustic file, you can run:
```python
    from pypam import acoustic_file

    acu_file = acoustic_file.AcuFile('tests/test_data/67416073.210610033655.wav', soundtrap, 1)
    acu_file.plot_power_spectrum()

    nfft = 8000  # Set to the same as sampling rate (or higher band limit, when downsampling) for 1s time resolution
    acu_file.hybrid_millidecade_bands(nfft=nfft, fft_overlap=0.5, binsize=None, bin_overlap=0, db=True,
                                                   method='density', band=None)
```

## Acoustic Survey ([acoustic_survey.ASA](asa.md))

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

### Save the results (processed data + metadata)

Finally, the output can be saved as a netCDF file. The output will contain all the metadata necessary to reproduce 
the results, such as all the metadata and selected parameters specified by the user.: 

```python
    oct_ds.to_netcdf('path_to_the_file.nc')
```

The saved files can afterward be loaded to analyze, or to be used with *pypam* utils / plots modules to produce plots.:

```python
    ds = xarray.load_dataset('path_to_the_file.nc')
```

## Acoustic Dataset ([dataset.Dataset](dataset.md))

Alternatively, we can process several deployments at once using the
Dataset class. To create an acoustic dataset made out of several
deployments (with different metadata), first it is necessary to create a
csv file where each row is a deployment. You can find an example in
docs/data_summary_example.csv. There is also a test file in
tests/test_data/data_summary.csv. This metadata information will be at
one point linked with the download output of ETN Underwater Acoustics
(<https://www.lifewatch.be/etn/>), but now the csv has to be manually
created.

So far, all the fields up to dc_subtract (see example) have to be
present in the csv file (even if they are left blank). If some extra
metadata should be added per deployment, then columns can be added (in
the example, etn_id, instrument_depth and method).

A Dataset is a conjunction of AcousticSurveys to be studied together.
The output is always in a structured folder.

``` none
📁 output_folder
└─ 📁 deployments/: contains one netcdf file per deployment processed
└─ 📁 detections/: contains the output of the detectors (if applicable)
└─ 📁 img/: contains graphs created (if applicable)
    ├─ 📁 data_overview/: general plots
    ├─ 📁 features_analysis/: stats from the features
    ├─ 📁 temporal_features/: graphs in time domain of the features
    └─ 📁 spatial_features/: spatial plots (in a map) of the features
```

To create a dataset, run:

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

## To produce plots

There are functions to plot compute and plot in one-go, such as:

```python
    h_db = 1
    percentiles = [1, 10, 50, 90, 95]
    min_val = 60
    max_val = 140

    # ASA defined before
    asa.plot_spd(db=True, h=h_db, percentiles=percentiles, min_val=min_val, max_val=max_val)
```

If you want to plot data which has already been processed, you can use the [plots](plots.md) module. 
This module which allows to pass the saved *.nc files to produce specific plots of interest.
