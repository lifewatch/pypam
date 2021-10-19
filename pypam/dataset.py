import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray
from tqdm import tqdm

from pypam import acoustic_survey


class DataSet:
    def __init__(self, summary_path, output_folder, instruments, features, third_octaves=None,
                 bands_list=None, binsize=60.0, nfft=512):
        """
        A DataSet object is a representation of a group of acoustic deployments.
        It allows to calculate all the acoustic features from all the deployments and store them in a structured way
        in the output folder. The structure is as follows:
        output_folder
          - deployments : a pkl file for each deployment
          - detections : files resulting of the events detections. One folder per detection type
          - img : graphs and figures
            - temporal_features : temporal evolution of all features per deployment
            - data_overview : spatial and temporal coverage, and methods used
            - features_analysis : ??
            - spatial_features : spatial distribution of features
          dataset.netCDF : pkl with all the datasets together
        Parameters
        ----------
        summary_path : string or Path
            Path to the csv file where all the metadata of the deployments is
        output_folder : string or Path
            Where to save the output files (pkl) of the deployments with the processed data
        instruments : dictionary of (name,  instrument_object) entries
            A dictionary of all the instruments used in the deployments
        features : list of strings
            A list of all the features to be calculated
        bands_list : list of tuples
            A list of all the bands to consider (low_freq, high_freq)
        third_octaves : False or band
            If False, no octave bands are calculated. Otherwise the parameter is passed to the pypam as a band
        binsize : float
            In seconds, duration of windows to consider
        nfft : int
            Number of samples of window to use for frequency analysis
        """
        self.metadata = pd.read_csv(summary_path)
        self.summary_path = summary_path
        self.instruments = instruments
        self.features = features
        self.third_octaves = third_octaves
        self.band_list = bands_list
        self.binsize = binsize
        self.nfft = nfft

        if not isinstance(output_folder, pathlib.Path):
            output_folder = pathlib.Path(output_folder)
        self.output_folder = output_folder
        self.output_folder.joinpath('img/temporal_features').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('img/data_overview').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('img/features_analysis').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('img/spatial_features').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('deployments').mkdir(parents=True, exist_ok=True)
        self.output_folder.joinpath('detections').mkdir(parents=True, exist_ok=True)

        self.deployments_created = list(self.output_folder.joinpath('deployments').glob("*.netcdf"))
        self.dataset = {}
        
    def __call__(self):
        """
        Calculates the acoustic features of every deployment and saves them as a pickle in the deployments folder with
        the name of the station of the deployment.
        Also adds all the deployment data to the self object in the general dataset,
        and the path to each deployment's pickle in the list of deployments
        """
        for idx, name, deployment_path, in self._deployments():
            if deployment_path.exists():
                ds = xarray.open_dataset(deployment_path)
            else:
                ds = self.generate_deployment(idx=idx)
                ds.to_netcdf(deployment_path)
            self.dataset[name] = ds
        self.metadata.to_csv(self.summary_path, index=False)
    
    def _deployments(self):
        for idx in tqdm(self.metadata.index, total=len(self.metadata)):
            deployment_row = self.metadata.iloc[idx]
            deployment_path = self.output_folder.joinpath('deployments/%s_%s.netcdf' % (idx,
                                                                                        deployment_row.deployment_name))
            yield idx, deployment_row['deployment_name'], deployment_path

    def generate_deployment(self, idx):
        hydrophone = self.instruments[self.metadata.loc[(idx, 'instrument_name')]]
        hydrophone.sensitivity = self.metadata.loc[(idx, 'instrument_sensitivity')]
        hydrophone.preamp_gain = self.metadata.loc[(idx, 'instrument_amp')]
        hydrophone.Vpp = self.metadata.loc[(idx, 'instrument_Vpp')]
        asa = acoustic_survey.ASA(hydrophone, **self.metadata.loc[(idx, ['folder_path',
                                                                         'utc',
                                                                         'include_dirs',
                                                                         'calibration_time',
                                                                         'cal_freq',
                                                                         'max_cal_duration',
                                                                         'dc_subtract'])].to_dict())
        ds = xarray.Dataset()
        if self.third_octaves is not False:
            freq_evo = asa.evolution_freq_dom('third_octaves_levels', band=self.third_octaves, db=True)
            ds = ds.merge(freq_evo['oct3'])
        if self.features is not None or len(self.features) != 0:
            temporal_evo = asa.evolution_multiple(method_list=self.features, band_list=self.band_list)
            for f in self.features:
                ds = ds.merge(temporal_evo[f])
        # Update the metadata in case the calibration changed the sensitivity
        self.metadata.loc[idx, 'instrument_sensitivity'] = hydrophone.sensitivity
        return ds

    def add_deployment_metadata(self, idx):
        deployment_row = self.metadata.iloc[idx]
        hydrophone = self.instruments[deployment_row['instrument_name']]
        hydrophone.sensitivity = deployment_row['instrument_sensitivity']
        hydrophone.preamp_gain = deployment_row['instrument_amp']
        hydrophone.Vpp = deployment_row['instrument_Vpp']
        asa = acoustic_survey.ASA(hydrophone=hydrophone, folder_path=deployment_row['folder_path'])
        start, end = asa.start_end_timestamp()
        duration = asa.duration()
        self.metadata.iloc[idx, ['start', 'end', 'duration']] = start, end, duration

    def add_temporal_metadata(self):
        """
        Return a db with a data overview of the folder
        """
        metadata_params = ['start_datetime', 'end_datetime', 'duration']
        for m_p in metadata_params:
            self.metadata[m_p] = None
        for idx, _, _, in self._deployments():
            self.add_deployment_metadata(idx)
        return self.metadata

    def plot_all_features_evo(self):
        """
        Creates the images of the temporal evolution of all the features and saves them in the correspondent folder
        """
        i = 0
        for station_name, deployment in self.dataset.items():
            for feature in self.features:
                deployment[feature].plot()
                plt.title('%s %s evolution' % (station_name, feature))
                plt.savefig(
                    self.output_folder.joinpath('img/temporal_features/%s_%s_%s.png' % (i, station_name, feature)))
                plt.show()
                i += 1

    def plot_third_octave_bands_prob(self, h=1.0, percentiles=None):
        """
        Create a plot with the probability distribution of the levels of the third octave bands
        Parameters
        ----------
        h: float
            Histogram bin size (in db)
        percentiles: list of floats
            Percentiles to plot (0 to 1). Default is 10, 50 and 90% ([0.1, 0.5, 0.9])
        """
        if percentiles is None:
            percentiles = []
        percentiles = np.array(percentiles)

        bin_edges = np.arange(start=self.dataset['oct3'].min().min(), stop=self.dataset['oct3'].max().max(), step=h)
        fbands = self.dataset['oct3'].columns
        station_i = 0
        for station_name, deployment in self.dataset.items():
            sxx = deployment['oct3'].values.T
            spd = np.zeros((sxx.shape[0], bin_edges.size - 1))
            p = np.zeros((sxx.shape[0], percentiles.size))
            for i in np.arange(sxx.shape[0]):
                spd[i, :] = np.histogram(sxx[i, :], bin_edges, density=True)[0]
                cumsum = np.cumsum(spd[i, :])
                for j in np.arange(percentiles.size):
                    p[i, j] = bin_edges[np.argmax(cumsum > percentiles[j] * cumsum[-1])]
            fig = plt.figure()
            im = plt.pcolormesh(fbands, bin_edges[:-1], spd.T, cmap='BuPu', shading='auto')
            # Plot the lines of the percentiles
            plt.plot(fbands, p, label=percentiles)
            plt.xlabel('Frequency [Hz]')
            plt.xscale('log')
            plt.ylabel('$L_{rms}$ [dB]')
            cbar = fig.colorbar(im)
            cbar.set_label('Empirical Probability Density', rotation=90)
            plt.title('1/3-octave bands probability distribution %s' % station_name)
            plt.savefig(self.output_folder.joinpath('img/features_analysis/%s_%s_third_oct_prob.png' %
                                                    (station_i, station_name)))
            plt.show()
            station_i += 1

    def plot_third_octave_bands_avg(self, group_by='station_name'):
        """
        Plot the average third octave bands
        Parameters
        ----------
        group_by: string
            Column in which to separate the plots. A figure will be generated for each group
        """
        if self.third_octaves is False:
            raise Exception('This is only possible if third-octave bands have been computed!')
        self.dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        station_i = 0
        for station_name, deployment in self.dataset.groupby((group_by, 'all')):
            deployment['oct3'].mean(axis=0).plot()
            plt.title('1/3-octave bands average %s' % station_name)
            plt.xlabel('Frequency [Hz]')
            plt.xscale('log')
            plt.ylabel('Average Sound Level [dB re 1 $\mu Pa$]')
            plt.savefig(self.output_folder.joinpath('img/features_analysis/%s_%s_third_oct_avg.png' %
                                                    (station_i, station_name)))
            plt.show()
            station_i += 1
