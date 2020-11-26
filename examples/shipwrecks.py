import pathlib
import geopandas
import pandas as pd
import pyhydrophone as pyhy
import matplotlib.pyplot as plt

from pypam import acoustic_survey, geolocation


# Sound Analysis
data_folder = pathlib.Path('C:/Users/cleap/Documents/PhD/Classifiers/sounnscapes/Data/recordings')
save_path_spl = pathlib.Path('C:/Users/cleap/Documents/PhD/Classifiers/sounnscapes/Data/10sec_analysis_spl.pkl')
save_path_aci = pathlib.Path('C:/Users/cleap/Documents/PhD/Classifiers/sounnscapes/Data/5sec_analysis_aci.pkl')
save_path_coast = pathlib.Path('C:/Users/cleap/Documents/PhD/Classifiers/sounnscapes/Data/distance_to_coast.pkl')
save_fig_path = pathlib.Path('C:/Users/cleap/Documents/PhD/Projects/COVID-19/distance_vs_params')
save_map_path = pathlib.Path('C:/Users/cleap/Documents/PhD/Projects/COVID-19/distribution')
zipped = False
include_dirs = False

# GPS Location data
gps_path = pathlib.Path("C:/Users/cleap/Documents/Data/Tracks/COVID-19/Track_2020-05-18 092212.gpx")
oostende = {'Lat': 51.237421, 'Lon': 2.921875}
coastfile = pathlib.Path("C:/Users/cleap/Documents/Data/Maps/basislijn_BE.shp")

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


# Acoustic params. Reference pressure 1 uPa
REF_PRESSURE = 1e-6

# SURVEY PARAMETERS
nfft = 512
binsize = 5.0
# h = 0.1
# percentiles = [10, 50, 90]
# period = None
band_lf = [100, 500]
band_mf = [500, 2000]
band_hf = [2000, 20000]


def generate_spl_dataset():
    """
    Generate a dataset in a pandas data frame with the spl values
    """
    # Read the metadata
    metadata = pd.read_csv(data_folder.joinpath('metadata.csv'))
    metadata = metadata.set_index('Location')

    # Read the gpx (time is in UTC)
    geoloc = geolocation.SurveyLocation(gps_path)

    spl_data = pd.DataFrame(columns=['location', 'datetime', 'geometry', 'shipwreck_dist', 'spl_lf',
                                     'spl_mf', 'spl_hf'])
    for shipwreck_dir in data_folder.glob('**/*/'):
        if shipwreck_dir.is_dir():
            shipwreck_name = shipwreck_dir.parts[-1]
            shipwreck_metadata = metadata.loc[shipwreck_name]
            if shipwreck_metadata['shipwreck'] == 'shipwreck':
                if shipwreck_metadata['Instrument'] == 'SoundTrap ST300HF':
                    hydrophone = soundtrap
                elif shipwreck_metadata['Instrument'] == 'B&K Nexus':
                    hydrophone = bk
                    bk.sensitivity = shipwreck_metadata['sensitivity']
                else:
                    raise Exception('This hydrophone is not defined!')

                lf_asa = acoustic_survey.ASA(hydrophone, shipwreck_dir, binsize=binsize, nfft=nfft, band=band_lf)
                mf_asa = acoustic_survey.ASA(hydrophone, shipwreck_dir, binsize=binsize, nfft=nfft, band=band_mf)
                hf_asa = acoustic_survey.ASA(hydrophone, shipwreck_dir, binsize=binsize, nfft=nfft, band=band_hf)

                lf_evo = lf_asa.evolution('rms', dB=True)
                mf_evo = mf_asa.evolution('rms', dB=True)
                hf_evo = hf_asa.evolution('rms', dB=True)

                shipwreck_data = pd.DataFrame({
                                                'location': shipwreck_name,
                                                'datetime': lf_evo.index.values,
                                                'spl_lf': lf_evo.rms.values,
                                                'spl_mf': mf_evo.rms.values,
                                                'spl_hf': hf_evo.rms.values,
                                                })
                shipwreck_data_loc = geoloc.add_distance_to(df=shipwreck_data, lat=shipwreck_metadata['Lat'],
                                                            lon=shipwreck_metadata['Lon'],
                                                            column='shipwreck_dist')[spl_data.columns]

                # Plot the map
                geoloc.plot_survey_color('spl_lf', units='dB', df=shipwreck_data_loc,
                                         save_path=save_fig_path.joinpath(shipwreck_name+'_map_SPL.png'))

                # Plot SPL vs distance to shipwreck
                plt.figure()
                plt.scatter(x=shipwreck_data_loc['shipwreck_dist'], y=shipwreck_data_loc['spl_lf'], label='LF')
                plt.scatter(x=shipwreck_data_loc['shipwreck_dist'], y=shipwreck_data_loc['spl_mf'], label='MF')
                plt.scatter(x=shipwreck_data_loc['shipwreck_dist'], y=shipwreck_data_loc['spl_hf'], label='HF')
                plt.xlabel('Distance [m]')
                plt.ylabel('SPL [dB]')
                plt.title('SPL vs distance to shipwreck %s' % shipwreck_name)
                plt.ylim(ymax=140, ymin=60)
                plt.legend()
                plt.savefig(save_fig_path.joinpath(shipwreck_name+'_SPL.png'))
                plt.close()

                spl_data = spl_data.append(shipwreck_data_loc, ignore_index=True)
                # spl_data.to_pickle(save_path_spl)

    return spl_data


def generate_aci_dataset():
    """
    Generate a dataset in a pandas data frame with the spl values
    """
    # Read the metadata
    metadata = pd.read_csv(data_folder.joinpath('metadata.csv'))
    metadata = metadata.set_index('Location')

    # Read the gpx (time is in UTC)
    geoloc = geolocation.SurveyLocation(gps_path)

    aci_data = pd.DataFrame(columns=['location', 'datetime', 'geometry', 'shipwreck_dist',
                                     'aci_lf', 'aci_mf', 'aci_hf'])
    for shipwreck_dir in data_folder.glob('**/*/'):
        if shipwreck_dir.is_dir():
            shipwreck_name = shipwreck_dir.parts[-1]
            shipwreck_metadata = metadata.loc[shipwreck_name]
            if shipwreck_metadata['shipwreck'] == 'shipwreck':
                if shipwreck_metadata['Instrument'] == 'SoundTrap ST300HF':
                    hydrophone = soundtrap
                elif shipwreck_metadata['Instrument'] == 'B&K Nexus':
                    hydrophone = bk
                    bk.sensitivity = shipwreck_metadata['sensitivity']
                else:
                    raise Exception('This hydrophone is not defined!')

                lf_asa = acoustic_survey.ASA(hydrophone, shipwreck_dir, binsize=binsize, nfft=nfft, band=band_lf)
                mf_asa = acoustic_survey.ASA(hydrophone, shipwreck_dir, binsize=binsize, nfft=nfft, band=band_mf)
                hf_asa = acoustic_survey.ASA(hydrophone, shipwreck_dir, binsize=binsize, nfft=nfft, band=band_hf)

                lf_evo = lf_asa.evolution('aci')
                mf_evo = mf_asa.evolution('aci')
                hf_evo = hf_asa.evolution('aci')

                shipwreck_data = pd.DataFrame({
                                                'location': shipwreck_name,
                                                'datetime': lf_evo.index.values,
                                                'aci_lf': lf_evo.aci.values,
                                                'aci_mf': mf_evo.aci.values,
                                                'aci_hf': hf_evo.aci.values,
                                                })

                shipwreck_data_loc = geoloc.add_distance_to(df=shipwreck_data, lat=shipwreck_metadata['Lat'],
                                                            lon=shipwreck_metadata['Lon'],
                                                            column='shipwreck_dist')[aci_data.columns]

                # Plot the map
                # geoloc.plot_survey_color('spl_lf', units='dB', df=shipwreck_data_loc)

                # Plot ACI vs distance to shipwreck
                plt.figure()
                plt.scatter(x=shipwreck_data_loc['shipwreck_dist'], y=shipwreck_data_loc['aci_lf'], label='LF')
                plt.scatter(x=shipwreck_data_loc['shipwreck_dist'], y=shipwreck_data_loc['aci_mf'], label='MF')
                plt.scatter(x=shipwreck_data_loc['shipwreck_dist'], y=shipwreck_data_loc['aci_hf'], label='HF')
                plt.xlabel('Distance [m]')
                plt.ylabel('ACI [dB]')
                plt.title('ACI vs distance to shipwreck %s' % shipwreck_name)
                plt.legend()
                plt.savefig(save_fig_path.joinpath(shipwreck_name+'_ACI.png'))
                plt.close()

                aci_data = aci_data.append(shipwreck_data_loc, ignore_index=True)
                aci_data.to_pickle(save_path_aci)

    return aci_data


def add_distance_to_coast():
    """
    Calculate the distance to the coast for 5 rows of each shipwreck
    """
    geoloc = geolocation.SurveyLocation(gps_path)
    dataset = pd.read_pickle(save_path_spl)
    dataset = geopandas.GeoDataFrame(dataset)
    new_dataset = geopandas.GeoDataFrame()
    for shipwreck in dataset.location.unique():
        new_dataset = new_dataset.append(dataset.loc[dataset.location == shipwreck].sample(n=5))
    new_dataset = geoloc.add_distance_to_coast(df=new_dataset, coastfile=coastfile)
    new_dataset = geoloc.add_distance_to(df=new_dataset, lat=oostende['Lat'], lon=oostende['Lon'], column='port_dist')

    # Plot SPL vs distance to coast
    plt.figure()
    plt.scatter(x=new_dataset['coast_dist'], y=new_dataset['spl_lf'], label='LF')
    plt.scatter(x=new_dataset['coast_dist'], y=new_dataset['spl_mf'], label='MF')
    plt.scatter(x=new_dataset['coast_dist'], y=new_dataset['spl_hf'], label='HF')
    plt.xlabel('Distance [m]')
    plt.ylabel('SPL [dB]')
    plt.title('SPL vs distance to Coast')
    plt.ylim(ymax=140, ymin=60)
    plt.legend()
    plt.savefig(save_fig_path.joinpath('SPL_vs_distance_coast.png'))
    plt.show()
    plt.close()

    # Plot SPL vs Oostende distance
    plt.figure()
    plt.scatter(x=new_dataset['port_dist'], y=new_dataset['spl_lf'], label='LF')
    plt.scatter(x=new_dataset['port_dist'], y=new_dataset['spl_mf'], label='MF')
    plt.scatter(x=new_dataset['port_dist'], y=new_dataset['spl_hf'], label='HF')
    plt.xlabel('Distance [m]')
    plt.ylabel('SPL [dB]')
    plt.title('SPL vs distance to Oostende Port')
    plt.ylim(ymax=140, ymin=60)
    plt.legend()
    plt.savefig(save_fig_path.joinpath('SPL_vs_distance_oost.png'))
    plt.close()

    new_dataset.to_pickle(save_path_coast)


def plot_aci_distribution():
    """
    Plot the distribution on the map
    """
    geoloc = geolocation.SurveyLocation(gps_path)
    dataset = pd.read_pickle(save_path_aci)
    dataset = geopandas.GeoDataFrame(dataset, crs=geoloc.geotrackpoints.crs.to_string())
    new_dataset = geopandas.GeoDataFrame()
    for shipwreck in dataset.location.unique():
        new_dataset = new_dataset.append(dataset.loc[dataset.location == shipwreck].sample(n=100))

    geoloc.plot_survey_color('aci_lf', df=new_dataset, save_path=save_map_path.joinpath('ACI_LF.png'))
    geoloc.plot_survey_color('aci_mf', df=new_dataset, save_path=save_map_path.joinpath('ACI_MF.png'))
    geoloc.plot_survey_color('aci_hf', df=new_dataset, save_path=save_map_path.joinpath('ACI_HF.png'))


def plot_spl_distribution():
    """
    Plot the distribution on the map
    """
    geoloc = geolocation.SurveyLocation(gps_path)
    dataset = pd.read_pickle(save_path_spl)
    dataset = geopandas.GeoDataFrame(dataset, crs=geoloc.geotrackpoints.crs.to_string())
    new_dataset = geopandas.GeoDataFrame()
    for shipwreck in dataset.location.unique():
        new_dataset = new_dataset.append(dataset.loc[dataset.location == shipwreck].sample(n=100))

    geoloc.plot_survey_color('spl_lf', df=new_dataset, units='SPL [dB]', save_path=save_map_path.joinpath('SPL_LF.png'))
    geoloc.plot_survey_color('spl_mf', df=new_dataset, units='SPL [dB]', save_path=save_map_path.joinpath('SPL_MF.png'))
    geoloc.plot_survey_color('spl_hf', df=new_dataset, units='SPL [dB]', save_path=save_map_path.joinpath('SPL_HF.png'))


if __name__ == "__main__":
    """
    Compute the spl of each band for each bin
    """
    # generate_spl_dataset()
    # generate_aci_dataset()
    # add_distance_to_coast()
    # plot_aci_distribution()
    # plot_spl_distribution()
