"""
Plots
=====

The module ``plots`` is an ensemble of functions to plot `pypam` output's in different ways


.. autosummary::
    :toctree: generated/

    plot_spd
    plot_spectrum_median
    plot_ltsa
    plot_summary_dataset
    plot_daily_patterns_from_ds
    plot_rms_evolution
    plot_aggregation_evolution

"""

import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray
import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import matplotlib.gridspec as gridspec
try:
    import pvlib
except ModuleNotFoundError:
    pvlib = None

import pypam

plt.rcParams.update({'text.usetex': True})
sns.set_theme('paper')
sns.set_style('ticks')
sns.set_palette('colorblind')


def plot_spd(spd, log=True, save_path=None, ax=None, show=True):
    """
    Plot the SPD graph of the bin

    Parameters
    ----------
    spd : xarray DataArray
        Data array with 2D data frequency-sound_pressure. Frequency needs to be the first dimension. It is prepared to
        be used with the output of pypam.utils.compute_spd()
    log : boolean
        If set to True the scale of the y-axis is set to logarithmic
    save_path : string or Path
        Where to save the images
    ax : matplotlib.axes class or None
        ax to plot on
    show : bool
        set to True to show the plot

    Returns
    -------
    ax : matplotlib.axes class
        The ax with the plot if something else has to be plotted on the same
    """
    # Plot the EPD
    if ax is None:
        fig, ax = plt.subplots()
    freq_axis = spd['spd'].dims[0]
    plot_2d(spd['spd'], x=freq_axis, y='spl', cmap='CMRmap_r',
            cbar_label=r'%s [$%s$]' % (re.sub('_', ' ', spd['spd'].standard_name).title(), spd['spd'].units),
            ax=ax, ylabel=r'%s [$%s$]' % (re.sub('_', ' ', spd['spl'].standard_name).title(), spd['spl'].units),
            xlabel='Frequency [Hz]', title='Spectral Probability Density (SPD)', vmin=0, robust=False)
    if len(spd.percentiles) > 0:
        ax.plot(spd['value_percentiles'][freq_axis], spd['value_percentiles'],
                label=spd['value_percentiles'].percentiles.values, linewidth=1)
        plt.legend(loc='upper right')

    if log:
        ax.set_xscale('symlog')

    ax.set_facecolor('white')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


def plot_spectrogram_per_chunk(ds_spectrogram, log=True, save_path=None, show=True, datetime_coord='time',
                               freq_coord='frequency'):
    """
    Plot the spectrogram for each id of the ds_spectrogram (separately)

    Parameters
    ----------
    ds_spectrogram : xarray DataArray
        Data array with 3D data (datetime, frequency and time as dimensions)
    log : boolean
        If set to True the scale of the y-axis is set to logarithmic
    save_path : string or Path
        Where to save the images (folder)
    show : bool
        set to True to show the plot
    datetime_coord : str
        Name of the coordinate representing time for the spectrogram (not for each chunk)
    freq_coord
    """

    for id_n in ds_spectrogram.id:
        sxx = ds_spectrogram['spectrogram'].sel(id=id_n)
        time_bin = sxx[datetime_coord]
        title = 'Spectrogram of bin %s' % time_bin.values
        if save_path is not None:
            if type(save_path) == str:
                save_path = pathlib.Path(save_path)
            file_name = pathlib.Path(ds_spectrogram.attrs['file_path']).name
            spectrogram_path = save_path.joinpath(file_name.replace('.wav', '_%s.png' % int(id_n)))
        # Plot the spectrogram
        plot_2d(ds=sxx, x=datetime_coord, y=freq_coord, xlabel='Time [s]', ylabel='Frequency [Hz]',
                cbar_label=r'%s [$%s$]' % (re.sub('_', ' ', ds_spectrogram['spectrogram'].standard_name).title(),
                                           ds_spectrogram['spectrogram'].units), ylog=log, title=title)

        if save_path is not None:
            plt.savefig(spectrogram_path)
        if show:
            plt.show()
        plt.close()


def plot_spectrum_per_chunk(ds, data_var, log=True, save_path=None, show=True):
    """
    Plot the spectrums contained on the dataset

    Parameters
    ----------
    ds : xarray Dataset
        Dataset resultant from psd or power spectrum calculation
    data_var : string
        Name of the data variable to use
    log : boolean
        If set to True the scale of the y-axis is set to logarithmic
    save_path: string or Path
        Where to save the image
    show : bool
        set to True to show the plot
    """
    xscale = 'linear'
    if log:
        xscale = 'log'

    freq_coord = ds[data_var].dims[1]
    for id_n in ds.id:
        ds_id = ds[data_var].sel(id=id_n)
        ds_id.plot.line(xscale=xscale)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'%s [$%s$]' % (re.sub('_', ' ', ds[data_var].standard_name).title(), ds[data_var].units))

        # Plot the percentiles as horizontal lines
        plt.hlines(y=ds['value_percentiles'].loc[id_n], xmin=ds[freq_coord].min(), xmax=ds[freq_coord].max(),
                   label=ds['percentiles'])
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()


def plot_multiple_spectrum_median(ds_dict, data_var, percentiles='default', frequency_coord='frequency',
                                  time_coord='id', log=True, save_path=None, ax=None, show=True, **kwargs):
    """
    Same than plot_spectrum_median but instead of one ds you can pass a dictionary of label: ds so they are all plot
    on one figure.

    Parameters
    ----------
    ds_dict : dict
        Dictionary of label: ds with all the ds to plot
    data_var : string
        Name of the data variable to use
    percentiles: Tuple or 'default'
        list or tuple with (min_percentile, max_percentile). If set to 'default' it will be [10, 90]
    time_coord: str
        Name of the coordinate representing time
    frequency_coord: str
        Name of the coordinate representing frequency
    log : boolean
        If set to True, y-axis in logarithmic scale
    save_path : string or Path
        Where to save the output graph. If None, it is not saved
    ax : matplotlib.axes class or None
        ax to plot on
    show : bool
        set to True to show the plot

    Returns
    -------
    matplotlib.axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    for label, ds in ds_dict.items():
        kwargs.update({'label': label})
        plot_spectrum_median(ds, data_var, percentiles=percentiles, frequency_coord=frequency_coord,
                             time_coord=time_coord, log=log, save_path=None, ax=ax, show=False, **kwargs)

    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


def plot_spectrum_median(ds, data_var, percentiles='default', frequency_coord='frequency', time_coord='id',
                         log=True, save_path=None, ax=None, show=True, **kwargs):
    """
    Plot the median spectrum

    Parameters
    ----------
    ds : xarray DataSet
        Dataset to plot
    data_var : string
        Name of the data variable to use
    percentiles: Tuple or 'default'
        list or tuple with (min_percentile, max_percentile). If set to 'default' it will be [10, 90]
    time_coord: str
        Name of the coordinate representing time
    frequency_coord: str
        Name of the coordinate representing frequency
    log : boolean
        If set to True, y-axis in logarithmic scale
    save_path : string or Path
        Where to save the output graph. If None, it is not saved
    ax : matplotlib.axes class or None
        ax to plot on
    show : bool
        set to True to show the plot

    Returns
    -------
    ax : matplotlib.axes class
        The ax with the plot if something else has to be plotted on the same
    """
    if ax is None:
        fig, ax = plt.subplots()
    if percentiles == 'default':
        percentiles = [10, 90]

    pxx = ds[data_var].to_numpy().T
    p = np.nanpercentile(a=pxx, q=np.array(percentiles), axis=1)
    ax.plot(ds[frequency_coord].values, ds[data_var].median(dim=time_coord).values, **kwargs)
    if 'color' in kwargs.keys():
        ax.fill_between(x=ds[frequency_coord].values, y1=p[0], y2=p[1], alpha=0.2, color=kwargs['color'])
    else:
        ax.fill_between(x=ds[frequency_coord].values, y1=p[0], y2=p[1], alpha=0.2)

    ax.set_facecolor('white')
    ax.set_title(data_var.replace('_', ' ').capitalize())
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'%s [$%s$]' % (re.sub('_', ' ', ds[data_var].standard_name).title(), ds[data_var].units))

    if log:
        ax.set_xscale('log')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


def plot_ltsa(ds, data_var, time_coord='id', freq_coord='frequency', log=True, save_path=None, ax=None, show=True):
    """
    Plot the evolution of the ds containing percentiles and band values

    Parameters
    ----------
    ds : xarray DataSet
        Output of evolution
    data_var : string
        Column name of the value to plot. Can be 'density' or 'spectrum' or 'millidecade_bands
    time_coord: string
        name of the coordinate which represents time (has to be type np.datetime64)
    freq_coord: string
        name of the coordinate which represents frequency.
    log : boolean
        If set to True the scale of the y-axis is set to logarithmic
    save_path : string or Path
        Where to save the output graph. If None, it is not saved
    ax : matplotlib.axes class or None
        ax to plot on
    show : boolean
        Set to True to show the plot

    Returns
    -------
    ax : matplotlib.axes class
        The ax with the plot if something else has to be plotted on the same
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the evolution
    # Extra axes for the colorbar and delete the unused one
    plot_2d(ds[data_var], x=time_coord, y=freq_coord, ax=ax,
            cbar_label=r'%s [$%s$]' % (re.sub('_', ' ', ds[data_var].standard_name).title(), ds[data_var].units),
            xlabel='Time', ylabel='Frequency [Hz]', title='Long Term Spectrogram', ylog=log)
    ax.set_facecolor('white')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


def plot_summary_dataset(ds, percentiles, data_var='band_density', time_coord='datetime', freq_coord='frequency',
                         min_val=None, max_val=None, show=True, log=True, save_path=None,
                         location=None):
    """
    Plots a summary of the data combining the LTSA and a SPD. If location is given, it also plots an extra
    colorbar showing the day/night patterns

    Parameters
    ----------
    ds: xarray Dataset
        dataset output of pypam
    data_var: string
        name of the data variable to plot as a spectrogram. default band_density.
    time_coord: string
        name of the coordinate which represents time (has to be type np.datetime64)
    freq_coord: string
        name of the coordinate which represents frequency.
    percentiles: list or numpy array
        percentiles to compute and plot (1 to 100).
    min_val: float
        minimum value (SPL) in db to compute the SPD. If None, minimum of the dataset will be used
    max_val: float
        maximum value (SPL) in db to compute the SPD. If None, maximum of the dataset will be used
    show: bool.
        Set to True to show the plot
    log: bool
        Set to True to set the frequencies axis in a log scale
    save_path: None, string or Path.
        Where to save the plot. If None, the plot is not saved.
    location: tuple or list
        [longitude, latitude] in decimal coordinates. If location is passed, a bar with the sun position is going
        to be added below the time axis
    """
    plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[10, 0.5], width_ratios=[3, 2])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharey=ax0)

    # LTSA plot
    xarray.plot.pcolormesh(ds[data_var], x=time_coord, y=freq_coord, add_colorbar=True,
                           cbar_kwargs={'label': r'%s [$%s$]' % (re.sub('_', ' ', ds[data_var].standard_name).title(),
                                                                 ds[data_var].units),
                                        'location': 'top', 'orientation': 'horizontal', 'shrink': 2/3}, ax=ax0,
                           extend='neither', cmap='YlGnBu_r')

    # SPD plot
    spd = pypam.utils.compute_spd(ds, data_var=data_var, percentiles=percentiles, min_val=min_val, max_val=max_val)

    xarray.plot.pcolormesh(spd['spd'], x='spl', y=freq_coord, cmap='binary', add_colorbar=True,
                           cbar_kwargs={'label': r'%s [$%s$]' % (re.sub('_', ' ', spd['spd'].standard_name).title(),
                                                                 spd['spd'].units),
                                        'location': 'top', 'orientation': 'horizontal'}, ax=ax1,
                           extend='neither', vmin=0, robust=False)

    ax1.plot(spd['value_percentiles'], spd['value_percentiles'][freq_coord],
             label=spd['value_percentiles'].percentiles.values, linewidth=1)

    if location is not None:
        if pvlib is None:
            raise Exception('To use this feature it is necessary to install pvlib ')
        ax2 = plt.subplot(gs[2], sharex=ax0)
        solpos = pvlib.solarposition.get_solarposition(
            time=ds[time_coord],
            latitude=location[1],
            longitude=location[0],
            altitude=0,
            temperature=20,
            pressure=pvlib.atmosphere.alt2pres(0),
        )

        solpos_arr = solpos[['elevation']].to_xarray()
        solpos_2d = solpos_arr['elevation'].expand_dims({'id': [0, 1]})
        # Plot the night/day bar
        xarray.plot.pcolormesh(solpos_2d, x='index', y='id', cmap='Greys', ax=ax2, add_colorbar=False, vmax=0, vmin=-12)

        night_moment = solpos.elevation.argmax()
        day_moment = solpos.elevation.argmin()
        ax2.text(solpos.iloc[night_moment].name, 0.3, 'Night', fontdict={'color': 'white'})
        ax2.text(solpos.iloc[day_moment].name, 0.3, 'Day', fontdict={'color': 'k'})

        # Adjust the axis
        ax2.get_yaxis().set_visible(False)
        ax2.set_xlabel('Time')
        ax0.get_xaxis().set_visible(False)
    else:
        ax0.set_xlabel('Time')
    # Adjust the axis names and visibilities
    ax0.set_ylabel('Frequency [Hz]')
    ax1.get_yaxis().set_visible(False)
    ax1.set_xlabel(r'%s [$%s$]' % (re.sub('_', ' ', spd['spl'].standard_name).title(), spd['spl'].units))

    if log:
        ax0.set_yscale('symlog')

    ax0.set_facecolor('white')
    ax1.set_facecolor('white')
    ax1.legend(loc='upper right')
    plt.tight_layout()
    if location is not None:
        plt.subplots_adjust(wspace=0.05, hspace=0.01)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_daily_patterns_from_ds(ds, data_var, interpolate=True, save_path=None, ax=None,
                                show=True, plot_kwargs=None, datetime_coord='datetime'):
    """
    Plot the daily rms patterns

    Parameters
    ----------
    ds : xarray DataSet
        Dataset to process. Should be an output of pypam, or similar structure
    data_var : str
        Name of the data variable to plot
    interpolate: bool
        Set to False if no interpolation is desired for the nan values
    save_path : string or Path
        Where to save the output graph. If None, it is not saved
    ax : matplotlib.axes
        Ax to plot on
    show : bool
        Set to True to show directly
    datetime_coord : str
        Name of the coordinate representing time

    Returns
    -------
    ax : matplotlib.axes class
        The ax with the plot if something else has to be plotted on the same
    """
    if plot_kwargs is None:
        plot_kwargs = {}

    daily_xr = ds.copy()
    hours_float = daily_xr[datetime_coord].dt.hour + daily_xr[datetime_coord].dt.minute / 60
    date_minute_index = pd.MultiIndex.from_arrays([daily_xr[datetime_coord].dt.floor('D').values, hours_float.values],
                                                  names=('date', 'hours'))
    daily_xr = daily_xr.assign({datetime_coord: date_minute_index}).unstack(datetime_coord)

    if interpolate:
        daily_xr = daily_xr.interpolate_na(dim='hours', method='linear')

    if ax is None:
        fig, ax = plt.subplots()

    xarray.plot.pcolormesh(daily_xr[data_var], x='date', y='hours', robust=True,
                           cbar_kwargs={'label': r'%s [%s]' % (re.sub('_', ' ', ds[data_var].standard_name).title(),
                                                               ds[data_var].units)},
                           ax=ax, cmap='magma', **plot_kwargs)
    ax.set_ylabel('Hours of the day')
    ax.set_xlabel('Days')
    ax.set_facecolor('white')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


def plot_rms_evolution(ds, save_path=None, ax=None, show=True):
    """
    Plot the rms evolution

    Parameters
    ----------
    ds : xarray DataSet
        Dataset to process
    save_path : string or Path
        Where to save the image
    ax : matplotlib.axes class or None
        ax to plot on
    show : boolean
        Set to True to show the plot

    Returns
    -------
    ax : matplotlib.axes class
        The ax with the plot if something else has to be plotted on the same
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(ds['rms'])
    ax.set_xlabel('Time')
    ax.set_facecolor('white')
    ax.set_title('Evolution of the broadband rms value')  # Careful when filter applied!
    ax.set_ylabel(r'%s [%s]' % (re.sub('_', ' ', ds['rms'].standard_name).title(), ds['rms'].units))
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


def _plot_aggregation_evolution(df_plot, data_var, standard_name, units, mode='boxplot', ax=None, save_path=None,
                                show=False, aggregation_time='D', **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if mode == 'boxplot':
        sns.boxplot(data=df_plot, x='aggregated_time', y=data_var, whis=2.5, ax=ax, **kwargs)
    elif mode == 'violin':
        sns.violinplot(data=df_plot, x='aggregated_time', y=data_var, ax=ax, **kwargs)
    elif mode == 'quantiles':
        if 'hue' in kwargs.keys():
            df_plot_list = df_plot.groupby(kwargs['hue'])
            kwargs.pop('hue')
        else:
            df_plot_list = [(0, df_plot)]
        for _, df_plot_i in df_plot_list:
            quantiles_plot = df_plot_i.groupby('aggregated_time').quantile([0.1, 0.5, 0.9], numeric_only=True)
            quantiles_plot = quantiles_plot.unstack()
            quantiles_plot = quantiles_plot[data_var]
            sns.lineplot(data=quantiles_plot, y=0.5, x='aggregated_time', ax=ax, **kwargs)

            if 'color' in kwargs.keys():
                ax.fill_between(x=quantiles_plot.index,
                                y1=quantiles_plot[0.1].values,
                                y2=quantiles_plot[0.9].values,
                                alpha=0.2, color=kwargs['color'])
            else:
                ax.fill_between(x=quantiles_plot.index,
                                y1=quantiles_plot[0.1].values,
                                y2=quantiles_plot[0.9].values,
                                alpha=0.2)
    else:
        raise ValueError('mode %s is not implemented. Only boxplot, quantiles and violin' % mode)

    ax.set_xlabel('Time [%s]' % aggregation_time)
    ax.set_ylabel(r'%s [$%s$]' % (re.sub('_', ' ', standard_name).title(), units))
    ax.set_facecolor('white')
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


def _prepare_aggregation_plot_data(ds, data_var, aggregation_freq_band=None, aggregation_time='D',
                                   freq_coord='frequency', datetime_coord='datetime'):
    """
    Prepare the data for the aggregation plot

    Parameters
    ----------
    ds : Dataset
        Dataset with all the ds to plot
    data_var : str
        Name of the data variable to plot
    datetime_coord : str
        Name of the coordinate representing time
    aggregation_time : str
        Resolution of the bin aggregation. Can be 'D' for day, 'H' for hour, 'W' for week and 'M' for month
    freq_coord : str
        Name of the frequency coordinate
    aggregation_freq_band : None, float or tuple
        If a float is given, this function compute aggregation for the frequency which is selected
        If a tuple is given, this function will compute aggregation for the average of all frequencies which are
        selected
        If None is given, this function will compute aggregation for the data_var given, assuming that there is no
        frequency dependence

    Returns
    -------
    ax : matplotlib.axes class
        The ax with the plot if something else has to be plotted on the same
    """
    ds_copy = pypam.utils.freq_band_aggregation(ds, data_var,
                                                aggregation_freq_band=aggregation_freq_band,
                                                freq_coord=freq_coord)
    df_plot = ds_copy[data_var].to_dataframe()
    df_plot['aggregated_time'] = pd.to_datetime(ds_copy[datetime_coord].values).to_period(aggregation_time).start_time

    return df_plot


def plot_multiple_aggregation_evolution(ds_dict, data_var, mode, save_path=None, ax=None, show=True,
                                        datetime_coord='datetime', aggregation_time='D', freq_coord='frequency',
                                        aggregation_freq_band=None, **kwargs):
    """
    Same than plot_aggregation_evolution but instead of one ds you can pass a dictionary of label: ds so they are
    all plot on one figure.

    Parameters
    ----------
    ds_dict : dict
        Dictionary of label: ds with all the ds to plot
    data_var : str
        Name of the data variable to plot
    mode : str
        'boxplot' or 'violin'
    save_path : string or Path
        Where to save the image
    ax : matplotlib.axes class or None
        ax to plot on
    show : boolean
        Set to True to show the plot
    datetime_coord : str
        Name of the coordinate representing time
    aggregation_time : str
        Resolution of the bin aggregation. Can be 'D' for day, 'H' for hour, 'W' for week and 'M' for month
    freq_coord : str
        Name of the frequency coordinate
    aggregation_freq_band : None, float or tuple
        If a float is given, this function compute aggregation for the frequency which is selected
        If a tuple is given, this function will compute aggregation for the average of all frequencies which are
        selected
        If None is given, this function will compute aggregation for the data_var given, assuming that there is no
        frequency dependence
    kwargs:
        Any other argument which can be passed to the seaborn plot function

    Returns
    -------
    ax : matplotlib.axes class
        The ax with the plot if something else has to be plotted on the same
    """
    total_df = pd.DataFrame()
    for label, ds in ds_dict.items():
        df_plot = _prepare_aggregation_plot_data(ds, data_var=data_var, aggregation_freq_band=aggregation_freq_band,
                                                 aggregation_time=aggregation_time, freq_coord=freq_coord,
                                                 datetime_coord=datetime_coord)
        df_plot['Data series'] = label
        total_df = pd.concat([total_df, df_plot])

    kwargs.update({'hue': 'Data series'})
    _plot_aggregation_evolution(total_df, data_var, standard_name=ds[data_var].standard_name, units=ds[data_var].units,
                                mode=mode, ax=ax, save_path=save_path,
                                show=show, aggregation_time=aggregation_time, **kwargs)

    return ax


def plot_aggregation_evolution(ds, data_var, mode, save_path=None, ax=None, show=True, datetime_coord='datetime',
                               aggregation_time='D', freq_coord='frequency', aggregation_freq_band=None, **kwargs):
    """
    Plot the aggregation evolution with boxplot, violin or quartiles, the limits of the box are Q1 and Q3.
    It will compute the median of all the values included in the frequency band specified in 'aggregation_freq_band'.
    Then it will plot the evolution considering the specified aggregation_time

    Parameters
    ----------
    ds : xarray DataSet
        Dataset to process
    data_var : str
        Name of the data variable to plot
    mode : str
        'boxplot', 'violin' or 'quartiles'
    save_path : string or Path
        Where to save the image
    ax : matplotlib.axes class or None
        ax to plot on
    show : boolean
        Set to True to show the plot
    datetime_coord : str
        Name of the coordinate representing time
    aggregation_time : str
        Resolution of the bin aggregation. Can be 'D' for day, 'H' for hour, 'W' for week and 'M' for month
    freq_coord : str
        Name of the frequency coordinate
    aggregation_freq_band : None, float or tuple
        If a float is given, this function compute aggregation for the frequency which is selected
        If a tuple is given, this function will compute aggregation for the average of all frequencies which are
        selected
        If None is given, this function will compute aggregation for the data_var given, assuming that there is no
        frequency dependence
    kwargs:
        Any parameter which can be passed to the plot function of seaborn

    Returns
    -------
    ax : matplotlib.axes class
        The ax with the plot if something else has to be plotted on the same
    """

    df_plot = _prepare_aggregation_plot_data(ds, data_var=data_var, aggregation_freq_band=aggregation_freq_band,
                                             aggregation_time=aggregation_time, freq_coord=freq_coord,
                                             datetime_coord=datetime_coord)
    _plot_aggregation_evolution(df_plot, data_var, standard_name=ds[data_var].standard_name, units=ds[data_var].units,
                                mode=mode, ax=ax, save_path=save_path,
                                show=show, aggregation_time=aggregation_time, **kwargs)

    return ax


def plot_2d(ds, x, y, cbar_label, xlabel, ylabel, title, ylog=False, ax=None, **kwargs):
    yscale = 'linear'
    if ylog:
        yscale = 'symlog'
    if ax is None:
        _, ax = plt.subplots()

    if 'cmap' not in kwargs.keys():
        kwargs['cmap'] = 'YlGnBu_r'
    if 'robust' not in kwargs.keys():
        kwargs['robust'] = True
    xarray.plot.pcolormesh(ds, x=x, y=y, add_colorbar=True, yscale=yscale,
                           cbar_kwargs={'label': cbar_label}, ax=ax,
                           extend='neither', **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_facecolor('white')
