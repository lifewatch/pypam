"""
Plots
=====

The module ``plots`` is an ensemble of functions to plot `pypam` output's in different ways


.. autosummary::
    :toctree: generated/

    plot_spd
    plot_spectrum_mean
    plot_ltsa
    plot_summary_dataset
    plot_daily_patterns_from_ds
    plot_rms_evolution
    plot_aggregation_evolution

"""

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
            cbar_label=r'%s [$%s$]' % (spd['spd'].standard_name, spd['spd'].units),
            ax=ax, ylabel=r'%s [$%s$]' % (spd['spl'].standard_name, spd['spl'].units), xlabel='Frequency [Hz]',
            title='Spectral Probability Density (SPD)', vmin=0, robust=False)
    if len(spd.percentiles) > 0:
        ax.plot(spd['value_percentiles'][freq_axis], spd['value_percentiles'],
                label=spd['value_percentiles'].percentiles.values, linewidth=1)
        plt.legend(loc='upper right')
    if log:
        plt.xscale('symlog')

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


def plot_spectrogram_per_chunk(ds_spectrogram, log=True, save_path=None, show=True):
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
    """

    for id_n in ds_spectrogram.id:
        sxx = ds_spectrogram['spectrogram'].sel(id=id_n)
        time_bin = sxx.datetime
        title = 'Spectrogram of bin %s' % time_bin.values
        if save_path is not None:
            if type(save_path) == str:
                save_path = pathlib.Path(save_path)
            file_name = pathlib.Path(ds_spectrogram.attrs['file_path']).name
            spectrogram_path = save_path.joinpath(file_name.replace('.wav', '_%s.png' % int(id_n)))
        # Plot the spectrogram
        plot_2d(ds=sxx, x='time', y='frequency', xlabel='Time [s]', ylabel='Frequency [Hz]',
                cbar_label=r'%s [$%s$]' % (ds_spectrogram['spectrogram'].standard_name,
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
        plt.ylabel(r'%s [$%s$]' % (ds[data_var].standard_name, ds[data_var].units))

        # Plot the percentiles as horizontal lines
        plt.hlines(y=ds['value_percentiles'].loc[id_n], xmin=ds[freq_coord].min(), xmax=ds[freq_coord].max(),
                   label=ds['percentiles'])

        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()


def plot_spectrum_mean(ds, data_var, log=True, save_path=None, ax=None, show=True):
    """
    Plot the mean spectrum

    Parameters
    ----------
    ds : xarray DataSet
        Output of evolution
    data_var : string
        Name of the data variable to use
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

    sns.lineplot(x=ds[data_var].dims[1], y='value', ax=ax, data=ds[data_var].to_pandas().melt(), errorbar='sd')

    if ('percentiles' in ds) and (len(ds['percentiles']) > 0):
        # Add the percentiles values
        ds['value_percentiles'].mean(dim='id').plot.line(hue='percentiles', ax=ax)

    ax.set_facecolor('white')
    plt.title(data_var.replace('_', ' ').capitalize())
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'%s [$%s$]' % (ds[data_var].standard_name, ds[data_var].units))

    if log:
        plt.xscale('log')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


def plot_ltsa(ds, data_var, log=True, save_path=None, ax=None, show=True):
    """
    Plot the evolution of the ds containing percentiles and band values

    Parameters
    ----------
    ds : xarray DataSet
        Output of evolution
    data_var : string
        Column name of the value to plot. Can be 'density' or 'spectrum' or 'millidecade_bands
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
    plot_2d(ds[data_var], x='datetime', y=ds[data_var].dims[1], ax=ax,
            cbar_label=r'%s [$%s$]' % (ds[data_var].standard_name, ds[data_var].units), xlabel='Time',
            ylabel='Frequency [Hz]', title='Long Term Spectrogram', ylog=log)
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
        [latitude, longitude] in decimal coordinates. If location is passed, a bar with the sun position is going
        to be added below the time axis
    """
    plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[10, 0.5], width_ratios=[3, 2])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharey=ax0)

    # LTSA plot
    xarray.plot.pcolormesh(ds[data_var], x=time_coord, y=freq_coord, add_colorbar=True,
                           cbar_kwargs={'label': r'%s [$%s$]' % (ds[data_var].standard_name, ds[data_var].units),
                                        'location': 'top', 'orientation': 'horizontal', 'shrink': 2/3}, ax=ax0,
                           extend='neither', cmap='YlGnBu_r')

    # SPD plot
    spd = pypam.utils.compute_spd(ds, data_var=data_var, percentiles=percentiles, min_val=min_val, max_val=max_val)

    xarray.plot.pcolormesh(spd['spd'], x='spl', y=freq_coord, cmap='binary', add_colorbar=True,
                           cbar_kwargs={'label': r'%s [$%s$]' % (spd['spd'].standard_name, spd['spd'].units),
                                        'location': 'top', 'orientation': 'horizontal'}, ax=ax1,
                           extend='neither', vmin=0, robust=False)

    ax1.plot(spd['value_percentiles'], spd['value_percentiles'][freq_coord],
             label=spd['value_percentiles'].percentiles.values, linewidth=1)

    if location is not None:
        if pvlib is None:
            raise Exception('To use this feature it is necessary to install pvlib ')
        ax2 = plt.subplot(gs[2], sharex=ax0)
        solpos = pvlib.solarposition.get_solarposition(
            time=ds.datetime,
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
    ax1.set_xlabel(r'%s [$%s$]' % (spd['spl'].standard_name, spd['spl'].units))

    if log:
        ax0.set_yscale('symlog')

    ax1.legend(loc='upper right')
    plt.tight_layout()
    if location is not None:
        plt.subplots_adjust(wspace=0.05, hspace=0.01)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_daily_patterns_from_ds(ds, data_var, interpolate=True, save_path=None, ax=None,
                                show=True, plot_kwargs=None):
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

    Returns
    -------
    ax : matplotlib.axes class
        The ax with the plot if something else has to be plotted on the same
    """
    if plot_kwargs is None:
        plot_kwargs = {}

    daily_xr = ds.swap_dims(id='datetime')
    daily_xr = daily_xr.sortby('datetime')

    hours_float = daily_xr.datetime.dt.hour + daily_xr.datetime.dt.minute / 60
    date_minute_index = pd.MultiIndex.from_arrays([daily_xr.datetime.dt.floor('D').values, hours_float.values],
                                                  names=('date', 'time'))
    daily_xr = daily_xr.assign(datetime=date_minute_index).unstack('datetime')

    if interpolate:
        daily_xr = daily_xr.interpolate_na(dim='time', method='linear')

    if ax is None:
        fig, ax = plt.subplots()

    xarray.plot.pcolormesh(daily_xr[data_var], x='date', y='time', robust=True,
                           cbar_kwargs={'label': r'%s [%s]' % (ds[data_var].standard_name, ds[data_var].units)},
                           ax=ax, cmap='magma', **plot_kwargs)
    ax.set_ylabel('Hours of the day')
    ax.set_xlabel('Days')

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

    ax.set_title('Evolution of the broadband rms value')  # Careful when filter applied!
    ax.set_ylabel(r'%s [%s]' % (ds['rms'].standard_name, ds['rms'].units))
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


def plot_aggregation_evolution(ds, data_var, mode, save_path=None, ax=None, show=True):
    """
    Plot the aggregation evolution with boxplot or violin, the limits of the box are Q1 and Q3

    Parameters
    ----------
    ds : xarray DataSet
        Dataset to process
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

    Returns
    -------
    ax : matplotlib.axes class
        The ax with the plot if something else has to be plotted on the same
    """
    if ax is None:
        fig, ax = plt.subplots()

    df_plot = ds.to_dataframe()
    if mode == 'boxplot':
        sns.boxplot(data=df_plot, x='time', y=data_var, whis=2.5, color='steelblue')
    if mode == 'violin':
        sns.violinplot(data=df_plot, x='time', y=data_var, color='steelblue')

    ax.set_xlabel('Time')
    ax.set_ylabel(r'%s [$%s$]' % (ds[data_var].standard_name, ds[data_var].units))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

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
