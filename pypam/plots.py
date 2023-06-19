import matplotlib.pyplot as plt
import xarray
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


def plot_spd(spd, db=True, p_ref=1.0, log=True, save_path=None, ax=None, show=True):
    """
    Plot the SPD graph of the bin

    Parameters
    ----------
    spd : xarray DataArray
        Data array with 2D data frequency-sound_pressure
    db : boolean
        If set to True the result will be given in db. Otherwise, in upa^2/Hz
    p_ref : Float
        Reference pressure in upa
    log : boolean
        If set to True the scale of the y axis is set to logarithmic
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
    if db:
        units = r'dB %s $\mu Pa^2/Hz$' % p_ref
    else:
        units = r'$\mu Pa^2/Hz$'
    # Plot the EPD
    if ax is None:
        fig, ax = plt.subplots()
    plot_2d(spd['spd'], x='frequency', y='spl', cmap='CMRmap_r', cbar_label='Empirical Probability Density', ax=ax,
            ylabel='PSD [%s]' % units, xlabel='Frequency [Hz]', title='Spectral Probability Density (SPD)', vmin=0,
            robust=False)
    ax.plot(spd['value_percentiles'].frequency, spd['value_percentiles'],
            label=spd['value_percentiles'].percentiles.values, linewidth=1)
    if log:
        plt.xscale('symlog')

    plt.legend(loc='upper right')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

    return ax


def plot_spectrogram_per_chunk(ds_spectrogram, db=True, p_ref=1.0, log=True, save_path=None, show=True):
    """
    Plot the spectrogram for each id of the ds_spectrogram (separately)

    Parameters
    ----------
    ds_spectrogram : xarray DataArray
        Data array with 3D data (datetime, frequency and time_bin as dimensions)
    db : boolean
        If set to True the result will be given in db. Otherwise in upa^2/Hz
    p_ref : Float
        Reference pressure in upa
    log : boolean
        If set to True the scale of the y axis is set to logarithmic
    save_path : string or Path
        Where to save the images (folder)
    show : bool
        set to True to show the plot
    """
    if db:
        units = r'dB ' + str(p_ref) + r' $\mu Pa$'
    else:
        units = r'$\mu Pa$'

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
                cbar_label=r'$L_{rms}$ [%s]' % units, ylog=log, title=title)

        if save_path is not None:
            plt.savefig(spectrogram_path)
        if show:
            plt.show()
        plt.close()


def plot_spectrum_per_chunk(ds, col_name, db=True, p_ref=1.0, log=True, save_path=None, show=True):
    """
    Plot the spectrums contained on the dataset

    Parameters
    ----------
    ds : xarray Dataset
        Dataset resultant from psd or power spectrum calculation
    col_name : string
        Name of the column where the data is (scaling type) 'spectrum' or 'density'
    db : boolean
        If set to True the result will be given in db. Otherwise in upa^2/Hz
    p_ref : Float
        Reference pressure in upa
    log : boolean
        If set to True the scale of the y axis is set to logarithmic
    save_path: string or Path
        Where to save the image
    show : bool
        set to True to show the plot
    """
    xscale = 'linear'
    if log:
        xscale = 'log'

    # TODO infer these units from the output dataset
    if col_name == 'band_density':
        if db:
            units = r'$[dB %s \mu Pa^2/Hz]$' % p_ref
        else:
            units = r'$[\mu Pa^2/Hz]$'

    else:  # col_name == 'band_spectrum':
        if db:
            units = r'$[dB %s \mu Pa^2]$' % p_ref
        else:
            units = r'$[\mu Pa^2]$'

    for id_n in ds.id:
        ds_id = ds[col_name].sel(id=id_n)
        ds_id.plot.line(xscale=xscale)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(units)

        # Plot the percentiles as horizontal lines
        plt.hlines(y=ds['value_percentiles'].loc[id_n], xmin=ds.frequency.min(), xmax=ds.frequency.max(),
                   label=ds['percentiles'])

        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()


def plot_spectrum_mean(ds, col_name, output_name, db=True, p_ref=1.0, log=True, save_path=None, ax=None, show=True):
    """
    Plot the mean spectrum

    Parameters
    ----------
    ds : xarray DataSet
        Output of evolution
    col_name : string
        Column name of the value to plot. Can be 'density' or 'spectrum'
    output_name : string
        Name of the label. 'PSD' or 'SPLrms'
    db : boolean
        If set to True the result will be given in db. Otherwise in upa^2/Hz
    p_ref : Float
        Reference pressure in upa
    log : boolean
        If set to True, y axis in logarithmic scale
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
    if col_name == 'band_density':
        if db:
            units = r'$[dB %s \mu Pa^2/Hz]$' % p_ref
        else:
            units = r'$[\mu Pa^2/Hz]$'

    else:  # col_name == 'band_spectrum':
        if db:
            units = r'$[dB %s \mu Pa^2]$' % p_ref
        else:
            units = r'$[\mu Pa^2]$'

    if ax is None:
        fig, ax = plt.subplots()

    sns.lineplot(x='frequency', y='value', ax=ax, data=ds[col_name].to_pandas().melt(), errorbar='sd')
    if len(ds['percentiles']) > 0:
        # Add the percentiles values
        ds['value_percentiles'].mean(dim='id').plot.line(hue='percentiles', ax=ax)

    ax.set_facecolor('white')
    plt.title(col_name.replace('_', ' ').capitalize())
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('%s [%s]' % (output_name, units))

    if log:
        plt.xscale('log')
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

    return ax


def plot_hmb_ltsa(da_sxx, db=True, p_ref=1.0, log=True, save_path=None, ax=None, show=True):
    """
    Plot the long-term spectrogram in hybrid millidecade bands
    Parameters
    ----------
    da_sxx : xarray DataArray
        Spectrogram data
    db : boolean
        If set to True, output in db
    p_ref : Float
        Reference pressure in upa
    log : boolean
        If set to True the scale of the y axis is set to logarithmic
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

    if db:
        units = r'db re 1V %s $\mu Pa^2/Hz$' % p_ref
    else:
        units = r'$\mu Pa^2/Hz$'

    if ax is None:
        fig, ax = plt.subplots()

    plot_2d(ds=da_sxx, x='datetime', y='frequency_bins', cbar_label='[%s]' % units, ax=ax, xlabel='Time',
            ylabel='Frequency [Hz]', title='Long Term Spectrogram', ylog=log)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

    return ax


def plot_summary_dataset(ds, percentiles, data_var='band_density', time_coord='datetime', freq_coord='frequency',
                         min_val=None, max_val=None, p_ref=1.0, show=True, log=True, save_path=None,
                         location=None):
    """
    :param ds: dataset output of pypam
    :param data_var: strig, name of the data variable to plot as a spectrogram. default band_density.
    :param time_coord: string, name of the coordinate which represents time (has to be type np.datetime64)
    :param freq_coord: string, name of the coordinate which represents frequency.
    :param percentiles: list or numpy array of the percentiles to compute and plot (1 to 100).
    :param min_val: minimum value (SPL) in db to compute the SPD. If None, minimum of the dataset will be used
    :param max_val: maximum value (SPL) in db to compute the SPD. If None, maximum of the dataset will be used
    :param p_ref: pressure reference in uPa. Default to 1uPa
    :param show: bool. Set to True to show the plot
    :param log: set to True to set the frequencies axis in a log scale
    :param save_path: None, string or Path. Where to save the plot. If None, the plot is not saved.
    :param location: tuple or list [latitude, longitude] in decimal coordinates. If location is passed, a bar with
    the sun position is going to be added below the time axis

    :return:
    """
    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[10, 0.5], width_ratios=[3, 2])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharey=ax0)

    units = r'db re 1V %s $\mu Pa^2/Hz$' % p_ref

    # LTSA plot
    xarray.plot.pcolormesh(ds[data_var], x=time_coord, y=freq_coord, add_colorbar=True,
                           cbar_kwargs={'label': '%s [%s]' % ('Spectrum Level', units),
                                        'location': 'top', 'orientation': 'horizontal', 'shrink': 2/3}, ax=ax0,
                           extend='neither', cmap='YlGnBu_r')

    # SPD plot
    spd = pypam.utils.compute_spd(ds, percentiles=percentiles, min_val=min_val, max_val=max_val)

    xarray.plot.pcolormesh(spd['spd'], x='spl', y=freq_coord, cmap='binary', add_colorbar=True,
                           cbar_kwargs={'label': 'Empirical Probability Density',
                                        'location': 'top', 'orientation': 'horizontal'}, ax=ax1,
                           extend='neither', vmin=0, robust=False)

    ax1.plot(spd['value_percentiles'], spd['value_percentiles'].frequency,
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
        solpos_arr = solpos[['zenith']].to_xarray()
        solpos_2d = solpos_arr['zenith'].expand_dims({'id': [0, 1]})
        # Plot the night/day bar
        xarray.plot.pcolormesh(solpos_2d, x='index', y='id', cmap='Greys', ax=ax2, add_colorbar=False)

        night_moment = solpos.zenith.argmax()
        day_moment = solpos.zenith.argmin()
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
    ax1.set_xlabel('Spectrum Level %s' % units)

    if log:
        ax0.set_yscale('symlog')

    ax1.legend(loc='upper right')
    plt.subplots_adjust(wspace=0.05, hspace=0.01)

    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)


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

