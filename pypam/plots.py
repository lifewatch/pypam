"""
Plots
=====

The module ``plots`` is an ensemble of functions to plot `pypam` output's in different ways


.. autosummary::
    :toctree: generated/

    plot_spd
    plot_spectrograms
    plot_spectrum
    plot_spectrum_mean
    plot_hmb_ltsa

"""

import matplotlib.pyplot as plt
import xarray
import seaborn as sns
import pathlib

plt.rcParams.update({'text.usetex': True})
sns.set_theme('paper')
sns.set_style('ticks')
sns.set_palette('colorblind')


def plot_spd(spd, db, p_ref, log=True, save_path=None, ax=None, show=True):
    """
    Plot the the SPD graph of the bin

    Parameters
    ----------
    spd : xarray DataArray
        Data array with 2D data frequency-sound_pressure
    db : boolean
        If set to True the result will be given in db. Otherwise in upa^2/Hz
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
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def plot_spectrograms(ds_spectrogram, log, db, p_ref=1.0, save_path=None):
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
        plt.show()
        plt.close()


def plot_spectrum(ds, col_name, ylabel, log=True, save_path=None):
    """
    Plot the spectrums contained on the dataset

    Parameters
    ----------
    ds : xarray Dataset
        Dataset resultant from psd or power spectrum calculation
    col_name : string
        Name of the column where the data is (scaling type) 'spectrum' or 'density'
    ylabel : string
        Label of the y axis (with units of the data)
    log : boolean
        If set to True the scale of the y axis is set to logarithmic
    save_path: string or Path
        Where to save the image
    """
    xscale = 'linear'
    if log:
        xscale = 'log'
    for id_n in ds.id:
        ds_id = ds[col_name].sel(id=id_n)
        ds_id.plot.line(xscale=xscale)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(ylabel)

        # Plot the percentiles as horizontal lines
        plt.hlines(y=ds['value_percentiles'].loc[id_n], xmin=ds.frequency.min(), xmax=ds.frequency.max(),
                   label=ds['percentiles'])

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()


def plot_spectrum_mean(ds, units, col_name, output_name, save_path=None, log=True):
    """
    Plot the mean spectrum

    Parameters
    ----------
    ds : xarray DataSet
        Output of evolution
    units : string
        Units of the spectrum
    col_name : string
        Column name of the value to plot. Can be 'density' or 'spectrum'
    output_name : string
        Name of the label. 'PSD' or 'SPLrms'
    log : boolean
        If set to True, y axis in logarithmic scale
    save_path : string or Path
        Where to save the output graph. If None, it is not saved
    """
    fig, ax = plt.subplots()
    ds[col_name].mean(dim='id').plot.line(x='frequency', ax=ax)
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
    plt.show()
    plt.close()


def plot_hmb_ltsa(da_sxx, db=True, p_ref=1.0, log=False, save_path=None, show=False):
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
    show : boolean
        Set to True to show the plot

    """

    if db:
        units = r'db re 1V %s $\mu Pa^2/Hz$' % p_ref
    else:
        units = r'$\mu Pa^2/Hz$'

    plot_2d(ds=da_sxx, x='datetime', y='frequency_bins', cbar_label='%s [%s]' % ('SPLrms', units), xlabel='Time',
            ylabel='Frequency [Hz]', title='Long Term Spectrogram', ylog=log)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


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

