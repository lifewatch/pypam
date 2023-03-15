#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 12:16:03 2021

This script is used to perform the NMF on hydrophone data

@author: Randall Ali
"""

import numpy as np
import sklearn.decomposition

from matplotlib import pyplot as plt
from scipy import signal as sig
import xarray

from pypam import utils, plots

SEED = 10


class NMF:
    def __init__(self, window_time=0.1, rank=15, save_path=None):
        """
        window_time: Frame length in seconds
        """
        self.window_time = window_time
        self.R = rank
        self.nfft = None
        self.noverlap = None
        self.save_path = save_path

    def __call__(self, s, V_type='Z_mag', normalize=True, verbose=False):
        """
        s: Signal object
            Signal to analyze
        V_type: string
            You can choose what you want to use for V : Z_mag, Z_phase, Gxx, etc...
            I've found that Z_mag works best, perhaps because some sources are coherent.
        normalize: bool
            Set to True if matrices should be normalized to represent probabilities
        verbose: bool
            Set to True to see the evolution of the decomposition and to plot the results

        Returns:
            dataset with W, H as variables
        """
        self.nfft = int(self.window_time * s.fs)
        self.noverlap = (self.nfft / 2)
        # df = s.fs / self.nfft

        # Computing both the spectrogram and the STFT
        # The STFT will be needed to do the filtering with the time-freq. mask.
        # The STFT function is used since it will take into account the COLA constraints so that an iSTFT can be done.
        # The STFT has a length (time dimension) longer than the spectrogram because it does some extra padding.

        # Apply the NMF Algorithm
        if V_type == 'Gxx':
            f, t, Z_stft = s.spectrogram(nfft=self.nfft, scaling='density', db=True, overlap=0.5)
            V = Z_stft

        elif V_type == 'Z_mag' or V_type == 'Z_phase':
            f, t, Z_stft = sig.stft(s.signal, fs=s.fs, nperseg=self.nfft, noverlap=self.noverlap, window='hann')
            Z_mag = np.abs(Z_stft)
            Z_phase = np.angle(Z_stft)
            # Magnitude of STFT
            z_mag = xarray.DataArray(data=Z_mag,
                                     coords={'frequency': f, 'time': t},
                                     dims=['frequency', 'time'])
            # Phase of STFT
            z_phase = xarray.DataArray(data=Z_phase,
                                       coords={'frequency': f, 'time': t},
                                       dims=['frequency', 'time'])
            z_ds = xarray.Dataset(data_vars={'magnitude': z_mag, 'phase': z_phase})
            if verbose:
                plots.plot_2d(z_ds['magnitude'], x='time', y='frequency', cbar_label='Magnitude', xlabel='Time',
                              ylabel='Frequency [Hz]', title='Z magnitude')
                plt.show()

            if V_type == 'Z_phase':
                V = Z_phase
            else:
                V = Z_mag
        else:
            raise Exception('This approach is not implemented!')

        # W, H, ErrConv = self.NMF_hals(V, init_type="rand_norm", max_iter=1000000, tol=1e-9, verbose=verbose)
        separator = sklearn.decomposition.NMF(n_components=self.R, init='random', tol=1e-9, verbose=verbose,
                                              max_iter=1000)
        W = separator.fit_transform(V)  # W=W_init, H=H_init
        H = separator.components_
        V_approx = W @ H  # approximated V

        # Normalise the columns of W and H if so desired - This scales the values so that they are
        # between 0 and 1 and hence represent probabilities
        if normalize:
            l1n_w = np.linalg.norm(W, axis=0, ord=1)  # l1 norm of W
            Dw = np.diag(l1n_w)
            Dw_inv = np.diag(1 / l1n_w)
    
            l1n_v = np.linalg.norm(V_approx, axis=0, ord=1)  # l1 norm of V
            Dv_inv = np.diag(1 / l1n_v)
    
            V_approx = V_approx @ Dv_inv
            W = W @ Dw_inv
            H = Dw @ H @ Dv_inv

        sources = np.arange(self.R)
        w_arr = xarray.DataArray(data=W, coords={'frequency': f, 'sources': sources},
                                 dims=['frequency', 'sources'])
        h_arr = xarray.DataArray(data=H, coords={'sources': sources, 'time': t},
                                 dims=['sources', 'time'])
        v_arr = xarray.DataArray(data=Z_stft, coords={'frequency': f, 'time': t},
                                 dims=['frequency', 'time'])
        nmf_ds = xarray.Dataset({'W': w_arr, 'H': h_arr, 'Z_stft': v_arr})

        # Error convergence and decomposition
        if verbose:
            # Decomposition Plots
            Vlg = utils.to_db(V, square=True, ref=1.0)
            Vlg_ap = utils.to_db(V_approx, square=True, ref=1.0)
            v = xarray.DataArray(data=Vlg, coords={'frequency': f, 'time': t}, dims=['frequency', 'time'])
            v_approx = xarray.DataArray(data=Vlg_ap, coords={'frequency': f, 'time': t}, dims=['frequency', 'time'])

            self.plot_decomposition(s, f, v, v_approx, w_arr, h_arr)

        return nmf_ds

    def time_freq_masks(self, ds):
        """
        Compute the time-frequency masks

        Returns:
            G_tf: time-freq masks. Not divided by W@H
            C_tf: filtered STFT,
        """
        W = ds['W'].values
        H = ds['H'].values
        Z_stft = ds['Z_stft']
        f = ds.frequency.values
        t = ds.time.values
        sources = ds.sources.values

        # Initialising arrays
        G_tf = np.zeros([ds.dims['frequency'], ds.dims['time'], self.R])
        C_tf = np.zeros([ds.dims['frequency'], ds.dims['time'], self.R], dtype='complex')

        for n in np.arange(0, self.R, 1):
            G_tf[:, :, n] = (np.dot(W[:, [n]], H[[n], :]))  # Compute TF mask. Before it was divided by W@H
            C_tf[:, :, n] = G_tf[:, :, n] * Z_stft

        gtf_arr = xarray.DataArray(data=G_tf, coords={'frequency': f, 'time': t, 'sources': sources},
                                   dims=['frequency', 'time', 'sources'])
        ctf_arr = xarray.DataArray(data=C_tf, coords={'frequency': f, 'time': t, 'sources': sources},
                                   dims=['frequency', 'time', 'sources'])
        nmf_tf_ds = xarray.Dataset({'G_tf': gtf_arr, 'C_tf': ctf_arr})
        return nmf_tf_ds

    def return_filtered_signal(self, s, C_tf):
        c_tf = np.zeros([len(s.signal), self.R])
        for n in np.arange(0, self.R, 1):
            _, sig_td = sig.istft(C_tf[:, :, n], s.fs, nperseg=self.nfft, noverlap=self.noverlap, window='hann')
            c_tf[:, n] = sig_td[0:len(s.signal)]

        return c_tf

    def reconstruct_sources(self, ds):
        # Step 5: Creation of time-frequency masks to obtain the individual components.
        ds_tf = self.time_freq_masks(ds)
        return ds_tf

    @staticmethod
    def plot_error_conv(ErrConv, Residual):
        ErrConv = np.array(ErrConv)
        plt.plot(np.log(ErrConv))
        plt.xlabel('Number of iterations')
        plt.title('Residual Frob. norm = ' + str(np.linalg.norm(Residual)))
        plt.show()
        
    def plot_decomposition(self, s, f_sg, V, V_approx, W, H):
        W_log = np.log10(W)
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))

        plots.plot_2d(V, x='time', y='frequency', xlabel='Time [mins]', ylabel='Frequency [Hz]',
                      cbar_label='SPL', title='Original spectrogram', ax=ax[0, 0])
        plots.plot_2d(W_log, x='sources', y='frequency', xlabel='Number of basis vectors', ylabel='Frequency [Hz]',
                      cbar_label='SPL', title='W = basis vectors', ax=ax[1, 0])
        plots.plot_2d(H, x='time', y='sources', xlabel='Time [mins]', ylabel='Number of basis vectors',
                      cbar_label='Activation', title='H = activations', ax=ax[1, 1])
        plots.plot_2d(V_approx, x='time', y='frequency', xlabel='Time [mins]', ylabel='Frequency [Hz]',
                      cbar_label='SPL', title='V approximation (WH)', ax=ax[0, 1])

        if self.save_path is not None:
            plt.savefig(self.save_path.joinpath('decomposition.png'))
        plt.show()

        # Basis Vector plots
        fig, axes = plt.subplots(1, self.R)
        fig.subplots_adjust(wspace=0.5)  # horizontal spacing

        for i in np.arange(0, self.R, 1):
            axes[i].plot((W[:, i]), f_sg, label='B-vct ' + str(i))
            axes[i].set_yscale('log')
            axes[i].set_ylim([10, s.fs // 2])
            axes[i].set_xlim([0, 1])  # adjust limits accordingly
            axes[i].set_title(str(i))
            if i != 0:
                axes[i].set_yticklabels([])
            elif i == 0:
                axes[i].set_ylabel('Frequency (Hz)')

        fig.text(0.5, 0.04, 'Amplitudes (x $10^{-3}$)', ha='center')

        if self.save_path is not None:
            plt.savefig(self.save_path.joinpath('sources.png'))
        plt.show()
