#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 12:16:03 2021

This script is used to perform the NMF on hydrophone data

@author: Randall Ali
"""

import numpy as np

from matplotlib import pyplot as plt
from scipy import signal as sig

from pypam import utils

SEED = 10


class NMF:
    def __init__(self, window_time=0.1, rank=15):
        """
        window_time: Frame length in seconds
        """
        self.window_time = window_time
        self.R = rank
        self.nfft = None
        self.noverlap = None

    def __call__(self, s, V_type='Z_mag', normalize=True, verbose=False):
        """
        s: Signal object
            Signal to analyze
        V_type: string
            You can choose what you want to use for V : Z_mag, Z_phase, Gxx, etc...
            I've found that Z_mag works best, perhaps because some sources are coherent.
        normalize: bool
            Set to True if matrices should be normalized to represent probabilities
        """
        self.nfft = int(self.window_time * s.fs)
        self.noverlap = (self.nfft / 2)
        df = s.fs / self.nfft

        # Computing both the spectrogram and the STFT
        # The STFT will be needed to do the filtering with the time-freq. mask.
        # The STFT function is used since it will take into account the COLA constraints so that an iSTFT can be done.
        # The STFT has a length (time dimension) longer than the spectrogram because it does some extra padding.
        f_sg, t_sg, Gxx = s.spectrogram(nfft=self.nfft, scaling='density', db=True, overlap=0.5)

        f, t, Z_stft = sig.stft(s.signal, fs=s.fs, nperseg=self.nfft, noverlap=self.noverlap, window='hann')
        Z_mag = np.abs(Z_stft)      # Magnitude of STFT
        Z_phase = np.angle(Z_stft)  # Phase of STFT

        if verbose:
            self.plot_z_mag(f, t, Z_mag)

        # Apply the NMF Algorithm
        if V_type == 'Z_mag':
            V = Z_mag
        elif V_type == 'Z_phase':
            V = Z_phase
        elif V_type == 'Gxx':
            V = Gxx
        else:
            raise Exception('This approach is not implemented!')

        W, H, ErrConv = self.NMF_hals(V, init_type="rand_norm", max_iter=1000, tol=1e-9)
        V_approx = W @ H  # approximated V
        Residual = V - V_approx
        Residual_norm = np.linalg.norm(V - (W @ H))

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

        # Error convergence
        if verbose:
            self.plot_error_conv(ErrConv, Residual)
            self.plot_decomposition(s, f_sg, V, V_approx, W, H)

        # Step 5: Creation of time-frequency masks to obtain the individual components.
        WH_prod, G_tf, C_tf, c_tf = self.time_freq_masks(s, Z_stft, V, W, H)

        return W, H, WH_prod, G_tf, C_tf, c_tf

    def NMF_hals(self, V, init_type, max_iter, tol):
        """
        Non-negative matrix factorisation (NMF) implementation using the
        Hierarchical Alternating Least Squares (HALS) algorithm, which was firstly described in:

        Cichocki, A., & Phan, A. H. (2009). Fast local algorithms for large scale nonnegative matrix and tensor
        factorizations. IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences,

        The implementation in this script is done following:
        Gillis, N. (2011 PhD thesis) Nonnegative Matrix Factorization Complexity, Algorithms and Applications
        Ch. 4.1.3 - Algorithm (3)

        The goal is to find an approximation of the (M x N) matrix V such that:
                    V ~= WH
        where W is an (M x R) matrix consisting of R basis vectors (columns)  and H is an (R x N) matrix with the
        associated activations to the basis vectors


        Arguments:
            V          - M x N matrix data to approximate
            R          - Rank: the number of assumed basis vectors
            init_type  - Type of initialisation for W and H
                         "rand_norm" - a random initialisation with a normalisation by the 2-norm of each column
                                     - Note that a fixed seed is used here (can change to include in the function
                                     if needed)
                         (can put others here for different initialisations)
            max_iter   - maximum number of iterations
            tol        - Tolerance for convergence. The algorithm is deemed to have converged if either this tolerance
                         is reached or max_iter has been exceeded

        Returns:

            W       - M x R matrix of basis vectors
            H       - R x N matrix of activations
            ErrConv - Difference between the residual before and after each iteration - Used to make a convergence plot

        """

        M = V.shape[0]
        N = V.shape[1]

        # Initialise W and H (can include other conditions for different types of normalisations)
        np.random.seed(SEED)
        W_init = np.random.rand(M, self.R)
        H_init = np.random.rand(self.R, N)
        if init_type == "rand_norm":
            # Normalise
            W_init = W_init / np.linalg.norm(W_init, axis=0)
            H_init = H_init / np.linalg.norm(H_init, axis=0)

        l = 0  # iteration count
        eps = 1e-16  # for enforcing +ve

        W = W_init
        H = H_init

        alpha = (np.trace((V @ H.T).T @ W)) / (np.trace((W.T @ W).T @ (H @ H.T)))
        W = alpha * W

        err = np.linalg.norm(V - (W @ H))
        ErrConv = []

        while err > tol:
            err_befu = np.linalg.norm(V - (W @ H))  # Error before updating W and H
            l += 1

            # Update W
            A = V @ H.T
            B = H @ H.T
            for r in np.arange(0, self.R, 1):
                wr = (A[:, r] + W[:, r] * B[r, r] - (W @ B[:, r])) / (B[r, r])
                W[:, r] = np.maximum(wr, eps)

            # Update H
            C = W.T @ V
            D = W.T @ W
            for r in np.arange(0, self.R, 1):
                hr = (C[r, :] + D[r, r] * H[r, :] - (D[:, r] @ H)) / (D[r, r])
                H[r, :] = np.maximum(hr, eps)

            err_aftu = np.linalg.norm(V - (W @ H))  # Error after updating W and H
            err = err_befu - err_aftu

            ErrConv.append(err)

            if (l % 10) == 0:  # condition to display convergence.
                print('Iteration: ' + str(l) + ', Error: ' + str(err))

            if l > max_iter:  # In case it gets stuck or takes too long, break earlier
                print('Algorithm stopped after ' + str(max_iter) + ' iterations')
                break

        return W, H, ErrConv

    def time_freq_masks(self, s, Z_stft, V, W, H):
        """
        Compute the time-frequency masks

        Returns:
            G_tf: time-freq masks,
            C_tf: filtered STFT,
            c_tf: filtered time domain signal
        """
        # Initialising arrays
        G_tf = np.zeros([V.shape[0], V.shape[1], self.R])
        C_tf = np.zeros([V.shape[0], len(Z_stft[1]), self.R], dtype='complex')
        c_tf = np.zeros([len(s.signal), self.R])

        WH_prod = W @ H

        for n in np.arange(0, self.R, 1):
            G_tf[:, :, n] = (np.dot(W[:, [n]], H[[n], :])) / WH_prod  # Compute TF mask
            C_tf[:, 0:V.shape[1], n] = G_tf[:, :, n] * Z_stft[0:, 0:V.shape[1]]
            _, sig_td = sig.istft(C_tf[:, :, n], s.fs, nperseg=self.nfft, noverlap=self.noverlap, window='hann')
            c_tf[:, n] = sig_td[0:len(s.signal)]

        return WH_prod, G_tf, C_tf, c_tf

    @staticmethod
    # TODO implement in signal 
    def plot_z_mag(f, t, Z_mag):
        # pad_xextent = t[2] - t[1]  # get time frame interval
        # xextent = np.min(t) - pad_xextent, np.max(t) + pad_xextent  # pad the beginning and end
        # tmin, tmax = xextent
        # extent = tmin / 60, tmax / 60, f[0], f[-1]  # this defines the 4 corners of the "image"
        # cmin, cmax = np.min(Z_mag), np.max(Z_mag)

        fig, ax = plt.subplots()
        sp = ax.pcolormesh(t, f, Z_mag, shading='auto')
        ax.set_xlabel('Time (mins)')
        ax.set_ylabel('Frequency (Hz)')
        plt.colorbar(sp, ax=[ax], location='right')
        plt.show()

    @staticmethod
    def plot_error_conv(ErrConv, Residual):
        ErrConv = np.array(ErrConv)
        plt.plot(np.log(ErrConv))
        plt.xlabel('Number of iterations')
        plt.title('Residual Frob. norm = ' + str(np.linalg.norm(Residual)))
        plt.show()
        
    def plot_decomposition(self, s, f_sg, V, V_approx, W, H):
        # Decomposition Plots
        Vlg = utils.to_db(V, square=True, ref=1.0)
        Vlg_ap = utils.to_db(V_approx, square=True, ref=1.0)
        W_sklg = np.log10(W)
        H_sklg = (H)

        fig, axes = plt.subplots(2, 2, figsize=(12, 4))
        fig.subplots_adjust(hspace=0.5)  # horizontal spacing

        sp = axes[0, 0].pcolormesh(Vlg, shading='auto')
        axes[0, 0].set_xlabel('Time (mins)')
        axes[0, 0].set_ylabel('Frequency (Hz)')
        plt.colorbar(sp, ax=[axes[0, 0]], location='right')

        sp2 = axes[1, 0].pcolormesh(W_sklg, shading='auto')
        axes[1, 0].set_title('W = basis vectors')
        axes[1, 0].set_xticks(np.arange(0, self.R, 5))
        axes[1, 0].set_xlabel('Number of basis vectors')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        plt.colorbar(sp2, ax=[axes[1, 0]], location='right')

        sp3 = axes[1, 1].pcolormesh(H_sklg, shading='auto')
        axes[1, 1].set_title('H = activations')
        axes[1, 1].set_xlabel('Time (mins)')
        axes[1, 1].set_ylabel('Number of basis vectors')
        axes[1, 1].set_yticks(np.arange(0, self.R, 5))
        plt.colorbar(sp3, ax=[axes[1, 1]], location='right')

        sp4 = axes[0, 1].pcolormesh(Vlg_ap, shading='auto')
        axes[0, 1].set_title('V approximation (WH)')
        axes[0, 1].set_xlabel('Time (mins)')
        axes[0, 1].set_ylabel('Frequency (Hz)')
        plt.colorbar(sp4, ax=[axes[0, 1]], location='right')

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

        plt.show()
