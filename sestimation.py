#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:40:21 2023

@author: bmhi
"""
import os.path as op
from mne.datasets import fetch_fsaverage
from mne import make_forward_solution, compute_covariance
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.viz import plot_cov
def source_estimate(Epochs, **kwgs):
    fs_dir = fetch_fsaverage()
    subject_dir = op.dirname(fs_dir)
    subject, trans = "fsaverage","fsaverage"
    src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
    fwd = make_forward_solution(Epochs.info, trans, src, bem, eeg=True, mindist=5.)
    noise_cov = compute_covariance(Epochs, tmax = 0., method=["shrunk","empirical"],rank=None)
    # fig_cov, fig_spectra = plot_cov(noise_cov, eegno.info)
    evoke = Epochs.average()
    inv = make_inverse_operator(evoke.info, fwd, noise_cov)
    method = "dSPM"
    snr = 3.
    lambda2 = 1. / snr ** 2
    stc, residual = apply_inverse(evoke, inv, lambda2,
                                  method=method, pick_ori=None,
                                  return_residual=True, verbose=True)
    return stc, residual
