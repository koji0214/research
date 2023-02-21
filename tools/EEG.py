#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:55:45 2023

@author: bmhi
"""
# %%
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from EMG import EMG

# %%
def add_cadence(eeg, emg, RAS=100):
  #length = min(eeg.last_samp, len(emg.raw)-1)
  #foot = emg.events.values[np.newaxis][:length]
  lng = eeg.last_samp
  sti = emg.mean_length * RAS/100
  foot = np.zeros(shape=[1,lng+1])
  i = 0
  while(sti*i < lng):
      foot[:,int(sti*i)] = 1
      i += 1
  stim = mne.create_info(ch_names = ["cadence"], sfreq=1000, ch_types = "stim")
  event = mne.io.RawArray(data = foot, info = stim)
  #eeg.crop(0, length/1000)
  eeg.load_data()
  return eeg.add_channels([event])
# %%
def add_RAS(eeg, RAS):
  lng = eeg.last_samp
  sti = 60/RAS*1000
  foot = np.zeros(shape=[1,lng+1])
  i = 0
  while(sti*i < lng):
      if i % 2 == 0:
        foot[:,int(sti*i)] = 2
      elif i % 2 == 1:
        foot[:,int(sti*i)] = 1
      i += 1
  stim = mne.create_info(ch_names = ["cadence"], sfreq=1000, ch_types = "stim")
  event = mne.io.RawArray(data = foot, info = stim)
  eeg.load_data()
  return eeg.add_channels([event])

# %%
def epoching_hc(raw, tmin=-.2, tmax=1, foot = "Rt"):
    if "Rt" in foot:
        foot=1
    elif "Lt" in foot:
        foot=2
    events = mne.find_events(raw, stim_channel="Heel")
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax)[f"{foot}"]
    return epochs
# %%
def epoching_cad(raw, tmin=-.2, tmax=1, half = True):
    events = mne.find_events(raw, stim_channel="cadence")
    if half:
      epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax)["1"]
    else:
      epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax)
    return epochs

# %%
def preprocess(eeg,l_freq=0.2,h_freq=100,sfreq=200,bad_channels=None):
  eeg.crop(10,)
  eeg.filter(l_freq=l_freq, h_freq=h_freq,method='iir')
  eeg.resample(sfreq=sfreq)
  epochs = epoching_hc(eeg)
  epochs.average().plot_joint()
  if bad_channels:
    eeg.info['bads'].extend(bad_channels)
  eeg.set_eeg_reference(ref_channels='average', projection=True)
  epoching_hc(eeg).average().plot_joint()
  return eeg
