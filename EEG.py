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
      foot[:,int(sti*i)] = 1
      i += 1
  stim = mne.create_info(ch_names = ["cadence"], sfreq=1000, ch_types = "stim")
  event = mne.io.RawArray(data = foot, info = stim)
  eeg.load_data()
  return eeg.add_channels([event])

# %%
# preprocessing
