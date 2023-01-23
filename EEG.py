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

# %%
import load
eeg_path = '../data/Data_original/Young/EEG/Nagashima_noRAS1_Segment_0.edf'
emg_path = '../data/Data_original/Young/EMG/sub01/Nagashima_noRAS1.mat'
eeg, emg = load.load(eeg_path, emg_path)
# %%
events = mne.find_events(eeg, stim_channel = "Heel")
event_dict = {"Heel":1}
epochs = mne.Epochs(eeg, events, event_id=event_dict, tmin=-0.5, tmax=1.5,preload=True)
epochs.plot_image(picks=["Cz"])
plt.show()

evoked = epochs.average()
evoked.plot_joint(picks="eeg")
plt.show()

# %%
