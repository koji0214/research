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
def make_emg_obj(fname):
  dat = sp.io.loadmat(fname)
  dat_seq = dat["data"]
  st = dat["datastart"]
  ed = dat["dataend"]
  labels = dat["titles"]
  dat_emg = pd.DataFrame()
  for s, e, la in zip(st, ed, labels):
    if s == -1:
      continue
    else:
      la = la.replace(" ", "")
      dat_emg[la] = dat_seq[0][int(s-1):int(e-1)]
  if dat_emg.shape[1] == 16:
    labels = ["Rt_TA", "Rt_SOL", "Rt_GM", "Rt_GL", "Rt_VM", "Rt_VL", "Rt_Ham", "Lt_TA", 
              "Lt_SOL", "Lt_GM", "Lt_GL", "Lt_VM", "Lt_VL", "Lt_Ham", "Rt_foot", "Lt_foot"]
    dat_emg.columns = labels
    val = dat_emg["Rt_foot"].values
    val = val - min(val)
    val[val < np.max(val)/3] = 0
    events = [1 if val[j-1] == 0 and val[j]>0 else 0 for j in range(len(val))]
    dat_emg_foot = dat_emg.iloc[:,:14]
    dat_emg_foot["Foot"] = events
  else:
    val = dat_emg["Foot"].values
    val = val - min(val)
    val[val < np.max(val)/3] = 0
    events = [1 if val[j-1] == 0 and val[j]>0 else 0 for j in range(len(val))]
    dat_emg_foot = dat_emg
    dat_emg_foot["Foot"] = events
  return dat_emg_foot



# set montage
def make_montage(fname):
  montage = mne.channels.read_custom_montage(fname)
  _montage = montage.get_positions()["ch_pos"]
  
  for mtg in _montage:
    _montage[mtg] += (0, 0.01, 0.04)
  
  return mne.channels.make_dig_montage(_montage)



def read_eeg(fname, montage):
  raw = mne.io.read_raw_edf(fname)
  ch_names = raw.info["ch_names"]
  new_names = [ch_name.replace("EEG ","").replace("-Ref","") for ch_name in ch_names]
  ch_names_dic = dict(zip(ch_names, new_names))
  mne.rename_channels(raw.info, ch_names_dic)
  retype = {"EOG":"eog"}
  raw.set_channel_types(retype)
  raw.set_montage(montage)
  return raw.set_montage(montage)



def add_event(eeg, emg):
  length = min(eeg.last_samp, len(emg)-1)
  foot = emg.loc[:length, ["Foot"]].values.T
  stim = mne.create_info(ch_names = ["Heel"], sfreq=1000, ch_types = "stim")
  event = mne.io.RawArray(data = foot, info = stim)
  eeg.crop(0, length/1000)
  eeg.load_data()
  return eeg.add_channels([event]), emg.iloc[:(length+1),:]

# %%
fname = '../data/elect_loc64_2.elc'
montage = make_montage(fname)
fname = '../data/Data_original/Young/EEG/Nagashima_noRAS1_Segment_0.edf'
eeg = read_eeg(fname, montage)
# eeg.plot()
fname = '../data/Data_original/Young/EMG/sub01/Nagashima_noRAS1.mat'
emg = make_emg_obj(fname)

# %%
eeg,emg = add_event(eeg, emg)

# %%
eeg.plot()
# %%
events = mne.find_events(eeg, stim_channel = "Heel")
event_dict = {"Heel":1}
epochs = mne.Epochs(eeg, events, event_id=event_dict, tmin=-0.5, tmax=1.5,preload=True)
epochs.plot_image(picks=["Cz"])
plt.show()

# %%
evoked = epochs.average()
evoked.plot_joint(picks="eeg")
plt.show()

# %%
eeg.crop(10,)
eeg.plot()

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
