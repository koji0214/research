# %%
from tools.EMG2 import EMG
import mne
import numpy as np
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

# def add_event(eeg, emg):
#   length = min(eeg.last_samp, len(emg)-1)
#   foot = emg.loc[:length, ["Foot"]].values.T
#   stim = mne.create_info(ch_names = ["Heel"], sfreq=1000, ch_types = "stim")
#   event = mne.io.RawArray(data = foot, info = stim)
#   eeg.crop(0, length/1000)
#   eeg.load_data()
#   return eeg.add_channels([event]), emg.iloc[:(length+1),:]

def add_event(eeg, emg):
  length = min(eeg.last_samp, len(emg.raw)-1)
  foot = emg.events.values[np.newaxis][:,:length+1]
  stim = mne.create_info(ch_names = ["Heel"], sfreq=1000, ch_types = "stim")
  event = mne.io.RawArray(data = foot, info = stim)
  eeg.crop(0, length/1000)
  eeg.load_data()
  return eeg.add_channels([event])



def load(eeg_path, emg_path):
    fname = '../data/elect_loc64_2.elc'
    montage = make_montage(fname)
    eeg = read_eeg(eeg_path, montage)
    emg = EMG(emg_path)
    eeg = add_event(eeg,emg)
    return eeg, emg
# %%
