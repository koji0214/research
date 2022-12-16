import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
import seaborn as sns
import os.path as op
import skfda
from sklearn.decomposition import NMF
import tslearn
import networkx
from glob import glob
from scipy.interpolate import interp1d

class EMG:
    # 基本変数の定義
    labels = ["Rt_TA", "Rt_SOL", "Rt_GM", "Rt_GL", "Rt_VM", "Rt_VL", "Rt_Ham", "Lt_TA", 
                "Lt_SOL", "Lt_GM", "Lt_GL", "Lt_VM", "Lt_VL", "Lt_Ham", "Rt_foot", "Lt_foot"]
    sampling_rate = 1000  # サンプリングレート
    low_freq = None   # ローパスフィルタ周波数
    high_freq = None  # ハイパスフィルタ周波数
    smoothing = False # 平滑化
    nyq = sampling_rate/2
    
    def __init__(self, fname):
        mat = sp.io.loadmat(fname)
        _dat = mat["data"]
        _start_idx = mat["datastart"]
        _end_idx = mat["dataend"]
        
        self.raw = pd.DataFrame()
        for s, e, la in zip(_start_idx, _end_idx, self.labels):
            if s == -1:
                continue
            else:
                la = la.replace(" ", "")
                self.raw[la] = _dat[0][int(s-1):int(e-1)]
        
        self.raw.columns = self.labels
        _rt = self.raw["Rt_foot"].values
        _rt = _rt - min(_rt)
        _rt[_rt < np.max(_rt)/2] = 0
        _events = [1 if _rt[j-1] == 0 and _rt[j]>0 else 0 for j in range(len(_rt))]
        
        _lt = self.raw["Lt_foot"].values
        _lt = _lt - min(_lt)
        _lt[_lt < np.max(_lt)/2] = 0
        _events2 = [1 if _lt[j-1] == 0 and _lt[j]>0 else 0 for j in range(len(_lt))]
        
        self.raw["Rt_foot"] = _events
        self.raw["Lt_foot"] = _events2
        
        _events3 = self.raw.iloc[:,14:]
        self.events = _events3.max(axis=1)# eventsというイベントデータ(0: no event,1: rt_foot,2: lt_foot)
        self.raw = self.raw.iloc[:,:14]   # rawというEMGデータ
        self.raw = self.raw - self.raw.mean()
        
    def approx(x, method, n):
        y = np.arange(0, len(x), 1)
        f = interp1d(y, x, kind = method)
        y_resample = np.linspace(0, len(x)-1, n)
        return f(y_resample)
    
    
    def filering(self, degree, high_freq = 0.5, low_freq = 500, btype = "bandpass"):
        self.high_freq, self.low_freq = high_freq, low_freq
        self.filtered = self.raw.copy()
        
        h_freq = high_freq/self.nyq
        l_freq = low_freq/self.nyq
        b, a = sp.signal.butter(degree, [h_freq, l_freq], btype=btype)
        for ch in self.filtered.columns:
            self.filtered[ch] = sp.signal.filtfilt(b, a, self.filtered[ch])
    
    
    
    def smooth(self, dat, freq, degree = 4, filtered = False):
        if dat:
            self.smoothed = dat
        elif self.high_freq == None or self.low_freq == None:
            self.smoothed = self.raw.copy()
        else:
            self.smoothed = self.filtered.copy()
        low_pass = degree/self.nyq
        
        b2, a2 = sp.signal.butter(degree, low_pass, btype = 'lowpass')
        for ch in self.smoothed.columns:
            self.smoothed[ch] = sp.signal.filtfilt(b2, a2, self.smoothed[ch])
            self.smoothed[ch] = np.abs(sp.signal.hilbert(self.smoothed[ch]))
        self.smoothing = True
    
  
    def epoching(self, dat=None, tmin=None, tmax=None, n=None, lln = False, list = False, foot = "Rt_foot"):
        if "Rt" in foot:
            foot = 1
        elif "Lt" in foot:
            foot = 2
        else:
            KeyError("foot must be 'Rt' or 'Lt'" )
        event_idx = self.events[self.events == foot].index
        
        # datはあればそのまま、それ以外では一番最新の前処理をしたところまでのもの
        if dat:
            dat = dat
        elif self.smoothing:
            dat = self.smoothed
        elif self.high_freq != None or self.low_freq != None:
            dat = self.filtered
        else:
            dat = self.raw
        
        if list or lln:
            emg_drop_log = []
            for i, idx in enumerate(event_idx):
                if i == len(event_idx):
                    break
                elif len(dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:]) < (tmax -tmin)*1000:
                    emg_drop_log.append(f"the {i+1}th epoch has too short! exclude this epoch. {len(dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:])}")
                    continue
                elif i == 0:
                    emg_epochs = dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:].values[np.newaxis]
                    print(emg_epochs)
                else:
                    emg_epochs = np.append(emg_epochs, dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:].values[np.newaxis], axis=0)
            # emg_epochs2 = emg_epochs.copy()
            # self.epochs = emg_epochs
            # self.epochs_list = emg_epochs2
            self.drop_log = emg_drop_log
        if lln or n:
            if n:
                n = n
            else:
                n = 100
            
            method = "linear"
            epochs = np.zeros([len(emg_epochs), n, emg_epochs[0].shape[1]])
            for i, epoch in enumerate(emg_epochs):
                for j, mus in enumerate(epoch):
                    if i == len(emg_epochs):
                        break
                    epochs[i,:,j] = self.approx(epoch[mus], method, n)
            self.epochs = epochs
            self.lln_epochs =  epochs

        else:
            emg_epochs = []
            for ev in range(len(event_idx)):
                if ev+1 == len(event_idx):
                    break
                else:
                    emg_epochs.append(dat.iloc[event_idx[ev]:event_idx[ev+1]])
            self.epochs = emg_epochs
        
      
    def lln_list(self, n):
        if n:
            n = n
        else:
            n =100
        method = "linear"
        epochs = np.zeros([len(self.epochs_list), n, self.epochs_list[0].shape[1]])
        for i, epoch in enumerate(self.epochs_list):
            for j, mus in enumerate(epoch):
                if i == len(self.epochs_list):
                    break
                epochs[i,:,j] = self.pprox(epoch[mus], method, n)
        self.lln_epochs =  epochs



class EEG:
  def __init__(self, fname):
    self.raw = mne.io.read_raw_edf(fname)
    
def make_montage(fname):
  montage = mne.channels.read_custom_montage(fname)
  _montage = montage.get_positions()["ch_pos"]
  
  for mtg in _montage:
    _montage[mtg] += (0, 0.01, 0.04)
  
  return mne.channels.make_dig_montage(_montage)

def read_eeg(fname, mtg_path):
    montage = mne.channels.read_custom_montage(mtg_path)
    _montage = montage.get_positions()["ch_pos"]
    for mtg in _montage:
        _montage[mtg] += (0, 0.01, 0.04)
    _montage = mne.channels.make_dig_montage(_montage)


    raw = mne.io.read_raw_edf(fname)
    ch_names = raw.info["ch_names"]
    new_names = [ch_name.replace("EEG ","").replace("-Ref","") for ch_name in ch_names]
    ch_names_dic = dict(zip(ch_names, new_names))
    mne.rename_channels(raw.info, ch_names_dic)
    retype = {"EOG":"eog"}
    raw.set_channel_types(retype)
    raw.set_montage(_montage)
    return raw.set_montage(_montage)


class Culc:
    a = 1
    b = 2
    def Sum(self):
        print(self.a + self.b)