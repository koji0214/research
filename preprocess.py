#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
from scipy.interpolate import interp1d
import seaborn as sns

class EMG:
    # 基本変数の定義
    labels = ["Rt_TA", "Rt_SOL", "Rt_GM", "Rt_GL", "Rt_VM", "Rt_VL", "Rt_Ham", "Lt_TA", 
                "Lt_SOL", "Lt_GM", "Lt_GL", "Lt_VM", "Lt_VL", "Lt_Ham", "Rt_foot", "Lt_foot"]
    sampling_rate = 1000  # サンプリングレート
    low_freq = None   # ローパスフィルタ周波数
    high_freq = None  # ハイパスフィルタ周波数
    smoothing = False # 平滑化
    nyq = sampling_rate/2
    lln = False
    
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
        
        self.foot_sensor = self.raw.iloc[:,14:]
        self.raw.columns = self.labels
        _rt = self.raw["Rt_foot"].values
        _rt = _rt - min(_rt)
        _rt[_rt < np.max(_rt)/2] = 0
        _events = [1 if _rt[j-1] == 0 and _rt[j]>0 else 0 for j in range(len(_rt))]
        
        _lt = self.raw["Lt_foot"].values
        _lt = _lt - min(_lt)
        _lt[_lt < np.max(_lt)/2] = 0
        _events2 = [2 if _lt[j-1] == 0 and _lt[j]>0 else 0 for j in range(len(_lt))]
        
        # self.foot_sensor = self.raw.iloc[:,14:]

        self.raw["Rt_foot"] = _events
        self.raw["Lt_foot"] = _events2
        
        _events3 = self.raw.iloc[:,14:]
        self.events = _events3.max(axis=1)# eventsというイベントデータ(0: no event,1: rt_foot,2: lt_foot)
        self.foot_sensor = self.raw.iloc[:,14:]
        self.raw = self.raw.iloc[:,:14]   # rawというEMGデータ
        self.raw = self.raw - self.raw.mean()
        
    def approx(self, x, method, n):
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
    
    
    
    def smooth(self, dat=None, freq=20, degree = 4, filtered = False):
        if self.smoothing and dat is None:
            ValueError("This Instance is already smoothed!")
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
    
  
    def epoching(self, dat=None, tmin=None, tmax=None, n=None, lln = False, list = True, foot = "Rt_foot"):
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

        self.epochs = None
        if tmin or tmax:
            self.emg_drop_log = []
            for i, idx in enumerate(event_idx):
                if i == len(event_idx):
                    break
                elif len(dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:]) < (tmax -tmin)*1000:
                    self.emg_drop_log.append(f"the {i+1}th epoch has too short! exclude this epoch. {len(dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:])}")
                    continue
                elif self.epochs is not None:
                    self.epochs = np.append(self.epochs, dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:].values[np.newaxis], axis=0)
                else:
                    self.epochs = dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:].values[np.newaxis]
            self.lln = True
            self.lln_epochs = self.epochs
        else:
            emg_epochs = []
            for ev in range(len(event_idx)):
                if ev+1 == len(event_idx):
                    break
                else:
                    emg_epochs.append(dat.iloc[event_idx[ev]:event_idx[ev+1]])
            self.epochs = emg_epochs
            
            if lln or n:
                self.lln_list(n)
        
      
    def lln_list(self, n=100):
        if self.lln:
            ValueError("This Instance cannot be done LLN! It is already done LLN.")
        method = "linear"
        epochs = np.zeros([len(self.epochs), n, self.epochs[0].shape[1]])
        for i, epoch in enumerate(self.epochs):
            for j, mus in enumerate(epoch):
                if i == len(self.epochs):
                    break
                epochs[i,:,j] = self.approx(epoch[mus], method, n)
        self.lln_epochs = epochs
        self.lln = True

    def plot_bar(self, foot = "Rt_foot", ax = None, ymax = None, ymin = None):
        if "Rt" in foot:
            idx = self.events[self.events == 1].index
        elif "Lt" in foot:
            idx = self.events[self.events == 2].index
        else:
            KeyError("foot must be 'Rt' or 'Lt'" )
        
        length = []
        for i in range(len(idx)):
            if i == len(idx)-1:
                break
            length.append(idx[i+1]-idx[i])

        if ax != None:
            ax.bar(range(len(length)), length)
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel("time(msec)")
        else:
            plt.bar(range(len(length)), length)
            plt.ylim(ymin, ymax)
            plt.ylabel("time(msec)")
    
    def plot_box(self, foot = "Rt_foot", ax = None, ymax = None, ymin = None, labels = None, strip = True):
        if "Rt" in foot:
            idx = self.events[self.events == 1].index
        elif "Lt" in foot:
            idx = self.events[self.events == 2].index
        else:
            KeyError("foot must be 'Rt' or 'Lt'" )
        
        length = []
        for i in range(len(idx)):
            if i == len(idx)-1:
                break
            length.append(idx[i+1]-idx[i])
        if labels is None:
            labels = 1
        df = pd.DataFrame({f"{labels}" : length})
        df_melt = pd.melt(df)
        
        sns.boxplot(x = "variable",y = "value", data=df_melt, ax = ax)
        if strip:
            sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax=ax)

        if ax is None:
            plt.ylim(ymin,ymax)
            plt.ylabel("time(msec)")
            plt.xlabel("RAS")
        else:    
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("RAS")
            ax.set_ylabel("time(msec)")
        
    
    @staticmethod
    def comp_box(emgs, labels = None, foot = "Rt_foot", ax = None, ymax = None, ymin = None, strip = True):
        if labels == None or len(emgs) != len(labels):
            labels = np.arange(len(emgs))
        if len(emgs) != len(labels):
            print("labels is not same length to emgs.")
        # dat = pd.DataFrame()
        for i, emg in enumerate(emgs):
            if "Rt" in foot:
                idx = emg.events[emg.events == 1].index
            elif "Lt" in foot:
                idx = emg.events[emg.events == 2].index
            else:
                KeyError("foot must be 'Rt' or 'Lt'" )
            
            length = []
            for j in range(len(idx)):
                if j == len(idx)-1:
                    break
                length.append(idx[j+1]-idx[j])
            # dat[labels[i]] = length  長さが違うのでこれではだめ。はじめから2列のデータフレームを作る。
            if i == 0:
                dat = pd.DataFrame({"variable":labels[i], "value":length})
            else:
                _dat = pd.DataFrame({"variable":labels[i], "value":length})
                dat = pd.concat([dat, _dat])

        sns.boxplot(x = "variable",y = "value", data=dat, ax = ax)
        if strip:
            sns.stripplot(x='variable', y='value', data=dat, jitter=True, color='black', ax=ax)

        if ax is None:
            plt.ylim(ymin,ymax)
            plt.ylabel("time(msec)")
            plt.xlabel("RAS")
        else:    
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("RAS")
            ax.set_ylabel("time(msec)")
        
    # 異常周期を除外するためのコード(±3SDで除外99.7％)(未実装)
    def exclude_epoch(self):
        epochs = self.epochs

    def plot_raw(self, figsize=(10, 18), add_mean = True, ymax=None, ymin=None):
        if not self.lln:
            ValueError("This Instance is not done LLN.")

        fig, ax = plt.subplots(7, 2, figsize = figsize)
        x = np.linspace(0, 100, self.lln_epochs.shape[1])
        for i, epoch in enumerate(self.lln_epochs):
            for j, (mus, a) in enumerate(zip(epoch.T, ax.flatten(order="F"))):
                if add_mean:
                    a.plot(x, mus, color="grey")
                else:
                    a.plot(x, mus)
        
        if add_mean:
            dat_mean = self.lln_epochs.mean(axis=0)
            for i, a in enumerate(ax.flatten(order="F")):
                a.plot(x, dat_mean[:,i], color="black")
            
        for i, a in enumerate(ax.flatten(order="F")):
            a.set_title(self.labels[i])
            a.set_ylabel("amplitude")
            a.set_xlabel("sycle(%)")
            a.set_ylim(ymin,ymax)

    def plot_corr(self, hist = False):
        if self.lln:
            ValueError("plot_corr() must be done after epoching(lln=True) or lln_list()")
        if not hist:
            corr = []
            for i in range(self.lln_epochs.shape[2]):
                ave = self.lln_epochs[:,:,i].mean(axis=0)
                plt.plot(ave)


# %%

import os
os.getcwd()

# %%

fname = "../data/Data_original/Elderly/EMG/sub05/noRAS1.mat"
emg = EMG(fname)
# %%

emg.filering(degree=4, high_freq=0.5,low_freq=250)
emg.smooth()
emg.epoching(n=1000)
emg.plot_raw(ymax=0.03, ymin=0, add_mean=True)
# %%
fname = "../data/Data_original/Elderly/EMG/sub05/noRAS1.mat"
emg_no = EMG(fname)
fname = "../data/Data_original/Elderly/EMG/sub05/noRAS2.mat"
emg_no2 = EMG(fname)
fname = "../data/Data_original/Elderly/EMG/sub05/RAS90.mat"
emg_90 = EMG(fname)
fname = "../data/Data_original/Elderly/EMG/sub05/RAS100.mat"
emg_100 = EMG(fname)
fname = "../data/Data_original/Elderly/EMG/sub05/RAS110.mat"
emg_110 = EMG(fname)

EMG.comp_box([emg_no, emg_no2, emg_90, emg_100, emg_110], labels=["no","no2","90","100","110"], ymin = 700, ymax = 1400)
# # %%
# emg_no.filering(degree=4, high_freq=0.5,low_freq=250)
# emg_no.smooth()
# emg_no.epoching(n=1000)
# emg_no.plot_raw(ymax=0.1, ymin=0, add_mean=True)
# plt.savefig("../misc/plot/1216/emg_signal_no.png")

# # %%
# emg_no2.filering(degree=4, high_freq=0.5,low_freq=250)
# emg_no2.smooth()
# emg_no2.epoching(n=1000)
# emg_no2.plot_raw(ymax=0.1, ymin=0, add_mean=True)
# plt.savefig("../misc/plot/1216/emg_signal_no2.png")
# # %%
# emg_90.filering(degree=4, high_freq=0.5,low_freq=250)
# emg_90.smooth()
# emg_90.epoching(n=1000)
# emg_90.plot_raw(ymax=0.1, ymin=0, add_mean=True)
# plt.savefig("../misc/plot/1216/emg_signal_90.png")
# # %%
# emg_100.filering(degree=4, high_freq=0.5,low_freq=250)
# emg_100.smooth()
# emg_100.epoching(n=1000)
# emg_100.plot_raw(ymax=0.1, ymin=0, add_mean=True)
# plt.savefig("../misc/plot/1216/emg_signal_100.png")
# # %%
# emg_110.filering(degree=4, high_freq=0.5,low_freq=250)
# emg_110.smooth()
# emg_110.epoching(n=1000)
# emg_110.plot_raw(ymax=0.1, ymin=0, add_mean=True)
# plt.savefig("../misc/plot/1216/emg_signal_110.png")

#%%
emg_no.filering(degree=4, high_freq=0.5,low_freq=250)
emg_no.smooth()
emg_no.epoching(n=1000)
emg_no.plot_corr()
# %%
