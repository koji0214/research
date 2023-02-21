import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
from scipy.interpolate import interp1d
import seaborn as sns
from tools.rpca import rpca
# from tools.Epochs import Epochs

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
    # coh_titles = ["alpha 8~13Hz", "beta 13~26Hz", "gamma 26~30Hz", "delta 0.5~4Hz", "theta 4~8Hz"]
    
    def __init__(self, fname):
        mat = sp.io.loadmat(fname)
        _dat = mat["data"]
        _start_idx = mat["datastart"]
        _end_idx = mat["dataend"]
        
        self.data_matrix = pd.DataFrame()
        for s, e, la in zip(_start_idx, _end_idx, self.labels):
            if s == -1:
                continue
            else:
                self.data_matrix[la] = _dat[0][int(s-1):int(e-1)]
        
        self.__data_matrix__ = self.data_matrix.copy()
        self.__foot_sensor__ = self.data_matrix.iloc[:,14:].copy()

        for i, foo in enumerate(["Rt_foot", "Lt_foot"]):
            _rt = self.data_matrix[foo].values
            _rt = _rt - min(_rt)
            _rt[_rt < np.max(_rt)/3] = 0
            _events = [i+1 if _rt[j-1] == 0 and _rt[j]>0 else 0 for j in range(len(_rt))]
            _events = pd.Series(_events)
            idx = _events[_events != 0].index
            skip=False
            for i in range(len(idx)):
                if i == len(idx)-2:
                    break
                elif idx[i] == 0|idx[i-1]==0:
                    continue
                elif skip:
                    skip = False
                    continue
                lng = idx[i+2]-idx[i+1]
                sht = idx[i+1]-idx[i]
                
                if lng*2/3>sht:
                    _events[idx[i+1]]=0
                    skip = True
            
            self.data_matrix[foo] = _events
        
        self.foot_sensor = self.data_matrix[["Rt_foot", "Lt_foot"]]#.iloc[:,14:]
        self.events = self.foot_sensor.max(axis=1) # eventsというイベントデータ(0: no event,1: rt_foot,2: lt_foot)
        
        emg = self.data_matrix.iloc[:,:14]   # rawというEMGデータ
        
        self.emg_matrix = emg - emg.mean()
        self.emg_raw = self.emg_matrix.copy()
        self.cadence = self.culc_cadence()
        
        print(self, len(self.emg_raw)) # pd.DataFrameでデータの詳細について返す
    
    def _reset_data(self):
        self.emg_matrix = self.data_matrix.iloc[:,:14]
        self.foot_sensor = self.data_matrix.iloc[:,14:]
        self.events = self.foot_sensor.max(axis=1)

    def filtering(self, degree = 4, high_freq = 0.5, low_freq = 250, btype = "bandpass"):
        self.high_freq, self.low_freq = high_freq, low_freq
        emg = self.emg_matrix.abs()
        
        h_freq = high_freq/self.nyq
        l_freq = low_freq/self.nyq
        b, a = sp.signal.butter(degree, [h_freq, l_freq], btype=btype)
        for ch in emg.columns:
            emg[ch] = sp.signal.filtfilt(b, a, emg[ch])
        self.data_matrix.iloc[:,:14] = emg
        self._reset_data
        # return emg
    
    def smooth(self, freq=20, degree = 4):
        low_pass = degree/self.nyq
        emg = self.emg_matrix
        
        b2, a2 = sp.signal.butter(degree, low_pass, btype = 'lowpass')
        for ch in emg.columns:
            emg[ch] = sp.signal.filtfilt(b2, a2, emg[ch])
            emg[ch] = np.abs(sp.signal.hilbert(emg[ch]))
        
        self.data_matrix.iloc[:,:14] = emg
        self._reset_data
        self.smoothing = True
    
    def crop(self, tmin = None, tmax = None):
        self.data_matrix = self.data_matrix.iloc[tmin:tmax,:]
        self.__data_matrix__ = self.__data_matrix__.iloc[tmin:tmax,:]
        self.__foot_sensor__ = self.__data_matrix__.iloc[tmin:tmax,14:]
        self.emg_raw = self.emg_raw.iloc[tmin:tmax,:]
        self._reset_data()
            
    # def epoching(self, dat=None, tmin=None, tmax=None, n=None, lln = False, list = True, foot = "Rt_foot"):
    #     if "Rt" in foot:
    #         foot = 1
    #     elif "Lt" in foot:
    #         foot = 2
    #     else:
    #         KeyError("foot must be 'Rt' or 'Lt'" )
    #     event_idx = self.events[self.events == foot].index
        
    #     # datはあればそのまま、それ以外では一番最新の前処理をしたところまでのもの
    #     if dat is locals():
    #         dat = dat
    #     elif self.smoothing:
    #         dat = self.smoothed
    #     elif self.high_freq != None or self.low_freq != None:
    #         dat = self.filtered
    #     else:
    #         dat = self.emg_matrix

    #     self.epochs = None
    #     if tmin or tmax:
    #         self.emg_drop_log = []
    #         for i, idx in enumerate(event_idx):
    #             if i == len(event_idx):
    #                 break
    #             elif len(dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:]) < (tmax -tmin)*1000:
    #                 self.emg_drop_log.append(f"the {i+1}th epoch has too short! exclude this epoch. {len(dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:])}")
    #                 continue
    #             elif self.epochs is not None:
    #                 self.epochs = np.append(self.epochs, dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:].values[np.newaxis], axis=0)
    #             else:
    #                 self.epochs = dat.iloc[int(idx+tmin*1000):int(idx+tmax*1000),:].values[np.newaxis]
    #         self.lln = True
    #         self.lln_epochs = self.epochs
    #         return self.epochs
    #     else:
    #         emg_epochs = []
    #         for ev in range(len(event_idx)):
    #             if ev+1 == len(event_idx):
    #                 break
    #             else:
    #                 emg_epochs.append(dat.iloc[event_idx[ev]:event_idx[ev+1]])
    #         self.epochs = emg_epochs
            
    #         if lln or n:
    #             self.lln_epochs = self.lln_list(n)
    #         # return self.lln_epochs
        
      
    # def lln_list(self, n=100):
    #     if self.lln:
    #         ValueError("This Instance cannot be done LLN! It has already done LLN.")
    #     method = "linear"
    #     epochs = np.zeros([len(self.epochs), n, self.epochs[0].shape[1]])
    #     for i, epoch in enumerate(self.epochs):
    #         for j, mus in enumerate(epoch):
    #             if i == len(self.epochs):
    #                 break
    #             epochs[i,:,j] = self.approx(epoch[mus], method, n)
    #     self.lln_epochs = epochs
    #     self.lln = True
    #     return self.lln_epochs

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
    
    def plot_box(self, foot = "Rt_foot", ax = None, ymax = None, ymin = None, labels = None, strip = True, showfliers = True):
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
        
        sns.boxplot(x = "variable",y = "value", data=df_melt, ax = ax, showfliers = showfliers)
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
    def comp_epochs(emgs, labels = None, foot = "Rt_foot", ax = None, ymax = None, ymin = None, strip = True, showfliers = True):
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

        sns.boxplot(x = "variable",y = "value", data=dat, ax = ax, showfliers = showfliers)
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
        

    def plot_raw(self, figsize=(15, 35), add_mean = True, ymax=None, ymin=None):
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

    def culc_epoch_len(self, foot = "Rt"):
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
        return length
    
    def culc_cadence(self,foot="Rt"):
        self.mean_length = np.median(self.culc_epoch_len())
        return 60/self.mean_length*1000
        
    def make_epochs(self):
        return Epochs()