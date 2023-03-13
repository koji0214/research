import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
from scipy.interpolate import interp1d
import seaborn as sns
from tools.rpca import rpca
from tools.function import func_gen_epochs, func_gen_rejects
from tools.prep import create_epochs, create_align_epochs

class EMG:
    # 基本変数の定義
    _labels = ["Rt_TA", "Rt_SOL", "Rt_GM", "Rt_GL", "Rt_VM", "Rt_VL", "Rt_Ham", "Lt_TA", 
                "Lt_SOL", "Lt_GM", "Lt_GL", "Lt_VM", "Lt_VL", "Lt_Ham", "Rt_foot", "Lt_foot"]
    labels = ["Rt_TA", "Rt_SOL", "Rt_GM", "Rt_GL", "Rt_VM", "Rt_VL", "Rt_Ham", "Lt_TA", 
                "Lt_SOL", "Lt_GM", "Lt_GL", "Lt_VM", "Lt_VL", "Lt_Ham"]
    sampling_rate = 1000  # サンプリングレート
    low_freq = None   # ローパスフィルタ周波数
    high_freq = None  # ハイパスフィルタ周波数
    smoothing = False # 平滑化
    nyq = sampling_rate/2
    lln = False
    # coh_titles = ["alpha 8~13Hz", "beta 13~26Hz", "gamma 26~30Hz", "delta 0.5~4Hz", "theta 4~8Hz"]
    
    def __init__(self, fname, name = None, verbose = True):
        mat = sp.io.loadmat(fname)
        _dat = mat["data"]
        _start_idx = mat["datastart"]
        _end_idx = mat["dataend"]
        
        self.data_matrix = pd.DataFrame()
        for s, e, la in zip(_start_idx, _end_idx, self._labels):
            if s == -1:
                continue
            else:
                self.data_matrix[la] = _dat[0][int(s-1):int(e-1)]
        
        self.__data_matrix__ = self.data_matrix.copy()
        self.__foot_sensor__ = self.data_matrix.iloc[:,14:].copy()

        for i, foo in enumerate(["Rt_foot", "Lt_foot"]):
            _rt = self.data_matrix[foo].values
            _rt = _rt - min(_rt)
            _rt[_rt < np.max(_rt)/5] = 0
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
                
                if lng/2>sht:
                    _events[idx[i+1]]=0
                    skip = True
            
            self.data_matrix[foo] = _events
        
        self.foot_sensor = self.data_matrix[["Rt_foot", "Lt_foot"]]#.iloc[:,14:]
        self.events = self.foot_sensor.max(axis=1) # eventsというイベントデータ(0: no event,1: rt_foot,2: lt_foot)
        
        emg = self.data_matrix.iloc[:,:14]   # rawというEMGデータ
        
        self.emg_matrix = emg - emg.mean()
        self.emg_raw = self.emg_matrix.copy()
        self.cadence = self.culc_cadence()
        if name:
            self.name = name
        else:
            self.name = fname
        
        # if verbose:
        #     la = pd.DataFrame(data={"file":fname,
        #                     "length":len(self.emg_matrix),
        #                     "cadence":self.cadence
        #                     })
        #     print(la) # pd.DataFrameでデータの詳細について返す
    
    def _reset_data(self):
        self.emg_matrix = self.data_matrix.iloc[:,:14].copy()
        self.foot_sensor = self.data_matrix.iloc[:,14:].copy()
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
        self._reset_data()
    
    def smooth(self, freq=20, degree = 4, method="low-path", window=100):
        if self.smoothing:
            ValueError("This instance has already smoothing!")
        if method == "low-path":
            low_pass = degree/self.nyq
            emg = self.emg_matrix
            
            b2, a2 = sp.signal.butter(degree, low_pass, btype = 'lowpass')
            for ch in emg.columns:
                emg[ch] = sp.signal.filtfilt(b2, a2, emg[ch])
                emg[ch] = np.abs(sp.signal.hilbert(emg[ch]))
        elif method == "movag":
            emg = self.emg_matrix
            emg = emg.abs()
            emg = emg.rolling(window=window).mean()
        else:
            KeyError("Method is value error. Use 'low-pass' or 'movag'.")
        self.data_matrix.iloc[:,:14] = emg
        self._reset_data()
        self.smoothing = True
    
    def crop(self, tmin = None, tmax = None):
        tmin, tmax = int(tmin*1000), int(tmax*1000)
        self.data_matrix = self.data_matrix.iloc[tmin:tmax,:].reset_index(drop=True)
        # self.data_matrix = self.data_matrix
        self.__data_matrix__ = self.__data_matrix__.iloc[tmin:tmax,:]
        self.__foot_sensor__ = self.__data_matrix__.iloc[tmin:tmax,14:]
        self.emg_raw = self.emg_raw.iloc[tmin:tmax,:]
        self._reset_data()
        return self

    def plot_bar(self, foot = "Rt_foot", ax = None, ymax = None, ymin = None):
        length = self.culc_epoch_len(foot=foot)
        if ax != None:
            ax.bar(range(len(length)), length)
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel("time(msec)")
        else:
            plt.bar(range(len(length)), length)
            plt.ylim(ymin, ymax)
            plt.ylabel("time(msec)")
    
    def plot_box(self, foot = "Rt_foot", ax = None, ymax = None, ymin = None, labels = None, strip = True, showfliers = True):
        length = self.culc_epoch_len(foot=foot)
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
        for i, emg in enumerate(emgs):
            length = emg.culc_epoch_len(foot=foot)
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
        

    def plot_raw(self, tmin,tmax,figsize=(7, 20), ymax=None, ymin=None):
        fig, ax = plt.subplots(7, 2, figsize = figsize)
        data = self.emg_matrix.iloc[tmin:tmax,:]
        for a,dd in zip(ax.flatten(order="F"),data):
            data[dd].plot(ax=a)
            a.set_title(dd)
        plt.suptitle(self.name)
        plt.tight_layout()

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
    
    def gen_norm_epochs(self,foot="Rt"):
        return create_align_epochs(self, foot=foot)  # return 3D-NdArray
    
    def gen_unnorm_epochs(self, foot="Rt"):
        return create_epochs(self, foot=foot)

    def plot_emg(self, figsize=(8,14), add_mean=True, ymin=None,ymax=None, show_all_outliers=True):
        fig, ax = plt.subplots(7,2,figsize=figsize)
        epochs = self.gen_norm_epochs()
        idx = func_gen_rejects(epochs, labels=None)

        x = np.linspace(0,100,100)
        for i, (a,label) in enumerate(zip(ax.flatten(order="F"), self.labels)):
            if add_mean:
                a.plot(x, epochs[idx,:,i].T, color="grey")
            else:
                a.plot(x, epochs[idx,:,i].T)
            a.set_ylabel("amplitude")
            a.set_xlabel("sycle(%)")
            a.set_ylim(ymin,ymax)
            a.set_title(label)

        if add_mean:
            epochs_mn = epochs[idx,:,:].mean(axis=0)
            for i, a in enumerate(ax.flatten(order="F")):
                a.plot(x, epochs_mn[:,i], color="black")

        if show_all_outliers:
            idx = func_gen_rejects(epochs, labels=None)
            idx = [not i for i in idx]
            print(idx.count(True))
            for i, a in enumerate(ax.flatten(order="F")):
                a.plot(x, epochs[idx,:,i].T,color="red")