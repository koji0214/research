import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
from scipy.interpolate import interp1d
import seaborn as sns
# from matplotlib import animation, rc
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from rpca import rpca

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
    coh_titles = ["alpha 8~13Hz", "beta 13~26Hz", "gamma 26~30Hz", "delta 0.5~4Hz", "theta 4~8Hz"]
    
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
        return self.filtered
    
    
    
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
        if dat is locals():
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
            return self.epochs
        else:
            emg_epochs = []
            for ev in range(len(event_idx)):
                if ev+1 == len(event_idx):
                    break
                else:
                    emg_epochs.append(dat.iloc[event_idx[ev]:event_idx[ev+1]])
            self.epochs = emg_epochs
            
            if lln or n:
                self.lln_epochs = self.lln_list(n)
            # return self.lln_epochs
        
      
    def lln_list(self, n=100):
        if self.lln:
            ValueError("This Instance cannot be done LLN! It has already done LLN.")
        method = "linear"
        epochs = np.zeros([len(self.epochs), n, self.epochs[0].shape[1]])
        for i, epoch in enumerate(self.epochs):
            for j, mus in enumerate(epoch):
                if i == len(self.epochs):
                    break
                epochs[i,:,j] = self.approx(epoch[mus], method, n)
        self.lln_epochs = epochs
        self.lln = True
        return self.lln_epochs

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
    def comp_epochs(emgs, labels = None, foot = "Rt_foot", ax = None, ymax = None, ymin = None, strip = True):
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

    def plot_corr(self, hist = False, figsize = (12, 35), heatmap = False):
        if self.lln:
            ValueError("plot_corr() must be done after epoching(lln=True) or lln_list()")
        
        # _mn = self.lln_epochs.mean(axis=0)  # 軸固定できる
        # plt.plot(_mn)
        fig, ax = plt.subplots(7,2,figsize=figsize)
        
        bigCorr = []
        for i, a in enumerate(ax.flatten(order="F")):
            ave = self.lln_epochs[:,:,i].mean(axis=0)
            # a.plot(ave)
            ave = pd.Series(ave)
            corr = []
            for sig in self.lln_epochs[:,:,i]:
                sig = pd.Series(sig)
                corr.append(ave.corr(sig))
            if not hist:
                a.plot(np.abs(corr))
                a.set_ylim(-0.05,1.05)
                a.set_ylabel("correlation")
            else:
                a.hist(np.abs(corr), range=(0,1), bins = 20)
                # a.set_xlim(-0.05,1.05)
            bigCorr.append(corr)    
            a.set_title(self.labels[i])
        self.bigCorr = pd.DataFrame(bigCorr, index=self.labels[:14])
        if heatmap:
            fig, ax = plt.subplots(figsize=(10,8))
            # plt.imshow(np.abs(self.bigCorr), cmap="inferno", aspect=3)
            # plt.colorbar()
            sns.heatmap(np.abs(self.bigCorr), cmap="inferno")
    
    # 波形のばらつき具合を条件間で検定
    @staticmethod
    def comp_corr(emgs, labels = None, foot = "Rt_foot", ax = None, ymax = None, ymin = None, strip = True):
        for i, emg in enumerate(emgs):
            if emg.bigCorr is not locals():
                ValueError(f"{i+1}th emg has no bigCorr object.")
            
        
        
       
    
    def culc_coherence(self, degree = 4, high_freq = 5, low_freq = 250, average = True):
        filt = self.filering(degree=degree, high_freq = high_freq, low_freq = low_freq)
        epoch = self.epoching(filt)
        nperseg = 512
        res = np.zeros([int(nperseg/2)+1, 14,14])
        self.coherence = np.ndarray((1,int(nperseg/2)+1,14,14))
        self.drop_log = []
        for a, df in enumerate(epoch):
            if df.shape[0] < 512:  # 異常な長さのエポックは処理をスキップする
                self.drop_log.append(f"{a}th epoch is droped because it is too short to calculate coherence")
                continue
            for i,la1 in enumerate(df):
                for j,la2 in enumerate(df):
                    x = df[la1]
                    y = df[la2]
                    f, Cxy = signal.coherence(y, x, fs=1000, nperseg=nperseg)
                    res[:,i,j] = Cxy
                    
            self.coherence = np.append(self.coherence, res[np.newaxis],axis=0)
        a = 8<=f
        b = f<13
        alpha = a==b  # alpha
        a = 13<=f
        b = f < 26
        beta = a==b  #beta
        a = 26<=f
        b = f < 30
        gamma = a==b  #gamma
        a = 0.5<=f
        b = f < 4
        delta = a==b  #delta
        a = 4<=f
        b = f < 8
        theta = a==b  # theta
        crt = {"alpha 8~13Hz" : alpha, "beta 13~26Hz" : beta, "gamma 26~30Hz" : gamma, "delta 0.5~4Hz" : delta, "theta 4~8Hz" : theta}
        self.coherence = self.coherence ** 2
        
        self.bigCoh = [[] for i in range(5)]  # 周期ごとのcoherenceを記録
        stack_coh = {"alpha 8~13Hz" : np.zeros([1,14,14]), "beta 13~26Hz" : np.zeros([1,14,14]), "gamma 26~30Hz" : np.zeros([1,14,14]),
                     "delta 0.5~4Hz" : np.zeros([1,14,14]), "theta 4~8Hz" : np.zeros([1,14,14])}  # epochごとのコヒーレンスをarrayとして記録し、平均する用
        for coh in self.coherence:
            for i, ct in enumerate(crt):
                coherence = coh[crt[ct],:,:].mean(axis=0)
                stack_coh[ct] = np.append(stack_coh[ct], coherence[np.newaxis], axis = 0)
                coherence = pd.DataFrame(coherence, index=self.labels[:14], columns=self.labels[:14])
                self.bigCoh[i].append(coherence)

        if average:
            self.bigCoh_average = []
            for st in stack_coh:
                coh = stack_coh[st]
                coh = coh.mean(axis=0)
                self.bigCoh_average.append(coh)
            
            return self.bigCoh_average

        return self.bigCoh
    
    
    def plot_coherence(self, average = True):
        if average:
            coh = self.bigCoh_average
            fig, ax = plt.subplots(5,1,figsize=(7,15))
            for i, (c, a) in enumerate(zip(coh, ax)):
                im = a.imshow(c)
                plt.colorbar(im, ax=a)
                a.set_title(self.coh_titles[i])
        else:
            coh = self.bigCoh
            self.coherence_map = []
            fig,ax = plt.subplots(5, 1, figsize=(10,20))
            for i, (_coh, a) in enumerate(zip(coh, ax)):
                for j, c in enumerate(_coh):
                    _coherence = []
                    index = []
                    for _r in range(14):
                        for _c in range(14):
                            if _c >= _r:
                                continue
                            index.append(f"{c.index[_c]}-{c.index[_r]}")
                            _coherence.append(c.iat[_r, _c])
                    _coherence = pd.Series(_coherence, index=index)
                    if j == 0:
                        coherence = _coherence
                    else:
                        coherence = pd.concat([coherence, _coherence], axis=1)
                # im = a.imshow(coherence, cmap="inferno", aspect=0.3)
                # plt.colorbar(im, ax=a)
                coherence.columns = np.arange(coherence.shape[1])
                self.coherence_map.append(coherence)
                sns.heatmap(coherence, ax=a, cmap="inferno")
                a.set_title(self.coh_titles[i])
    
    
