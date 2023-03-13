from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


class Coherence:
    coh_titles = ["alpha 8~13Hz", "beta 13~26Hz", "gamma 26~30Hz", "delta 0.5~4Hz", "theta 4~8Hz"]
    nperseg = 512
    f = np.linspace(0,500,nperseg//2+1)
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
    labels = ['Rt_TA', 'Rt_SOL', 'Rt_GM', 'Rt_GL', 'Rt_VM', 'Rt_VL', 'Rt_Ham', 
              'Lt_TA', 'Lt_SOL', 'Lt_GM', 'Lt_GL', 'Lt_VM', 'Lt_VL', 'Lt_Ham']
    # def __init__(self):
    #     self.

    def culc_coherence(self, epochs, degree = 4, high_freq = 5, low_freq = 250, average = True):
        nperseg = self.nperseg
        res = np.zeros([int(nperseg/2)+1, 14,14])
        self.coherence = np.ndarray((1,int(nperseg/2)+1,14,14))
        self.drop_log = []
        for a, df in enumerate(epochs):
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
        
        self.coherence = self.coherence ** 2
        
        self.bigCoh = [[] for i in range(5)]  # 周期ごとのcoherenceを記録
        self.bigCoh = {la:[] for la in self.coh_titles}
        # stack_coh = {"alpha 8~13Hz" : np.zeros([1,14,14]), "beta 13~26Hz" : np.zeros([1,14,14]), "gamma 26~30Hz" : np.zeros([1,14,14]),
        #              "delta 0.5~4Hz" : np.zeros([1,14,14]), "theta 4~8Hz" : np.zeros([1,14,14])}  # epochごとのコヒーレンスをarrayとして記録し、平均する用
        stack_coh = {la:np.zeros([1,14,14]) for la in self.coh_titles}
        for coh in self.coherence:
            for i, ct in enumerate(self.crt):
                coherence = coh[self.crt[ct],:,:].mean(axis=0)
                stack_coh[ct] = np.append(stack_coh[ct], coherence[np.newaxis], axis = 0)
                coherence = pd.DataFrame(coherence, index=self.labels, columns=self.labels)
                self.bigCoh[ct].append(coherence)

        self.bigCoh_average = {}
        for st in stack_coh:
            coh = stack_coh[st]
            coh = coh.mean(axis=0)
            # self.bigCoh_average.append(coh)
            coh = pd.DataFrame(coh,index=self.labels,columns=self.labels)
            self.bigCoh_average[st]=coh

        if average:    
            return self.bigCoh_average

        return self.bigCoh
    
    def _plot_coherence(self, freq="beta"):
        freq = [f for f in self.coh_titles if freq in f][0]
        # freq = self.coh_titles[freq in self.coh_titles]
        dat = self.bigCoh_average[freq]
        plt.imshow(dat)
        plt.colorbar()
        plt.title(freq)
        plt.xticks(ticks=np.arange(14),labels=self.labels,rotation=90)
        plt.yticks(ticks=np.arange(14),labels = self.labels)
        # plt.show()


    # def plot_coherence(self, average = True):
    #     if average:
    #         coh = self.bigCoh_average
    #         fig, ax = plt.subplots(5,1,figsize=(7,15))
    #         for i, (c, a) in enumerate(zip(coh, ax)):
    #             im = a.imshow(c)
    #             plt.colorbar(im, ax=a)
    #             a.set_title(self.coh_titles[i])
    #     else:
    #         coh = self.bigCoh
    #         self.coherence_map = []
    #         fig,ax = plt.subplots(5, 1, figsize=(10,20))
    #         for i, (_coh, a) in enumerate(zip(coh, ax)):
    #             for j, c in enumerate(_coh):
    #                 _coherence = []
    #                 index = []
    #                 for _r in range(14):
    #                     for _c in range(14):
    #                         if _c >= _r:
    #                             continue
    #                         index.append(f"{c.index[_c]}-{c.index[_r]}")
    #                         _coherence.append(c.iat[_r, _c])
    #                 _coherence = pd.Series(_coherence, index=index)
    #                 if j == 0:
    #                     coherence = _coherence
    #                 else:
    #                     coherence = pd.concat([coherence, _coherence], axis=1)
    #             # im = a.imshow(coherence, cmap="inferno", aspect=0.3)
    #             # plt.colorbar(im, ax=a)
    #             coherence.columns = np.arange(coherence.shape[1])
    #             self.coherence_map.append(coherence)
    #             sns.heatmap(coherence, ax=a, cmap="inferno")
    #             a.set_title(self.coh_titles[i])
    def plot_coherence(self, average=True):
        print('This method has not used yet')
    
    
