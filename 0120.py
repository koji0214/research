# %%
import preprocess as pp
import msynergy as ms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fname = '../data/Data_original/Young/EMG/sub01/Nagashima_noRAS1.mat'
degree = 4
h_freq = 0.5
l_freq = 250
n = 100
dat = pp.EMG(fname)
dat.filering(degree=degree,high_freq=h_freq,low_freq=l_freq)
dat.smooth()
dat.epoching(n=n)
dat_ep = dat.lln_epochs
dat_mn = dat_ep.mean(axis=0)
msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
msgy.fit(dat_mn, 'young_sub01_noRAS')
msgy.est_best_n()
msgy.plot_vaf()
msgy.plot_synergies()
# %%
