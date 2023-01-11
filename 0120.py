# %%
# 0120
import preprocess as pp
import msynergy as ms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rpca import rpca
from glob import glob
from matplotlib import pyplot as plt

degree = 4
h_freq = 0.5
l_freq = 250
n = 100

isYoung = "Young"
n=1
fname = glob(f'../data/Data_original/{isYoung}/EMG/sub0{n}/*noRAS1.mat')[0]
fname
#%%
dat = pp.EMG(fname)
dat.filering(degree=degree,high_freq=h_freq,low_freq=l_freq)
dat.smooth()
dat.epoching(n=n)
dat_ep = dat.lln_epochs
dat.plot_raw()
plt.suptitle(f'{isYoung}_sub0{n}_noRAS', y=0.995)
plt.tight_layout()
# %%
# dat_mn = rpca(dat_ep, fix_ep=False)
#%%
dat_mn = rpca(dat_ep)
plt.suptitle(f'{isYoung}_sub0{n}_noRAS', y=0.999)
plt.tight_layout()
plt.savefig(f"misc/plot/0120/{isYoung}_sub0{n}_noRAS_rpca.png")
# %%
import msynergy as ms
msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
msgy.fit(dat_mn, f'{isYoung}_sub0{n}_noRAS')
msgy.est_best_n()
msgy.plot_vaf()
plt.savefig(f"misc/plot/0120/{isYoung}_sub0{n}_noRAS_comp_vaf.png")
msgy.plot_synergies()
plt.savefig(f"misc/plot/0120/{isYoung}_sub0{n}_noRAS_synergy.png")