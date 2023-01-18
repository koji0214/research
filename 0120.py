# %%
import os
os.getcwd()
os.chdir("./Desktop/research/script")
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
import msynergy as ms

# %%
# degree = 4
# h_freq = 0.5
# l_freq = 250
# n = 100

# isYoung = "Young"
# n=1
# fname = glob(f'../data/Data_original/{isYoung}/EMG/sub0{n}/*noRAS1.mat')[0]
# fname
# #%%
# dat = pp.EMG(fname)
# dat.filering(degree=degree,high_freq=h_freq,low_freq=l_freq)
# dat.smooth()
# dat.epoching()
# dat.lln_list()
# dat_ep = dat.lln_epochs
# dat.plot_raw()
# plt.suptitle(f'{isYoung}_sub0{n}_noRAS', y=0.995)
# plt.tight_layout()
# # %%
# # dat_mn = rpca(dat_ep, fix_ep=False)
# #%%
# dat_mn = rpca(dat_ep)
# plt.suptitle(f'{isYoung}_sub0{n}_noRAS', y=0.999)
# plt.tight_layout()
# plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_noRAS_rpca.png")
# # %%
# msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
# msgy.fit(dat_mn, f'{isYoung}_sub0{n}_noRAS')
# msgy.est_best_n()
# msgy.plot_vaf()
# plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_noRAS_comp_vaf.png")
# msgy.plot_synergies()
# plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_noRAS_synergy.png")



# %%
# 高齢者と若年者全ての人にシナジーかいせきによって生じる違いをまとめる

# ハイパーパラメータ
degree = 4
h_freq = 0.5
l_freq = 250
nn = 100
RAS = "RAS90"

sub_list = ["Young", "Elderly"]
for isYoung in sub_list:
    for n in range(4):
        n += 1
        fname = glob(f'../data/Data_original/{isYoung}/EMG/sub0{n}/*{RAS}.mat')[0]
        dat = pp.EMG(fname)
        dat.filering(degree=degree,high_freq=h_freq,low_freq=l_freq)
        dat.smooth()
        dat.epoching(n = nn)
        dat_ep = dat.lln_list()
        #dat_ep = dat.lln_epochs
        dat.plot_raw()
        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.995)
        plt.tight_layout()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_raw.png")
        
        dat_mn = rpca(dat_ep)
        dat_mn = dat_mn["mean"]
        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.999)
        plt.tight_layout()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_rpca.png")

        msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
        msgy.fit(dat_mn, f'{isYoung}_sub0{n}_{RAS}')
        msgy.est_best_n()
        msgy.plot_vaf()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_comp_vaf.png")
        msgy.plot_synergies()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_synergy.png")

# %%
ras = ["noRAS1", "RAS90", "RAS100", "RAS110"]
sub_list = ["Elderly"]
for isYoung in sub_list:
    for n in range(4):
        RAS = ras[n]
        n = 5
        fname = glob(f'../data/Data_original/{isYoung}/EMG/sub0{n}/*{RAS}.mat')[0]
        dat = pp.EMG(fname)
        dat.filering(degree=degree,high_freq=h_freq,low_freq=l_freq)
        dat.smooth()
        dat.epoching(n = nn)
        dat_ep = dat.lln_list()
        #dat_ep = dat.lln_epochs
        dat.plot_raw()
        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.995)
        plt.tight_layout()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_raw.png")
        
        dat_mn = rpca(dat_ep)
        dat_mn = dat_mn["mean"]
        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.999)
        plt.tight_layout()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_rpca.png")

        msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
        msgy.fit(dat_mn, f'{isYoung}_sub0{n}_{RAS}')
        msgy.est_best_n()
        msgy.plot_vaf()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_comp_vaf.png")
        msgy.plot_synergies()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_synergy.png")

# %%

# noRAS1
degree = 4
h_freq = 0.5
l_freq = 250
nn = 100
RAS = "noRAS1"

sub_list = ["Young", "Elderly"]
for isYoung in sub_list:
    for n in range(4):
        n += 1
        fname = glob(f'../data/Data_original/{isYoung}/EMG/sub0{n}/*{RAS}.mat')[0]
        dat = pp.EMG(fname)
        dat.filering(degree=degree,high_freq=h_freq,low_freq=l_freq)
        dat.smooth()
        dat.epoching(n = nn)
        dat_ep = dat.lln_list()
        #dat_ep = dat.lln_epochs
#        dat.plot_raw()
#        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.995)
#        plt.tight_layout()
#        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_raw_scale.png")
        
        dat_mn = rpca(dat_ep)
        dat_mn = dat_mn["mean"]
#        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.999)
#        plt.tight_layout()
#        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_rpca.png")

        msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
        msgy.fit(dat_mn, f'{isYoung}_sub0{n}_{RAS}')
        msgy.est_best_n()
        msgy.plot_vaf()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_comp_vaf_scale.png")
        msgy.plot_synergies()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_synergy_scale.png")
# %%

# RAS90
degree = 4
h_freq = 0.5
l_freq = 250
nn = 100
RAS = "RAS90"

sub_list = ["Young", "Elderly"]
for isYoung in sub_list:
    for n in range(4):
        n += 1
        fname = glob(f'../data/Data_original/{isYoung}/EMG/sub0{n}/*{RAS}.mat')[0]
        dat = pp.EMG(fname)
        dat.filering(degree=degree,high_freq=h_freq,low_freq=l_freq)
        dat.smooth()
        dat.epoching(n = nn)
        dat_ep = dat.lln_list()
        #dat_ep = dat.lln_epochs
#        dat.plot_raw()
#        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.995)
#        plt.tight_layout()
#        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_raw_scale.png")
        
        dat_mn = rpca(dat_ep)
        dat_mn = dat_mn["mean"]
#        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.999)
#        plt.tight_layout()
#        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_rpca.png")

        msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
        msgy.fit(dat_mn, f'{isYoung}_sub0{n}_{RAS}')
        msgy.est_best_n()
        msgy.plot_vaf()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_comp_vaf_scale.png")
        msgy.plot_synergies()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_synergy_scale.png")

# %%

# RAS100
degree = 4
h_freq = 0.5
l_freq = 250
nn = 100
RAS = "RAS100"

sub_list = ["Young", "Elderly"]
for isYoung in sub_list:
    for n in range(4):
        n += 1
        fname = glob(f'../data/Data_original/{isYoung}/EMG/sub0{n}/*{RAS}.mat')[0]
        dat = pp.EMG(fname)
        dat.filering(degree=degree,high_freq=h_freq,low_freq=l_freq)
        dat.smooth()
        dat.epoching(n = nn)
        dat_ep = dat.lln_list()
        #dat_ep = dat.lln_epochs
#        dat.plot_raw()
#        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.995)
#        plt.tight_layout()
#        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_raw_scale.png")
        
        dat_mn = rpca(dat_ep)
        dat_mn = dat_mn["mean"]
#        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.999)
#        plt.tight_layout()
#        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_rpca.png")

        msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
        msgy.fit(dat_mn, f'{isYoung}_sub0{n}_{RAS}')
        msgy.est_best_n()
        msgy.plot_vaf()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_comp_vaf_scale.png")
        msgy.plot_synergies()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_synergy_scale.png")

# %%

# RAS110
degree = 4
h_freq = 0.5
l_freq = 250
nn = 100
RAS = "RAS110"

sub_list = ["Young", "Elderly"]
for isYoung in sub_list:
    for n in range(4):
        n += 1
        fname = glob(f'../data/Data_original/{isYoung}/EMG/sub0{n}/*{RAS}.mat')[0]
        dat = pp.EMG(fname)
        dat.filering(degree=degree,high_freq=h_freq,low_freq=l_freq)
        dat.smooth()
        dat.epoching(n = nn)
        dat_ep = dat.lln_list()
        #dat_ep = dat.lln_epochs
#        dat.plot_raw()
#        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.995)
#        plt.tight_layout()
#        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_raw_scale.png")
        
        dat_mn = rpca(dat_ep)
        dat_mn = dat_mn["mean"]
#        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.999)
#        plt.tight_layout()
#        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_rpca.png")

        msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
        msgy.fit(dat_mn, f'{isYoung}_sub0{n}_{RAS}')
        msgy.est_best_n()
        msgy.plot_vaf()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_comp_vaf_scale.png")
        msgy.plot_synergies()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_synergy_scale.png")

# %%
degree = 4
h_freq = 0.5
l_freq = 250
nn = 100
ras = ["noRAS1", "RAS90", "RAS100", "RAS110"]
sub_list = ["Elderly"]
for isYoung in sub_list:
    for n in range(4):
        RAS = ras[n]
        n = 5
        fname = glob(f'../data/Data_original/{isYoung}/EMG/sub0{n}/*{RAS}.mat')[0]
        dat = pp.EMG(fname)
        dat.filering(degree=degree,high_freq=h_freq,low_freq=l_freq)
        dat.smooth()
        dat.epoching(n = nn)
        dat_ep = dat.lln_list()
        #dat_ep = dat.lln_epochs
        dat.plot_raw()
        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.995)
        plt.tight_layout()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_raw2.png")
        
        dat_mn = rpca(dat_ep, fix_ep=False)
        dat_mn = dat_mn["mean"]
        plt.suptitle(f'{isYoung}_sub0{n}_{RAS}', y=0.999)
        plt.tight_layout()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_rpca2.png")

        msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
        msgy.fit(dat_mn, f'{isYoung}_sub0{n}_{RAS}')
        msgy.est_best_n()
        msgy.plot_vaf()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_comp_vaf_scale2.png")
        msgy.plot_synergies()
        plt.savefig(f"../misc/plot/0120/{isYoung}_sub0{n}_{RAS}_synergy_scale2.png")

# %%


# %%
# Normalization
plt.plot(dat_mn)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(dat_mn)

# %%
# コヒーレンス解析dat.epoching(n=nn)
plt.plot(x)
