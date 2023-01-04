# %%
from preprocess import EMG
import matplotlib.pyplot as plt

isElderly = "Young"
sub = "sub01"
ras = "noRAS1"

from glob import glob
fname = glob(f"../data/Data_original/{isElderly}/EMG/{sub}/*{ras}*.mat")
#%%
emg = EMG(fname[0])
emg.filering(degree=4, high_freq=0.5,low_freq=250)
emg.smooth()
emg.epoching(n=1000)

# 各筋ごとにばらつきがどの程度かを相関係数で評価
emg.plot_corr(hist=True) 
emg.plot_corr(hist=False)
emg.plot_corr(hist=True, heatmap=True)
plt.savefig(f"../misc/plot/1223/{isElderly}_{sub}_{ras}_corr.png")

# # %%

# # 筋のばらつきが特定の周期で優位に見られるか、階層的クラスタリングで評価
# seabornのclustermapをちゃんと理解する
# https://nykergoto.hatenablog.jp/entry/2018/11/19/seaborn_の_clustermap_をちゃんと理解する
import seaborn as sns
cluster = sns.clustermap(abs(emg.bigCorr))
plt.savefig(f"../misc/plot/1223/{isElderly}_{sub}_{ras}_cluster_cor.png")

# %%
fname = glob(f"../data/Data_original/{isElderly}/EMG/{sub}/*{ras}*.mat")
emg = EMG(fname[0])

coherence = emg.culc_coherence()
emg.plot_coherence(average=False)
plt.savefig(f"../misc/plot/1223/{isElderly}_{sub}_{ras}_coh.png")


# %%
for i, _coherence in enumerate(emg.coherence_map):
    g = cluster = sns.clustermap(_coherence)
    g.ax_col_dendrogram.set_title(emg.coh_titles[i])
    plt.savefig(f"../misc/plot/1223/{isElderly}_{sub}_{ras}_{emg.coh_titles[i]}_coh.png")

plt.show()
# %%
