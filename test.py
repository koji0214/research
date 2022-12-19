#%%
from preprocess import EMG
import matplotlib.pyplot as plt

fname = "../data/Data_original/Elderly/EMG/sub05/noRAS1.mat"
emg = EMG(fname)
emg.filering(degree=4, high_freq=0.5,low_freq=250)
emg.smooth()
emg.epoching(n=1000)

# 各筋ごとにばらつきがどの程度かを相関係数で評価
# emg.plot_corr(hist=True) 
# emg.plot_corr(hist=False)

#%%

# # 筋のばらつきが特定の周期で優位に見られるか、階層的クラスタリングで評価
# import seaborn as sns
# cluster = sns.clustermap(emg.bigCorr)

#%%
# plt.show()

coherence = emg.culc_coherence(average=False)
plt.imshow(coherence[0][3])
plt.show()
# print(len(coherence))
# print(coherence[0].shape)

# %%
# print(emg.coherence[0].mean(axis=0))