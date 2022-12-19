# %%
from preprocess import EMG
import matplotlib.pyplot as plt

# %%

fname = "../data/Data_original/Elderly/EMG/sub05/noRAS1.mat"
emg = EMG(fname)
# %%

emg.filering(degree=4, high_freq=0.5,low_freq=250)
emg.smooth()
emg.epoching(n=1000)

emg.plot_raw(ymax=0.03, ymin=0, add_mean=True, figsize=(15,30))

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
fig, ax = plt.subplots(figsize=(10,18))
emg_no.plot_corr(ax=ax)

# %%
