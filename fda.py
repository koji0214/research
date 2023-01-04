# %%
import skfda
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as signal
import preprocess as pp
from glob import glob
fname = "../data/Data_original/Elderly/EMG/sub05/noRAS1.mat"
n =1000
emg = pp.EMG(fname)
emg.filering(degree=4, high_freq=0.5,low_freq=250)
emg.smooth()
emg.epoching(n = n)

# %%
epochs = emg.lln_epochs
epochs.shape

# %%
t = np.linspace(0 ,100, n)

# %%
fd = skfda.FDataGrid(data_matrix=epochs[:,:,0], grid_points=t)
# %%
fd.scatter()
# %%
fd
# %%
