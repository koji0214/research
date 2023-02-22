from scipy.interpolate import interp1d
import numpy as np

def approx(x, method, n):
    y = np.arange(0, len(x), 1)
    f = interp1d(y, x, kind = method)
    y_resample = np.linspace(0, len(x)-1, n)
    return f(y_resample)

def func_cad(emg, label):
    return emg.cadence*2

import matplotlib.pyplot as plt
from tools.EMG2 import EMG
def func_comp_ep(emgs, label):
    fig,ax = plt.subplots()
    # emgs = [emg.crop(10,120) for emg in emgs]
    EMG.comp_epochs(emgs, strip=True ,showfliers = False, labels=["noRAS1","RAS90","RAS100","RAS110"])
    plt.title(label)

def func_plt_bar(emg, label):
    fig, ax = plt.subplots()
    emg.plot_bar()
    plt.title(label)

def func_gen_epochs(emg, label):
    from tools.prep import create_align_epochs
    return create_align_epochs(emg)