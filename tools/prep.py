import numpy as np
import scipy as sp
import skfda
from tools.EMG2 import EMG
from tools.function import approx

def make_epochs(fname):
    dat = EMG(fname)
    raw = dat.filtering(low_freq = 250)
    raw = raw.abs()
    raw = raw.rolling(100).mean()
    dat.filtered = raw
    dat.epoching()
    dat.lln_list(100)
    return dat.lln_epochs

def smoothing(emg):
    dat = emg.emg_matrix
    dat = dat.abs()
    dat = dat.rolling(100).mean()
    emg.emg_matrix = dat

def create_epochs(emg, foot="Rt"):
    if "Rt" in foot:
        foot = 1
    elif "Lt" in foot:
        foot = 2
    else:
        KeyError("foot must be 'Rt' or 'Lt'" )
    event_idx = emg.events[emg.events == foot].index
    
    dat = emg.emg_matrix
    emg_epochs = []
    for ev in range(len(event_idx)):
        if ev+1 == len(event_idx):
            break
        else:
            emg_epochs.append(dat.iloc[event_idx[ev]:event_idx[ev+1]])
    return emg_epochs

def align_epochs(epochs, n=100):
    method = "linear"
    aln_epochs = np.zeros([len(epochs), n, epochs[0].shape[1]])
    for i, epoch in enumerate(epochs):
        for j, mus in enumerate(epoch):
            if i == len(epochs):
                break
            aln_epochs[i,:,j] = approx(epoch[mus], method, n)
    return aln_epochs

def create_align_epochs(emg, foot="Rt", n=100):
    epochs = create_epochs(emg,foot)
    epochs = align_epochs(epochs,n)
    return epochs