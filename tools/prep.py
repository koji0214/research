import mne
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io
from glob import glob
from matplotlib import pyplot as plt
import skfda
from EMG import EMG

def make_epochs(fname):
    dat = EMG(fname)
    raw = dat.filtering(low_freq = 250)
    raw = raw.abs()
    raw = raw.rolling(100).mean()
    dat.filtered = raw
    dat.epoching()
    dat.lln_list(100)
    return dat.lln_epochs
