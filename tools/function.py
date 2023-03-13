from scipy.interpolate import interp1d
import numpy as np
import skfda
from skfda.exploratory.visualization import Boxplot
import pandas as pd

def approx(x, method, n):
    y = np.arange(0, len(x), 1)
    f = interp1d(y, x, kind = method)
    y_resample = np.linspace(0, len(x)-1, n)
    return f(y_resample)

def func_cad(emg, label):
    return emg.cadence*2

import matplotlib.pyplot as plt
def func_comp_ep(emgs, label):
    from tools.EMG2 import EMG
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

def func_gen_rejects(emg, labels):
    idx = []
    i = emg.shape[1]
    gp = np.linspace(1,100,i)
    for i in range(emg.shape[2]):
        dat = emg[:,:,i]
        fd = skfda.FDataGrid(data_matrix = dat, grid_points = gp)
        fdBoxplot = Boxplot(fd)
        outliers = [not i for i in fdBoxplot.outliers]
        idx.append(outliers)
    idx = np.array(idx)
    idx = idx.min(axis=0)
    return idx

def func_to_fd(emg, labels):
    fData = []
    gp = np.linspace(1,100,emg.shape[1])
    idx = func_gen_rejects(emg, labels)
    for i in range(emg.shape[2]):
        dat = emg[idx,:,i]
        fd = skfda.FDataGrid(data_matrix=dat, grid_points=gp)
        fData.append(fd)
    return {'data':fData, 'outlier':idx}


def func_calc_var(emgs, labels):
    from data.label import muscle_labels as labels
    res = np.zeros([len(labels), len(emgs)])
    res = pd.DataFrame(res, columns=emgs.keys(), index=labels)

    for j, ras in enumerate(emgs):
        emg = emgs[ras]
        for i, fd in enumerate(emg['data']):
            variance = np.sqrt(fd.var().data_matrix)
            res.iat[i,j] = variance.sum()
    return res

def func_calc_cv(emgs, labels):
    from data.label import muscle_labels as labels
    res = np.zeros([len(labels), len(emgs)])
    res = pd.DataFrame(res, columns=emgs.keys(), index=labels)

    for j, ras in enumerate(emgs):
        emg = emgs[ras]
        for i, fd in enumerate(emg['data']):
            variance = np.sqrt(fd.var().data_matrix[0,:,0])
            mean = fd.mean().data_matrix[0,:,0]
            cv = variance/mean
            res.iat[i,j] = cv.sum()
    return res
