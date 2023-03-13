# %%
import os
os.chdir(os.environ["HOME"]+'/Desktop/research/script/script')
print(os.getcwd())
import sys
sys.path.append("./../")

import mne
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io
from glob import glob
from matplotlib import pyplot as plt
import skfda
from tools.EMG2 import EMG


from data.reject import reject_list

import tools.prep as pp
ras = ["noRAS1","RAS100","RAS110"]
ep_list = []
for isY in range(2):
    sub_list = []
    if isY == 0:
        fpath = "../data/Data_original/Young/EMG/"
    else:
        fpath = "../data/Data_original/Elderly/EMG/"
    sub = len(glob(fpath+"*"))
    
    for i in range(sub):
        ras_list = []
        for j, _ras in enumerate(ras):
            fname = glob(fpath + f"sub0{i+1}/*{_ras}.mat")[0]
            epochs = pp.make_epochs(fname)
            print(fname)
            reject_idx = reject_list[isY][i][j]
            reject_idx.append(0)
            idx = np.ones(epochs.shape[0], dtype=bool)
            idx[reject_idx] = False
            epochs = epochs[idx,:,:]
            ras_list.append(epochs)
        sub_list.append(ras_list)
    ep_list.append(sub_list)
# %%
from data.label import muscle_labels as labels
#%%
# MFPCAによる異常周期の除外
cmap = plt.get_cmap("tab10")
for s in range(2):
    _ = len(ep_list[s])
    for t in range(_):
        ep_mini = ep_list[s][t]
        fig, axes = plt.subplots(7,2,figsize=(10,20))
        for i, ep in enumerate(ep_mini):
            for j,ax in enumerate(axes.flatten(order="F")):
                raw = ep[:,:,j]
                grid_points = np.linspace(1,100,len(raw.T))
                # raw = mus
                fd = skfda.FDataGrid(data_matrix = raw, grid_points = grid_points)  # 関数データとして読み込み
                fd.plot(axes=ax, color=cmap(i),alpha=.3)
                ax.set_title(labels[j])

# %%
# Boxplotによる異常波形の除外
from skfda.exploratory.depth import ModifiedBandDepth, IntegratedDepth
from skfda.exploratory.visualization import Boxplot

# %%
cmap = plt.get_cmap("tab10")
for s in range(2):
    _ = len(ep_list[s])
    for t in range(_):
        ep_mini = ep_list[s][t]
        fig, axes = plt.subplots(7,2,figsize=(10,20))
        for i, ep in enumerate(ep_mini):
            for j,ax in enumerate(axes.flatten(order="F")):
                raw = ep[:,:,j]
                grid_points = np.linspace(1,100,len(raw.T))
                fd = skfda.FDataGrid(data_matrix = raw, grid_points = grid_points)  # 関数データとして読み込み
                fdBoxplot = Boxplot(fd,axes=ax)
                outliers = [not i for i in fdBoxplot.outliers]
                raw = raw[outliers,:]
                fd = skfda.FDataGrid(data_matrix = raw, grid_points = grid_points)  # 異常波形の除外
                fd.plot(axes=ax, color=cmap(i),alpha=.3)
                ax.set_title(labels[j])
# %%
cmap = plt.get_cmap("tab10")
var_res_list = []
for s in range(2):
    _ = len(ep_list[s])
    sub_list = []
    for t in range(_):
        ep_mini = ep_list[s][t]
        fig, axes = plt.subplots(7,2,figsize=(10,20))
        res = np.zeros([len(labels),3])
        res = pd.DataFrame(res, columns=["no","100","110"],index = labels)
        
        for i, ep in enumerate(ep_mini):
            for j,ax in enumerate(axes.flatten(order="F")):
                raw = ep[:,:,j]
                grid_points = np.linspace(1,100,len(raw.T))
                fd = skfda.FDataGrid(data_matrix = raw, grid_points = grid_points)  # 関数データとして読み込み
                fdBoxplot = Boxplot(fd,axes=ax)
                outliers = [not i for i in fdBoxplot.outliers]
                raw = raw[outliers,:]
                fd = skfda.FDataGrid(data_matrix = raw, grid_points = grid_points)  # 異常波形の除外
                # fd.plot(axes=ax, color=cmap(i),alpha=.3)
                variance = np.sqrt(fd.var().data_matrix)
                ax.plot(variance[0,:,0],color=cmap(i))
                ax.set_title(labels[j])
                res.iat[j,i]=variance.sum()
            ax.set_ylim(0,None)
            plt.suptitle("no RAS : Blue, RAS100 : Orange, RAS110 : Green")
            plt.tight_layout()
        sub_list.append(res)
    var_res_list.append(sub_list)
                
# %%
fig, ax = plt.subplots()
height1 = res["no"].values  # 点数1
height2 = res["100"].values  # 点数2
height3 = res["110"].values  # 点数3
 
left = np.arange(len(height1))  # numpyで横軸を設定

width = 0.2
 
plt.bar(left, height1, color=cmap(0), width=width, align='center')
plt.bar(left+width, height2, color=cmap(1), width=width, align='center')
plt.bar(left+2*width, height3, color=cmap(2), width=width, align='center')
 
plt.xticks(left + width, labels,rotation=45)
plt.show()
# %%
# scale
fig, ax = plt.subplots()
height1 = res["no"].values/res["no"].values  # 点数1
height2 = res["100"].values/res["no"].values  # 点数2
height3 = res["110"].values/res["no"].values  # 点数3
 
left = np.arange(len(height1))  # numpyで横軸を設定

width = 0.2
 
plt.bar(left, height1, color=cmap(0), width=width, align='center')
plt.bar(left+width, height2, color=cmap(1), width=width, align='center')
plt.bar(left+2*width, height3, color=cmap(2), width=width, align='center')
 
plt.xticks(left + width, labels,rotation=45)
plt.show()

# %%
for i,ye in enumerate(["young","elderly"]):
    for j in range(len(var_res_list[i])):
        res = var_res_list[i][j]
        fig, ax = plt.subplots(figsize=[8,4])
        height1 = res["no"].values/res["no"].values  # 点数1
        height2 = res["100"].values/res["no"].values  # 点数2
        height3 = res["110"].values/res["no"].values  # 点数3
        
        left = np.arange(len(height1))  # numpyで横軸を設定

        width = 0.2
        
        plt.bar(left, height1, color=cmap(0), width=width, align='center')
        plt.bar(left+width, height2, color=cmap(1), width=width, align='center')
        plt.bar(left+2*width, height3, color=cmap(2), width=width, align='center')
        
        plt.xticks(left + width, labels,rotation=45)
        plt.title(f"{ye}_{j+1}")
        plt.savefig(f"../misc/plot/0217/{ye}_{j+1}_variance.png")

# %%
# 変動係数CVの計算
cmap = plt.get_cmap("tab10")
cv_res_list = []
for s in range(2):
    _ = len(ep_list[s])
    sub_list = []
    for t in range(_):
        ep_mini = ep_list[s][t]
        fig, axes = plt.subplots(7,2,figsize=(10,20))
        res = np.zeros([len(labels),3])
        res = pd.DataFrame(res, columns=["no","100","110"],index = labels)
        
        for i, ep in enumerate(ep_mini):
            for j,ax in enumerate(axes.flatten(order="F")):
                raw = ep[:,:,j]
                grid_points = np.linspace(1,100,len(raw.T))
                fd = skfda.FDataGrid(data_matrix = raw, grid_points = grid_points)  # 関数データとして読み込み
                fdBoxplot = Boxplot(fd,axes=ax)
                outliers = [not i for i in fdBoxplot.outliers]
                raw = raw[outliers,:]
                fd = skfda.FDataGrid(data_matrix = raw, grid_points = grid_points)  # 異常波形の除外
                # fd.plot(axes=ax, color=cmap(i),alpha=.3)
                variance = np.sqrt(fd.var().data_matrix)[0,:,0]
                mean = fd.mean().data_matrix[0,:,0]
                cv = variance/mean
                ax.plot(cv,color=cmap(i))
                ax.set_title(labels[j])
                res.iat[j,i]=cv.sum()
            ax.set_ylim(0,None)
            plt.suptitle("no RAS : Blue, RAS100 : Orange, RAS110 : Green")
            plt.tight_layout()
        sub_list.append(res)
    cv_res_list.append(sub_list)

# 変動係数で（分散を平均で補正してみて）みると、それほどRASによってばらつきがあるようには見えない。

# %%
for i,ye in enumerate(["young","elderly"]):
    for j in range(len(cv_res_list[i])):
        res = cv_res_list[i][j]
        fig, ax = plt.subplots(figsize=[8,4])
        height1 = res["no"].values/res["no"].values  # 点数1
        height2 = res["100"].values/res["no"].values  # 点数2
        height3 = res["110"].values/res["no"].values  # 点数3
        
        left = np.arange(len(height1))  # numpyで横軸を設定

        width = 0.2
        
        plt.bar(left, height1, color=cmap(0), width=width, align='center')
        plt.bar(left+width, height2, color=cmap(1), width=width, align='center')
        plt.bar(left+2*width, height3, color=cmap(2), width=width, align='center')
        
        plt.xticks(left + width, labels,rotation=45)
        plt.title(f"{ye}_{j+1}")
        plt.savefig(f"../misc/plot/0217/{ye}_{j+1}_cv.png")

# %%
dat = pd.read_csv("../歩行データ.csv")
dat

# %%
for isy in np.unique(dat["Y or E"]):
    d = dat[dat["Y or E"]==isy]
    # print(d)
    for sub in np.unique(d["sub"]):
        dd = d[d["sub"]==sub]
        # print(dd)
        for ras in np.unique(dd["conditions"]):
            ddd = dd[dd["conditions"]==ras]
            RAS = ddd.iat[0,3]
            cadence = ddd.iat[0,4]
            dat.query(f'`Y or E`=="{isy}" and sub == {sub} and conditions == "{ras}"').replace({"RAS":RAS})
            dat.query(f'`Y or E`=="{isy}" and sub == {sub} and conditions == "{ras}"').replace({"cadence":cadence})
# %%
dat
# %%
dd
# %%
ddd
# %%
ddd.iat[0,3]
# %%
dat[np.all([dat["sub"]==sub,dat[dat["Y or E"]==isy]])]
# %%
np.all([dat["sub"]==sub,dat[dat["Y or E"]==isy]])
# %%
dat[dat["YorE"]==isy].query(f"sub == {sub-2}")["RAS"]
# %%
dat.query(f'`Y or E`=="{isy}" and sub == {sub} and conditions == "{ras}"')
# %%
dat.columns[0]="YorE"

# %%
RAS=111
dat.query(f'`Y or E`=="{isy}" and sub == {sub} and conditions == "{ras}"')["RAS"] = RAS
# %%
dat.query(f'`Y or E`=="{isy}" and sub == {sub} and conditions == "{ras}"')["RAS"]
# %%
