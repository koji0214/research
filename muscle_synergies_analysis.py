# %%
# モジュールのインポート
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import preprocess as pp
from sklearn.decomposition import NMF

x = np.array([[1,2,3,4],[2,3,4,5],[1,3,5,7],[2,4,6,8],[5,6,7,8]])

# lossを計算する関数を定義
def culc_loss(X, n_components = None):
    nmf = NMF(n_components=n_components)
    nmf.fit(X)
    W = nmf.components_
    C = nmf.fit_transform(X)
    WC = np.dot(C, W)
    return np.sum((X - WC)**2), nmf


culc_loss(x, 2)

# %%
# 局所解に対する関数
def f_nmf(X, n_components = None, max_iter=20):
    old_loss, old_nmf = culc_loss(X, n_components=n_components)
    for i in range(max_iter - 1):
        loss, nmf = culc_loss(X, n_components=n_components)
        if loss < old_loss:
            old_loss = loss
            old_nmf = nmf
    return old_nmf


nmf = f_nmf(x, 2)

# %%
# vafを計算する関数
def culc_vaf(X, nmf):
    W = nmf.components_
    C = nmf.fit_transform(X)
    WC = np.dot(C, W)
    e = X - WC
    vaf = 1 - (np.sum(e**2)/np.sum(X**2))
    return vaf


culc_vaf(x, nmf)

# %%
# 筋ごとのvafを計算する関数
def culc_vaf_mus(X, nmf, axis = 0):
    W = nmf.components_
    C = nmf.fit_transform(X)
    WC = np.dot(C, W)
    e = X - WC
    vaf_mus = []
    if axis == 0:
        for i in range(X.shape[1]):
            xi = X.T[i]
            ei = e.T[i]
            vaf_i = 1 - (np.sum(ei**2)/np.sum(xi**2))
            vaf_mus.append(vaf_i)
    else:
        for i in range(X.shape[0]):
            xi = X[i]
            ei = e[i]
            vaf_i = 1 - (np.sum(ei**2)/np.sum(xi**2))
            vaf_mus.append(vaf_i)
    return vaf_mus


culc_vaf_mus(x, nmf, axis=0)

# %%
# 最適なシナジー数を計算する関数
def est_best_n(x, max_n_components, threshold = 0.75):
    vaf_log = []
    nmf_log = {}
    for n in range(max_n_components):
        nmf = f_nmf(x,n + 1)
        vaf = culc_vaf(x, nmf)
        print(n + 1, vaf)
        vaf_log.append(vaf)
        nmf_log[f"{n+1}"] = nmf
    
    for i, vaf in enumerate(vaf_log):
        if vaf > threshold:
            print(i + 1, vaf)
            break
    return i+1, nmf_log





est_best_n(x, 3)


# %%
