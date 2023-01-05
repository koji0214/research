# %%
# モジュールのインポート
import seaborn as sns
import preprocess as pp

fname = '../data/Data_original/Young/EMG/sub01/Nagashima_noRAS1.mat'
degree = 4
h_freq = 0.5
l_freq = 250
n = 100
dat = pp.EMG(fname)
# %%
dat.filering(degree=degree, high_freq=h_freq,low_freq=l_freq)
dat.smooth()
dat.epoching(n=n)
dat_ep = dat.lln_epochs

# %%
dat.plot_raw()
# %%
dat_ep.shape
# %%
import numpy as np
dat_mn = dat_ep.mean(axis=0)
dat_mn.shape
# %%
import msynergy as ms
msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
msgy.fit(dat_mn, 'young_sub01_noRAS')
msgy.est_best_n()
msgy.plot_vaf()
msgy.plot_synergies()
# %%
# elderly
fname = '../data/Data_original/Elderly/EMG/sub05/noRAS1.mat'
degree = 4
h_freq = 0.5
l_freq = 250
n = 100
dat = pp.EMG(fname)

dat.filering(degree=degree, high_freq=h_freq,low_freq=l_freq)
dat.smooth()
dat.epoching(n=n)
dat_ep = dat.lln_epochs

dat.plot_raw()

dat_mn = dat_ep.mean(axis=0)

msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
msgy.fit(dat_mn, 'elderly_sub05_noRAS')
msgy.est_best_n()
msgy.plot_vaf()
msgy.plot_synergies()
# %%
import numpy as np
import numpy.linalg as LA
def rpca(M, max_iter=800,p_interval=50):
    def shrinkage_operator(x, tau):
        return np.sign(x) * np.maximum((np.abs(x) - tau), np.zeros_like(x))

    def svd_thresholding_operator(X, tau):
        U, S, Vh = LA.svd(X, full_matrices=False)
        return U @ np.diag(shrinkage_operator(S, tau)) @ Vh

    i = 0
    S = np.zeros_like(M)
    Y = np.zeros_like(M)
    error = np.Inf
    tol = 1e-4 * LA.norm(M, ord="fro")
    mu = M.shape[0] * M.shape[1]/(4 * LA.norm(M, ord=1))
    mu_inv = 1/mu
    lam = 1/np.sqrt(np.max(M.shape))

    while i < max_iter:
        L = svd_thresholding_operator(M - S + mu_inv * Y, mu_inv)
        S = shrinkage_operator(M - L + mu_inv * Y, lam * mu_inv)
        Y = Y + mu * (M - L - S)
        error = LA.norm(M - L - S, ord='fro')
        if i % p_interval == 0:
            print("step:{} error:{}".format(i, error))

        if error <= tol:
            print("converted! error:{}".format(error))
            break
        i+=1
    else:
        print("Not converged")

    return L, S
# %%
L,S = rpca(dat_ep[:,:,0],max_iter=100000)
# %%
L.shape
# %%
import matplotlib.pyplot as plt
plt.plot(L.T)
# %%
plt.plot(S.T)
# %%
plt.plot(dat_ep[:,:,0].T)
# %%
len(abs(S).mean(axis=1))
# %%
np.var(S)
# %%
fig, ax = plt.subplots()
ax.plot(S.T, color = 'grey')
ax.plot(S.mean(axis=0),color='red')
# ax.plot(S.mean(axis=0) - 4*np.std(S,axis=0), color='pink')
# ax.plot(S.mean(axis=0) + 4*np.std(S,axis=0), color='pink')
ax.fill_between(np.arange(100),S.mean(axis=0) - 4*np.std(S,axis=0), S.mean(axis=0) + 4*np.std(S,axis=0), color='pink')
# %%
type(S)
# %%
