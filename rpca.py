# %%
import numpy as np
import numpy.linalg as LA
def rpca_native(M, max_iter=800,p_interval=50):
    def shrinkage_operator(x, tau):
        return np.sign(x) * np.maximum((np.abs(x) - tau), np.zeros_like(x))

    def svd_thresholding_operator(X, tau):
        U, S, Vh = LA.svd(X, full_matrices=False)
        return U @ np.diag(shrinkage_operator(S, tau)) @ Vh

    i = 0
    S = np.zeros_like(M)
    Y = np.zeros_like(M)
    error = np.Inf
    tol = 1e-6 * LA.norm(M, ord="fro")
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

def rpca(epochs, max_iter=800,p_interval=50,fix_ep=True, plot=True):
    if plot:
        fig, ax = plt.subplots(epochs.shape[2], 2, figsize = (10, 4*epochs.shape[2]))
    tot_idx = np.ones([epochs.shape[0], epochs.shape[2]], dtype="bool")
    epochs_mn = np.zeros([epochs.shape[1], epochs.shape[2]])
    for i in range(epochs.shape[2]):
        mus_i = epochs[:,:,i].T
        L, S = rpca_native(mus_i, max_iter=10000)
        idx = [all([all(S.T[i] > ymin), all(S.T[i] < ymax)]) for i in range(S.shape[1])]
        tot_idx[:,i] = idx
        mus_i_fix = mus_i.T[idx].T
        
        if not fix_ep:
            mus_i_mn = mus_i_fix.mean(axis = 1)
            epochs_mn[:,i] = mus_i_mn
            
            if plot:
                ax[i, 0].plot(mus_i)
                ax[i, 1].plot(mus_i_fix)
                ax[i, 1].plot(mus_i_mn,lw=5, color="k")
    if fix_ep:
        idx = tot_idx.all(axis = 1)
        epochs_fix = epochs[idx,:,:]
        epochs_mn = epochs_fix.mean(axis=0)
    
        if plot:
            for i in range(epochs.shape[2]):
                ax[i, 0].plot(epochs[:,:,i].T)
                ax[i, 1].plot(epochs_fix[:,:,i].T)
                ax[i, 1].plot(epochs_mn[:,i],lw=5, color="k")
    return epochs_mn

# %%
import preprocess as pp
import msynergy as ms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fname = '../data/Data_original/Young/EMG/sub01/Nagashima_noRAS1.mat'
degree = 4
h_freq = 0.5
l_freq = 250
n = 100
dat = pp.EMG(fname)
dat.filering(degree=degree,high_freq=h_freq,low_freq=l_freq)
dat.smooth()
dat.epoching(n=n)
dat_ep = dat.lln_epochs
dat.plot_raw()

# %%
mus_i = dat_ep[:,:,0].T
plt.plot(mus_i)

# %%
L,S = rpca_native(mus_i, max_iter=100000)
plt.plot(L)
# %%
plt.plot(S)
# %%
fig, ax = plt.subplots()
ax.plot(S, color = 'grey')
ax.plot(S.mean(axis=1),color='red')
ymin = S.mean(axis=1) - 4*np.std(S,axis=1)
ymax = S.mean(axis=1) + 4*np.std(S,axis=1)
# ax.plot(ymin, color='pink')
# ax.plot(ymax, color='pink')
ax.fill_between(np.arange(100),ymin, ymax, color='pink')

# %%
idx = [all([all(S.T[i] > ymin), all(S.T[i] < ymax)]) for i in range(S.shape[1])]
idx

# %%
mus_i_fix = mus_i.T[idx]
plt.plot(mus_i_fix.T)
fig, ax = plt.subplots()
plt.plot(mus_i)

# %%
dat_mn = rpca(dat_ep, fix_ep=False)
#%%
dat_mn = rpca(dat_ep)
# %%
msgy = ms.MuscleSynergy(max_n_components=10, max_iter=1000)
msgy.fit(dat_mn, 'young_sub01_noRAS')
msgy.est_best_n()
msgy.plot_vaf()
msgy.plot_synergies()

