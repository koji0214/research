# %%
import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
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

def rpca(epochs, max_iter=800,p_interval=50,fix_ep=True, plot=True, threshold = 4):
    if plot:
        fig, ax = plt.subplots(epochs.shape[2], 2, figsize = (10, 4*epochs.shape[2]))
    tot_idx = np.ones([epochs.shape[0], epochs.shape[2]], dtype="bool")
    epochs_mn = np.zeros([epochs.shape[1], epochs.shape[2]])
    for i in range(epochs.shape[2]):
        mus_i = epochs[:,:,i].T
        L, S = rpca_native(mus_i, max_iter=10000)
        ymin = S.mean(axis=1) - threshold*np.std(S,axis=1)
        ymax = S.mean(axis=1) + threshold*np.std(S,axis=1)
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
    res = {"mean":epochs_mn,"fix_epoch":epochs_fix,"fix_idx":idx}
    return epochs_mn
