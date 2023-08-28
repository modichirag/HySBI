import numpy as np
import sys, os
sys.path.append('../../src/')
import pt

k_quijote = np.load('../../data/kmatter_quijote.npy')
nmodes_quijote = np.load('../../data/nmodes_quijote.npy')

def get_cov(cp, k=None, nmodes=None, return_k=False):
    
    pkmatter = pt.PkMatter()
    pklinfunc_nb = pt.NBPklin()

    if k is None: 
        k = k_quijote
        return_k = True
    if nmodes is None: 
        nmodes = nmodes_quijote
    
    cov = 2*(pkmatter.interp(pklinfunc_nb(*cp), 1)(k))**2/nmodes

    if return_k:
        return k, cov
    else:
        return cov
    

if __name__=="__main__":

    import sbitools
    cp = sbitools.quijote_params()[1]
    cp = sbitools.quijote_params()[0][0]
    print("For cosmology : ", cp)
    k, cov = get_cov(cp)
    print(cov)
    cov_save = np.load('../../data/cov_disconnected_cs1_quijote.npy')

    print("Check k")
    print(k/cov_save[:, 0])
    print("Check cov")
    print(cov/cov_save[:, 1])