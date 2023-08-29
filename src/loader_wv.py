import numpy as np
import sys, os
import sbitools
import argparse
import pickle


#####

def lh_features(args, verbose=True):
    s1 = np.load('/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/s1_M4_J7_Q4_e0.1_kc0.67.npy')
    s0 = np.load('/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/s0_M4_J7_Q4_e0.1_kc0.67.npy')
    assert len(s0.shape) == 2
    assert len(s1.shape) == 4
    s1 = s1[:, :args.M+1, :args.J+1].reshape(s1.shape[0], -1)
    s0 = s0[:, :args.M+1].reshape(s0.shape[0], -1)
    if args.s1only:
        features = s1.copy()
    else:
        features = np.concatenate([s0, s1], axis=1)

    return features



def cosmolh_params():
    #
    cosmo_params = sbitools.quijote_params()[0]
    ncosmop = cosmo_params.shape[-1]
    return cosmo_params



def loader(args):
    """
    Data:
    power spectrum multipoles and ngals
    Offset multipoles with a random constant amplitude scaled with offset_amp.
    """
    
    features = lh_features(args)
    params = cosmolh_params()                    
    return features, params



def folder_path(cfgd, verbose=True):
    cuts = cfgd['datacuts']
    run = cfgd['analysis']
    analysis_path = f"/mnt/ceph/users/cmodi/HySBI/matter/networks/{run['alg']}/"
    
    #folder name is decided by data-cuts imposed
    folder = ''
    for key in sorted(cuts):
        if cuts[key]:
            if verbose: print("key-val pair : ", key, str(cuts[key]))
            if type(cuts[key]) == bool: folder = folder + f"{key}"
            else: folder = folder + f'{key}{cuts[key]}'
            folder += '-'
    folder = folder[:-1] + f"{run['suffix']}/"

    return analysis_path + folder

