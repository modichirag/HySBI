import numpy as np
import sys, os
import sbitools
import argparse
import pickle


#####
def k_cuts(args, pk, k=None, verbose=True):
    if k is None:
        k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')
    ikmin = np.where(k>args.kmin)[0][0]
    ikmax = np.where(k>args.kmax)[0][0]
    pk = pk[..., ikmin:ikmax]
    k = k[ikmin:ikmax]
    if verbose: print("pk shape after k-cuts : ", pk.shape)
    return k, pk


def _add_offset(offset_amp, pk, seed):
    offset = offset_amp*np.random.uniform(1, 10, np.prod(pk.shape[:2]))
    offset = offset.reshape(pk.shape[0], pk.shape[1]) # different offset for sim & HOD realization
    pk = pk + offset[..., None] #add k dimension
    return pk, offset

def add_offset(args, pk, seed=None, verbose=True):
    if seed is not None: np.random.seed(seed)
    if args.offset_amp:
        if verbose: print(f"Offset power spectra with amplitude: {args.offset_amp}")
        pk, offset = _add_offset(args.offset_amp, pk, seed)
    else:
        offset = None
    return pk, offset


def _ampnorm(ampnorm, pk):
    pk /= pk[..., ampnorm:ampnorm+1]
    return pk

def normalize_amplitude(args, pk, verbose=True):
    if args.ampnorm:
        pk = _ampnorm(args.ampnorm, pk)
        if verbose: print(f"Normalize amplitude at scale-index: {args.ampnorm}")
    return pk



def process_pk(args, k, pk, verbose=True):
    #Offset here
    pk, offset = add_offset(args, pk, verbose=verbose)
    #k cut
    k, pk = k_cuts(args, pk, k, verbose=verbose)
    # Normalize at large sacles
    pk = normalize_amplitude(args, pk, verbose=verbose)

    return k, pk, offset


# def lh_features(args, verbose=True):
#     pk = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/power_split{args.splits}.npy")
#     k = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/k_split{args.splits}.npy")
#     pk = pk[:, :args.nsubs]
#     print("Loaded power spectrum data with shape : ", pk.shape)

    # return process_pk(args, k, pk, verbose)

def lh_features(args, seed=99, verbose=True):
    if args.splits > 1:

        try:
            if args.dk != 1: #factor which with bin-width for 1Gpc/h is multiplied, default is 1
                pk = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/power_split{args.splits}-dk{args.dk}.npy")
                k = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/k_split{args.splits}-dk{args.dk}.npy")        
            else:
                pk = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/power_split{args.splits}.npy")
                k = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/k_split{args.splits}.npy")
        except Exception as e:
            print("Exception in lh_features in loader_pk_splits", e)
            print(f"loading from power_split{args.splits}.npy")
            pk = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/power_split{args.splits}.npy")
            k = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/k_split{args.splits}.npy")            
        if args.nsubs == 1:
            pk = pk[:, :1]
        else:
            np.random.seed(seed)
            pk2 = []
            for i in range(pk.shape[0]):
                idx = np.random.choice(np.arange(args.splits**3), args.nsubs, replace=False)
                if args.meanf : #take mean of all sub-boxes
                    pk2.append(np.expand_dims(pk[i][idx].mean(axis=0), axis=0))
                else: 
                    pk2.append(pk[i][idx])
            pk = np.array(pk2)
    elif args.splits == 1:
        pk = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/pk.npy")
        k = pk[0, :, 0]
        pk = np.expand_dims(pk[..., 1], axis=1)
    return process_pk(args, k, pk, verbose)


def cosmolh_params():
    #
    cosmo_params = sbitools.quijote_params()[0]
    ncosmop = cosmo_params.shape[-1]
    return cosmo_params



def loader(args, return_k=False):
    """
    Data:
    power spectrum multipoles and ngals
    Offset multipoles with a random constant amplitude scaled with offset_amp.
    """
    
    k, features, offset = lh_features(args)
    params = cosmolh_params()

    nsubs = args.nsubs
    if  (args.splits == 1) or args.meanf: nsubs = 1
    params = np.repeat(params, nsubs, axis=0).reshape(-1, nsubs, params.shape[-1])    

    if offset is not None:
        print("offset shape: ", offset.shape)
        params = np.concatenate([params, offset[..., None]], axis=-1)
        print("Params shape after adding offset: ", params.shape)
                    
    if return_k:
        return k, features, params
    else:
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

