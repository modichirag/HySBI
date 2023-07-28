import numpy as np
import sbitools
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


def process_pk(args, k, pk, verbose=True):
    #Offset here
    # pk, offset = add_offset(args, pk, verbose=verbose)
    offset = None
    #k cut
    k, pk = k_cuts(args, pk, k, verbose=verbose)

    return k, pk, offset


def lh_features(args, seed=99, verbose=True):
    if args.splits > 1:
        pk = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/power_split{args.splits}.npy")
        k = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/k_split{args.splits}.npy")
        if args.nsubs == 1:
            pk = pk[:, :1]
        else:
            np.random.seed(seed)
            pk2 = []
            for i in range(pk.shape[0]):
                idx = np.random.choice(np.arange(args.splits**3), args.nsubs, replace=False)
                pk2.append(pk[i][idx])
            pk = np.array(pk2)
    elif args.splits == 1:
        pk = np.load(f"/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/pk.npy")
        k = pk[0, :, 0]
        pk = np.expand_dims(pk[..., 1], axis=1)
    print("Loaded power spectrum data with shape : ", pk.shape)
    small_args = {'kmin': args.ksplit, 'kmax': args.kmax, 'offset_amp': args.offset_amp}
    small_args = sbitools.Objectify(small_args)
    return process_pk(small_args, k, pk, verbose)


def pkcond(args, verbose=True):
    pk = np.load('/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/pk.npy')[:, 1:, :] #skip first row of k=0
    print("Loaded power spectrum conditioning data with shape : ", pk.shape)
    k = pk[0, :, 0]
    pk = pk[..., 1]
    large_args = {'kmax': args.ksplit, 'kmin': args.kmin, 'offset_amp': args.offset_amp}
    large_args = sbitools.Objectify(large_args)
    
    if args.standardize_cond:
        fname = "scaler_cond.pkl"
        try:
            scaler = sbitools.load_scaler(args.analysis_path, fname=fname)
            pk = sbitools.standardize(pk, scaler=scaler, log_transform=args.logit_cond)[0]
        except Exception as e:
            print("EXCEPTION occured in loading scaler for conditioning", e)
            print("Fitting for the scaler and saving it")
            pk, scaler = sbitools.standardize(pk, log_transform=args.logit_cond)
            with open(args.analysis_path + fname, "wb") as handle:
                pickle.dump(scaler, handle)
    return process_pk(large_args, k, pk, verbose)


def cosmolh_params():
    #
    cosmo_params = sbitools.quijote_params()[0]
    return cosmo_params


def loader(args, return_k=False):
    """
    Data:
    """
    if args.offset_amp:
        print("offset is currently disabled for hybrid version")
        raise NotImplementedError    

    k, features, offset = lh_features(args)
    k_cond, conditioning, _ = pkcond(args)
    params = cosmolh_params()
    params = np.concatenate([params, conditioning], axis=-1)

    nsubs = args.nsubs
    if  args.splits == 1: nsubs = 1
    params = np.repeat(params, nsubs, axis=0).reshape(-1, nsubs, params.shape[-1])    

    if offset is not None:
        print("offset shape: ", offset.shape)
        params = np.concatenate([params, offset[..., None]], axis=-1)
        print("Params shape after adding offset: ", params.shape)
                    
    if return_k:
        return k, k_cond, features, params
    else:
        return features, params



def folder_path(cfgd, verbose=True):
    cuts = cfgd['datacuts']
    run = cfgd['analysis']
    analysis_path = f"/mnt/ceph/users/cmodi/HySBI/matter/networks/hybrid/"
    
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

