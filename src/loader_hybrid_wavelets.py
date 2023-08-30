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



def lh_features(args, seed=99, verbose=True):
    if args.splits > 1:
        s1 = np.load(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/s1_M4_J7_Q4_e0.1_kc0.67_split{args.splits}.npy')
        s0 = np.load(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/s0_M4_J7_Q4_e0.1_kc0.67_split{args.splits}.npy')
        if args.nsubs == 1:
            s0, s1 = s0[:, :1], s1[:, :1]
        else:
            np.random.seed(seed)
            s0_n, s1_n = [], []
            for i in range(s0.shape[0]):
                idx = np.random.choice(np.arange(args.splits**3), args.nsubs, replace=False)
                if args.meanf : #take mean of all sub-boxes
                    s0_n.append(np.expand_dims(s0[i][idx].mean(axis=0), axis=0))
                    s1_n.append(np.expand_dims(s1[i][idx].mean(axis=0), axis=0))
                else: 
                    s0_n.append(s0[i][idx])
                    s1_n.append(s1[i][idx])
            s0, s1 = np.array(s0_n), np.array(s1_n)

    else :
        s1 = np.load('/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/s1_M4_J7_Q4_e0.1_kc0.67.npy')
        s0 = np.load('/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/s0_M4_J7_Q4_e0.1_kc0.67.npy')
        s0, s1 = np.expand_dims(s0, axis=1), np.expand_dims(s1, axis=1)

    assert len(s0.shape) == 3
    assert len(s1.shape) == 5
    n_lh, n_reps = s0.shape[0], s0.shape[1]
    print("Loaded shapes of s0 and s1 : ", s0.shape, s1.shape)
    s1 = s1[:, :, :args.M+1, :args.J+1].reshape(n_lh, n_reps, -1)
    s0 = s0[:, :, :args.M+1]
    print("Shapes of s0 and s1 after M and J cuts: ", s0.shape, s1.shape)
    if args.s1only:
        features = s1.copy()
    else:
        features = np.concatenate([s0, s1], axis=2)

    return features



def pkcond(args, verbose=True):
    pk = np.load('/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/matter/N0256/pk.npy')[:, 1:, :] #skip first row of k=0
    print("Loaded power spectrum conditioning data with shape : ", pk.shape)
    k = pk[0, :, 0]
    pk = pk[..., 1]
    large_args = {'kmax': args.ksplit, 'kmin': args.kmin}
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

    k, pk = k_cuts(large_args, pk, k, verbose=verbose)
    return k, pk


def cosmolh_params():
    #
    cosmo_params = sbitools.quijote_params()[0]
    return cosmo_params


def loader(args):
    """
    Data:
    """
    features = lh_features(args)
    k_cond, conditioning = pkcond(args)
    params = cosmolh_params()
    params = np.concatenate([params, conditioning], axis=-1)

    nsubs = args.nsubs
    if  (args.splits == 1) or args.meanf: nsubs = 1
    params = np.repeat(params, nsubs, axis=0).reshape(-1, nsubs, params.shape[-1])    

                    
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

