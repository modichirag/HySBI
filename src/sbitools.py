import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils
import sbiplots
import pickle
from collections import namedtuple
import torch.optim as optim

class Objectify(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
###
def quijote_params():
    params = np.load('/mnt/ceph/users/cmodi/Quijote/params_lh.npy')
    params_fid = np.load('/mnt/ceph/users/cmodi/Quijote/params_fid.npy')
    ndim = len(params_fid)
    cosmonames = r'$\Omega_m$,$\Omega_b$,$h$,$n_s$,$\sigma_8$'.split(",")
    return params, params_fid, cosmonames

###
def sbi_prior(params, offset=0.25):
    '''
    Generate priors for parameters of the simulation set with offset from min and max value
    '''
    lower_bound, upper_bound = .1 * np.round(10 * params.min(0)) * (1-offset),\
                                      .1 * np.round(10 * params.max(0)) * (1+offset)
    lower_bound, upper_bound = (torch.from_numpy(lower_bound.astype('float32')), 
                                torch.from_numpy(upper_bound.astype('float32')))
    prior = utils.BoxUniform(lower_bound, upper_bound)
    return prior


###
# def test_train_split(x, y, train_size_frac=0.8, random_state=0, reshape=True):
#     '''
#     Split the data into test and training dataset
#     '''
#     train, test = train_test_split(np.arange(x.shape[0])[:, np.newaxis], 
#                                    train_size=train_size_frac, random_state=random_state)
#     data = namedtuple("data", ["trainx", "trainy", "testx", "testy"])
    
#     train_id = train.ravel()
#     test_id = test.ravel()
#     data.tidx = [train_id, test_id]
#     data.trainx = x[train_id]
#     data.testx =  x[test_id]
#     data.trainy = y[train_id]
#     data.testy = y[test_id]    
#     if reshape:
#         if len(data.trainx.shape) > 2:
#             nsim = data.trainx.shape[1] # assumes that features are on last axis
#             nfeats, nparams = data.trainx.shape[-1], data.trainy.shape[-1]
#             data.nsim, data.nfeatures, data.nparams = nsim, nfeats, nparams
#             data.trainx = data.trainx.reshape(-1, nfeats)
#             data.testx = data.testx.reshape(-1, nfeats)
#             data.trainy = data.trainy.reshape(-1, nparams)
#             data.testy = data.testy.reshape(-1, nparams)

#     return data


def split_index(n, test_frac, seed):

    np.random.seed(seed)
    idxpath = '/mnt/ceph/users/cmodi/contrastive/analysis/test-train-splits/'
    idx = np.random.permutation(np.arange(n))
    split = int((1-test_frac)*n)
    train = idx[:split]
    test = idx[split:]
    fname = f"N{n}-f{test_frac:0.2f}-S{seed}"
    np.save(f"{idxpath}train-{fname}", train)
    np.save(f"{idxpath}test-{fname}", test)
    print(f"Test-train splits saved at {idxpath}train-{fname}")
    return train, test 

    
def test_train_split(x, y, train_size_frac=0.8, random_state=0, reshape=True, retindex=False):
    '''
    Split the data into test and training dataset
    '''

    idxpath = '/mnt/ceph/users/cmodi/contrastive/analysis/test-train-splits/'
    n = x.shape[0]
    test_frac = 1 - train_size_frac
    fname = f"N{n}-f{test_frac:0.2f}-S{random_state}"

    try:
        print(f"Loading test-train split index from {idxpath}train-{fname}.npy")
        train_id = np.load(f"{idxpath}train-{fname}.npy")
        test_id = np.load(f"{idxpath}test-{fname}.npy")
        print("Successfully loaded")
        if retindex: return train_id, test_id
        
    except Exception as e:
        print("\nEXCEPTION occured in loading test_train_split")
        print(e)
        print("Generate splits now and save them")
        train_id, test_id = split_index(n, test_frac, random_state)
 
    data = namedtuple("data", ["trainx", "trainy", "testx", "testy"])
    data.tidx = [train_id, test_id]
    data.trainx = x[train_id]
    data.testx =  x[test_id]
    data.trainy = y[train_id]
    data.testy = y[test_id]    
    if reshape:
        if len(data.trainx.shape) > 2:
            nsim = data.trainx.shape[1] # assumes that features are on last axis
            nfeats, nparams = data.trainx.shape[-1], data.trainy.shape[-1]
            data.nsim, data.nfeatures, data.nparams = nsim, nfeats, nparams
            data.trainx = data.trainx.reshape(-1, nfeats)
            data.testx = data.testx.reshape(-1, nfeats)
            data.trainy = data.trainy.reshape(-1, nparams)
            data.testy = data.testy.reshape(-1, nparams)

    return data


###
def standardize(data, secondary=None, log_transform=True, scaler=None):
    '''
    Given a dataset, standardize by removing mean and scaling by standard deviation
    '''
    if log_transform:
        data = np.log10(data)
        if secondary is not None:
            secondary = np.log10(secondary)
    if scaler is None: 
        scaler = StandardScaler()
        data_s = scaler.fit_transform(data)
    else: 
        data_s = scaler.transform(data)
    if secondary is not None:
        secondary_s = scaler.transform(secondary)
        return data_s, secondary_s, scaler
    return data_s, scaler


###
def minmax(data, log_transform=True, scaler=None):
    '''
    Given a dataset, standardize by removing mean and scaling by standard deviation
    '''
    if log_transform:
        data = np.log10(data)
    if scaler is None: 
        scaler = MinMaxScaler()
        data_s = scaler.fit_transform(data)
    else: 
        data_s = scaler.transform(data)
    return data_s, scaler


###
def save_posterior(scaler, savepath):
    with open(savepath + "scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)

def save_posterior(posterior, savepath):
    with open(savepath + "posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

def save_inference(inference, savepath):
    with open(savepath + "inference.pkl", "wb") as handle:
        pickle.dump(inference, handle)


def load_scaler(savepath):
    with open(savepath + "scaler.pkl", "rb") as handle:
        return pickle.load(handle)

def load_posterior(savepath):
    with open(savepath + "posterior.pkl", "rb") as handle:
        return pickle.load(handle)

def load_inference(savepath):
    with open(savepath + "inference.pkl", "rb") as handle:
        return pickle.load(handle)


###
def sbi(trainx, trainy, prior, savepath=None, model_embed=torch.nn.Identity(),
        model='maf', nhidden=32, nlayers=5, nblocks=2,
        batch_size=128, lr=0.0005,
        validation_fraction=0.2, 
        retrain=False, summarize=False, verbose=True):

    if (savepath is not None) & (not retrain):
        try:
            print("Load an existing posterior model")
            posterior = load_posterior(savepath)
            inference = load_inference(savepath)
            if summarize:
                return posterior, inference, None
            else:
                return posterior

        except Exception as e:
            print("##Exception##\n", e)

    print("Training a new NF")
    density_estimator_build_fun = posterior_nn(model=model, \
                                               hidden_features=nhidden, \
                                               num_transforms=nlayers,
                                               num_blocks=nblocks,
                                               embedding_net=model_embed)
    
    inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)
    inference.append_simulations(
        torch.from_numpy(trainy.astype('float32')), 
        torch.from_numpy(trainx.astype('float32')))
    
    density_estimator = inference.train(training_batch_size=batch_size, 
                                        validation_fraction=validation_fraction, 
                                        learning_rate=lr,
                                        show_train_summary=verbose)
    
    posterior = inference.build_posterior(density_estimator)
    
    if savepath is not None:
        save_posterior(posterior, savepath)
        save_inference(inference, savepath)

    if summarize:
        # Log summary
        summary = {"train_log_probs":[], "validation_log_probs":[]}
        for i in range(len(inference.summary['train_log_probs'])):
            summary['train_log_probs'].append(inference.summary['train_log_probs'][i])
            summary['validation_log_probs'].append(inference.summary['validation_log_probs'][i])
        summary["best_validation_log_prob"] = inference.summary['best_validation_log_probs'][0]
        if savepath is not None:
            np.save(savepath + 'train_log_probs', summary['train_log_probs'])
            np.save(savepath + 'validation_log_probs', summary['validation_log_probs'])
            np.save(savepath + 'best_validation_log_prob', summary['best_validation_log_prob'])
        return posterior, inference, summary
    else:
        return posterior


#############
def analysis(cfgd, cfgm, features, params, verbose=True):
    data = test_train_split(features, params, train_size_frac=cfgd.train_fraction)

    ### Standaradize
    scaler = None
    if cfgd.standardize:
        try:
            scaler = load_scaler(cfgd.analysis_path)
            data.trainx = standardize(data.trainx, scaler=scaler, log_transform=cfgd.logit)[0]
            data.testx = standardize(data.testx, scaler=scaler, log_transform=cfgd.logit)[0]
        except Exception as e:
            print("EXCEPTION occured in loading scaler", e)
            print("Fitting for the scaler and saving it")
            data.trainx, data.testx, scaler = standardize(data.trainx, secondary=data.testx, log_transform=cfgd.logit)
            with open(cfgd.analysis_path + "scaler.pkl", "wb") as handle:
                pickle.dump(scaler, handle)

    ### SBI
    prior = sbi_prior(params.reshape(-1, params.shape[-1]), offset=0.2)
    print("trainx and trainy shape : ", data.trainx.shape, data.trainy.shape)
    posterior, inference, summary = sbi(data.trainx, data.trainy, prior, \
                                        model=cfgm.model,
                                        nlayers=cfgm.ntransforms,
                                        nhidden=cfgm.nhidden,
                                        nblocks=cfgm.nblocks,
                                        batch_size=cfgm.batch,
                                        lr=cfgm.lr,
                                        validation_fraction=cfgd.validation_fraction,
                                        savepath=cfgm.model_path,
                                        retrain=bool(cfgm.retrain),
                                        summarize=True,
                                        verbose=verbose)

    return data, posterior, inference, summary

    

# #
# def analysis(dataloader, args, savepath, model_embed=torch.nn.Identity()):
#     features, params = dataloader()
#     data = test_train_split(features, params, train_size_frac=0.8)

#     ### Standaradize
#     data.trainx, data.testx, scaler = standardize(data.trainx, secondary=data.testx, log_transform=True)
#     with open(savepath + "scaler.pkl", "wb") as handle:
#         pickle.dump(scaler, handle)

#     #############
#     ### SBI
#     prior = sbi_prior(params.reshape(-1, params.shape[-1]), offset=0.1)
#     posterior = sbi(data.trainx, data.trainy, prior, model_embed=model_embed, \
#                                   model=args.model, nlayers=args.nlayers, \
#                                   nhidden=args.nhidden, batch_size=args.batch, savepath=savepath)

#     ### Diagnostics
#     cosmonames = r'$\Omega_m$,$\Omega_b$,$h$,$n_s$,$\sigma_8$'.split(",")
#     cosmonames = cosmonames + ["Mcut", "sigma", "M0", "M1", "alpha"]
#     for _ in range(args.nposterior):
#         ii = np.random.randint(0, data.testx.shape[0], 1)[0]
#         savename = savepath + 'posterior%04d.png'%(data.tidx[1][ii//params.shape[1]])
#         fig, ax = sbiplots.plot_posterior(data.testx[ii], data.testy[ii], posterior, titles=cosmonames, savename=savename)
#     sbiplots.test_diagnostics(data.testx, data.testy, posterior, titles=cosmonames, savepath=savepath, test_frac=0.2, nsamples=500)



def train(x, y, model, criterion, batch_size=32, niter=100, lr=1e-3, optimizer=None, nprint=20, scheduler=None):

    if optimizer is None: optimizer = optim.Adam(model.parameters(), lr=lr)
    if scheduler is not None: scheduler = scheduler(optimizer) 
    # in your training loop:
    losses = []
    for j in range(niter+1):
        optimizer.zero_grad()   # zero the gradient buffers
        idx = np.random.randint(0, x.shape[0], batch_size)
        inp = torch.tensor(x[idx], dtype=torch.float32)
        target = torch.tensor(y[idx], dtype=torch.float32)  # a dummy target, for example
        output = model(inp)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update
        losses.append(loss.detach().numpy())
        if (j*nprint)%niter == 0: print(j, losses[-1])
        if (scheduler is not None) & ((j * batch_size)%x.shape[0] == 0) : 
            #print('scheduel step ')
            scheduler.step()

    return losses, optimizer




def embed_data(x, model, batch=256, device='cuda'):
    em = []

    for i in range(x.shape[0]//batch + 1):
        em.append(model(torch.tensor(x[i*batch : (i+1)*batch], dtype=torch.float32).to(device)).detach().cpu().numpy())
        
    em = np.concatenate(em, axis=0)
    return em
