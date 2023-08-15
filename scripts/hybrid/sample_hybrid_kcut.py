import numpy as np
import os, sys, time

sys.path.append('../../src/')
import pt, sbitools
import loader_hybrid, loader_pk, loader_pk_splits
from utils import BoltzNet

import argparse
import emcee
import torch
from torch.distributions import Normal


parser = argparse.ArgumentParser(description='Arguments for simulations to run')
parser.add_argument('--isim', type=int, help='Simulation number to run')
parser.add_argument('--testsims', default=False, action='store_true')
parser.add_argument('--no-testsims', dest='testsims', action='store_false')
parser.add_argument('--cfgfolder', type=str, help='folder of the sweep')
parser.add_argument('--subdata', default=False, action='store_true')
parser.add_argument('--no-subdata', dest='subdata', action='store_false')
parser.add_argument('--kcut', type=float, default=0., help='cut PT model at this scale')
args = parser.parse_args()
print()

nposterior = 10
nsteps, nwalkers, ndim = 10000, 20, 6
burn_in, thin = nsteps//10, 10


## Parse arguments
if args.testsims:
    #testidx = np.load('/mnt/ceph/users/cmodi/HySBI/test-train-splits/test-N2000-f0.15-S0.npy')
    testidx = np.load('/mnt/home/cmodi/Research/Projects/HySBI/data/testidx_p0-0.15-0.45_p4-0.65-0.95.npy')
    isim = testidx[args.isim]
else:
    isim = args.isim
print(f"Sample for LH {isim}")


# Setup paths
base_path = "/mnt/ceph/users/cmodi/HySBI/matter/networks/hybrid/"
cfg_path = f"{base_path}/{args.cfgfolder}/"
if not os.path.isdir(cfg_path):
    print(f'Configuration folder does not exist at path {cfg_path}.\nCheck cfgfolder argument')
    sys.exit()

# if still running
if args.subdata:
    save_path = cfg_path.replace('networks/hybrid/', 'samples/hybrid2_sub/')
else:
    save_path = cfg_path.replace('networks/hybrid/', 'samples/hybrid2/') 
if args.kcut != 0:
    save_path = save_path + f'/kcut{args.kcut}/'
save_path = save_path + f'ens{nposterior}/'
os.makedirs(save_path, exist_ok=True)
print("samples will be saved at : ", save_path)
if os.path.isfile(f"{save_path}/LH{isim}.npy"):
    print(f"Already sampled. File {save_path}/LH{isim}.npy already exists. Exiting.")
    sys.exit()


# load PT objects
data_path = '../../data/'
pklinfunc_nb = pt.NBPklin()
pkmatter = pt.PkMatter()
# load NN models
model = BoltzNet(None, d_in=5, d_out=500, nhidden=1000, log_it=True)
model.load_model(f'{data_path}/boltznet/ep3k/')
modelspt = BoltzNet(k=None, d_in=5, d_out=120, nhidden=500, log_it=False)
modelspt.load_model(f'{data_path}/sptnet/ep3k/')
# get sweepdict and data
print("\nloading sweep")
sweepdict = sbitools.setup_sweepdict(cfg_path)
cfg = sweepdict['cfg']

################################################
# load and process data

# setup large scale data and likelihood
print("\nSetting up large scale data")
pk = np.load(f'{data_path}/pkmatter_quijote.npy')[isim, :, 1]
k =  np.load(f'{data_path}/kmatter_quijote.npy')
if args.kcut !=0:
    idx = (k > cfg.kmin) & (k < args.kcut)
else:
    idx = (k > cfg.kmin) & (k < cfg.ksplit)
klarge, pk_large_data = k[idx], pk[idx]
cov = np.load(f'{data_path}/cov_disconnected_cs1_quijote.npy')[1:klarge.size+1, 1] #1st row is k=0

# setup conditioning for small scales
kcond, pk_cond, _ = loader_hybrid.pkcond(cfg)
pk_cond = pk_cond[isim]

# setup small scales
print("\nSetting up small scale data")
if args.subdata:
    ksmall, pk_small, _ = loader_hybrid.lh_features(cfg)
    pk_small = pk_small[isim]
else:
    ksmall, pk_small, _ = loader_pk.lh_features(cfg)
    pk_small = pk_small[isim]
    idx = (ksmall > cfg.ksplit) & (ksmall < cfg.kmax)
    ksmall, pk_small = ksmall[idx], pk_small[idx]

if sweepdict['scaler'] is not None:
    pk_small_data = sbitools.standardize(pk_small.reshape(1, -1).copy(), scaler=sweepdict['scaler'], log_transform=cfg.logit)[0]
    pk_small_data = pk_small_data[0] #remove batch dimension

################################################
# log probability
#prior_cs = [-5, 5]
prior_cs = Normal(0., 10.)
prior = sbitools.quijote_prior(offset=0., round=False)

# get log prob
sweepid = sweepdict['sweepid']    
posteriors = []
for j in range(nposterior):
    name = sweepdict['names'][j]
    model_path = f"{cfg.analysis_path}/{sweepid}/{name}/"
    posteriors.append(sbitools.load_posterior(model_path))


def log_prob_small(theta, data):
    batch = theta.shape[0]
    data = torch.from_numpy(np.array([data]*batch).astype(np.float32).reshape(batch, data.shape[-1]))
    weights = 1/nposterior
    logweights = np.log(weights)
    lks = np.stack([logweights + p.potential_fn.likelihood_estimator.log_prob(data, theta).detach() for p in posteriors], axis=0)
    lk = torch.logsumexp(torch.from_numpy(lks), dim=0).detach().numpy()
    return lk


def log_prob_large_vec(theta, data, kdata, cov):
    cp, cs = theta[:, :-1], theta[:, -1]
    pklin = model.interp(cp)
    pct = np.array(list(map(lambda p: pkmatter.pct(p[0])(kdata, p[1]), zip(pklin, cs) )))
    p1loop = np.array(list(map(lambda f: f(kdata), modelspt.interp(cp))))
    pred = p1loop + pct
    chisq = (pred - data)**2/cov
    lk = -0.5 * np.sum(chisq, axis=1)
    if np.isnan(lk).sum():
        raise ValueError("log prob is NaN")
    return lk


def log_prob(theta, params_large, params_small):
    cp, cs = theta[:, :-1], theta[:, -1]
    cp = torch.from_numpy(cp.astype(np.float32))
    data_large, kdata, cov = params_large
    data_small, pk_cond = params_small
    conditioning = np.repeat(pk_cond.reshape(1, -1), cp.shape[0], axis=0).astype(np.float32)
    cp_cond = np.concatenate([cp, conditioning], axis=-1)

    # get likelihoods
    lk_large = log_prob_large_vec(theta, data_large, kdata, cov)
    lk_small = log_prob_small(cp_cond, data_small)
    # priors
    lpr = prior.log_prob(cp).detach().numpy()
    lpr += prior_cs.log_prob(torch.from_numpy(cs)).numpy()
    # total log prob
    lp = lk_large + lk_small + lpr
    return lp


print("shape large, small, cond : ", pk_large_data.shape, pk_small_data.shape, pk_cond.shape)
params_large = [pk_large_data, klarge, cov]
params_small = [pk_small_data, pk_cond]
# Initialize and sample
np.random.seed(42)
cp0 = np.stack([prior.sample() for i in range(nwalkers)])
#cs0 = np.random.uniform(prior_cs[0], prior_cs[1], nwalkers).reshape(-1, 1)
cs0 = np.random.uniform(-5, 5, nwalkers).reshape(-1, 1)
theta0 = np.concatenate([cp0, cs0], axis=-1)
print("initial sample shape : ", theta0.shape)
print(f"Log prob at initialization : ", log_prob(theta0, params_large, params_small))
print()


# Run it for emcee
print('emcee it')
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, vectorize=True, args=(params_large, params_small))
start = time.time()
sampler.run_mcmc(theta0, nsteps + burn_in, progress=True)
print("Time taken : ", time.time()-start)
chain = sampler.get_chain(flat=False, discard=burn_in, thin=thin)
print(chain.shape)
np.save(f"{save_path}/LH{isim}", chain)
