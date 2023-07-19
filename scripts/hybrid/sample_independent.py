import numpy as np
import os, sys, time

sys.path.append('../../src/')
import pt, sbitools, loader_pk, loader_pk_splits
from utils import BoltzNet

import argparse
import emcee
import torch


parser = argparse.ArgumentParser(description='Arguments for simulations to run')
parser.add_argument('--isim', type=int, help='Simulation number to run')
parser.add_argument('--testsims', default=False, action='store_true')
parser.add_argument('--no-testsims', dest='testsims', action='store_false')
parser.add_argument('--cfgfolder', type=str, help='folder of the sweep')
args = parser.parse_args()
print()

nposterior = 5
nsteps, nwalkers, ndim = 10000, 20, 6
burn_in, thin = nsteps//10, 10


## Parse arguments
if args.testsims:
    testidx = np.load('/mnt/ceph/users/cmodi/HySBI/test-train-splits/test-N2000-f0.15-S0.npy')
    isim = testidx[args.isim]
else:
    isim = args.isim
print(f"Sample for LH {isim}")


# Setup paths
base_path = "/mnt/ceph/users/cmodi/HySBI/matter/networks/snle/"
cfg_path = f"{base_path}/{args.cfgfolder}/"
if not os.path.isdir(cfg_path):
    print(f'Configuration folder does not exist at path {cfg_path}.\nCheck cfgfolder argument')
    sys.exit()
save_path = cfg_path.replace('networks/snle/', 'samples/hybrid_independent/emcee_chains/')
os.makedirs(save_path, exist_ok=True)
print("samples will be saved at : ", save_path)


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


################################################
# load and process data
pk = np.load(f'{data_path}/pkmatter_quijote.npy')[isim, :, 1].reshape(1, -1)
k =  np.load(f'{data_path}/kmatter_quijote.npy')
# params = np.load(f'{data_path}/params_quijote_lh.npy')[isim]
ksplit, kmax = 0.15, sweepdict['cfg'].kmax

# setup large scale data and likelihood
print("\nSetting up large scale data")
cfg_large = {"kmin": 0.001, "kmax":ksplit, "offset_amp":0, "ampnorm":False}
cfg_large = sbitools.Objectify(cfg_large)
klarge, pk_large = loader_pk.k_cuts(cfg_large, pk=pk.copy(), k=k.copy())
cov = np.load(f'{data_path}/cov_disconnected_cs1_quijote.npy')[1:klarge.size+1, 1] #1st row is k=0

# setup small scales
print("\nSetting up small scale data")
ksmall, pk_small, offset = loader_pk_splits.process_pk(sweepdict['cfg'], k, pk, verbose=True)
if sweepdict['scaler'] is not None:
    pk_small_processed = sbitools.standardize(pk_small.copy(), scaler=sweepdict['scaler'], log_transform=sweepdict['cfg'].logit)[0]
# get log prob
prior = sbitools.quijote_prior(offset=0., round=False)
sweepid = sweepdict['sweepid']    
posteriors = []
for j in range(nposterior):
    name = sweepdict['names'][j]
    model_path = f"{sweepdict['cfg'].analysis_path}/{sweepid}/{name}/"
    posteriors.append(sbitools.load_posterior(model_path))



################################################
# log probability
prior_cs = [-5, 5]

def log_prob_small(theta, data):
    batch = theta.shape[0]
    data = torch.from_numpy(np.array([data]*batch).astype(np.float32).reshape(batch, data.shape[-1]))
    # cp = torch.from_numpy(cp.astype(np.float32))
    lp = 0.
    for p in posteriors:
        lp += p.potential_fn.likelihood_estimator.log_prob(data, theta)
    lp /= nposterior
    lp = lp.detach().numpy()
    return lp


def log_prob_large_vec(theta, data, kdata, cov):
    cp, cs = theta[:, :-1], theta[:, -1]
    pklin = model.interp(cp)
    pct = np.array(list(map(lambda p: pkmatter.pct(p[0])(kdata, p[1]), zip(pklin, cs) )))
    p1loop = np.array(list(map(lambda f: f(kdata), modelspt.interp(cp))))
    pred = p1loop + pct
    chisq = (pred - data)**2/cov
    lk = -0.5 * np.sum(chisq, axis=1)
    #prior only on cs
    lpr = np.zeros_like(lk)
    lpr[(cs < prior_cs[0])] = -np.inf
    lpr[(cs > prior_cs[1])] = -np.inf
    #  logprob
    lp = lpr + lk
    if np.isnan(lp).sum():
        raise ValueError("log prob is NaN")
    return lp

def log_prob(theta, params_large, params_small):
    cp = theta[:, :-1]
    cp = torch.from_numpy(cp.astype(np.float32))
    data_large, kdata, cov = params_large
    data_small = params_small[0]
    lk_large = log_prob_large_vec(theta, data_large, kdata, cov)
    lk_small = log_prob_small(cp, data_small)
    lpr = prior.log_prob(cp).detach().numpy()
    lp = lk_large + lk_small + lpr
    return lp


params_large = [pk_large, klarge, cov]
params_small = [pk_small_processed]
# Initialize and sample
np.random.seed(42)
cp0 = np.stack([prior.sample() for i in range(nwalkers)])
cs0 = np.random.uniform(prior_cs[0], prior_cs[1], nwalkers).reshape(-1, 1)
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
