import numpy as np
import sys, time
import os
import argparse

sys.path.append('../../src/')
import sbitools, loader_pk

import emcee, zeus
import torch

parser = argparse.ArgumentParser(description='Arguments for simulations to run')
parser.add_argument('--isim', type=int, help='Simulation number to run')
parser.add_argument('--testsims', default=False, action='store_true')
parser.add_argument('--no-testsims', dest='testsims', action='store_false')
parser.add_argument('--cfgfolder', type=str, help='folder of the sweep')
args = parser.parse_args()
print(args)

nposterior = 5
nsteps, nwalkers, ndim = 10000, 20, 5
burn_in, thin = nsteps//10, 10

## Parse arguments
if args.testsims:
    testidx = np.load('/mnt/ceph/users/cmodi/HySBI/test-train-splits/test-N2000-f0.15-S0.npy')
    isim = testidx[args.isim]
else:
    isim = args.isim


# Setup paths
print(f"Sample for LH {isim}")
base_path = "/mnt/ceph/users/cmodi/HySBI/matter/networks/snle/"
cfg_path = f"{base_path}/{args.cfgfolder}/"
if not os.path.isdir(cfg_path):
    print(f'Configuration folder does not exist at path {cfg_path}.\nCheck cfgfolder argument')
    sys.exit()
save_path = cfg_path.replace('networks/snle/', 'samples/snle/emcee_chains/')
os.makedirs(save_path, exist_ok=True)
print("samples will be saved at : ", save_path)


# get sweepdict and data
sweepdict = sbitools.setup_sweepdict(cfg_path)
print(f"Run analysis for LH {isim}")
features, params = loader_pk.loader(sweepdict['cfg'])
x = features[isim].reshape(1, -1)
if sweepdict['scaler'] is not None:
    x = sbitools.standardize(x, scaler=sweepdict['scaler'], log_transform=sweepdict['cfg'].logit)[0]


# get log prob
prior = sbitools.sbi_prior(params, offset=0., round=False)
sweepid = sweepdict['sweepid']    
posteriors = []
for j in range(nposterior):
    name = sweepdict['names'][j]
    model_path = f"{sweepdict['cfg'].analysis_path}/{sweepid}/{name}/"
    posteriors.append(sbitools.load_posterior(model_path))


def log_prob(theta, x):
    batch = theta.shape[0]
    x = torch.from_numpy(np.array([x]*batch).astype(np.float32).reshape(batch, x.shape[-1]))
    theta = torch.from_numpy(theta.astype(np.float32))
    lp = 0.
    for p in posteriors:
        lp += p.potential_fn.likelihood_estimator.log_prob(x, theta)
    lp /= nposterior
    lp += prior.log_prob(theta)
    lp = lp.detach().numpy()
    return lp


# Initialize and sample
np.random.seed(42)
theta0 = np.stack([prior.sample() for i in range(nwalkers)])
print(theta0.shape)
print(f"Log prob at initialization : ", log_prob(theta0[0:4], x))


# Run it for emcee
print('emcee it')
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, vectorize=True, args=(x,))
start = time.time()
sampler.run_mcmc(theta0, nsteps + burn_in, progress=True)
print("Time taken : ", time.time()-start)
chain = sampler.get_chain(flat=False, discard=burn_in, thin=thin)
np.save(f"{save_path}/LH{isim}", chain)

