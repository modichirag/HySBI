import numpy as np
import sys, time
import os
import argparse

sys.path.append('../../src/')
import sbitools, loader_pk

import emcee, zeus
import torch
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

parser = argparse.ArgumentParser(description='Arguments for simulations to run')
parser.add_argument('--isim', type=int, default=-1, help='Simulation number to run')
parser.add_argument('--testsims', default=False, action='store_true')
parser.add_argument('--no-testsims', dest='testsims', action='store_false')
parser.add_argument('--cfgfolder', type=str, help='folder of the sweep')
args = parser.parse_args()
print(args)

nposterior = 5
nsamples = 5000

## Parse arguments
if args.testsims:
    testidx = np.load('/mnt/ceph/users/cmodi/HySBI/test-train-splits/test-N2000-f0.15-S0.npy')
else:
    testidx = np.arange(2000)

isim = args.isim


# Setup paths
print(f"Sample for LH {isim}")
base_path = "/mnt/ceph/users/cmodi/HySBI/matter/networks/snpe/"
cfg_path = f"{base_path}/{args.cfgfolder}/"
if not os.path.isdir(cfg_path):
    print(f'Configuration folder does not exist at path {cfg_path}.\nCheck cfgfolder argument')
    sys.exit()
save_path = cfg_path.replace('networks/snpe/', 'samples/snpe/')
os.makedirs(save_path, exist_ok=True)
print("samples will be saved at : ", save_path)


# get sweepdict and data
sweepdict = sbitools.setup_sweepdict(cfg_path)
print(f"Run analysis for LH {isim}")
features, params = loader_pk.loader(sweepdict['cfg'])
if sweepdict['scaler'] is not None:
    features = sbitools.standardize(features, scaler=sweepdict['scaler'], log_transform=sweepdict['cfg'].logit)[0]


sweepid = sweepdict['sweepid']    
posteriors = []
for j in range(nposterior):
    name = sweepdict['names'][j]
    model_path = f"{sweepdict['cfg'].analysis_path}/{sweepid}/{name}/"
    posteriors.append(sbitools.load_posterior(model_path))
posterior = NeuralPosteriorEnsemble(posteriors=posteriors)

if isim != -1:
    x = features[isim]
    samples = posterior.sample((nsamples,), x=torch.from_numpy(x.astype('float32')), show_progress_bars=False).detach().numpy()
    chain = np.expand_dims(samples, 1)
    np.save(f"{save_path}/LH{isim}", chain)

else:
    for i, isim in enumerate(testidx):
        if i%50 == 0:
            print(f"Iteration {i} of {len(testidx)}")
        x = features[isim]
        samples = posterior.sample((nsamples,), x=torch.from_numpy(x.astype('float32'))).detach().numpy()
        chain = np.expand_dims(samples, 1)
        np.save(f"{save_path}/LH{isim}", chain)

