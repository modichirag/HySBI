import numpy as np
import os, sys, time
import argparse

sys.path.append('../../src/')
import pt, sbitools, loader_pk
from utils import BoltzNet
import pt_covariance

import zeus, emcee
import torch
from torch.distributions import Normal

parser = argparse.ArgumentParser(description='Arguments for simulations to run')
parser.add_argument('--isim', type=int, help='Simulation number to run')
parser.add_argument('--kmax', type=float,help='folder of the sweep')
parser.add_argument('--testsims', default=False, action='store_true')
parser.add_argument('--no-testsims', dest='testsims', action='store_false')
parser.add_argument('--varycov', default=False, action='store_true')
parser.add_argument('--no-varycov', dest='varycov', action='store_false')
args = parser.parse_args()
print(args)

isim = args.isim
kmax = args.kmax
# testsim = bool(int(sys.argv[2]))
# kmax = float(sys.argv[3])
# kmax = float(sys.argv[3])

print()
# print(f"args read : isim {isim}, testsim {atestsim}, kmax {kmax}")
# print()
nsteps, nwalkers, ndim = 10000, 20, 6
burn_in, thin = nsteps//10, 10

## Parse arguments
if args.testsims:
    #testidx = np.load('/mnt/ceph/users/cmodi/HySBI/test-train-splits/test-N2000-f0.15-S0.npy')
    testidx = np.load('/mnt/home/cmodi/Research/Projects/HySBI/data/testidx_p0-0.15-0.45_p4-0.65-0.95.npy')
    isim = testidx[isim]
else:
    pass
print(f"Sample for LH {isim}")

if args.varycov:
    save_path = f"/mnt/ceph/users/cmodi/HySBI/matter/samples/PT_varycov/kmax{kmax}/"
else:
    save_path = f"/mnt/ceph/users/cmodi/HySBI/matter/samples/PT/kmax{kmax}/"
os.makedirs(save_path, exist_ok=True)
if os.path.isfile(f"{save_path}/LH{isim}.npy"):
    print(f"Already sampled. File {save_path}/LH{isim}.npy already exists. Exiting.")
    sys.exit()
#
data_args = {"kmin": 0.001, "kmax":kmax, "offset_amp":0, "ampnorm":False}
data_args = sbitools.Objectify(data_args)
kcut, features, params = loader_pk.loader(data_args, return_k=True)

# load PT objects
data_path = '../../data/'
pklinfunc_nb = pt.NBPklin()
pkmatter = pt.PkMatter()
# load NN models
model = BoltzNet(None, d_in=5, d_out=500, nhidden=1000, log_it=True)
model.load_model(f'{data_path}/boltznet/ep3k/')
modelspt = BoltzNet(k=None, d_in=5, d_out=120, nhidden=500, log_it=False)
modelspt.load_model(f'{data_path}/sptnet/ep3k/')


# specify fake observed data
data = features[isim].copy()
kdata = np.load(f'{data_path}/kmatter_quijote.npy')[1:data.size+1]
if args.varycov:
    cp = params[isim]
    cov = pt_covariance.get_cov(cp)[1][1:data.size+1]
else:
    cov = np.load(f'{data_path}/cov_disconnected_cs1_quijote.npy')[1:data.size+1, 1]

# log probability
# priors = (list(model.lower_bounds) , list(model.upper_bounds))
prior = sbitools.quijote_prior(offset=0., round=False)
prior_cs = Normal(0., 10.)

def log_prob(x, data, kdata, cov):
    cp, cs = x[:-1], x[-1:]
    pklin = model.interp(cp)
    pct = pkmatter.pct(pklin)(kdata, cs)
    p1loop = modelspt.interp(cp)(kdata)
    pred = p1loop + pct
    chisq = (pred - data)**2/cov
    lk = -0.5 * np.sum(chisq)
    #prior
    lpr = prior.log_prob(torch.from_numpy(cp)).detach().numpy()
    # lb = (x[:-1] < priors[0]).sum()
    # ub = (x[:-1] > priors[1]).sum()
    # if lb+ub:
    #     lpr = -np.inf
    # else:
    #     lpr = 0 
    lpr_cs = prior_cs.log_prob(torch.from_numpy(cs)).numpy()[0]
    # logprob
    lp = lpr + lk + lpr_cs
    if np.isnan(lp):
        raise ValueError("log prob is NaN")
    return lp


# generate initial points
np.random.seed(42)
x0 = []
for i in range(ndim):
    if i < ndim-1: x0.append(np.random.uniform(model.lower_bounds[i], model.upper_bounds[i], nwalkers))
    else: x0.append(np.random.normal(0, 1, nwalkers))
x0 = np.array(x0).T
# check
print(x0[0])
print(f"Log prob at initialization : ", log_prob(x0[0], data, kdata, cov))

# # Run it for zeus
# print('zeus it')
# sampler = zeus.EnsembleSampler(nwalkers, ndim, log_prob, args=[data, kdata, cov])
# sampler.run_mcmc(x0, nsteps + burn_in)
# chain = sampler.get_chain(flat=False, discard=burn_in, thin=thin)
# np.save(f"/mnt/ceph/users/cmodi/HySBI/matter/samples/PT/zeus_chains/LH{isim}_kmax{kmax}", chain)

# Run it for emcee
start = time.time()
print('emcee it')
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[data, kdata, cov])
sampler.run_mcmc(x0, nsteps + burn_in, progress=True)
chain = sampler.get_chain(flat=False, discard=burn_in, thin=thin)
#np.save(f"/mnt/ceph/users/cmodi/HySBI/matter/samples/PT/emcee_chains/LH{isim}_kmax{kmax}", chain)
np.save(f"{save_path}/LH{isim}.npy", chain)
print("Time taken : ", time.time()-start)
