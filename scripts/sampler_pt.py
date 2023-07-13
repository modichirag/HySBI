import numpy as np
import sys

sys.path.append('../src/')
import pt, sbitools, loader_pk
from utils import BoltzNet

import zeus, emcee
import torch

# load PT objects
pklinfunc_nb = pt.NBPklin()
pkmatter = pt.PkMatter()
# load NN models
model = BoltzNet(k=None, d_in=5, d_out=500, nhidden=1000)
model.load_model('../data/boltznet/ep3k/')
modelspt = BoltzNet(k=None, d_in=5, d_out=120, nhidden=500, log_it=False)
modelspt.load_model('../data/sptnet/ep3k/')


#specify fake observed data

isim = 100
kmax = 0.15
args = {"kmin": 0.001, "kmax":kmax, "offset_amp":0, "ampnorm":False}
args = sbitools.Objectify(args)
kcut, features, params = loader_pk.loader(args, return_k=True)

data = features[isim].copy()
kdata = np.load('../data/kmatter_quijote.npy')[1:data.size+1]
cov = np.load('../data/cov_disconnected_cs1_quijote.npy')[1:data.size+1]


# def log_prob(x, data, kdata, cov):
#     cp, cs = x[:-1], x[-1]
#     try:
#         pklin = model.interp(cp)
#         pred = pkmatter(kdata, pklin, cs)
#         chisq = (pred - data)**2/cov
#         lp = -0.5 * np.sum(chisq)
#     except Exception as e:
#         print(e)
#         lp = -np.inf
#     return lp

# priors = [torch.distributions.Uniform(model.lower_bounds[i], model.upper_bounds[i]) for i in range(5)]
# priors = priors + [torch.distributions.Uniform(-5, 5)]
priors = (list(model.lower_bounds) + [-5], list(model.upper_bounds) + [5])

def log_prob(x, data, kdata, cov):
    cp, cs = x[:-1], x[-1]
    try:
        pklin = model.interp(cp)
        pct = pkmatter.pct(pklin)(kdata, cs)
        p1loop = modelspt.interp(cp)(kdata)
        pred = p1loop + pct
        chisq = (pred - data)**2/cov
        lk = -0.5 * np.sum(chisq)
        #prior
        lb = (x < priors[0]).sum()
        ub = (x > priors[1]).sum()
        if lb+ub:
            lpr = -np.inf
        else:
            lpr = 0 
        # logprob
        lp = lpr + lk
        if np.isnan(lp):
            raise ValueError("log prob is NaN")
    except Exception as e:
        print(e)
        lp = -np.inf
    return lp



# Zeus it
nsteps, nwalkers, ndim = 10000, 20, 6

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

# Run it for zeus
print('zeus it')
sampler = zeus.EnsembleSampler(nwalkers, ndim, log_prob, args=[data, kdata, cov])
sampler.run_mcmc(x0, nsteps)
chain = sampler.get_chain(flat=False)
np.save(f"zeus_chain_LH{isim}_kmax{kmax}", chain)

# Run it for emcee
print('emcee it')
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[data, kdata, cov])
sampler.run_mcmc(x0, nsteps)
chain = sampler.get_chain(flat=False)
np.save(f"emcee_chain_LH{isim}_kmax{kmax}", chain)
