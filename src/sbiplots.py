import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
#from timeout import timeout
import sys, os, errno
#warnings.filterwarnings("error")

cosmonames = r'$\Omega_m$,$\Omega_b$,$h$,$n_s$,$\sigma_8$,$c_s$'.split(",")

###
def plot_ranks_histogram(ranks, nbins=10, npars=5, titles=None, savepath=None, figure=None, suffix=""):

    ncounts = ranks.shape[0]/nbins
    if titles is None: titles = [""]*npars

    if figure is None: 
        fig, ax = plt.subplots(1, npars, figsize=(npars*3, 4))
    else:
        fig, ax = figure
    
    for i in range(npars):
        ax[i].hist(np.array(ranks)[:, i], bins=nbins)
        ax[i].set_title(titles[i])
        ax[0].set_ylabel('counts')

    for axis in ax:
        axis.set_xlim(0, ranks.max())
        axis.set_xlabel('rank')
        axis.grid(visible=True)
        axis.axhline(ncounts, color='k')
        axis.axhline(ncounts - ncounts**0.5, color='k', ls="--")
        axis.axhline(ncounts + ncounts**0.5, color='k', ls="--")
    plt.tight_layout()
    if savepath is not None:
        suptitle = savepath.split('/')[-2]
        plt.suptitle(suptitle)
        plt.tight_layout()
        plt.savefig(savepath + 'rankplot%s.png'%suffix)
    return fig, ax


###
def plot_coverage(ranks, npars=5, titles=None, savepath=None, figure=None, suffix="", label="", plotscatter=True):

    ncounts = ranks.shape[0]
    if titles is None: titles = [""]*npars
    unicov = [np.sort(np.random.uniform(0, 1, ncounts)) for j in range(20)]

    if figure is None: 
        fig, ax = plt.subplots(1, npars, figsize=(npars*3, 4))
    else:
        fig, ax = figure
    
    for i in range(npars):
        xr = np.sort(ranks[:, i])
        xr = xr/xr[-1]
        cdf = np.arange(xr.size)/xr.size
        if plotscatter: 
            for j in range(len(unicov)): ax[i].plot(unicov[j], cdf, lw=1, color='gray', alpha=0.2)
        ax[i].plot(xr, cdf, lw=2, label=label)
        ax[i].set_title(titles[i])
        ax[0].set_ylabel('CDF')

    for axis in ax:
        #axis.set_xlabel('rank')
        axis.grid(visible=True)

    plt.tight_layout()
    if savepath is not None:
        suptitle = savepath.split('/')[-2]
        plt.suptitle(suptitle)
        plt.tight_layout()
        plt.savefig(savepath + 'coverage%s.png'%suffix)
    return fig, ax


###
def plot_posterior(x, y, posterior, nsamples=1000, titles=None, savename="", ndim=None):
    
    posterior_samples = posterior.sample((nsamples,), x=torch.from_numpy(x.astype('float32'))).detach().numpy()
    mu, std = posterior_samples.mean(axis=0), posterior_samples.std(axis=0)
    p = y
    if ndim is None: ndim = len(y)
    #
    fig, axar = plt.subplots(ndim, ndim, figsize=(3*ndim, 3*ndim), sharex='col')
    nbins = 'auto'
    for i in range(ndim):
        axar[i, i].hist(np.array(posterior_samples[:, i]), density=True, bins=nbins);
        axar[i, i].axvline(mu[i], color='r');
        axar[i, i].axvline(mu[i] + std[i], color='r', ls="--");
        axar[i, i].axvline(mu[i] - std[i], color='r', ls="--");
        axar[i, i].axvline(p[i], color='k')
        for j in range(0, i):
            axar[i, j].plot(posterior_samples[:, j], posterior_samples[:, i], '.')
        for j in range(i+1, ndim):
            axar[i, j].set_axis_off()
            
    if titles is None:
        titles = cosmonames
        if ndim > len(titles): 
            titles = titles + ["a_%d"%i for i in range(ndim - len(titles))]
    for i in range(ndim): 
        axar[i, i].set_title(titles[i])

    if savename != "": 
        plt.savefig(savename)
        plt.close()
    return fig, axar


###
#@timeout(100, os.strerror(errno.ETIMEDOUT))
def _sample_for_rank(posterior, x, nsamples):
    posterior_samples = posterior.sample((nsamples,),
                                             x=torch.from_numpy(x.astype('float32')), 
                                             show_progress_bars=False).detach().numpy()
    return posterior_samples

#@timeout(5000, os.strerror(errno.ETIMEDOUT))
def get_ranks(x, y, posterior, test_frac=1.0, nsamples=500, ndim=None):
    if ndim is None:
        ndim = y.shape[1]
        ndim = min(ndim, posterior.sample((1,),  x=torch.from_numpy(x[0].astype('float32')), 
                                             show_progress_bars=False).detach().numpy().shape[1])

    ranks = []
    mus, stds = [], []
    trues = []
    for ii in range(x.shape[0]):
        if ii%1000 == 0: print("Test iteration : ",ii)
        if np.random.uniform() > test_frac: continue

        # try:
        #     posterior_samples = posterior.sample((nsamples,),
        #                                      x=torch.from_numpy(x[ii].astype('float32')), 
        #                                      show_progress_bars=False).detach().numpy()
        # except Warning as w:
        #     #except :
        #     print("WARNING\n", w)
        #     continue
        try:
            posterior_samples = _sample_for_rank(posterior, x[ii], nsamples)
        except Exception as e:
            print(f"Exception at index {ii}\n", e)
            continue
        mu, std = posterior_samples.mean(axis=0)[:ndim], posterior_samples.std(axis=0)[:ndim]
        rank = [(posterior_samples[:, i] < y[ii, i]).sum() for i in range(ndim)]
        mus.append(mu)
        stds.append(std)
        ranks.append(rank)
        trues.append(y[ii][:ndim])
    mus, stds, ranks = np.array(mus), np.array(stds), np.array(ranks)
    trues = np.array(trues)
    return trues, mus, stds, ranks


###
def plot_predictions(trues, mus, stds, npars=5, titles=None, savepath=None, suffix=""):

    #plot predictions
    if npars > 5: fig, ax = plt.subplots(npars//5, 5, figsize=(15, 4*npars//5))
    else: fig, ax = plt.subplots(1, 5, figsize=(15, 4))
    for j in range(min(npars, len(ax.flatten()))):
        if stds is not None: ax.flatten()[j].errorbar(trues[:, j], mus[:, j], stds[:, j], fmt="none", elinewidth=0.5, alpha=0.5)
        else: ax.flatten()[j].plot(trues[:, j], mus[:, j], ".", alpha=0.5)
        #if j == 0 : ax.flatten()[0].set_ylabel(lbls[iss], fontsize=12)
        ax.flatten()[j].plot(trues[:, j], trues[:, j], 'k.', ms=0.2, lw=0.5)
        ax.flatten()[j].grid(which='both', lw=0.5)
        ax.flatten()[j].set_title(titles[j], fontsize=12)

    plt.tight_layout()
    if savepath is not None:
        suptitle = savepath.split('/')[-2]
        plt.suptitle(suptitle)
        plt.tight_layout()
        plt.savefig(savepath + 'predictions%s.png'%suffix)
    return fig, ax


###
def test_diagnostics(x, y, posterior, nsamples=500, titles=None, savepath="", test_frac=1.0, suffix=""):

    ndim = y.shape[1]
    ndim = min(ndim, posterior.sample((1,),  x=torch.from_numpy(x[0].astype('float32')), 
                                             show_progress_bars=False).detach().numpy().shape[1])
    if titles is None: titles = []*ndim

    trues, mus, stds, ranks = get_ranks(x, y, posterior, test_frac, nsamples=nsamples)

    #plot ranks and coverage
    _ = plot_ranks_histogram(ranks, titles=titles, savepath=savepath, suffix=suffix)
    plt.close()
    _ = plot_coverage(ranks, titles=titles, savepath=savepath, suffix=suffix)
    plt.close()
    _ = plot_predictions(trues, mus, stds, npars=ndim, titles=titles, savepath=savepath, suffix=suffix)
    plt.close()

    



###
def test_fiducial(x, y, posterior, nsamples=500, rankplot=True, titles=None, savepath="", test_frac=1.0, suffix=""):

    ndim = y.shape[1]
    ndim = min(ndim, posterior.sample((1,),  x=torch.from_numpy(x[0].astype('float32')), 
                                             show_progress_bars=False).detach().numpy().shape[1])
    if titles is None: titles = []*ndim

    trues, mus, stds, ranks = get_ranks(x, y, posterior, test_frac, nsamples=nsamples)

    #plot ranks
    _ = plot_ranks_histogram(ranks, titles=titles, savepath=savepath, suffix=suffix)
    _ = plot_coverage(ranks, titles=titles, savepath=savepath, suffix=suffix)


    #plot predictions
    if ndim > 5: fig, ax = plt.subplots(ndim//5, 5, figsize=(15, 4*ndim//5))
    else: fig, ax = plt.subplots(1, 5, figsize=(15, 4))
    for j in range(min(ndim, len(ax.flatten()))):
        axis = ax.flatten()[j]
        axis.errorbar(np.arange(mus.shape[0]), mus[:, j], stds[:, j], fmt="none", elinewidth=0.5, alpha=0.5)
        axis.axhline(trues[0, j], color='k')
        #axis.plot(y[:, j], y[:, j], 'k.', ms=0.2, lw=0.5)
        axis.grid(which='both', lw=0.5)
        axis.set_title(titles[j], fontsize=12)
    suptitle = savepath.split('/')[-2]
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(savepath + 'predictions%s.png'%suffix)
    


###
def mlp_diagnostics(x, y, model, titles=None, savepath="", test_frac=1.0, suffix=""):

    ndim = y.shape[1]
    nsamples = int(test_frac*x.shape[0])
    idx = np.random.randint(0, x.shape[0], nsamples)
    inp = torch.tensor(x[idx], dtype=torch.float32)
    output = model(inp).detach().numpy()
    target = y[idx]
    ndim = min(ndim, output.shape[-1])
    _ = plot_predictions(target, output, stds=None, npars=ndim, titles=titles, savepath=savepath, suffix=suffix)
    plt.close()

    

