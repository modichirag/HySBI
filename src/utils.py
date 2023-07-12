import numpy as np
import os
import torch
from torch import nn
from scipy.interpolate import InterpolatedUnivariateSpline
import numdifftools as ndiff


class BoltzNet(nn.Module):

    def __init__(self, d_in=5, d_out=500, nhidden=500, log_it=True, loc=0., scale=1.):
        super().__init__()
        self.in_layer = nn.Linear(d_in, nhidden)
        self.hidden_layers0 = nn.Linear(nhidden, nhidden)
        self.hidden_layers1 = nn.Linear(nhidden, nhidden)
        self.out_layer = nn.Linear(nhidden, d_out)
        self.log_it = log_it
        self.loc = loc
        self.scale = scale

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+'/net')
        np.save(path+'/loc', self.loc)
        np.save(path+'/scale', self.scale)

    def load_model(self, path):
        self.load_state_dict(torch.load(path+'/net'))
        self.loc = np.load(path+'/loc.npy')
        self.scale = np.load(path+'/scale.npy')


    def transform(self, x):
        if x.dtype == torch.float32:
            x = x.detach().numpy()
        if self.log_it: 
                x = np.log(x)
        x = (x-self.loc)/self.scale
        return x

    def inv_transform(self, x):
        if x.dtype == torch.float32:
            x = x.detach().numpy()
        x = x*self.scale + self.loc
        if self.log_it: 
            x = np.exp(x)
        return x

    def forward(self, x):

        x = self.in_layer(x)
        x = nn.ReLU()(x)
        x = self.hidden_layers0(x)
        x = nn.Tanh()(x)        
        x = self.hidden_layers1(x)
        x = nn.Tanh()(x)        
        x = self.out_layer(x)
        return x 

    def predict(self, x):

        y = self.forward(x)
        y = self.inv_transform(y)
        return y



def loginterp(x, y, yint = None, side = "both", lp = 1, rp = -1, err_order=4, verbose=True):
    """
    Interpolate (x,y) after extrapolating linearly on log-scale. 
    """

    if yint is None:
        yint = InterpolatedUnivariateSpline(x, y, k = 5)

    if side == "both":
        side = "lr"
        l = int(lp)
        r = int(rp)

    lneff = ndiff.Derivative(yint, order=err_order)(x[l])*x[l]/y[l]
    rneff = ndiff.Derivative(yint, order=err_order)(x[r])*x[r]/y[r]
    if verbose: print("log derivative on left and right edge : ", lneff, rneff)

    xl = np.logspace(-18, np.log10(x[l]), 10**6)
    xr = np.logspace(np.log10(x[r]), 10., 10**6)
    yl = y[l]*(xl/x[l])**lneff
    yr = y[r]*(xr/x[r])**rneff

    xint = x[l+1:r].copy()
    yint = y[l+1:r].copy()
    if side.find("l") > -1:
        xint = np.concatenate((xl, xint))
        yint = np.concatenate((yl, yint))
    if side.find("r") > -1:
        xint = np.concatenate((xint, xr))
        yint = np.concatenate((yint, yr))
    yint2 = InterpolatedUnivariateSpline(xint, yint, k = 5)

    return yint2
