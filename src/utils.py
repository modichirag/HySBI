import numpy as np
import os
import torch
from torch import nn
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import numdifftools as ndiff
import json


class BoltzNet(nn.Module):

    def __init__(self, k, d_in=5, d_out=500, nhidden=1000, log_it=True, loc=0., scale=1.):
        super().__init__()
        self.kvals = k
        self.in_layer = nn.Linear(d_in, nhidden)
        self.hidden_layers0 = nn.Linear(nhidden, nhidden)
        self.hidden_layers1 = nn.Linear(nhidden, nhidden)
        self.out_layer = nn.Linear(nhidden, d_out)
        self.log_it = log_it
        self.loc = loc
        self.scale = scale
        self.lower_bounds = np.array([0.1 , 0.03, 0.5, 0.8, 0.6 ])
        self.upper_bounds =  np.array([0.5 , 0.07, 0.9 , 1.2 , 1.0 ])
        self.input_params = {'d_in': d_in, 
                            'd_out': d_out, 
                            'nhidden': nhidden,
                            'log_it': log_it
                            }

    def check_bounds(self, x):
        inbounds  = (x >= self.lower_bounds) & (x <= self.upper_bounds)
        for ip, p in enumerate(inbounds):
            if not p: print(f"Parameter {ip} is not within prior range")

        if inbounds.sum() == len(inbounds): 
            return True
        else:
            return False



    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path+'/net')
        np.save(path+'/loc', self.loc)
        np.save(path+'/scale', self.scale)
        np.save(path+'/k', self.kvals)
        with open(path+'/params.json', 'w', encoding='utf-8') as f:
            json.dump(self.input_params, f, ensure_ascii=False, indent=4)


    def load_model(self, path):
        self.load_state_dict(torch.load(path+'/net'))
        self.loc = np.load(path+'/loc.npy')
        self.scale = np.load(path+'/scale.npy')
        self.kvals = np.load(path+'/k.npy')


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
        if type(x) == np.ndarray: 
            x = torch.from_numpy(x.astype(np.float32))
        y = self.forward(x)
        y = self.inv_transform(y)
        return y

    def interp(self, x):
        if type(x) == np.ndarray: 
            x = torch.from_numpy(x.astype(np.float32))
        y = self.forward(x)
        y = self.inv_transform(y)
        if len(x.shape) == 1:
            pkl = InterpolatedUnivariateSpline(self.kvals, y, ext='zeros')
        else:
            pkl = [InterpolatedUnivariateSpline(self.kvals, y_i, ext='zeros') for y_i in y]
        return pkl



def loginterp(x, y, yint = None, side = "both", lp = 1, rp = -1, err_order=4, verbose=True, fitline=False):
    """
    Interpolate (x,y) after extrapolating linearly on log-scale. 
    """

    if yint is None:
        yint = InterpolatedUnivariateSpline(x, y, k = 5)

    if side == "both":
        side = "lr"
        l = int(lp)
        r = int(rp)

    # Extra smoothing the edges by fitting a straight line
    if fitline:
        line = lambda x, m, c : m*x + c
        m, c = curve_fit(line, x[:2*lp], y[:2*lp], )[0]
        y[:2*lp] = line(x[:2*lp], m, c)
        m, c = curve_fit(line, x[2*rp:], y[2*rp:], )[0]
        y[2*rp:] = line(x[2*rp:], m, c)
    
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
