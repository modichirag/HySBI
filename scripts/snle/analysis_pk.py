import numpy as np
import sys, os
sys.path.append('../../src/')
import sbitools, sbiplots
import argparse
import pickle, json
import loader_pk as loader
import yaml


cfg_data = sys.argv[1]
cfg_model = sys.argv[2]
cfgd_dict = yaml.load(open(f'{cfg_data}'), Loader=yaml.Loader)
cfgm_dict = yaml.load(open(f'{cfg_model}'), Loader=yaml.Loader)['flow']
cuts = cfgd_dict['datacuts']
args = {}
for i in cfgd_dict.keys(): #hack to flatten nested dict
    args.update(**cfgd_dict[i])
# cfgd = sbitools.Objectify(**cfgd_dict)
cfgm = sbitools.Objectify(**cfgm_dict)
cfgd = sbitools.Objectify(**args)
np.random.seed(cfgd.seed)

print(cfgd_dict)

#
analysis_path = loader.folder_path(cfgd_dict)
folder = 'tmp/'
model_path = f'tmp/'

cfgd.analysis_path = analysis_path + folder
cfgm.model_path = cfgd.analysis_path + model_path
os.makedirs(cfgm.model_path, exist_ok=True)

print("\nWorking directory : ", cfgm.model_path)
os.system(f'cp {cfg_data} {cfgd.analysis_path}/{cfg_data}')  

#############
kdata, features, params = loader.loader(cfgd, return_k=True)
print("features and params shapes : ", features.shape, params.shape)
data, posterior, inference, summary = sbitools.analysis(cfgd, cfgm, features, params)

ndiagnostics = 5
print("Diagnostics for test dataset")
for i in range(ndiagnostics):
    fig, ax = sbiplots.plot_posterior(data.testx[i], data.testy[i], posterior, savename=f'{cfgm.model_path}/corner{i}.png')


print("Diagnostics for training dataset")
for i in range(ndiagnostics):
    fig, ax = sbiplots.plot_posterior(data.testx[i], data.testy[i], posterior, savename=f'{cfgm.model_path}/corner-train{i}.png')
