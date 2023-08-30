import numpy as np
import sys, os
sys.path.append('../../src/')
import sbitools, sbiplots
import argparse
import pickle, json
import loader_hybrid_wavelets as loader
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

#
cfgd.analysis_path = loader.folder_path(cfgd_dict)
model_path = f'tmp/'
cfgm.model_path = cfgd.analysis_path + model_path
os.makedirs(cfgm.model_path, exist_ok=True)

print("\nWorking directory : ", cfgm.model_path)
os.system(f'cp {cfg_data} {cfgd.analysis_path}/{cfg_data}')  

#############
features, params = loader.loader(cfgd)
print("features and params shapes : ", features.shape, params.shape)
data, posterior, inference, summary = sbitools.analysis(cfgd, cfgm, features, params)

# ndiagnostics = 1
# print("Diagnostics for test dataset")
# for i in range(ndiagnostics):
#     fig, ax = sbiplots.plot_posterior(data.testx[i], data.testy[i], posterior, nsamples=500, savename=f'{cfgm.model_path}/corner{i}.png')


# print("Diagnostics for training dataset")
# for i in range(ndiagnostics):
#     fig, ax = sbiplots.plot_posterior(data.testx[i], data.testy[i], posterior, nsamples=500, savename=f'{cfgm.model_path}/corner-train{i}.png')
