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
# datapath = f'/mnt/ceph/users/cmodi/contrastive/data/{cfgd.simulation}/{cfgd.finder}/z{int(cfgd.z*10):02d}-N{int(cfgd.nbar/1e-4):04d}/{cfgd.hodmodel}/'
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



##
#############
# def analysis(features, params):
#     data = sbitools.test_train_split(features, params, train_size_frac=cfgd.train_frac)

#     ### Standaradize
#     scaler = None
#     if cfgd.standardize:
#         try:
#             scaler = sbitools.load_scaler(analysis_path)
#             data.trainx = sbitools.standardize(data.trainx, scaler=scaler, log_transform=cfgd.logit)[0]
#             data.testx = sbitools.standardize(data.testx, scaler=scaler, log_transform=cfgd.logit)[0]
#         except Exception as e:
#             print("EXCEPTION occured in loading scaler", e)
#             print("Fitting for the scaler and saving it")
#             data.trainx, data.testx, scaler = sbitools.standardize(data.trainx, secondary=data.testx, log_transform=cfgd.logit)
#             with open(analysis_path + "scaler.pkl", "wb") as handle:
#                 pickle.dump(scaler, handle)

#     ### SBI
#     prior = sbitools.sbi_prior(params.reshape(-1, params.shape[-1]), offset=0.2)
#     print("trainx and trainy shape : ", data.trainx.shape, data.trainy.shape)
#     posterior = sbitools.sbi(data.trainx, data.trainy, prior, \
#                                   model=cfgm.model, nlayers=cfgm.ntransforms, \
#                                   nhidden=cfgm.nhidden, batch_size=cfgm.batch, savepath=model_path, retrain=bool(cfgm.retrain))

#     return data, posterior

