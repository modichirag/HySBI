import numpy as np
#import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')
import sbitools
import loader_hybrid_wavelets as loader
import wandb
import yaml

## Parse arguments
config_data = sys.argv[1]
nmodels = int(sys.argv[2])
cfgd_dict = yaml.load(open(f'{config_data}'), Loader=yaml.Loader)
sweep_id = cfgd_dict['sweep']['id']
print(sweep_id)

cuts = cfgd_dict['datacuts']
args = {}
for i in cfgd_dict.keys():
    args.update(**cfgd_dict[i])
cfgd = sbitools.Objectify(**args)

#

cfgd.analysis_path = loader.folder_path(cfgd_dict)
cfgd.model_path = cfgd.analysis_path + f'/{sweep_id}/'
os.makedirs(cfgd.model_path, exist_ok=True)
print("\nWorking directory : ", cfgd.model_path)
os.system('cp {config_data} {cfgd.model_path}')
##


#############
features, params = loader.loader(cfgd)

#####
#############
def train_sweep(config=None):

    
    with wandb.init(config=config) as run:

        # Copy your config 
        cfgm = wandb.config
        cfgm = sbitools.Objectify(**cfgm)
        cfgm.retrain = True

        print("running for model name : ", run.name)
        cfgm.model_path = f"{cfgd.model_path}/{run.name}/"
        os.makedirs(cfgm.model_path, exist_ok=True)

        data, posterior, inference, summary = sbitools.analysis(cfgd, cfgm, features, params, verbose=False)

        # Make the loss and optimizer
        for i in range(len(summary['train_log_probs'])):
            metrics = {"train_log_probs": summary['train_log_probs'][i],
                   "validation_log_probs": summary['validation_log_probs'][i]}
            wandb.log(metrics)
        wandb.run.summary["best_validation_log_prob"] = summary['best_validation_log_prob']
        print(wandb.run.summary["best_validation_log_prob"])
        wandb.log({'output_directory': cfgm.model_path})
               


if __name__ == '__main__':

    print(f"run for {nmodels} models")
    wandb.agent(sweep_id=sweep_id, function=train_sweep, count=nmodels, project='hysbi')

