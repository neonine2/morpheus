import os, sys
import numpy as np
import json
import pickle

def run_one(config):
    from run_one import optimize_one ## This will be executed on each compute instance, hence needs to be imported within the function
    optimize_one(_config=config)

# get which job we need to do. The int is necessary because environment variables are stored as strings
run_id = 46400

with open(os.path.join(sys.path[0],"queue.json"), "r") as f:
    simulation_parameter_list = json.load(f)
    config = simulation_parameter_list[0] # first entry are general parameters
    instance = simulation_parameter_list[run_id+1] # starting from second entry

# Load inputs into correct format
with open(config['data_dir']+"/data_info.pkl", 'rb') as f:
    data_dict = pickle.load(f)
instance['data_dict'] = data_dict
instance['X_orig'] = np.load(instance['X_orig_path'])
instance.pop('X_orig_path')
instance['y_orig'] = np.array(instance['y_orig'])
instance['model_path'] = config['model_path']
instance['channel_to_perturb'] = np.array(config['channel_to_perturb'])
instance['optimization_param'] = config['optimization_param']
instance['X_train_path'] = config['X_train_path']
instance['save_dir'] = config['save_dir']
instance['model_arch'] = config['model_arch']
instance['threshold'] = config['threshold']

# Now run it
print(f'running number {run_id}')
run_one(instance)