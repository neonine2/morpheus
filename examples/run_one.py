import sys
sys.path.append('/home/zwang2/morpheus')

from get_cf import generate_cf

def optimize_one(_config):
    generate_cf(X_orig = _config["X_orig"], 
                y_orig = _config["y_orig"],
                model_path = _config["model_path"],
                channel_to_perturb = _config["channel_to_perturb"], 
                data_dict = _config["data_dict"],
                optimization_params = _config['optimization_param'],
                X_train_path = _config['X_train_path'],
                save_dir = _config['save_dir'],
                patch_id = _config['patch_id'],
                model_arch = _config['model_arch'],
                threshold = _config['threshold'],
                SAVE = True)

if __name__ == '__main__':
    pass