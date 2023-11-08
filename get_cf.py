import os
import numpy as np
import time
import torch
import tensorflow.compat.v1 as tf
from morpheus.explainers.cfproto import CounterfactualProto
from morpheus.morpheus.utils.models import TissueClassifier

EPSILON = 1e-20

def generate_cf(X_orig, 
                y_orig, 
                model_path, 
                channel_to_perturb, 
                data_dict, 
                model_arch=None,
                X_train_path=None, 
                optimization_params=dict(), 
                SAVE=False, 
                save_dir=None, 
                patch_id=None, 
                threshold=0.5):
    """
    Generate counterfactuals for a given patch.
    Parameters
    ----------
    X_orig : numpy array
        Original patch to be explained.
    y_orig : numpy array
        Original label of the patch.
    model_path : str
        Path to the model.
    channel_to_perturb : list
        List of channels to perturb.
    data_dict : dict
        Dictionary containing the mean and standard deviation of each channel.
    model_arch : str
        Model architecture. Either 'mlp' or 'cnn'.
    X_train_path : str
        Path to the training data.
    optimization_params : dict
        Dictionary containing the parameters for the optimization.
    SAVE : bool
        Whether to save the counterfactual.
    save_dir : str
        Directory where output will be saved.
    patch_id : str
        ID of the patch.
    threshold : float
        Threshold for the prediction probability.
    Returns
    -------
    None
    """
    # Obtain data features
    channel, sigma, mu = np.array(data_dict['channel']), data_dict['stdev'], data_dict['mean']
    H, _, C = X_orig.shape
    X_orig = (X_orig - mu)/sigma
    X_mean = np.mean(X_orig,axis=(0,1))
    
    if model_arch == 'mlp':
        X_orig = X_mean
        
    # initialize model
    tf.disable_eager_execution()

    print('Loading model')
    model = TissueClassifier.load_from_checkpoint(model_path, 
                                                    in_channels=C, 
                                                    img_size=H,
                                                    modelArch=model_arch)
    model.eval()
    
    # Adding init layer to model
    # make sure X_orig is unnormalized when passed into add_init_layer
    unnormed_mean = X_mean*sigma+mu
    if model_arch == 'mlp':
        def altered_model(x): 
            return torch.nn.functional.softmax(model(torch.from_numpy(x).float()),dim=1)
        def input_transform(x): return x
    else:
        print('Modifying model')
        unnormed_patch = X_orig[None,:]*sigma+mu
        def init_fun(y):
            return alter_image(y, unnormed_patch, mu, sigma, unnormed_mean)
        altered_model, input_transform = add_init_layer(init_fun,model)

    # Set range of each channel to perturb
    channel_to_perturb = [name for name in channel if name in channel_to_perturb] # IMPORTANT: keep channel in appropriate order
    isPerturbed = np.array([True if name in channel_to_perturb 
                            else False for name in channel])
    feature_range = (np.maximum(-mu/sigma, np.ones(C)*-4),np.ones(C)*4)
    feature_range[0][~isPerturbed] = X_mean[~isPerturbed]-EPSILON
    feature_range[1][~isPerturbed] = X_mean[~isPerturbed]+EPSILON

    # define predict function
    predict_fn = lambda x: altered_model(x).detach().numpy()
    
    print('check instance')
    # Terminate if model incorrectly classifies patch as the target class
    target_class = optimization_params.pop('target_class')
    print(predict_fn(X_mean[None,]))
    pred = predict_fn(X_mean[None,])[0,1] > threshold
    if pred == target_class:
        print('instance already classified as target class, no counterfactual needed')
        return 
    
    shape = (1,) + X_orig.shape
    cf = CounterfactualProto(predict_fn, input_transform, shape, 
                             feature_range=feature_range, 
                             **optimization_params)

    print('Building kdtree')
    if not os.path.exists(optimization_params['trustscore']):
        X_train = np.load(X_train_path)
        X_train = (X_train - mu)/sigma
        # generate predicted label to build tree
        if model_arch == 'mlp':
            X_t = torch.from_numpy(np.mean(X_train,axis=(1,2))).float()
        else:
            X_t = torch.permute(torch.from_numpy(X_train), (0,3,1,2)).float()
        preds = np.argmax(model(X_t).detach().numpy(), axis=1)

        X_train = np.mean(X_train,axis=(1,2))
        cf.fit(X_train, preds)
    else:
        cf.fit()
    
    print('kdtree built!')
    t1 = time.time()
    explanation = cf.explain(X=X_mean[None,:], Y=y_orig[None,:], 
                             target_class=[target_class], verbose=False)
    t2 = time.time()
    print(f'explain step time elapsed = {t2 - t1}')

    if explanation.cf is not None:
        cf_prob = explanation.cf['proba'][0]
        cf = explanation.cf['X'][0]
        
        print(f'compute probability: {predict_fn(cf[None,])}')
        cf = input_transform(cf[None,])
        if model_arch == 'mlp':
            pred_proba = altered_model(cf)
        else:
            pred_proba = model(cf)
        if model_arch != 'mlp':
            cf = torch.permute(cf, (0,2,3,1)).numpy()

        print(f"cf probability: {cf_prob}")
        print(f"compute probability: {pred_proba}")
        X_perturbed = mean_skipfew(np.mean, cf*sigma+mu, preserveAxis=cf.ndim-1)
        X_orig = X_mean*sigma+mu
        cf_delta = (X_perturbed  - X_orig) / X_orig * 100 
        cf_perturbed = dict(zip(channel[isPerturbed],cf_delta[isPerturbed]))
        print(f"cf perturbed: {cf_perturbed}")

        if SAVE:
            if patch_id is None:
                raise ValueError("Value of file_id must be passed if SAVE is true.")
            if save_dir is None:
                raise ValueError("Please provide directory where output will be saved.")
            else:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            savedFile = os.path.join(save_dir, "patch_{}.npz".format(patch_id))
            np.savez(savedFile, explanation=explanation,
                                cf_perturbed=cf_perturbed,
                                channel_to_perturb=channel_to_perturb)
    return

def alter_image(y, unnormed_patch, mu, sigma, unnormed_mean):
    unnormed_y = y*sigma + mu
    new_patch = unnormed_patch*((unnormed_y/unnormed_mean)[:,None,None,:])
    return (new_patch-mu)/sigma
    
def add_init_layer(init_fun, model):
    class input_fun(torch.nn.Module):
        def forward(self, input):
            return torch.permute(torch.from_numpy(init_fun(input)), (0,3,1,2)).float()
    input_transform = input_fun()
    completeModel = torch.nn.Sequential(input_transform, model)
    return completeModel, input_transform

def mean_skipfew(ufunc, foo, preserveAxis=None):
    r = np.arange(foo.ndim)   
    if preserveAxis is not None:
        preserveAxis = tuple(np.delete(r, preserveAxis))
    return ufunc(foo, axis=preserveAxis)