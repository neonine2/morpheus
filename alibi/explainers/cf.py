import os
import _pickle as pickle
import numpy as np
import tensorflow.compat.v1 as tf
import time
from tensorflow.keras import models
from alibi.explainers.cfprotoOG import CounterfactualProto

def load_object(filename):
    with open(filename, 'rb') as outp: 
        return pickle.load(outp)
    
def add_init_layer(patch, mu, sigma, orig_mean, model, ml_framework='tf'):
    # orig_mean = np.float32(orig_mean)
    if len(patch.shape) > 3:
        _,H,W,C = patch.shape
    else:
        H,W,C = 1,1,patch.shape[-1]

    new_weights = np.zeros((C*H*W,C))
    for c_ind in range(C):
        start = (H*W)*c_ind
        fin = (H*W)*(c_ind+1)
        new_weights[start:fin,c_ind] = 1
    def alter_image(y):
            return ((tf.math.minimum(1.0, (y*sigma + mu)/(orig_mean*sigma + mu))
                 *((patch*sigma + mu)+
                   tf.math.maximum(0.0, (y*sigma + mu)-(orig_mean*sigma + mu))))
                   -mu)/sigma
    newInput = tf.keras.Input(shape=(C,))
    x = tf.keras.layers.Lambda(alter_image)(newInput)
    input_transform = tf.keras.Model(newInput, x)
    if ml_framework != 'pytorch':
        newOutputs = model(x)
        completeModel = tf.keras.Model(newInput, newOutputs)
    else:
        fc1 = torch.nn.Linear(C, C*H*W, bias=True)
        fc1.weight.data = torch.from_numpy(new_weights).float()
        fc1.bias.data = torch.from_numpy(patch-minVal).float().permute(2,0,1).flatten()
        activation = torch.nn.ReLU()
        fc2 = torch.nn.Linear(C*H*W, C*H*W, bias=True)
        fc2.weight.data = torch.nn.init.eye_(torch.empty(C*H*W, C*H*W))
        fc2.bias.data = torch.from_numpy(np.repeat(minVal,H*W)).float()
        if H > 1 or W > 1:
            reshape_layer = torch.nn.Unflatten(1, (C, H, W))
            completeModel = torch.nn.Sequential(fc1, activation, fc2, reshape_layer, model)
        completeModel = torch.nn.Sequential(fc1, activation, fc2, model)
    return completeModel, input_transform

def mean_skipfew(ufunc, foo, preserveAxis=None):
    r = np.arange(foo.ndim)   
    if preserveAxis is not None:
        preserveAxis = tuple(np.delete(r, preserveAxis))
    return ufunc(foo, axis=preserveAxis)

def load_trained_model(model_path, ml_framework=None):
    #load model
    if ml_framework == 'pytorch':
        model = TissueClassifier.load_from_checkpoint(model_path, in_channels=len(data_dict['channel']), img_size=X.shape,
                                                        modelArch=model_arch).float()
        model.eval()
    else:
        model = models.load_model(model_path, compile=False)
    return model

def generate_cf(X_orig, y_orig, model_path, channel_to_perturb, data_dict, optimization_params=dict(), 
                SAVE=False, save_dir=None, patch_id=None):
    
    # For MLP, make sure to average across each channel, # for CNNs, make sure to adjust input shape
    model_name = model_path.split(os.sep)[-1]
    model_arch = model_name.split('.')[0].lower()
    model_ext = model_name.split('.')[1].lower()
    if model_ext == 'h5':
        ml_framework = 'tensorflow'
    else:
        ml_framework = 'pytorch'

    # Obtain data features
    channel = np.array(data_dict['channel'])
    sigma = data_dict['stdev']
    mu = data_dict['mean']
    C = X_orig.shape[-1]

    # Change image shape
    input_shape = (32,32)
    resize_fn = tf.keras.layers.Resizing(input_shape[0], input_shape[1], interpolation='nearest')
    X_orig = resize_fn(X_orig).numpy()
    X_train = resize_fn(data_dict['X_train']).numpy()
    
    if model_arch == 'mlp':
        X_orig = np.mean(X_orig,axis=(0,1))
        X_train = np.mean(X_train,axis=(1,2))
        
    X_orig = (X_orig - mu) / sigma
    if X_orig.ndim > 2:
        X_mean = np.mean(X_orig,axis=(0,1))
    
    # initialize model
    tf.disable_eager_execution()
    model = load_trained_model(model_path, ml_framework)

    # only procedure if model correctly classifies patch as not having T cells
    pred = np.argmax(model.predict(X_orig[None,]))
    if pred == 1:
        print('instance already positively classified, no counterfactual needed')
        return 
    
    # Adding init layer to model
    # make sure X_orig is unnormalized when passed into add_init_layer
    altered_model, input_transform = add_init_layer(X_orig[None,:],  mu, sigma, 
                                                        X_mean, model, ml_framework)

    # Set range of each channel to perturb
    isPerturbed = np.array([True if name in channel_to_perturb else False for name in channel])
    feature_range = ((0*np.ones(C) - mu)/sigma,(1e2*np.ones(C) - mu)/sigma)
    feature_range[0][~isPerturbed] = X_mean[~isPerturbed]-1e-20
    feature_range[1][~isPerturbed] = X_mean[~isPerturbed]+1e-20
    # maxVal = mean_skipfew(np.max, X_orig, preserveAxis=X_orig.ndim-1)

    # define predict function
    if ml_framework == 'pytorch':
        predict_fn = lambda x: torch.nn.functional.softmax(altered_model(torch.from_numpy(x)
                                                                 .float()),dim=0).detach().numpy()
    else:
        # predict_fn = lambda x : altered_model.predict(x) 
        predict_fn = altered_model
   
    cf = CounterfactualProto(predict_fn, input_transform, (1,) + X_orig.shape, 
                             feature_range=feature_range, **optimization_params)
        # generate predicted label to build tree
    print('shape : ', X_train.shape)
    if optimization_params['trustscore'] is None:
        if ml_framework == 'pytorch':
            preds = np.argmax(model(torch.from_numpy(X_train).permute(0, 3, 1, 2).float())
                            .detach().numpy(), axis=1)
        else:
            preds = model.predict(X_train)
            preds = np.argmax(preds,axis=1)
        cf.fit(X_train, preds)
    else:
        cf.fit(train_data=np.array([]),preds=np.array([]))
    
    t1 = time.time()
    explanation = cf.explain(X=X_mean[None,:], Y=y_orig[None,:], target_class=[1], verbose=True)
    t2 = time.time()
    print(f'explain step time elapsed = {t2 - t1}')

    cf_delta = None
    cf_prob = None
    cf_perturbed = None
    if explanation.cf is not None:
        cf_prob = explanation.cf['proba'][0]
        cf = explanation.cf['X'][0]
        # cf_all = np.maximum(0,cf + X_orig - minVal - X_mean_normed) + minVal
        if X_orig.ndim>2:
            cf =((np.minimum(1.0, (cf*sigma + mu)/(X_mean*sigma + mu))*
                  ((X_orig*sigma + mu)+
                   np.maximum(0.0, (cf*sigma + mu)-(X_mean*sigma + mu))))
                   -mu)/sigma
        print(f"cf probability: {cf_prob}")
        print(f"compute probability:{model.predict(cf[None,])}")
        X_perturbed = mean_skipfew(np.mean, cf*sigma+mu, preserveAxis=cf.ndim-1)
        X_orig = X_mean*sigma+mu
        cf_delta = (X_perturbed  - X_orig) / X_orig * 100
        print(f"cf unperturbed: {np.max(np.abs(cf_delta[~isPerturbed]))}")
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
                            cf_delta=cf_delta,
                            cf_perturbed=cf_perturbed,
                            channel_to_perturb=channel_to_perturb)
    return cf_delta, cf_prob, cf_perturbed