import os
import numpy as np
import torch
from counterfactuals.cfproto import CounterfactualProto
from models.ModelClass import TissueClassifier

EPSILON = torch.tensor(1e-20, dtype=torch.float32)


def generate_cf(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    model_path: str,
    channel_to_perturb: list,
    data_dict,
    model_arch=None,
    X_train_path=None,
    optimization_params=dict(),
    SAVE=False,
    save_dir=None,
    patch_id=None,
    threshold=0.5,
):
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
    channel, sigma, mu = (
        np.array(data_dict["channel"]),
        torch.from_numpy(data_dict["stdev"]).float(),
        torch.from_numpy(data_dict["mean"]).float(),
    )
    H, _, C = X_orig.shape
    X_orig = (torch.from_numpy(X_orig).float() - mu) / sigma
    y_orig = torch.from_numpy(y_orig).long()
    X_mean = torch.mean(X_orig, dim=(0, 1))

    if model_arch == "mlp":
        X_orig = X_mean

    print("Loading model")
    model = TissueClassifier.load_from_checkpoint(
        model_path, in_channels=C, img_size=H, modelArch=model_arch
    )
    model.eval()

    # Adding init layer to model
    # make sure X_orig is unnormalized when passed into add_init_layer
    unnormed_mean = X_mean * sigma + mu
    if model_arch == "mlp":

        def altered_model(x):
            return torch.nn.functional.softmax(model(x), dim=1)

        def input_transform(x):
            return x

    else:
        print("Modifying model")
        unnormed_patch = X_orig[None, :] * sigma + mu

        def init_fun(y):
            return alter_image(y, unnormed_patch, mu, sigma, unnormed_mean)

        altered_model, input_transform = add_init_layer(init_fun, model)

    # Set range of each channel to perturb
    channel_to_perturb = [
        name for name in channel if name in channel_to_perturb
    ]  # IMPORTANT: keep channel in appropriate order
    isPerturbed = np.array(
        [True if name in channel_to_perturb else False for name in channel]
    )
    feature_range = (torch.maximum(-mu / sigma, torch.ones(C) * -4), torch.ones(C) * 4)
    feature_range[0][~isPerturbed] = X_mean[~isPerturbed] - EPSILON  # type: ignore
    feature_range[1][~isPerturbed] = X_mean[~isPerturbed] + EPSILON  # type: ignore

    # define predict function
    predict_fn = lambda x: altered_model(x)

    print("check instance")
    # Terminate if model incorrectly classifies patch as the target class
    target_class = optimization_params.pop("target_class")
    orig_proba = predict_fn(X_mean[None,])
    print("initial probability: ", orig_proba)
    pred = orig_proba[0, 1] > threshold
    if pred == target_class:
        print("instance already classified as target class, no counterfactual needed")
        return

    # define counterfactual object
    print("defining counterfactual object")
    shape = (1,) + X_orig.shape
    cf = CounterfactualProto(
        predict_fn,
        input_transform,
        shape,
        feature_range=feature_range,
        **optimization_params,
    )

    print("Building kdtree")
    if not os.path.exists(optimization_params["trustscore"]):
        X_train = np.load(X_train_path)
        X_train = (X_train - mu) / sigma
        # generate predicted label to build tree
        if model_arch == "mlp":
            X_t = torch.from_numpy(np.mean(X_train, axis=(1, 2))).float()
        else:
            X_t = torch.permute(torch.from_numpy(X_train), (0, 3, 1, 2)).float()
        preds = np.argmax(model(X_t).detach().numpy(), axis=1)

        X_train = torch.mean(X_train, dim=(1, 2))
        cf.fit(X_train, preds)
    else:
        cf.fit()

    print("kdtree built!")
    explanation = cf.explain(
        X=X_mean[None, :], Y=y_orig[None, :], target_class=[target_class]
    )

    if explanation.cf is not None:
        cf_prob = explanation.cf["proba"][0]
        cf = explanation.cf["X"][0]

        # manually compute probability of cf
        cf = input_transform(torch.from_numpy(cf[None,]))
        if model_arch == "mlp":
            pred_proba = altered_model(cf)
        else:
            pred_proba = model(cf)
        if model_arch != "mlp":
            cf = torch.permute(cf, (0, 2, 3, 1))

        print(f"cf probability: {cf_prob}")
        print(f"compute probability: {pred_proba}")
        X_perturbed = mean_preserve_dimensions(
            cf * sigma + mu, preserveAxis=cf.ndim - 1
        )
        X_orig = X_mean * sigma + mu
        cf_delta = (X_perturbed - X_orig) / X_orig * 100
        print(f"cf delta: {cf_delta}")
        cf_perturbed = dict(zip(channel[isPerturbed], cf_delta[isPerturbed].numpy()))
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
            np.savez(
                savedFile,
                explanation=explanation,
                cf_perturbed=cf_perturbed,
                channel_to_perturb=channel_to_perturb,
            )
    return


def alter_image(y, unnormed_patch, mu, sigma, unnormed_mean):
    unnormed_y = y * sigma + mu
    new_patch = unnormed_patch * ((unnormed_y / unnormed_mean)[:, None, None, :])
    return (new_patch - mu) / sigma


def add_init_layer(init_fun, model):
    class input_fun(torch.nn.Module):
        def forward(self, input):
            return torch.permute(init_fun(input), (0, 3, 1, 2)).float()

    input_transform = input_fun()
    completeModel = torch.nn.Sequential(input_transform, model)
    return completeModel, input_transform


def mean_preserve_dimensions(tensor, preserveAxis=None):
    if isinstance(preserveAxis, int):
        preserveAxis = (preserveAxis,)

    # Compute the mean along all dimensions except those in preserveAxis
    dims_to_reduce = [i for i in range(tensor.ndim) if i not in preserveAxis]
    result = tensor.mean(dim=dims_to_reduce)
    return result
