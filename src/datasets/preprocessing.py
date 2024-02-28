import numpy as np
import os
import pickle
import random
import pandas as pd

import sys

sys.path.append("/home/jerrywang/morpheus/src/")
from utils.misc import unison_shuffled_copies, check_split_constraints
from pprint import pprint


def get_stratified_splits(
    img_dir: str,
    patient_dir: str,
    patient_split=None,
    overwrite=False,
    save_path=None,
    param={
        "eps": 0.01,
        "train_lb": 0.65,
        "split_ratio": [0.65, 0.15, 0.2],
        "celltype": "Tcytotoxic",
        "ntol": 100,
    },
):
    # get folder path of image data as output path being saved to
    if save_path is None:
        save_path = os.path.dirname(img_dir)

    # generate data split if not already done or overwrite set to True
    if not os.path.isdir(os.path.join(save_path, "train")) or overwrite:
        print(f"Generating data splits and saving to {save_path}")
        stratified_data_split(
            img_dir,
            patient_dir,
            save_path=save_path,
            patient_split=patient_split,
            **param,
        )
    else:
        print(f"Given data directory already created: {save_path}")
        pprint(describe_data_split(save_path))
    return save_path


def describe_data_split(save_path):
    y_mean = {}
    pat = np.load(save_path + "/data_info.pkl", allow_pickle=True)["patient_df"]
    for group in ["train", "validate", "test"]:
        data = pd.read_csv(save_path + f"/{group}/label.csv")
        n_pat = len(
            np.unique(pat[pat["ImageNumber"].isin(data["ImageNumber"])]["PatientID"])
        )
        y = data["Tcytotoxic"].mean()
        y_mean.update({group: [round(y, 3), len(data), n_pat]})
    return y_mean


def stratified_data_split(
    img_dir: str,
    patient_path: str,
    save_path=None,
    patient_split={},
    celltype=str,
    split_ratio=[0.6, 0.2, 0.2],
    eps=0.05,
    train_lb=0.65,
    ntol=100,
):
    if save_path is None:
        save_path = os.path.dirname(img_dir)

    # Ratio of patients in different groups
    train_ratio, valid_ratio, test_ratio = split_ratio

    # load patient and image id
    pat_df = pd.read_csv(patient_path)
    pat_df = pat_df[["PatientID", "ImageNumber"]]
    pat_id = np.unique(pat_df["PatientID"].tolist())

    # load image data
    print("loading image data")
    try:
        with open(img_dir, "rb") as f:
            intensity, label, channel, _ = pickle.load(f)
    except Exception as e:
        print(f"Error loading image data: {e}")
        return
    npatches = intensity.shape[0]

    # split patient into train-test-validation group stratified by T cell level
    tr_te_diff = 1
    tr_va_diff = 1
    tr_prop = 0
    counter = 0
    while (
        not check_split_constraints(tr_te_diff, tr_va_diff, tr_prop, eps, train_lb)
        and counter < ntol
    ):
        if patient_split is None:
            patient_split["train"] = random.sample(
                list(pat_id), round(len(pat_id) * train_ratio)
            )
            remain = [pat for pat in pat_id if pat not in patient_split["train"]]
            patient_split["validate"] = random.sample(
                remain, round(len(remain) * valid_ratio / (test_ratio + valid_ratio))
            )
            patient_split["test"] = [
                pat for pat in remain if pat not in patient_split["validate"]
            ]
        # obtain image number corresponding to patient split
        image_split = {}
        for key, val in patient_split.items():
            image_split[key] = pat_df[pat_df["PatientID"].isin(val)]["ImageNumber"]

        # shuffle image patch in each split
        patch_split = {}
        label_split = {}
        index_split = {}
        split_balance = {}
        for key, val in image_split.items():
            _label = label[label["ImageNumber"].isin(val)]
            _index, _label = unison_shuffled_copies(_label.index, _label)
            patch_split[key] = intensity[_index, :]
            label_split[key] = _label[celltype]
            index_split[key] = _index
            split_balance[key] = label_split[key].mean()

        # compute sample condition values
        tr_prop = patch_split["train"].shape[0] / npatches
        tr_te_diff = abs(split_balance["train"] - split_balance["test"])
        tr_va_diff = abs(split_balance["train"] - split_balance["validate"])
        print(tr_te_diff, tr_va_diff, tr_prop)
        counter += 1

        # if sample conditions satisfied, save splits
        if check_split_constraints(tr_te_diff, tr_va_diff, tr_prop, eps, train_lb) or (
            image_split is not None
        ):
            print("Sample Proportions:")
            for key, val in patch_split.items():
                print("{}: {:.3f}".format(key, val.shape[0] / npatches))

            print("\nPositive Proportions:")
            for key, val in split_balance.items():
                print("{}: {:.3f}".format(key, val))

            # save splits
            split_info = {
                "channel": channel,
                "patient_df": pat_df,
                "train_set_mean": np.mean(patch_split["train"], axis=(0, 1, 2)),
                "train_set_stdev": np.std(patch_split["train"], axis=(0, 1, 2)),
                "patch_shape": intensity.shape[1:],
                "test_patient": patient_split["test"],
                "validate_patient": patient_split["validate"],
                "train_patient": patient_split["train"],
                "test_index": index_split["test"],
                "validate_index": index_split["validate"],
                "train_index": index_split["train"],
            }
            save_splits(save_path, patch_split, label_split, split_info)
            return
    print("Could not satisfy data split constraints, try again or adjust constraints")


def save_splits(save_path, data_dict, label_dict, split_info):
    with open(os.path.join(save_path, "split_info.pkl"), "wb") as f:
        pickle.dump(split_info, f, protocol=4)

    for key, val in data_dict.items():
        # make dir
        save_path = os.path.join(save_path, key)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
            os.makedirs(os.path.join(save_path, "0"))
            os.makedirs(os.path.join(save_path, "1"))

        # save labels
        label_dict[key].to_csv(os.path.join(save_path, "label.csv"), index=False)

        # save images
        np.save(os.path.join(save_path, "img.npy"), val)
        nimage = val.shape[0]
        for ind in range(nimage):
            label = str(label_dict[key].iloc[ind, 1])
            np.save(
                os.path.join(save_path, f"{label}/patch_{ind}.npy"),
                val[ind, ...],
            )
