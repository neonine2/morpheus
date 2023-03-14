import os, sys, pickle
import random 
from pathlib import Path
from utils.misc import *
import pandas as pd

import torch
import torch.nn.functional as F

def make_dir(data_path, model_arch):
    # create model path and filename
    model_name = Path(data_path).stem + '_' + model_arch
    model_dir = 'model/{}'.format(model_name)
    splitdata_dir = 'data/{}'.format(model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(splitdata_dir):
        os.makedirs(splitdata_dir)
    model_path = uniquify(os.path.join(model_dir, model_arch + '.h5'))
    return model_path, splitdata_dir

def stratified_data_split(metadata_path, data_path, splitdata_path, split_ratio, 
                          eps=0.05, train_lb=0.65, ntol=100, celltype='Tcytotoxic'):
    #parameters
    train_ratio = split_ratio[0]
    valid_ratio = split_ratio[1]
    test_ratio = split_ratio[2]
    
    #metadata about patient tumor hot/cold status
    pat_df = pd.read_csv(metadata_path)[['IHC_T_score','PatientID','ImageNumber']]

    #filter out images without immune status classification
    nan_imagenumber = pat_df[~pat_df['IHC_T_score'].notna()]['ImageNumber']
    pat_df = pat_df[pat_df['PatientID'].notna()]

    #classify patients as hot if at least one tissue profile classified as inflammed, cold otherwise
    pat_id = np.unique(pat_df['PatientID'].tolist())
    pat_status = dict.fromkeys(pat_id)
    for ID in pat_id:
        T_scores = pat_df[pat_df['PatientID'] == ID]['IHC_T_score'].tolist()
        count = {item: T_scores.count(item) for item in T_scores}
        pat_status[ID] = max(count, key=count.get)
    hot_pat = [k for k, v in pat_status.items() if ('I' in v)]
    cold_pat = [k for k in pat_id if k not in hot_pat]

    #load image data
    with open(data_path, 'rb') as f:
        intensity, label, channel = pickle.load(f)

    #stratified data train-test-validation split patient tumor infiltration status
    tr_te_diff = 1
    tr_va_diff = 1
    tr_prop = 0
    counter = 0
    while not sample_cond(tr_te_diff, tr_va_diff, tr_prop, eps, train_lb):
        # first sort patients into train, validation and test groups
        train_pat = random.sample(hot_pat, round(len(hot_pat)*train_ratio))+ \
                           random.sample(cold_pat, round(len(cold_pat)*train_ratio))
        remain_hot = [pat for pat in hot_pat if pat not in train_pat]
        remain_cold = [pat for pat in cold_pat if pat not in train_pat]
        valid_pat = random.sample(remain_hot, round(len(remain_hot)*valid_ratio/(test_ratio + valid_ratio))) + \
                            random.sample(remain_cold, round(len(remain_cold)*valid_ratio/(test_ratio + valid_ratio)))
        test_pat = [pat for pat in pat_id if (pat not in train_pat and pat not in valid_pat)]

        # obtain image number corresponding to patient split
        train_image = pd.concat([pat_df[pat_df['PatientID'].isin(train_pat)]['ImageNumber'], nan_imagenumber])
        valid_image = pat_df[pat_df['PatientID'].isin(valid_pat)]['ImageNumber']
        test_image = pat_df[pat_df['PatientID'].isin(test_pat)]['ImageNumber']

        # obtain patch data from image number
        train_label = label[np.isin(label['ImageNumber'],train_image)]
        val_label = label[np.isin(label['ImageNumber'],valid_image)]
        test_label = label[np.isin(label['ImageNumber'],test_image)]
        X_train,X_val,X_test = intensity[train_label.index,:],intensity[val_label.index,:],intensity[test_label.index,:]
        X_train,train_label = unison_shuffled_copies(X_train, train_label)
        X_val,val_label = unison_shuffled_copies(X_val, val_label)
        X_test,test_label = unison_shuffled_copies(X_test, test_label)
        y_train,y_val,y_test = train_label[celltype],val_label[celltype],test_label[celltype]

        # compute sample condition values 
        tr_prop = X_train.shape[0]/intensity.shape[0]
        y_train_mean,y_val_mean,y_test_mean = y_train.mean(),y_val.mean(),y_test.mean()
        tr_te_diff = abs(y_train_mean - y_test_mean)
        tr_va_diff = abs(y_train_mean - y_val_mean)

        # if sample conditions satisfied, save split
        if sample_cond(tr_te_diff, tr_va_diff, tr_prop, eps, train_lb):
            print("""Train sample prop: {} \nValidation sample prop: {} \nTest sample prop: {} 
            \nNumber of test patient: {} \nTrain class prop: {}  \nValid class prop: {} \nTest class prop: {}"""
                  .format(tr_prop, X_val.shape[0]/intensity.shape[0],X_test.shape[0]/intensity.shape[0],
                          len(test_pat), y_train_mean, y_val_mean, y_test_mean))
            mu = np.mean(X_train,axis=(0,1,2))
            std = np.std(X_train,axis=(0,1,2))
            data_dict = {"channel":channel, "patient_df":pat_df, "mean":mu, "stdev":std, "X_train":X_train}
            file = open(os.path.join(splitdata_path,'data_info.pkl'), 'wb')
            pickle.dump(data_dict, file, protocol=4)
            file.close()
            
            data_dict = {"train":X_train,"validation":X_val,"test":X_test}
            label_dict = {"train":train_label,"validation":val_label,"test":test_label}
            for key, val in data_dict.items():
                # make dir
                save_path = os.path.join(splitdata_path,key)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                # save labels
                label_dict[key].to_csv(save_path+'/label.csv', index=False)
                # save images
                nimage = val.shape[0]
                for ind in range(nimage):
                    np.save(os.path.join(save_path, 'patch_{}.npy'.format(ind)), val[ind,...])  
            return
        elif counter>ntol:
            sys.exit("Could not satisfy data split requirements")
        else:
            counter+=1
            

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(os.path.join(self.img_dir, 'label.csv'))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        # Get data and label   
        # image = self.img_file[idx,::]
        img_path = os.path.join(self.img_dir, 'patch_{}.npy'.format(idx))
        image = np.load(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    