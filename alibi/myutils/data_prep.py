import os, sys, pickle
import random 
from pathlib import Path
from alibi.myutils.misc import *
import pandas as pd
from pprint import pprint

def describe_data_split(output_path):
    y_mean = {}
    pat = np.load(output_path+f"/data_info.pkl", allow_pickle=True)['patient_df']
    for group in ['train','validate','test']:
        data = pd.read_csv(output_path+f"/{group}/label.csv")
        n_pat = len(np.unique(pat[pat['ImageNumber'].isin(data['ImageNumber'])]['PatientID']))
        y = data['Tcytotoxic'].mean()
        y_mean.update({group:[round(y,3), len(data), n_pat]})
    return y_mean

def generate_split_from_data(DATA_NAME, metadata_path, image_split=None,
                             param = {'eps':0.01, "train_lb":0.65, "split_ratio":[0.65,0.15,0.2]}):
    DIR = os.path.dirname(metadata_path)
    output_path = os.path.expanduser(f'{DIR}/{DATA_NAME}')
    rawdata_path = f'{output_path}/patched.dat'
    
    # split data and save to splitdata_path
    if not os.path.isdir(output_path):
        print("creating directory")
        os.makedirs(output_path)
    else:
        print(f"Given data directory already created: {output_path}")

    if len(os.listdir(output_path))<=3:
        stratified_data_split(rawdata_path, metadata_path, output_path, image_split, **param)
    else:
        print("Data directory is already filled with the following split:")
        pprint(describe_data_split(output_path))
    return output_path

def stratified_data_split(data_path, metadata_path, splitdata_path, image_split=None, 
                          split_ratio=[0.6,0.2,0.2], eps=0.05, train_lb=0.65, ntol=100, celltype='Tcytotoxic'):
    # Ratio of patients in different groups
    train_ratio = split_ratio[0]
    valid_ratio = split_ratio[1]
    test_ratio = split_ratio[2]
    
    #metadata about patient tumor hot/cold status
    metadata = pd.read_csv(metadata_path)
    if 'IHC_T_score' in metadata:
        pat_df = metadata[['IHC_T_score','PatientID','ImageNumber']]

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
    else:
        pat_df = metadata[['PatientID','ImageNumber']]
        pat_id = np.unique(pat_df['PatientID'].tolist())
    
    #load image data
    with open(data_path, 'rb') as f:
        intensity, label, channel, _ = pickle.load(f)

    #stratified data train-test-validation split patient tumor infiltration status
    tr_te_diff = 1
    tr_va_diff = 1
    tr_prop = 0
    counter = 0
    while not sample_cond(tr_te_diff, tr_va_diff, tr_prop, eps, train_lb):
        if image_split is None:
            if 'IHC_T_score' in metadata:
                # first sort patients into train, validation and test groups
                train_pat = random.sample(hot_pat, round(len(hot_pat)*train_ratio))+ \
                                random.sample(cold_pat, round(len(cold_pat)*train_ratio))
                remain_hot = [pat for pat in hot_pat if pat not in train_pat]
                remain_cold = [pat for pat in cold_pat if pat not in train_pat]
                valid_pat = random.sample(remain_hot, round(len(remain_hot)*valid_ratio/(test_ratio + valid_ratio))) + \
                                    random.sample(remain_cold, round(len(remain_cold)*valid_ratio/(test_ratio + valid_ratio)))
                test_pat = [pat for pat in pat_id if (pat not in train_pat and pat not in valid_pat)]
                train_image = pd.concat([pat_df[pat_df['PatientID'].isin(train_pat)]['ImageNumber'], nan_imagenumber])
            else:
                train_pat = random.sample(list(pat_id), round(len(pat_id)*train_ratio))
                remain = [pat for pat in pat_id if pat not in train_pat]
                valid_pat = random.sample(remain, round(len(remain)*valid_ratio/(test_ratio + valid_ratio)))
                test_pat = [pat for pat in remain if pat not in valid_pat]
                train_image = pat_df[pat_df['PatientID'].isin(train_pat)]['ImageNumber']

            # obtain image number corresponding to patient split
            valid_image = pat_df[pat_df['PatientID'].isin(valid_pat)]['ImageNumber']
            test_image = pat_df[pat_df['PatientID'].isin(test_pat)]['ImageNumber']
        else:
            train_image = image_split['train']
            valid_image = image_split['val']
            test_image = image_split['test']
            test_pat = pat_df.loc[pat_df['ImageNumber'].isin(test_image), 'PatientID'].unique()
            valid_pat = pat_df.loc[pat_df['ImageNumber'].isin(valid_image), 'PatientID'].unique()
            train_pat = pat_df.loc[pat_df['ImageNumber'].isin(train_image), 'PatientID'].unique()

        # obtain patch data from image number
        train_label = label[np.isin(label['ImageNumber'],train_image)]
        val_label = label[np.isin(label['ImageNumber'],valid_image)]
        test_label = label[np.isin(label['ImageNumber'],test_image)]
        
        train_index,train_label = unison_shuffled_copies(train_label.index, train_label)
        val_index,val_label = unison_shuffled_copies(val_label.index, val_label)
        test_index,test_label = unison_shuffled_copies(test_label.index, test_label)
        X_train,X_val,X_test = intensity[train_index,:],intensity[val_index,:],intensity[test_index,:]
        y_train,y_val,y_test = train_label[celltype],val_label[celltype],test_label[celltype]

        # compute sample condition values 
        tr_prop = X_train.shape[0]/intensity.shape[0]
        y_train_mean,y_val_mean,y_test_mean = y_train.mean(),y_val.mean(),y_test.mean()
        tr_te_diff = abs(y_train_mean - y_test_mean)
        tr_va_diff = abs(y_train_mean - y_val_mean)

        # if sample conditions satisfied, save split
        print(tr_te_diff, tr_va_diff, tr_prop)
        if sample_cond(tr_te_diff, tr_va_diff, tr_prop, eps, train_lb) or (image_split is not None):
            print('Balanced data splitting done with conditions satisfied!')
            print("""Train sample prop: {} \nValidation sample prop: {} \nTest sample prop: {} 
            \nNumber of test patients: {} \nTrain class prop: {}  \nValid class prop: {} \nTest class prop: {}"""
                .format(tr_prop, X_val.shape[0]/intensity.shape[0],X_test.shape[0]/intensity.shape[0],
                        len(test_pat), y_train_mean, y_val_mean, y_test_mean))
            mu = np.mean(X_train,axis=(0,1,2))
            std = np.std(X_train,axis=(0,1,2))
            data_dict = {"channel":channel, "patient_df":pat_df, 
                        "mean":mu, "stdev":std, "shape":X_train.shape[1:], "train_index":train_index, 
                        "val_index":val_index, "test_index":test_index, "test_pat":test_pat,
                        "validate_pat":valid_pat, "train_pat":train_pat}
            file = open(os.path.join(splitdata_path,'data_info.pkl'), 'wb')
            pickle.dump(data_dict, file, protocol=4)
            file.close()
            
            data_dict = {"train":X_train,"validate":X_val,"test":X_test}
            label_dict = {"train":train_label,"validate":val_label,"test":test_label}
            for key, val in data_dict.items():
                # make dir
                save_path = os.path.join(splitdata_path,key)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                    os.makedirs(os.path.join(save_path,'0'))
                    os.makedirs(os.path.join(save_path,'1'))
                # save labels
                label_dict[key].to_csv(save_path+'/label.csv', index=False)
                # save images
                np.save(os.path.join(save_path, 'img.npy'),val)
                nimage = val.shape[0]
                for ind in range(nimage):
                    label = str(label_dict[key].iloc[ind,1])
                    np.save(os.path.join(save_path, f'{label}/patch_{ind}.npy'), val[ind,...])  
            return
        elif counter>ntol:
            sys.exit("Could not satisfy data split requirements")
        else:
            counter+=1
        



    