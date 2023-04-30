import numpy as np
import pandas as pd
import pickle, os
from alibi.myutils.my_models import *
from alibi.myutils.imcwrangler import *

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

class IMCDataset:
    def __init__(self, name, data_dir, model_path=None, patient_path=None, classify_threshold=0.5, modelArch='unet'):
        self.name = name
        self.data_dir = data_dir
        self.patient_path = patient_path
        self.info_path = f'{self.data_dir}/data_info.pkl'
        self.minsize = 7*7

        info = np.load(self.info_path, allow_pickle=True)
        self.channel = info['channel']
        self.img_size = info['shape']
        self.mu = info['mean']
        self.stdev = info['stdev']
        self.test_df = None

        if model_path is not None:
            self.model_path = os.path.join(self.data_dir+model_path, os.listdir(self.data_dir+model_path)[0])
            self.get_classifier(self.model_path, classify_threshold, modelArch)
        else:
            self.classifier = None

    def get_full_df(self):
        return pd.read_csv(f'{self.data_dir}/singlecell_df.csv', index_col=False)
    
    def get_test_df(self, channel_to_remove=[]):
        # get full dataframe
        data = self.get_full_df()

        # Keep only test image numbers
        test_label = pd.read_csv(f'{self.data_dir}/test/label.csv', index_col=False)
        test_image = np.unique(test_label['ImageNumber'])
        test_df = data.loc[data['original_ImageNumber'].isin(test_image),:]

        # Remove channels
        genes_to_keep = [i for i in self.channel if i not in channel_to_remove]

        # Remove all signals from T cells
        test_df.loc[test_df['celltype_rf']=='Tcytotoxic', genes_to_keep] = 0

        self.test_df = test_df

        return test_df

    def get_classifier(self, model_path, classify_threshold=0.5, modelArch='unet'):
        self.classify_threshold = classify_threshold
        nnmodel = TissueClassifier.load_from_checkpoint(model_path, 
                                                    in_channels=len(self.channel),
                                                    img_size=self.img_size[0],
                                                    modelArch=modelArch)
        def classifier_fun(x):
            x = torch.from_numpy(x).float()
            if x.ndim == 4:
                x = torch.permute(x, (0,3,1,2))
            return torch.nn.functional.softmax(nnmodel(x), dim=1).detach().numpy()
        self.classifier = classifier_fun

    def compute_prediction(self):
        X_test = np.load(f'{self.data_dir}/test/img.npy')
        test_label = pd.read_csv(self.data_dir+'/test/label.csv')
        img_num = test_label[['ImageNumber']].values.flatten()
        y_test = test_label[['Tcytotoxic']].values.flatten()

        # predict patch label
        pred_orig = self.classifier((X_test-self.mu)/self.stdev)[:,1] 

        # map each patch to patient
        pre_post_df = pd.DataFrame({'ImageNumber': img_num.flatten(), 
                                    'orig': y_test.flatten() == 1, 
                                    'predict': pred_orig > self.classify_threshold})
        img_mean = pre_post_df.groupby(['ImageNumber']).mean().reset_index()
        img_mean['patch_count'] = pre_post_df.groupby(['ImageNumber']).size().reset_index().iloc[:,-1]

        # keep only big enough images
        self.img_mean = img_mean[img_mean['patch_count'] > self.minsize]

    def regression_plus_T_Test(self):
        x = self.img_mean['orig'].to_numpy()
        y = self.img_mean['predict'].to_numpy()

        # create and fit the linear regression model
        self.linmodel = LinearRegression()
        X = x.reshape(-1, 1)
        self.linmodel.fit(X, y)
        self.r_sq = self.linmodel.score(X, y)

        # perform a paired t-test
        _, p_value = stats.ttest_rel(x, y)
        print(f"{self.name} p-value:", p_value)
    
    def correlation_channel_Tcell(self):
        # Get data
        label_ = pd.read_csv(f'{self.data_dir}/train/label.csv')
        X_train = np.load(f'{self.data_dir}/train/img.npy')
        
        # Average across space and pre-process intensity value
        df = pd.DataFrame(np.mean(X_train, axis=(1,2)), columns=self.channel)

        # Normalize intensity
        df = (df-self.mu)/self.stdev

        corr_coef = {key: None for key in df.columns}
        p_coef = {key: None for key in df.columns}
        for id in df.columns:
            corr_coef[id], p_coef[id] = stats.pointbiserialr(df[id],label_['Tcytotoxic'])

        sorted_ind = list(np.argsort(list(corr_coef.values()))[::-1])
        key = np.array([list(corr_coef.keys())[i] for i in sorted_ind])
        corr = np.array([list(corr_coef.values())[i] for i in sorted_ind])
        p_val = np.array([list(p_coef.values())[i] for i in sorted_ind])

        return key, corr, p_val

    def single_cell_type_perturbation(self, target, perturbation=dict(), channel_to_remove=[]):
        df = self.test_df.copy(deep=True)

        # Apply perturbation
        delta = 0
        for celltype, delta in perturbation.items():
            original = np.sum(df.loc[df['celltype_rf']==celltype, target])
            new_val = df.loc[df['celltype_rf']==celltype, target] * (delta/100 + 1)
            df.loc[df['celltype_rf']==celltype, target] = new_val
            perturbed = np.sum(df.loc[df['celltype_rf']==celltype, target])
            delta = perturbed - original

        X, _, _ , _ = patch_to_matrix(df, width=self.img_size[0], height=self.img_size[1], 
                                        typeName='celltype_rf', 
                                        genelist=self.channel, celltype=['Tcytotoxic','Tumor'],
                                        channel_to_remove=channel_to_remove)
        X = np.swapaxes(np.swapaxes(X, 1, 2), 2, 3)

        # Normalize data and get prediction
        return self.classifier((X-self.mu)/self.stdev)[:,1] > self.classify_threshold, delta