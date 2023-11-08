import numpy as np
import pandas as pd
import os
import json
import warnings
import seaborn as sns
from morpheus.morpheus.utils.models import TissueClassifier
from morpheus.utils.imcwrangler import patch_to_matrix
import torch

from statsmodels.stats import multitest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class IMCDataset:
    def __init__(self, name, data_dir, model_path=None, patient_path=None, threshold=None, metadata=None, modelArch='unet', minsize=49):
        self.name = name
        self.data_dir = data_dir
        self.patient_ID = pd.read_csv(patient_path,index_col=[0])
        self.info_path = f'{self.data_dir}/data_info.pkl'
        self.figure_path = f'{self.data_dir}/figure'
        self.minsize = minsize
        self.patient_cluster=None
        self.modelArch = modelArch

        info = np.load(self.info_path, allow_pickle=True)
        self.channel = info['channel']
        self.img_size = info['shape']
        self.mu = info['mean']
        self.stdev = info['stdev']
        self.test_df = None
        self.threshold = threshold

        if model_path is not None:
            self.model_path = os.path.join(self.data_dir+model_path, os.listdir(self.data_dir+model_path)[0])
            self.get_classifier(self.model_path, threshold, self.modelArch)
        else:
            self.classifier = None

        if not os.path.isdir(self.figure_path):
            os.makedirs(self.figure_path)

        if metadata is not None:
            self.metadata = pd.read_csv(metadata)
            if self.name == 'Liver tumor':
                self.metadata.loc[self.metadata['ImageNumber']==134, 'type'] = 'metaT' # manually correct the type of 134
        else:
            self.metadata = None

        # initialize without counterfactual data
        self.cfdata = 0

    def set_cf(self, cf_dir):
        # relative perturbation result
        _path = os.path.join(self.data_dir+'/cf', cf_dir, 'cf_rel.csv')
        try:
            self.cf_rel = pd.read_csv(_path)
            if len(self.cf_rel.loc[self.cf_rel['PatientID'].isna(),'PatientID'])>0:
                self.cf_rel.loc[self.cf_rel['PatientID'].isna(),'PatientID'] = 'X'
                nan_image = self.cf_rel.loc[self.cf_rel['PatientID']=='X','ImageNumber'].unique()
                self.patient_ID.loc[self.patient_ID['ImageNumber'].isin(nan_image),'PatientID'] = 'X'
        except IOError:
            print(f'{_path} not found.')
        
        _path = os.path.join(self.data_dir+'/cf', cf_dir, 'queue.json')
        try:
            with open(_path, "r") as f:
                simulation_parameter_list = json.load(f)
                config = simulation_parameter_list[0]
                self.channel_to_perturb = config['channel_to_perturb']

                self.cf_data_dir = config['data_dir']
                if self.data_dir != self.cf_data_dir:
                    warnings.warn('counterfactual data path does not match current data path')

                self.cf_model_path = config['model_path']
                if self.model_path != self.cf_model_path:
                    warnings.warn('counterfactual classifier model path does not match data model path')

        except IOError:
            print(f'{_path} not found.')

    def get_data_split(self, split='train'):
        X = np.load(f'{self.data_dir}/{split}/img.npy')
        label = pd.read_csv(self.data_dir+f'/{split}/label.csv')
        return X, label

    def get_perturbation_cluster(self, n_cluster, show_plot=False):
        grouped_cf = self.cf_rel[self.cf_rel['prob']>self.threshold][['PatientID'] + self.channel_to_perturb].copy()
        grouped_cf = grouped_cf.groupby(['PatientID']).median()
        vmin = grouped_cf.min().min()
        vmax = grouped_cf.max().max()
        axe= sns.clustermap(grouped_cf, row_cluster=True, col_cluster=True, method = "ward",
                     norm=colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
        if not show_plot:
            plt.close(axe.fig)
        clusters = sch.fcluster(axe.dendrogram_row.linkage, n_cluster, 'maxclust')
        self.patient_cluster = []
        for ii in range(2):
            self.patient_cluster.append(np.array(grouped_cf.index[clusters==ii+1]))
    
    def compare_cancer_stage(self):
        stage_1 = self.patient_ID.loc[self.patient_ID['PatientID'].isin(self.patient_cluster[0]),:].groupby('PatientID')['Cancer_Stage'].first()
        stage_2 = self.patient_ID.loc[self.patient_ID['PatientID'].isin(self.patient_cluster[1]),:].groupby('PatientID')['Cancer_Stage'].first()
        stage_2[stage_2 == 'III or IV'] = 'unknown'
        stage_1_counts = stage_1.value_counts()
        stage_2_counts = stage_2.value_counts()
        stage_1_counts.index = 'Stage ' + stage_1_counts.index
        stage_2_counts.index = 'Stage ' + stage_2_counts.index
        cmap = plt.get_cmap('Pastel1')
        colors = cmap.colors
        colors = ['darkgray' if label == 'Stage unknown' else color for label, color in zip(stage_1_counts.index, colors)]
        stage_1_counts.plot.pie(autopct='%1.1f%%', colors=colors)
        plt.ylabel('')  # This line can be used to remove the ylabel.
        # plt.savefig(f'{self.figure_path}/cancer_stage_pie_1_{self.name}.svg')
        plt.show()

        stage_2_counts.plot.pie(autopct='%1.1f%%', colors=colors)
        plt.ylabel('')  # This line can be used to remove the ylabel.
        # plt.savefig(f'{self.figure_path}/cancer_stage_pie_2_{self.name}.svg')
        plt.show()

    def plot_volcano(self, n_cluster, patient_cluster=None, compare='gene', p_cutoff=0.05, show_heatmap=False, save_volcanoplot=False):
        """
        df: A pandas DataFrame with patient ID and gene expressions.
        partitions: A list of numpy arrays of patient IDs, specifying partitions of the DataFrame.
        """
        self.patient_cluster = patient_cluster
        if self.patient_cluster is None:
            self.get_perturbation_cluster(n_cluster, show_heatmap)

        if compare == 'celltype':
            # df = self.get_df(subset='train')
            df_raw = self.get_full_df()
            df = df_raw.pivot_table(index='ImageNumber', columns='celltype_rf', aggfunc='size', fill_value=0)
            df = df.reset_index()
            df['ImageNumber'] = df['ImageNumber'].apply(
                                lambda score: df_raw[df_raw['ImageNumber'] == score]['original_ImageNumber'].values[0])
            df = df[df['ImageNumber'].isin(self.metadata[self.metadata['type']!='Nor']['ImageNumber'])]
            column = [col for col in df.columns if (col != 'ImageNumber')]
        elif compare == 'gene':
            X, label = self.get_data_split(split='train')
            if self.metadata is not None:
                filter = label['ImageNumber'].isin(self.metadata[self.metadata['type']!='Nor']['ImageNumber'])
                X = X[filter,...]
                label = label[filter].reset_index(drop=True)
            X = X.sum(axis=(1,2)) 
            X = pd.DataFrame(X, columns=self.channel)
            # X = np.arcsinh(X) # Arc-sinh transformation since t test assumes normality
            df = pd.concat([label, X], axis=1) 
            # df = df.loc[(df['Tumor']==1),:]
            column = self.channel_to_perturb
        self.partitioned_dfs = self.partition_dataframe(df, column+['ImageNumber'])

        fold_changes = []
        g1_mean = []
        g2_mean = []
        p_values = []

        # Iterate through each column (gene)
        for item in column:
            # Calculate mean expression levels for the gene in each partition
            if compare == 'celltype':
                expression_i = self.partitioned_dfs[0].groupby('ImageNumber').mean()[item].dropna()
                expression_j = self.partitioned_dfs[1].groupby('ImageNumber').mean()[item].dropna()
            elif compare == 'gene':
                expression_i = self.partitioned_dfs[0][item].dropna()
                expression_j = self.partitioned_dfs[1][item].dropna()

            # Calculate fold change
            fold_change = np.mean(expression_i) / np.mean(expression_j)
            fold_changes.append(np.log2(fold_change))  # we log transform the fold change for better visualization and stability
            g1_mean.append(np.mean(expression_i))
            g2_mean.append(np.mean(expression_j))
            
            # Calculate p-value
            if compare == 'celltype':
                _, p_value = stats.ranksums(expression_i, expression_j)
            else:
                _, p_value = stats.ttest_ind(expression_i, expression_j)
            p_values.append(p_value)  # we transform p-value to -log10 for better visualization

        # Adjust p-values for multiple testing
        self.p_values = p_values
        _, p_values_adj, _, alphacBonf = multitest.multipletests(p_values, method='sidak')
        # Create DataFrame for the plot
        plot_df = pd.DataFrame({
            compare: column,
            'log2(fold_change)': fold_changes,
            'g1_mean': g1_mean,
            'g2_mean': g2_mean,
            '-log10(p_value_adj)': [-np.log10(val) if val != 0 else 150 for val in p_values_adj],
        })
        self.p_values_adj = p_values_adj
        
        # Create the volcano plot
        plt.figure(figsize=(1.7, 3.5))

        # Create masks for significant and non-significant points
        significant_mask = plot_df['-log10(p_value_adj)'] > -np.log10(p_cutoff)
        positive_fc_mask = plot_df['log2(fold_change)'] > 0
        negative_fc_mask = plot_df['log2(fold_change)'] < 0

        # Color significant points with positive fold change red
        plt.scatter('log2(fold_change)', '-log10(p_value_adj)', data=plot_df[significant_mask & positive_fc_mask], color='red')

        # Color significant points with negative fold change blue
        plt.scatter('log2(fold_change)', '-log10(p_value_adj)', data=plot_df[significant_mask & negative_fc_mask], color='blue')

        # Color non-significant points gray
        plt.scatter('log2(fold_change)', '-log10(p_value_adj)', data=plot_df[~significant_mask], color='gray')

        # Adding vertical line at x=0
        plt.axvline(x=0, color='black', linestyle='--')

        # Adding horizontal line for threshold on p-value
        plt.axhline(y=-np.log10(0.05), color='black', linestyle='--')

        plt.xlabel('log2(Fold Change between patient cluster 1 & 2)')
        plt.ylabel('-log10(Adjusted p-value)')
        
        # plt.title(f'Volcano plot of {compare} differences between partition 1 and 2')

        for i in range(plot_df.shape[0]):
            plt.text(plot_df['log2(fold_change)'][i], plot_df['-log10(p_value_adj)'][i], plot_df[compare][i], fontsize=8)
        plt.yscale("log")
        if save_volcanoplot:
            plt.savefig(f'{self.figure_path}/{compare}_volcano_plot_{self.name}.svg')

        plt.show()

        return plot_df

    def partition_dataframe(self, df, items):
        """
        df: A pandas DataFrame with patient ID and gene expressions.
        partitions: A list of numpy arrays of patient IDs, specifying partitions of the DataFrame.
        """
        # Store the partitions as DataFrames in a list
        partitioned_dfs = []
        for partition in self.patient_cluster:
            part_image = self.patient_ID[self.patient_ID['PatientID'].isin(partition)]['ImageNumber']
            partitioned_dfs.append(df.loc[df['ImageNumber'].isin(part_image), items])
        return partitioned_dfs

    def get_full_df(self):
        df = pd.read_csv(f'{self.data_dir}/singlecell_df.csv', index_col=False)
        # df['ImageNumber'] = df['original_ImageNumber']
        # df = df.drop(columns=['original_ImageNumber'])
        return df
    
    def get_df(self, subset='test', channel_to_remove=[]):
        # get full dataframe
        data = self.get_full_df()

        # Keep only test image numbers
        _label = pd.read_csv(f'{self.data_dir}/{subset}/label.csv', index_col=False)
        _image = np.unique(_label['original_ImageNumber'])
        test_df = data.loc[data['original_ImageNumber'].isin(_image),:]

        # Remove channels
        genes_to_keep = [i for i in self.channel if i not in channel_to_remove]

        # Remove all signals from T cells
        test_df.loc[test_df['celltype_rf']=='Tcytotoxic', genes_to_keep] = 0

        self.test_df = test_df

        return test_df

    def get_classifier(self, model_path, threshold=None, modelArch='unet'):
        if threshold is not None:
            self.threshold = threshold
        nnmodel = TissueClassifier.load_from_checkpoint(model_path, 
                                                    in_channels=len(self.channel),
                                                    img_size=self.img_size[0],
                                                    modelArch=self.modelArch)
        # disable randomness, dropout, etc...
        nnmodel.eval()
        def classifier_fun(x):
            x = torch.from_numpy(x).float()
            if x.ndim == 4:
                x = torch.permute(x, (0,3,1,2))
            pred = nnmodel(x)
            if self.modelArch == 'lr':
                pred = torch.column_stack((pred, 1 - pred[:, 0]))
            return torch.nn.functional.softmax(pred, dim=1).detach().numpy()
        self.classifier = classifier_fun

    @staticmethod
    def softmax(x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # for numerical stability
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def compute_prediction(self, threshold=None, split='test'):
        if threshold is None:
            threshold = self.threshold
        X, label = self.get_data_split(split)
        
        img_num = label[['ImageNumber']].values.flatten()
        y_test = label[['Tcytotoxic']].values.flatten()

        # predict patch label
        X = (X-self.mu)/self.stdev
        if self.modelArch != 'unet':
            X = np.mean(X, axis=(1,2))
        pred = self.classifier(X)
        if pred.shape[1] == 1:
            new_col = 1 - pred[:, 0]
            pred = np.column_stack((pred, new_col))
        pred = pred[:,1]

        # map each patch to patient
        pre_post_df = pd.DataFrame({'ImageNumber': img_num.flatten(), 
                                    'orig': y_test.flatten() == 1, 
                                    'predict': pred.flatten() > threshold})
        img_mean = pre_post_df.groupby(['ImageNumber']).mean().reset_index()
        img_mean['patch_count'] = pre_post_df.groupby(['ImageNumber']).size().reset_index().iloc[:,-1]

        # keep only big enough images
        self.img_mean = img_mean[img_mean['patch_count'] > self.minsize]

        rmse = np.sqrt(mean_squared_error(self.img_mean['orig'], self.img_mean['predict']))
        print('Root Mean Squared Error:', rmse)
        return pred, X

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
        self.p_value = p_value
        print(f"{self.name} p-value:", self.p_value)
    
    def correlation_channel_Tcell(self, split='train'):
        # Get data
        X, label = self.get_data_split(split)
        
        # Average across space and pre-process intensity value
        df = pd.DataFrame(np.mean(X, axis=(1,2)), columns=self.channel)

        # Normalize intensity
        df = (df-self.mu)/self.stdev

        corr_coef = {key: None for key in df.columns}
        p_coef = {key: None for key in df.columns}
        for id in df.columns:
            corr_coef[id], p_coef[id] = stats.pointbiserialr(df[id],label['Tcytotoxic'])

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
        return self.classifier((X-self.mu)/self.stdev)[:,1] > self.threshold, delta
    
    def get_split_statistic(self):
        # compute proportion of T-cell positive patches
        self.t_cell_freq = dict()
        self.core_patient = dict()
        groups_ = ['train','test','validate']
        for group_ in groups_:
            label_ = pd.read_csv(self.data_dir+f'/{group_}/label.csv')
            label_['PatientID'] = label_['ImageNumber'].apply(
                                lambda score: self.patient_ID[self.patient_ID['ImageNumber'] == score]['PatientID'].values[0])
            t_cell_prop = label_.groupby('ImageNumber').sum()['Tcytotoxic']/label_.value_counts('ImageNumber')
            core_per_patient = label_.groupby('PatientID')['ImageNumber'].nunique().values
            self.t_cell_freq[group_] = t_cell_prop
            self.core_patient[group_] = core_per_patient
    
    def assess_masking(self, dataset, n, k):
        """
        Assess the impact of random masking on the prediction probability of a given dataset.

        Parameters:
        - dataset (str): The name (or path) of the dataset directory containing the image data.
        - n (int): Number of randomized versions to create for each image in the test set.
        - k (int): Number of indices to randomly mask (set to zero) in each randomized version of an image.

        Returns:
        - prob_diff (ndarray): Difference in predicted probabilities between the randomized and original images.
        - decision_diff (ndarray): Difference in decision outcomes (based on a threshold) 
                                between the randomized and original images.

        Description:
        1. Loads the test images from the specified dataset.
        2. Filters out any samples where the number of cells is less than or equal to zero.
        3. Computes the predicted probability for each image in the original test set using the model's classifier.
        4. For each image:
        - Identifies the indices of positive numbers (non-zero cells).
        - Generates n randomized versions of the image. In each version, randomly selects k of the identified 
            indices and masks them (sets them to zero).
        5. Computes the predicted probabilities for each of the randomized images.
        6. Calculates the differences in probabilities (prob_diff) and decision outcomes (decision_diff) 
        between the randomized and original images.

        Usage Notes:
        - Ensure that the image files are in the format 'img.npy' and are located in the specified dataset directory.
        - The function assumes the presence of instance attributes `data_dir`, `mu`, `stdev`, `classifier`, and `threshold`.
        """
        #compute predicted probability on test set
        X_test = np.load(f'{self.data_dir}/{dataset}/img.npy')
        cond = np.sum(np.max(X_test, axis=-1)>0, axis=(1,2))<=0
        X_test = X_test[~cond,...]
        print(f'{np.sum(cond)} samples removed due to having #cells <= {0}')

        n_sample = X_test.shape[0]
        # pred_orig = self.classifier((X_test-self.mu)/self.stdev)[:,1].reshape(-1, 1)

        prob_diffs = []
        decision_diffs = []

        for j in range(n_sample):
            orig = X_test[j,...][np.newaxis, ...]
            arr = np.max(orig, axis=-1)

            # Find the indices of positive numbers
            indices = np.argwhere(arr > 0)

            pred_orig = self.classifier((orig-self.mu)/self.stdev)[0][1]

            for _ in range(n):
                randomized_img = orig.copy()
                
                if indices.shape[0] < k:
                    chosen_indices = np.random.choice(indices.shape[0], k, replace=True)
                else:
                    chosen_indices = np.random.choice(indices.shape[0], k, replace=False)

                randomized_img[0, indices[chosen_indices, 0], indices[chosen_indices, 1], :] = 0

                pred_randomized = self.classifier((randomized_img-self.mu)/self.stdev)[0][1]

                prob_diffs.append(pred_randomized - pred_orig)
                decision_diffs.append((pred_randomized > self.threshold).astype(int) - (pred_orig > self.threshold).astype(int))

        return np.array(prob_diffs).reshape(n_sample, n), np.array(decision_diffs).reshape(n_sample, n)
        # X_randomized = np.empty((n*n_sample,)+X_test.shape[1:])
        # for j in range(n_sample):
        #     orig = X_test[j,...]
        #     arr = np.max(orig, axis=-1)

        #     # Find the indices of positive numbers
        #     indices = np.argwhere(arr > 0)

        #     # # Generate n versions of the array
        #     arr_versions = np.repeat(orig[np.newaxis, :, :, :], n, axis=0)

        #     # # Randomly choose k indices to set to zero in each version
        #     for i in range(n):
        #         if indices.shape[0] < k:
        #             chosen_indices = np.random.choice(indices.shape[0], k, replace=True)
        #         else:
        #             chosen_indices = np.random.choice(indices.shape[0], k, replace=False)
        #         # Set chosen indices to zero in the version
        #         arr_versions[i, indices[chosen_indices, 0], indices[chosen_indices, 1], :] = 0
        #     X_randomized[j*n:(j+1)*n,...] = arr_versions
        # pred_randomized = self.classifier((X_randomized-self.mu)/self.stdev)[:,1].reshape(n_sample,n)
        # prob_diff = pred_randomized - pred_orig
        # decision_diff = (pred_randomized > self.threshold).astype(int) - (pred_orig > self.threshold).astype(int)
        # return prob_diff, decision_diff

