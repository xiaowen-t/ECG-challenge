import torch
import ast
import random
import pandas as pd
import os
import numpy as np
import pickle
from tqdm import tqdm
import wfdb
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch
import ast
import random

# for preprocessing
def load_data_ptbxl(df, sampling_rate, path, standardize=False):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    if standardize:
        ss = StandardScaler()
        ss.fit(np.vstack(data).flatten()[:,np.newaxis].astype(float))
        return apply_standardizer(data, ss)
    return data
    
def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


def code_to_MI(code):
    scp = pd.read_csv("data/scp_statements.csv").rename({'Unnamed: 0':'scp_code'},axis=1)
    MI_codes = scp[scp.diagnostic_class=='MI']['scp_code']
    code_dict = ast.literal_eval(code)
    if 'NORM' in code_dict.keys() and code_dict['NORM'] == 100:
        return 0
    for key in code_dict:
        if key in list(MI_codes) and code_dict[key]>=80:
            return 1

class ECG_Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True,groups=[1,2,3]):
        """
        Dataloader for MI prediction
        """
        df = pd.read_csv('data/ptbxl_database.csv')#.reset_index(drop=True)
        X = load_data_ptbxl(df,100,'data/',standardize=True)
        df['BMI'] = df['weight']/(df['height']/100)/(df['height']/100)
        df['MI'] = df['scp_codes'].apply(lambda x: code_to_MI(x))
        # reshape from (n_samples, seq_len) to (n_samples, 1, seq_len)
        # X = X.reshape(X.shape[0],1,X.shape[1]*X.shape[2])
        df.loc[random.sample(sorted(df[(df['BMI']>0)&(df['BMI']<25)&(~df['MI'].isna())].index),128),'bmi_group'] = 1 
        df.loc[random.sample(sorted(df[(df['BMI']>=25)&(df['BMI']<30)&(~df['MI'].isna())].index),128),'bmi_group'] = 2 
        df.loc[random.sample(sorted(df[(df['BMI']>=30)&(df['BMI']<100)&(~df['MI'].isna())].index),128),'bmi_group'] = 3 
        target_indices = df[~df['MI'].isna()].index
        test_indices = df[~df['bmi_group'].isna()].index
        if train is True:
            indicies = df[(~df.index.isin(test_indices))&(df.index.isin(target_indices))].index
        else:
            indicies = df[df['bmi_group'].isin(groups)].index
        y = np.array(df.iloc[indicies]['MI'])
        X = X[indicies.to_list()]
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(np.array(y).reshape(len(y),1))

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :, :].T.float(), self.Y[idx,:].float()