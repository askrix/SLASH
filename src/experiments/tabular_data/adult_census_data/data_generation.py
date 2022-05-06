import os
import wget
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


def getDataset():
    # 1. Download the data if necessary 
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    dataset_name = 'census-income'
    data_file = Path(os.getcwd()+'/data/'+dataset_name+'.csv')
    data_file.parent.mkdir(parents=True, exist_ok=True)
    if data_file.exists():
        print("File already exists.")
    else:
        print("Downloading file...")
        wget.download(url, data_file.as_posix())
    # 2. Load the data and split accordingly
    train = pd.read_csv(data_file)
    target = ' <=50K'
    if "Set" not in train.columns:
        indices_file = Path(os.getcwd()+'/data/indices')
        indices_file.parent.mkdir(parents=True, exist_ok=True)
        indices = []
        if indices_file.exists():
            print("File already exists. Load the indices...")
            indices = np.load(infices_file)
        else:    
            print("Pick the indeces at random and save these as the file for the future usage.")
            indices = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))
            np.save(indices_file, indices)
        train["Set"] = indices
    train_indices = train[train.Set=="train"].index
    valid_indices = train[train.Set=="valid"].index
    test_indices = train[train.Set=="test"].index
    # 3. Label encode categorical features and fill empty cells.
    nunique = train.nunique()
    types = train.dtypes
    categorical_columns = []
    categorical_dims =  {}
    for col in train.columns:
        if types[col] == 'object' or nunique[col] < 200:
            print(col, train[col].nunique())
            l_enc = LabelEncoder()
            train[col] = train[col].fillna("VV_likely")
            train[col] = l_enc.fit_transform(train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            train.fillna(train.loc[train_indices, col].mean(), inplace=True)
    # 4. Define categorical features for categorical embeddings
    unused_feat = ['Set']
    features = [ col for col in train.columns if col not in unused_feat+[target]] 
    cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]
    cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]
    # 5. Define the data subsets.
    X_train = train[features].values[train_indices]
    y_train = train[target].values[train_indices]
    X_test = train[features].values[test_indices]
    y_test = train[target].values[test_indices]
    X_valid = train[features].values[valid_indices]
    y_valid = train[target].values[valid_indices]
    #6. Return everything
    return cat_idxs, cat_dims, X_train, y_train, X_test, y_test, X_valid, y_valid


class A(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        data = self.X[idx]
        label = self.y[idx]
        return torch.tensor(data, dtype=torch.float), int(label.item())
    

class B(Dataset):
    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        p1, s1, p2, s2 = self.data[index]
        return self.dataset[p1][0], s1, self.dataset[p2][0], s2

    
class C(Dataset):
    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        p, w = self.data[index]
        return self.dataset[p][0], w
    

def generate_queries(data:A, mark:str='train'):
    data_file = Path(os.getcwd()+'/data/'+mark+'_data.txt')
    with open(data_file, 'w') as f:
        if len(data) % 2 == 1:
            n = len(data)-1
        else: 
            n = len(data)
        for i in np.arange(0, n, 2):
            if str(data[i][0][9]) != str(data[i+1][0][9]):
                f.write(str(i)+' '+str(int(data[i][0][9].item()))+' '+str(i+1)+' '+str(int(data[i+1][0][9].item())))
                f.write('\n')
            
            
def get_data_and_query_list(train_dataset:B):
    dataList = []
    queryList = []
    for p1, s1, p2, s2 in train_dataset:
        dataList.append({'p1': p1, 'p2': p2})
        queryList.append(':- not wealth_gap(p1,{},p2,{}).'.format(s1, s2))

    dataList = np.array(dataList)
    queryList = np.array(queryList)
    
    return dataList, queryList


def generate_queries_classification(data:A, mark:str='train'):
    data_file = Path(os.getcwd()+'/data/classification_'+mark+'_data.txt')
    with open(data_file, 'w') as f:
        if len(data) % 2 == 1:
            n = len(data)-1
        else: 
            n = len(data)
        for i in np.arange(n):
            f.write(str(i)+' '+str(data[i][1]))
            f.write('\n')
            

def get_data_and_query_list_classification(train_dataset:A):
    dataList = []
    queryList = []
    for p, w in train_dataset:
        dataList.append({'p': p})
        queryList.append(':- not wealthy(p,{}).'.format(w))

    dataList = np.array(dataList)
    queryList = np.array(queryList)
    
    return dataList, queryList


if __name__ == '__main__':
    # Obtain the data
    cat_idxs, cat_dims, X_train, y_train, X_test, y_test, X_valid, y_valid = getDataset()
    # Generate queries for training
    mark = 'train'
    train_set = A(X_train, y_train)
    generate_queries(train_set, mark)
    generate_queries_classification(train_set, mark)
    # Generate queries for testing
    mark = 'test'
    test_set = A(X_test, y_test)
    generate_queries(test_set, mark)
    generate_queries_classification(test_set, mark)
