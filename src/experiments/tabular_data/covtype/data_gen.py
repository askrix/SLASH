import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import datasets


indices = np.random.permutation(581012)



def get_data_and_object_list(dataset, batch_size, shuffle=True):
    """
    Returns a list of hashmaps containing data input for the NeurAsp program, the queries and the query attributes as an array.
    NeurASP expects the data in form of lists of a hashmap. Thats because we need to map the input for example two images to its corresponding atoms in the logic program.
    Example MNIST digit addition: [{'im1':[...], 'im2':[...] }, {'im1':[...], 'im2':[...] }, ...]
    """

    
    loader = get_loader(dataset, batch_size, 8, shuffle)
    
    data_list = []
    query_list = []
    obj_list = []
    for im, query, obj in loader:
        data_list.append({'im': im})
        query_list.append(query[0]) #query is a list but needs to be string
        obj_list.append(obj)
        
    data_list = np.array(data_list)
    query_list= np.array(query_list)
    obj_list = np.array(obj_list)

    
    return data_list, query_list, obj_list
        


def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    '''
    Returns and iterable dataset with specified batchsize and shuffling.
    '''
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
)
    
    
class COVTYPE(Dataset):
    def __init__(self, root, mode):
        
        datasets.maybe_download_covtype()

        
        self.root = root
        self.mode = mode
        assert os.path.exists(root), 'Path {} does not exist'.format(root)
        
        #load data from file
        data = np.loadtxt(root, delimiter=',')
        
        
        
        #normalize to be in [0,1]
        data[:,:-1] = (data[:,:-1] - data[:,:-1].min(0))/ (data[:,:-1].max(0)- data[:,:-1].min(0))
        data[:,-1] = data[:,-1] -1 #we want our class labels from 0-6 instead of 1-7
        
        if mode == 'train':
            self.X = torch.Tensor(data[indices[:460000],:-1])
            self.y = torch.Tensor(data[indices[:460000],-1])
            self.len= 460000
        else: 
            self.X = torch.Tensor(data[indices[460000:],:-1])
            self.y = torch.Tensor(data[indices[460000:],-1])          
            self.len = data.shape[0]-460000
            
    
                    
    def __getitem__(self, index):
        
        return self.X[index], int(self.y[index])
       
    def __len__(self):
        return self.len


train_dataset = COVTYPE(root='../../data/covtype/covtype.data' , mode='train')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(COVTYPE(root='../../data/covtype/covtype.data',mode='test'), batch_size=100, shuffle=True)
                                           
dataList = []
queryList = []
for t1, c in train_dataset:
    dataList.append({'t1': t1})
                                           
    random_noise_class = np.random.randint(1,8) #c= 1,2,...,7
    #query = ":- not forest(p1,{}) ;  not forest(p2,{}). ".format(c, random_noise_class)
    query = ":- not forest(p1,{}). ".format(c+1)

    queryList.append(query)

dataList= np.array(dataList)
queryList= np.array(queryList)