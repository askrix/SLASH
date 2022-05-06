import torch
from torch.utils.data import Dataset
import numpy as np

class MNIST_Addition(Dataset):

    def __init__(self, dataset, examples, flat_for_spn):
        self.data = list()
        self.dataset = dataset
        self.flat_for_spn = flat_for_spn
        
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    
    def __getitem__(self, index):
        i1, i2, l = self.data[index]
        l = ':- not addition(i1, _, {}).'.format(l)
        
        if self.flat_for_spn:
            return {'i1': self.dataset[i1][0].flatten(), 'i2': self.dataset[i2][0].flatten()}, l
        else:
            return {'i1': self.dataset[i1][0], 'i2': self.dataset[i2][0]}, l
    
    
    def __len__(self):
        return len(self.data)