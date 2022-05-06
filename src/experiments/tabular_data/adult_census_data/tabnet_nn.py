import time
from typing import List
import sys
sys.path.append('../')
sys.path.append('../tabnet/')
import torch

from tabnet.pytorch_tabnet.tab_model import TabNetClassifier
from tabnet.pytorch_tabnet import tab_network

class TabNetClass(tab_network.TabNet):
    def __init__(self,
                 input_dim=14,
                 output_dim=2,
                 n_d=8,
                 n_a=8,
                 n_steps=3,
                 gamma=1.3,
                 cat_idxs=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                 cat_dims=[73, 9, 16, 16, 7, 15, 6, 5, 2, 119, 92, 94, 42],
                 cat_emb_dims=1,
                 n_independent=2,
                 n_shared=2,
                 epsilon=1e-15,
                 virtual_batch_size=128,
                 momentum=0.02,
                 mask_type='entmax',
                 device=torch.device('cpu'),
                 lambda_sparse=1e-3
                ):
        
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dims = cat_emb_dims
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.mask_type = mask_type
        self.device = device
        self.lambda_sparse: float = lambda_sparse
        
        super(tab_network.TabNet, self).__init__()
        self.network = tab_network.TabNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=self.cat_emb_dims,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
        ).to(self.device)        
                
    
    def forward(self, X, sparsity_loss=False):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor`
            Input data

        Returns
        -------
        predictions : a :tensor: `torch.Tensor`
        M_loss : a :tensor: `torch.Tensor`
        """
        output, M_loss = self.network(X)
        predictions = torch.nn.Softmax(dim=1)(output)
        if sparsity_loss:
            return predictions, -self.lambda_sparse * M_loss
        return predictions
