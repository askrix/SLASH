#!/usr/bin/env python
# coding: utf-8

# # SpnAsp experiment notebook

# In[1]:


import train
from dataGen import get_loader, get_data_and_object_list, CLEVR
import numpy as np
import torch
import torchvision
import datetime


from importlib import reload  


# In[5]:


seed = 1
obj_num = 6
date_string = datetime.datetime.today().strftime('%d-%m-%Y')
experiments = {f'CLEVR{obj_num}_seed_{seed}': {'structure':'poon-domingos', 'pd_num_pieces':[4], 'learn_prior':True,
                         'lr': 0.01, 'bs':512, 'epochs':1100, 'pretrained_slot_module':False, 'debug':False,
                        'lr_warmup_steps':8, 'lr_decay_steps':368, 'use_em':False, 'train_slot':True, 'resume':False,
                         'start_date':date_string, 'credentials':'DO', 'p_num':32, 'seed':seed, 'obj_num':obj_num
                        }}


for exp_name in experiments:
    print(exp_name)
    train.slash_slot_attention(exp_name, experiments[exp_name])



