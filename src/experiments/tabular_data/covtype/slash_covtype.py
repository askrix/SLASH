#!/usr/bin/env python
# coding: utf-8


import train
import numpy as np
import torch
import torchvision
import datetime



date_string = datetime.datetime.today().strftime('%d-%m-%Y')
#Python script to start the shapeworld4 slot attention experiment
#Define your experiment(s) parameters as a hashmap having the following parameters
example_structure = {'exp_name': 
                   {'structure': 'binary-trees',#structure of the EinsumNetwork, either binary-trees or poon-domingos
                    #binary tree structure params
                    'num_repetitions': 20,
                    'depth': 5,
                    'num_sums':20,
                    'k':20,
                    #poon-domingos structure params
                    'pd_num_pieces':[6],
                    #general parameters
                    'lr': 0.01, 'bs':512, 'epochs':200, 'use_em':False,
                    'start_date':date_string, 'resume':False, 'p_num':4, 'credentials':'DO',
              'explanation': """Covtype experiment using WMC for a simple classification."""}}




experiments = {'covtype_bt_20_5_bs_512_s_20_k_20': 
                   {'structure': 'binary-trees', 'num_repetitions': 20, 'depth': 5,
                    'num_sums':20, 'k':20,
                    'lr': 0.01, 'bs':512, 'epochs':200, 'use_em':False,
                    'start_date':date_string, 'resume':False, 'p_num':1, 'credentials':'DO',
              'explanation': """Covtype experiment using WMC for a simple classification."""}}






# train the network
for exp_name in experiments:
    print(exp_name)
    train.slash_covtype(exp_name, experiments[exp_name])





