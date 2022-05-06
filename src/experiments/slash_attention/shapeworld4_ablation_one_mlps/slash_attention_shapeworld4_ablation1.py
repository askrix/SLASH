#!/usr/bin/env python
# coding: utf-8


import train
import numpy as np
import torch
import torchvision
import datetime


#Python script to start the shapeworld4 slot attention experiment
#Define your experiment(s) parameters as a hashmap having the following parameters
example_structure = {'experiment_name': 
                   {'structure': 'poon-domingos',
                    'pd_num_pieces': [4],
                    'bs':50, #the batchsize
                    'epochs':1000, #number of epochs to train
                    'lr_warmup':True, #boolean indicating the use of learning rate warm up
                    'lr_warmup_steps':25, #number of epochs to warm up the slot attention module, warmup does not apply to the SPNs
                    'start_date':"01-01-0001", #current date
                    'resume':False, #you can stop the experiment and set this parameter to true to load the last state and continue learning
                    'credentials':'DO', #your credentials for the rtpt class
                    'explanation': """Running the whole SlotAttention+Slash pipeline using poon-domingos as SPN structure learner."""}}





#EXPERIMENTS
date_string = datetime.datetime.today().strftime('%d-%m-%Y')
experiments = {'shapeworld4_ablation1_1024':
                   {'structure': 'poon-domingos',
                    'bs':512, 'epochs':1100, 
                    'lr_warmup_steps':8, 'lr_decay_steps':368,
                    'start_date':date_string, 'resume':False, 'credentials':'DO','seed':1,
                    'p_num':8,
              'explanation': """Running the whole SlotAttention+Slash pipeline using poon-domingos as SPN structure learner."""}}



# train the network
for exp_name in experiments:
    print(exp_name)
    train.slash_slot_attention(exp_name, experiments[exp_name])





