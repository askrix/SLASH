print("start importing...")

import time
import sys
sys.path.append('../../../')
sys.path.append('../../../SLASH/')
sys.path.append('../../../EinsumNetworks/src/')


#torch, numpy, ...
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import importlib

#own modules
from data_generation import getDataset, A, B, C, get_data_and_query_list, get_data_and_query_list_classification
from einsum_wrapper import EiNet
from tabnet_nn import TabNetClass

import slash
from slash import SLASH
import utils
from utils import set_manual_seed
from pathlib import Path
from rtpt import RTPT

print("...done")


def slash_tabular_learning(exp_name, exp_dict):
    
    # Set the seeds for PRNG
    set_manual_seed(exp_dict['seed'])

    # Create RTPT object
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name='SLASH ACI', max_iterations=int(exp_dict['epochs']))

    # Start the RTPT tracking
    rtpt.start()
    
    # Load data
    cat_idxs, cat_dims, X_train, y_train, X_test, y_test, X_valid, y_valid = getDataset()
    if exp_dict['use_spn']:
        std_eps = 1e-7
        mean = np.mean(X_train, axis=0)
        std = np.mean(X_train, axis=0)
        X_train = (X_train - mean) / (std + std_eps) 
        X_test = (X_test - mean) / (std + std_eps) 
        X_valid = (X_valid - mean) / (std + std_eps)
    A_train = A(X_train, y_train)
    A_test = A(X_test, y_test)
    if exp_dict['type'] == 'classification':
        train_dataset = C(dataset=A_train, examples='data/classification_train_data.txt')
        test_dataset = C(dataset=A_test, examples='data/classification_test_data.txt')
        dataList, queryList = get_data_and_query_list_classification(train_dataset)
    else:
        train_dataset = B(dataset=A_train, examples='data/train_data.txt')
        test_dataset = B(dataset=A_test, examples='data/test_data.txt')
        dataList, queryList = get_data_and_query_list(train_dataset)

    train_loader = torch.utils.data.DataLoader(A_train, batch_size=exp_dict['bs'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(A_test, batch_size=exp_dict['bs'], shuffle=True)

    # Program code
    if exp_dict['type'] == 'classification':
        program = '''
        person(p).
        wealthy(P,W) :- wealth(0,P,W).
        '''
    else:
        program = '''
        person(p1). person(p2).
        1{sex(p1,0); sex(p1,1)}1.
        1{sex(p2,0); sex(p2,1)}1.
        different_wealth(P1,W1,P2,W2) :- wealth(0,P1,W1), wealth(0,P2,W2), W1!=W2.
        different_sex(P1,S1,P2,S2) :- sex(P1,S1), sex(P2,S2), S1!=S2.
        wealth_gap(P1,S1,P2,S2) :- different_sex(P1,S1,P2,S2), different_wealth(P1,W1,P2,W2).        
        '''
    
    # Use tabnet or probabilisitc circuit
    if exp_dict['use_spn']:
    
        # Setup new CP given the parameters
        if exp_dict['structure'] == 'poon-domingos':
            m = EiNet(structure=exp_dict['structure'],
                      pd_num_pieces=exp_dict['pd_num_pieces'],
                      use_em=False,
                      num_var=14,
                      class_count=2,
                      pd_width=14,
                      pd_height=1,
                      learn_prior=exp_dict['learn_prior'])
        else: 
            m = EiNet(structure=exp_dict['structure'],
                      depth=exp_dict['depth'],
                      num_repetitions=exp_dict['num_repetitions'],
                      use_em=False,
                      num_var=14,
                      class_count=2,
                      learn_prior=exp_dict['learn_prior'])
        
        # Extend program with pc atom
        program += '''pc(wealth(1,X), [0,1]) :- person(X).'''
    
    else:
        # Define TabNet        
        m = TabNetClass(cat_idxs=cat_idxs, cat_dims=cat_dims, virtual_batch_size=exp_dict['bs']/8)
        
        # Extend program with tabnet atom
        program += '''tabnet(wealth(1,X), [0,1]) :- person(X).'''    
    
    print(program)
    
    saveModelPath = 'data/'+exp_name+'/slash_ACI.pt'
    Path("data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)

    print("Experiment parameters:", exp_dict)
     
    # Determine the number of trainable parameters            
    num_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in m.parameters())
    print("training with {} trainable params and {} params in total".format(num_trainable_params,num_params))
            
        
    # Create the SLASH Program
    nnMapping = {'wealth': m}
    optimizers = {'wealth': torch.optim.Adam(m.parameters(), lr=exp_dict['lr'], eps=1e-7)}
    SLASHobj = SLASH(program, nnMapping, optimizers)


    # Metric lists
    train_accuracy_list = []
    test_accuracy_list = []
    confusion_matrix_list = []
    startTime = time.time()
    
    
    # Evaluate the performanve directly after initialisation
    time_test = time.time()
    test_acc, _ = SLASHobj.testNetwork('wealth', test_loader)
    train_acc, _ = SLASHobj.testNetwork('wealth', train_loader)
    test_accuracy_list.append([test_acc, 0])
    train_accuracy_list.append([train_acc, 0])
    timestamp_test = utils.time_delta_now(time_test, simple_format=True)
    timestamp_total = utils.time_delta_now(startTime, simple_format=True)
    time_array = [0.0, timestamp_test, timestamp_total]

    
    # Save and print statistics
    print('Train Acc: {:0.2f}%, Test Acc: {:0.2f}%'.format(train_acc, test_acc))
    print('--- train time:  ---', 0)
    print('--- test time:  ---' , timestamp_test)
    print('--- total time from beginning:  ---', timestamp_total)

    
    # Export results and networks
    print('Storing the trained model into {}'.format(saveModelPath))
    torch.save({"tabnet":m.state_dict(),
                "test_accuracy_list":test_accuracy_list,
                "train_accuracy_list":train_accuracy_list,
                "num_params":num_trainable_params,
                "exp_dict":exp_dict,
                "exp_name":exp_name,
                "time":time_array,
                "program":program}, saveModelPath)
    
    
    # Train and evaluate the performance
    for e in range(exp_dict['epochs']):
        print('Epoch {}...'.format(e+1))

        time_train= time.time()
        SLASHobj.learn(dataList=dataList, queryList=queryList, 
                       epoch=1, batchSize=exp_dict['bs'],
                       p_num=exp_dict['p_num'], method='exact')  # 'network_prediction'
        timestamp_train = utils.time_delta_now(time_train, simple_format=True)

        time_test = time.time()
        test_acc, _ = SLASHobj.testNetwork('wealth', test_loader)
        train_acc, _ = SLASHobj.testNetwork('wealth', train_loader)        
        test_accuracy_list.append([test_acc, e])
        train_accuracy_list.append([train_acc, e])
        timestamp_test = utils.time_delta_now(time_test, simple_format=True)
        timestamp_total = utils.time_delta_now(startTime, simple_format=True)

        time_array = [timestamp_train, timestamp_test, timestamp_total]

        
        # Save and print statistics
        print('Train Acc: {:0.2f}%, Test Acc: {:0.2f}%'.format(train_acc, test_acc))
        print('--- train time:  ---', timestamp_train)
        print('--- test time:  ---' , timestamp_test)
        print('--- total time from beginning:  ---', timestamp_total)
        
        
        # Export results and networks
        print('Storing the trained model into {}'.format(saveModelPath))
        torch.save({"addition_net":m.state_dict(), 
                    "test_accuracy_list":test_accuracy_list,
                    "train_accuracy_list":train_accuracy_list,
                    "num_params":num_trainable_params,
                    "exp_dict":exp_dict,
                    "exp_name":exp_name,
                    "time":time_array,
                    "program":program}, saveModelPath)
        
        
        # Update the RTPT
        rtpt.step()
