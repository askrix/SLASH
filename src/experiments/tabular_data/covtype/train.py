print("start importing...")

import time
import sys
sys.path.append('../../../')
sys.path.append('../../../SLASH/')
sys.path.append('../../../EinsumNetworks/src/')

#torch, numpy, ...
import torch


import numpy as np
import importlib

#own modules
from data_gen import test_loader, train_loader, dataList, queryList
from einsum_wrapper import EiNet
from slash import SLASH
import utils
from pathlib import Path

from rtpt import RTPT

#seeds
torch.manual_seed(42)
np.random.seed(42)
print("...done")


program ='''
tab(t1).
pred(p1).
pred(p2).

npp(covtype(1,T),[1,2,3,4,5,6,7]) :- tab(T).
forest(N,C) :- covtype(0,+T,-C), pred(N).

'''
#:- not forest(p1,1) ;  not forest(p2,3).



def slash_covtype(exp_name , exp_dict):
    
    # Create RTPT object
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name='SLASH covtype', max_iterations=int(exp_dict['epochs']))

    # Start the RTPT tracking
    rtpt.start()
    
    

    saveModelPath = 'data/'+exp_name+'/slash_models.pt'
    Path("data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)

    print("Experiment parameters:", exp_dict)


    #setup new SLASH program given the network parameters
    if exp_dict['structure'] == 'poon-domingos':
        exp_dict['depth'] = None
        exp_dict['num_repetitions'] = None
        print("using poon-domingos")

    elif exp_dict['structure'] == 'binary-trees':
        exp_dict['pd_num_pieces'] = None
        print("using binary-trees")

    
    #NETWORKS
        
    #covtype network
    cov_net = EiNet(structure = exp_dict['structure'],
        pd_num_pieces = exp_dict['pd_num_pieces'],
        depth = exp_dict['depth'],
        num_repetitions = exp_dict['num_repetitions'],
        num_var = 54,
        class_count=7,
        K = exp_dict['k'],
        num_sums= exp_dict['num_sums'],
        use_em= exp_dict['use_em'],
        pd_height=9,
        pd_width=6)

  
    
    

    #trainable params
    num_trainable_params = [sum(p.numel() for p in cov_net.parameters() if p.requires_grad)]

    num_params = [sum(p.numel() for p in cov_net.parameters())]

    
    print("training with {}({}) trainable params and {}({}) params in total".format(np.sum(num_trainable_params),num_trainable_params,np.sum(num_params),num_params))
    
     
    
    #create the SLASH Program
    nnMapping = {'covtype': cov_net}
    
    

    #OPTIMIZERS and LEARNING RATE SHEDULING

    optimizers = {'cov': torch.optim.Adam([
                                            {'params':cov_net.parameters()}],
                                            lr=exp_dict['lr'], eps=1e-7)}
  
    
    SLASHobj = SLASH(program, nnMapping, optimizers)

    

    
    #metric lists
    train_acc_list = [] #stores acc for train 
    test_acc_list = []  #and test

    startTime = time.time()
  
    
    start_e= 0
    if exp_dict['resume']:
        print("resuming experiment")
        saved_model = torch.load(saveModelPath)
        
        #load pytorch models
        cov_net.load_state_dict(saved_model['cov_net'])
      
        #optimizers and shedulers
        optimizers['cov'].load_state_dict(saved_model['resume']['optimizer_cov'])
        start_e = saved_model['resume']['epoch']
       
        #metrics
        train_acc_list = saved_model['train_acc_list']
        test_acc_list = saved_model['test_acc_list']        
        
        
    
    for e in range(start_e, exp_dict['epochs']):
        
                
        
        #TRAIN
        print('Epoch {}/{}...'.format(e+1, exp_dict['epochs']))
        time_train= time.time()
        
        
        SLASHobj.learn(dataList=dataList, queryList=queryList, method='slot', p_num = 1,
                        epoch=1, batchSize=exp_dict['bs'], use_em=exp_dict['use_em']) #smPickle='data/stableModels.pickle',
        
        
        #TEST
        time_test = time.time()

        #test accuracy
        train_acc, _, = SLASHobj.testNN('covtype', train_loader, ret_confusion=False)
        test_acc, _, = SLASHobj.testNN('covtype', test_loader, ret_confusion=False)

        print("Test Accuracy:",test_acc)
        print("Train Accuracy:",train_acc)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        timestamp_train = utils.time_delta_now(time_train)
                            

        timestamp_test = utils.time_delta_now(time_test)
        timestamp_total =  utils.time_delta_now(startTime)
        
        print('--- train time:  ---', timestamp_train)
        print('--- test time:  ---' , timestamp_test)
        print('--- total time from beginning:  ---', timestamp_total )
        time_array = [timestamp_train, timestamp_test, timestamp_total]
        
        #save the neural network  such that we can use it later
        print('Storing the trained model into {}'.format(saveModelPath))
        torch.save({"cov_net":  cov_net.state_dict(), 
                    "resume": {
                        "optimizer_cov":optimizers['cov'].state_dict(),
                        "epoch":e
                    },
                    "train_acc_list":train_acc_list,
                    "test_acc_list":test_acc_list,
                    "num_params": num_params,
                    "time": time_array,
                    "exp_dict":exp_dict,
                    "program":program}, saveModelPath)
        
        # Update the RTPT
        rtpt.step()

        



