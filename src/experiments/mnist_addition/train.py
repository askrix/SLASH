print("start importing...")

import time
import sys
sys.path.append('../../')
sys.path.append('../../SLASH/')
sys.path.append('../../EinsumNetworks/src/')


#torch, numpy, ...
import torch
#from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import torchvision

import numpy as np
import random

#own modules
from dataGen import MNIST_Addition
from einsum_wrapper import EiNet
from network_nn import Net_nn

#import slash
from slash import SLASH

import utils
from utils import set_manual_seed
from pathlib import Path
from rtpt import RTPT

print("...done")



def slash_mnist_addition(exp_name , exp_dict):
    
    # Set the seeds for PRNG
    set_manual_seed(exp_dict['seed'])

    # Create RTPT object
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name='SLASH MNIST Addition', max_iterations=int(exp_dict['epochs']))

    # Start the RTPT tracking
    rtpt.start()
    
    program = '''
    img(i1). img(i2).
    addition(A,B,N):- digit(0,+A,-N1), digit(0,+B,-N2), N=N1+N2.
    npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
    '''
    
    
    saveModelPath = 'data/'+exp_name+'/slash_digit_addition_models.pt'
    Path("data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)

    print("Experiment parameters:", exp_dict)
    
    #use neural net or probabilisitc circuit
    if exp_dict['use_spn']:
    
        #setup new SLASH program given the network parameters
        if exp_dict['structure'] == 'poon-domingos':
            m = EiNet(structure = exp_dict['structure'],
                      pd_num_pieces = exp_dict['pd_num_pieces'],
                      use_em = False,
                      num_var = 784,
                      class_count = 10,
                      pd_width = 28,
                      pd_height = 28,
                      learn_prior = exp_dict['learn_prior'])
        else: 
            m = EiNet(structure = exp_dict['structure'],
                      depth = exp_dict['depth'],
                      num_repetitions = exp_dict['num_repetitions'],
                      use_em = False,
                      num_var = 784,
                      class_count = 10,
                      learn_prior = exp_dict['learn_prior'])
    else:
        m = Net_nn()    

         

    
    num_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in m.parameters())
    print("training with {} trainable params and {} params in total".format(num_trainable_params,num_params))
            
        
    #create the SLASH Program
    nnMapping = {'digit': m}
    optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=exp_dict['lr'], eps=1e-7)}
    SLASHobj = SLASH(program, nnMapping, optimizers)


    #metric lists
    train_accuracy_list = []
    test_accuracy_list = []
    confusion_matrix_list = []
    loss_list = []
    startTime = time.time()
    
    
    #load data
    #if we are using spns we need to flatten the data(Tensor has form [bs, 784])
    if exp_dict['use_spn']: 
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, )), transforms.Lambda(lambda x: torch.flatten(x))])
    #if not we can keep the dimensions(Tensor has form [bs,28,28])
    else: 
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, ))])


    mnist_addition_dataset = MNIST_Addition(torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform), 'data/labels/train_data.txt', exp_dict['use_spn'])
    train_dataset_loader = torch.utils.data.DataLoader(mnist_addition_dataset, shuffle=True,batch_size=exp_dict['bs'],pin_memory=True, num_workers=8)
    
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, transform=transform), batch_size=100, shuffle=True)
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, transform=transform), batch_size=100, shuffle=True)


    # if the training with missing data is enabled
    marg_masks = []
    if exp_dict['drop_out'] < 0.0 or exp_dict['drop_out'] >= 1.0:
        raise Exception("%r not in range [0.0, 1.0)" % (exp_dict['drop_out'],))
    elif exp_dict['drop_out'] > 0.0:
        marg_masks = []
        train_N = len(train_dataset_loader)
        idx_batches = torch.tensor(range(train_N), dtype=torch.int).split(exp_dict['bs'])
        # iterate over batches and generate a marginalization mask for each according to the requested drop-out rate
        for idx in idx_batches:
            marg_idx = random.sample(range(0, m.num_var-1), round(m.num_var*exp_dict['drop_out']))
            marg_idx.sort()
            marg_masks.append(marg_idx)
    else:
        marg_masks = None


    start_e= 0
    if exp_dict['resume']:
        print("resuming experiment")
        saved_model = torch.load(saveModelPath)
        
        #load pytorch models
        m.load_state_dict(saved_model['addition_net'])

        #optimizers and shedulers
        optimizers['digit'].load_state_dict(saved_model['resume']['optimizer_digit'])
        start_e = saved_model['resume']['epoch']
                
        #metrics
        test_accuracy_list = saved_model['test_accuracy_list']
        train_accuracy_list = saved_model['train_accuracy_list']
        confusion_matrix_list = saved_model['confusion_matrix_list']
        loss_list = saved_model['loss']
    

    # Evaluate the performanve directly after initialisation
    time_test = time.time()
    test_acc, _, confusion_matrix = SLASHobj.testNetwork('digit', test_loader, ret_confusion=True)
    train_acc, _ = SLASHobj.testNetwork('digit', train_loader)
    confusion_matrix_list.append(confusion_matrix)
    train_accuracy_list.append([train_acc,0])
    test_accuracy_list.append([test_acc, 0])
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
    torch.save({"addition_net": m.state_dict(),
                "test_accuracy_list": test_accuracy_list,
                "train_accuracy_list":train_accuracy_list,
                "confusion_matrix_list":confusion_matrix_list,
                "num_params": num_trainable_params,
                "exp_dict":exp_dict,
                "exp_name":exp_name,
                "time": time_array,
                "program":program}, saveModelPath)
    
    # Train and evaluate the performance
    for e in range(start_e, exp_dict['epochs']):
        print('Epoch {}...'.format(e+1))

        time_train= time.time()
        loss = SLASHobj.learn(dataset_loader = train_dataset_loader,
                       epoch=1, batchSize=exp_dict['bs'],
                       p_num=exp_dict['p_num'], marginalisation_masks=marg_masks)
        timestamp_train = utils.time_delta_now(time_train, simple_format=True)

        time_test = time.time()
        test_acc, _, confusion_matrix = SLASHobj.testNetwork('digit', test_loader, ret_confusion=True)
        confusion_matrix_list.append(confusion_matrix)
        train_acc, _ = SLASHobj.testNetwork('digit', train_loader)        
        train_accuracy_list.append([train_acc,e])
        test_accuracy_list.append([test_acc, e])
        timestamp_test = utils.time_delta_now(time_test, simple_format=True)
        timestamp_total = utils.time_delta_now(startTime, simple_format=True)
        loss_list.append(loss)
        time_array = [timestamp_train, timestamp_test, timestamp_total]

        # Save and print statistics
        print('Train Acc: {:0.2f}%, Test Acc: {:0.2f}%'.format(train_acc, test_acc))
        print('--- train time:  ---', timestamp_train)
        print('--- test time:  ---' , timestamp_test)
        print('--- total time from beginning:  ---', timestamp_total)
        
        # Export results and networks
        print('Storing the trained model into {}'.format(saveModelPath))
        torch.save({"addition_net": m.state_dict(),
                    "resume": {
                        "optimizer_digit":optimizers['digit'].state_dict(),
                        "epoch":e
                            },
                    "test_accuracy_list": test_accuracy_list,
                    "train_accuracy_list":train_accuracy_list,
                    "confusion_matrix_list":confusion_matrix_list,
                    "num_params": num_trainable_params,
                    "exp_dict":exp_dict,
                    "exp_name":exp_name,
                    "time": time_array,
                    "loss": loss_list,
                    "program":program}, saveModelPath)
        
        # Update the RTPT
        rtpt.step(subtitle=f"accuracy={test_acc:2.2f}")
