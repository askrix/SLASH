print("start importing...")

import time
import sys
sys.path.append('../../')
sys.path.append('../../SLASH/')
sys.path.append('../../EinsumNetworks/src/')

import os
#torch, numpy, ...
import torch
#from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import torchvision

import numpy as np
import random
import importlib
from tqdm import tqdm


#own modules
from dataGen import MNIST_Addition
from einsum_wrapper import EiNet
from EinsumNetwork import EinsumNetwork

from EinsumNetwork.SumLayer import SumLayer
from EinsumNetwork.FactorizedLeafLayer import FactorizedLeafLayer
#import slash
from slash import SLASH

import utils
from utils import set_manual_seed
from pathlib import Path
from rtpt import RTPT

print("...done")



def complient_with_em(einet):
    """Normalize model parameters after the gradient descent step."""
    with torch.no_grad():
        for layer in einet.einet_layers:
            if issubclass(type(layer), SumLayer):
                # clamp into [0, inf)
                layer.params.data = torch.clamp(layer.params, 1e-16)
                if layer.params_mask is not None:
                    layer.params.data *= layer.params_mask

                # normalize
                layer.params.data = layer.params / (
                    layer.params.sum(layer.normalization_dims, keepdim=True)
                )
                layer.params.grad = None

            if issubclass(type(layer), FactorizedLeafLayer):
                # project back
                layer.ef_array.params.data = layer.ef_array.project_params(
                    layer.ef_array.params.data
                )


def complient_with_gd(einet):
    """Project model's parameters after the EM step."""
    with torch.no_grad():
        for layer in einet.einet_layers:
            if issubclass(type(layer), SumLayer):
                # clamping is not necessary
                if layer.params_mask is not None:
                    layer.params.data *= layer.params_mask
                # normalize
                layer.params.data = layer.params.log()
                layer.params.grad = None


def einet_set_use_em(einet, use_em):
    """Set 'use_em' for all layers in the EiNet object."""
    einet.einet_layers[0].ef_array._use_em = use_em
    for layer in einet.einet_layers:
        layer._use_em = use_em


def slash_mnist_addition(exp_name , exp_dict):
    
    # Set the seeds for PRNG
    set_manual_seed(exp_dict['seed'])

    # Create RTPT object
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name='SLASH MNIST Addition', max_iterations=int(exp_dict['epochs']))

    # Start the RTPT tracking
    rtpt.start()
    
    program = '''
    img(i1). img(i2).
    addition(A,B,N):- digit(0,+A,-N1), digit(0,+B,+N2), A != B, N=N1+N2.
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
                      #K = 40,
                      #num_sums = 40,
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
   

         

    
    num_trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in m.parameters())
    print("training with {} trainable params and {} params in total".format(num_trainable_params,num_params))
            
        
    #create the SLASH Program
    nnMapping = {'digit': m}
    optimizers = {'digit': torch.optim.Adam(m.parameters(), lr=exp_dict['lr'], eps=1e-7)}
    SLASHobj = SLASH(program, nnMapping, optimizers)

    ll_optimizer = torch.optim.Adam(m.parameters(), lr=0.01, eps=1e-7)

    #metric lists
    train_accuracy_list = []
    test_accuracy_list = []
    confusion_matrix_list = []
    startTime = time.time()
    
    
    #load data
    #if we are using spns we need to flatten the data(Tensor has form [bs, 784])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081, )), transforms.Lambda(lambda x: torch.flatten(x))])

    mnist_addition_dataset = MNIST_Addition(torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform), 'data/labels/train_data.txt', exp_dict['use_spn'])
    train_dataset_loader = torch.utils.data.DataLoader(mnist_addition_dataset, shuffle=True,batch_size=exp_dict['bs'],pin_memory=False, num_workers=8)
    
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, transform=transform), batch_size=100, shuffle=True)
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, transform=transform), batch_size=100, shuffle=True)


    
    # Evaluate the performanve directly after initialisation
    time_test = time.time()
    test_acc, _, confusion_matrix = SLASHobj.testNetwork('digit', test_loader, ret_confusion=True)
    confusion_matrix_list.append(confusion_matrix)
    train_acc, _ = SLASHobj.testNetwork('digit', train_loader)
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

    samples_dir = 'samples/'+exp_name+ "/"
    utils.mkdir_p(samples_dir)
    train_N = 30000
    batch_size = 100

    warmup = 5
    complient_with_em(m)
    einet_set_use_em(m,True)
    m.train()

    def evaluate_ll(message):
        m.eval()
        train_ll = 0
        for data_batch, _ in train_dataset_loader:
            batch_x = data_batch['i1'].to(device='cuda')
            train_ll += EinsumNetwork.eval_loglikelihood_batched(m, batch_x, batch_size=batch_size)
        print(("[{}]   train LL {} "+message).format(e,train_ll / train_N))
        m.train()

    # Train and evaluate the performance
    for e in range(exp_dict['epochs']):
        print('Epoch {}...'.format(e+1))

        
        ##### evaluate
        evaluate_ll("before")

        #maximize log likelihhods using EM 
        for i in range(0,warmup):
            for data_batch, _ in tqdm(train_dataset_loader):

                ll_optimizer.zero_grad()
                batch_x = data_batch['i1'].to(device='cuda')
                outputs = m.get_log_likelihoods(batch_x) #shape [bs,1]

                ll_sample = EinsumNetwork.log_likelihoods(outputs) #shape [bs,1]
                log_likelihood = ll_sample.sum()
                log_likelihood.backward()

                m.em_process_batch()
            m.em_update()

            ##### evaluate
            evaluate_ll("after LL max")


        #finish EM warmup
        warmup = 1

        #sample after EM step
        samples = m.sample(num_samples=25).cpu().numpy()
        samples = samples.reshape((-1, 28, 28))
        file_name = "samples_ll"+str(e)+".png" 
        utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, file_name), margin_gray_val=0.)

        saveModelPath = 'data/'+exp_name+'/slash_digit_addition_models_generative_e'+str(e)+'.pt'
        Path("data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)
        
        # Export results and networks after generative training
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
        

        
        #swap EiNet parameter domain for gradient descent
        m.train()
        complient_with_gd(m)
        einet_set_use_em(m,False)

        #SLASH training 
        time_train= time.time()
        SLASHobj.learn(dataset_loader = train_dataset_loader,
                       epoch=1, batchSize=exp_dict['bs'],
                       p_num=exp_dict['p_num'])
        timestamp_train = utils.time_delta_now(time_train, simple_format=True)

        time_test = time.time()
        test_acc, _, confusion_matrix = SLASHobj.testNetwork('digit', test_loader, ret_confusion=True)
        confusion_matrix_list.append(confusion_matrix)
        train_acc, _ = SLASHobj.testNetwork('digit', train_loader)        
        train_accuracy_list.append([train_acc,e])
        test_accuracy_list.append([test_acc, e])
        timestamp_test = utils.time_delta_now(time_test, simple_format=True)
        timestamp_total = utils.time_delta_now(startTime, simple_format=True)
        time_array = [timestamp_train, timestamp_test, timestamp_total]


        #sample after SLASH step
        samples = m.sample(num_samples=25).cpu().numpy()
        samples = samples.reshape((-1, 28, 28))
        file_name = "samples_slash"+str(e)+".png" 
        utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, file_name), margin_gray_val=0.)

        #for each class
        for i in range(0,10):
            samples = m.sample(num_samples=25, class_idx=i).cpu().numpy()
            samples = samples.reshape((-1, 28, 28))
            
            utils.mkdir_p(samples_dir+str(i)+"/")
            file_name = str(i)+"/samples_slash_c"+str(i)+"_e"+str(e)+".png"
            utils.save_image_stack(samples, 5, 5, os.path.join(samples_dir, file_name), margin_gray_val=0.)


        #swap EiNet parameter domain back to EM and evaluate again
        complient_with_em(m)
        einet_set_use_em(m,True)
        evaluate_ll("after SLASH training")
        m.train()


        # Save and print statistics
        print('Train Acc: {:0.2f}%, Test Acc: {:0.2f}%'.format(train_acc, test_acc))
        print('--- train time:  ---', timestamp_train)
        print('--- test time:  ---' , timestamp_test)
        print('--- total time from beginning:  ---', timestamp_total)

        saveModelPath = 'data/'+exp_name+'/slash_digit_addition_models_slash_e'+str(e)+'.pt'
        Path("data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)
        
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
        
        # Update the RTPT
        rtpt.step(subtitle=f"accuracy={test_acc:2.2f}")
