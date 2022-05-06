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
from dataGen import get_data_and_object_list, SHAPEWORLD4
from einsum_wrapper import EiNet
from slash import SLASH
import utils
import ap_utils
from utils import set_manual_seed
from slot_attention_module import SlotAttention_model, NPP_SlotAttention_classifier
from pathlib import Path

from rtpt import RTPT
print("...done")


program ='''
slot(s1).
slot(s2).
slot(s3).
slot(s4).
name(o1).
name(o2).
name(o3).
name(o4).

%assign each name a slot
{slot_name_comb(N,X): slot(X) }=1 :- name(N). %problem we have dublicated slots
%remove each model which has multiple slots asigned to the same name
:-  slot_name_comb(N1,X1), slot_name_comb(N2,X2), X1 == X2, N1 != N2.

%build the object ontop of the slot assignment
object(N,C,S,H,Z) :- color(0, +X, -C), shape(0, +X, -S), shade(0, +X, -H), size(0, +X, -Z), slot(X), name(N), slot_name_comb(N,X).

npp(shade(1,X),[bright, dark, bg]) :- slot(X).
npp(color(1,X),[red, blue, green, gray, brown, magenta, cyan, yellow, black]) :- slot(X).
npp(shape(1,X),[circle, triangle, square, bg]) :- slot(X).
npp(size(1,X),[small, big, bg]) :- slot(X).
'''



def slash_slot_attention(exp_name , exp_dict):
    
    # Set the seeds for PRNG
    set_manual_seed(exp_dict['seed'])
    
    # Create RTPT object
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name='SLASH Shapeworld4 ablation4', max_iterations=int(exp_dict['epochs']))

    # Start the RTPT tracking
    rtpt.start()
    
    

    saveModelPath = 'data/'+exp_name+'/slash_slot_models.pt'
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
        
    #NETWORKS
        
    # color network
    color_net = NPP_SlotAttention_classifier(in_channels=32, out_channels=9)

    # shape network
    shape_net = NPP_SlotAttention_classifier(in_channels=32, out_channels=4)

    # shade network
    shade_net = NPP_SlotAttention_classifier(in_channels=32, out_channels=3)

    # size network
    size_net = NPP_SlotAttention_classifier(in_channels=32, out_channels=3)
    

    
    #create the Slot Attention network
    slot_net = SlotAttention_model(n_slots=4, n_iters=3, n_attr=18,
                                encoder_hidden_channels=32, attention_hidden_channels=64,
                                decoder_hidden_channels=32, decoder_initial_size=(32, 32))# was 32*32 and 128
    slot_net = slot_net.to(device='cuda')
    
    
    
    

    #trainable params
    num_trainable_params = [sum(p.numel() for p in color_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in shape_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in shade_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in size_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in slot_net.parameters() if p.requires_grad)]
    num_params = [sum(p.numel() for p in color_net.parameters()), 
                  sum(p.numel() for p in shape_net.parameters()),
                  sum(p.numel() for p in shade_net.parameters()),
                  sum(p.numel() for p in size_net.parameters()),
                  sum(p.numel() for p in slot_net.parameters())]
    
    print("training with {}({}) trainable params and {}({}) params in total".format(np.sum(num_trainable_params),num_trainable_params,np.sum(num_params),num_params))
    
     
            
    slot_net_params = list(slot_net.parameters())
    csss_params = list(color_net.parameters()) + list(shape_net.parameters()) + list(shade_net.parameters()) + list(size_net.parameters())
    
    #create the SLASH Program
    nnMapping = {'color': color_net,
                 'shape':shape_net,
                 'shade':shade_net,
                 'size':size_net}
    
    

    #OPTIMIZERS and LEARNING RATE SHEDULING
    optimizers = {'csss': torch.optim.Adam([
                                            {'params':csss_params}],
                                            lr=0.0004, eps=1e-7),
                 'slot': torch.optim.Adam([
                                            {'params': slot_net_params}],
                                            lr=0.0004, eps=1e-7)}

    
    SLASHobj = SLASH(program, nnMapping, optimizers)

    
    
    
    print("using learning rate warmup")
    warmup_epochs = exp_dict['lr_warmup_steps'] #warmup for x epochs
    decay_epochs = exp_dict['lr_decay_steps']
    epochs = exp_dict['epochs'] 
    slot_base_lr = 0.0004

    
        #only affects slot attention lr
        #cheduler_steplr_slot = CosineAnnealingLR(optimizers['slot'], T_max =cosine_steps,eta_min=0.00005)
        #scheduler_warmup_slot = GradualWarmupScheduler(optimizers['slot'], multiplier=1, total_epoch=warmup_steps, after_scheduler=scheduler_steplr_slot)
        #schedulers = {'slot': scheduler_warmup_slot}
    

    
    #metric lists
    test_ll_list = [] #stores log likelihoods
    test_ap_list = [] #stores average precsion values
    test_metric_list = [] #stores tp, fp, tn values
    lr_list = [] # store learning rate
    loss_list = []  # store training loss
    startTime = time.time()
  
    
    print("loading data...")
    # train the network and evaluate the performance
    dataListTrain, queryListTrain, objListTrain= get_data_and_object_list(SHAPEWORLD4('../../data/shapeworld4/',"train"), 1, shuffle=True)
    dataListTest, queryListTest, objListTest  = get_data_and_object_list(SHAPEWORLD4('../../data/shapeworld4/',"val"), 1, shuffle=True)
    print("...done")

    
    start_e= 0
    if exp_dict['resume']:
        print("resuming experiment")
        saved_model = torch.load(saveModelPath)
        
        #load pytorch models
        color_net.load_state_dict(saved_model['color_net'])
        shape_net.load_state_dict(saved_model['shape_net'])
        shade_net.load_state_dict(saved_model['shade_net'])
        size_net.load_state_dict(saved_model['size_net'])
        slot_net.load_state_dict(saved_model['slot_net'])
        
        
        #optimizers and shedulers
        optimizers['csss'].load_state_dict(saved_model['resume']['optimizer_csss'])
        optimizers['slot'].load_state_dict(saved_model['resume']['optimizer_slot'])
        start_e = saved_model['resume']['epoch']
        
        
        #metrics
        test_ll_list = saved_model['test_ll_list']
        test_ap_list = saved_model['test_ap_list']
        test_metric_list = saved_model['test_metric_list']
        lr_list = saved_model['lr_list']
        loss_list = saved_model['loss_list']
        

    for e in range(start_e, exp_dict['epochs']):
        
        
        #we have three datasets right now train, val and test with 20k, 5k and 100 samples
        
        
        #TRAIN
        print('Epoch {}/{}...'.format(e+1, exp_dict['epochs']))
        time_train= time.time()
        
        #apply lr schedulers
        if e < warmup_epochs:
            lr = slot_base_lr * ((e+1)/warmup_epochs)
        else:
            lr = slot_base_lr
        lr = lr * 0.5**((e+1)/decay_epochs)
        optimizers['slot'].param_groups[0]['lr'] = lr
        optimizers['csss'].param_groups[0]['lr'] = lr
        lr_list.append([lr,e])


        print("LR:", "{:.6f}".format(lr), optimizers['slot'].param_groups[0]['lr'])
        
        loss = SLASHobj.learn(dataList=dataListTrain, queryList=queryListTrain, slot_net=slot_net, method='slot', p_num = 8,
                        epoch=1, batchSize=exp_dict['bs'], train_slot= True) #smPickle='data/stableModels.pickle',
        

        loss_list.append(loss)

        timestamp_train = utils.time_delta_now(time_train)
        
        
        #TEST
        time_test = time.time()

        #forward test batch
        inference, ll = SLASHobj.forward_slot_attention_pipeline(slot_net=slot_net, data_batch= dataListTest)

        #collect log likelihoods summed for test data and append it to the list
        ll_dict= {'color':0, 'shape':0, 'shade':0, 'size':0}
        for slot in ll:
            for prop in ll[slot]:
                ll_dict[prop] += torch.sum(ll[slot][prop])
        test_ll_list.append( [torch.Tensor(list(ll_dict.values())), e]) #color, shape
                                  
                            
        #compute the average precision, tp, fp, tn for color+shape+shade+size, color, shape, shade, size
        #color+shape+shade+size
        pred = ap_utils.inference_map_to_array(inference).cpu().numpy()
        ap, true_positives,false_positives, true_negatives, correctly_classified = ap_utils.average_precision(pred, objListTest,-1, "SHAPEWORLD4")
        print("avg precision ",ap, "tp", true_positives, "fp", false_positives, "tn", true_negatives, "correctly classified",correctly_classified)
            
        
        #color
        pred_c = ap_utils.inference_map_to_array(inference, only_color=True).cpu().numpy()
        ap_c, true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c= ap_utils.average_precision(pred, objListTest,-1, "SHAPEWORLD4", only_color = True)
        print("avg precision color",ap_c, "tp", true_positives_c, "fp", false_positives_c, "tn", true_negatives_c, "correctly classified",correctly_classified_c)

        #shape              
        pred_s = ap_utils.inference_map_to_array(inference, only_shape=True).cpu().numpy()
        ap_s, true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s= ap_utils.average_precision(pred_s, objListTest,-1, "SHAPEWORLD4", only_shape = True)
        print("avg precision shape",ap_s, "tp", true_positives_s, "fp", false_positives_s, "tn", true_negatives_s, "correctly classified",correctly_classified_s)
        
        #shade              
        pred_h = ap_utils.inference_map_to_array(inference, only_shade=True).cpu().numpy()
        ap_h, true_positives_h, false_positives_h, true_negatives_h, correctly_classified_h= ap_utils.average_precision(pred_h, objListTest,-1, "SHAPEWORLD4", only_shade = True)
        print("avg precision shade",ap_h, "tp", true_positives_h, "fp", false_positives_h, "tn", true_negatives_h, "correctly classified",correctly_classified_h)
        
        #shape              
        pred_x = ap_utils.inference_map_to_array(inference, only_size=True).cpu().numpy()
        ap_x, true_positives_x, false_positives_x, true_negatives_x, correctly_classified_x= ap_utils.average_precision(pred_x, objListTest,-1, "SHAPEWORLD4", only_size = True)
        print("avg precision size",ap_x, "tp", true_positives_x, "fp", false_positives_x, "tn", true_negatives_x, "correctly classified",correctly_classified_x)

        
        #store ap, tp, fp, tn
        test_ap_list.append([ap, ap_c, ap_s, ap_h, ap_x, e])                    
        test_metric_list.append([true_positives, false_positives, true_negatives,correctly_classified,
                                 true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c,
                                 true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s,
                                 true_positives_h, false_positives_h, true_negatives_h, correctly_classified_h,
                                 true_positives_x, false_positives_x, true_negatives_x, correctly_classified_x])
                            
        
        #store learning rates

        timestamp_test = utils.time_delta_now(time_test)
        timestamp_total =  utils.time_delta_now(startTime)
        
        print('--- train time:  ---', timestamp_train)
        print('--- test time:  ---' , timestamp_test)
        print('--- total time from beginning:  ---', timestamp_total )
        time_array = [timestamp_train, timestamp_test, timestamp_total]
        
        #save the neural network  such that we can use it later
        print('Storing the trained model into {}'.format(saveModelPath))
        torch.save({"shape_net":  shape_net.state_dict(), 
                    "color_net": color_net.state_dict(),
                    "shade_net": shade_net.state_dict(),
                    "size_net": size_net.state_dict(),
                    "slot_net": slot_net.state_dict(),
                    "resume": {
                        "optimizer_csss":optimizers['csss'].state_dict(),
                        "optimizer_slot": optimizers['slot'].state_dict(),
                        "epoch":e
                    },
                    "test_ll_list": test_ll_list,
                    "test_ap_list":test_ap_list,
                    "loss_list":loss_list,
                    "test_metric_list":test_metric_list,
                    "lr_list": lr_list,
                    "num_params": num_params,
                    "time": time_array,
                    "exp_dict":exp_dict,
                    "program":program}, saveModelPath)
        
        # Update the RTPT
        rtpt.step(subtitle=f"ap={ap:2.2f}")

        



