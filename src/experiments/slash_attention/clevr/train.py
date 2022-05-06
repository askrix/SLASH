print("start importing...")

import time
import sys
sys.path.append('../../../')
sys.path.append('../../../SLASH/')
sys.path.append('../../../EinsumNetworks/src/')

#torch, numpy, ...
import torch
from torch.utils.tensorboard import SummaryWriter
torch.cuda.empty_cache()


import numpy as np
import importlib

#own modules
from dataGen import CLEVR
from auxiliary import get_files_names_and_paths, get_slash_program
from einsum_wrapper import EiNet
from slash import SLASH
import utils
import ap_utils
from utils import set_manual_seed
from slot_attention_module import SlotAttention_model
from slot_attention_module import SlotAttention_model
from pathlib import Path
from rtpt import RTPT
print("...done")



def slash_slot_attention(exp_name , exp_dict):
    
    saveModelPath = f"data/{exp_name}/SLASH_Attention_CLEVR{exp_dict['obj_num']}.pt"
    Path("data/"+exp_name+"/").mkdir(parents=True, exist_ok=True)

    print("Experiment parameters:", exp_dict)
    
    # Set the seeds for PRNG
    set_manual_seed(exp_dict['seed'])

    #setup new SLASH program given the network parameters
    program = get_slash_program(exp_dict['obj_num'])
    
    if exp_dict['structure'] == 'poon-domingos':
        print("using poon-domingos")
              
        #size network
        size_net = EiNet(structure = exp_dict['structure'],
            pd_num_pieces = exp_dict['pd_num_pieces'],
            num_var = 64,
            pd_width = 8,
            pd_height = 8,
            class_count = 3,
            use_em = exp_dict['use_em'],
            learn_prior = exp_dict['learn_prior'])
        
        #material network
        material_net = EiNet(structure = exp_dict['structure'],
            pd_num_pieces = exp_dict['pd_num_pieces'],
            num_var = 64,
            pd_width = 8,
            pd_height = 8,
            class_count = 3,
            use_em = exp_dict['use_em'],
            learn_prior = exp_dict['learn_prior'])
        
        #shape network
        shape_net = EiNet(structure = exp_dict['structure'],
            pd_num_pieces = exp_dict['pd_num_pieces'],
            num_var = 64,
            pd_width = 8,
            pd_height = 8,
            class_count = 4,
            use_em = exp_dict['use_em'],
            learn_prior = exp_dict['learn_prior'])
        
        #color network
        color_net = EiNet(structure = exp_dict['structure'],
            pd_num_pieces = exp_dict['pd_num_pieces'],
            num_var = 64,
            pd_width = 8,
            pd_height = 8,
            class_count = 9,
            use_em = exp_dict['use_em'],
            learn_prior = exp_dict['learn_prior'])
        
    elif exp_dict['structure']== 'binary-trees':
        print("using binary-trees")
                      
        #size network
        size_net = EiNet(structure = exp_dict['structure'],
            depth = exp_dict['depth'],
            num_repetitions = exp_dict['num_repetitions'],
            num_var = 64,
            class_count = 3,
            use_em = exp_dict['use_em'],
            learn_prior = exp_dict['learn_prior'])
        
        #materil network
        material_net = EiNet(structure = exp_dict['structure'],
            depth = exp_dict['depth'],
            num_repetitions = exp_dict['num_repetitions'],
            num_var = 64,
            class_count = 3,
            use_em = exp_dict['use_em'],
            learn_prior = exp_dict['learn_prior'])
        
        #shape network
        shape_net = EiNet(structure = exp_dict['structure'],
            depth = exp_dict['depth'],
            num_repetitions = exp_dict['num_repetitions'],
            num_var = 64,
            class_count = 4,
            use_em = exp_dict['use_em'],
            learn_prior = exp_dict['learn_prior'])
        
        #color network
        color_net = EiNet(structure = exp_dict['structure'],
            depth = exp_dict['depth'],
            num_repetitions = exp_dict['num_repetitions'],
            num_var = 64,
            class_count = 9,
            use_em = exp_dict['use_em'],
            learn_prior = exp_dict['learn_prior'])

    
    #create the Slot Attention network
    slot_net = SlotAttention_model(n_slots=exp_dict['obj_num'], n_iters=3, n_attr=18,
                                   encoder_hidden_channels=64, attention_hidden_channels=128)
        
    if exp_dict['pretrained_slot_module']:
        model_path = "data/slot-attention-shapeworld2-objdiscovery-pretrain-0" #load the pretrained model
        pretrained_model = torch.load(model_path)['weights']
        print("preloading slot module")

        #print(pretrained_model)
        keys = list(pretrained_model.keys())

        #replace module. with withespaces
        for key in keys:
            new_key = key.replace('module.','')
            #print(new_key," ", key)
            pretrained_model[new_key] = pretrained_model.pop(key)

        #load the pretrained model
        slot_net.load_state_dict(pretrained_model, strict=True)
    slot_net = slot_net.to(device='cuda')
        
    
    if exp_dict['debug']:
        #preload the network
        print("preloading the color and shape networks")
        MODEL = "most_prob_2"
        #data/one_to_one_pretrained_test_2/spn_slot_models.pt
        class_count = 4

        loaded_models_path = "data/"+MODEL+"/spn_slot_models.pt" 
        loaded_models = torch.load(loaded_models_path)
        print(loaded_models.keys())
        
        s_net.load_state_dict(loaded_models['s_net'])
        c_net.load_state_dict(loaded_models['c_net'])
    

    #trainable params
    num_trainable_params = [sum(p.numel() for p in size_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in material_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in shape_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in color_net.parameters() if p.requires_grad),
                            sum(p.numel() for p in slot_net.parameters() if p.requires_grad)]
    
    num_params = [sum(p.numel() for p in size_net.parameters()), 
                  sum(p.numel() for p in material_net.parameters()), 
                  sum(p.numel() for p in shape_net.parameters()), 
                  sum(p.numel() for p in color_net.parameters()),
                  sum(p.numel() for p in slot_net.parameters())]
    
    print("training with {}({}) trainable params and {}({}) params in total".format(np.sum(num_trainable_params),num_trainable_params,np.sum(num_params),num_params))
         
            
    slot_net_params = list(slot_net.parameters())
    #s_params = list(s_net.parameters()) #+ list(slot_net.parameters()) #dont train slot attention now
    #c_params = list(c_net.parameters()) #+ list(slot_net.parameters())
    smsc_params = list(size_net.parameters()) + list(material_net.parameters()) + list(shape_net.parameters())  + list(color_net.parameters()) 
    
    #create the SLASH Program
    nnMapping = {'size': size_net,
                 'material': material_net,
                 'shape': shape_net,
                 'color': color_net
                }
        
    
    if exp_dict['train_slot']:    
        print("training EiNets and SlotAttention module")
        optimizers = {'smsc': torch.optim.Adam([
                                                {'params':smsc_params}],
                                                lr=exp_dict['lr'], eps=1e-7),
                      'slot': torch.optim.Adam([
                                                {'params': slot_net_params}],
                                                lr=0.0004, eps=1e-7)}
    else: 
        print("training EiNets without SlotAttention module")
        optimizers = {'smsc': torch.optim.Adam([{'params':smsc_params}],
                                                lr=exp_dict['lr'], eps=1e-7)}
    
    SLASHobj = SLASH(program, nnMapping, optimizers)

    

    print("using learning rate warmup")
    warmup_epochs = exp_dict['lr_warmup_steps'] #warmup for x epochs
    decay_epochs = exp_dict['lr_decay_steps']
    epochs = exp_dict['epochs'] 
    slot_base_lr = 0.0004
        
        
    #metric lists
    test_ll_list = [] #stores log likelihoods
    test_ap_list = [] #stores average precsion values
    test_metric_list = [] #stores tp, fp, tn values
    lr_list = [] # store learning rate
    loss_list = []  # store training loss
    startTime = time.time()
  
    
    # Load data
    if exp_dict['obj_num'] == 4 or exp_dict['obj_num'] == 6:
        obj_num = exp_dict['obj_num']
        root = '/SLASH/data/CLEVR_v1.0/'
        mode = 'train'
        img_paths, files_names = get_files_names_and_paths(root=root, mode=mode, obj_num=obj_num)
        dataListTrain, obsListTrain, objListTrain = get_data_and_object_list(CLEVR(root,mode,img_paths,files_names,obj_num), 1, shuffle=True)
        mode = 'val'
        img_paths, files_names = get_files_names_and_paths(root=root, mode=mode, obj_num=obj_num)
        dataListTest, obsListTest, objListTest = get_data_and_object_list(CLEVR(root,mode,img_paths,files_names,obj_num), 1, shuffle=True)
    else:
        dataListTrain, obsListTrain, objListTrain = get_data_and_object_list(CLEVR(root,mode), 1, shuffle=True)
        dataListTest, obsListTest, objListTest = get_data_and_object_list(CLEVR(root,mode), 1, shuffle=True)
    print("loaded data")
    
    
    # Resume the training if requested
    start_e= 0
    if exp_dict['resume']:
        print("resuming experiment")
        saved_model = torch.load(saveModelPath)
        
        #load pytorch models
        color_net.load_state_dict(saved_model['color_net'])
        shape_net.load_state_dict(saved_model['shape_net'])
        material_net.load_state_dict(saved_model['material_net'])
        size_net.load_state_dict(saved_model['size_net'])
        slot_net.load_state_dict(saved_model['slot_net'])
        
        
        #optimizers and shedulers
        optimizers['smsc'].load_state_dict(saved_model['resume']['optimizer_smsc'])
        optimizers['slot'].load_state_dict(saved_model['resume']['optimizer_slot'])
        start_e = saved_model['resume']['epoch']
    
        #metrics
        test_ll_list = saved_model['test_ll_list']
        test_ap_list = saved_model['test_ap_list']
        test_metric_list = saved_model['test_metric_list']
        lr_list = saved_model['lr_list']
        loss_list = saved_model['loss_list']
        
    
    # Create RTPT object
    rtpt = RTPT(name_initials=exp_dict['credentials'], experiment_name=f'SLASH Attention CLEVR %s' % exp_dict['obj_num'], max_iterations=int(exp_dict['epochs'] - start_e))
    
    # Start the RTPT tracking
    rtpt.start()
    

    # train the network and evaluate the performance
    for e in range(start_e, exp_dict['epochs']):
        #we have three datasets right now train, val and test with 20k, 5k and 100 samples
                
        #TRAIN
        print('Epoch {}/{}...'.format(e+1, exp_dict['epochs']))
        time_train = time.time()
        
        #apply lr schedulers
        if e < warmup_epochs:
            lr = slot_base_lr * ((e+1)/warmup_epochs)
        else:
            lr = slot_base_lr
        lr = lr * 0.5**((e+1)/decay_epochs)
        optimizers['slot'].param_groups[0]['lr'] = lr
        print("LR", lr)



        print("QueryList",len(obsListTrain))
        loss = SLASHobj.learn(dataList=dataListTrain, queryList=obsListTrain, slot_net=slot_net, method='slot', p_num=exp_dict['p_num'],
                              epoch=1, batchSize=exp_dict['bs'], use_em=exp_dict['use_em'], train_slot=exp_dict['train_slot'])
        
        #store loss
        loss_list.append(loss)
        
        timestamp_train = utils.time_delta_now(time_train)
                
        #TEST
        time_test = time.time()

        #forward test batch
        inference, ll = SLASHobj.forward_slot_attention_pipeline(slot_net=slot_net, data_batch=dataListTest)

        #collect log likelihoods summed for test data and append it to the list
        ll_dict= {'size':0 ,'material':0, 'shape':0 , 'color':0}
        for slot in ll:
            for prop in ll[slot]:
                ll_dict[prop] += torch.sum(ll[slot][prop])
        test_ll_list.append( [torch.Tensor(list(ll_dict.values())), e]) #color, shape
                                    
        #compute the average precision, tp, fp, tn for color+shape+material+size
        pred = ap_utils.inference_map_to_array(inference).cpu().numpy()
        ap, true_positives,false_positives, true_negatives, correctly_classified  = ap_utils.average_precision(pred, objListTest, -1, "CLEVR")
        print("avg precision ",ap, "tp", true_positives, "fp", false_positives, "tn", true_negatives, "correctly classified", correctly_classified)
        
        #color
        pred_c = ap_utils.inference_map_to_array(inference, only_color=True).cpu().numpy()
        ap_c, true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c = ap_utils.average_precision(pred, objListTest, -1, "CLEVR", only_color=True)
        print("avg precision color", ap_c, "tp", true_positives_c, "fp", false_positives_c, "tn", true_negatives_c, "correctly classified", correctly_classified_c)

        #shape              
        pred_s = ap_utils.inference_map_to_array(inference, only_shape=True).cpu().numpy()
        ap_s, true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s = ap_utils.average_precision(pred_s, objListTest, -1, "CLEVR", only_shape=True)
        print("avg precision shape", ap_s, "tp", true_positives_s, "fp", false_positives_s, "tn", true_negatives_s, "correctly classified", correctly_classified_s)
        
        #material              
        pred_m = ap_utils.inference_map_to_array(inference, only_material=True).cpu().numpy()
        ap_m, true_positives_m, false_positives_m, true_negatives_m, correctly_classified_m = ap_utils.average_precision(pred_m, objListTest, -1, "CLEVR", only_material=True)
        print("avg precision material", ap_m, "tp", true_positives_m, "fp", false_positives_m, "tn", true_negatives_m, "correctly classified", correctly_classified_m)
        
        #size              
        pred_x = ap_utils.inference_map_to_array(inference, only_size=True).cpu().numpy()
        ap_x, true_positives_x, false_positives_x, true_negatives_x, correctly_classified_x = ap_utils.average_precision(pred_x, objListTest, -1, "CLEVR", only_size=True)
        print("avg precision size", ap_x, "tp", true_positives_x, "fp", false_positives_x, "tn", true_negatives_x, "correctly classified", correctly_classified_x)

        
        #store ap, tp, fp, tn
        test_ap_list.append([ap, ap_c, ap_s, ap_m, ap_x, e])                    
        test_metric_list.append([true_positives, false_positives, true_negatives, correctly_classified,
                                 true_positives_c, false_positives_c, true_negatives_c, correctly_classified_c,
                                 true_positives_s, false_positives_s, true_negatives_s, correctly_classified_s,
                                 true_positives_m, false_positives_m, true_negatives_m, correctly_classified_m,
                                 true_positives_x, false_positives_x, true_negatives_x, correctly_classified_x])
        
        
        #store learning rates
        lr_temp = []
        for key in optimizers:
            lr_temp.append(optimizers[key].param_groups[0]['lr'])
        lr_list.append([lr_temp,e])

        timestamp_test = utils.time_delta_now(time_test)
        timestamp_total =  utils.time_delta_now(startTime)
        
        print('--- train time: ---', timestamp_train)
        print('--- test time: ---' , timestamp_test)
        print('--- total time from beginning: ---', timestamp_total )
        time_array = [timestamp_train, timestamp_test, timestamp_total]
        
        #save the neural network  such that we can use it later
        print('Storing the trained model into {}'.format(saveModelPath))
        torch.save({"color_net":color_net.state_dict(),
                    "shape_net":shape_net.state_dict(),                    
                    "material_net":material_net.state_dict(),
                    "size_net":size_net.state_dict(),
                    "slot_net":slot_net.state_dict(),
                    "resume": {
                        "optimizer_smsc":optimizers['smsc'].state_dict(),
                        "optimizer_slot":optimizers['slot'].state_dict(),
                        "epoch":e
                    },
                    "test_ll_list":test_ll_list,
                    "test_ap_list":test_ap_list,
                    "loss_list":loss_list,
                    "test_metric_list":test_metric_list,
                    "lr_list":lr_list,
                    "num_params":num_params,
                    "time":time_array,
                    "exp_dict":exp_dict,
                    "program":program}, saveModelPath)
        
        # Update the RTPT
        rtpt.step()
        