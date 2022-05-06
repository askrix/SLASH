#!/usr/bin/env python
# coding: utf-8
# MNIST Digit Addition Experiment
import train

seed = 4 
# drop_out = 0.5

# experiments = {f'pc-poon-domingos-drop-out-{int(100*drop_out)}-percent-seed-{seed}':
#                {'structure':'poon-domingos', 'pd_num_pieces':[4,7,28],
#                 'use_spn':True, 'credentials':'AS', 'seed':seed, 'learn_prior':True,
#                 'lr':0.01, 'bs':100, 'epochs':100, 'p_num':8, 'drop_out':drop_out}
#               }

# experiments = {f'pc-poon-domingos-normal6-seed-{seed}-epochs-100':
#                {'structure':'poon-domingos', 'pd_num_pieces':[4,7,28],
#                 'use_spn':True, 'credentials':'AS', 'seed':seed, 'learn_prior':True,
#                 'lr':0.01, 'bs':100, 'epochs':100, 'p_num':8, 'drop_out':0.0}
#               }

#experiments = {f'pc-binary-trees-normal-seed-{seed}':
#               {'structure':'binary-trees', 'K':10, 'depth':3, 'num_repetitions':20, 'num_sums':10,
#                'use_spn':True, 'credentials':'AS', 'seed':1, 'learn_prior':True,
#                'lr':0.01, 'bs':100, 'epochs':15, 'p_num':8, 'drop_out':0.0}
#              }

experiments = {f'dnn-normal-seed-{seed}':
               {'use_spn':False, 'credentials':'AS', 'seed':seed,
                'lr':0.005, 'bs':100, 'epochs':100, 'p_num':8, 'drop_out':0.0}
              }

# experiments = {f'dnn-drop-out-{int(100*drop_out)}-percent-seed-{seed}':
#                 {'use_spn':False, 'credentials':'AS', 'seed':seed,
#                  'lr':0.005, 'bs':100, 'epochs':100, 'p_num':8, 'drop_out':drop_out}
#               }

for exp_name in experiments:
    print("Experiment's folder is %s" % exp_name)
    train.slash_mnist_addition(exp_name, experiments[exp_name])
