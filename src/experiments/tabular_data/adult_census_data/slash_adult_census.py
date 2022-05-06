#!/usr/bin/env python
# coding: utf-8
# Adult Census Income Experiment
import train


experiments = {'tabnet-classification-seed-1':
               {'credentials':'AS', 'seed':1, 'lr':0.02, 'use_spn':False,
                'bs':16384, 'epochs':100, 'p_num':8, 'type':'classification'}
              }

# experiments = {'tabnet-seed-1':
#                {'credentials':'AS', 'seed':1, 'lr':0.02, 
#                 'bs':1024, 'epochs':100, 'p_num':8, 'type':'social_study'}
#               }

# experiments = {'pc-binary-trees-classification-seed-1':
#               {'structure':'binary-trees', 'K':14, 'depth':3, 'num_repetitions':20, 'num_sums':14,  # 'K':10, 'depth':3, 'num_repetitions':20, 'num_sums':10,
#                'use_spn':True, 'credentials':'AS', 'seed':1, 'learn_prior':True,
#                'lr':0.01, 'bs':16384, 'epochs':50, 'p_num':8, 'type':'classification'}  # classification 16384
#              }

# experiments = {'pc-poon-domingos-normal-seed-5':
#                {'structure':'poon-domingos', 'pd_num_pieces':[4,7,28],
#                 'use_spn':True, 'credentials':'AS', 'seed':5, 'learn_prior':True,
#                 'lr':0.01, 'bs':100, 'epochs':100, 'p_num':8, 'type':'classification'}
#               }


for exp_name in experiments:
    print("Experiment's folder is %s" % exp_name)
    train.slash_tabular_learning(exp_name, experiments[exp_name])
