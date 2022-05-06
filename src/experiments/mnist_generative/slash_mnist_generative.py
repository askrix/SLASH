#!/usr/bin/env python
# coding: utf-8
# MNIST Digit Addition Experiment
import train

seed = 1
drop_out = 0.0


experiments = {f'pc-generative-poon-domingos-normal-seed-{seed}-epochs-100-pd-7-k10':
               {'structure':'poon-domingos', 'pd_num_pieces':[7],
                'use_spn':True, 'credentials':'DO', 'seed':seed, 'learn_prior':True,
                'lr':0.01, 'bs':100, 'epochs':100, 'p_num':8, 'drop_out':0.0}
              }

for exp_name in experiments:
    print("Experiment's folder is %s" % exp_name)
    train.slash_mnist_addition(exp_name, experiments[exp_name])
