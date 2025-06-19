#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------
Automatic Ice Composition Estimator (AICE) v 1.0
------------------------------------------
Script for combining the weights of the models into a single file.

Andrés Megías
"""

import os
import numpy as np

#%% Packing all the models into the final one.
       
model_variables = ['temp', 'H2O', 'CO', 'CO2', 'CH3OH', 'NH3', 'CH4']
model_seeds = [1,2,3,4,5,6,7,8,9,10]    
model_name = 'aice-lite'
    
all_models_available = True
weights = []
all_weights = []
for seed in model_seeds:
    weights_i = []
    for var in model_variables:
        try:
            weights_ij = np.load(os.path.join('models', model_name, 'weights',
                                              str(seed), f'weights-{var}.npy'),
                                 allow_pickle=True)
        except:
            all_models_available = False
            break
        weights_i += [weights_ij]
    all_weights += [weights_i]
if all_models_available:
    all_weights = np.array(all_weights, object)
    filename = os.path.join('models', f'{model_name}-weights.npy')
    np.save(filename, all_weights)
    print('Saved file {}.'.format(filename))
else:
    print('Not all individual models available.')