#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
------------------------------------------------
Automatic Ice Composition Estimator (AICE) v 1.1
------------------------------------------
Script for combining the weights of the models into a single file.

Andrés Megías
"""

import os
import pickle
import numpy as np

#%% Packing all the models into the final one.
       
model_variables = ['temp', 'H2O', 'CO', 'CO2', 'CH3OH', 'NH3', 'CH4']
model_seeds = [1,2,3,4,5,6,7,8,9,10]    
model_name = 'aice'
    
all_models_available = True
weights, all_weights = [], []
for seed in model_seeds:
    seed = str(seed)
    weights_i = []
    for var in model_variables:
        try:
            weights_folder = os.path.join('models', model_name, 'weights', seed)
            weights_path = os.path.join(weights_folder, f'weights-{var}.npy')
            weights_ij = np.load(weights_path, allow_pickle=True)
        except:
            all_models_available = False
            break
        weights_i += [weights_ij]
    all_weights += [weights_i]
if all_models_available:
    all_weights = np.array(all_weights, object)
    for var in model_variables:
        spectral_range_prev, spacing_prev, resolution_prev = None, None, None
        model_info_path = os.path.join('models', model_name, 'model-info', 
                                       f'model-info-{var}.txt')
        with open(model_info_path, 'r') as file:
            for line in file:
                if line.startswith('spectral range'):
                    text = line.split(': ')[1]
                    text = (text.replace('(', '').replace(')', '')
                            .replace('[', '').replace(']', '').replace(' ', ''))
                    x1, x2 = text.split(',')
                    x1x2 = [float(x1), float(x2)]
                    x1, x2 = min(x1x2), max(x1x2)
                    spectral_range = [x1, x2]
                    if (spectral_range_prev is not None
                            and spectral_range != spectral_range_prev):
                        print('Error: Not all models have the same spectral range.')
                        break
                elif line.startswith('spectral spacing'):
                    text = line.split(': ')[1]
                    spacing = float(text)
                    if spacing_prev is not None and spacing != spacing_prev:
                        print('Error: Not all models have the same spectral spacing.')
                        break
                    spectral_range[1] = spectral_range[1] + spacing
                elif line.startswith('spectral resolution'):
                    text = line.split(': ')[1]
                    resolution = float(text)
                    if resolution_prev is not None and resolution != resolution_prev:
                        print('Error: Not all models have the same resolution.')
                        break
                elif line.startswith('use '):
                    break
            spectral_range_prev = spectral_range
            spacing_prev = spacing
            resolution_prev = resolution
    filename = os.path.join('models', f'{model_name}-model.pkl')
    model_info = {'variables': model_variables, 'wavenumber_range': spectral_range,
                  'spacing': spacing, 'resolution': resolution}
    with open(filename, 'wb') as file:
        pickle.dump([all_weights, model_info], file)
    print('Saved file {}.'.format(filename))
else:
    print('Error: Not all individual models available.')