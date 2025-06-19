#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.0
------------------------------------------
Script for evaluating the errors of the neural networks,
using the method of nested cross-validation.

Andrés Megías
"""

# Libraries.
import os
import gc
import copy
import time
import pickle
import random
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom functions.

relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))

def invert_indices(inds, N):
    """Return the complementary indices to the input ones."""
    _inds = []
    for i in range(N):
        if i not in inds:
            _inds += [i]
    return _inds

def resample_spectra(x_new, x, y):
    "Resample input spectra (x, y) to the new wavenumber points (x_new)."
    yr = []
    for (i, yi) in enumerate(y):
        yri = np.interp(x_new, x, yi)
        yri /= np.mean(yri)
        yr += [yri]
    yr = np.array(yr)
    return yr

def add_saturation(x, y, names, saturation_levels):
    """Add simulated saturation to the dataset (x, y)."""
    x_new, y_new, names_new = [], [], []
    for (x_i, y_i, names_i) in zip(x, y, names):
        x_new += [x_i]
        y_new += [y_i]
        names_new += [names_i]
        for level in saturation_levels:
            xi_new = copy.copy(x_i)
            limit = (1 - level) * np.max(x_new)
            xi_new[xi_new > limit] = limit
            xi_new /= np.mean(x_new)
            x_new += [xi_new]
            y_new += [y_i]
            names_new += [f's({level*100:.0f})' + names_i]
    x_new = np.array(x_new)
    y_new = np.array(y_new)
    names_new = np.array(names_new)
    return x_new, y_new, names_new

#%% Preparation.

print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Script for evaluating the errors of the neural networks.')

# Start counting time.
initial_time = time.time()

# Options for the training.
num_trials = 5
random_seeds = [1,2,3,4,5,6,7,8,9,10]  # [1,2,3,4,5,6,7,8,9,10]
laboratory_dataset_path = 'training-dataset-laboratory.pkl'
lincombs_dataset_path = 'training-dataset-lincombs.pkl'
model_name = 'aice'
variables = ['temp', 'H2O', 'CO', 'CO2', 'CH3OH', 'NH3', 'CH4']  # ['temp', 'H2O', 'CO', 'CO2', 'CH3OH', 'NH3', 'CH4']
wavenumber_points = np.arange(980., 4001., 1.)
use_laboratory_dataset = True
use_lincombs_dataset = True
use_lincombs_in_validation = True
use_lincombs_in_test = {var: True for var in variables}
simulate_saturation = False
simulate_saturation_in_validation = False
simulate_saturation_in_test = False
saturation_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
num_lincombs = 282
f_ = trainval_fraction = 0.9
f = training_fraction = 0.8
num_epochs = 160
num_batches = 15
dropout_prob = 0.3
exceptions = []
# Some checks.
full_training = True if training_fraction == 1. else False
if full_training:
    use_lincombs_in_validation = False
    simulate_saturation_in_validation = False
if num_lincombs == 0:
    use_limcombs_dataset = False
if not use_lincombs_dataset:
    num_lincombs = 0
if not use_lincombs_dataset:
    use_lincombs_in_validation = False
if not simulate_saturation:
    simulate_saturation_in_validation = False
    
# Test errors variable.
test_errors = {}
for var in variables:
    test_errors[var] = []
test_errors['species'] = []

# Laboratory dataset.
if use_laboratory_dataset:
    with open(laboratory_dataset_path, 'rb') as file:
        laboratory_dataset = pickle.load(file)
    (absorbances_lab, parameters_lab, names_lab,
         all_variables, wavenumber, _) = laboratory_dataset
    absorbances_lab = resample_spectra(wavenumber_points,
                                       wavenumber, absorbances_lab)
    x_lab = np.array(absorbances_lab)
    y_lab = np.array(parameters_lab)
    names_lab = np.array(names_lab)
# Linear combinations dataset.
if use_lincombs_dataset:
    with open(lincombs_dataset_path, 'rb') as file:
        lincombs_dataset = pickle.load(file)
    absorbances_lc, parameters_lc, names_lc, all_variables, wavenumber = \
        lincombs_dataset
    absorbances_lc = resample_spectra(wavenumber_points,
                                      wavenumber, absorbances_lc)
    x_lc = np.array(absorbances_lc)[:num_lincombs]
    y_lc = np.array(parameters_lc)[:num_lincombs]
    names_lc = np.array(names_lc)[:num_lincombs]
Lx = x_lab.shape[1]
Ly = y_lab.shape[1]
    
# Order of variables.
num_vars = len(variables)
vars_inds_dic = {}
for (i,variable) in enumerate(all_variables):
    vars_inds_dic[variable] = i
inds_vars = [vars_inds_dic[var] for var in variables]
    
# Time count.
t1 = time.time()

#%%

# Loop for trials for test.
for trial in range(1, num_trials+1):
    
    # Random seed.
    print(f'\nTrial {trial}.')
    random.seed(trial)
    np.random.seed(trial)
    
    if use_laboratory_dataset:
        # Indices for spliting in training+validation and test subsets.
        N = len(y_lab)
        inds_trval = np.array([], int)
        for (i,yi) in enumerate(y_lab):
            if ((any(yi == 1.) or yi[-1] != 0.
                    or any([names_lab[i].startswith(text) for text in exceptions]))
                    and i not in inds_trval):
                inds_trval = np.append(inds_trval, i)
        inds_test = invert_indices(inds_trval, N)
        f0 = len(inds_trval) / N
        Nn = round(N * (f_ - f0))
        new_inds = random.sample(sorted(inds_test), Nn)
        inds_trval = np.append(inds_trval, new_inds)
        inds_test = invert_indices(inds_trval, N)  
        # Spliting in training+validation and tests subsets.
        x_trval = x_lab[inds_trval]
        y_trval = y_lab[inds_trval]
        names_trval = names_lab[inds_trval]
        x_test = x_lab[inds_test]
        y_test = y_lab[inds_test]
        names_test = names_lab[inds_test]
        if len(y_trval.shape) == 1:
            y_trval = y_trval.reshape(-1,1)
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1,1)
        # Augmented data with synthetic linear combinations.
        if use_lincombs_dataset:
            N = len(y_lc)
            # f_ = len(x_trval) / (len(x_trval) + len(x_test))
            idx_ = int(f_*N) if use_lincombs_in_test else N
            x_lc_ = x_lc[:idx_]
            y_lc_ = y_lc[:idx_]
            names_lc_ = names_lc[:idx_]
            if use_lincombs_in_test:
                x_test = np.append(x_test, x_lc[idx_:], axis=0)
                y_test = np.append(y_test, y_lc[idx_:], axis=0)
                names_test = np.append(names_test, names_lc[idx_:])
    else:
        N = len(y_lc)
        inds_trval= np.arange(int(f_*N))
        inds_test = invert_indices(inds_trval, N)
        x_trval = x_lc[inds_trval]
        y_trval = y_lc[inds_trval]
        names_trval = names_lc[inds_trval]
        x_test = x_lc[inds_test]
        y_test = y_lc[inds_test]
        names_test = names_lc[inds_test]

    # Loop for splits for validation.
    for seed in random_seeds:
        
        # Random seed.
        print(f'\nTrial {trial}. Seed {seed}.')
        gc.collect()
        plt.close('all')
        random.seed(seed)
        np.random.seed(seed)
        keras.utils.set_random_seed(seed)
        # Folder for saving files.
        if not os.path.exists(os.path.join('models', model_name, 'errors', 'test')):
            os.makedirs(os.path.join('models', model_name, 'errors', 'test'))
        if use_laboratory_dataset:
            # Indices for spliting in training and validation subsets.
            N = len(y_trval)
            inds_train = np.array([], int)
            for (i,yi) in enumerate(y_trval):
                if ((any(yi == 1.) or yi[-1] != 0.
                        or any([names_trval[i].startswith(text) for text in exceptions]))
                        and i not in inds_train):
                    inds_train = np.append(inds_train, i)
            inds_val = invert_indices(inds_train, N)
            f0 = len(inds_train) / N
            Nn = round(N * (f - f0))
            new_inds = random.sample(sorted(inds_val), Nn)
            inds_train = np.append(inds_train, new_inds)
            inds_val = invert_indices(inds_train, N)       
            # Spliting in training and validation subsets.
            x_train = x_trval[inds_train]
            y_train = y_trval[inds_train]
            names_train = names_trval[inds_train]
            x_val = x_trval[inds_val]
            y_val = y_trval[inds_val]
            names_val = names_trval[inds_val]
            # Augmented data with synthetic linear combinations.
            if use_lincombs_dataset:
                N = len(y_lc)
                # f = len(x_train) / (len(x_train) + len(x_val))
                idx = int(f*N) if use_lincombs_in_validation else N
                x_train = np.append(x_train, x_lc_[:idx], axis=0)
                y_train = np.append(y_train, y_lc_[:idx], axis=0)
                names_train = np.append(names_train, names_lc_[:idx])
                inds = random.sample(range(len(x_train)), len(x_train))
                x_train, y_train = x_train[inds], y_train[inds]
                names_train = names_train[inds]
                if use_lincombs_in_validation:
                    x_val = np.append(x_val, x_lc_[idx:], axis=0)
                    y_val = np.append(y_val, y_lc_[idx:], axis=0)
                    names_val = np.append(names_val, names_lc_[idx:])
        else:
            N = len(y_lc_)
            inds_train = np.arange(int(f*N))
            inds_val = invert_indices(inds_train, N)
            x_train = x_lc_[inds_train]
            y_train = y_lc_[inds_train]
            names_train = names_lc_[inds_train]
            x_val = x_lc_[inds_val]
            y_val = y_lc_[inds_val]
            names_val = names_lc_[inds_val]
        # Data augmentation to simulate saturation.
        if simulate_saturation:
            x_train, y_train, names_train = add_saturation(x_train, y_train,
                                                names_train, saturation_levels)
            if simulate_saturation_in_validation:
                x_val, y_val, names_val = add_saturation(x_val, y_val,
                                                  names_val, saturation_levels)
        # Batch size.
        batch_size = int(np.ceil(len(x_train)/num_batches))
        
        #%% Training of the models.
        
        species, models = [], {}
        # Loop for variables.
        for var in variables:
            
            # Set random seed at the start.
            print(f'\nTrial {trial}. Seed {seed}. Variable {var}.')
            keras.utils.set_random_seed(seed)
            # Identification of type of variable.
            idx = vars_inds_dic[var]
            if idx == 0:
                fit_mol = False
                fit_temp = True
            else:
                species += [var]
                fit_mol = True
                fit_temp = False
            end_act = 'relu' if fit_temp else 'sigmoid'
            # Network architecture.
            inputs = keras.Input(shape=(Lx,))
            mid = keras.layers.Dense(120, activation='relu')(inputs)
            mid = keras.layers.BatchNormalization()(mid)
            mid = keras.layers.Dropout(dropout_prob)(mid)
            mid = keras.layers.Dense(60, activation='relu')(mid)
            mid = keras.layers.BatchNormalization()(mid)
            mid = keras.layers.Dense(30, activation='relu')(mid)
            mid = keras.layers.BatchNormalization()(mid)
            outputs = keras.layers.Dense(1, activation=end_act)(mid)
            model = keras.Model(inputs, outputs)
            # Optimizer and loss.
            optimizer = keras.optimizers.Adam(learning_rate=0.02,
                                              beta_1=0.9, beta_2=0.999)
            loss = keras.losses.MeanSquaredLogarithmicError()
            model.compile(loss=loss, optimizer=optimizer)
            loss_monitor = 'val_loss' if not full_training else 'loss'
            stop = keras.callbacks.EarlyStopping(monitor=loss_monitor,
                                patience=35, start_from_epoch=45, min_delta=0,
                                restore_best_weights=True)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=loss_monitor,
                                        factor=1/2, patience=15, min_delta=0)
            checkpoint_file = os.path.join('models', model_name, 'weights',
                                            str(seed), 'checkpoint.weights.h5')
            checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_file,
                                   monitor=loss_monitor,
                                   save_weights_only=True, save_best_only=True)
            # Training.
            xy_val = (x_val, y_val[:,idx]) if not full_training else None
            result = model.fit(x_train, y_train[:,idx],
                    validation_data=xy_val, batch_size=batch_size,
                    epochs=num_epochs, callbacks=[stop, reduce_lr, checkpoint],
                    shuffle=False)
            models[var] = model
        print()
            
        #%% Model definition.
        
        # Keras models.
        def nn_model(x):
            y = np.zeros((x.shape[0], num_vars))
            for (i, var) in enumerate(variables):
                y[:,i] = models[var].predict(x).flatten()
            return y
        
        # Model predictions.
        Y_train = nn_model(x_train)
        if not full_training:
            Y_val = nn_model(x_val)
        Y_test = nn_model(x_test)
        
        # Determination of variables trained.
        fit_temp = True if 0 in inds_vars else False
        fit_mol = False if inds_vars == [0] else True
        si = 1 if fit_temp else 0
        
        #%% Calculation of errors.
        
        # Preparation of data to be analyzed.
        x_ = x_test.copy()
        y_ = y_test[:,inds_vars]
        Y_ = Y_test.copy()
        names_ = names_test.copy()
        # Errors.
        inds_err = {}
        for var in variables:
            if (full_training
                    or (use_lincombs_in_test[var] and simulate_saturation_in_test)
                    or (not use_lincombs_dataset and not simulate_saturation)):
                inds_err = np.ones(len(y_), bool)
            else:
                names_err = names_lab[inds_test]
                if use_lincombs_in_test[var]:
                    names_err = np.append(names_err, names_lc[int(f*len(y_lc)):])
                if simulate_saturation_in_test:
                    _, _, names_err_ = add_saturation(x_lab[inds_test], y_lab[inds_test],
                                            names_lab[inds_test], saturation_levels)
                    names_err = np.append(names_err, names_err_)
                inds_err[var] = []
                for name_j in names_err:
                    for (i, name_i) in enumerate(names_):
                        if name_i == name_j:
                            inds_err[var] += [i]
                            break
        errors_mols = [np.abs(y_[inds_err[var],si+i] - Y_[inds_err[var],si+i])
                       for (i,var) in enumerate(species)]
        all_errors_mols = []
        for errors_i in errors_mols:
            all_errors_mols += list(errors_i)
        all_errors_mols = np.array(all_errors_mols)
        if fit_temp:
            errors_temp = np.abs(y_[inds_err['temp'],0] - Y_[inds_err['temp'],0])
        # Errors for variables.
        error_function = lambda x: np.nanmean(x**2)**0.5
        errors = {}
        if fit_mol:
            for (i, var) in enumerate(species):
                errors[var] = error_function(errors_mols[i])
        if fit_temp:
            errors['temp'] = error_function(errors_temp)
        errors['species'] = error_function(all_errors_mols)
        # Test errors table for each trial and seed.
        errors_filename = os.path.join('models', model_name, 'errors', 'test',
                                       f'errors{trial}.csv')
        errors_new = pd.DataFrame(errors, index=[seed])
        if os.path.exists(errors_filename):
            errors_prev = pd.read_csv(errors_filename, index_col=[0])
            errors_df = errors_prev
            for name in errors_new:
                if name in errors_prev:
                    errors_df.at[seed,name] = errors_new[name].values[0]
                else:
                    errors_df[name] = errors_new[name]
        else:
            errors_df = errors_new
        errors_df.to_csv(errors_filename, float_format='%.4f')
        print(f'\nSaved file {errors_filename}.')
        
    # Average test errors table for each trial.
    errors_filename = os.path.join('models', model_name, 'errors', 'test',
                                   f'errors{trial}.csv')
    test_errors_df = pd.read_csv(errors_filename, index_col=[0])
    for var in variables:
        test_errors[var] += [test_errors_df[var].mean()]
    test_errors['species'] += [test_errors_df['species'].mean()]


print()     
test_errors_df = pd.DataFrame(test_errors, index=np.arange(num_trials)+1)
filename = os.path.join('models', model_name, 'errors',
                        f'{model_name}-test-errors.csv')
test_errors_df.to_csv(filename, float_format='%.4f')
print(f'Saved file {filename}.\n')
            
# Time check.         
final_time = time.time()
elapsed_time = final_time - initial_time
print(f'Total elapsed time: {elapsed_time//60:.0f} min + {elapsed_time%60:.0f} s.')