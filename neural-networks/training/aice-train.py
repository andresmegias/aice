#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.1
------------------------------------------
Script for training the neural networks.

Andrés Megías
"""

# Libraries.
import os
import gc
import copy
import time
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Custom functions.

relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))

def format_func(x, pos):
    """Formatting of numbers mixing normal and logarithmic scale."""
    if x >= 1:
        g = '{:.0f}'.format(x)
    elif x == 0:
        g = '0'
    else:
        n = int(abs((np.floor(np.log10(abs(x))))))
        g = '{:.{}f}'.format(x, n).replace('-','$-$')
    return g

def axis_conversion(x):
    """Axis conversion from wavenumber to wavelength and viceversa"""
    with np.errstate(divide='ignore'):
        y = 1e4 / x
    return y

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
    for (xi, yi, names_i) in zip(x, y, names):
        x_new += [xi]
        y_new += [yi]
        names_new += [names_i]
        for level in saturation_levels:
            xi_new = copy.copy(xi)
            limit = (1 - level) * np.max(xi_new)
            xi_new[xi_new > limit] = limit
            xi_new /= np.mean(xi_new)
            x_new += [xi_new]
            y_new += [yi]
            names_new += [f's({level*100:.0f})' + names_i]
    x_new = np.array(x_new)
    y_new = np.array(y_new)
    names_new = np.array(names_new)
    return x_new, y_new, names_new

def shuffle_dataset(x, y, names):
    """Shuffle given dataset (x: spectra, y: parameters)."""
    N = len(names)
    inds = random.sample(range(N), N)
    x_new = x[inds]
    y_new = y[inds]
    names_new = names[inds]
    return x_new, y_new, names_new

# Custom losses.
eps_molfrac = 0.1  # threshold for molecular fraction
def custom_loss_molfrac(y_true, y_pred):
    """Loss function for the training of the neural networks."""
    log_y_true = tf.math.log(y_true + eps_molfrac)
    log_y_pred = tf.math.log(y_pred + eps_molfrac)
    diffs = log_y_pred - log_y_true
    loss = tf.math.reduce_mean(diffs**2, axis=-1)
    return loss
eps_temp = 0.1  # threshold for temperature
def custom_loss_temp(y_true, y_pred):
    """Loss function for the training of the neural networks."""
    log_y_true = tf.math.log(y_true + eps_temp)
    log_y_pred = tf.math.log(y_pred + eps_temp)
    diffs = log_y_pred - log_y_true
    loss = tf.math.reduce_mean(diffs**2, axis=-1)
    return loss

#%% Preparation.

print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Script for training the neural networks.')

# Start counting time.
initial_time = time.time()

# Options for the training.
random_seeds = [1]  # [1,2,3,4,5,6,7,8,9,10]
laboratory_dataset_path = 'datasets/aice2/training-dataset-laboratory.pkl'
lincombs_dataset_path = 'datasets/aice/training-dataset-lincombs.pkl'
model_name = 'aice2'
train_nn = True
save_results = False
variables =  ['temp', 'H2O']  # ['temp', 'H2O', 'CO', 'CO2', 'CH3OH', 'NH3', 'CH4']
formatted_names = {'H2O': r'${\rm H_2O}$', 'CO': 'CO', 'CO2': r'${\rm CO_2}$',
                   'CH3OH': r'${\rm CH_3OH}$', 'NH3': r'${\rm NH_3}$',
                   'CH4': r'${\rm CH_4}$', 'temp': 'temp. (K)'}
wavenumber_points = np.arange(980, 3751., 1.)
use_custom_losses = True
use_laboratory_dataset = True
use_lincombs_dataset = False
use_lincombs_in_validation = True
use_lincombs_for_validation_errors = False
simulate_saturation = False
simulate_saturation_in_validation = False
use_saturated_data_for_validation_errors = False
saturation_levels = [0.2, 0.4, 0.6]
num_lincombs = 282 // 3
f = training_fraction = 0.9
f_lc = training_fraction_lc = 0.8
num_epochs = 160
num_batches = {'temp': 7, 'H2O': 6, 'CO': 12, 'CO2': 6,
               'CH3OH': 6, 'NH3': 8, 'CH4': 8}
dropout_prob = {'temp': 0.05, 'H2O': 0.10, 'CO': 0.20, 'CO2': 0.10,
                'CH3OH': 0.05, 'NH3': 0.15, 'CH4': 0.10}
relu_negative_slope = {'temp': 0.5, 'H2O': 0., 'CO': 0, 'CO2': 0.,
                       'CH3OH': 0., 'NH3': 0., 'CH4': 0.}
fixed_train_spectra = []
# Some checks.
if num_lincombs == 0:
    use_limcombs_dataset = False
if not use_lincombs_dataset:
    num_lincombs = 0
    use_lincombs_in_validation = False
    training_fraction_lc = None
if not use_lincombs_in_validation:
    use_lincombs_for_validation_errors = False
if (not use_lincombs_in_validation and f == 1. or
        use_lincombs_in_validation and f == 1. and f_lc == 1.):
    full_training = True
else:
    full_training = False
if full_training:
    use_lincombs_in_validation = False
    simulate_saturation_in_validation = False
if not simulate_saturation:
    simulate_saturation_in_validation = False
if not simulate_saturation_in_validation:
    use_saturated_data_for_validation_errors = False
if type(dropout_prob) is not dict:
    dropout_prob = {var: dropout_prob for var in variables}
if type(num_batches) is not dict:
    num_batches = {var: num_batches for var in variables}
if type(relu_negative_slope) is not dict:
    relu_negative_slope = {var: relu_negative_slope for var in variables}
spectral_spacing = np.median(np.diff(wavenumber_points))

# Laboratory dataset.
if use_laboratory_dataset:
    with open(laboratory_dataset_path, 'rb') as file:
        dataset = pickle.load(file)
    absorbances_lab = dataset['absorbances']
    descriptors_lab = dataset['target_values']
    names_lab = dataset['spectra_names']
    all_variables = dataset['target_variables']
    wavenumber = dataset['wavenumber_points']
    resolution = dataset['resolution']
    absorbances_lab = resample_spectra(wavenumber_points,
                                       wavenumber, absorbances_lab)
    x_lab = np.array(absorbances_lab)
    y_lab = np.array(descriptors_lab)
    names_lab = np.array(names_lab)
# Linear combinations dataset.
if use_lincombs_dataset:
    with open(lincombs_dataset_path, 'rb') as file:
        dataset = pickle.load(file)
    absorbances_lc = dataset['absorbances']
    descriptors_lc = dataset['target_values']
    names_lc = dataset['spectra_names']
    all_variables_lc = dataset['target_variables']
    wavenumber_lc = dataset['wavenumber_points']
    absorbances_lc = resample_spectra(wavenumber_points, wavenumber_lc, absorbances_lc)
    x_lc = np.array(absorbances_lc)[:num_lincombs]
    y_lc = np.array(descriptors_lc)[:num_lincombs]
    names_lc = np.array(names_lc)[:num_lincombs]
    
# Order of variables.
species = [var for var in variables if 'temp' not in var]
if 'temp' in variables:
    variables = ['temp'] + species
num_vars = len(variables)

#%%

# Loop for seeds.
for seed in random_seeds:
    
    # Set of random seed.
    print(f'\nSeed {seed}.')
    gc.collect()
    plt.close('all')
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    # Folder for saving files.
    for folder in ('weights', 'histories', 'plots'):
        if not os.path.exists(os.path.join('models', model_name, folder, str(seed))):
            os.makedirs(os.path.join('models', model_name, folder, str(seed)))
    for folder in ('indices', 'errors'):
        if not os.path.exists(os.path.join('models', model_name, folder)):
            os.makedirs(os.path.join('models', model_name, folder))
    # Preparation of the dataset.
    if use_laboratory_dataset:
        # Indices for spliting in training and validation subsets.
        inds_file = os.path.join('models', model_name, 'indices',
                                 f'val-indices-{seed}.txt')
        N = len(y_lab)
        if train_nn:
            inds_train = np.array([], int)
            for (i, yi) in enumerate(y_lab):
                if i not in inds_train and ((any(yi == 1.) or yi[-1] != 0.
                        or any([names_lab[i].startswith(text)
                                for text in fixed_train_spectra]))):
                    inds_train = np.append(inds_train, i)
            inds_val = invert_indices(inds_train, N)
            f0 = len(inds_train) / N
            Nn = round(N * (f - f0))
            new_inds = random.sample(sorted(inds_val), Nn)
            inds_train = np.append(inds_train, new_inds)
            inds_val = invert_indices(inds_train, N)
            if save_results:
                np.savetxt(inds_file, inds_val, fmt='%i')
        else:
            inds_val = np.loadtxt(inds_file, dtype=int)
            inds_train = invert_indices(inds_val, N)        
        # Spliting in training and validation subsets.
        x_train = x_lab[inds_train]
        y_train = y_lab[inds_train]
        names_train = names_lab[inds_train]
        x_val = x_lab[inds_val]
        y_val = y_lab[inds_val]
        names_val = names_lab[inds_val]
        x_train, y_train, names_train = shuffle_dataset(x_train, y_train, names_train)
        names_e = names_val.copy() if not full_training else names_train.copy()
        # Augmented data with synthetic linear combinations.
        if use_lincombs_dataset:
            if use_lincombs_in_validation:
                N = len(names_lc)
                inds_train = random.sample(range(N), round(f_lc*N))
                inds_val = invert_indices(inds_train, N)
                x_train = np.append(x_train, x_lc[inds_train], axis=0)
                y_train = np.append(y_train, y_lc[inds_train], axis=0)
                names_train = np.append(names_train, names_lc[inds_train])
                x_val = np.append(x_val, x_lc[inds_val], axis=0)
                y_val = np.append(y_val, y_lc[inds_val], axis=0)
                names_val = np.append(names_val, names_lc[inds_val])
                if use_lincombs_for_validation_errors:
                    names_e = names_val.copy()
            else:
                x_train = np.append(x_train, x_lc, axis=0)
                y_train = np.append(y_train, y_lc, axis=0)
                names_train = np.append(names_train, names_lc)
                if full_training:
                    names_e = names_train.copy()  
            x_train, y_train, names_train = shuffle_dataset(x_train, y_train, names_train) 
    else:
        # Spliting in training and validation subsets.
        N = len(y_lc)
        inds_train = random.sample(range(N), round(f_lc*N))
        inds_val = invert_indices(inds_train, N)
        x_train = x_lc[inds_train]
        y_train = y_lc[inds_train]
        names_train = names_lc[inds_train]
        x_val = x_lc[inds_val]
        y_val = y_lc[inds_val]
        names_val = names_lc[inds_val]
        names_e = names_val.copy() if not full_training else names_train.copy()
    # Data augmentation to simulate saturation.
    if simulate_saturation:
        x_train, y_train, names_train = add_saturation(x_train, y_train,
                                                names_train, saturation_levels)
        if simulate_saturation_in_validation:
            x_val, y_val, names_val = add_saturation(x_val, y_val, names_val,
                                                     saturation_levels)
            if use_saturated_data_for_validation_errors:
                names_e = names_val.copy()
        if full_training:
            names_e = names_train.copy()
        x_train, y_train, names_train = shuffle_dataset(x_train, y_train, names_train)
    # Input and output_sizes
    Lx = x_train.shape[1]
    Ly = y_train.shape[1]
    # Copies.
    y_train_copy = copy.copy(y_train)
    y_val_copy = copy.copy(y_val)
    # Batch size
    batch_size = {var: int(np.ceil(len(x_train)/num_batches[var])) for var in variables}
    
    #%% Training of the models.
    
    t1 = time.time()
    models, histories = {}, {}
    
    # Loop for variables.
    for var in variables:
        
        # Set random seed at the start.
        print(f'\nSeed {seed}. Variable {var}.')
        keras.utils.set_random_seed(seed)
        # Identification of type of variable.
        var_is_molfrac = True if var != 'temp' else False
        # Get hyperparameters for the variables.
        end_act = 'sigmoid' if var_is_molfrac else 'softplus'
        alpha = relu_negative_slope[var]
        prob = dropout_prob[var]
        # Network architecture.
        inputs = keras.Input(shape=(Lx,))
        mid = keras.layers.Dense(120, activation=None)(inputs)
        mid = keras.layers.LeakyReLU(alpha)(mid)
        mid = keras.layers.BatchNormalization()(mid)
        mid = keras.layers.Dropout(prob)(mid)
        mid = keras.layers.Dense(60, activation=None)(mid)
        mid = keras.layers.LeakyReLU(alpha)(mid)
        mid = keras.layers.BatchNormalization()(mid)
        mid = keras.layers.Dense(30, activation=None)(mid)
        mid = keras.layers.LeakyReLU(alpha)(mid)
        mid = keras.layers.BatchNormalization()(mid)
        outputs = keras.layers.Dense(1, activation=end_act)(mid)
        model = keras.Model(inputs, outputs)
        # Optimizer and loss.
        optimizer = keras.optimizers.Adam(learning_rate=0.02,
                                          beta_1=0.9, beta_2=0.999)
        if use_custom_losses:
            loss = custom_loss_molfrac if var_is_molfrac else custom_loss_temp
        else:
            loss = keras.losses.MeanSquaredLogarithmicError()
        model.compile(loss=loss, optimizer=optimizer)
        loss_monitor = 'val_loss' if not full_training else 'loss'
        stop = keras.callbacks.EarlyStopping(monitor=loss_monitor, patience=35,
                   start_from_epoch=45, min_delta=0., restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=loss_monitor,
                                         factor=1/2, patience=15, min_delta=0.)
        checkpoint_file = os.path.join('models', model_name, 'weights',
                                       str(seed), 'checkpoint.weights.h5')
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_file,
                                     monitor=loss_monitor, save_best_only=True,
                                     save_weights_only=True)
        # Training.
        if train_nn:
            idx = np.argwhere(np.array(all_variables) == var)[0,0]
            if var == 'temp':
                mask = y_train[:,-1] == 0.
                x_train_ = x_train[mask]
                y_train_ = y_train[mask,:]
                batch_size_ = int(np.ceil(len(x_train_)/num_batches[var]))
            else:
                x_train_ = copy.copy(x_train)
                y_train_ = copy.copy(y_train)
                batch_size_ = batch_size[var]
            xy_val = (x_val, y_val[:,idx]) if not full_training else None
            result = model.fit(x_train_, y_train_[:,idx], validation_data=xy_val,
                               batch_size=batch_size_, epochs=num_epochs,
                               callbacks=[checkpoint, reduce_lr, stop])
            model.load_weights(checkpoint_file)
            histories[var] = result.history
            os.remove(checkpoint_file)
        # Loading of pre-trained model.
        else:
            weights = np.load(os.path.join('models', model_name, 'weights',
                        str(seed), f'weights-{var}.npy'), allow_pickle=True)
            model.set_weights(weights)
            with open(os.path.join('models', model_name, 'histories', str(seed),
                                   f'history-{var}.pkl'), 'rb') as file:
                histories[var] = pickle.load(file)
        models[var] = model
        
    # Time count.
    print()
    if train_nn:
        t2 = time.time()
        print(f'Elapsed time for training: {(t2-t1):.0f} s.')
        print()
        
    #%% Model definition.
    
    # Keras models.
    def nn_model(x):
        y = np.zeros((x.shape[0], num_vars), float)
        for (i, var) in enumerate(variables):
            y[:,i] = models[var].predict(x).flatten()
        return y
    
    # Model predictions.
    Y_train = nn_model(x_train)
    if not full_training:
        Y_val = nn_model(x_val)
    print()
    
    #%% Plots and calculation of errors
    
    # Visualization options.
    shown_dataset = 'training'
    if full_training and shown_dataset == 'validation':
        shown_dataset = 'training'
    shown_variables = copy.copy(variables)   # ['temp', 'H2O']
    shown_figures = [1, 4]
    show_temp_as_size = True
    show_spectra_names = False
    use_logscale_for_errors = True
    only_shown_species_temps = True
    colors = {'temp': 'dimgray', 'H2O': 'tab:blue', 'CO': 'tab:red',
              'CO2': 'tab:orange', 'CH3OH': 'tab:green',
              'NH3': 'tab:cyan', 'CH4': 'tab:purple', 'other': 'gray'}
    shown_species = [var for var in shown_variables if var != 'temp']
    num_shown_species = len(shown_species)
    if num_shown_species == 0:
        only_shown_species_temps = False
    idx = example_idx = 0
    array_species = np.array(species)
    
    # Preparation of data to be shown.
    inds = [i for (i, var) in enumerate(all_variables) if var in variables]
    y_train = y_train_copy[:,inds]
    y_val = y_val_copy[:,inds]
    if shown_dataset == 'training':
        x_ = x_train.copy()
        y_ = y_train.copy()
        Y_ = Y_train.copy()
        names_ = names_train.copy()
    elif shown_dataset == 'validation':
        x_ = x_val.copy()
        y_ = y_val.copy()
        Y_ = Y_val.copy()
        names_ = names_val
    show_molfracs = True if len(shown_species) > 0 else False
    show_temp = True if 'temp' in variables else False
    si = 1 if 'temp' in variables else 0
    # Errors.
    if shown_dataset == 'training':
        inds_err = np.arange(len(y_))
    else:
        inds_err = []
        for (i, name) in enumerate(names_):
            if name in names_e:
                inds_err += [i]
    y_e = y_[inds_err,:]
    Y_e = Y_[inds_err,:]
    if len(species) > 0:
        errors_mols = [np.abs(y_e[:,si+i] - Y_e[:,si+i])
                       for (i,var) in enumerate(species)]
        all_errors_mols = []
        for errors_i in errors_mols:
            all_errors_mols += list(errors_i)
        all_errors_mols = np.array(all_errors_mols)
    if 'temp' in variables:
        y_ref = y_train_copy if shown_dataset == 'training' else y_val_copy
        mask = mask_temp = y_ref[:,-1] == 0.
        mask_ = np.zeros(len(mask), bool)
        if only_shown_species_temps:
            for (i, name) in enumerate(variables):
                if name in shown_species:
                    mask_ |= y_ref[:,i] != 0.
        mask = mask & mask_
        errors_temp = np.abs(y_e[mask,0] - Y_e[mask,0])
    # Error function.
    error_function = lambda x: np.nanmean(x**2)**0.5
    # Errors for present an absent species.
    if len(species) > 0:
        for (i, name) in enumerate(species):
            with warnings.catch_warnings():
                mask = y_e[:,si+i] > 0.
                warnings.simplefilter('ignore', category=RuntimeWarning)
                print('Error in molecular fraction for present {}: {:.4f}.'
                      .format(name, error_function(errors_mols[i][mask])))
                print('Error in molecular fraction for absent {}: {:.4f}.'
                      .format(name, error_function(errors_mols[i][~mask])))
    # Errors for variables.
    errors, num_spectra = {}, {}
    if 'temp' in variables:
        errors['temp'] = error_function(errors_temp)
        print('Error in temperature (K): {:.2f}.'
              .format(error_function(errors_temp)))
    if len(species) > 0:
        for (i, var) in enumerate(species):
            errors[var] = error_function(errors_mols[i])
            num_spectra[var] = (y_train[:,si+i] != 0.).sum()
        print('Error in molecular fraction: {:.4f}.'
              .format(error_function(all_errors_mols)))
        errors['species'] = error_function(all_errors_mols)
    num_spectra = pd.DataFrame(num_spectra, index=[seed])
    # Errors table.
    errtype = 'val' if shown_dataset == 'validation' else 'train'
    errors_filename = os.path.join('models', model_name, 'errors',
                                   f'{model_name}-{errtype}-errors.csv')
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
    if save_results:
        errors_df.to_csv(errors_filename, float_format='%.4f')
        print(f'Saved file in {errors_filename}.')
     
    # Labels for each variable.
    labels = {var: formatted_names[var] for var in variables}
    labels['other'] = 'other'
    
    # Evolution of the training.
    if 1 in shown_figures:
        plt.close(1)
        plt.figure(1, figsize=(6.88,5.0))
        for var in shown_variables:
            if var == 'other':
                continue
            label = labels[var]
            history = histories[var]
            color = colors[var]
            loss = history['loss']
            epochs = np.arange(1, len(loss)+1, 1)
            plt.plot(epochs, loss, color=color, ls='--', alpha=0.6)
            plt.plot([], label=labels[var], color=color, ls='-')
            if not full_training:
                val_loss = history['val_loss']
                plt.plot(epochs, val_loss, color=color, ls='-')
            loss_ = loss if full_training else val_loss
            i = np.argmin(loss_)
            plt.plot(epochs, loss_, color=color, ls='-')
            plt.hlines(min(loss_), 1., epochs[i], color=color, ls=':')
            plt.vlines(epochs[i], 1e-9, min(loss_), color=color, ls=':')
        plt.legend(ncol=len(shown_variables), fontsize=8, loc='upper left')
        plt.yscale('log')
        plt.margins(x=0.)
        plt.ylim(5e-5, 2e1)
        plt.xlabel('epoch')
        plt.ylabel('loss (MSLE)')
        plt.title('Training process', pad=10)
        ax = plt.twinx(plt.gca())
        plt.plot([], '-', color='gray', label='validation')
        plt.plot([], '--', color='gray',  label='training')
        plt.margins(x=0.)
        plt.yticks([])
        plt.legend(loc=(0.796,0.815), fontsize=9)
        plt.tight_layout()
        filename = os.path.join('models', model_name, 'plots', str(seed),
                                 'training.png')
        if save_results:
            plt.savefig(filename)
            print(f'Saved plot in {filename}.')
    
    # Example of the model predictions.
    if 2 in shown_figures:
        plt.close(2)
        fig = plt.figure(2, figsize=(12,5))
        idx = min(idx, len(x_)-1)
        absorbance = x_[idx]
        num_positions = num_vars
        if 'temp' in shown_variables:
            num_positions -= 1
        xlabels = [formatted_names[name] for name in species]
        positions = np.arange(num_positions)
        error_mols = np.nanmean(errors_mols, axis=1)
        if 'temp' in shown_variables:
            error_temp = np.nanmean(errors_temp, axis=0)
        plt.clf()
        gs = plt.GridSpec(1, 2, width_ratios=[1.5,1.0], wspace=0.25,
                          bottom=0.15, top=0.82)
        # Left plot of the spectrum.
        fig.add_subplot(gs[0,0])
        plt.plot(wavenumber_points, absorbance, color='black', lw=1.)
        plt.axhline(y=0, color='k', ls='--', lw=0.8)
        plt.margins(x=0, y=0.02)
        plt.gca().invert_xaxis()
        plt.xlabel('wavenumber (/cm)', labelpad=8)
        plt.ylabel('normalized absorbance', labelpad=10)
        ax = plt.gca()
        ax2 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
        wavelength_ticks = [1, 2, 3, 4, 5, 7, 10, 20, 30]  # μm
        ax2.set_xticks(wavelength_ticks)
        ax2.set_xlabel('wavelength (μm)', labelpad=10, fontsize=9)
        # Right plot of the predictions.
        gs = plt.matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1],
                                                       width_ratios=[8,1], wspace=0.1)
        if show_molfracs:
            fig.add_subplot(gs[0,0])
            plt.scatter(positions, y_[idx,si:].flatten(), color='orchid', zorder=3,
                        label='real')
            plt.bar(positions, Y_[idx,si:], edgecolor='black', width=0.5,
                    color='tab:gray', label='prediction')
            plt.errorbar(positions, Y_[idx,si:], error_mols, fmt='.', color='k')
            plt.xticks(positions, xlabels, rotation=30.)
            plt.margins(x=0.03)
            plt.ylim(0., 1.)
            plt.ylabel('species fraction')
            plt.suptitle(names_[idx], y=0.98)
            plt.legend(loc='upper right')
        if show_temp:
            fig.add_subplot(gs[0,1])
            plt.scatter([1], y_[idx,0], color='orchid', zorder=3)
            plt.bar([1], Y_[idx,0], edgecolor='black', color='tab:gray')
            plt.errorbar([1], Y_[idx,0], error_temp, color='k')
            plt.ylim(0., 100.)
            plt.xticks([1], ['$T_\\mathrm{ice}$'])
            plt.gca().yaxis.tick_right()
            plt.gca().yaxis.set_label_position('right')
            plt.ylabel('temperature (K)')
            plt.margins(x=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            plt.tight_layout()
    
    # Errors histograms.
    if 3 in shown_figures:
        plt.close(3)
        plt.figure(3, figsize=(9,5))
        if show_molfracs:
            plt.subplot(1,2,1)
            for var in shown_species:
                i = int(np.where(array_species == var)[0,0])
                label = labels[var]
                color = colors[var]
                zorder = 2. if var != 'other' else 3.
                mask = mask_temp if var == 'temp' else np.ones(len(y_), bool)
                if use_logscale_for_errors:
                    eps = eps_molfrac
                    errors = np.log(Y_[mask,i+si]+eps) - np.log(y_[mask,i+si]+eps)
                else:
                    errors = Y_[mask,i+si]-y_[mask,i+si]
                plt.hist(errors, bins=20, histtype='step', label=label,
                     edgecolor=color, zorder=zorder)
            plt.xlim(left=0)
            plt.xlabel('absolute error in molecular fraction')
            plt.ylabel('number', labelpad=14)
            plt.legend(ncol=2, fontsize=8, loc='upper right')
        if show_temp:
            plt.subplot(1,2,2)
            mask = mask_temp
            plt.hist(y_[mask,0]-Y_[mask,0], bins=20, histtype='stepfilled',
                     edgecolor='black', color='gray')
            plt.gca().yaxis.tick_right()
            plt.xlim(left=0)
            plt.xlabel('absolute error in temperature (K)')
        plt.suptitle('Distributions of errors - {} dataset'.format(shown_dataset),
                     y=0.96)
        plt.tight_layout()
    
    # Correlation between predicted and actual values.
    if 4 in shown_figures:
        plt.close(4)
        fig = plt.figure(4, figsize=(10,6))
        if not show_temp:
            show_temp_as_size = False
        ms = 40.
        size = y_[:,0] if show_temp_as_size else ms
        lims = np.array([0., 1.])
        if use_logscale_for_errors:
            ticks = np.array([0., 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
            ticklabels = ['0.00', 0.04, 0.1, 0.2, '', 0.4, '', 0.6, '', '', '', 1.]
            ticks = np.log(ticks + eps_molfrac)
            lims = np.log(lims + eps_molfrac) 
        else:
            ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ticklabels = [0.0, '', 0.2, '', 0.4, '', 0.6, '', 0.8, '', 1.0]
        gs = plt.GridSpec(1, 2, width_ratios=[1,1], wspace=0.3)
        # Left plot for molecules.
        if show_molfracs:
            alpha = (0.5 if shown_dataset == 'validation'
                     else 0.6 - min(0.04*len(x_), 0.2))
            gs1 = plt.matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=gs[0,0], height_ratios=[2,1], hspace=0)
            ax = fig.add_subplot(gs1[0,0])
            for var in shown_species:
                i = int(np.where(array_species == var)[0][0])
                label = labels[var]
                color = colors[var]
                zorder = 2. if var != 'other' else 3.
                if not use_logscale_for_errors:
                    y_true, y_pred = y_[:,si+i], Y_[:,si+i]
                else:
                    y_true = np.log(y_[:,si+i] + eps_molfrac)
                    y_pred = np.log(Y_[:,si+i] + eps_molfrac)
                ax.scatter(y_true, y_pred, color=color, alpha=alpha,
                           marker='o', edgecolor='black', s=size, zorder=zorder)
                ax.scatter([], [], color=color, alpha=0.5, marker='o',
                           edgecolor='black', s=ms, label=label, zorder=zorder)
                if show_temp_as_size:
                    ax2 = ax.twinx()
                    for temp in [10, 90]:
                        ax2.scatter([], [], color=[0.9]*3, edgecolor='black',
                                    s=temp, label='{} K'.format(temp))
                        ax2.set_ylim(lims)
                        ax2.axis('off')
                        ax2.legend(loc='lower right')
                if show_spectra_names:
                    for (j, (xj, yj)) in enumerate(zip(y_true, y_pred)):
                        tj = '  ' + names_[j]
                        ax.text(xj, yj, tj, fontsize=5., alpha=alpha,
                                ha='left', va='top', rotation=-45.)
            ax.axline((0.,0.), slope=1., ls='--', color='gray')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xticks(ticks, ticklabels)
            ax.set_yticks(ticks, ticklabels)
            ax.set_ylabel('predicted molecular fraction')
            ax.legend(ncol=2, fontsize=8., loc='upper left')
            plt.setp(ax.get_xticklabels(), visible=False)
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
            # Error.
            ax = fig.add_subplot(gs1[1,0], sharex=ax)
            for var in shown_species:
                i = int(np.where(array_species == var)[0][0])
                color = colors[var]
                zorder = 2. if var != 'other' else 3.
                error_i = Y_[:,si+i] - y_[:,si+i]
                y_true = (y_[:,si+i] if not use_logscale_for_errors
                          else np.log(y_[:,si+i] + eps_molfrac))
                ax.scatter(y_true, error_i, color=color,
                           alpha=alpha, marker='o', label=var,
                           edgecolor='black', s=size, zorder=zorder) 
            ax.axhline(y=0, ls='--', color='gray')
            use_symlog_scale = False
            if use_symlog_scale:
                for yi in (-0.1, 0.1):
                    ax.axhline(y=yi, ls='--', color=[0.7]*3)
                ax.set_ylim([-1., 8.])
                ax.set_yscale('symlog', linthresh=0.1)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_func))
            else:
                ax.set_ylim([-0.23,0.27])
            ax2 = ax.secondary_xaxis('top')
            ax2.tick_params(axis='x', top=True, bottom=False, direction='in',
                            labeltop=False)
            ax2.set_xticks(ticks, ticklabels)
            ax.set_xlabel('labeled molecular fraction')
            ax.set_ylabel('error')
            plt.sca(ax)
        # Right plot for temperature.
        if show_temp:
            alpha = 0.3 if shown_dataset == 'validation' else 0.2
            gs2 = plt.matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=gs[0,1], height_ratios=[2,1], hspace=0)
            ax = fig.add_subplot(gs2[0,0])
            mask = mask_temp
            plt.scatter(y_[mask,0], Y_[mask,0], s=size[mask],
                        color=[0.2]*3, alpha=alpha)
            plt.axline((0.,0.), slope=1., ls='--', color='gray')
            plt.xlim([0., 100.])
            plt.ylim([0., 100.])
            plt.ylabel('predicted temperature (K)')
            plt.setp(ax.get_xticklabels(), visible=False)
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                ax2 = ax.twinx()
                for temp in [10, 90]:
                    ax2.scatter([], [], color=[0.9]*3, edgecolor='black', s=temp,
                               label='{} K'.format(temp))
                    ax2.set_ylim([0,1])
                    ax2.axis('off')
                    ax2.legend(loc='lower right')
            if show_spectra_names:
                for (j, (xj, yj)) in enumerate(zip(y_[mask,0], Y_[mask,0])):
                    tj = '  ' + names_[mask][j].split(' ')[0]
                    ax.text(xj, yj, tj, fontsize=5., alpha=alpha,
                            ha='left', va='top', rotation=-45.)
            # Error.
            ax = fig.add_subplot(gs2[1,0], sharex=ax)
            errort = Y_[mask,0] - y_[mask,0]
            plt.scatter(y_[mask,0], errort, color=[0.2]*3, s=size[mask],
                        alpha=alpha)
            ax.axhline(y=0., ls='--', color='gray')
            plt.ylim(-40., 57.)
            ax2 = ax.secondary_xaxis('top')
            ax2.tick_params(axis='x', top=True, bottom=False, direction='in',
                            labeltop=False)
            plt.xlabel('labeled temperature (K)')
            plt.ylabel('error (K)')
        plt.suptitle('Correlation between predicted and actual values - {} dataset'
                     .format(shown_dataset), y=0.95)
        suffix = '-train' if shown_dataset == 'training' else ''
        filename = os.path.join('models', model_name, 'plots', str(seed),
                                f'validation{suffix}.png')
        if save_results:
            plt.savefig(filename)
            print(f'Saved plot in {filename}.')
    
    # Error versus number of training data.
    if show_molfracs and 5 in shown_figures:
        plt.close(5)
        plt.figure(5)
        xlabels = num_spectra[species]
        for (i,name) in enumerate(species):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                mask = y_[:,si+i] > 0.
                error_present = np.nanmean(errors_mols[mask,i])
                error_absent = np.nanmean(errors_mols[~mask,i])
                plt.scatter(num_spectra[name], errors_df.at[seed,name],
                            color=colors[name], label=labels[name])
                plt.scatter(num_spectra[name], error_present, color=colors[name],
                            marker='*')
                plt.scatter(num_spectra[name], error_absent, color=colors[name],
                            marker='o', facecolors='none')
        plt.ylim(0., 0.15)
        plt.xlabel('number of spectra in training subset')
        plt.ylabel('{} error'.format(shown_dataset))
        plt.title('Relation between error and training spectra for species', pad=10)
        plt.legend(loc='upper right', fontsize='small')
        ax2 = plt.gca().twinx()
        plt.yticks([])
        plt.plot([], 'o', color='gray', label='total')
        plt.plot([], '*', color='gray', label='present')
        plt.plot([], 'o', color='gray', fillstyle='none', label='absent')
        plt.legend(loc='upper left', fontsize='small')
        plt.tight_layout()
        fractions_train = np.unique(y_train[:,si:], axis=0)
        fractions_val = np.unique(y_train[:,si:], axis=0)
        fractions = np.concatenate((fractions_train, fractions_val))
        train_ratios = []
        for fracs in fractions:
            num_train = sum([np.array_equal(yi[si:], fracs) for yi in y_train])
            num_val = sum([np.array_equal(yi[si:], fracs) for yi in y_val])
            num_total = num_train + num_val
            train_ratio = num_train / num_total
            train_ratios += [train_ratio]
        frac_labels = [('{:.1f}'*len(species)).format(*fracs) for fracs in fractions]
        frac_positions = np.arange(len(fractions))
    
    plt.show()
    
    #%% Saving individual models.
    
    if train_nn and save_results:
        info_folder = os.path.join('models', model_name, 'model-info')
        if not os.path.exists(info_folder):
            os.makedirs(info_folder)
        # Models for each variable.
        for var in variables:
            model = models[var]
            history = histories[var]
            weights_i = []
            for w in model.weights:
                weights_i += [w.numpy()]
            weights_i = np.array(weights_i, object)
            filename = os.path.join('models', model_name, 'weights', str(seed),
                                    f'weights-{var}.npy')
            np.save(filename, weights_i)
            print(f'Saved file {filename}.')
            filename = os.path.join('models', model_name, 'histories', str(seed),
                                    f'history-{var}.pkl')
            with open(filename, 'wb') as file:
                pickle.dump(history, file)
            print(f'Saved file {filename}.')
            # Information about the model.
            model_info = [
                f'Model: {model_name}',
                '',
                f'variable: {var}',
                f'spectral range (/cm): ({wavenumber_points[0]}, {wavenumber_points[-1]})',
                f'spectral spacing (/cm): {spectral_spacing}',
                f'spectral resolution (/cm): {resolution}',
                'use linear combinations dataset in training/validation: ' + \
                    f'{use_lincombs_dataset}/{use_lincombs_in_validation}',
                'simulate saturation in training/validation: ' + \
                    f'{simulate_saturation}/{simulate_saturation_in_validation}',
                f'training fraction for laboratory: {training_fraction}',
                f'training fraction for linear combinations: {training_fraction_lc}',
                f'maximum number of epochs: {num_epochs}',
                f'number of batches: {num_batches[var]}',
                f'dropout probability: {dropout_prob[var]}',
                f'ReLU negative slope: {relu_negative_slope[var]}', 
                ''
            ]
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            model_info += model_summary[0].split('\n')
            for (i, line) in enumerate(model_info):
                if line.startswith('Model: ') and i > 0:
                    model_info[i] = 'Structure:'
            filename = os.path.join(info_folder, f'model-info-{var}.txt')
            with open(filename, 'w') as file:
                for line in model_info:
                    file.write(f'{line}\n')
            print(f'Saved file {filename}.')
    
print()
            
# Time check.         
final_time = time.time()
elapsed_time = final_time - initial_time
print(f'Total elapsed time: {elapsed_time//60:.0f} min + {elapsed_time%60:.0f} s.')
