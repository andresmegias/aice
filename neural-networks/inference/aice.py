#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.0
------------------------------------------
Inference module

Andrés Megías
"""

config_file = 'NIR38.yaml'

# Libraries.
import os
import sys
import yaml
import spectres
import numpy as np
import pandas as pd
import richvalues as rv
import matplotlib.pyplot as plt

# Functions.

relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))

def supersample_spectrum(new_wavs, spec_wavs, spec_fluxes, spec_errs=None,
                         fill=np.nan):
    """Supersample spectrum to the input points."""
    new_fluxes = np.interp(new_wavs, spec_wavs, spec_fluxes,
                           left=fill, right=fill)
    if spec_errs is None:
        new_errs = None
    else:
        new_errs = np.zeros(len(new_wavs))
        for (i,xi) in enumerate(new_wavs):
            if xi <  min(spec_wavs) or xi > max(spec_wavs):
                new_errs[i] = fill
            elif any(spec_wavs == xi):
                new_errs[i] = spec_errs[np.argwhere(spec_wavs==xi)[0][0]]
            else:
                inds = np.argsort(np.abs(spec_wavs - xi))[:2]
                x1x2 = spec_wavs[inds]
                x1, x2 = min(x1x2), max(x1x2)
                mask = (new_wavs > x1) & (new_wavs < x2)
                num_points = np.sum(mask)
                err1 = spec_errs[np.argwhere(spec_wavs == x1)][0][0]
                err2 = spec_errs[np.argwhere(spec_wavs == x2)][0][0]
                err_m = np.mean([err1, err2])
                new_errs[i] = np.sqrt(num_points) * err_m
        return new_fluxes, new_errs
    
def resample_spectrum(new_wavs, spec_wavs, spec_fluxes, spec_errs=None,
                      fill=np.nan, verbose=True):
    """Resample spectrum to the input points."""
    mask = (new_wavs >= min(spec_wavs)) & (new_wavs <= max(spec_wavs))
    if len(new_wavs[mask]) <= len(spec_wavs):
        new_fluxes, new_errs = spectres.spectres(new_wavs, spec_wavs,
                                    spec_fluxes, spec_errs, fill, verbose)
    else:
        new_fluxes, new_errs = supersample_spectrum(new_wavs, spec_wavs,
                                                spec_fluxes, spec_errs, fill)
    return new_fluxes, new_errs

def axis_conversion(x):
    """Axis conversion from wavenumber to wavelength and viceversa"""
    with np.errstate(divide='ignore'):
        y = 1e4 / x
    return y

#%% Initial options.

print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('')

# Default options.
default_options = {
    'input spectrum': '',
    'column indices': {'x': 1, 'y': 2, 'y unc.': 3},
    'figure size': [12., 5.],
    'save results': True,
    'output file': 'auto',
    'model weights': os.path.join('..', 'training', 'models', 'aice-weights.npy'),
    'spectral variable': 'wavenumber (/cm)',
    'intensity variable': 'absorbance',
    'wavenumber range (/cm)': [4001., 980., 1.],
    'resampling edges (/cm)': [],
    'target variables': {0: 'temp. (K)', 1: 'H2O', 2: 'CO', 3: 'CO2',
                         4: 'CH3OH', 5: 'NH3', 6: 'CH4'},
    'formatted names': {'H2O': 'H$_2$O', 'CO': 'CO', 'CO2': 'CO$_2$',
                        'CH3OH': 'CH$_3$OH', 'NH3': 'NH$_3$', 'CH4': 'CH$_4$'},
    'spectrum vertical range': None,
    'molecular fraction range': [0., 1.0],
    'temperature range (K)': [0., 100.],
    'show predictions for all submodels': False,
    'propagate observational uncertainties': False,
    'normalization': 'total',
    'reference values': {},
    'used reference values': [],
    'reference colors': ['orchid', 'orange', 'palevioletred'],
    'AICE label': 'AICE'
    }

# Configuration file.
spectrum_name = config_file.replace('.yaml', '').replace('.', '')
config_file = (os.path.realpath(config_file) if len(sys.argv) == 1
               else os.path.realpath(sys.argv[1]))
with open(config_file) as file:
    config = yaml.safe_load(file)
config = {**default_options, **config}

# Options.
file = config['input spectrum']
column_inds = config['column indices']
figsize = config['figure size']
weights_path = config['model weights']
spectral_variable = config['spectral variable']
intensity_variable = config['intensity variable']
wavenumber_range = config['wavenumber range (/cm)']
resampling_edges = config['resampling edges (/cm)']
target_vars = config['target variables']
formatted_names = config['formatted names']
frac_range = config['molecular fraction range']
absorb_range = config['spectrum vertical range']
temp_range = config['temperature range (K)']
normalization = config['normalization']
all_references = config['reference values']
used_references = config['used reference values']
aice_label = config['AICE label']
show_individual_predictions = config['show predictions for all submodels']
propagate_obs_uncs = config['propagate observational uncertainties']
ref_colors = config['reference colors']
save_results = config['save results']
output_file = config['output file']
show_references = used_references != []
x1, x2, dx = wavenumber_range
x1x2 = x1, x2
x1, x2 = min(x1x2), max(x1x2)
wavenumber = np.arange(x1, x2, dx)
if resampling_edges == []:
    resampling_edges = [x1, x2]
else:
    if resampling_edges[0] != x1:
        resampling_edges = [x1] + resampling_edges
    if resampling_edges[-1] != x2:
        resampling_edges = resampling_edges + [x2]
inds = list(target_vars.keys())
num_vars = len(target_vars)
variables = list(target_vars.values())
species = [var for var in variables if 'temp' not in var]
idx_x, idx_y = column_inds['x'] - 1, column_inds['y'] - 1
idx_dy = column_inds['y unc.'] - 1 if 'y unc.' in column_inds else None


#%% Loading of files.

# Reading of input spectrum.
if '.csv' in file:
    data = pd.read_csv(file).values
    x, y = data[:,[idx_x, idx_y]]
    data[:,idx_dy] if idx_dy is not None else np.zeros(len(y))
else:
    data = np.loadtxt(file)
x = data[:,idx_x]
y = data[:,idx_y]
dy = data[:,idx_dy] if idx_dy is not None else np.zeros(len(y))
if spectral_variable == 'wavelength (μm)':
    x = 1e4 / x
if intensity_variable == 'optical depth':
    y /= np.log(10)
    dy /= np.log(10)
inds = np.argsort(x)
x = x[inds]
y = y[inds]
dy = dy[inds]

# Resampling.
x_, y_, dy_ = np.array([]), np.array([]), np.array([])
for i in range(len(resampling_edges)-1):
    xi1, xi2 = resampling_edges[i], resampling_edges[i+1]
    mask = (x >= xi1) & (x <= xi2)
    mask_ = (wavenumber >= xi1) & (wavenumber <= xi2)
    xi = wavenumber[mask_]
    yi, dyi = resample_spectrum(xi, x[mask], y[mask], dy[mask], verbose=False)
    x_ = np.append(x_, xi)
    y_ = np.append(y_, yi)
    dy_ = np.append(dy_, dyi)
absorbance = np.interp(wavenumber, x_, y_, left=0., right=0.)
absorbance_unc = np.interp(wavenumber, x_, dy_, left=0., right=0.)
mask_nan = np.isnan(absorbance)
absorbance = np.nan_to_num(absorbance, nan=0.)
absorbance_unc = np.nan_to_num(absorbance_unc, nan=0.)

# Normalization.
norm = np.mean(absorbance)
absorbance /= norm
absorbance_unc /= norm

print(f'Read spectrum in file {file}.')
print()

# %% Preparation of the neural network model.

# Model weights.
weights = np.load(weights_path, allow_pickle=True)
inds = list(target_vars.keys())
if len(weights.shape) == 2:
    weights = weights.reshape(1,*weights.shape)
weights = weights[:,inds,:]

# Multi-layer perceptron.
def model_i(x, weights, end_act=sigmoid):
    """Apply a multi-layer perceptron to input vector x."""
    w = weights
    w1, b1 = w[0], w[1]  # first hidden layer weights
    ga1, be1, m1, s1 = w[2], w[3], w[4], w[5]  # first batch-norm weights
    w2, b2 = w[6], w[7]  # second  hidden layer weights
    ga2, be2, m2, s2 = w[8], w[9], w[10], w[11]  # second batch-norm weights
    w3, b3 = w[12], w[13]  # third hidden layer weights
    ga3, be3, m3, s3 = w[14], w[15], w[16], w[17]  # third batch-norm weights
    w4, b4 = w[18], w[19]  # final layer weights
    e = 1e-3  # correction for bath-norm variance
    a1 = relu(np.dot(w1.T, x) + b1)  # first hidden layer
    a1 = ga1 * (a1 - m1) / (s1 + e)**0.5 + be1  # first batch-norm
    a2 = relu(np.dot(w2.T, a1) + b2)  # second hidden layer
    a2 = ga2 * (a2 - m2) / (s2 + e)**0.5 + be2  # second batch-norm
    a3 = relu(np.dot(w3.T, a2) + b3)  # third hidden layer
    a3 = ga3 * (a3 - m3) / (s3 + e)**0.5 + be3  # third batch-norm
    y = end_act(np.dot(w4.T, a3) + b4)  # final layer
    return y

# Neural network model.
def nn_model(x, weights):
    """Neural network model the targeted variables."""
    ya = []
    for j in range(len(weights)):
        yj = np.zeros(num_vars)
        for (i,var) in target_vars.items():
            end_act = relu if 'temp' in var else sigmoid
            yj[i] = model_i(x, weights[j,i], end_act)[0]
        ya += [yj]
    ya = np.array(ya)
    dy = np.std(ya, ddof=1, axis=0)
    y = np.mean(ya, axis=0)
    return y, dy, ya

#%% Calculations of AICE.

# Predictions.
predictions, stdevs, predictions_all = nn_model(absorbance, weights)
if propagate_obs_uncs:
    print('Propagating observational uncertainties...\n')
    predictions_rv = rv.array_function(lambda x: nn_model(x, weights)[0],
                        rv.RichArray(absorbance, absorbance_unc),
                    domain=[0,np.inf], len_samples=400, consider_intervs=False)
    obs_uncs = predictions_rv.uncs
if (propagate_obs_uncs and any(rv.isnan(predictions).flatten())
        or not propagate_obs_uncs):
    obs_uncs = 0.

# Uncertainty estimation.
stdevs = np.array([stdevs, stdevs]).T  # deviation of nn-predictions
uncs = (obs_uncs**2 + stdevs**2)**0.5
predictions = rv.RichArray(predictions, uncs, domains=[0,np.inf])

# Preparation of results dataframe.
results_df = rv.rich_dataframe({aice_label: predictions}, index=variables).T

# Normalization.
if normalization != 'total':
    norm_results = {aice_label: results_df[normalization].values[0]}
    other_species = [name for name in species if name != normalization]
    results_df[other_species] = (results_df[other_species].values
                                 / results_df[normalization].values)

# Adjustment of results dataframe.
results_df = results_df.T

# Implementation of input references in results dataframe.
if show_references:
    references = {}
    for key1 in all_references:
        if any([key1.startswith(key2) for key2 in used_references]):
            references[key1] = all_references[key1]
    for key in references:
        reference_df = rv.rich_dataframe(references[key], index=[0],
                                         domains=[0,np.inf])
        norm = 0.
        for name in reference_df:
            if 'temp' not in name:
                norm += reference_df[name].values[0]
        for name in reference_df:
            if 'temp' not in name:
                if name not in species:
                    reference_df = reference_df.drop(columns=name)
                    continue
                reference_df[name] /= norm
        for name in results_df[aice_label].keys():
            if name not in reference_df.columns:
                reference_df[name] = rv.rval('nan')
        if normalization != 'total':
            norm_results[key] = reference_df[normalization].values[0]
            with np.errstate(divide='ignore', invalid='ignore'):
                reference_df[other_species] = (reference_df[other_species].values
                                               / reference_df[normalization].values)
        reference_df = reference_df.transpose()
        reference_df.columns = [key]
        results_df = pd.concat((results_df, reference_df), axis=1)
        
# Change of names for normalized fractions.
if normalization != 'total':
    old_vars = list(results_df.index)
    new_vars = []
    for name in old_vars:
        if name == normalization or 'temp' in name:
            new_vars += [name]
        else:
            new_vars += [f'{name}/{normalization}']
    results_df.index = new_vars
        
#%% Displaying predictions and plots.

plt.close('all')    
print('Predictions for the ice composition:')
print()

# Result dataframe.
with pd.option_context('display.max_columns', 4):
    print(results_df, '\n')
if normalization == 'total':
    print(f'Sum of predicted molecules: {results_df.values[1:,0].sum()}')

# Graphic options.
if show_references:  # offsets for predictions
    refs_names = list(references.keys())
    num_refs = len(refs_names)
    if num_refs == 1:
        offsets = {aice_label: -0.15, refs_names[0]: 0.15}
    elif num_refs == 2:
        offsets = {aice_label: 0, refs_names[0]: -0.15, refs_names[1]: 0.15}
    elif num_refs == 3:
        offsets = {aice_label: -0.10, refs_names[0]: 0.10,
                   refs_names[1]: -0.30, refs_names[2]: 0.30}
else:
    offsets = {aice_label: 0}
fig = plt.figure(1, figsize=figsize)   # figure dimensions and ratios
width_ratios = [2.,1.] if normalization == 'total' else [3.,2.]
width_ratios[-1] = width_ratios[-1] - (6. - len(species))/6.
gs = plt.GridSpec(1, 2, width_ratios=width_ratios, wspace=0.18,
                  left=0.07, right=0.93, bottom=0.15, top=0.82)
absorbance[mask_nan] = np.nan
absorbance_unc[mask_nan] = np.nan

# Plot of the spectrum.
fig.add_subplot(gs[0,0])
plt.errorbar(wavenumber, absorbance, absorbance_unc, linewidth=1.,
             color='black', ecolor='gray', label='observations')
plt.axhline(y=0, color='k', ls='--', lw=0.6)
if absorb_range is not None:
    plt.ylim(absorb_range)
plt.xlim(wavenumber_range[0], wavenumber_range[1])
plt.xlabel('wavenumber (cm$^{-1}$)', labelpad=8,)
plt.ylabel('normalised absorbance', labelpad=10)
plt.legend(loc='upper right')
ax = plt.gca()  
ax2 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
               [1, 2, 3, 4, 5, '', 7, '', '', 10, 20, 30])
ax2.set_xlabel('wavelength (μm)', labelpad=10, fontsize=9)

# Predicted fraction for normalization molecule.
if normalization != 'total':
    norm_results = pd.DataFrame(norm_results, index=[normalization])
    gs = plt.matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, gs[1],
                                               width_ratios=[1.,5.], wspace=0.5)
    fig.add_subplot(gs[0,0])
    rv.errorbar([0. + offsets[aice_label]], norm_results[aice_label],
                fmt='.', color='black')
    plt.bar([0], norm_results[aice_label].values[0].main,
            edgecolor='black', color='gray')
    if show_references:
        for (i,key) in enumerate(references):
            reference = norm_results[key].values
            rv.errorbar([0. + offsets[key]], reference[0], color=ref_colors[i])
    plt.margins(x=0.7)
    plt.ylim(frac_range)
    plt.xticks([0], [formatted_names[normalization]], rotation=45.,
               fontsize=10.)
    plt.ylabel('molecular fraction')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
gs = plt.matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, gs[1],
                                            width_ratios=[8.,1.], wspace=0.15)
# Predictions bar plot.
fig.add_subplot(gs[0,0])
predictions = []
for name in results_df[aice_label].index:
    if name != normalization:
        predictions += [results_df[aice_label][name]]
predictions = rv.rarray(predictions)
species = [name for name in species if name != normalization]
labels = [formatted_names[name] for name in species]
positions = np.arange(len(labels))
plt.bar(positions, predictions.mains[1:], width=0.6,  # ice composition
        edgecolor='black', color='gray')
rv.errorbar(positions + offsets[aice_label], predictions[1:], fmt='.',
            color='black')
plt.plot([], [], '.', color='black', label='AICE (this work)')
if show_individual_predictions:
    num_seeds = len(predictions_all)
    for (i, predictions_i) in enumerate(predictions_all):
        offset_i = (i / (num_seeds-1) - 0.5) * 0.6
        plt.plot(positions + offset_i, predictions_i[1:], '.',
                 color='black', alpha=0.3)
if show_references:  # reference
    for (i,key) in enumerate(references):
        reference = []
        for name in (results_df[key].index):
            if name != normalization and 'temp' not in name:
                reference += [results_df[key][name]]
        rv.errorbar(positions + offsets[key], reference, color=ref_colors[i])
        plt.plot([], [], '.', color=ref_colors[i], label=key)
plt.xticks(positions, labels, rotation=45., fontsize=10.)
plt.xlim(positions[0]-0.7, positions[-1]+0.7)
for tick in plt.gca().xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
plt.ylim(frac_range)
ylabel = ('molecular fraction' if normalization == 'total' else
          'abundance with respect to '+ formatted_names[normalization])
plt.ylabel(ylabel, labelpad=4)
plt.legend(fontsize=8)
title = 'AICE predictions  '
if normalization != 'total':
    title += ' '*26
plt.title(title, x=0.55, pad=16.)
fig.add_subplot(gs[0,1])  # temperature
plt.bar([1.], predictions[0].main, edgecolor='black', color='tab:gray')
rv.errorbar([1. + offsets[aice_label]], predictions[0], color='black')
if show_individual_predictions:
    num_seeds = len(predictions_all)
    for (i, predictions_i) in enumerate(predictions_all):
        offset_i = (i / (num_seeds-1) - 0.5) * 0.6
        plt.plot(1. + offset_i, predictions_i[0], '.',
                 color='black', alpha=0.3)
if show_references:  # reference
    for (i,key) in enumerate(references):
        reference = results_df[key].values
        if rv.isfinite(reference[0]):
            rv.errorbar([1. + offsets[key]], reference[0], color=ref_colors[i])
plt.ylim(temp_range)
plt.locator_params(axis='y', nbins=5)
plt.xticks([1], ['$T_\\mathrm{ice}$'], rotation=0., fontsize=10.)
plt.tick_params(axis='x', which='major', pad=8.)
for tick in plt.gca().xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position('right')
plt.ylabel('temperature (K)')
plt.margins(x=0.9)

plt.suptitle(spectrum_name, y=0.94, fontweight='bold')

# plt.tight_layout()

plt.show()

#%% Save output.

if save_results:
    if output_file == 'auto':
        output_file = config_file.replace('.yaml', '-aice-results.csv')
    results_df.to_csv(output_file)
    print(f'\nSaved results in {output_file}.')