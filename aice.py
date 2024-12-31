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
import re
import sys
import copy
import yaml
import pickle
import numpy as np
import pandas as pd
import richvalues as rv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Functions.

relu = lambda x: np.maximum(0, x)
sigmoid = lambda x: 1 / (1 + np.exp(-x))

def format_species_name(input_name, simplify_numbers=True):
    """
    Format text as a molecule name, with subscripts and upperscripts.

    Parameters
    ----------
    input_name : str
        Input text.
    simplify_numbers : bool, optional
        Remove the brackets between single numbers.

    Returns
    -------
    output_name : str
        Formatted molecule name.
    """
    original_name = copy.copy(input_name)
    # upperscripts
    possible_upperscript, in_upperscript = False, False
    output_name = ''
    upperscript = ''
    inds = []
    for i, char in enumerate(original_name):
        if (char.isupper() and not possible_upperscript
                and '-' in original_name[i:]):
            inds += [i]
            possible_upperscript = True
        elif char.isupper():
            inds += [i]
        if char == '-' and not in_upperscript:
            inds += [i]
            in_upperscript = True
        if not possible_upperscript:
            output_name += char
        if in_upperscript and i != inds[-1]:
            if char.isdigit():
                upperscript += char
            if char == '-' or i+1 == len(original_name):
                if len(inds) > 2:
                    output_name += original_name[inds[0]:inds[-2]]
                output_name += ('$^{' + upperscript + '}$'
                                + original_name[inds[-2]:inds[-1]])
                upperscript = ''
                in_upperscript, possible_upperscript = False, False
                inds = []
    if output_name.endswith('+') or output_name.endswith('-'):
        symbol = output_name[-1]
        output_name = output_name.replace(symbol, '$^{'+symbol+'}$')
    original_name = copy.copy(output_name)
    # subscripts
    output_name, subscript, prev_char = '', '', ''
    in_bracket = False
    for i,char in enumerate(original_name):
        if char == '{':
            in_bracket = True
        elif char == '}':
            in_bracket = False
        if (char.isdigit() and not in_bracket
                and prev_char not in ['=', '-', '{', ',']):
            subscript += char
        else:
            if len(subscript) > 0:
                output_name += '$_{' + subscript + '}$'
                subscript = ''
            output_name += char
        if i+1 == len(original_name) and len(subscript) > 0:
            output_name += '$_{' + subscript + '}$'
        prev_char = char
    output_name = output_name.replace('^$_', '$^').replace('$$', '')
    # vibrational numbers
    output_name = output_name.replace(',vt', ', vt')
    output_name = output_name.replace(', vt=', '$, v_t=$')
    output_name = output_name.replace(',v=', '$, v=$')
    for i,char in enumerate(output_name):
        if output_name[i:].startswith('$v_t=$'):
            output_name = output_name[:i+5] + \
                output_name[i+5:].replace('$_','').replace('_$','')
    # some formatting
    output_name = output_name.replace('$$', '').replace('__', '')
    # remove brackets from single numbers
    if simplify_numbers:
        single_numbers = re.findall('{(.?)}', output_name)
        for number in set(single_numbers):
            output_name = output_name.replace('{'+number+'}', number)
    return output_name

def generate_spectrum(wavenumber, coeffs, temp):
    """
    Generate a synthetic spectrum in the given wavenumber range.
    """
    original_wavenumber = np.arange(980., 4001., 1.)
    new_spectrum = np.zeros(len(wavenumber))
    for key in base_spectra:
        spectra = base_spectra[key]
        temps = base_temps[key]
        coeff = coeffs[key] if key in coeffs else 0.
        interp = interp1d(temps, spectra, axis=1,
                          bounds_error=False, fill_value=0.)
        temp = max(temp, min(temps))
        temp = min(temp, max(temps))
        spectrum = interp(temp)
        spectrum = np.interp(wavenumber, original_wavenumber, spectrum)
        new_spectrum += coeff * spectrum
    return new_spectrum

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
    'figure size': [12., 5.],
    'save results': False,
    'model weights': '../training/aice-weights.npy',
    'model errors': '../training/aice-errors.csv',
    'use logarithmic errors': False,
    'spectral variable': 'wavenumber (/cm)',
    'intensity variable': 'absorbance',
    'wavenumber range (/cm)': [4001., 980., 1.],
    'target variables': {0: 'temp. (K)', 1: 'H2O', 2: 'CO', 3: 'CO2',
                         4: 'CH3OH', 5: 'NH3', 6: 'CH4'},
    'molecular fraction range': [0., 1.0],
    'temperature range (K)': [0., 100.],
    'plot molecular contributions': True,
    'propagate observational uncertainties': False,
    'normalization': 'total',
    'reference values': {},
    'reference colors': ['orchid', 'tab:orange', 'palevioletred'],
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
figsize = config['figure size']
weights_path = config['model weights']
errors_path = config['model errors']
use_logerrors = config['use logarithmic errors']
spectral_variable = config['spectral variable']
intensity_variable = config['intensity variable']
target_vars = config['target variables']
frac_range = config['molecular fraction range']
temp_range = config['temperature range (K)']
plot_molecular_contributions = config['plot molecular contributions']
normalization = config['normalization']
references = config['reference values']
aice_label = config['AICE label']
propagate_obs_uncs = config['propagate observational uncertainties']
ref_colors = config['reference colors']
save_results = config['save results']
show_references = references != {}
if not file.startswith('./'):
    file = './' + file
if not use_logerrors and 'log' in errors_path.split('/')[-1]:
    use_logerrors = True
x1, x2, dx = config['wavenumber range (/cm)']
x1x2 = x1, x2
x1, x2 = min(x1x2), max(x1x2)
wavenumber = np.arange(x1, x2, dx)
inds = list(target_vars.keys())
num_vars = len(target_vars)
variables = list(target_vars.values())
species = [var for var in variables if 'temp' not in var]

#%% Loading of files.

# Reading of input spectrum.
if '.csv' in file:
    data = pd.read_csv(file).values
else:
    data = np.loadtxt(file)
x = data[:,0]
y = data[:,1]
dy = data[:,2] if data.shape[1] == 3 else np.zeros(len(y))
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
absorbance = np.interp(wavenumber, x, y, left=0., right=0.)
absorbance_unc = np.interp(wavenumber, x, dy, left=0., right=0.)
absorbance = np.nan_to_num(absorbance, nan=0.)
absorbance_unc = np.nan_to_num(absorbance_unc, nan=0.)

# Normalization.
norm = np.mean(absorbance)
absorbance /= norm
absorbance_unc /= norm

print('Read spectrum in file {}.\n'.format(file))

# %% Preparation of the neural network model.

# Model weights.
weights = np.load(weights_path, allow_pickle=True)
inds = list(target_vars.keys())
if len(weights.shape) == 2:
    weights = weights.reshape(1,*weights.shape)
weights = weights[:,inds,:]

# Multi-layer perceptron.
def modeli(x, weights, end_act=sigmoid):
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
def nnmodel(x, weights):
    """Neural network model with for all the targeted variables."""
    y = []
    for j in range(len(weights)):
        yj = np.zeros(num_vars)
        for (i,var) in target_vars.items():
            fit_temp = True if 'temp' in var else False
            end_act = relu if fit_temp else sigmoid
            x_ = copy.copy(x)
            yj[i] = modeli(x_, weights[j,i], end_act)[0]
        y += [yj]
    y = np.array(y)
    dy = np.std(y, ddof=1, axis=0)
    y = np.mean(y, axis=0)
    return y, dy

#%% Calculations of AICE.

# Predictions.
predictions, stdevs = nnmodel(absorbance, weights)
if propagate_obs_uncs:
    print('Propagating observational uncertainties...\n')
    predictions_rv = rv.array_function(lambda x: nnmodel(x, weights)[0],
                        rv.RichArray(absorbance, absorbance_unc),
                    domain=[0,np.inf], len_samples=400, consider_intervs=False)
    uncs = predictions_rv.uncs
if (propagate_obs_uncs and any(rv.isnan(predictions).flatten())
        or not propagate_obs_uncs):
    uncs = 0.
predictions = rv.RichArray(predictions, uncs, domains=[0,np.inf])

# Uncertainty estimation.
stdevs = np.array([stdevs, stdevs]).T  # deviation of nn-predictions
errors_df = pd.read_csv(errors_path, index_col=0)
errors = []  # intrinsec model uncs.
for name in variables:
    name = name.split(' ')[0].replace('.', '')
    errors += [errors_df[name].values.mean()]
errors = np.array(errors)
if use_logerrors:
    errors = (predictions.mains + 1) * (np.exp(errors)-1)
errors = 0*np.array([errors, errors]).T
uncs = (predictions.uncs**2 + np.maximum(errors, stdevs)**2)**0.5
num_seeds = weights.shape[0]
uncs_r = (predictions.uncs**2 + stdevs**2)**0.5  # reduced uncs.
predictions_mains = predictions.mains
predictions = rv.RichArray(predictions_mains, uncs)
for (i,rval) in enumerate(predictions):  # correction for lower uncs.
    y = rval.main
    s = uncs[i][0]
    sr = uncs_r[i][0]
    if y/s > 2.:
        unc1 = s
    elif y/s < 1.:
        unc1 = sr
    else:
        unc1 = s*(y/s-1.) + sr*(2.-y/s)
    a = min(max(0., 2.-y/s), 1.)
    predictions[i].unc[0] = unc1

# Preparation of results dataframe.
results_df = rv.rich_dataframe({aice_label: predictions}, index=variables).T

# Estimations of molecular contributions.
if plot_molecular_contributions:
    with open('pure-ices-spectra.pkl', 'rb') as file:
        base_spectra, base_temps = pickle.load(file)
    temp = predictions[0].main
    fractions = predictions[1:].mains
    coeffs = {name: frac for (name,frac) in zip(species, fractions)}
    synthetic_spectrum = generate_spectrum(wavenumber, coeffs, temp)
    norm_total = np.mean(synthetic_spectrum)
    contributions = np.zeros((len(species), len(wavenumber)))
    for (i,(name,coeff)) in enumerate(zip(species, fractions)):
        spectrum_i = generate_spectrum(wavenumber, {name: coeff}, temp)
        spectrum_i /= norm_total
        contributions[i,:] = spectrum_i
        if name == 'H2O':
            contributions[i,:] *= 1/2
    synthetic_spectrum /= norm_total
    colors = {'H2O': 'tab:blue', 'CO': 'tab:red', 'CO2': 'tab:orange',
              'CH3OH': 'tab:green', 'NH3': 'tab:cyan', 'CH4': 'tab:purple'}

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
            new_vars += ['{}/{}'.format(name, normalization)]
    results_df.index = new_vars
        
#%% Displaying predictions and plots.

plt.close('all')    
print('Predictions for the ice composition:')
print()

# Result dataframe.
with pd.option_context('display.max_columns', 4):
    print(results_df, '\n')
if normalization == 'total':
    print('Sum of predicted molecules: {}'.format(results_df.values[1:,0].sum()))

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

# Plot of the spectrum.
fig.add_subplot(gs[0,0])
plt.errorbar(wavenumber, absorbance, absorbance_unc, linewidth=1.,
             color='black', ecolor='gray', label='observations')
plt.axhline(y=0, color='k', ls='--', lw=0.6)
if plot_molecular_contributions:  # molecular contributions in colors
    inds = np.argmax(contributions, axis=0)
    y1, y2 = plt.ylim()
    for (i,name) in enumerate(species):
        plt.fill_between(wavenumber, absorbance, where=inds==i,
                          color=colors[name], alpha=0.8)
    plt.ylim(y1, y2)
    ax = plt.gca()
    loc = (0.36 - 0.02*(6.-len(species))/3., 0.96)
    plt.text(*loc, ' predicted main contribution : \n\n\n\n',
             ha='right', va='top', transform=ax.transAxes,
             bbox=dict(edgecolor=[0.8]*3, facecolor='white', boxstyle='round'))
    ax2 = plt.twinx(ax)
    for name in species:
        text = format_species_name(name)
        if name == 'H2O':
            text += ' (½)'
        plt.plot([], color=colors[name], label=text)
    plt.axis('off')
    loc = [0.027, 0.71]
    if normalization != 'total':
        loc[0] -= 0.034
    plt.legend(loc=loc, ncols=2, fontsize=9.5, facecolor='none', edgecolor='none')
    plt.sca(ax)
plt.margins(x=0.03, y=0.02)
plt.xlim(plt.xlim()[::-1])
plt.xlabel('wavenumber (cm$^{-1}$)', labelpad=8,)
plt.ylabel('normalized absorbance', labelpad=10)
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
    plt.xticks([0], [format_species_name(normalization)], rotation=45.,
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
labels = [format_species_name(name) for name in species]
positions = np.arange(len(labels))
plt.bar(positions, predictions.mains[1:], width=0.6,  # ice composition
        edgecolor='black', color='gray')
rv.errorbar(positions + offsets[aice_label], predictions[1:], fmt='.',
            color='black')
plt.plot([], [], '.', color='black', label='AICE (this work)')
if show_references:  # reference predictions
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
          'abundance with respect to '+ format_species_name(normalization))
plt.ylabel(ylabel, labelpad=4)
plt.legend(fontsize=8)
title = 'AICE predictions  '
if normalization != 'total':
    title += ' '*26
plt.title(title, x=0.55, pad=16.)
fig.add_subplot(gs[0,1])  # temperature
plt.bar([1.], predictions[0].main, edgecolor='black', color='tab:gray')
rv.errorbar([1. + offsets[aice_label]], predictions[0], color='black')
if show_references:
    for (i,key) in enumerate(references):
        reference = results_df[key].values
        if rv.isfinite(reference[0]):
            rv.errorbar([1. + offsets[key]], reference[0], color=ref_colors[i])
plt.ylim(temp_range)
plt.locator_params(axis='y', nbins=5)
plt.xticks([1], ['$T_\mathrm{ice}$'], rotation=0., fontsize=10.)
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
    filename = config_file.replace('.yaml', '-results.csv')
    results_df.to_csv(filename)
    print(f'Saved results in {filename}.')