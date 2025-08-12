#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.0
------------------------------------------
Script for averaging the errors of the neural networks

Andrés Megías
"""

import os
import pathlib
import platform
import numpy as np
import pandas as pd
import richvalues as rv
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 90)

#%%

print('AICE errors')
print('-----------')
print()
plt.close('all')

# Options.
model_name = 'aice'
species = ['H2O', 'CO', 'CO2', 'CH3OH', 'NH3', 'CH4']
formatted_names = {'H2O': 'H$_2$O', 'CO': 'CO', 'CO2': 'CO$_2$',
                   'CH3OH': 'CH$_3$OH', 'NH3': 'NH$_3$', 'CH4': 'CH$_4$'}
variables = ['temp'] + species + ['species']
colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:purple']
labels = (['$T_\\mathrm{ice}$'] + [formatted_names[name] for name in species]
          + ['avg.'])

# Validation errors.
errors_filename = os.path.join('models', model_name, 'errors',
                               f'{model_name}-val-errors.csv')
val_errors_df = pd.read_csv(errors_filename, index_col=[0])
seeds = list(val_errors_df.index)
for name in (species + ['species']):
    val_errors_df[name] = 100*val_errors_df[name]
for var in variables:
    val_errors_df.at['avg',var] = val_errors_df[var].mean()
val_errors_rdf = val_errors_df.copy().astype(object)
for (i,row) in val_errors_rdf.iterrows():
    for var in variables:
        val_errors_rdf.at[i,var] = f'{row[var]:.1f}'
val_errors_rdf = rv.rich_df(val_errors_rdf, ignore_columns=variables)
val_errors_rdf_latex = val_errors_rdf[species+['temp']].latex()
print('Validation errors:\n')
print(val_errors_rdf)
print()

# Plot of validation errors.
fig = plt.figure(1)
gs = plt.GridSpec(1, 2, width_ratios=[1,len(species)+1], wspace=0.28,
                  left=0.14, right=0.95, bottom=0.15, top=0.90)
fig.add_subplot(gs[0,0])
plt.scatter([0], val_errors_df.at['avg','temp'], color='gray',
            s=100., alpha=0.5)
for i in seeds:
    o = (0.1*i - 0.5)*0.9
    plt.scatter([0+o], val_errors_df.at[i,'temp'], color='gray',
                s=3., alpha=0.5)
plt.xticks([0], [labels[0]])
plt.tick_params(axis='x', which='major', pad=8.)
plt.ylim(0., 20.)
plt.margins(x=0.5)
for tick in plt.gca().xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
plt.ylabel('temperature (K)')
fig.add_subplot(gs[0,1])
positions = np.append(1+np.arange(len(species)), len(species)+1.2)
plt.scatter(positions[:-1], val_errors_df.loc['avg'][species], c=colors,
            s=100., alpha=0.5)
plt.scatter(positions[-1], val_errors_df.loc['avg']['species'], c='chocolate',
            s=100., alpha=0.5)
for i in seeds:
    o = (0.1*i - 0.5)*0.9
    plt.scatter(positions[:-1]+o, val_errors_df.loc[i][species], c=colors,
                s=3., alpha=0.5)
    plt.scatter(positions[-1]+o, val_errors_df.loc[i]['species'], c='chocolate',
                s=3., alpha=0.5)
plt.ylim(0., 6.)
plt.margins(x=0.04)
plt.axvline(positions[-1]-0.6, color='black', lw=1.)
plt.xticks(positions, labels[1:], rotation=45.)
for tick in plt.gca().xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
plt.ylabel('molecular fraction (%)')
plt.suptitle(f'Mean validation errors - {model_name}', fontweight='bold')
ax2 = plt.twinx(plt.gca())
plt.ylim(0., 6.)

# Test errors.
sep = '\\' if platform.system() == 'Windows' else '/'  # folder separator
test_folder = os.path.join('models', model_name, 'errors', 'test')
consider_test_errors = os.path.exists(test_folder)
if consider_test_errors:
    test_errors = {}
    for var in variables:
        test_errors[var] = []
    test_files = os.path.join(test_folder, '*.csv')
    test_files = list([str(pp) for pp in pathlib.Path('.').glob(test_files)])
    trials = [folder.split(sep)[-1].replace('errors', '').replace('.csv', '')
              for folder in test_files]
    trials = sorted(trials)
    test_files = sorted(test_files)
    for (trial, file) in zip(trials, test_files):
        test_errors_df = pd.read_csv(file, index_col=[0])
        for name in (species + ['species']):
            test_errors_df[name] = 100*test_errors_df[name]
        for var in variables:
            test_errors[var] += [test_errors_df[var].mean()]
    test_errors_df = pd.DataFrame(test_errors, index=trials, dtype=object)
    for var in variables:
        test_errors_df.at['avg',var] = test_errors_df[var].mean()
    test_errors_rdf = test_errors_df.copy()
    for (i,row) in test_errors_rdf.iterrows():
        for var in variables:
            test_errors_rdf.at[i,var] = f'{row[var]:.1f}'
    test_errors_rdf = rv.rich_df(test_errors_rdf, ignore_columns=variables)
    test_errors_rdf_latex = test_errors_rdf.latex()
    print('Test errors:\n')
    print(test_errors_rdf)
    
# Save figure.
plt.show()
filename = os.path.join('models', model_name, 'plots', 'errors.png')
plt.savefig(filename)
print(f'\nSaved plot in {filename}.')
