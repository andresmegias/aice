#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.0
------------------------------------------
Band Integrator module

Andrés Megías
"""

# Configuration file.
config_file = 'NIR38.yaml'

# Libraries.
import os
import re
import sys
import copy
import yaml
import numpy as np
import pandas as pd
import richvalues as rv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from matplotlib.ticker import ScalarFormatter

# Functions.

def axis_conversion(x):
    """Axis conversion from wavenumber to wavelength and viceversa"""
    with np.errstate(divide='ignore'):
        y = 1e4 / x
    return y

def gaussian(x, x0, s, h):
    """Gaussian with given center (x0), width (s) and height (h)."""
    y = h * np.exp(-0.5*((x-x0)/s)**2)
    return y

def format_species_name(input_name, simplify_numbers=True, acronyms={}):
    """
    Format text as a molecule name, with subscripts and upperscripts.

    Parameters
    ----------
    input_name : str
        Input text.
    simplify_numbers : bool, optional
        Remove the brackets between single numbers.
    acronyms : dict
        Dictionary of acronyms for species name. If the input text is one of
        the dictionary keys, it will be replaced by the corresponding value,
        and then the formatting function will be still applied.

    Returns
    -------
    output_name : str
        Formatted molecule name.
    """
    original_name = copy.copy(input_name)
    # acronyms
    for name in acronyms:
        if original_name == name:
            original_name = acronyms[name]
    # removing the additional information of the transition
    original_name = original_name.replace('_k',',k').split(',')[0]
    # prefixes
    possible_prefixes = ['#', '@', '$']
    if '-' in original_name and original_name.split('-')[0].isalpha():
        prefix = original_name.split('-')[0]
    else:
        prefix = ''
        for text in possible_prefixes:
            if original_name.startswith(text):
                prefix = text
                break
    original_name = original_name.replace(prefix, '')
    # upperscripts
    possible_upperscript, in_upperscript = False, False
    output_name = ''
    upperscript = ''
    inds = []
    for (i, char) in enumerate(original_name):
        if (char.isupper() and not possible_upperscript
                and '-' in original_name[i:]):
            inds += [i]
            possible_upperscript = True
        elif char.isupper():
            inds += [i]
        if char == '-' and not in_upperscript:
            inds += [i]
            in_upperscript = True
        if in_upperscript and not (char.isdigit() or char == '-'):
            in_upperscript = False
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
    if output_name == '':
        output_name = original_name
    output_name = output_name.replace('[', '^{').replace(']', '}')
    if output_name.endswith('+') or output_name.endswith('-'):
        symbol = output_name[-1]
        output_name = output_name.replace(symbol, '$^{'+symbol+'}$')
    original_name = copy.copy(output_name)
    # subscripts
    original_name = original_name.replace('_','')
    output_name, subscript, prev_char = '', '', ''
    in_bracket = False
    for (i, char) in enumerate(original_name):
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
    # some formatting
    output_name = output_name.replace('$$', '').replace('__', '')
    # remove brackets from single numbers
    if simplify_numbers:
        single_numbers = re.findall('{(.?)}', output_name)
        for number in set(single_numbers):
            output_name = output_name.replace('{'+number+'}', number)
    # prefix
    if prefix == '$':
        prefix = '\$'
    if prefix in ('\$', '#', '@'):
        prefix += '$\,$'
    output_name = prefix + output_name
    output_name = output_name.replace('$^$', '')
    return output_name

def correct_water_band(x, y, yf):
    """Correct left wing of water band in input spectra and fit (x, y, yf)."""
    mask = x > 3300.
    ys = rv.rolling_function(np.median, y, 15)
    s = len(x)*(1.*np.std(ys-y))**2
    interp = UnivariateSpline(x, y, s=s)
    yf[mask] = interp(x[mask])
    yf[yf < 0.] = 0.
    yf[x > 3700.] = 0.
    yf = rv.rolling_function(np.mean, yf, 45)
    return yf

#%% Initial options.

plt.close('all')
print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Band Integrator module')
print()

# Default options.
default_options = {
    'input file': '',
    'output folder': 'abs-coeffs/',
    'figure size': [9., 5.],
    'data column': None,
    'show fractional abundance': False,
    'propagate observational uncertainties': False,
    'integrate fit': True,
    'normalize spectra': False,
    'integration sigmas': 3.,
    'normalization band': 'all',
    'normalization column': 'first',
    'wavenumber range (/cm)': (4001., 980., 1.)
    }

# Configuration file.
config_path = './' + config_file if len(sys.argv) == 1 else sys.argv[1]
name = config_path.replace('.yaml', '').replace('./', '/').replace('.', '')
config_path = os.path.realpath(config_path)
config_folder = '/'.join(config_path.split('/')[:-1]) + '/'
os.chdir(config_folder)
if os.path.isfile(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
else:
    raise FileNotFoundError('Configuration file not found.')
with open(config_file) as file:
    config = yaml.safe_load(file) 
config = {**default_options, **config}

# Loading of options.
file = config['input file']
output_folder = config['output folder']
figsize = config['figure size']
data_column = config['data column']
show_fractions = config['show fractional abundance']
propagate_obs_uncs = config['propagate observational uncertainties']
integrate_fit = config['integrate fit']
normalize_spectra = config['normalize spectra']
norm_band = config['normalization band']
norm_column = config['normalization column']
sigmas = config['integration sigmas']
bands = config['bands']
if len(list(bands.keys())) < 2:
    show_fractions = False
x1, x2, dx = config['wavenumber range (/cm)']
x1x2 = (x1, x2)
x1, x2 = min(x1x2), max(x1x2)
x = wavenumber = np.arange(x1, x2, int(dx))
# Some more options.
input_format = '.' + file.split('.')[-1]
output_format = input_format

#%% Reading and analyzing spectra.

# Reading of spectra.
if input_format == '.txt':
    data = np.loadtxt(file)
    x_ = data[:,0]
    inds = np.argsort(x_)
    data = data[inds,:]
    x_, y_ = data[:,[0,1]].transpose()
    dy_ = data[:,2] if data.shape[1] > 2 else np.zeros(len(x))
    y_ = np.nan_to_num(y_, nan=0.)
    y = np.interp(x, x_, y_, left=0., right=0.)
    dy = np.interp(x, x_, dy_, left=0., right=0.)
    spectra = pd.DataFrame({'x': x, 'y': y})
    columns = list(spectra.columns)
elif input_format == '.csv':
    spectra = pd.read_csv(file)
    data = spectra.values
    x = data[:,0]
    inds = np.argsort(x)
    x = x[inds]
    spectra = spectra.reindex(inds)
    columns = list(spectra.columns)
    if data_column is not None:
        columns = [columns[0]] + [data_column]
    for column in spectra.columns[1:]:
        spectra[column] = spectra[column].fillna(0.)
    if norm_column == 'first':
        norm_column = columns[1]
    elif norm_column == 'last':
        norm_column = columns[-1]
    dy = np.zeros(len(x))


# Fitting and integration of bands.
results = {}
ylims = []
for (k,column) in enumerate(columns[1:]):
    y = spectra[column].values
    result = {'bands': {}}
    for band in bands:
        options = bands[band]
        center = options['center (/cm)']
        width = options['width (/cm)']
        band_strength = options['band strength (cm)']
        fix_band = True if 'fixed' in options and options['fixed'] else False
        xrange = [center - width/2, center + width/2]
        ampl = sigmas * width/2
        if fix_band:
            ampl *= 2
            std = width/2
        mask = (x >= xrange[0] - ampl) & (x <= xrange[1] + ampl)
        x_ = x[mask]
        y_ = y[mask]
        dy_ = dy[mask]
        if integrate_fit:
            guess = [np.nanmax(y_)] if fix_band else [center, width/4, np.nanmax(y_)]
            gaussian_function = ((lambda x,h: gaussian(x, center, std, h))
                                 if fix_band else gaussian)
            mask = (x_ >= xrange[0]) & (x_ <= xrange[1])
            if 'excluded width (/cm)' in options:
                width = options['excluded width (/cm)']
                semiwidth = width / 2
                xrange = [center - semiwidth, center + semiwidth]
                mask &= (x_ <= xrange[0]) | (x_ >= xrange[1])
            mask &= (y_ >= 0.3*np.max(y_[mask]))
            try:
                if not propagate_obs_uncs:
                    params = curve_fit(gaussian_function, x_[mask], y_[mask], p0=guess)[0]
                else:
                    fit_results = rv.curve_fit(x_[mask], rv.RichArray(y_[mask], dy_[mask]),
                                    gaussian_function, guess, num_samples=400,
                                    consider_param_intervs=False)
                    params = fit_results['parameters']
                    params_samples = fit_results['parameters samples']
            except:
                params = guess
            params = rv.rich_array(params)
            y_f = gaussian_function(x_, *params.centers)
            if band.startswith('H2O'):
                y_f = correct_water_band(x_, y_, y_f)
        if not propagate_obs_uncs:
            area = (rv.rval(np.trapz(y_f, x_)) if integrate_fit
                    else rv.rval(np.trapz(y_, x_)))
        else:
            areas = []
            if integrate_fit:
                if fix_band:
                    for h in params_samples:
                        y_f = gaussian_function(x_, h)
                        if band.startswith('H2O'):
                            y_f = correct_water_band(x_, y_, y_f)
                        areas += [np.trapz(y_f, x_)]
                else:
                    for (x0,s,h) in params_samples:
                        y_f = gaussian_function(x_, x0, s, h)
                        if band.startswith('H2O'):
                            y_f = correct_water_band(x_, y_, y_f)
                        areas += [np.trapz(y_f, x_)]
            else:
                areas = rv.array_distribution(lambda x,y: np.trapz(y,x), [x_, y_],
                                           num_samples=640, consider_intervs=False)
            area = rv.evaluate_distr(areas, domain=[0.,np.inf], consider_intervs=False)
        with np.errstate(invalid='ignore'):
            coldens = np.log(10) * area / band_strength
        result['bands'][band] = {}
        if integrate_fit:
            result['bands'][band]['fit parameters'] = params
            result['bands'][band]['fit'] = {'x': x_, 'y': y_f}
        else:
            result['bands'][band]['band'] = {'x': x_, 'y': y_}
        result['bands'][band]['column density (/cm2)'] = coldens
        results[column] = result
        y_max = max(y.max(), y_f.max()) if integrate_fit else y.max()
        ylims += [y.min(), y_max]
    
#%% Displaying results.

print()
if output_format == '.csv' and norm_column == 'each' or output_format == '.txt':
    shown_results = results
else:
    shown_results = {norm_column: results[norm_column]}
# Column densities and fractional abundances.
for column in shown_results:
    result = results[column]
    title = file.split('/')[-1]
    if input_format == '.csv':
        title += f' - {column}'.replace('abs. ', '').replace('.0 ', ' ')
    print(title)
    print('-'*len(title))
    print('Column densities:')
    for band in result['bands']:
        colden = result['bands'][band]['column density (/cm2)']
        print('- {}: {} /cm2'.format(band, colden))
    if show_fractions:
        print('Fractional abundances:')
        coldens = np.array([result['bands'][band]['column density (/cm2)']
                            for band in result['bands']])
        with np.errstate(invalid='ignore'):
            fractions = coldens / np.sum(coldens)
        for (band, fraction) in zip(result['bands'], fractions):
            print('- {}: {} %'.format(band, 100*fraction))
    print()

for (k,column) in enumerate(columns[1:]):
    y = spectra[column].values
    if normalize_spectra:
        if output_format == '.csv':
            result = (results[columns[1+k]] if norm_column == 'each'
                      else results[norm_column])
        if norm_band in ('all', 'average'):
            coldens = np.array([result['bands'][band]['column density (/cm2)']
                                for band in result['bands']])
            if norm_band == 'average':
                norm = np.mean(coldens)
            elif norm_band == 'all':
                norm = np.sum(coldens)
        else:
            norm = result['bands'][norm_band]['column density (/cm2)']
        results[column]['norm'] = norm
    
#%% Plotting and optional normalization.

# Graphic options.
ylims = [min(ylims), max(ylims)]
margin = 0.02*np.diff(ylims)
ylims = np.array([min(ylims)-margin, max(ylims)+6*margin])
# Plots.
for (k,column) in enumerate(columns[1:]):
    title = file.split('/')[-1]
    if input_format == '.csv':
        title += f' - {column}'.replace('abs. ', '').replace('.0 ', ' ')
    y = spectra[column].values
    result = results[column]
    plt.figure(k+1, figsize=figsize)
    plt.clf()    
    plt.errorbar(x, y, dy, ms=1., color='black', ecolor='gray', label='observations')
    plt.axhline(y=0, color='k', ls='--', lw=0.6)
    if integrate_fit:
        plt.plot([], color='palevioletred', label='AICE fit')
    plt.fill_between([], [], color='palevioletred', alpha=0.2, label='integrated area')
    std = np.nanstd(y)
    for band in result['bands']:
        if integrate_fit:
            fit_data = result['bands'][band]['fit']
            plt.plot(fit_data['x'], fit_data['y'], color='palevioletred', zorder=3.)
            x_ = fit_data['x']
            y_ = fit_data['y']
        else:
            x_ = result['bands'][band]['band']['x']
            y_ = result['bands'][band]['band']['y']
        plt.fill_between(x_, y_, color='palevioletred', alpha=0.2)
        y_text = np.max(y_) + 0.5*std
        text = format_species_name(band)
        plt.text(bands[band]['center (/cm)']+34., y_text, text,
                 ha='center', va='center', fontsize=5.)
    plt.xlim(wavenumber[-1], wavenumber[0])
    plt.ylim(ylims)
    plt.axhline(y=0, color='black', lw=0.5)
    plt.xlabel('wavenumber (cm$^{-1}$)', labelpad=6)
    plt.ylabel('absorbance')
    plt.title(title, fontweight='bold', pad=12)
    ax = plt.gca()
    ax2 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
    wavelength_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # μm
    wavelength_ticklabels = [1, 2, 3, 4, 5, '', 7, '', '', '', '', '', '', '', 15]
    ax2.set_xticks(wavelength_ticks, wavelength_ticklabels)
    ax2.set_xlabel('wavelength (μm)', labelpad=6., fontsize=9.)
    plt.legend()
    if normalize_spectra:
        ax3 = plt.twinx(ax)
        norm = result['norm'].main
        ax3.plot(x, y/norm, alpha=0.)
        ax3.set_ylim(ylims/norm)
        ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True)) 
        ax3.set_ylabel('absorption coefficient (cm$^2$)')
    plt.tight_layout()

# Normalization and saving of normalized spectra.
if normalize_spectra:
    if output_format == '.txt':
        data = [x, y/norm]
        data = np.array(data).T
        header = 'wavenum.(/cm) abs.coeff.(cm2)'
        filename = (file.split('/')[-1].replace('.txt', '-n.txt')
                    .replace('-n-n', '-n'))
        if '-r-' in filename:
            filename = filename.replace('-r-', '-').replace('-n.txt', '-r-n.txt')
        np.savetxt(output_folder+filename, data, fmt='%.1f %.3e',
                   header=header, delimiter='\t')
        print(f'Saved file in {output_folder}{filename}')
    elif output_format == '.csv':
        filename = (file.split('/')[-1].replace('.csv', '-n.csv')
                    .replace('-n-n', '-n'))
        if data_column is not None:
            spectra = spectra[['wavenumber (/cm)'] + [data_column]]
        for column in columns[1:]:
            spectra[column] = spectra[column] / results[column]['norm'].main
        new_columns = list(spectra.columns)
        for (i,column) in enumerate(new_columns[1:]):
            new_columns[i+1] = column.replace('abs.', 'abs. coeff. (cm2)')
        spectra.columns = new_columns
        spectra['wavenumber (/cm)'] = spectra['wavenumber (/cm)'].map(
                                                lambda x: '{:.1f}'.format(x))
        spectra.to_csv(output_folder+filename, index=False, float_format='%.3e')
        print(f'Saved file in {output_folder}{filename}.')
