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
config_file = 'J110621.yaml'

# Libraries.
import os
import sys
import time
import platform
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

def gaussian(x, height=1, mean=0, std=1, ssf=1):
    """
    Apply a Gaussian function to the input array (x), with given height, mean,
    standard deviation (std) and supersampling factor (ssf).
    """
    if ssf == 1:
        y = height * np.exp(-(x-mean)**2 / (2*std**2))
    else:
        dx = x[1] - x[0]
        x_ = np.linspace(x[0]-dx/2, x[-1]+dx/2, int(ssf*len(x)))
        y_ = height * np.exp(-(x_-mean)**2 / (2*std**2))
        y = y_.reshape(-1, ssf).mean(axis=1)
    return y

def correct_band_fit(x, y, yf, thresh, excluded_regions=[]):
    """Correct left wing of the band in input spectra and fit (x, y, yf)."""
    mask = x > thresh
    for x1x2 in excluded_regions:
        x1, x2 = min(x1x2), max(x1x2)
        mask &= (x <= x1) | (x >= x2)
    ys = rv.rolling_function(np.median, y, size=15)
    s = len(x)*(1.*np.std(ys-y))**2
    interp = UnivariateSpline(x, y, s=s)
    yf[mask] = interp(x[mask])
    yf = rv.rolling_function(np.mean, yf, size=45)
    return yf

sep = '\\' if platform.system() == 'Windows' else '/'  # folder separator

#%% Initial options.

plt.close('all')
print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Band Integrator')
print()

# Default options.
default_options = {
    'output file': None,
    'figure size': [9., 5.],
    'data column': None,
    'show fractional abundance': False,
    'propagate observational uncertainties': False,
    'do gaussian fit': True,
    'normalize spectra': False,
    'normalization band': 'all',
    'normalization column': 'first',
    'wavenumber limits (/cm)': 'auto',
    'considered bands': 'all',
    'band labels': {}
    }

# Configuration file.
config_path = config_file if len(sys.argv) == 1 else sys.argv[1]
name = config_path.replace('.yaml', '').replace('.'+sep, sep).replace('.', '')
config_path = os.path.realpath(config_path)
config_folder = sep.join(config_path.split(sep)[:-1])
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
input_file = config['input file']
output_file = config['output file']
figsize = config['figure size']
data_column = config['data column']
show_fractions = config['show fractional abundance']
propagate_obs_uncs = config['propagate observational uncertainties']
do_gaussian_fit = config['do gaussian fit']
normalize_spectra = config['normalize spectra']
norm_band = config['normalization band']
norm_column = config['normalization column']
xlims = config['wavenumber limits (/cm)']
band_labels = config['band labels']
bands = config['all bands']
considered_bands = config['considered bands']
if considered_bands == 'all':
    considered_bands = list(bands.keys())
if len(considered_bands) < 2:
    show_fractions = False
# Some more options.
input_format = '.' + input_file.split('.')[-1]
output_format = input_format

#%% Reading and analyzing spectra.

# Reading of spectra.
if input_format == '.txt':
    data = np.loadtxt(input_file)
    x = data[:,0]
    inds = np.argsort(x)
    data = data[inds,:]
    x, y = data[:,[0,1]].transpose()
    dy = data[:,2] if data.shape[1] > 2 else np.zeros(len(x))
    spectra = pd.DataFrame({'x': x, 'y': y})
    columns = list(spectra.columns)
elif input_format == '.csv':
    spectra = pd.read_csv(input_file)
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
    
t1 = time.time()

# Fitting and integration of bands.
results = {}
ylims = []
for (k, column) in enumerate(columns[1:]):
    y = spectra[column].values
    result = {'bands': {}}
    for band in considered_bands:
        options = bands[band]
        extension = options['extension (/cm)']
        x1, x2 = min(extension), max(extension)
        xrange = x2 - x1
        width = xrange / 6
        band_strength = options['band strength (cm)']
        fix_band = True if 'fixed' in options and options['fixed'] else False
        do_gaussian_fit_ = (options['do gaussian fit'] if 'do gaussian fit'
                            in options else do_gaussian_fit)
        base = options['base level'] if 'base level' in options else 0.
        remove_baseline = True if 'baseline points (/cm)' in options else False
        if remove_baseline:
            x1x2 = options['baseline points (/cm)']
            x1, x2 = min(x1x2), max(x1x2)
            y1 = y[np.argmin(np.abs(x-x1))]
            y2 = y[np.argmin(np.abs(x-x2))]
            m = (y2 - y1) / (x2 - x1)
            base += y1 + m*(x-x1)
        else:
            base = base * np.ones(len(x))
        x1, x2 = min(extension), max(extension)
        mask = (x >= x1) & (x <= x2) & np.isfinite(y)
        x_ = x[mask]
        y_ = y[mask]
        dy_ = dy[mask]
        base_ = base[mask]
        if fix_band:
            center = options['center (/cm)']
            std = options['width (/cm)']
        else:
            center = x_[np.nanargmax(y_)]
        if do_gaussian_fit_:
            guess = [np.nanmax(y_)] if fix_band else [center, width, np.nanmax(y_)]
            gaussian_ = ((lambda x,h: gaussian(x, center, std, h)) if fix_band
                         else lambda x,x0,s,h: gaussian(x, x0, s, h))
            mask = (x_ >= x1) & (x_ <= x2)
            if 'excluded region (/cm)' in options:
                region = options['excluded region (/cm)']
                if type(region[0]) is not list:
                    region = [region]
                for x1x2 in region:
                    x1, x2 = min(x1x2), max(x1x2)
                    mask &= (x_ <= x1) | (x_ >= x2)
            else:
                region = []
            if propagate_obs_uncs:
                y_ra = rv.RichArray(y_, dy_)
                fit_results = rv.curve_fit(x_[mask], y_ra[mask]-base_[mask],
                                           gaussian_, guess, num_samples=600,
                                           consider_param_intervs=False)
                params = fit_results['parameters']
                params_samples = fit_results['parameters samples']
            else:
                params = curve_fit(gaussian_, x_[mask], y_[mask]-base_[mask],
                                   p0=guess)[0]
            if (band.startswith('H2O') and 'threshold to correct left part'
                    in options):
                correct_band_shape = True
                thresh = options['threshold to correct left part']
                if thresh is None:
                    correct_band_shape = False
            else:
                correct_band_shape = False
            if propagate_obs_uncs:
                if len(x_) > 400:
                    print('Integrating band...')
                areas = []
                if fix_band:
                    for h in params_samples:
                        y_f = gaussian_(x_, h)
                        if correct_band_shape:
                            y_f = correct_band_fit(x_, y_, y_f, thresh, region)
                        areas += [np.trapezoid(y_f, x_)]
                else:
                    for (x0,s,h) in params_samples:
                        y_f = gaussian_(x_, x0, s, h)
                        if correct_band_shape:
                            y_f = correct_band_fit(x_, y_, y_f, thresh, region)
                        areas += [np.trapezoid(y_f, x_)]
                area = rv.evaluate_distr(areas, domain=[0.,np.inf],
                                         consider_intervs=False)
            else:
                y_f = gaussian_(x_, *params)
                area = np.trapezoid(y_f, x_)
            params = rv.rich_array(params)
            if len(x_) < 25:
                x1, x2 = x_[0], x_[-1]
                x_new = np.append(x_, np.linspace(x1, x2, 2*len(x_)))
                x_new = np.unique(x_new)
                base_ = np.interp(x_new, x_, base_)
                x_ = x_new
            y_f = gaussian_(x_, *params.centers) + base_
            if correct_band_shape:
                y_f = correct_band_fit(x_, y_, y_f, thresh, region)
        else:
            if propagate_obs_uncs:
                y_ra = rv.RichArray(y_, dy_)
                areas = rv.array_distribution(lambda x,y: np.trapezoid(y-base_,x),
                                              [x_, y_ra], num_samples=1200,
                                              consider_intervs=False)
                area = rv.evaluate_distr(areas, domain=[0.,np.inf],
                                         consider_intervs=False)
            else:
                area = np.trapezoid(y_-base_, x_)
        with np.errstate(invalid='ignore'):
            coldens = np.log(10) * area / band_strength
        result['bands'][band] = {}
        if do_gaussian_fit_:
            result['bands'][band]['fit parameters'] = params
            result['bands'][band]['fit'] = {'x': x_, 'y': y_f, 'base': base_}
        else:
            result['bands'][band]['band'] = {'x': x_, 'y': y_, 'base': base_}
        result['bands'][band]['column density (/cm2)'] = coldens
        results[column] = result
        y_max = max(y.max(), y_f.max()) if do_gaussian_fit_ else y.max()
        ylims += [y.min(), y_max]
        
t2 = time.time()
    
#%% Displaying results.

print()
if output_format == '.csv' and norm_column == 'each' or output_format == '.txt':
    shown_results = results
else:
    shown_results = {norm_column: results[norm_column]}
# Column densities and fractional abundances.
for column in shown_results:
    result = results[column]
    title = input_file.split('/')[-1]
    if input_format == '.csv':
        title += f' - {column}'.replace('abs. ', '').replace('.0 ', ' ')
    print(title)
    print('-'*len(title))
    print('Column densities:')
    for band in result['bands']:
        colden = result['bands'][band]['column density (/cm2)']
        print(f'- {band}: {colden} /cm2')
    if show_fractions:
        print('Fractional abundances:')
        coldens = np.array([result['bands'][band]['column density (/cm2)']
                            for band in result['bands']])
        with np.errstate(invalid='ignore'):
            fractions = coldens / np.sum(coldens)
        for (band, fraction) in zip(result['bands'], fractions):
            print(f'- {band}: {100*fraction} %')
    print()

for (k, column) in enumerate(columns[1:]):
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
        norm /= np.log(10)
        results[column]['norm'] = norm
       
elapsed_time = t2 - t1
mins = elapsed_time // 60
secs = elapsed_time % 60
print(f'Elapsed time for calculations: {mins} mins + {secs:.4f} s.')
    
#%% Plotting and optional normalization.

# Graphic options.
if xlims == 'auto':
    xlims = [x.max(), x.min()]
# Plots.
for (k,column) in enumerate(columns[1:]):
    title = input_file.split('/')[-1]
    if input_format == '.csv':
        title += f' - {column}'.replace('abs. ', '').replace('.0 ', ' ')
    y = spectra[column].values
    result = results[column]
    plt.figure(k+1, figsize=figsize)
    plt.clf()    
    plt.errorbar(x, y, dy, ms=1., color='black', ecolor='gray',
                 drawstyle='steps-mid', label='observations')
    plt.axhline(y=0, color='k', ls='--', lw=0.6)
    if do_gaussian_fit:
        plt.plot([], color='palevioletred', label='AICE fit')
    plt.fill_between([], [], color='palevioletred', alpha=0.2, label='integrated area')
    std = np.nanstd(y)
    are_excluded_regions = False
    for band in result['bands']:
        band_options = bands[band]
        band_results = result['bands'][band]
        if 'fit' in result['bands'][band]:
            fit_data = band_results['fit']
            x_ = fit_data['x']
            y_ = fit_data['y']
            base_ = fit_data['base']
            plt.plot(x_, y_, color='palevioletred', zorder=3., alpha=0.7)
            fixed_band = (True if 'fixed' in band_options and band_options['fixed']
                          else False)
            center = (band_options['center (/cm)'] if fixed_band
                      else band_results['fit parameters'][0].main)
            if 'excluded region (/cm)' in band_options:
                are_excluded_regions = True
                region = band_options['excluded region (/cm)']
                if type(region[0]) is not list:
                    region = [region]
                for x1x2 in region:
                    x1, x2 = min(x1x2), max(x1x2)
                    for (fc, ec) in zip(['none', 'gray'], ['lightgray', 'none']):
                        plt.axvspan(x1, x2, facecolor=fc, edgecolor=ec,
                                    hatch='/', alpha=0.3)
        else:
            x_ = band_results['band']['x']
            y_ = band_results['band']['y']
            base_ = band_results['band']['base']
            extension = band_options['extension (/cm)']
            center = np.mean(extension)
        plt.fill_between(x_, base_, y_, color='palevioletred', alpha=0.4)
        y_text = np.max(y_) + 0.5*std
        text = band_labels[band] if band in band_labels else band
        plt.text(center+34., y_text, text,
                 ha='center', va='center', fontsize=6.)
    if are_excluded_regions:
        plt.axvspan(10., 10., facecolor='lightgray', edgecolor='gray',
                    hatch='/', alpha=0.3, label='excluded regions')
    plt.xlim(xlims)
    plt.axhline(y=0, color='black', lw=0.5)
    plt.xlabel('wavenumber (cm$^{-1}$)', labelpad=6.)
    plt.ylabel('absorbance')
    plt.title(title, fontweight='bold', pad=12.)
    ax = plt.gca()
    ax2 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
    wavelength_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # μm
    wavelength_ticklabels = [1, 2, 3, 4, 5, '', 7, '', '', 10,
                             '', 12, '', '', '', '', '', '', '', '']
    ax2.set_xticks(wavelength_ticks, wavelength_ticklabels)
    ax2.set_xlabel('wavelength (μm)', labelpad=6., fontsize=9.)
    plt.legend()
    plt.tight_layout()
    if normalize_spectra:
        ax3 = plt.twinx(ax)
        norm = result['norm']
        ylims = plt.ylims()
        ax3.plot(x, y/norm, alpha=0.)
        ax3.set_ylim(ylims/norm)
        ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True)) 
        ax3.set_ylabel('absorption coefficient (cm$^2$)')

# Normalization and saving of normalized spectra.
if normalize_spectra and output_file is not None:
    if output_format == '.txt':
        data = [x, y/norm]
        data = np.array(data).T
        header = 'wavenum.(/cm) abs.coeff.(cm2)'
        np.savetxt(output_file, data, fmt='%.1f %.3e',
                   header=header, delimiter='\t')
        print(f'Saved file in {output_file}')
    elif output_format == '.csv':
        if data_column is not None:
            spectra = spectra[['wavenumber (/cm)'] + [data_column]]
        for column in columns[1:]:
            spectra[column] = spectra[column] / results[column]['norm']
        new_columns = list(spectra.columns)
        for (i,column) in enumerate(new_columns[1:]):
            new_columns[i+1] = column.replace('abs.', 'abs. coeff. (cm2)')
        spectra.columns = new_columns
        spectra['wavenumber (/cm)'] = spectra['wavenumber (/cm)'].map(
                                                lambda x: '{:.1f}'.format(x))
        spectra.to_csv(output_file, index=False, float_format='%.3e')
        print(f'Saved file in {output_file}.')
