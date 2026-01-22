#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.0
------------------------------------------
Pre-processing module 3 - Silicate fit

Andrés Megías
"""

# Configuration file.
config_file = 'NIR38.yaml'

# Libraries
import os
import sys
import yaml
import time
import optool
import platform
import numpy as np
import pandas as pd
import richvalues as rv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Functions.

def xaxis_conversion(x):
    """Convert wavelength to wavenumber and viceversa."""
    with np.errstate(divide='ignore'):
        y = 1e4 / x
    return y

sep = '\\' if platform.system() == 'Windows' else '/'  # folder separator

#%% Initial options.

print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Pre-processing module 3 - Silicate fit')
time.sleep(0.3)
print()

# Default options.
default_options = {
    'figure size': (9., 7.),
    'comment character in input file': '#',
    'column indices': {'x': 1, 'y': 2, 'y unc.': 3},
    'input spectral variable': 'wavenumber',
    'intensity variable': 'absorbance',
    'silicate file': None,
    'silicate spectral variable': 'wavelength',
    'silicate intensity variable': 'optical depth',
    'fit offset': True
    }

# Configuration file.
config_path = config_file if len(sys.argv) == 1 else sys.argv[1]
name = config_path.replace('.yaml', '')
config_path = os.path.realpath(config_path)
config_folder = sep.join(config_path.split(sep)[:-1]) + sep
os.chdir(config_folder)
if os.path.isfile(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
else:
    raise FileNotFoundError('Configuration file not found.')
config['silicate removal'] = {**default_options, **config['silicate removal']}

# Options.
folder = config['parent folder'] if 'parent folder' in config else ''
options = config['silicate removal']
figsize = options['figure size']
input_file = options['input file']
comment_char = options['comment character in input file']
column_inds = options['column indices']
input_spectral_variable = options['input spectral variable']
intensity_variable = options['intensity variable']
silicate_file = options['silicate file']
silicate_spectral_variable = options['silicate spectral variable']
silicate_intensity_variable = options['silicate intensity variable']
composition = options['composition']
grain_size = options['grain size (μm)']
fit_offset = options['fit offset'] if 'fit offset' in options else True
optool_path = options['OpTool path']
output_file = options['output file']
if not optool_path.endswith(' '):
    optool_path += ' '
idx_x, idx_y = column_inds['x'] - 1, column_inds['y'] - 1
idx_dy = column_inds['y unc.'] - 1 if 'y unc.' in column_inds else None
composition_labels = {}
for (i, comp_i) in enumerate(composition):
    comp = list(comp_i.keys())[0] if type(comp_i) is dict else comp_i
    if '(' in comp:
        label = comp.split('(')[1].split(')')[0]
        comp_ = comp.split('(')[0].replace(' ', '')
        if type(comp_i) is dict:
            composition[i] = {comp_: comp_i[comp]}
        else:
            composition[i] = comp_
        composition_labels[comp_] = label
    else:
        composition_labels[comp] = comp
fit_silicate = True if silicate_file is None else False
if 'fit regions (μm)' in options:
    regions = options['fit regions (μm)']
    use_wavelength = True
elif 'fit regions (/cm)' in options:
    regions = options['fit regions (/cm)']
    use_wavelength = False
    
#%% Reading and fit.

# Reading of file.
if '.csv' in input_file:
    df = rv.rich_df(pd.read_csv(os.path.join(folder, input_file)))
    data = df.values
    x = data[:,idx_x].mains
    y = data[:,idx_y].mains
    dy = data[:,idx_y].uncs
else:
    data = np.loadtxt(os.path.join(folder, input_file), comments=comment_char)
    x = data[:,idx_x]
    y = data[:,idx_y]
    dy = data[:,idx_dy] if idx_dy is not None else np.zeros(len(y))
if input_spectral_variable == 'wavelength':
    x = 1e4 / x
if intensity_variable == 'optical depth':
    y /= np.log(10)
    dy /= np.log(10)
inds = np.argsort(x)
x = x[inds][::-1]
y = y[inds][::-1]
dy = dy[inds][::-1]
wavenumber = x
absorbance = y
absorbance_uncs = dy
absorbance_rv = rv.RichArray(absorbance, absorbance_uncs)
wavelength = 1e4 / wavenumber
if 'xrange (μm)' in options:
    xrange = options['xrange (μm)']
elif 'xrange (/cm)' in options:
    x1x2 = options['xrange (/cm)']
    x1, x2 = min(x1x2), max(x1x2)
    xrange = [1e4 / x2, 1e4 / x1]
else:
    xrange = [1e4 / wavenumber[1], 1e4 / wavenumber[0]]
mask = (wavelength >= xrange[0]) & (wavelength <= xrange[1])
wavelength_silreg = wavelength[mask]
absorbance_silreg = absorbance[mask]

# Silicate fit.
if fit_silicate:
    
    # Individual simulations.
    sil_spectra = []
    comps, fracs = [], []
    for comp_entry in composition:
        if type(comp_entry) is str:
            comp = comp_entry
            comps += [composition_labels[comp]]
            fit_fracs = True
            frac = None
        elif type(comp_entry) is dict:
            comp = list(comp_entry.keys())[0]
            comps += [composition_labels[comp]]
            fit_fracs = False
            frac = float(list(comp_entry.values())[0])
        silicate = optool.particle(optool_path + '{} -a {}'
                                   .format(comp, grain_size))
        sil_wavelength = silicate.lam
        sil_abscoeff = silicate.kabs.flatten() + silicate.ksca.flatten()
        mask = (sil_wavelength >= xrange[0]) & (sil_wavelength <= xrange[1])
        sil_wavelength = sil_wavelength[mask]
        sil_abscoeff = sil_abscoeff[mask]
        sil_absorbance = -sil_abscoeff
        sil_absorbance /= np.max(sil_absorbance)
        interpolation = interp1d(sil_wavelength, sil_absorbance, kind='quadratic',
                                 bounds_error=False, fill_value='extrapolate')
        sil_absorbance = interpolation(wavelength_silreg)
        sil_spectra += [sil_absorbance]
        fracs += [frac]
        
    # Preparation of the model to fit.
    sil_wavelength = wavelength_silreg.copy()
    sf_bounds = (0., 40.)
    off_bounds = [-2., 2.]
    frac_bounds = [0., 1.]
    if not fit_offset:
        off = options['fixed offset'] if 'fixed offset' in options else 0.
    if len(composition) == 1:
        if fit_offset:
            def model(x, sf, off):
                spectrum = np.interp(x, sil_wavelength, sil_spectra[0])
                absorbance = sf * spectrum + off
                return absorbance
            bounds = ([sf_bounds[0], off_bounds[0]], [sf_bounds[1], off_bounds[1]])
        else:
            def model(x, sf):
                spectrum = np.interp(x, sil_wavelength, sil_spectra[0])
                absorbance = sf * spectrum + off
                return absorbance
            bounds = ([sf_bounds[0]], [sf_bounds[1]])
        
    elif len(composition) == 2:
        if fit_fracs:
            if fit_offset:
                def model(x, sf, off, c):
                    c1, c2 = c, 1-c
                    spectrum1 = np.interp(x, sil_wavelength, sil_spectra[0])
                    spectrum2 = np.interp(x, sil_wavelength, sil_spectra[1])
                    absorbance = sf * (c1 * spectrum1 + c2 * spectrum2) + off
                    return absorbance
                bounds = ([sf_bounds[0], off_bounds[0], frac_bounds[0]],
                          [sf_bounds[1], off_bounds[1], frac_bounds[1]])
            else:
                def model(x, sf, c):
                    c1, c2 = c, 1-c
                    spectrum1 = np.interp(x, sil_wavelength, sil_spectra[0])
                    spectrum2 = np.interp(x, sil_wavelength, sil_spectra[1])
                    absorbance = sf * (c1 * spectrum1 + c2 * spectrum2) + off
                    return absorbance
                bounds = [sf_bounds[0], frac_bounds[0]], [sf_bounds[1], frac_bounds[1]]
        else:
            fracs = np.array(fracs)
            fracs /= np.sum(fracs)
            if fit_offset:
                def model(x, sf, off):
                    c1, c2 = fracs
                    spectrum1 = np.interp(x, sil_wavelength, sil_spectra[0])
                    spectrum2 = np.interp(x, sil_wavelength, sil_spectra[1])
                    absorbance = sf * (c1 * spectrum1 + c2 * spectrum2) + off
                    return absorbance
                bounds = ([sf_bounds[0], off_bounds[0]], [sf_bounds[1], off_bounds[1]])
            else:
                def model(x, sf):
                    c1, c2 = fracs
                    spectrum1 = np.interp(x, sil_wavelength, sil_spectra[0])
                    spectrum2 = np.interp(x, sil_wavelength, sil_spectra[1])
                    absorbance = sf * (c1 * spectrum1 + c2 * spectrum2) + off
                    return absorbance
                bounds = ([sf_bounds[0]], [sf_bounds[1]])
    
    # Silicate fit.
    mask = np.zeros(len(wavelength), bool)
    for x1x2 in regions:
        x1, x2 = min(x1x2), max(x1x2)
        if use_wavelength:
            mask += (wavelength >= x1) & (wavelength <= x2)
        else:
            mask += (wavenumber >= x1) & (wavenumber <= x2)
    wavelength_mask = wavelength[mask]
    params = curve_fit(model, wavelength[mask], absorbance[mask], bounds=bounds)[0]
    fitted_sf = params[0]
    if fit_offset:
        fitted_off = params[1]
    if len(params) > 2:
        fitted_fracs = params[2:]
        fitted_fracs = np.append(fitted_fracs, 1 - fitted_fracs.sum())
    else:
        fitted_fracs = [1.] if fit_fracs else fracs
    print('\nFitted parameters:')
    print('- Scale factor: {:.3f}'.format(fitted_sf))
    if fit_offset:
        print('- Offset: {:.3f}'.format(fitted_off))
    if fit_fracs:
        for (comp, frac) in zip(comps, fitted_fracs):
            print('- {} fraction: {:.0f} %'.format(comp.capitalize(), 100*frac))
    print()      

    # Resampling.
    x1, x2 = min(wavelength_silreg), max(wavelength_silreg)
    mask = (wavelength >= x1) & (wavelength <= x2)
    interp_wavelength = np.linspace(x1, x2, 10*mask.sum())
    model_absorbance = model(wavelength_silreg, *params)
    interp = interp1d(wavelength_silreg, model_absorbance, kind='quadratic')
    interp_absorbance = interp(interp_wavelength)
    wavelength_fine = wavelength[~mask]
    sil_wavelength = np.concatenate((wavelength_fine, interp_wavelength))
    sil_absorbance = np.concatenate((np.zeros(len(wavelength_fine)), interp_absorbance))
    sil_absorbance[sil_absorbance < 0.] = 0.
    
else:    

    # Reading of silicate file.
    if silicate_file.endswith('.csv'):
        data = pd.read_csv(os.path.join(folder, silicate_file)).values
    else:
        data = np.loadtxt(silicate_file)
    x, y = data.T
    if silicate_intensity_variable == 'optical depth':
        y /= np.log(10)
    sil_wavelength = x
    sil_absorbance = y
    
# Silicate subtraction.
sil_absorbance_interp = np.interp(wavelength, sil_wavelength, sil_absorbance)
absorbance_corr = absorbance - sil_absorbance_interp
wavenumber_ = wavenumber.copy()
mask = wavelength <= np.max(sil_wavelength)
wavelength = wavelength[mask]
sil_absorbance_interp = sil_absorbance_interp[mask]
absorbance_corr = absorbance_corr[mask]
absorbance_corr_rv = rv.RichArray(absorbance_corr, absorbance_uncs[mask])
wavenumber = 1e4 / wavelength
sil_wavenumber = 1e4 / sil_wavelength
    

#%% Plots.

plt.close('all')
plt.figure(1, figsize=figsize)

plt.subplot(2,1,1)
plt.axvspan(wavenumber_[-1], wavenumber_[-1], color='gray', alpha=0.1,
            label='regions of the fit')
plt.plot(sil_wavenumber, sil_absorbance, color='chocolate', alpha=0.8,
         zorder=3., label='fitted silicate')
rv.errorbar(wavenumber_, absorbance_rv, fmt='.-', alpha=0.6, color='black',
            ecolor='gray', ms=1., lw=1., drawstyle='steps-mid')
for (x1,x2) in regions:
    for (fc, ec) in zip(['gray', 'none'], ['none', 'gray']):
        plt.axvspan(1e4/x1, 1e4/x2, facecolor=fc, edgecolor=ec, hatch='/', alpha=0.1)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.margins(x=0.)
plt.xlim(plt.xlim()[::-1])
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.ylabel('absorbance')

ax1 = ax = plt.gca()
ax.invert_yaxis()
ax2 = ax.secondary_xaxis('top', functions=(xaxis_conversion, xaxis_conversion))
wavelength_ticks = [2.5, 3, 4, 5, 7, 10, 15, 30]
ax2.set_xticks(wavelength_ticks, wavelength_ticks)
ax2.set_xlabel('wavelength (μm)', labelpad=6, fontsize=9)
if fit_silicate:
    for (comp, frac) in zip(comps, fitted_fracs):
        plt.plot([], alpha=0., label='{}: {:.0f} %'.format(comp, 100*frac))
plt.legend(fontsize='small', loc='lower right')

plt.subplot(2,1,2, sharex=ax)
rv.errorbar(wavenumber, absorbance_corr_rv/np.log(10), fmt='.-', ms=1., lw=1.,
            color='black', ecolor='gray', alpha=0.6, drawstyle='steps-mid')
for (x1,x2) in regions:
    for (fc, ec) in zip(['gray', 'none'], ['none', 'gray']):
        plt.axvspan(1e4/x1, 1e4/x2, facecolor=fc, edgecolor=ec, hatch='/', alpha=0.1)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.ylabel('corrected absorbance')
ax = plt.gca()
ax.invert_yaxis()

plt.suptitle(name, fontweight='bold')
plt.tight_layout()

#%% Saving of plot and files.

path = os.path.join(folder, 'silicatefit.png')
plt.savefig(path)
print('Saved plot in {}.'.format(path))

path = os.path.join(folder, f'{name}-silic.txt')
np.savetxt(path, np.array([wavelength, sil_absorbance_interp]).T,
           header='wavenumber_(/cm) absorbance_(mJy)', fmt=['%.7f','%.9f'])
print('Saved silicate fit in {}.'.format(path))

if output_file.endswith('.txt') or output_file.endswith('.dat'):
    absorbance_corr_txt = absorbance_corr_rv.mains
    absorbance_corr_unc_txt = absorbance_corr_rv.uncs.mean(axis=1)
    path = os.path.join(folder, output_file)
    np.savetxt(path, np.array([wavenumber, absorbance_corr_txt, absorbance_corr_unc_txt]).T,
               header='wavenumber_(/cm) absorbance absorbance_unc',
               fmt=['%.7f','%.9f', '%.9f'])
    print(f'Saved absorbance spectrum in {path}.')

if output_file.endswith('.csv'):
    df = rv.RichDataFrame({'wavenumber (/cm)': wavenumber_,
                           'absorbance': absorbance_corr_rv})
    df['wavenumber (/cm)'] = df['wavenumber (/cm)'].map(
                                                 lambda x: '{:.{}f}'.format(x, 2))
    df.set_params({'num_sf': 3})
    path = os.path.join(folder, output_file)
    df.to_csv(path, index=False)
    print(f'Saved absorbance spectrum in {path}.')
