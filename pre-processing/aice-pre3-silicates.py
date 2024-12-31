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
config_file = 'NIR38.yaml'  # NIR38, J110621

# Libraries
import os
import sys
import yaml
import time
import optool
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

composition_abbreviations = {'pyroxene': 'pyr', 'olivine': 'ol'}

#%% Initial options.

print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Pre-processing module 3 - Silicate fit')
time.sleep(0.5)


# Reading.
config_path = './' + config_file if len(sys.argv) == 1 else sys.argv[1]
name = config_path.replace('.yaml', '').replace('./', '').replace('.', '')
config_path = os.path.realpath(config_path)
config_folder = '/'.join(config_path.split('/')[:-1]) + '/'
os.chdir(config_folder)
if os.path.isfile(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
else:
    raise FileNotFoundError('Configuration file not found.')

# Options.
folder = config['parent folder'] if 'parent folder' in config else ''
figsize = config['figure size'] if 'figure size' in config else (9., 7.)
options = config['silicate removal']
file = options['input file']
folder += '/'.join(file.split('/')[:-1])
file = file.split('/')[-1]
xrange = options['range (μm)']
regions = options['fit regions (μm)'] if 'fit regions (μm)' in options else [xrange]
composition = options['composition']
grain_size = options['grain size (μm)']
optool_path = (options['OpTool path'] if 'OpTool path' in options
               else '~/Documents/optool/optool')
if not optool_path.endswith(' '):
    optool_path += ' '
    
#%% Reading and fit.

# Reading of file.
if '.csv' in file:
    df = rv.rich_df(pd.read_csv('{}/{}'.format(folder, file)))
    wavenumber_rv, absorbance_rv = df.values.T
    wavenumber = wavenumber_rv.mains
    absorbance = absorbance_rv.mains
    absorbance_uncs = absorbance_rv.uncs
else:
    data = np.loadtxt('{}/{}'.format(folder, file))
    wavenumber, absorbance, absorbance_uncs = data.transpose()
absorbance_rv = rv.RichArray(absorbance, absorbance_uncs)
wavelength = 1e4 / wavenumber
mask = (wavelength >= xrange[0]) & (wavelength <= xrange[1])
wavelength_silreg = wavelength[mask]
absorbance_silreg = absorbance[mask]

# Individual simulations.
sil_spectra = []
comps, fracs = [], []
for comp_entry in composition:
    if type(comp_entry) is str:
        comp = comp_entry
        comps += [comp]
        comp = composition_abbreviations[comp_entry]
        fit_fracs = True
        frac = None
    elif type(comp_entry) is dict:
        comp = list(comp_entry.keys())[0]
        comps += [comp]
        comp = composition_abbreviations[comp]
        fit_fracs = False
        frac = float(list(comp_entry.values())[0])
    silicate = optool.particle(optool_path + '{} -a {}'
                               .format(comp, grain_size))
    sil_wavelength = silicate.lam
    sil_abscoeff = silicate.kabs.flatten()
    mask = (sil_wavelength >= xrange[0]) & (sil_wavelength <= xrange[1])
    sil_wavelength = sil_wavelength[mask]
    sil_abscoeff = sil_abscoeff[mask]
    sil_absorbance = - sil_abscoeff
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
if len(composition) == 1:
    def model(x, sf, off):
        spectrum = np.interp(x, sil_wavelength, sil_spectra[0])
        absorbance = sf * spectrum + off
        return absorbance
    bounds = ([sf_bounds[0], off_bounds[0]], [sf_bounds[1], off_bounds[1]])
elif len(composition) == 2:
    if fit_fracs:
        def model(x, sf, off, c):
            c1, c2 = c, 1-c
            spectrum1 = np.interp(x, sil_wavelength, sil_spectra[0])
            spectrum2 = np.interp(x, sil_wavelength, sil_spectra[1])
            absorbance = sf * (c1 * spectrum1 + c2 * spectrum2) + off
            return absorbance
        bounds = ([sf_bounds[0], off_bounds[0], frac_bounds[0]],
                  [sf_bounds[1], off_bounds[1], frac_bounds[1]])
    else:
        fracs = np.array(fracs)
        fracs /= np.sum(fracs)
        def model(x, sf, off):
            c1, c2 = fracs
            spectrum1 = np.interp(x, sil_wavelength, sil_spectra[0])
            spectrum2 = np.interp(x, sil_wavelength, sil_spectra[1])
            absorbance = sf * (c1 * spectrum1 + c2 * spectrum2) + off
            return absorbance
        bounds = ([sf_bounds[0], off_bounds[0]], [sf_bounds[1], off_bounds[1]])

# Silicate fit.
mask = np.zeros(len(wavelength), bool)
for region in regions:
    mask += (wavelength >= region[0]) & (wavelength <= region[1])
wavelength_mask = wavelength[mask]
params = curve_fit(model, wavelength[mask], absorbance[mask], bounds=bounds)[0]
fitted_sf = params[0]
fitted_off = params[1]
if len(params) > 2:
    fitted_fracs = params[2:]
    fitted_fracs = np.append(fitted_fracs, 1 - fitted_fracs.sum())
else:
    fitted_fracs = [1.] if fit_fracs else fracs
print('\nFitted parameters:')
print('- Scale factor: {:.2f}'.format(fitted_sf))
print('- Offset: {:.2f}'.format(fitted_off))
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
sil_absorbance_interp = np.interp(wavelength, sil_wavelength, sil_absorbance)
absorbance_corr = absorbance - sil_absorbance_interp
absorbance_corr_rv = rv.RichArray(absorbance_corr, absorbance_uncs)
wavenumber = 1e4 / wavelength
sil_wavenumber = 1e4 / sil_wavelength

#%% Plots.

plt.close('all')
plt.figure(1, figsize=figsize)

plt.subplot(2,1,1)
plt.axvspan(wavenumber[-1], wavenumber[-1], color='gray', alpha=0.1,
            label='regions of the fit')
plt.plot(sil_wavenumber, sil_absorbance, color='chocolate', alpha=0.8,
         zorder=3., label='fitted silicate')
rv.errorbar(wavenumber, absorbance_rv, fmt='.-', alpha=0.6, color='black',
            ecolor='gray', ms=1., lw=1.)
for (x1,x2) in regions:
    for (fc, ec) in zip(['gray', 'none'], ['none', 'gray']):
        plt.axvspan(1e4/x1, 1e4/x2, facecolor=fc, edgecolor=ec, hatch='/', alpha=0.1)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.margins(x=0.02)
plt.xlim(plt.xlim()[::-1])
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.ylabel('absorbance')

ax1 = ax = plt.gca()
ax.invert_yaxis()
ax2 = ax.secondary_xaxis('top', functions=(xaxis_conversion, xaxis_conversion))
wavelength_ticks = [2.5, 3, 4, 5, 7, 10, 15, 30]
ax2.set_xticks(wavelength_ticks, wavelength_ticks)
ax2.set_xlabel('wavelength (μm)', labelpad=6, fontsize=9)
for (comp, frac) in zip(comps, fitted_fracs):
    plt.plot([], alpha=0., label='{}: {:.0f} %'.format(comp, 100*frac))
plt.legend(fontsize='small', loc='lower right')

plt.subplot(2,1,2, sharex=ax)
rv.errorbar(wavenumber, absorbance_corr_rv/np.log(10), fmt='.-', ms=1., lw=1.,
            color='black', ecolor='gray', alpha=0.6)
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

path = '{}/silicatefit.png'.format(folder)
plt.savefig(path)
print('Saved plot in {}.'.format(path))

path = '{}/{}-silic.txt'.format(folder, name)
np.savetxt(path, np.array([wavelength, sil_absorbance_interp]).T,
           header='wavenumber_(/cm) absorbance_(mJy)', fmt=['%.7f','%.9f'])
print('Saved silicate fit in {}.'.format(path))

absorbance_corr_txt = absorbance_corr_rv.mains
absorbance_corr_unc_txt = absorbance_corr_rv.uncs.mean(axis=1)
path = '{}/{}-c-s.txt'.format(folder, name)
np.savetxt(path, np.array([wavenumber, absorbance_corr_txt, absorbance_corr_unc_txt]).T,
           header='wavenumber_(/cm) absorbance absorbance_unc',
           fmt=['%.7f','%.9f', '%.9f'])
print('Saved absorbance spectrum in {}.'.format(path))
df = rv.RichDataFrame({'wavenumber (/cm)': wavenumber,
                       'absorbance': absorbance_corr_rv})
df['wavenumber (/cm)'] = df['wavenumber (/cm)'].map(
                                             lambda x: '{:.{}f}'.format(x, 2))
df.set_params({'num_sf': 3})
path = '{}/{}-c-s.csv'.format(folder, name)
df.to_csv(path, index=False)
print('Saved absorbance spectrum in {}.'.format(path))
