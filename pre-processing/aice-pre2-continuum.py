#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.0
------------------------------------------
Pre-processing module 2 - Continuum fit

Andrés Megías
"""

# Configuration file.
config_file = 'NIR38.yaml'  # NIR38, J110621

# Libraries.
import os
import sys
import yaml
import numpy as np
import richvalues as rv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
rv.set_default_params({'sigmas to define upper/lower limits from read values': 1.})

# Functions.

def polynomial(x, *coeffs):
    """Apply a polinomial to x with the given coefficients (coeffs)."""
    order = len(coeffs) - 1
    y = coeffs[-1]
    for i in reversed(range(1, order+1)):
        y += coeffs[i-1] * x**i
    return y

def black_body(wl, T, sf):
    """
    Black body spectrum with the given temperature (T) and scale factor (sf),
    with respect to the wavelength (l).
    """
    wl = wl*1e-6
    F = sf * 2*h*c**2/wl**5 / np.exp(h*c/(wl*kB*T) - 1)
    F *= wl**2 / c
    return F

def xaxis_conversion(x):
    """Convert wavelength to wavenumber and viceversa."""
    with np.errstate(divide='ignore'):
        y = 1e4 / x
    return y

# Constants.
h = planck_constant = 6.626e-34  # J*s
kB = boltzmann_constant = 1.381e-23  # J/K 
c = light_speed = 2.998e8  # m/s

#%% Initial options.

print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Pre-processing module 2 - Continuum fit')
print()

# Configuration file. 
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
options = config['continuum fit']
file = options['input file']
folder += '/'.join(file.split('/')[:-1])
file = file.split('/')[-1]
log_fit = options['logarithmic fit'] if 'logairthmic fit' in options else True
color = options['fit color'] if 'fit color' in options else 'palevioletred'
fits = options['fits']
use_accurate_uncs = (options['estimate accurate uncertainties'] if
                     'estimate accurate uncertainties' in options else True)
show_approx_uncs = (options['show approximate uncertainties'] if
                    'show approximate uncertainties' in options else False)
if not use_accurate_uncs:
    show_approx_uncs = False

#%% Reading of data and continuum fit.

# Data.
data = np.loadtxt('{}/{}'.format(folder, file))
wavelength, flux, flux_unc = data.transpose()
mask = flux > 0.
wavelength = wavelength[mask]
flux = flux[mask]
flux_unc = flux_unc[mask]
wavenumber = 1e4 / wavelength
continuous, all_regions = [], []

# Separate fits.
for (i, fit) in enumerate(fits):
    fit_type = list(fit.keys())[0]
    params = fit[fit_type]
    if fit_type in ('polynomial', 'black body'):
        if 'range (μm)' in params:
            xrange = params['range (μm)']
        elif 'range (/cm)' in params:
            xrange = params['range (/cm)']
            xrange = [1e4 / xrange[1], 1e4 / xrange[0]]
        if 'fit regions (μm)' in params:
            regions = params['fit regions (μm)']
            use_wavelength = True
        elif 'fit regions (/cm)' in params:
            regions = params['fit regions (/cm)']
            use_wavelength = False
        mask = np.zeros(len(wavelength), bool)
        for x1x2 in regions:
            x1, x2 = min(x1x2), max(x1x2)
            if use_wavelength:
                mask += (wavelength >= x1) & (wavelength <= x2)
            else:
                mask += (wavenumber >= x1) & (wavenumber <= x2)
        wavelength_mask = wavelength[mask]
        flux_mask = flux[mask]
        all_regions += regions
    elif fit_type == 'continuum file':
        params = fit
        cont_file = '../' + params['continuum file']
        cont_data = np.loadtxt(cont_file)
    if fit_type == 'polynomial':
        order = int(params['order'])
        guess = np.zeros(order+1)
        guess[-1] = np.nanmean(flux)
        fit_function = polynomial
    elif fit_type == 'black body':
        temperature = float(params['temperature (K)'])
        scale_factor = float(params['scale factor'])
        guess = (temperature, scale_factor)
        fit_function = black_body
    if fit_type in ('polynomial', 'black body'):
        if fit_type == 'polynomial' and log_fit:
            flux_mask = np.log(flux_mask)
        result = curve_fit(fit_function, wavelength_mask, flux_mask, p0=guess)
        params = result[0]
        mask = (wavelength >= xrange[0]) & (wavelength <= xrange[-1])
        cont_wavelength = wavelength[mask]
        cont_flux = fit_function(cont_wavelength, *params)
        if fit_type == 'polynomial' and log_fit:
            cont_flux = np.exp(cont_flux)
        cont_data = np.array([cont_wavelength, cont_flux]).T
    continuous += [cont_data]

# Merging.
cont_wavelength, cont_flux = np.concatenate([*continuous]).transpose()
inds = np.argsort(cont_wavelength)
cont_wavelength = cont_wavelength[inds]
cont_flux = cont_flux[inds]
mask = (cont_wavelength >= wavelength[0]) & (cont_wavelength <= wavelength[-1])
cont_wavelength = cont_wavelength[mask]
cont_flux = cont_flux[mask] 
cont_flux_res = np.interp(wavelength, cont_wavelength, cont_flux)

# Optical depth.
absorbance = -np.log10(flux / cont_flux_res)
if use_accurate_uncs:
    print('Calculating...')
    flux_rv = rv.RichArray(flux, flux_unc, domains=[0,np.inf])
    absorbance_rv = rv.function_with_rich_arrays('-np.log10({}/{})',
            [flux_rv, rv.RichArray(cont_flux_res)], elementwise=True,
                consider_intervs=False, len_samples=800)
else:
    absorbance_unc = flux_unc / flux / np.log(10)
    absorbance_rv = rv.RichArray(absorbance, absorbance_unc)
wavenumber = 1e4 / wavelength
cont_wavenumber = 1e4 / cont_wavelength

#%% Plots.

plt.close('all')
plt.figure(1, figsize=figsize)

plt.subplot(2,1,1)
plt.axvspan(wavenumber[-1], wavenumber[-1], color='gray', alpha=0.1,
            label='regions of the fit')
plt.errorbar(wavenumber, flux, flux_unc, fmt='.-', color='black', ecolor='gray',
             alpha=0.6, ms=1., lw=1.)
plt.plot(cont_wavenumber, cont_flux, color=color, alpha=0.8,
         zorder=2.5, label='fitted continuum')
for (x1,x2) in all_regions:
    for (fc, ec) in zip(['none', 'gray'], ['gray', 'none']):
        plt.axvspan(1e4/x1, 1e4/x2, facecolor=fc, edgecolor=ec, hatch='/', alpha=0.1)
plt.margins(x=0.02)
plt.xlim(plt.xlim()[::-1])
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.yscale('log')
plt.ylabel('spectral flux (mJy)')
plt.legend(fontsize='small', loc='lower right')
ax = plt.gca()
ax2 = ax.secondary_xaxis('top', functions=(xaxis_conversion, xaxis_conversion))
wavelength_ticks = [1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 30]
ax2.set_xticks(wavelength_ticks, wavelength_ticks)
ax2.set_xlabel('wavelength (μm)', labelpad=6, fontsize=9)

plt.subplot(2,1,2, sharex=ax)
rv.errorbar(wavenumber, absorbance_rv, fmt='.-', color='black',
            ecolor='gray', ms=1., lw=1., alpha=0.5)
if show_approx_uncs:
    ylims = plt.ylim()
    optdepth_unc = flux_unc / flux
    plt.errorbar(wavenumber+3e-4, absorbance,absorbance_unc, fmt='.-',
                 color='chocolate', ecolor='brown', alpha=0.5, zorder=1.5)
    plt.ylim(ylims)
    plt.errorbar([], [], [], fmt='.', alpha=0.6, color='chocolate',
                 ecolor='brown', label='(approx. uncertainties)')
for (x1,x2) in all_regions:
    for (fc, ec) in zip(['none', 'gray'], ['gray', 'none']):
        plt.axvspan(1e4/x1, 1e4/x2, facecolor=fc, edgecolor=ec, hatch='/', alpha=0.1)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.ylabel('absorbance')
plt.gca().invert_yaxis()

plt.suptitle(name, fontweight='bold')
plt.tight_layout()  # h_pad = 0


#%% Saving of plot and files.

path = '{}/continuumfit.png'.format(folder)
plt.savefig(path)
print('Saved plot in {}.'.format(path))

path = '{}/{}-cont.txt'.format(folder, name)
np.savetxt(path, np.array([cont_wavenumber, cont_flux]).T,
           header='wavenumber_(/cm) continuum_(mJy)', fmt=['%.7f','%.9f'])
print('Saved fitted continuum in {}.'.format(path))

absorbance_txt = absorbance_rv.mains
absorbance_unc_txt = absorbance_rv.uncs.mean(axis=1)
path = '{}/{}-c.txt'.format(folder, name)
np.savetxt(path, np.array([wavenumber, absorbance_txt, absorbance_unc_txt]).T,
           header='wavenumber_(μm) absorbance absorbance_unc',
           fmt=['%.7f','%.7f', '%.7f'])
print('Saved absorbance spectrum in {}.'.format(path))

df = rv.RichDataFrame({'wavelength (μm)': wavelength,
                       'absorbance': absorbance_rv})
df.set_params({'num_sf': 3})
path = '{}/{}-c.csv'.format(folder, name)
df.to_csv(path, index=False)
print('Saved absorbance spectrum in {}.'.format(path))