#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.1
------------------------------------------
Pre-processing module 2 - Continuum fit

Andrés Megías
"""

# Configuration file.
config_file = 'NIR38.yaml'

# Libraries.
import os
import sys
import yaml
import platform
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

sep = '\\' if platform.system() == 'Windows' else '/'  # folder separator

#%% Initial options.

print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Pre-processing module 2 - Continuum fit')
print()

# Default options.
default_options = {
    'figure size': (9., 7.),
    'comment character in input file': '#',
    'column indices': {'x': 1, 'y': 2, 'y unc.': 3},
    'input spectral variable': 'wavelength',
    'output spectral variable': 'wavenumber',
    'logarithmic fit': False,
    'fit color': 'palevioletred',
    'calculate accurate uncertainties': False,
    'show approximate uncertainties': False
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
config['continuum fit'] = {**default_options, **config['continuum fit']}

# Options.
folder = config['parent folder'] if 'parent folder' in config else '' 
options = config['continuum fit']
figsize = options['figure size']
input_file = options['input file']
comment_char = options['comment character in input file']
column_inds = options['column indices']
input_spectral_variable = options['input spectral variable']
output_spectral_variable = options['output spectral variable']
log_fit = options['logarithmic fit']
color = options['fit color']
fits = options['fits']
use_accurate_uncs = options['calculate accurate uncertainties']
show_approx_uncs = options['show approximate uncertainties']
output_file = options['output file']
if not use_accurate_uncs:
    show_approx_uncs = False
idx_x, idx_y = column_inds['x'] - 1, column_inds['y'] - 1
idx_dy = column_inds['y unc.'] - 1 if 'y unc.' in column_inds else None

#%% Reading of data and continuum fit.

# Data.
data = np.loadtxt(os.path.join(folder, input_file), comments=comment_char)
x = data[:,idx_x]
y = data[:,idx_y]
dy = data[:, idx_dy] if idx_dy is not None else np.zeros(len(y))
if input_spectral_variable == 'wavenumber':
    x = 1e4 / x
wavelength = x
flux = y
flux_unc = dy
mask = flux > 0.
wavelength = wavelength[mask]
flux = flux[mask]
flux_unc = flux_unc[mask]
wavenumber = 1e4 / wavelength
continuous, all_regions = [], []

# Separate fits.
for fit in fits:
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
        cont_file = params['file']
        comment_char = (params['comment character in continuum file'] if
                     'comment character in continuum file' in params else '#')
        column_inds = (params['column indices'] if 'column indices' in params
                       else {'x': 1, 'y': 2})
        scale_factor = (float(params['scale factor'])
                        if 'scale factor' in params else 1.)
        cont_data = np.loadtxt(os.path.join(folder, cont_file),
                               comments=comment_char)
        idx_x, idx_y = column_inds['x'] - 1, column_inds['y'] - 1
        cont_wavelength = cont_data[:,idx_x]
        cont_flux = cont_data[:,idx_y] * scale_factor
        if 'description' in params:
            for fit in params['description']:
                fit_type_ = list(fit.keys())[0]
                params_ = fit[fit_type_]
                if 'fit regions (μm)' in params_:
                    regions = params_['fit regions (μm)']
                elif 'fit regions (/cm)' in params_:
                    regions = params_['fit regions (/cm)']
                all_regions += regions
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
wavenumber_ = wavenumber.copy()
flux_ = flux.copy()
flux_unc_ = flux_unc.copy()
mask = ((wavelength >= cont_wavelength.min())
        & (wavelength <= cont_wavelength.max()))
wavelength = wavelength[mask]
wavenumber = wavenumber[mask]
flux = flux[mask]
flux_unc = flux_unc[mask]
cont_wavelength, cont_flux = np.concatenate([*continuous]).transpose()
inds = np.argsort(cont_wavelength)
cont_wavelength = cont_wavelength[inds]
cont_flux = cont_flux[inds]
mask = (cont_wavelength >= wavelength.min()) & (cont_wavelength <= wavelength.max())
cont_wavelength = cont_wavelength[mask]
cont_flux = cont_flux[mask] 
cont_flux_res = np.interp(wavelength, cont_wavelength, cont_flux)

# Optical depth.
with np.errstate(invalid='ignore'):
    absorbance = -np.log10(flux / cont_flux_res)
if not use_accurate_uncs or show_approx_uncs:
    absorbance_unc = flux_unc / flux / np.log(10)
    if not show_approx_uncs:
        abs_absorbance = np.abs(absorbance)
        mask = ((absorbance > 0.5*np.max(absorbance)) & (absorbance_unc > 0.5*abs_absorbance)
                | (absorbance - absorbance_unc < -np.nanmedian(abs_absorbance)))
        # absorbance[mask] = np.nan
        absorbance_unc[mask] = np.nan
        absorbance_rv = rv.RichArray(absorbance, absorbance_unc)
if use_accurate_uncs:
    print('Propagating observational uncertainties...')
    flux_rv = rv.RichArray(flux, flux_unc)
    with np.errstate(invalid='ignore'):
        absorbance_rv = rv.function_with_rich_arrays('-np.log10({}/{})',
                [flux_rv, rv.RichArray(cont_flux_res)], elementwise=True,
                    consider_intervs=False, len_samples=800)
    print()
wavenumber = 1e4 / wavelength
cont_wavenumber = 1e4 / cont_wavelength

#%% Plots.

plt.close('all')
plt.figure(1, figsize=figsize)

plt.subplot(2,1,1)
if len(all_regions) > 0:
    plt.axvspan(1e4/all_regions[0][0], 1e4/all_regions[0][0], color='gray', alpha=0.1,
                label='regions of the fit')
plt.errorbar(wavenumber_, flux_, flux_unc_, fmt='.-', color='black', ecolor='gray',
             alpha=0.6, ms=1., lw=1., drawstyle='steps-mid')
plt.plot(cont_wavenumber, cont_flux, color=color, alpha=0.8,
         zorder=2.5, label='fitted continuum')
for (x1,x2) in all_regions:
    for (fc, ec) in zip(['none', 'gray'], ['gray', 'none']):
        plt.axvspan(1e4/x1, 1e4/x2, facecolor=fc, edgecolor=ec, hatch='/', alpha=0.1)
plt.margins(x=0.)
plt.xlim(plt.xlim()[::-1])
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.yscale('log')
plt.ylabel('spectral flux density (mJy)')
plt.legend(fontsize='small', loc='lower right')
ax = plt.gca()
ax2 = ax.secondary_xaxis('top', functions=(xaxis_conversion, xaxis_conversion))
wavelength_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 25, 30, 40, 50, 100]
wavelength_ticklabels = [1, 2, 3, 4, 5, 6, 7, 8, '', 10, '', 12, '', '', 15,
                         '', '', '', '', 20, '', 30, '', 50, 100]
ax2.set_xticks(wavelength_ticks, wavelength_ticklabels)
ax2.set_xlabel('wavelength (μm)', labelpad=6, fontsize=9)

plt.subplot(2,1,2, sharex=ax)
rv.errorbar(wavenumber, absorbance_rv, fmt='.-', color='black',
            ecolor='gray', ms=1., lw=1., alpha=0.5, drawstyle='steps-mid')
if show_approx_uncs:
    ylims = plt.ylim()
    optdepth_unc = flux_unc / flux
    plt.errorbar(wavenumber+3e-4, absorbance, absorbance_unc, fmt='.-',
                 color='chocolate', ecolor='brown', alpha=0.5, zorder=1.5,
                 drawstyle='steps-mid', label='(approx. uncertainties)')
    plt.ylim(ylims)
    plt.legend(fontsize='small', loc='lower right')
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

# path = os.path.join(folder, 'continuumfit.png')
# plt.savefig(path)
# print(f'Saved plot in {path}.')

path = os.path.join(folder, f'{name}-cont.txt')
np.savetxt(path, np.array([cont_wavenumber, cont_flux]).T,
           header='wavenumber_(/cm) continuum_(mJy)', fmt=['%.7f','%.9f'])
print(f'Saved fitted continuum in {path}.')

if output_file.endswith('.txt') or output_file.endswith('.dat'):
    absorbance_txt = absorbance_rv.mains
    absorbance_unc_txt = absorbance_rv.uncs.mean(axis=1)
    path = os.path.join(folder, output_file)
    np.savetxt(path, np.array([wavenumber, absorbance_txt, absorbance_unc_txt]).T,
               header='wavenumber_(/cm) absorbance absorbance_unc',
               fmt=['%.7f','%.7f', '%.7f'])
    print(f'Saved absorbance spectrum in {path}.')

if output_file.endswith('.csv'):
    df = rv.RichDataFrame({'wavenumber (/cm)': wavenumber,
                           'absorbance': absorbance_rv})
    df.set_params({'num_sf': 3})
    path = os.path.join(folder, output_file)
    df.to_csv(path, index=False)
    print(f'Saved absorbance spectrum in {path}.')
