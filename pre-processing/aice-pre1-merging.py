#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.0
------------------------------------------
Pre-processing module 1 - Spectra merging

Andrés Megías
"""

# Configuration file.
config_file = 'NIR38.yaml'  # 'J110621.yaml'

# Libraries.
import os
import sys
import yaml
import warnings
import spectres
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Functions.

def axis_conversion(x):
    """Axis conversion from wavenumber to wavelength and viceversa"""
    with np.errstate(divide='ignore'):
        y = 1e4 / x
    return y

def supersample_spectrum(new_wavs, spec_wavs, spec_fluxes, spec_errs=None,
                         fill=np.nan):
    """Supersample spectrum to the input points."""
    new_fluxes = np.interp(new_wavs, spec_wavs, spec_fluxes, left=fill, right=fill)
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

#%% Initial options.

print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Pre-processing module 1 - Spectra merging')
print()

# Reading.
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

# Options.
folder = config['parent folder'] if 'parent folder' in config else '' 
figsize = config['figure size'] if 'figure size' in config else (9., 7.)
options = config['merging']
input_files = options['input files']
new_wavenumber_range = options['new wavenumber range (/cm)']
new_resolution = options['new resolution (/cm)']
xrange = [min(new_wavenumber_range), max(new_wavenumber_range)]
new_wavenumber = np.arange(*xrange, new_resolution)
colors = options['colors'] if 'colors' in options else 'auto'
flux_units = options['flux units'] if 'flux units' in options else 'mJy'
output_file = options['output file']
if colors == 'auto':
    colors = {}
    for (i, name) in enumerate(input_files):
        colors[name] = 'C{}'.format(i+1)
if folder.endswith('/'):
    folder = folder[:-1]
 
#%% Original spectra and resampling.

# Reading of files.
spectra_orig, spectra_res = {}, {}
for name in input_files:
    file = options['input files'][name]['file']
    if file.endswith('.txt') or file.endswith('.dat'):
        spectrum = np.loadtxt('{}/{}'.format(folder, file), comments=';')
    else:
        spectrum_df = pd.read_csv(file, index_col=0)
        spectrum = spectrum_df.values
    wavelength = spectrum[:,0]
    wavenumber = 1e4 / wavelength
    spectrum[:,0] = wavenumber
    inds = np.argsort(wavenumber)
    spectrum = spectrum[inds,:]
    file_options = options['input files'][name]
    xrange = file_options['wavenumber range (/cm)']
    wavenumber = spectrum[:,0]
    mask = (wavenumber >= min(xrange)) & (wavenumber <= max(xrange))
    spectrum = spectrum[mask,:]
    offset = file_options['offset'] if 'offset' in file_options else 0.
    spectrum[:,1] = spectrum[:,1] + offset
    spectra_orig[name] = spectrum

# Individual resampling.
for (name,spectrum) in spectra_orig.items():
    flux, flux_unc = resample_spectrum(new_wavenumber, spectrum[:,0],
                      spectrum[:,1], spectrum[:,2], fill=np.nan, verbose=False)
    if 'wavenumber gaps (/cm)' in options['input files'][name]:
        for gap in options['input files'][name]['wavenumber gaps (/cm)']:
            mask = (new_wavenumber >= min(gap)) & (new_wavenumber <= max(gap))
            flux[mask] = np.nan
    spectrum_res = np.array([new_wavenumber, flux, flux_unc]).T
    spectra_res[name] = spectrum_res
# Merging of the spectra.
fluxes, fluxes_unc = [], []
for spectrum in spectra_res.values():
    fluxes += [spectrum[:,1]]
    fluxes_unc += [spectrum[:,2]]
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    flux = np.nanmean(fluxes, axis=0)
with np.errstate(divide='ignore'):
    flux_unc = 1 / (np.nansum(1/np.array(fluxes_unc)**2, axis=0))**0.5
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    num_values = np.maximum(1., np.isfinite(fluxes).sum(axis=0))
    flux_std = np.nanstd(fluxes, axis=0) / np.sqrt(num_values)
flux_unc = np.maximum(flux_unc, flux_std)
new_wavelength = 1e4 / new_wavenumber
flux[flux < 0.] = 0.
spectrum_final = np.array([new_wavelength[::-1], flux[::-1], flux_unc[::-1]]).T
mask = np.isfinite(spectrum_final[:,1])
spectrum_final = spectrum_final[mask,:]

#%% Plots.
 
plt.close('all')
plt.figure(1, figsize=figsize)

gs = plt.GridSpec(2, 1, height_ratios=[3,1], hspace=0)

ax = plt.gcf().add_subplot(gs[0])
for (name,spectrum) in spectra_orig.items():
    plt.errorbar(spectrum[:,0], spectrum[:,1], yerr=spectrum[:,2], fmt='.-',
                 color=colors[name], ecolor='black', label=name, ms=7, alpha=0.8)
plt.errorbar(1e4/spectrum_final[:,0], spectrum_final[:,1], yerr=spectrum_final[:,2],
              fmt='.-', color='black', ecolor='gray', ms=1., lw=1., label='merged')
ax.axhline(y=0, color='black', ls='--', lw=0.8)
ax.set_ylabel(f'spectral flux ({flux_units})')
ax.set_title(config_file.replace('.yaml', ''), pad=15, fontweight='bold')
ax.legend()
ax.invert_xaxis()
plt.setp(ax.get_xticklabels(), visible=False)
ax2 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
wavelength_ticks = [1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20, 40]  # μm
ax2.set_xticks(wavelength_ticks, wavelength_ticks)
ax2.set_xlabel('wavelength (μm)', labelpad=10, fontsize=9)

ax = plt.gcf().add_subplot(gs[1], sharex=ax)
for (name,spectrum) in spectra_orig.items():
    plt.plot(spectrum[:,0][:-1], np.diff(spectrum[:,0]), '.', color=colors[name])
plt.axhline(y=new_resolution, ls='--', lw=0.5, color='gray')

ax.set_yscale('log')
ax.set_xlabel('wavenumber (cm$^{-1}$)', labelpad=10)
ax.set_ylabel('difference in' '\n' 'wavenumber (cm$^{-1}$)', fontsize=8)
ax2 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
ax2.xaxis.set_ticks([])
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
plt.margins(x=0.016)
xlims = plt.xlim()
ax3 = plt.twiny(ax2)
plt.xlim(xlims)
plt.tick_params(axis='x', which='both', direction='in', labeltop=False)
plt.tick_params(axis='x', which='both')

with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    plt.tight_layout()


#%% Saving of files.

folder += '/'.join(output_file.split('/')[:-1])
output_file = output_file.split('/')[-1]
path = folder + '/spectra.png'
plt.savefig(path)
print('Saved plot in {}.'.format(path))

if not os.path.exists(folder):
    os.makedirs(folder)
path = '{}/{}'.format(folder, output_file)
np.savetxt(path, spectrum_final,
           header=f'wavelength_(μm) flux_({flux_units}) flux_unc_({flux_units})',
           fmt=['%.7f','%.9f', '%.9f'])
print('Saved merged spectrum in {}.'.format(path))
