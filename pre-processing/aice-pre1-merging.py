#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.1
------------------------------------------
Pre-processing module 1 - Spectra merging

Andrés Megías
"""

config_file = 'J110621.yaml'

# Libraries.
import os
import sys
import yaml
import platform
import warnings
import numpy as np
import pandas as pd
import richvalues as rv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Functions.

def axis_conversion(x):
    """Axis conversion from wavenumber to wavelength and viceversa"""
    with np.errstate(divide='ignore'):
        y = 1e4 / x
    return y


def spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None,
             fill=None, supersample_linearly=False, verbose=True):

    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.
    SpectRes function by Adam Carnall, slightly modified by Andrés Megías.

    Parameters
    ----------

    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.

    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.

    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.

    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.

    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.

    verbose : bool (optional)
        Setting verbose to False will suppress the default warning about
        new_wavs extending outside spec_wavs and "fill" being used.

    Returns
    -------

    new_fluxes : numpy.ndarray
        Array of resampled flux values, last dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.

    new_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    def make_bins(wavs):
        """ Given a series of wavelength points, find the edges and widths
        of corresponding wavelength bins. """
        edges = np.zeros(wavs.shape[0]+1)
        widths = np.zeros(wavs.shape[0])
        edges[0] = wavs[0] - (wavs[1] - wavs[0])/2
        widths[-1] = (wavs[-1] - wavs[-2])
        edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2])/2
        edges[1:-1] = (wavs[1:] + wavs[:-1])/2
        widths[:-1] = edges[1:-1] - edges[:-2]
        return edges, widths

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Make arrays of edge positions and widths for the old and new bins.
    old_edges, old_widths = make_bins(old_wavs)
    new_edges, new_widths = make_bins(new_wavs)

    # Generate output arrays to be populated.
    new_fluxes = np.zeros(old_fluxes[...,0].shape + new_wavs.shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        else:
            new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0

    # Calculate new flux and uncertainty values, looping over new bins.
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs.
        if (new_edges[j] < old_edges[0]) or (new_edges[j+1] > old_edges[-1]):
            new_fluxes[...,j] = fill

            if spec_errs is not None:
                new_errs[...,j] = fill

            if (j == 0 or j == new_wavs.shape[0]-1) and verbose:
                warnings.warn(
                    "Spectres: new_wavs contains values outside the range "
                    "in spec_wavs, new_fluxes and new_errs will be filled "
                    "with the value set in the 'fill' keyword argument "
                    "(by default NaN).",
                    category=RuntimeWarning,
                )
            continue

        # Find first old bin which is partially covered by the new bin.
        while old_edges[start+1] <= new_edges[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin.
        while old_edges[stop+1] < new_edges[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal.
        if stop == start:
            if supersample_linearly:
                i1 = max(0, start-1)
                i2 = min(start+1, old_wavs.shape[0]-1)
                if supersample_linearly:
                    new_fluxes[...,j] = np.interp(new_wavs[j],
                                        old_wavs[i1:i2+1], old_fluxes[i1:i2+1])
                else:
                    new_fluxes[...,j] = old_fluxes[...,start]
            else:
                new_fluxes[...,j] = old_fluxes[...,start]
            if old_errs is not None:
                new_errs[...,j] = old_errs[...,start]
                # Artificially enlarge uncertainties to be consistent.
                new_errs[...,j] *= np.sqrt(old_widths[start] / new_widths[j])        

        # Otherwise multiply the first and last old bin widths by P_ij.
        else:
            start_factor = ((old_edges[start+1] - new_edges[j])
                            / (old_edges[start+1] - old_edges[start]))

            end_factor = ((new_edges[j+1] - old_edges[stop])
                          / (old_edges[stop+1] - old_edges[stop]))
            old_widths_local = old_widths[start:stop+1].copy()
            if not supersample_linearly or supersample_linearly and stop != start+1:
                old_widths_local[0] *= start_factor
                old_widths_local[stop-start] *= end_factor

            # Populate new_fluxes spectrum array.
            f_widths = old_widths_local * old_fluxes[...,start:stop+1]
            new_fluxes[...,j] = np.sum(f_widths, axis=-1)
            new_fluxes[...,j] /= np.sum(old_widths_local)
            # Populate new_fluxes uncertainty arrays.
            if old_errs is not None:
                # Case of new bin partially overlapping only one old bin.
                if stop == start+1:
                    # Including old flux value at new left edge.
                    if old_wavs[...,start] == new_edges[...,j]:
                        start = max(0, start-1)
                # Artificially enlarge uncertainties to be consistent.
                factor = np.sqrt(np.sum(old_widths[start:stop+1])/new_widths[j])
                # Compute uncertainties.
                e_wid = old_widths[start:stop+1] * old_errs[...,start:stop+1]
                new_errs[...,j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                new_errs[...,j] /= np.sum(old_widths[start:stop+1])
                new_errs[...,j] *= factor

    # If errors were supplied return both new_fluxes and new_errs.
    if old_errs is not None:
        return new_fluxes, new_errs

    # Otherwise just return the new_fluxes spectrum array
    else:
        return new_fluxes
    
def resample_spectrum(x_new, x, y, y_unc=None, supersample_linearly=False):
    """Use SpectRes to resamople input spectrum."""
    ssl = supersample_linearly
    result = spectres(x_new, x, y, y_unc, fill=np.nan,
                      supersample_linearly=ssl, verbose=False)
    return result

def fill_spectrum(x, y, y_unc=None, threshold=2.):
    """Fill the gaps in the input spectrum with NaNs."""
    x_new = np.array([x[0]], float)
    y_new = np.array([y[0]], float)
    y_new_unc = np.array([y_unc[0]], float) if y_unc is not None else None
    x_prev = x[0]
    dx_prev = np.median(np.diff(x[:3]))
    for i in range(1, len(x)):
        dx_i = x[i] - x_prev
        if dx_i > threshold * dx_prev:
            num_points = round(dx_i / dx_prev)
            dx_new = dx_i / num_points
            x_ = np.arange(x_prev+dx_prev, x[i]-dx_prev, dx_new)
            y_ = np.nan * np.ones(len(x_))
            x_new = np.append(x_new, x_)
            y_new = np.append(y_new, y_)
            if y_unc is not None:
                y_new_unc = np.append(y_new_unc, y_)
        x_new = np.append(x_new, x[i])
        y_new = np.append(y_new, y[i])
        if y_unc is not None:
            y_new_unc = np.append(y_new_unc, y_unc[i])
        x_prev = x_new[-1]
        dx_prev = dx_i
    if y_unc is None:
        return x_new, y_new
    else:
        return x_new, y_new, y_new_unc
    
def rolling_function(function, x, size, **kwargs):
    """
    Apply a function in a rolling way, in windows of the specified size.

    Parameters
    ----------
    x : array
        Input data (1-dimensional).
    func : function
        Function to be applied. It should admit arrays or lists as inputs.
    size : int or float
        Size of the windows to group the data.
    **kwargs : (various)
        Keyword arguments of the function to be applied.

    Returns
    -------
    y : array
        Resultant array.
    """
    if size <= 0:
        raise Exception('Window size must be positive.')
    size = int(size)
    N = len(x)
    if size >= round(N/2):
        raise Exception('Window size must be less than half of the data size.')
    elif size == 1:
        return x
    y = np.zeros(N)
    if size % 1 == 0:
        ic1 = size
        ic2 = N - size
        a = size // 2
        with warnings.catch_warnings(action="ignore"):
            if size % 2 == 1:
                for i in range(ic1, ic2):
                    y[i] = function(x[i-a:i+a+1], **kwargs)
                for i in range(0, ic1):
                    y[i] = function(x[max(0,i-a):i+a+1], **kwargs)
                for i in range(ic2, N):
                    y[i] = function(x[i-a:min(N,i+a+1)], **kwargs)
            else:
                for i in range(ic1, ic2):
                    y[i] = np.mean([function(x[i-a:i+a], **kwargs),
                                    function(x[i+1-a:i+1+a], **kwargs)])
                for i in range(0, ic1):
                    y[i] = np.mean([function(x[max(0,i-a):i+a], **kwargs),
                                    function(x[max(0,i+1-a):i+1+a], **kwargs)])
                for i in range(ic2, N):
                    y[i] = np.mean([function(x[i-a:min(N,i+a)], **kwargs),
                                    function(x[i+1-a:min(N,i+1+a)], **kwargs)])
    else:
        size1 = int(np.floor(size))
        size2 = int(np.ceil(size))
        ic1 = size2
        ic2 = N - size2
        a1 = size1 // 2
        a2 = size2 // 2
        with warnings.catch_warnings(action="ignore"):
            if size1 % 2 != 1:
                a1, a2 = a2, a1
                size1, size2 = size2, size1
                o = -1
            else:
                o = 1
            for i in range(ic1, ic2):
                y1 = function(x[i-a1:i+a1+1], **kwargs)
                y2 = np.mean([function(x[i-a2:i+a2], **kwargs),
                              function(x[i+1-a2:i+1+a2], **kwargs)])
                y[i] = np.interp(size, [size1, size2][::o], [y1, y2][::o])
            for i in range(0, ic1):
                y1 = function(x[max(0,i-a1):i+a1+1], **kwargs)
                y2 = np.mean([function(x[max(0,i-a2):i+a2], **kwargs),
                              function(x[max(0,i+1-a2):i+1+a2], **kwargs)])
                y[i] = np.interp(size, [size1, size2][::o], [y1, y2][::o])
            for i in range(ic2, N):
                y1 = function(x[i-a1:min(N,i+a1+1)], **kwargs)
                y2 = np.mean([function(x[i-a2:min(N,i+a2)], **kwargs),
                              function(x[i+1-a2:min(N,i+1+a2)], **kwargs)])
                y[i] = np.interp(size, [size1, size2][::o], [y1, y2][::o])
    y[np.isnan(x)] = np.nan
    return y
    
def divide_by_intersections(intervals):
    """Divide input intervals (x1, x2) by their intersections."""
    edges = []
    for (x1, x2) in intervals:
        edges += [x1, x2]
    edges = sorted(set(edges))
    subintervals = []
    for i in range(len(edges)-1):
        a, b = edges[i], edges[i+1]
        for (x1, x2) in intervals:
            if a >= x1 and b <= x2:
                subintervals += [(a, b)]
                break
    return subintervals

sep = '\\' if platform.system() == 'Windows' else '/'  # folder separator

#%% Initial options.

print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Pre-processing module 1 - Spectra merging')
print()

default_options = {
    'figure size': 'auto',
    'comment character in input files': '#',
    'column indices': {'x': 1, 'y': 2, 'y unc.': 3},
    'channels per spectral resolution element': 2.,
    'output spectral variable': 'wavelength',
    'show full original spectra': True,
    'flux units': 'a.u.',
    'flux offset': 0.,
    'use logarithmic scale for flux': False,
    'show spectral spacing': True,
    'show resolving power': True,
    'show signal-to-noise': True,
    'colors': 'auto',
    }

# Configuration file.
config_path = config_file if len(sys.argv) == 1 else sys.argv[1]
name = config_path.replace('.yaml', '')
config_path = os.path.realpath(config_path)
config_folder = sep.join(config_path.split('/')[:-1]) + sep
os.chdir(config_folder)
if os.path.isfile(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
else:
    raise FileNotFoundError('Configuration file not found.')
config['merging'] = {**default_options, **config['merging']}

# Options.
folder = config['parent folder'] if 'parent folder' in config else '' 
options = config['merging']
figsize = options['figure size']
input_files = options['input files']
comment_char = options['comment character in input files']
column_inds = options['column indices']
colors = options['colors']
flux_units = options['flux units']
y0 = flux_offset = options['flux offset']
spectral_factor = options['channels per spectral resolution element']
show_full_originlal_spectra = options['show full original spectra']
use_logscale = options['use logarithmic scale for flux']
show_spectral_spacing = options['show spectral spacing']
show_resolving_power = options['show resolving power']
show_signal_to_noise = options['show signal-to-noise']
output_file = options['output file']
idx_x, idx_y = column_inds['x'] - 1, column_inds['y'] - 1
idx_yu = column_inds['y unc.'] - 1 if 'y unc.' in column_inds else None
spectral_variable = options['output spectral variable']
resample_whole_data = (True if 'new spectral range (/cm)' in options
                 or 'new spectral range (μm)' in options else False)
if resample_whole_data:
    new_xrange = (options['new spectral range (μm)'] if 'new spectral range (μm)'
                  in options else options['new spectral range (/cm)'])
    new_xrange = sorted([new_xrange[0], new_xrange[1]]) +  [new_xrange[2]]
    resampling_variable = ('wavelength' if 'new spectral spacing (μm)' in options
                           else 'wavenumber')
if colors == 'auto':
    colors = {}
    for (i, name) in enumerate(input_files):
        colors[name] = 'C{}'.format(i+1)
if figsize == 'auto':
    figsize = [9., 5.]
    if show_spectral_spacing:
        figsize[1] += 1.
    if show_resolving_power:
        figsize[1] += 1.
    if show_signal_to_noise:
        figsize[1] += 1.
if folder.endswith(sep):
    folder = folder[:-1]
 
#%% Original spectra and resampling.

# Reading of files.
spectra_orig, spectra_new = {}, {}
for name in input_files:
    file_options = options['input files'][name]
    filename = file_options['file']
    if filename.endswith('.txt') or filename.endswith('.dat'):
        data = np.loadtxt(os.path.join(folder, filename), comments=comment_char)
    else:
        spectrum_df = pd.read_csv(file, index_col=0)
        data = spectrum_df.values
    x = data[:,idx_x]
    y = data[:,idx_y]
    y_unc = data[:,idx_yu] if data.shape[1] > 1 else np.zeros(len(y))
    if spectral_variable == 'wavenumber':
        x = 1e4/x
    x, inds = np.unique(x, return_index=True)
    y = y[inds]
    y_unc = y_unc[inds]
    x, y, y_unc = fill_spectrum(x, y, y_unc)
    dx = np.append(np.diff(x), [np.nan])
    dx[np.isnan(y)] = np.nan
    r = spectral_factor * dx
    if show_full_originlal_spectra:
        if spectral_variable != 'wavenumber':
            x = 1e4/x
        spectra_orig[name] = np.array([x, y, y_unc, r]).T
    if 'spectral range (μm)' in file_options:
        xranges = file_options['spectral range (μm)']
        if type(xranges[0]) is not list:
            xranges = [xranges]
        if spectral_variable != 'wavelength':
            for (i, x1x2) in enumerate(xranges):
                xranges[i] = [1e4/x1x2[0], 1e4/x1x2[1]]
    elif 'spectral range (/cm)' in file_options:
        xranges = file_options['spectral range (/cm)']
        if type(xranges[0]) is not list:
            xranges = [xranges]
        if spectral_variable != 'wavenumber':
            for (i, x1x2) in enumerate(xranges):
                xranges[i] = [1e4/x1x2[0], 1e4/x1x2[1]]
    else:
        xranges = [[x[0], x[-1]]]
    if not show_full_originlal_spectra:
        mask = np.zeros(len(x), bool)
        for (i, x1x2) in enumerate(xranges):
            x1, x2 = min(x1x2), max(x1x2)
            mask |= (x >= x1) & (x <= x2)
            x_ = x[mask]
            y_ = y[mask]
            y_unc_ = y_unc[mask]
            r_ = r[mask]
            _, r_ = fill_spectrum(x_, r_)
            x_, y_, y_unc_ = fill_spectrum(x_, y_, y_unc_)
            if spectral_variable != 'wavenumber':
                x_ = 1e4/x_
            spectra_orig[name] = np.array([x_, y_, y_unc_, r_]).T
    for (i, x1x2) in enumerate(xranges):
        x1, x2 = min(x1x2), max(x1x2)
        mask = (x >= x1) & (x <= x2)
        x_ = x[mask]
        y_ = y[mask]
        y_unc_ = y_unc[mask]
        r_ = r[mask]
        new_name = f'{name}-{i+1}' if len(xranges) > 1 else name
        spectra_new[new_name] = np.array([x_, y_+y0, y_unc_, r_]).T
    
# Combining of the spectra.
if not resample_whole_data:
    num_spectra = len(spectra_new)
    if num_spectra == 1:
        spectrum_new = spectra_new[0]
    else:
        # Resampling in the regions where spectra overlap (if any).
        xranges = [(float(spectrum[0,0]), float(spectrum[-1,0]))
                   for spectrum in spectra_new.values()]
        subranges = divide_by_intersections(xranges)
        xx_new = []
        for (x1, x2) in subranges:
            xx = []
            for spectrum in spectra_new.values():
                x = spectrum[:,0]
                mask = (x >= x1) & (x <= x2)
                if sum(mask) > 1:
                    xx += [x[mask]]
            dxx = [np.median(np.diff(x)) for x in xx]
            idx = np.argmax(dxx)
            x_new = xx[idx]
            xx_new += [x_new]
        for (i, name) in enumerate(spectra_new):
            spectrum = spectra_new[name]
            x, y, y_unc, r = spectrum.T
            spectrum_new = np.array([], float).reshape(-1,4)
            for x_new in xx_new:
                dx_new = np.append(np.diff(x_new), np.nan)
                mask = (x >= x_new[0]) & (x <= x_new[-1])
                if np.array_equal(x[mask], x_new):
                    y_new, y_new_unc, r_new = y[mask], y_unc[mask], r[mask]
                else:
                    if not any(mask):
                        num_points = len(x_new)
                        y_new = np.nan * np.ones(num_points)
                        y_new_unc = np.nan * np.ones(num_points)
                        r_new = np.nan * np.ones(num_points)
                    else:
                        y_new, y_new_unc = resample_spectrum(x_new, x, y, y_unc,
                                                      supersample_linearly=True)
                        r_new = spectral_factor * dx_new
                        r_old = np.interp(x_new, x, r, left=np.nan, right=np.nan)
                        r_new = np.maximum(r_new, r_old)
                spectrum_new_j = np.array([x_new, y_new, y_new_unc, r_new]).T
                spectrum_new = np.append(spectrum_new, spectrum_new_j, axis=0)
            spectra_new[name] = spectrum_new
        # Merging and averaging of the spectra.
        spectrum_new = np.array([], float).reshape(-1,4)
        for (x1, x2) in subranges:
            xx, yy, yy_unc, rr = [], [], [], []
            for spectrum in spectra_new.values():
                x, y, y_unc, r = spectrum.T
                mask = (x >= x1) & (x <= x2)
                xx += [x[mask]]
                yy += [y[mask]]
                yy_unc += [y_unc[mask]]
                rr += [r[mask]]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                x_new = xx[0]
                y_new = np.nanmean(yy, axis=0)
                r_new = np.nanmax(rr, axis=0)
            with np.errstate(divide='ignore'):
                y_new_unc = 1 / (np.nansum(1/np.array(yy_unc)**2, axis=0))**0.5
            zz = []
            for spectrum in spectra_new.values():
                x, y, y_unc, r = spectrum.T
                mask = (x >= x1) & (x <= x2)
                y = y[mask]
                y_unc = y_unc[mask]
                zz += [((y - y_new) / y_unc)**2]
            num_values = np.array([len(z[np.isfinite(z)])
                                   for z in np.array(zz).T])
            chi2 = np.nansum(zz, axis=0)
            chi2r = np.ones(len(chi2))
            mask = num_values > 1
            chi2r[mask] = chi2[mask] / (num_values[mask] - 1)
            size = min(9, round(0.7*len(chi2r)//2))
            chi2r = rv.rolling_function(np.mean, chi2r, size)
            chi2r[num_values==1] = 1.
            y_new_stdc = y_new_unc * np.sqrt(chi2r)
            y_new_unc = np.maximum(y_new_unc, y_new_stdc)
            y_new_unc[np.isinf(y_new_unc)] = np.nan
            x_new, inds = np.unique(x_new, return_index=True)
            y_new = y_new[inds]
            y_new_unc = y_new_unc[inds]
            r_new = r_new[inds]
            spectrum_new_i = np.array([x_new, y_new, y_new_unc, r_new]).T
            spectrum_new = np.append(spectrum_new, spectrum_new_i, axis=0)
if resample_whole_data:
    # Resampling.
    x_new = np.arange(*new_xrange)
    if resampling_variable != spectral_variable:
        x_new = 1e4/x_new[::-1]
    dx_new = np.append(np.diff(x_new), np.nan)
    spectra_res = {}
    for (name, spectrum) in spectra_new.items():
        x, y, y_unc, r = spectrum.T
        y_new, y_new_unc = resample_spectrum(x_new, x, y, y_unc,
                                             supersample_linearly=True)
        r_new = spectral_factor * dx_new
        r_old = np.interp(x_new, x, r, left=np.nan, right=np.nan)
        r_new = np.maximum(r_new, r_old)
        spectrum = np.array([x_new, y_new, y_new_unc, r_new]).T
        spectra_res[name] = spectrum
    # Merging and averaging of the spectra.
    yy, yy_unc, rr = [], [], []
    for spectrum in spectra_res.values():
        x, y, y_unc, r = spectrum.T
        yy += [y]
        yy_unc += [y_unc]
        rr += [r]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        y_new = np.nanmean(yy, axis=0)
        r_new = np.nanmax(rr, axis=0)
    with np.errstate(divide='ignore'):
        y_new_unc = 1 / (np.nansum(1/np.array(yy_unc)**2, axis=0))**0.5
    zz = []
    for spectrum in spectra_new.values():
        x, y, y_unc, _ = spectrum.T
        y = np.interp(x_new, x, y, left=np.nan, right=np.nan)
        y_unc = np.interp(x_new, x, y_unc, left=np.nan, right=np.nan)
        zz += [((y - y_new) / y_unc)**2]
    num_values = np.array([len(z[np.isfinite(z)])
                           for z in np.array(zz).T])
    chi2 = np.nansum(zz, axis=0)
    chi2r = np.ones(len(chi2))
    mask = num_values > 1
    chi2r[mask] = chi2[mask] / (num_values[mask] - 1)
    size = min(9, round(0.7*len(chi2r)//2))
    chi2r = rv.rolling_function(np.mean, chi2r, size)
    chi2r[num_values==1] = 1.
    y_new_stdc = y_new_unc * np.sqrt(chi2r)
    y_new_unc = np.maximum(y_new_unc, y_new_stdc)
    y_new_unc[np.isinf(y_new_unc)] = np.nan
    spectrum_new = np.array([x_new, y_new, y_new_unc, r_new]).T
x_new, y_new, y_new_unc, r_new = spectrum_new.T
x_new, inds = np.unique(x_new, return_index=True)
spectrum_new = spectrum_new[inds,:]
if spectral_variable != 'wavenumber':
    x_new = 1e4/x_new

#%% Plots.
 
plt.close('all')
plt.figure(1, figsize=figsize)

num_rows = 1
height_ratios = [3.]
if show_signal_to_noise:
    num_rows += 1
    height_ratios += [1.]
if show_spectral_spacing:
    num_rows += 1
    height_ratios += [1.] 
if show_resolving_power:
    num_rows += 1
    height_ratios += [1.] 

gs = plt.GridSpec(num_rows, 1, height_ratios=height_ratios, hspace=0)

ax = plt.gcf().add_subplot(gs[0])

for (name, spectrum) in spectra_orig.items():
    x, y, y_unc, _ = spectrum.T
    plt.errorbar(x, y, y_unc, color=colors[name], ecolor='black', label=name,
                 lw=3., alpha=0.8, drawstyle='steps-mid')
_, y, y_unc, _ = spectrum_new.T
x = x_new
plt.errorbar(x, y, y_unc, color='black', ecolor='gray', lw=1.,
             zorder=3., drawstyle='steps-mid', label='merged')
plt.axhline(y=0, color='black', ls='-', lw=0.8)
if use_logscale:
    ymin = np.nanmin(np.abs(y[y>0]))
    plt.ylim(bottom=ymin/2)
    plt.yscale('log')
plt.ylabel(f'spectral flux density ({flux_units})')
plt.title(config_file.replace('.yaml', ''), pad=10, fontweight='bold')
plt.legend()
plt.margins(x=0.02)
# plt.ylim(bottom=0.)
ax.invert_xaxis()
xlims = plt.xlim()
wavelength_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100,
                    200]  # μm
wavelength_ticklabels = [1, 2, 3, 4, 5, 6, 7, 8, '', 10, '', 12 , '', '', 15,
                         '', '', '', '', 20, '', 30, 40, '', 60, '', '', '', '',
                         200]
ax = plt.gca()
if spectral_variable == 'wavelength':
    ax.set_xlabel('wavenumber (cm$^{-1}$)', labelpad=10., fontsize=9.)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax2 = ax.secondary_xaxis('bottom', functions=(axis_conversion, axis_conversion))
    ax2.set_xticks(wavelength_ticks, wavelength_ticklabels)
    ax2.set_xlabel('wavelength (μm)', fontsize=9.)
    if show_signal_to_noise or show_spectral_spacing or show_resolving_power:
        ax2.tick_params(axis='x', which='both', bottom=False, top=False,
                        labelbottom=False)
else:
    ax.set_xlabel('wavenumber (cm$^{-1}$)')
    ax2 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
    ax2.set_xticks(wavelength_ticks, wavelength_ticklabels)
    ax2.set_xlabel('wavelength (μm)', labelpad=10, fontsize=9)
    if show_signal_to_noise or show_spectral_spacing or show_resolving_power:
        ax.tick_params(axis='x', which='both', bottom=False, top=False,
                       labelbottom=False)
plt.axvline(x=0., color='black', ls='--', lw=0.8)
ax.tick_params(axis='y', which='both', right=True)
interp = interp1d(x[::-1], 1e4/x[::-1])
def custom_format_coord(x, y):
    x = interp(x)
    text = f'(x, y) = ({x:.3f}, {y:.6f})'
    return text
ax.format_coord = custom_format_coord
formatter = plt.FuncFormatter(lambda x, _: str(int(x)) if x.is_integer() else str(x))

igs = 0  # last gridspec index

if show_signal_to_noise:

    igs += 1
    ax = plt.gcf().add_subplot(gs[igs], sharex=ax)
    ax_1 = ax
    
    for (name, spectrum) in spectra_orig.items():
        x, y, y_unc, _ = spectrum.T
        sn = y / y_unc
        sn_ = sn * np.sqrt(spectral_factor)
        plt.plot(x, sn, color=colors[name], lw=2.5, drawstyle='steps-mid')
        plt.plot(x, sn_, '--', color=colors[name], lw=1., alpha=0.7,
                 drawstyle='steps-mid')
    _, y, y_unc, _ = spectrum_new.T
    x = x_new
    sn = y / y_unc
    sn_ = sn * np.sqrt(spectral_factor)
    plt.plot(x, sn, color='black', lw=1., drawstyle='steps-mid')
    plt.plot(x, sn_, '--', color='black', lw=0.5, alpha=0.7, drawstyle='steps-mid')
    plt.plot([], [], '--', color='gray', lw=1., label='per resolution element')
    plt.yscale('log')
    plt.ylabel('signal-to-noise', fontsize=8)
    if spectral_variable == 'wavelength':
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax2 = ax.secondary_xaxis('bottom', functions=(axis_conversion, axis_conversion))
        ax2.set_xticks(wavelength_ticks, wavelength_ticklabels)
        ax2.set_xlabel('wavelength (μm)', labelpad=6.)
        ax3 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
        ax3.set_xticks(wavelength_ticks, wavelength_ticklabels)
        ax3.tick_params(axis='x', which='both', direction='in', labeltop=False)
    else:
        plt.xlabel('wavenumber (cm$^{-1}$)')
        ax2 = ax.secondary_xaxis('top')
        ax2.tick_params(axis='x', which='both', direction='in', labeltop=False)
    ax.yaxis.set_major_locator(plt.LogLocator(base=10., numticks=100))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10., subs=np.arange(2, 10)*0.1,
                                              numticks=100))
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    for yi in (1., 10., 100.):
        plt.axhline(y=yi, color='gray', ls='--', lw=0.4)
    # plt.axhline(y=1., color='black', ls='--', lw=0.8)
    plt.axvline(x=0., color='black', ls='--', lw=0.8)
    ax.tick_params(axis='y', which='both', right=True)
    plt.legend(loc='lower right', fontsize='small')
    ax.format_coord = custom_format_coord

if show_spectral_spacing:

    igs += 1
    ax = plt.gcf().add_subplot(gs[igs], sharex=ax)
    ax_2 = ax
    
    for (name, spectrum) in spectra_orig.items():
        x, _, _, r = spectrum.T
        dx = r / spectral_factor
        plt.plot(x, dx, color=colors[name], lw=2.5, drawstyle='steps-mid')
        plt.plot(x, r, '--', color=colors[name], lw=2., alpha=0.7,
                 drawstyle='steps-mid')
    _, y, _, r = spectrum_new.T
    x = x_new
    dx = np.diff(x) if spectral_variable == 'wavenumber' else np.diff(1e4/x)
    dx = np.append(dx, np.nan)
    dx[np.isnan(y)] = np.nan
    plt.plot(x, dx, color='black', lw=1., drawstyle='steps-mid')
    plt.plot(x, r, '--', color='black', alpha=0.7, drawstyle='steps-mid')
    plt.plot([], [], '--', color='gray', lw=1., label='resolution')
    plt.yscale('log')
    units = 'μm' if spectral_variable == 'wavelength' else 'cm$^{-1}$'
    plt.ylabel(f'{spectral_variable}' '\n' f'spacing ({units})', fontsize=8)
    if spectral_variable == 'wavelength':
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax2 = ax.secondary_xaxis('bottom', functions=(axis_conversion, axis_conversion))
        ax2.set_xticks(wavelength_ticks, wavelength_ticklabels)
        ax2.set_xlabel('wavelength (μm)', labelpad=6.)
        ax3 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
        ax3.set_xticks(wavelength_ticks, wavelength_ticklabels)
        ax3.tick_params(axis='x', which='both', direction='in', labeltop=False)
        if show_resolving_power:
            ax2.tick_params(axis='x', which='both', bottom=False, top=False,
                            labelbottom=False)
    else:
        plt.xlabel('wavenumber (cm$^{-1}$)')
        ax2 = ax.secondary_xaxis('top')
        ax2.tick_params(axis='x', which='both', direction='in', labeltop=False)
        ax_ = ax
    ax.yaxis.set_major_locator(plt.LogLocator(base=10., numticks=100))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10., subs=np.arange(2, 10)*0.1,
                                              numticks=100))
    ax.yaxis.set_major_formatter(formatter)
    plt.axvline(x=0., color='black', ls='--', lw=0.8)
    ax.tick_params(axis='y', which='both', right=True)
    ylines = [1., 10.] if spectral_variable == 'wavenumber' else [0.001, 0.01, 0.1]
    for yi in ylines:
        plt.axhline(y=yi, color='gray', ls='--', lw=0.4)
    plt.legend(loc='lower right', fontsize='small')
    ax.format_coord = custom_format_coord
    
if show_resolving_power:

    igs += 1
    ax = plt.gcf().add_subplot(gs[igs], sharex=ax)
    ax_3 = ax
    
    for (name, spectrum) in spectra_orig.items():
        x, _, _, r = spectrum.T
        rp = x / r if spectral_variable == 'wavenumber' else 1e4/x / r
        plt.plot(x, rp, color=colors[name], lw=2.5, drawstyle='steps-mid')
    _, y, _, r = spectrum_new.T
    x = x_new
    rp = x / r if spectral_variable == 'wavenumber' else 1e4/x / r
    plt.plot(x, rp, color='black', lw=1., drawstyle='steps-mid')
    plt.yscale('log')
    units = 'μm' if spectral_variable == 'wavelength' else 'cm$^{-1}$'
    plt.ylabel('resolving' '\n' 'power', fontsize=8)
    if spectral_variable == 'wavelength':
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax2 = ax.secondary_xaxis('bottom', functions=(axis_conversion, axis_conversion))
        ax2.set_xticks(wavelength_ticks, wavelength_ticklabels)
        ax2.set_xlabel('wavelength (μm)', labelpad=6.)
        ax3 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
        ax3.set_xticks(wavelength_ticks, wavelength_ticklabels)
        ax3.tick_params(axis='x', which='both', direction='in', labeltop=False)
    else:
        plt.xlabel('wavenumber (cm$^{-1}$)')
        ax2 = ax.secondary_xaxis('top')
        ax2.tick_params(axis='x', which='both', direction='in', labeltop=False)
        ax_ = ax
    ax.yaxis.set_major_locator(plt.LogLocator(base=10., numticks=100))
    ax.yaxis.set_minor_locator(plt.LogLocator(base=10., subs=np.arange(2, 10)*0.1,
                                              numticks=100))
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    plt.axvline(x=0., color='black', ls='--', lw=0.8)
    ax.tick_params(axis='y', which='both', right=True)
    for yi in [100., 1000.]:
        plt.axhline(y=yi, color='gray', ls='--', lw=0.4)
    ax.format_coord = custom_format_coord

plt.xlim(xlims)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    plt.tight_layout()

if show_signal_to_noise:
    for tick in ax_1.yaxis.get_major_ticks():
        if float(tick.get_loc()) not in [1., 10., 100.]:
            tick.label1.set_visible(False)
if show_spectral_spacing and spectral_variable == 'wavenumber':
    for tick in ax_2.yaxis.get_major_ticks():
        if float(tick.get_loc()) not in [1., 10., 100.]:
            tick.label1.set_visible(False)
if show_resolving_power:
    for tick in ax_3.yaxis.get_major_ticks():
        if float(tick.get_loc()) not in [100., 1000.]:
            tick.label1.set_visible(False)

#%% Saving of files.

folder += sep.join(output_file.split(sep)[:-1])
output_file = output_file.split(sep)[-1]

# path = os.path.join(folder, 'spectra.png')
# plt.savefig(path)
# print(f'Saved plot in {path}.')

if not os.path.exists(folder):
    os.makedirs(folder)
path = os.path.join(folder, output_file)
spectral_units = 'μm' if spectral_variable == 'wavelength' else '/cm'
n = 6 if spectral_variable == 'wavelength' else 3
np.savetxt(path, spectrum_new,
           header=f'{spectral_variable}_({spectral_units})'
           f' flux_({flux_units}) flux_unc_({flux_units})',
           fmt=[f'%.{n}f','%.9f', '%.9f', f'%.{n}f'])
print(f'Saved merged spectrum in {path}.')
