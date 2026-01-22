#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.0
------------------------------------------
Interactive Toolkit

Andrés Megías.
"""

# AICE parameters.
aice_labels = ['temp. (K)', 'H2O', 'CO', 'CO2', 'CH3OH', 'NH3', 'CH4']
weights_path = '/Users/andres/Proyectos/AICE/neural-networks/training/models/aice-weights.npy'
aice_xrange_params = [980, 4001., 1.]  # /cm
aice_resolution = 2.  # /cm
# Matplotlib backend.
backend = 'qtagg'
# Graphical options.
colors = {'edited': 'cornflowerblue', 'baseline': 'darkorange',
          'selected-points': 'crimson', 'lines': 'tab:red'}
colormaps = {'original': {'name': 'brg', 'offset': 0.0, 'scale': 0.5},
             'edited': {'name': 'viridis', 'offset': 0.4, 'scale': 0.6}}
ds = drawstyle = 'steps-mid'
dlw = default_linewidth = 2.0
rel_margin_x = 0.015
rel_margin_y = 0.040
# Default reduction options.
baseline_smooth_size = 21
interp_smooth_size = 9
smooth_size = 5
# Number of actions stored in cache for undo/redo.
max_actions_stored = 80
# Predefined spectral windows.
species_windows = {
    'H2O': [[3780, 2805], [2570, 2010], [1850, 1200], [1030, 480]],
    'CO': [[2165, 2120]],
    'CO2': [[3723, 3695], [3612, 3588], [2396, 2316], [687, 636]],
    'CH3OH': [[3560, 2712], [1178, 982]],
    'CH4': [[3103, 2968], [1334, 1274]],
    'NH3': [[3638, 2928], [1223, 976]],
    'H2CO': [[3040, 2780], [1780, 1690], [1530, 1470], [1280, 1150]],
    'C2H5NH2': [[3450, 3040], [3010, 2750], [1670, 1220], [1180, 800]],
    '13CO': [[2110, 2075]],
    }
# Predefined macros.
predefined_macros = {
    'M1':
        """
        - smooth (median) :
            smoothing factor : 15
        - modify windows :
            windows : 'auto'
        - estimate baseline :
            smoothing factor : 45
        - reduce
        - modify windows :
            windows :
            - (2992.03, 2942.30)
            - (1280.70, 1237.19)
            - (899.49, 686.09)
        - remove/interpolate :
            smoothing factor : 1
        - modify windows :
            windows :
            - (1235.12, 891.20)
        - smooth :
            smoothing factor : 15
        """,
    'M2':
        """
        - smooth (median) :
            smoothing factor : 75
        - modify windows :
            windows : 'auto'
        - estimate baseline :
            smoothing factor : 45
        - reduce
        - modify windows :
            windows :
            - (2980.00, 2945.30)
            - (1300.70, 1200.19)
            - (899.49, 600.09)
        - remove/interpolate :
            smoothing factor: 1
        - modify windows :
            windows :
            - (1235.12, 891.20)
        - smooth :
            smoothing factor : 15
        - modify windows :
            windows :
            - (6000.00, 3500.00)
        - smooth :
            smoothing factor: 45
        """
    }

# Libraries.
import os
import sys
import copy
import time
import pathlib
import warnings
import platform
import yaml
import numpy as np
import pandas as pd
import richvalues as rv
import matplotlib.pyplot as plt
import scipy.optimize
from types import SimpleNamespace
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.backend_bases import MouseEvent, KeyEvent
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy.stats import median_abs_deviation
plt.matplotlib.use(backend)
if backend == 'qtagg':
    from PyQt5.QtWidgets import QInputDialog

# General functions.

def fit_baseline(x, y, smooth_size=1, windows=None, interpolation='spline'):
    """
    Fit the baseline of the input data using the specified windows.

    Parameters
    ----------
    x, y : arrays
        Data to fit the baseline.
    smooth_size : int, optional
        Size of the filter applied for the fitting of the baseline.
        By default it is 1 (no smoothing).
    windows : array, optional
        Windows that specify regions of the data to use for the fit.
        By default, it is all the spectral range.
    interpolation : str, optional
        Type of interpolation.
        Possible values are 'spline' (default) or 'pchip'.

    Returns
    -------
    yb : array
        Resulting baseline.
    """
    if interpolation not in ('spline', 'pchip'):
        raise Exception("Wrong interpolation type. Should be 'spline' or 'pchip'.")
    if interpolation == 'pchip':
        y = rv.rolling_function(np.nanmedian, y, smooth_size)
    np_isfinite_y = np.isfinite(y)
    if windows is None:
        mask = np_isfinite_y
        x_ = x[mask]
        y_ = y[mask]
    else:
        mask = np.zeros(len(x), bool)
        for (x1, x2) in windows:
            mask |= (x >= x1) & (x <= x2) & np_isfinite_y
        x_ = x[mask]
        y_ = y[mask]
    if interpolation == 'pchip':
        spl = PchipInterpolator(x_, y_)
    elif interpolation == 'spline':
        y_s = rv.rolling_function(np.median, y_, smooth_size)
        s = np.nansum((y_s-y_)**2)
        k = len(y_)-1 if len(y_) <= 3 else 3
        spl = UnivariateSpline(x_, y_, s=s, k=k)
    yb = spl(x)
    return yb

def create_baseline(x, y, p, interpolation='pchip'):
    """
    Create a baseline from the input points.

    Parameters
    ----------
    x, y : arrays
        Data to fit the baseline.
    p : list / array (2, N)
        Reference points for the baseline.
    smooth_size : int, optional
        Size of the filter applied for the fitting of the baseline.
        Only valid if interpolation = 'pchip'.
    interpolation : str, optional
        Type of interpolation.
        Possible values are 'spline' or 'pchip' (default).

    Returns
    -------
    yb : array
        Resulting baseline.
    """ 
    x_, y_ = np.array(p).T
    x_, inds = np.unique(x_, return_index=True)
    y_ = y_[inds]
    if interpolation == 'pchip':
        spl = PchipInterpolator(x_, y_)
    elif interpolation == 'spline':
        k = len(y_)-1 if len(y_) <= 3 else 3
        spl = UnivariateSpline(x_, y_, s=0., k=k)
    yb = spl(x)
    return yb

def axis_conversion(x):
    """Axis conversion from wavenumber to wavelength and viceversa."""
    with np.errstate(divide='ignore'):
        y = 1e4 / x
    return y

def get_windows_from_points(selected_points):
    """Format the selected points into windows."""
    are_points_even = len(selected_points) % 2 == 0
    windows = selected_points[:] if are_points_even else selected_points[:-1]
    windows = np.array(windows).reshape(-1,2)
    for (i, x1x2) in enumerate(windows):
        x1, x2 = min(x1x2), max(x1x2)
        windows[i,:] = [x1, x2]
    return windows

def get_windows_from_mask(mask, x):
    """Obtain the ranges of the input mask associated with the array x."""
    windows = []
    in_window = False
    for i in range(len(x)-1):
        if not in_window and mask[i] == True:
            in_window = True
            window = [x[i]]
        elif in_window and mask[i] == False:
            in_window = False
            window += [(x[i-1] + x[i])/2]
            windows += [window]
    if in_window:
        window += [x[-1]]
        windows += [window]
    elif not in_window and x[-1] == True:
        window = [(x[-2] + x[-1])/2, x[-1]]
        windows += [window]
    return windows

def get_mask_from_windows(windows, x):
    """Obtain a mask corresponding to the input windows on the array x"""
    mask = np.zeros(len(x), bool)
    for x1x2 in windows:
        x1, x2 = min(x1x2), max(x1x2)
        mask |= (x >= x1) & (x <= x2)
    return mask

def invert_windows(windows, x):
    """Obtain the complementary of the input windows for the array x."""
    mask = get_mask_from_windows(windows, x)
    windows = get_windows_from_mask(~mask, x)
    return windows

def calculate_ylims(spectra, x_lims, perc1=0., perc2=100., rel_margin=0.):
    """Calculate vertical limits for the given spectra."""
    yy = np.array([], float)
    x1, x2 = x_lims
    for spectrum in spectra:
        x = spectrum['x']
        y = spectrum['y']
        mask = (x >= x1) & (x <= x2)
        mask_ = mask & np.isfinite(y)
        yy = np.append(yy, y[mask_])
        if 'y-lines' in spectrum:
            y_lines = spectrum['y-lines']
            mask_ = mask & np.isfinite(y_lines)
            yy = np.append(yy, y_lines[mask_])
    y1 = np.percentile(yy, perc1)
    y2 = np.percentile(yy, perc2)
    if rel_margin > 0.:
        yrange = y2 - y1
        margin = rel_margin * yrange
    else:
        margin = 0.
    y_lims = [y1 - margin, y2 + margin]
    return y_lims 

def calculate_robust_ylims(spectra, x_lims, perc1=0., perc2=100., rel_margin=0.):
    """Compute robust vertical limits for the given spectra."""
    yy = np.array([], float)
    x1, x2 = x_lims
    for spectrum in spectra:
        x = spectrum['x']
        y = spectrum['y']
        mask = (x >= x1) & (x <= x2)
        mask_ = mask & np.isfinite(y)
        yy = np.append(yy, y[mask_])
        if 'y-lines' in spectrum:
            y_lines = spectrum['y-lines']
            mask_ = mask & np.isfinite(y_lines)
            yy = np.append(yy, y_lines[mask_])
    yy = np.unique(yy)
    y_min = np.min(yy)
    y_max = np.max(yy)
    data_yrange = y_max - y_min
    margin = rel_margin * data_yrange
    y_lims1 = calculate_ylims(spectra, x_lims, 0., 100., rel_margin)
    y_lims2 = calculate_ylims(spectra, x_lims, perc1, perc2, rel_margin)
    yrange = y_lims2[1] - y_lims2[0]
    y_lims = [max(y_lims1[0], y_lims2[0]) - 0.1*rel_margin*yrange,
              min(y_lims1[1], y_lims2[1]) + rel_margin*yrange]
    y_lims = [max(y_min - margin, y_lims[0]), min(y_max + margin, y_lims[1])]
    return y_lims

def compute_noise(spectra):
    """Compute the median noise level of input spectra."""
    residuals = []
    for spectrum in spectra:
        y = spectrum['y']
        y_smoothed = rv.rolling_function(np.nanmean, y, size=7)
        mask = np.isfinite(y)
        residuals += [np.abs(y - y_smoothed)[mask]]
    residuals = np.concatenate(tuple(residuals))
    noise = median_abs_deviation(residuals, scale='normal')
    return noise
    
def parse_composition(text, folder):
    """Parse input text as containing species."""
    species_list = []
    if '{' in text:
        if 'phase_' in folder:
            pos = folder.index('phase_')
            num_groups = min(int(folder[pos+6]), text.count('{'))
        else:
            num_groups = text.count('{')
        for i in range(num_groups):
            species = text.split('{')[1].split('}')[0]
            text = '}'.join(text.split('}')[1:])
            for name in species.split('+'):
                species_list += [name]
    elif '-' in text and text.split('-')[0].isnumeric():
        species_list = text.split('-')[1].split('+')
    elif '+' in text:
        species_list = text.split('+')
    elif ',' in text:
        species_list = text.split(',')
    else:
        species_list = [text]
    species_list = [name.split('.')[0] for name in species_list]
    return species_list

def aice_model(wavenumber, absorbance, wavenumber_aice, weights):
    """
    Neural network model of AICE.
    
    Predict the composition and temperature of the input spectrum.
    The composition is given in terms of H2O, CO, CO2, CH3OH, NH3 and CH4.
    
    Parameters
    ----------
    wavenumber : array (float)
        Wavenumber points of the spectrum.
    absorbance : array (float)
        Absorbance points of the spectrum.
    wavenumber_aice : array (float)
        Reference wavenumber points used to train AICE.
    weights : array (float)
        Weights of the neural network ensemble.
    
    Returns
    -------
    prediction_df : dataframe (float)
        Predictions for the temperature and molecular fractions.
    """
    relu = lambda x: np.maximum(0, x)
    def sigmoid(x):
        with np.errstate(all='ignore'):
            y = 1 / (1 + np.exp(-x))
        return y
    def nn_model(x, weights, end_act=relu):
        """Multi-layer perceptron of AICE."""
        w = weights
        w1, b1 = w[0], w[1]
        ga1, be1, m1, s1 = w[2], w[3], w[4], w[5]
        w2, b2 = w[6], w[7]
        ga2, be2, m2, s2 = w[8], w[9], w[10], w[11]
        w3, b3 = w[12], w[13]
        ga3, be3, m3, s3 = w[14], w[15], w[16], w[17]
        w4, b4 = w[18], w[19]  
        e = 1e-3
        a1 = relu(np.dot(w1.T, x) + b1)
        a1 = ga1 * (a1 - m1) / (s1 + e)**0.5 + be1
        a2 = relu(np.dot(w2.T, a1) + b2)
        a2 = ga2 * (a2 - m2) / (s2 + e)**0.5 + be2
        a3 = relu(np.dot(w3.T, a2) + b3)
        a3 = ga3 * (a3 - m3) / (s3 + e)**0.5 + be3
        y = end_act(np.dot(w4.T, a3) + b4)
        return y
    baseline = fit_baseline(wavenumber, absorbance, smooth_size=3)
    mask = np.isnan(absorbance)
    absorbance_ = copy.copy(absorbance)
    absorbance_[mask] = baseline[mask]
    absorbance_aice = resample_spectrum(wavenumber_aice, wavenumber, absorbance_,
                                        supersample_linearly=True)
    absorbance_aice = np.nan_to_num(absorbance_aice, nan=0.)
    size = int(round(aice_resolution / aice_xrange_params[-1]))
    absorbance_aice = rv.rolling_function(np.mean, absorbance_aice, size)
    absorbance_aice /= np.mean(absorbance_aice)
    results = []
    for j in range(weights.shape[0]):
        yj = np.zeros(weights.shape[1])
        for i in range(len(yj)):
            end_act = relu if i == 0 else sigmoid
            yj[i] = nn_model(absorbance_aice, weights[j,i], end_act)[0]
        results += [yj]
    results = np.array(results)
    stdevs = np.std(results, axis=0)
    predictions = np.mean(results, axis=0)
    predictions = rv.RichArray(predictions, stdevs)
    labels = aice_labels + ['all molecules']
    sum_predictions = predictions[1:].sum()
    predictions = list(predictions) + [sum_predictions]
    predictions_df = rv.rich_dataframe({'AICE prediction': predictions},
                                       index=labels)
    return predictions_df

def spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None,
             fill=None, supersample_linearly=False, verbose=True):

    """
    Function for r–esampling spectra (and optionally associated
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
    y_unc_new = np.array([y_unc[0]], float) if y_unc is not None else None
    x_prev = x[0]
    dx_prev = np.median(np.diff(x[:3]))
    for i in range(1, len(x)):
        dx_i = x[i] - x_prev
        if dx_i > threshold * dx_prev:
            num_points = round(dx_i / dx_prev)
            dx_new = dx_i / num_points
            x_ = np.arange(x_prev+dx_prev, x[i]-dx_prev, dx_new)
            y_ = np.full(len(x_), np.nan)
            x_new = np.append(x_new, x_)
            y_new = np.append(y_new, y_)
            if y_unc is not None:
                y_unc_new = np.append(y_unc_new, y_)
        x_new = np.append(x_new, x[i])
        y_new = np.append(y_new, y[i])
        if y_unc is not None:
            y_unc_new = np.append(y_unc_new, y_unc[i])
        x_prev = x_new[-1]
        dx_prev = dx_i
    return x_new, y_new, y_unc_new

def custom_input(prompt, window_title=''):
    """Custom input call that uses Qt if using qtagg backend."""
    if backend == 'qtagg':
        prompt = prompt.replace('- ', '')
        text, _ = QInputDialog.getText(None, window_title, prompt)
    else:
        text = input(prompt)
    return text

# Removing default keymaps for interactive plot.
keymaps = ('back', 'copy', 'forward', 'fullscreen', 'grid', 'grid_minor',
           'help', 'home', 'pan', 'quit', 'quit_all', 'save', 'xscale',
           'yscale', 'zoom')
for keymap in keymaps:            
    plt.rcParams.update({'keymap.' + keymap: []})
    
# Folder separator.
sep = '\\' if platform.system() == 'Windows' else '/'

#%% Functions used in interactive mode.

def plot_data(spectra, spectra_old, active_indices, idx,
              using_manual_baseline_mode):
    """Plot the input spectra."""
    global x_min, x_max, x_lims, y_lims
    global use_logscale, use_microns, invert_yaxis, use_optical_depth
    global spectra_colors, spectra_colors_old
    plt.clf()
    factor = np.log(10) if use_optical_depth else 1.
    for (i,spectrum_old) in enumerate(spectra_old):
        x = spectrum_old['x'] if not use_microns else 1e4 / spectrum_old['x']
        y = spectrum_old['y'] * factor
        plt.plot(x, y, color='dimgray', drawstyle=ds, lw=dlw, alpha=0.4,
                 zorder=2.4)
    x = spectra_old[idx]['x'] if not use_microns else 1e4 / spectra_old[idx]['x']
    y = spectra_old[idx]['y'] * factor
    y_unc = copy.copy(spectra_old[idx]['y-unc'])
    if y_unc is not None:
        y_unc *= factor
    plt.errorbar(x, y, y_unc, color='black', ecolor=[0.7]*3, drawstyle=ds,
                 lw=dlw, label='original spectrum', zorder=2.6) 
    spectrum = spectra[idx]
    if 'y-base' in spectrum:
        x = spectrum['x'] if not use_microns else 1e4 / spectrum['x']
        y = spectrum['y-base'] * factor
        plt.plot(x, y, linestyle='--', lw=0.9*dlw, zorder=3.0,
                 color=colors['baseline'], drawstyle=ds, label='computed baseline')
    if 'y-lines' in spectrum:
        x = spectrum['x'] if not use_microns else 1e4 / spectrum['x']
        y = spectrum['y-lines'] * factor
        plt.plot(x, y, linestyle='-', lw=0.9*dlw, zorder=2.9, drawstyle=ds,
                 color=colors['lines'], label='fitted lines')
    if spectrum['edited']:
        x = spectrum['x'] if not use_microns else 1e4 / spectrum['x']
        y = spectrum['y'] * factor
        y_unc = copy.copy(spectrum['y-unc'])
        if y_unc is not None:
            y_unc *= factor
        plt.errorbar(x, y, y_unc, color=colors['edited'], ecolor='gray',
                     drawstyle=ds, lw=0.8*dlw, zorder=2.8,
                     label='edited spectrum')
    for i in all_indices:
        if i == idx or i not in active_indices:
            continue
        if i in active_indices:
            color = spectra_colors[i] if spectrum['edited'] else spectra_colors_old[i]
        else:
            color = 'gray'
        spectrum = spectra[i]
        x = spectrum['x'] if not use_microns else 1e4 / spectrum['x']
        y = spectrum['y'] * factor
        plt.plot(x, y, color=color, drawstyle=ds, lw=0.8*dlw,
                 alpha=0.4, zorder=2.5)
        if 'y-base' in spectrum and i in all_indices:
            x = spectrum['x'] if not use_microns else 1e4 / spectrum['x']
            y = spectrum['y-base'] * factor
            plt.plot(x, y, '--', lw=0.8*dlw, color='darkorange',
                     alpha=0.2, zorder=2.5)
            
    ylabel = variable_y
    y_lims_ = list(np.array(y_lims) * factor)
    if ylabel == 'absorbance' and use_optical_depth:
        ylabel = 'optical depth'
    elif ylabel.startswith('absorption coefficient'):
        ylabel = ylabel.replace('(cm2)', '(cm$^2$)')
    plt.xlim(x_lims)
    plt.ylim(y_lims_)
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.axhline(y=0., color='black', lw=0.5)
    if variable_y == 'transmittance':
        plt.axhline(y=1., color='black', lw=0.5)
    plt.axvline(x=0., color='black', lw=0.5)
    xlabel = 'wavenumber (cm$^{-1}$)' if not use_microns else 'wavelength (μm)'
    plt.xlabel(xlabel, labelpad=6.)
    plt.ylabel(ylabel, labelpad=12.)
    plt.margins(x=0.01)
    if not using_manual_baseline_mode:
        plt.axvspan(0., 0., 0., 0., edgecolor='lightgray', facecolor='white',
                    alpha=1., label='windows')
    else:
        plt.plot([], '.', color=colors['selected-points'],
                 label='reference points')
    ax = plt.gca()
    if use_logscale:
        yy = np.concatenate([spectrum['y'] for spectrum in spectra])
        linthresh = (1e3*np.min(np.abs(yy[yy>0])) if 'abs' not in variable_y
                     else 5*noise_level)
        linthresh = min(0.3*np.max(yy), linthresh)
        if 'abs' in variable_y:
            linthresh = max(0.04*np.max(yy), linthresh)
        linthresh = 10**(np.floor(np.log(linthresh)))
        plt.yscale('symlog', linthresh=linthresh)
        y1, y2 = plt.ylim()
        log_locator = LogLocator(base=10., subs='auto')
        y1_ = 10**(round(np.log(abs(y1))))
        y2_ = 10**(round(np.log(y2)))
        minor_ticks_pos = log_locator.tick_values(linthresh, 1e3*y2_)
        minor_ticks_neg = -log_locator.tick_values(linthresh, 1e4*y1_)
        minor_ticks = np.append(minor_ticks_pos, minor_ticks_neg)
        ax.set_yticks(minor_ticks, minor=True)
        log_locator = LogLocator(base=10.)
        major_ticks_pos = log_locator.tick_values(linthresh, 1e3*y2_)
        major_ticks_neg = -log_locator.tick_values(linthresh, 1e4*y1_)
        major_ticks = np.concatenate([major_ticks_pos, major_ticks_neg, [0.]])
        major_ticks = np.array(sorted(major_ticks))
        ax.set_yticks(major_ticks, minor=False)
        for (tick, label) in zip(ax.get_yticks(), ax.get_yticklabels()):
            if abs(tick) == linthresh/10:
                label.set_visible(False)
        plt.ylim(y1, y2)
    else:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    loc = ('upper left' if 'abs' in variable_y and not invert_yaxis
           or variable_y == 'transmittance' and invert_yaxis else 'lower left')
    plt.legend(loc=loc)
    plt.title(spectra_names[idx], fontweight='bold', pad=12.)
    ax = plt.gca()
    if not use_microns:
        ax.invert_xaxis()
    ax2 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
    if not use_microns:
        xticks2 = [1.5, 1.7, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8,
                   10, 12, 15, 20, 30, 50, 300] # μm
        xlabel2 = 'wavelength (μm)'
    else:
        xticks2 = [20000, 10000, 8000, 6000, 5000,
                   4000, 3000, 2500, 2000, 1500, 1200,
                   1000, 800, 700, 600, 500,
                   400, 350, 300, 250, 200, 150, 120,
                   100, 90, 80, 70, 60, 50, 45, 40, 35, 30]
        xlabel2 = 'wavenumber (cm$^{-1}$)'
    ax2.set_xticks(xticks2, xticks2)
    ax2.set_xlabel(xlabel2, labelpad=6., fontsize=9.)
    plt.tight_layout()
    loc = ((0.98, 0.96) if 'abs' in variable_y and not invert_yaxis
           or variable_y == 'transmittance' and invert_yaxis else (0.98, 0.125))
    plt.text(*loc, '      AICE Interactive Toolkit' '\n ' ,
             ha='right', va='top', fontweight='bold', transform=ax.transAxes,
             bbox=dict(edgecolor=[0.8]*3, facecolor='white', alpha=0.7))
    plt.text(*loc, ' \n' 'check terminal for instructions',
             ha='right', va='top', transform=ax.transAxes)
    plt.gca().set_facecolor([0.9]*3)

def plot_windows(selected_points, use_microns):
    """Plot the current selected windows."""
    for x in selected_points:
        if use_microns:
            x = 1e4/x
        plt.axvline(x, color='lightgray', alpha=1.)
    windows = get_windows_from_points(selected_points)
    for (x1, x2) in windows:
        if use_microns:
            x1, x2 = 1e4/x2, 1e4/x1
        plt.axvspan(x1, x2, transform=plt.gca().transAxes,
                    color='white', alpha=1.)

def plot_baseline_points(selected_points):
    """Plot the current selected baseline points."""
    if selected_points not in (0, None) and len(selected_points) > 0:
        x, y = np.array(selected_points).T
        x = 1e4 / x if use_microns else x
        plt.plot(x, y, '.', color=colors['selected-points'], zorder=3.)

def estimate_baseline(spectra, active_indices, selected_points, smooth_size,
                      using_manual_baseline_mode, interpolation='spline'):
    """Estimate a baseline for the input spectra and the given parameters."""
    global x_min, x_max
    windows = get_windows_from_points(selected_points)
    for i in active_indices:
        spectrum = spectra[i] 
        x = spectrum['x']
        y = spectrum['y']
        if not using_manual_baseline_mode:
            y_base = fit_baseline(x, y, smooth_size, windows, interpolation)
        else:
            y_base = create_baseline(x, y, selected_points, interpolation)
        if 'y-base' in spectrum:
            y_base_prev = copy.copy(spectrum['y-base'])
            if not using_manual_baseline_mode:
                x1, x2 = np.min(windows), np.max(windows)
            else:
                x1, x2 = np.min(selected_points), np.max(selected_points)
            mask = (x < x1) | (x > x2)
            y_base[mask] = y_base_prev[mask]
        spectra[i]['y-base'] = y_base

def reduce_spectra(spectra, active_indices, mode, use_lines=False):
    """Reduce the spectra subtracting the pre-computed existing baselines."""
    for i in active_indices:
        spectrum = spectra[i]
        if 'y-base' in spectrum:
            y = spectrum['y']
            y_unc = spectrum['y-unc']
            y_base = spectrum['y-base']
            if mode == 'subtract':
                y_red = y - y_base 
                y_red_unc = y_unc if y_unc is not None else None
            elif mode == 'divide':
                y_red = y / y_base
                y_red_unc = np.abs(y_unc / y_base) if y_unc is not None else None
            spectra[i]['edited'] = True
            spectra[i]['y'] = y_red
            spectra[i]['y-unc'] = y_red_unc 
            
def subtract_lines(spectra, active_indices, windows):
    """Remove the previously fitted lines in current windows."""
    for i in active_indices:
        spectrum = spectra[i]
        if 'y-lines' in spectrum:
            y = spectrum['y']
            y_unc = spectrum['y-unc']
            y_lines = np.nan_to_num(spectrum['y-lines'], nan=0.)
            y_red = copy.copy(y)
            y_red_unc = copy.copy(y_unc) if y_unc is not None else None
            for (x1, x2) in windows:
                mask = (x >= x1) & (x <= x2)
                y_red[mask] = y[mask] - y_lines[mask]
                y_lines[mask] = np.nan
            y_lines[y_lines==0.] = np.nan
            spectra[i]['edited'] = True
            spectra[i]['y'] = y_red
            spectra[i]['y-unc'] = y_red_unc 
            spectra[i]['y-lines'] = y_lines

def remove_regions(spectra, active_indices, selected_points):
    """Remove the selected regions of the data."""
    windows = get_windows_from_points(selected_points)
    for i in active_indices: 
        spectrum = spectra[i]
        x = spectrum['x']
        y = spectrum['y']
        y_unc = spectrum['y-unc']
        if 'y-base' in spectrum:
            y_base = spectrum['y-base']
        if 'y-lines' in spectrum:
            y_lines = spectrum['y-lines']
        for (x1,x2) in windows:
            is_inferior_edge = x1 <= np.min(x)
            is_superior_edge = x2 >= np.max(x)
            if is_inferior_edge or is_superior_edge:
                mask = (x >= x2 if is_inferior_edge else x <= x1)
                x = x[mask]
                y = y[mask]
                if y_unc is not None:
                    y_unc = y_unc[mask]
                if 'y-base' in spectrum:
                    y_base = y_base[mask]
                if 'y-lines' in spectrum:
                    y_lines = y_lines[mask]
            else:
                mask = (x >= x1) & (x <= x2)
                y[mask] = np.nan
                if y_unc is not None:
                    y_unc[mask] = np.nan
                if 'y-base' in spectrum:
                    y_base[mask] = np.nan
                if 'y-lines' in spectrum:
                    y_lines[mask] = np.nan
        spectra[i]['edited'] = True
        spectra[i]['x'] = x
        spectra[i]['y'] = y
        spectra[i]['y-unc'] = y_unc
        if 'y-base' in spectrum:
            spectra[i]['y-base'] = y_base
        if 'y-lines' in spectrum:
            spectra[i]['y-lines'] = y_lines
                
def interpolate_regions(spectra, active_indices, selected_points, smooth_size,
                        interpolation='spline', exclude_selected_regions=True):
    """Interpolate the selected regions of the data."""
    windows = get_windows_from_points(selected_points)
    for i in active_indices: 
        spectrum = spectra[i]
        x = spectrum['x']
        y = spectrum['y']
        y_unc = spectrum['y-unc']
        windows_ = (invert_windows(windows, x) if exclude_selected_regions
                    else None)
        y_interp = fit_baseline(x, y, smooth_size, windows_, interpolation)
        for (x1, x2) in windows:
            mask = (x >= x1) & (x <= x2)
            y[mask] = y_interp[mask]
            if y_unc is not None:
                y_unc[mask] = 0.
        spectra[i]['edited'] = True
        spectra[i]['y'] = y
        spectra[i]['y-unc'] = y_unc
        
def gaussian(x, mean=0, std=1, height=1, ssf=1):
    """
    Apply a Gaussian function to the input array (x), with given height, mean,
    standard deviation (std) and supersampling factor (ssf).
    """
    if ssf == 1:
        y = height * np.exp(-((x-mean)/std)**2/2)
    else:
        dx = np.median(np.diff(x))
        dx_ = dx / ssf
        N = round(ssf * len(x))
        x1_ = x[0] - dx/2 + dx_/2
        x2_ = x[-1] + dx/2 - dx_/2 
        x_ = np.linspace(x1_, x2_, N)
        y_ = height * np.exp(-((x_-mean)/std)**2/2)
        y = y_.reshape(-1, ssf).mean(axis=1)
    return y

def fit_gaussian(spectra, active_indices, windows, line_borders, center=None):
    """Fit Gaussians to given spectra in the current windows."""
    fix_center = False if center is None else True
    for j in active_indices:
        if not fix_center:
            center = None
        spectrum = spectra[j]
        x = spectrum['x']
        y = spectrum['y']
        y_lines = (spectrum['y-lines'] if 'y-lines' in spectrum
                   else np.zeros(len(x)))
        y_lines = np.nan_to_num(y_lines, nan=0.)
        mask = np.zeros(len(x), bool)
        for (x1, x2) in windows:
            mask |= (x >= x1) & (x <= x2)
        if line_borders not in (0, None):
            if line_borders[0][0] > line_borders[1][0]:
                line_borders = [line_borders[1], line_borders[0]]
            x1, y1 = line_borders[0]
            x2, y2 = line_borders[1]
            mask &= (x >= x1) & (x <= x2)
        mask &= np.isfinite(y)
        x_ = x[mask]
        y_ = y[mask]
        y_lines_ = y_lines[mask]
        if line_borders not in (0, None):
            x1w = np.min(windows)
            x2w = np.max(windows)
            x1 = max(x1, x1w)
            x2 = min(x2, x2w)
        else:
            x1, x2 = x_[0], x_[-1]
            y1, y2 = (y_[0], y_[-1]) if line_borders != 0 else (0, 0)
        baseline = lambda x: y1 + (y2 - y1) / (x2 - x1) * (x - x1)
        y_b = baseline(x_) + y_lines_
        xrange = x2 - x1
        width = xrange / 4
        height = np.max(y_)
        ssf = round(max(1., 10. - 9/25 * (len(x_)-1.)))
        if center is None:
            center = np.mean(x_)
            guess = [center, width, height]
            function = lambda x,m,s,h: gaussian(x, m, s, h, ssf)
            bounds = [[x1, 0., 0.], [x2, 5*xrange, 40*height]]
        else:
            guess = [width, height]
            m = center
            function = lambda x,s,h: gaussian(x, m, s, h, ssf)
        fit_params = scipy.optimize.curve_fit(function, x_, y_-y_b, p0=guess,
                                              bounds=bounds)[0]
        if not fix_center:
            center, width, height = fit_params
        else:
            width, height = fit_params
        if use_microns:
            width = 1e3 * (1e4 / (center - width/2) - 1e4 / (center + width/2))
            center = 1e4 / center
            nd = 3
            units1 = '(μm)'
            units2 = '(nm)'
        else:
            nd = 1
            units1 = units2 = '(/cm)'
        print(f'Gaussian parameters: center {units1}, {center:.{nd}f}; '
              f'width {units2}, {width:.1f}; height, {height:.2f}.')
        y_lines = (spectrum['y-lines'] if 'y-lines' in spectrum
                   else np.full(len(x), np.nan))
        y_lines = np.nan_to_num(y_lines, nan=0.)
        x1 = min(x1, center - 4*width)
        x2 = max(x2, center + 4*width)
        mask = (x >= x1) & (x <= x2)
        x_ = x[mask]
        y_f = function(x_, *fit_params) + baseline(x_)
        y_lines[mask] += y_f
        y_lines[y_lines == 0.] = np.nan
        spectra[j]['y-lines'] = y_lines

def gaussians(x, ssf, *params, centers=None):
    """
    Apply several Gaussian functions to the input array, where params should
    be a tuple with 3 parameters for each function: height, mean and std.
    """
    y = np.zeros(len(x))
    if centers is None:
        for i in range(0, len(params), 3):
            mean, std, height = params[i:i+3]
            y += gaussian(x, mean, std, height, ssf)
    else:
        for i in range(0, len(params), 2):
            std, height = params[i:i+2]
            mean = centers[i//2]
            y += gaussian(x, mean, std, height, ssf)
    return y

def multigaussian_fit(x, y, num_curves=1, ssf=1, centers=None, max_iters=10,
                      verbose=False, old_results=None):
    """
    Make a fit of multiple Gaussians to the input data.

    Parameters
    ----------
    x, y : array
        Input data to fit.
    num_curves : int, optional
        Starting number of curves to fit. The default is 1.
    ssf : int, optional
        Number of bins in which to supersample the gaussian function.
        The default is 1.
    max_iters : int, optional
        Number of maximum calls of the function. The default is 10.
    centers : list, optional
        List of fixed centers for the Gaussians.
    verbose : bool, optional
        If True, return informative messages of the fit.
    old_results : list, optional
        Internal variable used in the calculations. Do not use.

    Returns
    -------
    popt: list
        List of the parameters of the curves fitted, in groups of 3.
    window: float
        Region of x where the fitted curves lie.
    r2: float
        Coefficient of determination of the fit.
    """
    # First guess of the parameters of the gaussians.
    x1, x2 = x[0], x[-1]
    xrange = x2 - x1
    height = np.mean(y)
    mean = np.mean(x)
    std = xrange / 4
    means = np.random.uniform(x1, x2, num_curves)
    fixed_num_curves = False if centers is None else True
    guess, bounds_inf, bounds_sup, fix_centers = [], [], [], []
    for i in range(num_curves):
        mean = centers[i] if fix_centers else means[i]
        guess += [mean, std, height]
        bound_mean_inf = x1 if not fix_centers else mean - 0.01*std
        bound_mean_sup = x2 if not fix_centers else mean + 0.01*std
        bounds_inf += [bound_mean_inf, 0., 0.]
        bounds_sup += [bound_mean_sup, 5*xrange, 40*height]
    bounds = (bounds_inf, bounds_sup)
    # Fit.
    multigaussian_ = lambda x,*params: gaussians(x, ssf, *params) 
    popt, pcov = scipy.optimize.curve_fit(multigaussian_, x, y, p0=guess,
                                          bounds=bounds)
    # Coefficient of determination.    
    res = y - multigaussian_(x, *popt)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    if verbose:
        print('num_curves, r2, max_iters: ', num_curves, r2, max_iters)
    # Condition for returning to previous fit.
    if old_results and (r2 - old_results[-1]) < 2e-3:
        if verbose:
            print('Returning to previous fit. Finished.')
        return old_results
    # Condition for finishing or adding another curve.
    if not fixed_num_curves:
        if r2 > 0.99 or max_iters == 1:
            if verbose:
                print('Finished.')
            return popt, r2
        else:
            old_results = (copy.copy(popt), copy.copy(r2))
            return multigaussian_fit(x, y, num_curves+1, ssf, centers,
                                     max_iters-1, verbose, old_results)
    else:
        if verbose:
            print('Finished.')
        return popt, r2
        
def fit_gaussians(spectra, active_indices, windows, line_borders,
                  centers=None):
    """Fit Gaussians to given spectra in the current windows."""
    global use_microns
    for j in active_indices:
        spectrum = spectra[j]
        x = spectrum['x']
        y = spectrum['y']
        y_lines = (spectrum['y-lines'] if 'y-lines' in spectrum
                   else np.zeros(len(x)))
        y_lines = np.nan_to_num(y_lines, nan=0.)
        mask = np.zeros(len(x), bool)
        for (x1, x2) in windows:
            mask |= (x >= x1) & (x <= x2)
        if line_borders not in (0, None):
            if line_borders[0][0] > line_borders[1][0]:
                line_borders = [line_borders[1], line_borders[0]]
            x1, y1 = line_borders[0]
            x2, y2 = line_borders[1]
            mask &= (x >= x1) & (x <= x2)
        mask &= np.isfinite(y)
        x_ = x[mask]
        y_ = y[mask]
        y_lines_ = y_lines[mask]
        if line_borders not in (0, None):
            x1w = np.min(windows)
            x2w = np.max(windows)
            x1 = max(x1, x1w)
            x2 = min(x2, x2w)
        else:
            y_b = np.zeros(len(x_))
            x1, x2 = x_[0], x_[-1]
            y1, y2 = (y_[0], y_[-1]) if line_borders != 0. else (0., 0.)
        baseline = lambda x: y1 + (y2 - y1) / (x2 - x1) * (x - x1)
        y_b = baseline(x_) + y_lines_
        ssf = round(max(1., 10. - 9/25 * (len(x_)-1.)))
        function = lambda x,*args: gaussians(x, ssf, *args)
        num_curves = len(centers) if centers is not None else 1
        fit_params = multigaussian_fit(x_, y_-y_b, num_curves, ssf, centers,
                                       verbose=False)[0]
        edges = []
        print('Gaussians parameters:')
        for i in range(0, len(fit_params), 3):
            center, width, height = fit_params[i:i+3]
            edges += [[center - 4*width, center + 4*width]]
            if use_microns:
                width = 1e3 * (1e4 / (center - width/2) - 1e4 / (center + width/2))
                center = 1e4 / center
                nd = 3
                units1 = '(μm)'
                units2 = '(nm)'
            else:
                nd = 1
                units1 = units2 = '(/cm)'
            print(f'- center {units1}, {center:.{nd}f}; '
                  f'width {units2}, {width:.1f}; height, {height:.2f}.')
        y_lines = (spectrum['y-lines'] if 'y-lines' in spectrum
                   else np.full(len(x), np.nan))
        y_lines = np.nan_to_num(y_lines, nan=0.)
        x1 = min(x1, np.min(edges))
        x2 = max(x2, np.max(edges))
        mask = (x >= x1) & (x <= x2)
        x_ = x[mask]
        y_b = baseline(x_)
        y_f = function(x_, *fit_params) + y_b
        y_lines[mask] += y_f
        y_lines[y_lines == 0.] = np.nan
        spectra[j]['y-lines'] = y_lines
        
def remove_fitted_lines(spectra, active_indices, windows):
    """Remove previously fitted lines in current windows."""
    for i in active_indices:
        spectrum = spectra[i]
        if 'y-lines' in spectrum:
            x = spectrum['x']
            y_lines = spectrum['y-lines']
            mask = np.zeros(len(x), bool)
            for (x1, x2) in windows:
                mask |= (x >= x1) & (x <= x2)
            y_lines[mask] = np.nan
            spectra[i]['y-lines'] = y_lines

def smooth_spectra(spectra, active_indices, selected_points, smooth_size,
                 function=np.nanmean):
    """Smooth the data in the selected regions.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    windows = get_windows_from_points(selected_points)
    for i in active_indices:
        spectrum = spectra[i] 
        x = spectrum['x']
        y = spectrum['y']
        y_unc = spectrum['y-unc']
        for (x1, x2) in windows:
            mask1 = (x >= x1 - smooth_size) & (x <= x2 + smooth_size)
            x_ = x[mask1]
            mask2 = (x_ >= x1) & (x_ <= x2)
            mask = (x >= x1) & (x <= x2)
            y[mask] = rv.rolling_function(function, y[mask1], smooth_size)[mask2]
            if y_unc is not None:
                y_unc[mask] = y_unc[mask] / np.sqrt(smooth_size)
        spectra[i]['edited'] = True
        spectra[i]['y'] = y
        spectra[i]['y-unc'] = y_unc
        
def multiply_spectra(spectra, active_indices, selected_points, factor,
                     multiply_lines=False):
    """Multiply the data in the selected regions with the input factor.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    windows = get_windows_from_points(selected_points)
    for i in active_indices:
        spectrum = spectra[i] 
        x = spectrum['x']
        y = spectrum['y'] if not multiply_lines else spectrum['y-lines']
        y_unc = spectrum['y-unc']
        for (x1,x2) in windows:
            mask = (x >= x1) & (x <= x2)
            y[mask] = factor * y[mask]
            if y_unc is not None:
                y_unc[mask] = factor * y_unc[mask] 
        if not multiply_lines:
            spectra[i]['edited'] = True
        yvar = 'y' if not multiply_lines else 'y-lines'
        spectra[i][yvar] = y
        spectra[i]['y-unc'] = y_unc
        
def set_zero(spectra, active_indices, selected_points, only_negatives=False):
    """Set selected region of the data to zero."""
    if selected_points == []:
        if only_negatives:
            global x_min, x_max
            selected_points = [x_max, x_min]
        else:
            print('Error: No points selected.')
            return
    windows = get_windows_from_points(selected_points)
    for i in active_indices:
        spectrum = spectra[i]
        x = spectrum['x']
        y = spectrum['y']
        y_unc = spectrum['y-unc']
        for (x1, x2) in windows:
            mask = (x >= x1) & (x <= x2)
            if only_negatives:
                mask &= y < 0.
            y[mask] = 0.
            if y_unc is not None:
                y_unc[mask] = 0.
        spectra[i]['edited'] = True
        spectra[i]['y'] = y
        spectra[i]['y-unc'] = y_unc
        
def set_one(spectra, active_indices, selected_points, only_gtr1=False):
    """Set selected region of the data to one."""
    if selected_points == []:
        if only_gtr1:
            global x_min, x_max
            selected_points = [x_max, x_min]
        else:
            print('Error: No points selected.')
            return
    windows = get_windows_from_points(selected_points)
    for i in active_indices:
        spectrum = spectra[i]
        x = spectrum['x']
        y = spectrum['y']
        y_unc = spectrum['y-unc']
        for (x1, x2) in windows:
            mask = (x >= x1) & (x <= x2)
            if only_gtr1:
                mask &= y > 1.
            y[mask] = 1.
            if y_unc is not None:
                y_unc[mask] = 0.
        spectra[i]['edited'] = True
        spectra[i]['y'] = y
        spectra[i]['y-unc'] = y_unc

def add_noise(spectra, active_indices, selected_points, noise_level):
    """Add noise to the selected regions.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    windows = get_windows_from_points(selected_points)
    for i in active_indices:
        spectrum = spectra[i] 
        x = spectrum['x']
        y = spectrum['y']
        y_unc = spectrum['y-unc']
        for (x1, x2) in windows:
            mask = (x >= x1) & (x <= x2)
            noise = np.random.normal(0., scale=noise_level, size=sum(mask))
            if y_unc is None:
                y[mask] = y[mask] + noise
            else:
                y_unc[mask] += noise_level
        spectra[i]['edited'] = True
        spectra[i]['y'] = y
        spectra[i]['y-unc'] = y_unc

def add_offset(spectra, active_indices, selected_points, offset):
    """Add offset to selected spectra.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    windows = get_windows_from_points(selected_points)
    for i in active_indices:
        spectrum = spectra[i] 
        x = spectrum['x']
        y = spectrum['y']
        if type(offset) is float:
            for (x1, x2) in windows:
                mask = (x >= x1) & (x <= x2)
                y[mask] = y[mask] + offset
        else:
            offset = spectrum['y-base']
            y += offset
        spectra[i]['edited'] = True
        spectra[i]['y'] = y
        
def do_resampling(spectra, active_indices, selected_points, x_ref, kind='simple'):
    """Resample the spectra to the input wavenumber array.""" 
    global x_min, x_max
    windows = (get_windows_from_points(selected_points)
               if selected_points != [] else [[x_min, x_max]])
    for i in active_indices:
        spectrum = spectra[i] 
        x = spectrum['x']
        y = spectrum['y']
        y_unc = spectrum['y-unc']
        x_new = np.array([])
        y_new = np.array([])
        y_unc_new = np.array([]) if y_unc is not None else None
        for (x1, x2) in invert_windows(windows, x_ref):
            mask = (x > x1) & (x < x2)
            x_i = x[mask]
            y_i = y[mask]
            y_unc_i = y_unc[mask] if y_unc is not None else None
            x_new = np.append(x_new, x_i)
            y_new = np.append(y_new, y_i)
            if y_unc is not None:
                y_unc_new = np.append(y_unc_new, y_unc_i)
        for (x1, x2) in windows:
            mask = (x_ref >= x1) & (x_ref <= x2)
            x_new_i = x_ref[mask]
            mask = (x >= x1) & (x <= x2)
            x_i = x[mask]
            y_i = y[mask]
            y_unc_i = y_unc[mask] if y_unc is not None else None
            if kind == 'simple':
                y_new_i = np.interp(x_new_i, x_i, y_i, left=np.nan, right=np.nan)
                y_unc_new_i = (np.interp(x_new_i, x_i, y_unc_i,
                                         left=np.nan, right=np.nan)
                               if y_unc is not None else None)
            else:
                y_new_i, y_unc_new_i = resample_spectrum(x_new_i, x_i, y_i,
                                                         y_unc_i, verbose=False)
            x_new = np.append(x_new, x_new_i)
            y_new = np.append(y_new, y_new_i)
            if y_unc is not None:
                y_unc_new = np.append(y_unc_new, y_unc_new_i)
        inds = np.argsort(x_new)
        x_new = x_new[inds]
        y_new = y_new[inds]
        if y_unc is not None:
            y_unc_new = y_unc_new[inds]
        spectra[i]['edited'] = True
        spectra[i]['x'] = x_new
        spectra[i]['y'] = y_new
        spectra[i]['y-unc'] = y_unc_new
        
def compute_absorbance(spectra):
    """Convert the current spectra from transmittance to absorbance."""
    with np.errstate(invalid='ignore'):
        for (i, spectrum) in enumerate(spectra):
            y = spectrum['y']
            y_unc = spectrum['y-unc']
            y_new = -np.log10(y)
            y_unc_new = (np.abs(y_unc / y) / np.log(10)
                         if y_unc is not None else None)
            spectra[i]['edited'] = True
            spectra[i]['y'] = y_new
            spectra[i]['y-unc'] = y_unc_new
            if 'y-base' in spectrum:
                del spectra[i]['y-base']
                
def compute_transmittance(spectra):
    """Convert the current spectra from absorbance to transmittance."""
    with np.errstate(invalid='ignore'):
        for (i, spectrum) in enumerate(spectra):
            y = spectrum['y']
            y_unc = spectrum['y-unc']
            y_new = 10**(-y)
            y_unc_new = (y_new * np.log(10) * y_unc
                         if y_unc is not None else None)
            spectra[i]['edited'] = True
            spectra[i]['y'] = y_new
            spectra[i]['y-unc'] = y_unc_new
            if 'y-base' in spectrum:
                del spectra[i]['y-base']
                
def integrate_spectrum(spectrum, selected_points, factor=1., borders=None,
                       integrate_fitted_lines=False):
    """Integrate the spectrum in the current region."""
    if selected_points == []:
        print('Error: No points selected.')
        return
    windows = get_windows_from_points(selected_points)
    x = spectrum['x']
    y = (spectrum['y-lines'] if integrate_fitted_lines and 'y-lines' in spectrum
         else spectrum['y'])
    if borders is None:
        x1 = np.min(windows)
        x2 = np.max(windows)
    else:
        if borders[0][0] > borders[1][0]:
            borders = [borders[1], borders[0]]
        x1, y1 = borders[0]
        x2, y2 = borders[1]
    mask = (x >= x1) & (x <= x2) & np.isfinite(y)
    x_ = x[mask]
    y_ = y[mask]
    if borders is None:
        y1, y2 = y_[0], y_[-1]
    y_b = y1 + (y2 - y1) / (x2 - x1) * (x_ - x1)
    y_ -= y_b
    area = 0.
    for (x1, x2) in windows:
        mask = (x_ >= x1) & (x_ <= x2)
        area += np.trapezoid(y_[mask], x_[mask])
    area /= factor
    return area

def copy_spectrum_part(spectrum, windows, copy_lines=False):
    """Make a copy of the spectrum in the windows defined by the input points."""
    x = spectrum['x']
    y = spectrum['y'] if not copy_lines else spectrum['y-lines']
    y_unc = spectrum['y-unc'] if  not copy_lines else None
    copied_data = []
    for x1x2 in windows:
        x1, x2 = min(x1x2), max(x1x2)
        mask = (x >= x1) & (x <= x2)
        xi = x[mask]
        yi = y[mask]
        yi_unc = y_unc[mask] if y_unc is not None else None
        copied_data += [{'x': xi, 'y': yi, 'y-unc': yi_unc}]
    return copied_data

def paste_spectrum_part(spectrum, copied_data):
    """Replace the input spectrum by the previously copied data in such regions."""
    x = spectrum['x']
    y = spectrum['y']
    y_unc = spectrum['y-unc']
    if y_unc is None:
        y_unc = np.zeros(len(x), float)
    mask = np.zeros(len(x), bool)
    x_new, y_new = np.array([], float), np.array([], float)
    y_new_unc = np.array([], float)
    for copied_part in copied_data:
        xi = copied_part['x']
        yi = copied_part['y']
        yi_unc = copied_part['y-unc']
        if yi_unc is None:
            yi_unc = np.zeros(len(xi), float)
        x1, x2 = xi.min(), xi.max()
        mask |= (x >= x1) & (x <= x2)
        x_new = np.append(x_new, xi)
        y_new = np.append(y_new, yi)
        y_new_unc = np.append(y_new_unc, yi_unc)
    _mask = ~mask
    x_new = np.append(x_new, x[_mask])
    y_new = np.append(y_new, y[_mask])
    y_new_unc = np.append(y_new_unc, y_unc[_mask])
    x_new, inds = np.unique(x_new, return_index=True)
    y_new = y_new[inds]
    y_new_unc = y_new_unc[inds] if np.sum(y_new_unc) != 0 else None
    new_spectrum = {'x': x_new, 'y': y_new, 'y-unc': y_new_unc, 'edited': True}
    return new_spectrum

def sum_spectra(spectra, x_new=None):
    """Sum all the input spectra, obtaining a final single spectrum."""
    num_points = len(spectra[0]['x']) if x_new is None else len(x_new)
    y_new = np.zeros(num_points, float)
    y_new_var = np.zeros(num_points, float)
    for (i, spectrum) in enumerate(spectra):
        yi = spectrum['y']
        yi_unc = spectrum['y-unc']
        if yi_unc is None:
            yi_unc = np.zeros(len(yi), float)
        if x_new is None:
            y_new += yi
            y_new_var += yi_unc**2
        else:
            xi = spectrum['x']
            yi_, yi_unc_ = resample_spectrum(x_new, xi, yi, yi_unc)
            y_new += yi_
            if np.sum(yi_unc) != 0.:
                y_new_var +=  yi_unc_**2
    y_new_unc = np.sqrt(y_new_var) if np.sum(y_new_var) != 0. else None
    if x_new is None:
        x_new = spectra[0]['x']
    new_spectrum = {'x': x_new, 'y': y_new, 'y-unc': y_new_unc, 'edited': True}
    return new_spectrum
        
def merge_spectra(spectra):
    """Perform the union of all the input spectra into a single one."""
    x_new = np.array([], float)
    y_new = np.array([], float)
    are_uncs = any([spectrum['y-unc'] is not None for spectrum in spectra])
    are_baselines = any(['y-base' in spectrum for spectrum in spectra])
    if are_uncs:
        y_unc_new = np.array([], float)
    if are_baselines:
        y_base_new = np.array([], float)
        x_ref_new = np.array([], float)
        y_ref_new = np.array([], float)
    for spectrum in spectra:
        x = spectrum['x']
        y = spectrum['y']
        mask = np.isfinite(y)
        x_new = np.append(x_new, x[mask])
        y_new = np.append(y_new, y[mask])
        if are_uncs:
            y_unc = spectrum['y-unc']
            if y_unc is None:
                y_unc = np.zeros(len(x), float)
            y_unc_new = np.append(y_unc_new, y_unc[mask])
        if are_baselines:
            y_base = (spectrum['y-base'] if 'y-base' in spectrum
                      else np.full(len(x), np.nan))
            y_base_new = np.append(y_base_new, y_base[mask])
    x_new, inds = np.unique(x_new, return_index=True)
    y_new = y_new[inds]
    y_unc_new = (y_unc_new[inds] if are_uncs and np.nansum(y_unc_new) > 0.
                 else None)
    if are_baselines:
        x_new_ = copy.copy(x_new)
    x_new, y_new, y_unc_new = fill_spectrum(x_new, y_new, y_unc_new)
    new_spectrum = {'x': x_new, 'y': y_new, 'y-unc': y_unc_new, 'edited': True}
    if are_baselines:
        y_base_new = y_base_new[inds]
        y_base_new = np.interp(x_new, x_new_, y_base_new)
        x_ref_new, inds = np.unique(x_ref_new, return_index=True)
        y_ref_new = y_ref_new[inds]
        new_spectrum['y-base'] = y_base_new
    return new_spectrum

def load_external_baseline(spectra, active_indices, selected_points,
                           filepath, column_inds):
    """Load file containing baseline into current spectra."""
    data = np.loadtxt(filepath)
    num_cols = data.shape[1]
    if num_cols < 2:
        print('Error: Data file should contain at least 2 columns.')
    if column_inds == 'auto':
        if num_cols == 2:
            column_inds = [1, 2]
        elif num_cols == 3:
            column_inds = [1, 2, 3]
        else:
            column_inds = ([1, 2, 4] if 'flux' in variable_y
                           else [1, 2, 3])
    column_inds = [i-1 for i in column_inds]
    i_xr = column_inds[0]
    i_yb = column_inds[-1]
    x_ref = data[:,i_xr]
    y_base = data[:,i_yb]
    if np.median(x_ref) < 200:
        x_ref = 1e4 / x_ref
    x_base, inds = np.unique(x_ref, return_index=True)
    y_base = y_base[inds]
    for i in active_indices:
        spectrum = spectra[i]
        x = spectrum['x']
        if 'y-base' in spectrum:
            y_base_prev = copy.copy(spectrum['y-base'])
            if selected_points != []:
                if not using_manual_baseline_mode:
                    windows = get_windows_from_points(selected_points)
                    x1, x2 = np.min(windows), np.max(windows)
                else:
                    x1, x2 = np.min(selected_points), np.max(selected_points)
            else:
                x1, x2 = x_ref[0], x_ref[-1]
            mask = (x < x1) | (x > x2)
            y_base[mask] = y_base_prev[mask]
        y_base_ = np.interp(x, x_base, y_base)
        spectra[i]['y-base'] = y_base_

def get_windows_text(windows):
    """Convert the input window points to text."""   
    text_list = []
    for x1x2 in windows:
        x1, x2 = min(x1x2), max(x1x2)
        text_list += ['({:.2f}, {:.2f})'.format(x2, x1)]
    text = str(text_list)
    return text

def get_points_text(selected_points):
    """Convert the input selected points to text."""
    points_text = []
    for (x,y) in selected_points:
        points_text += ['({:.2f}, {:.4g})'.format(x, y)]
    return points_text
         
def save_files(spectra, active_indices, spectra_names, original_filemames,
               saved_var='spectrum'):
    """Save processed spectra."""
    if saved_var not in ('spectrum', 'baseline', 'lines'):
        print('Error: Wrong variable to save.')
        return None, None
    else:
        if saved_var == 'spectrum':
            yvar = 'y'
            extra_text = ''
        elif saved_var == 'baseline':
            yvar = 'y-base'
            extra_text = ' (baseline)'
        elif saved_var == 'lines':
            yvar = 'y-lines'
            extra_text = ' (fitted lines)'
    global original_folder, columns, variable_y, use_csv_format_for_input
    spectra_names_active_indices = np.array(spectra_names)[active_indices]
    nonedited_spectra = [not spectra[i]['edited'] for i in active_indices]
    if all(nonedited_spectra):
        print('No editing of the spectra was performed.\n')
    elif any(nonedited_spectra):
        print('Not all spectra have been edited.\n')
    filename = custom_input(f'- Output filename{extra_text}: ', 'Save files')
    if filename == '':
        return None, None
    if filename.endswith(' '):
        filename = filename[:-1]
    if filename in ('same', '*', '') and len(original_filenames) == 1:
        filename = original_filenames[0]
    elif '*' in filename or filename == '':
        if use_csv_format_for_input:
            default_filename = original_filenames[0]
        else:
            default_filename = os.path.commonprefix(spectra_names_active_indices)
        default_filename = '.'.join(default_filename.split('.')[:-1])
        filename = filename.replace('*', default_filename)
    num_spectra = len(active_indices)
    selected_spectra = np.array(spectra)[active_indices].tolist()
    if (num_spectra > 1 or num_spectra == 1 and filename.endswith('.csv')
            or '.' not in filename):
        use_csv_format_for_output = True
    else:
        use_csv_format_for_output = False
    variable_x_ = 'wavelength (μm)' if use_microns else 'wavenumber (/cm)'
    if use_csv_format_for_output:
        if use_csv_format_for_input:
            columns = np.array(columns)[active_indices].tolist()
        else:
            columns = spectra_names_active_indices.tolist()
            columns = ['.'.join(column.split('.')[:-1]) for column in columns]
            columns = [f'{variable_y} {column}' for column in columns]
        for (i, column) in enumerate(columns):
            column = (column.replace('transmittance', 'transm.')
                      .replace('absorbance', 'abs.')
                      .replace('optical depth', 'opt. depth'))
            if variable_y == 'absorbance' and use_optical_depth:
                column = column.replace('abs.', 'opt. depth')
            if 'coefficient' in variable_y:
                column = column.replace('abs.', 'abs. coeff. (cm2)')
            columns[i] = column
        if not filename.endswith('.csv'):
            filename += '.csv'
        xx = [spectrum['x'] for spectrum in selected_spectra]
        x = np.unique(xx)
        if use_microns:
            x = 1e4/x
        new_df = pd.DataFrame({variable_x_: x})
        for (i,spectrum) in enumerate(selected_spectra):
            column = columns[i]
            x_i = spectrum['x']
            y_i = spectrum[yvar]
            if use_microns:
                x_i = 1e4/x_i
            if variable_y == 'absorbance' and use_optical_depth:
                y_i *= np.log(10)
            y_i = np.interp(x, x_i, y_i)
            new_df[column] = y_i
        nd = max([len(xi.split('.')[-1]) if '.' in xi else 0
                  for xi in x.astype(str)])
        new_df[variable_x_] = new_df[variable_x_].map(lambda x: '{:.{}f}'.format(x, nd))
        new_df.to_csv(filename, index=False, float_format='%.3e')
    else: 
        variable_y = variable_y.replace('spectral flux density', 'flux density')
        if not filename.endswith('.txt') and not filename.endswith('.dat'):
            filename += '.txt'
        idx = active_indices[0]
        columns = spectra_names[idx]
        spectrum = spectra[idx]
        x = spectrum['x']
        y = spectrum[yvar]
        y_unc = spectrum['y-unc'] if yvar == 'y' else None
        variable_x_ = variable_x_.replace(' ', '_')
        variable_y_ = variable_y.replace(' ', '_')
        if variable_y == 'absorbance' and use_optical_depth:
            y *= np.log(10)
            if y_unc is not None:
                y_unc *= np.log(10)
            variable_y_ = 'optical_depth'
        if 'coefficient' in variable_y:
            variable_y_ = 'abs._coeff._(cm2)'
        if use_microns:
            x = 1e4/x
        data = np.array([x, y]).T if y_unc is None else np.array([x, y, y_unc]).T 
        inds = np.argsort(x)
        data = data[inds]
        nd = max([len(xi.split('.')[-1]) if '.' in xi else 0
                  for xi in x.astype(str)])
        fmt = f'%.{nd}f %.3e' if y_unc is None else f'%.{nd}f %.3e %.3e'
        header = f' {variable_x_} {variable_y_}'
        if y_unc is  not None:
            if '(' not in variable_y:
                header += f' {variable_y_}_unc.'
            else:
                variable, units = variable_y_.split('(')
                units = units[:-1]
                header += f' {variable}_unc._({units})'
        np.savetxt(filename, data, fmt=fmt, header=header)
    if yvar == 'y-lines':
        saved_var = 'set of fitted line(s)'
    if num_spectra > 1:
        saved_var = (saved_var.replace('spectrum', 'spectra')
                     .replace('baseline', 'baselines').replace('set', 'sets'))
    print('\n'f'Saved {num_spectra} {saved_var} in {filename}')
    output_data_info = {'file': filename.split(sep)[-1], 'columns': columns,
                        'number of spectra': len(spectra)}
    return output_data_info, filename

def save_action_record(action_log, spectra, filename, output_data_info):
    """Write and save action record file."""
    del action_log[0]
    action_record, count, previous_action = [], 0, ''
    for (i,entry) in enumerate(action_log):
        save_entry = False
        action = entry['action'] if type(entry) is dict else entry
        if action != previous_action:
            save_entry = True
        else:
            if len(action_record) == 0:
                continue
            if action in ('smooth', 'smooth (median)', 'estimate baseline',
                          'remove/interpolate', 'add noise', 'sigma-clip',
                          'multiply'):
                if action in ('smooth', 'smooth (median)',
                              'reduce', 'remove/interpolate'):
                    factor_name = 'smoothing factor'
                elif action == 'add noise':
                    factor_name = 'noise level'
                elif action == 'sigma-clip':
                    factor_name = 'threshold'
                elif action == 'multiply':
                    factor_name = 'factor'
                key = list(action_record[-1].keys())[0]
                old_value = action_record[-1][key][factor_name]
                new_value = '{}, {}'.format(old_value, entry[factor_name])
                action_record[-1][key][factor_name] = new_value
            elif action in ('switch spectrum', 'modify windows', 'modify points'):
                if len(action_record) > 0:
                    del action_record[-1]
                save_entry = True
                count -= 1
                if ('modify' in action and 'modify' not in str(action_record)):
                    params = list(entry.keys())
                    params.remove('action')
                    if entry[params[0]] == []:
                        save_entry = False
        if i+1 == len(action_log) and entry['action'] == 'modify windows':
            save_entry = False
        if save_entry:
            count += 1
            name = '{} ({})'.format(entry['action'], count)
            params = list(entry.keys())
            params.remove('action')
            if len(params) == 0:
                action_record += [name]
            else:
                action_record += [{name: {}}]
                for param in params:
                    action_record[-1][name][param] = entry[param]
            previous_action = entry['action'] if type(entry) is dict else entry
    info_dic = {'input data': input_data_info, 'output data': output_data_info,
                'action record': action_record}
    log_file = '.'.join(filename.split('.')[:-1]) + '-log.txt'
    with open(log_file, 'w') as file:
        yaml.dump(info_dic, file, default_flow_style=False, sort_keys=False)
    print('Saved record file in {}.'.format(log_file))

#%% Functions to create interactive mode.

def click1(event):
    """Interact with the plot by clicking on it."""
    if type(event) is not MouseEvent:
        pass
    global click_time
    click_time = time.time()

def click2(event):
    """Interact with the plot by clicking on it."""
    if type(event) is not MouseEvent:
        pass
    button = str(event.button).lower().split('.')[-1]
    if button in ('left', 'right', '1', '3'):
        global click_time, use_microns
        elapsed_click_time = time.time() - click_time
        x = event.xdata
        if use_microns:
            x = 1e4/x
        if (elapsed_click_time > 0.5  # s
                or x is None or x is not None and not np.isfinite(x)):
            return 
        global spectra, spectra_old, selected_points, idx
        global using_manual_baseline_mode, waiting_for_click
        global action_log, jlog
        if waiting_for_click:
            if button in ('left', '1'):
                global click_action_options
                clicked_borders = click_action_options['clicked_borders']
                y = event.ydata
                clicked_borders += [[x, y]]
                click_action_options['clicked_borders'] = clicked_borders
                plot_baseline_points(clicked_borders)
                if len(clicked_borders) == 2:
                    press_key(SimpleNamespace(key=click_action_options['key']))
            elif button in ('right', '3'):
                waiting_for_click = False
                print('Operation cancelled.')
        elif using_manual_baseline_mode:
            if button in ('left', '1'):
                y = event.ydata
                selected_points += [[x, y]]
                if use_microns:
                    x = 1e4/x
                plt.plot(x, y, '.', color=colors['selected-points'],
                         zorder=3.)
            else:
                if len(selected_points) == 0:
                    return
                xp, yp = np.array(selected_points).T
                i = np.argmin(np.abs(xp - x))
                del selected_points[i]
                plot_data(spectra, spectra_old, active_indices, idx,
                          using_manual_baseline_mode)
                plot_baseline_points(selected_points)
        else:
            if button in ('left', '1'):
                selected_points += [x]
                if use_microns:
                    x = 1e4/x
                plt.axvline(x, color='lightgray', alpha=1.)
                are_points_even = len(selected_points) % 2 == 0
                if are_points_even:
                    x1, x2 = selected_points[-2:]
                    if use_microns:
                        x1, x2 = 1e4/x2, 1e4/x1
                    plt.axvspan(x1, x2, transform=plt.gca().transAxes,
                                color='white', alpha=1.)
            else:
                if len(selected_points) == 0:
                    return
                are_points_even = len(selected_points) % 2 == 0
                was_removed = False
                if are_points_even:
                    windows = np.array(selected_points).reshape(-1,2)
                    for x1x2 in windows:
                        x1, x2 = min(x1x2), max(x1x2)
                        if x1 < x < x2:
                            selected_points.remove(x1)
                            selected_points.remove(x2)
                            was_removed = True
                            break
                if not was_removed:
                    del selected_points[-1]
                plot_data(spectra, spectra_old, active_indices, idx,
                          using_manual_baseline_mode)
                plot_windows(selected_points, use_microns)
        if not waiting_for_click:
            if not using_manual_baseline_mode:
                windows = get_windows_from_points(selected_points)
                windows_text = get_windows_text(windows)
                action_info = {'action': 'modify windows', 'windows': windows_text}
            else:
                points_text = get_points_text(selected_points)
                action_info = {'action': 'modify points', 'points': points_text}
            action_log = action_log[:jlog+1] + [copy.deepcopy(action_info)]
            jlog += 1
        plt.draw()    

def press_key(event):
    """Interact with the plot when pressing a key."""
    if type(event) is not KeyEvent:
        pass
    global original_filenames, spectra, data_log, action_log, ilog, jlog
    global spectra_old, active_indices, idx, all_indices, spectra_names
    global baseline_smooth_size, interp_smooth_size, smooth_size, noise_level
    global using_joint_editing_mode, using_manual_baseline_mode, variable_y
    global selected_points, x_lims, y_lims, old_x_lims, rel_margin_y
    global use_logscale, use_optical_depth, use_microns, invert_yaxis
    global in_macro, k, macro_actions, copied_data, x_min, x_max
    global spectra_colors, spectra_colors_old
    global waiting_for_click, click_action_options
    action_info = None
    if in_macro and event.key in ('enter', ' ', 'escape'):
        if event.key == 'escape':
            in_macro = False
            print('Macro exited.')
        elif event.key in ('enter', ' '):
            if event.key == 'enter':
                in_macro = True
                if type(macro_actions[k]) is str:
                    action = macro_actions[k]
                else:
                    action = list(macro_actions[k].keys())[0]
                    params = macro_actions[k][action]
                if action == 'modify windows':
                    windows = params['windows']
                    if windows != 'auto':
                        windows = [list(np.array(text[1:-1].split(', '),float))
                                   for text in windows]
                        selected_points = list(np.array(windows).flatten())
                    event.key = 'w'
                elif action == 'estimate baseline':
                    event.key = 'b'
                    factor = params['smoothing factor']
                elif action == 'reduce':
                    event.key = 'r'
                elif action == 'interpolate' in action:
                    event.key = 'x'
                    factor = params['smoothing factor']
                elif action in ('smooth', 'smooth (median)'):
                    event.key = 's' if action == 'smooth' else 'ctrl+s'
                    factor = params['smoothing factor']
                elif action == 'add noise':
                    event.key == 'n'
                    factor = params['smoothing factor']
                elif action == 'offset':
                    event.key = 'o'
                    offset = params['offset']
                elif action == 'set to zero':
                    event.key = '0'
                elif action == 'set negatives to zero':
                    event.key = '='
                elif action == 'set to one':
                    event.key = '1'
                elif action == 'set one as maximum':
                    event.keu = '!'
                elif action == 'multiply':
                    event.key = 'f'
                    factor = params['factor']
                elif action == 'convert to absorption coefficient':
                    event.key = 'F'
                    coldens = params['column density']
                elif action == 'convert to absorbance':
                    event.key = 'a'
                elif action.startswith('resample'):
                    event.key = ',' if 'simple' in action else ';'
                    text = params['wavenumber array (start, end, step)']
                elif action in ('delete spectrum', 'remove region'):
                    event.key == 'backspace'
                else:
                    print(f'Unknown action: {action}.')
            k += 1
    elif waiting_for_click:
        if event.key == 'escape':
            waiting_for_click = False
        elif event.key == 'enter':
            click_action_options['clicked_borders'] = None
            event.key = click_action_options['key']
        elif event.key == '0':
            click_action_options['clicked_borders'] = 0
            event.key = click_action_options['key']
    elif event.key == 'escape':
        plt.close(1)
        return
    elif event.key in ('shift+enter', 'ctrl+enter', 'cmd+enter', 'alt+enter'):
        data_log = data_log[:ilog+1]
        action_log = action_log[:jlog+1]
        if event.key in ('cmd+enter', 'alt+enter'):
            saved_var = custom_input('- Variable to save (spectrum, baseline, lines)',
                                     'Save files')
        else:
            saved_var = 'spectrum'
        output_data_info, filename = save_files(spectra, active_indices,
                                  spectra_names, original_filenames, saved_var)
        if event.key == 'shift+enter':
            save_action_record(action_log, spectra, filename, output_data_info)
    x_lims = list(sorted(plt.xlim()))
    y_lims = list(sorted(plt.ylim()))
    if use_optical_depth:
        y_lims = list(np.array(y_lims) / np.log(10))
    if using_joint_editing_mode and len(all_indices) == 1:
        using_joint_editing_mode = False
    if not using_joint_editing_mode:
        active_indices = [idx]
    if 'flux' in variable_y and invert_yaxis:
        invert_yaxis = False
    if event.key in ('y', 'Y', 'ctrl+y'):
        prev_y_lims = copy.copy(y_lims)
        x_lims_ = [1e4/x_lims[1], 1e4/x_lims[0]] if use_microns else x_lims
        if x_lims_[1] < 0:
            x_lims_[1] = x_max
        if event.key == 'y':
            ylims1 = calculate_ylims(spectra, x_lims_, 1., 99., rel_margin_y)
            ylims2 = calculate_ylims(spectra_old, x_lims_, 1., 99., rel_margin_y)
            y_lims = [min(ylims1[0], ylims2[0]), max(ylims1[1], ylims2[1])]
        elif event.key == 'Y':
            ylims1 = calculate_ylims([spectra[idx]], x_lims_, 1., 99., rel_margin_y)
            ylims2 = calculate_ylims([spectra_old[idx]], x_lims_, 1., 99., rel_margin_y)
            y_lims = [min(ylims1[0], ylims2[0]), max(ylims1[1], ylims2[1])]
        else:
            y_lims = calculate_ylims([spectra[idx]], x_lims_, 1., 99., rel_margin_y)
        if y_lims == prev_y_lims:
            y_lims = calculate_robust_ylims(spectra, x_lims_, 0., 100., rel_margin_y)
    elif event.key == 'ctrl+-':
        if 'flux' not in variable_y:
            invert_yaxis = not invert_yaxis
    elif event.key in ('z', 'Z', '<', '+', '-', 'left', 'right'):
        x_range = x_lims[1] - x_lims[0]
        if event.key in ('z', 'Z', '<', '+', '-'):
            if event.key in ('z', '+'):  # zoom
                x_lims = [x_lims[0] + x_range/8, x_lims[1] - x_range/8]
            else:  # de-zoom   (f_unzoom = f_zoom - 2)
                x_lims = [x_lims[0] - x_range/6, x_lims[1] + x_range/6]
                x_lims[0] = max(0.1, x_lims[0])
            old_x_lims_ = ([1e4/old_x_lims[1], 1e4/old_x_lims[0]]
                           if use_microns else old_x_lims)
            if x_lims[0] < old_x_lims_[0] and x_lims[1] > old_x_lims_[1]:
                x_lims = copy.copy(old_x_lims_)
        else:  # move horizontaly
            s = 1 if use_microns else -1
            if event.key == 'left':
                in_edge = x_lims[0] < 0. if use_microns else x_lims[1] > 6000.
                if not in_edge:
                    x_lims = [x_lims[0] - s * x_range/6, x_lims[1] - s * x_range/6]
            elif event.key == 'right':
                in_edge = x_lims[1] > 300. if use_microns else x_lims[0] < 0.
                if not in_edge:
                    x_lims = [x_lims[0] + s * x_range/6, x_lims[1] + s * x_range/6]
    elif event.key in ('up', 'down'):
        if len(spectra) > 1:
            idx = idx+1 if event.key == 'up' else idx-1
            idx = idx % len(spectra)
            if not using_joint_editing_mode:
                active_indices = [idx]
        action_info = {'action': 'switch spectrum', 'spectrum': spectra_names[idx]}
    elif event.key in ('tab', '\t'):
        if selected_points == [] and not using_manual_baseline_mode:
            selected_points = [x_min, x_max]
        else:
            selected_points = []
        if not using_manual_baseline_mode:
            action_info = {'action': 'modify windows', 'windows': []}
        else:
            action_info = {'action': 'modify points', 'points': []}
    elif event.key in ('alt+-', 'cmd+-'):
        if not using_manual_baseline_mode:
            windows = get_windows_from_points(selected_points)
            xx = np.concatenate(tuple([spectrum['x'] for spectrum in spectra]))
            x = np.unique(xx)
            windows = np.array(invert_windows(windows, x))
            selected_points = list(windows.flatten())
            windows_text = get_windows_text(windows)
            action_info = {'action': 'modify windows', 'windows': windows_text}
    elif event.key == 'backspace':
        if len(selected_points) > 1 and not using_manual_baseline_mode:
            remove_regions(spectra, active_indices, selected_points)
            print('Removed regions in spectral windows.')
            action_info = {'action': 'remove region'}
        elif len(all_indices) > 1:
            action_info = {'action': 'delete spectrum',
                           'deleted spectrum': spectra_names[idx]}
            del spectra[idx]
            del spectra_old[idx]
            del spectra_names[idx]
            idx = max(0, idx-1)
            all_indices = list(range(len(spectra)))
            active_indices = (copy.copy(all_indices) if using_joint_editing_mode
                              else [idx])
            if len(spectra) > 1 and using_joint_editing_mode:
                action_info['current selected spectrum'] = spectra_names[idx]
    elif event.key in ('w', 'W', 'ctrl+w'):
        if event.key == 'ctrl+w':
            text = custom_input('- Write new window (/cm): ', 'Add window')
            text = (text.replace('(','').replace(')','')
                    .replace('[','').replace(']','').replace(', ',','))
            x1, x2 = text.split(',')
            x1x2 = [float(x1), float(x2)]
            x1, x2 = min(x1x2), max(x1x2)
            selected_points += [x1, x2]
        else:
            if not in_macro or in_macro and windows == 'auto':
                global original_folder
                text = (spectra_names[idx] if event.key == 'w'
                        else custom_input('- Write species to mask: '), 'Add windows')
                species_list = parse_composition(text, original_folder)
                x = spectra[idx]['x']
                if len(selected_points) == 0:
                    mask = np.zeros(len(x), bool)
                else:
                    previous_windows = get_windows_from_points(selected_points)
                    mask = ~get_mask_from_windows(previous_windows, x)
                for species in species_list:
                    if species in species_windows:
                        for x1x2 in species_windows[species]:
                            x1, x2 = min(x1x2), max(x1x2)
                            mask |= (x >= x1) & (x <= x2)
                windows = get_windows_from_mask(~mask, x)
                selected_points = list(np.array(windows).flatten()[::-1])
        print('Modified windows.')
        windows = get_windows_from_points(selected_points)
        windows_text = get_windows_text(windows)
        action_info = {'action': 'modify windows', 'windows': windows_text}
    elif event.key in ('s', 'S', 'ctrl+s', 'ctrl+S'):
        if not in_macro:
            factor = copy.copy(smooth_size)
        if 'S' in event.key:
            text = custom_input('- Enter smoothing factor: ', 'Smooth')
            try:
                factor = round(float(text))
            except:
                factor = 1.
        if factor <= 1 or factor % 1 != 0:
            print('Error: Smoothing factor should be an integer greater than 1.')
        else:
            function = np.nanmedian if 'ctrl' in event.key else np.nanmean
            smooth_spectra(spectra, active_indices, selected_points, factor, function)
            print(f'Smoothed regions in current windows with smoothing factor {factor}.')
            action_name = 'smooth (median)' if 'ctrl' in event.key else 'smooth'
            action_info = {'action': action_name, 'smoothing factor': factor}
    elif event.key in ('x', 'X', 'ctrl+x', 'ctrl+X',
                       'alt+x', 'alt+X', 'cmd+x', 'cmd+X'):
        if selected_points == []:
            print('Error: No points selected.')
        else:
            if not in_macro:
                factor = copy.copy(interp_smooth_size)
            if 'X' in event.key:
                text = custom_input('- Enter smoothing factor for interpolation: ',
                                    'Interpolate')
                try:
                    factor = round(float(text))
                except:
                    factor = 1.
            if factor < 1 or factor % 1 != 0:
                print('Error: Smoothing factor should be an integer greater than 0.')
            else:
                interpolation = 'pchip' if 'ctrl' in event.key else 'spline'
                exclude_selected_regions = (False if 'alt' in event.key
                                            or 'cmd' in event.key else True)
                interpolate_regions(spectra, active_indices, selected_points,
                                    factor, interpolation, exclude_selected_regions)
                print('Interpolated regions in current windows with'
                      f' smoothing factor {factor}.')
                action_info = {'action': 'interpolate',
                               'smoothing factor': factor}
    elif event.key in ('n', 'N'):
        if not in_macro:
            factor = copy.copy(noise_level)
        if event.key == 'N':
            factor = custom_input('- Enter noise level: ', 'Noise (N)')
            try:
                factor = float(text)
            except:
                factor = 0.
        if factor != 0.:
            add_noise(spectra, active_indices, selected_points, factor)
            print(f'Added normal noise with a standard deviation of {factor:.3e}.')
            action_info = {'action': 'add noise', 'noise level': f'{factor:.3e}'}
    elif event.key in ('b', 'B', 'ctrl+b', 'ctrl+B'):
        if len(selected_points) < 2:
            print('Error: No windows selected.')
            return
        if not in_macro:
            factor = copy.copy(baseline_smooth_size)
        if event.key in ('B', 'ctrl+B') and not using_manual_baseline_mode:
            text = custom_input('- Enter smoothing factor for baseline: ',
                                'Baseline (B)')
            try:
                factor = round(float(text))
            except:
                factor = 0
        if factor < 1 or factor % 1 != 0:
            print('Error: Smoothing factor should be an integer greater than 0.')
        else:
            interpolation = 'pchip' if 'ctrl' in event.key else 'spline'
            estimate_baseline(spectra, active_indices, selected_points, factor, 
                              using_manual_baseline_mode, interpolation)
            suffix = '' if len(active_indices) == 1 else 's'
            if not using_manual_baseline_mode:
                print(f'Computed baseline{suffix} from windows'
                      f' with smoothing factor {factor}.')
            else:
                print(f'Computed baseline{suffix} from reference points.')
                factor = 1
            action_info = {'action': 'estimate baseline', 'smoothing factor': factor}
    elif event.key in ('r', 'R') or 'flux' in variable_y and event.key == 't':
        baselines_in_spectra = ['y-base' in spectra[i] for i in active_indices]
        if ('abs' not in variable_y and not all(baselines_in_spectra)
                or ('abs' in variable_y and event.key == 'r'
                and not all(baselines_in_spectra))):
            if using_joint_editing_mode:
                print('Error: Not all spectra have baselines computed.')
            else:
                print('Error: No baseline computed.')
            return
        else:
            lines_in_spectra = ['y-lines' in spectra[i] for i in active_indices]
            if not all(lines_in_spectra):
                if using_joint_editing_mode:
                    print('Error: Not all specra have fitted lines.')
                else:
                    print('Error: No fitted lines.')
                return
        if 'abs' not in variable_y:
            if 'flux' in variable_y and event.key == 't':
                event.key = 'r'
            mode = 'divide' if event.key == 'r' else 'subtract'
        else:
            mode = 'subtract'
            use_lines = True if event.key == 'R' else False
        if not use_lines:
            reduce_spectra(spectra, active_indices, mode)
        else:
            windows = (get_windows_from_points(selected_points)
                       if len(selected_points) > 1 else [[x_min, x_max]])
            subtract_lines(spectra, active_indices, windows)
        suffix = 'um' if len(active_indices) == 1 else 'a'
        extra_msg = ('by converting to transmittance using the baseline'
                     if 'flux' in variable_y and event.key == 'r'
                     else 'by subtracting the baseline')
        print(f'Reduced selected spectr{suffix} {extra_msg}.')
        if 'flux' in variable_y and event.key == 'r':
            variable_y = 'transmittance'
            spectra_old = copy.deepcopy(spectra)
            x_lims_ = [1e4/x_lims[1], 1e4/x_lims[0]] if use_microns else x_lims
            y_lims = calculate_robust_ylims(spectra, x_lims_, rel_margin_y)
            for i in active_indices:
                del spectra[i]['y-base']
        action_info = {'action': 'reduce'}
    elif event.key in ('0', '='):
        if not using_manual_baseline_mode:
            if event.key == '0':
                only_negatives = False
                action_info = {'action': 'set to zero'}
            else:
                only_negatives = True
                action_info = {'action': 'set negatives to zero'}
            set_zero(spectra, active_indices, selected_points, only_negatives)
            if only_negatives:
                print('Negative values of region in current windows set to 0.')
            else:
                print('Region in current windows set to 0.')
        else:
            points_text = get_points_text(selected_points)
            action_info = {'action': 'modify points', 'points': points_text}
            for x in np.linspace(x_min, x_max, 40):
                selected_points += [[x, 0.]] 
    elif event.key in ('1', '!'):
        if variable_y == 'transmittance':
            if not using_manual_baseline_mode:
                if event.key == '1':
                    only_gtr1 = False
                    action_info = {'action': 'set to one'}
                else:
                    only_gtr1 = True
                    action_info = {'action': 'set one as maximum'}
                set_one(spectra, active_indices, selected_points, only_gtr1)
                if only_gtr1:
                    print('Values greater than 1 of region in current windows set to 1.')
                else:
                    print('Region in current windows set to 1.')
            else:
                points_text = get_points_text(selected_points)
                action_info = {'action': 'modify points', 'points': points_text}
                for x in np.linspace(x_min, x_max, 40):
                    selected_points += [[x, 1.]] 
    elif event.key in ('o', 'O'):
        if event.key == 'o' and not in_macro:
            text = custom_input('- Enter an offset value: ', 'Offset (O)')
            try:
                offset = float(text)
            except:
                offset = None
        if event.key == 'O':
            baselines_in_spectra = ['y-base' in spectra[i] for i in active_indices]
            if not all(baselines_in_spectra):
                print('Error: Baseline should be computed for all selected spectra.')
                return
            offset = ('baseline' if using_manual_baseline_mode
                      and selected_points != [] else None)
        if offset is not None:
            add_offset(spectra, active_indices, selected_points, offset)
            if not using_manual_baseline_mode or selected_points == []:
                print(f'Added offset of {offset} to selected spectra.')
                action_info = {'action': 'add offset', 'offset': offset}
            else:
                print('Added baseline offset to selected spectra.')
                points_text = get_points_text(selected_points)
                action_info = {'action': 'add baseline offset',
                               'points': points_text}
    elif event.key in ('g', 'G', 'ctrl+g', 'ctrl+G'):
        if 'abs' not in variable_y:
            print('Error: Cannot fit a Gaussian in flux mode.')
        elif len(selected_points) < 2:
            print('Error: You must select a window where to fit a Gaussian.')
        else:
            windows = get_windows_from_points(selected_points)
            if not waiting_for_click:
                if 'G' in event.key:
                    text = custom_input('- Write the center(s) of the line(s): ',
                                        'Gaussian fit (G)').replace(' ', '')
                    if text == '':
                        return
                    centers = text.split(',')
                    for (i, x) in enumerate(centers):
                        if x == '':
                            centers[i] = None
                        else:
                            centers[i] = float(x)
                else:
                    centers = None
                if 'ctrl' in event.key:
                    fit_function = fit_gaussians
                else:
                    fit_function = fit_gaussian
                print('Click two reference points for the line borders'
                      ' (press Enter to skip).')
                click_action_options = {'key': event.key, 'centers': centers,
                        'fit_function': fit_function, 'clicked_borders': []}
                waiting_for_click = True
            else:
                clicked_borders = click_action_options['clicked_borders']
                if clicked_borders in (0, None) or len(clicked_borders) == 2:
                    waiting_for_click = False
                    centers = click_action_options['centers']
                    fit_function = click_action_options['fit_function']
                    fit_function(spectra, active_indices, windows, clicked_borders,
                                 centers)
                    windows_text = get_windows_text(windows)
                    action_info = {'action': 'fit line', 'windows': windows_text}
                    if clicked_borders not in (0, None):
                        x1, y1 = clicked_borders[0]
                        x2, y2 = clicked_borders[1]
                        line_borders_text = (f'({x1:.3f}), ({y1:.3f});'
                                             ' ({x2:.3f}), ({y2:.3f})')
                        action_info['line borders'] = line_borders_text
                    action_info['centers'] = centers
    elif event.key == 'alt+tab':
        windows = get_windows_from_points(selected_points)
        remove_fitted_lines(spectra, active_indices, windows)
        windows_text = get_windows_text(windows)
        action_info = {'action': 'remove fitted lines', 'windows': windows_text}
    elif event.key in ('i', 'I', 'ctrl+i', 'ctrl+I'):
        if len(selected_points) < 2:
            print('Error: You must select a region to integrate.')
            return
        if not waiting_for_click:
            if 'i' in event.key:
                factor = 1.
                area_variable = 'area'
                area_units = '/cm'
            elif 'I' in event.key:
                text = custom_input('- Introduce the band strength (cm): ',
                                    'Integrate (I)')
                try:
                   factor =  float(text) / np.log(10)
                except:
                    return
                area_variable = 'column density'
                area_units = '/cm2'
            print('- Click two continuum points for estimating the baseline'
                  ' (press Enter to skip).')
            click_action_options = {'key': event.key, 'clicked_borders': [],
                    'area_variable': area_variable, 'area_units': area_units,
                    'factor': factor}
            waiting_for_click = True
        else:
            clicked_borders = click_action_options['clicked_borders']
            if clicked_borders in (0, None) or len(clicked_borders) == 2:
                waiting_for_click = False
                key = click_action_options['key']
                factor = click_action_options['factor']
                area_variable = click_action_options['area_variable']
                area_units = click_action_options['area_units']
                integrate_fitted_lines = 'ctrl' in key
                area = integrate_spectrum(spectra[idx], selected_points,
                           factor, clicked_borders, integrate_fitted_lines)
                print('Integrated {}: {:.3e} {}'.format(area_variable,
                                                        area, area_units))
    elif event.key in ('f', 'F', 'ctrl+f', 'ctrl+F'):
        if event.key in ('F', 'ctrl+F') and 'abs' not in variable_y:
            print('Error: Spectrum must be in absorbance in order to'
                  ' convert to absorption coefficient, or viceversa.')  
            return
        elif (event.key in ('F', 'ctrl+F')
                  and (len(spectra) > 1 and not using_joint_editing_mode)):
            using_joint_editing_mode = True
        if event.key in ('f', 'ctrl+f'):
            text = custom_input('- Enter factor to multiply: ', 'Multiply (F)')
            if not in_macro:
                factor = 1. if text == '' else float(text)
            if factor != 1.:
                multiply_lines = event.key == 'ctrl+f'
                multiply_spectra(spectra, active_indices, selected_points,
                                 factor, multiply_lines)
                suffix = 'um' if len(active_indices) == 1 else 'a'
                print(f'Multiplied spectr{suffix} by a factor of {factor} in'
                      ' selected regions.')
                action_info = {'action': 'multiply', 'factor': factor}
        else:
            if not in_macro:
                new_variable_y = ('absorption coefficient'
                                  if variable_y == 'absorbance' else 'absorbance')
                window_title = ('Normalize (Shift+F)' if variable_y == 'absorbance'
                                else 'Scale (Shift+F)')
                text = custom_input('- Enter column density (/cm2) to convert'
                                    f' to {new_variable_y}: ', window_title)
                try:
                    coldens = float(text)
                except:
                    coldens = None
                    print('Error: Could not read input column density.' )
            if coldens is not None:
                if variable_y == 'absorbance':
                    factor = np.log(10) / coldens
                else:
                    factor = coldens
                variable_y = ('absorption coefficient (cm2)'
                              if variable_y == 'absorbance' else 'absorbance')
                multiply_spectra(spectra, active_indices, [], factor)
                spectra_old = copy.deepcopy(spectra)
                x_lims_ = [1e4/x_lims[1], 1e4/x_lims[0]] if use_microns else x_lims
                y_lims = calculate_ylims(spectra, x_lims_, 1., 99., rel_margin_y)
                for (i, spectrum) in enumerate(spectra):
                    if 'y-base' in spectrum:
                        del spectra[i]['y-base']
                if variable_y == 'absorption coefficient (cm2)':
                    print('Converted to absorption coefficient.')
                    action_info = {'action': 'convert to absorption coefficient',
                                   'column density (/cm2)': coldens}
                else:
                    action_info = {'action': 'convert to absorbance',
                                   'column density (/cm2)': coldens}
    elif event.key in ('ctrl+c', 'cmd+c', 'ctrl+C', 'cmd+C'):
        windows = get_windows_from_points(selected_points)
        copy_lines = 'C' in event.key
        copied_data = copy_spectrum_part(spectra[idx], windows, copy_lines)
        print('Copied spectrum in selected regions.')
        windows_text = get_windows_text(windows)
        action_info = {'action': 'copy spectrum', 'spectrum': spectra_names[idx],
                       'windows': windows_text}
    elif event.key in ('ctrl+v', 'cmd+v'):
        if copied_data is None:
            print('Warning: No data to be pasted.')
        else:
            new_spectrum = paste_spectrum_part(spectra[idx], copied_data)
            spectra[idx] = new_spectrum
            windows_text = []
            for spectrum in copied_data:
                x = spectrum['x']
                x1, x2 = x.min(), x.max()
                windows_text += [[f'({x2:.2f}, {x1:.2f})']]
            print('Pasted previously copied data in current spectrum.')
            action_info = {'action': 'copy spectrum', 'spectrum': spectra_names[idx],
                           'windows': windows_text}
    elif event.key == 'u' and len(spectra) > 1:
        new_name = os.path.commonprefix(spectra_names)
        if new_name[-1] in ('-', '_', ' '):
            new_name = new_name[:-1]
        if new_name == '':
            new_name = custom_input('Write a name for the resulting joint spectrum: ',
                                    'Union (U)')
        new_spectrum = merge_spectra(spectra)
        spectra = [new_spectrum]
        spectra_old = copy.deepcopy(spectra)
        spectra_names = [new_name]
        active_indices = [0]
        all_indices = [0]
        idx = 0
        print('Performed union of all spectra.')
        action_info = {'action': 'union of spectra'}
    elif event.key == 'U' and len(spectra) > 1:
        new_name = custom_input('Write a short name for the resulting spectrum: ',
                                "Sum (Shift+U)")
        x_new = None
        x = spectra[0]['x']
        for i in range(1, len(spectra)):
            xi = spectra[i]['x']
            if not np.array_equal(x, xi):
                print('Warning: Mismatch between spectral points.')
                text = custom_input('- Insert new wavenumber array (/cm) to'
                                    ' resample spectra (start, end, stop): ',
                                    "Resample / Sum (Alt+'+')")
                text = (text.replace('(','').replace(')','')
                        .replace('[','').replace(']','').replace(' ',''))
                params = text.split(',')
                x1x2 = np.array(params[:2], float)
                step = float(params[2])
                x1, x2 = min(x1x2), max(x1x2)
                x_new = np.arange(x1, x2, step)
                break
        new_spectrum = sum_spectra(spectra, x_new)
        spectra = [new_spectrum]
        spectra_old = copy.deepcopy(spectra)
        spectra_names = [new_name]
        active_indices = [0]
        all_indices = [0]
        idx = 0
        print('Summed up all spectra.')
        action_info = {'action': 'sum of spectra'}
    elif event.key == 'ctrl+g':
        1
    elif event.key in ('e', 'E'):
        text = custom_input('- Write the path of the baseline file: ',
                            'Load external baseline (E)')
        if text != '':
            if text.endswith(' '):
                text = text[:-1]
            filepath = copy.copy(text)
            if event.key == 'e':
                column_inds = 'auto'
            else:
                text = custom_input('Select the column indices for the baseline'
                                    ' (x, y).')
                column_inds = 'auto' if text == '' else list(text)
            load_external_baseline(spectra, active_indices, selected_points,
                                   filepath, column_inds)
            print('Loaded external baseline.')
            action_info = {'action': 'load external baseline',
                           'file': filepath}
    elif event.key in ('p', 'P'):
        global weights, weights_path
        if 'flux' in variable_y:
            print('Error: Cannot use AICE with flux. Convert first to'
                  'transmittance or absorbance / optical depth.')
        elif weights is None:
            print(f'Error: Could not find AICE weights in {weights_path}.')
        else:
            spectrum = spectra[idx]
            x1, x2, dx = aice_xrange_params
            x1x2 = (x1, x2)
            x1, x2 = min(x1x2), max(x1x2)
            x_aice = np.arange(x1, x2, dx)    
            x = spectrum['x']
            if 'abs' in variable_y:   
                y = copy.copy(spectrum['y'])
                if event.key == 'P' and 'y-lines' in spectrum:
                    y_lines = copy.copy(spectrum['y-lines'])
                    mask = np.isfinite(y_lines)
                    y[mask] = y_lines[mask]
            else:
                y = 10**spectrum['y']
            predictions_df = aice_model(x, y, x_aice, weights)
            print(predictions_df)
    elif event.key in (',', ';', 'ctrl+,', 'ctrl+;'):
        if not in_macro:
            variable = 'wavelength' if 'ctrl' in event.key else 'wavenumber'
            units = 'μm' if variable == 'wavelength' else '/cm'
            text = custom_input(f'- Enter new {variable} array ({units}) to'
                                ' resample: (start, end, step): ', 'Resample (,)')
        if text != '':
            text = (text.replace('(','').replace(')','')
                    .replace('[','').replace(']','').replace(' ',''))
            params = text.split(',')
            kind = 'simple' if ',' in event.key else 'precise'
            x1x2 = np.array(params[:2], float)
            step = float(params[2])
            x1, x2 = min(x1x2), max(x1x2)
            x_new = np.arange(x1, x2, step)
            if variable == 'wavelength':
                x_new = 1e4 / x_new
            do_resampling(spectra, active_indices, selected_points, x_new, kind)
            print(f'Resampling performed ({kind}).')
            action_info = {'action': f'resample ({kind})',
                           f'{variable} ({units}) array (start, end, step)':
                               list(params)}
    elif event.key in ('a', 'd'):
        if variable_y == 'absorbance':
            if event.key == 'a' and use_optical_depth:
                use_optical_depth = False
            elif event.key == 'd' and not use_optical_depth:
                use_optical_depth = True
        elif variable_y != 'transmittance':
            desired_variable_y = ('absorbance' if event.key == 'a'
                                  else 'optical depth')
            print('Error: Spectra must be in transmittance to convert to'
                  f' {desired_variable_y}.')
        else:
            compute_absorbance(spectra)
            variable_y = 'absorbance'
            use_optical_depth = True if event.key == 'd' else False
            use_logscale = False
            invert_yaxis = not invert_yaxis
            x_lims_ = [1e4/x_lims[1], 1e4/x_lims[0]] if use_microns else x_lims
            y_lims = calculate_ylims(spectra, x_lims_, 1., 99., rel_margin_y)
            noise_level = compute_noise(spectra)
            spectra_old = copy.deepcopy(spectra)
            if event.key == 'a':
                print('Converted to absorbance.')
                action_info = {'action': 'convert to absorbance'}
            else:
                print('Converted to optical depth.')
                action_info = {'action': 'convert to optical depth'}
    elif event.key == 't':
        if variable_y == 'transmittance':
            pass
        elif variable_y != 'absorbance':
            print('Error: Spectra must be in absorbance / optical depth to'
                  ' convert to transmittance.')
        else:
            compute_transmittance(spectra)
            variable_y = 'transmittance'
            use_logscale = False
            invert_yaxis = not invert_yaxis
            x_lims_ = [1e4/x_lims[1], 1e4/x_lims[0]] if use_microns else x_lims
            y_lims = calculate_ylims(spectra, x_lims_, 1., 99., rel_margin_y)
            noise_level = compute_noise(spectra)
            spectra_old = copy.deepcopy(spectra)
            print('Converted to transmittance.')
            action_info = {'action': 'convert to transmittance'}
    elif event.key == 'j':
        if len(all_indices) > 1:
            using_joint_editing_mode = not using_joint_editing_mode
            if not using_joint_editing_mode:
                active_indices = [idx]
                print('Processing mode has changed to individual.')
                action_info = {'action': 'activate individual processing mode',
                               'spectrum': spectra_names[idx]}
            else:
                active_indices = copy.copy(all_indices)
                print('Processing mode has changed to joint.')
                action_info = {'action': 'activate joint processing mode'}
    elif event.key == '.':
        using_manual_baseline_mode = not using_manual_baseline_mode
        if len(spectra) > 1:
            extra_msg = (' (for all spectra)' if using_joint_editing_mode
                         else ' (only for selected spectrum)')
        else:
            extra_msg = ''
        selected_points = []
        if using_manual_baseline_mode:
            print('Using manual baseline mode.' + extra_msg)
            action_info = {'action': 'activate manual baseline mode',
                           'spectrum': spectra_names[idx]}
        else:
            print('Using windows baseline mode.' + extra_msg)
            action_info = {'action': 'activate windows baseline mode'}
    elif event.key == 'l':
        use_logscale = not use_logscale
    elif event.key == 'm':
        x_min_, x_max_ = ([x_min, x_max] if not use_microns else
                          [1e4/x_max, 1e4/x_min])
        use_microns = not use_microns
        x_lim1 = max(x_min_, x_lims[0])
        x_lim2 = min(x_max_, x_lims[1])
        x_lim1, x_lim2 = [1e4/x_lim2, 1e4/x_lim1]
        if x_lims[0] < x_min_ or x_lims[1] > x_max_:
            margin = rel_margin_x * (x_lim2 - x_lim1)
            if x_lims[0] < x_min_:
                x_lim1 -= margin
            if x_lims[1] > x_max_:
                x_lim2 += margin
        x_lims = [x_lim1, x_lim2]
    elif event.key in ('M', 'ctrl+M') and not in_macro:
        if event.key == 'M':
            text = custom_input('Write the path of the macro file: ',
                                'Macro (Shift+M)')
            if text != '':
                if text.endswith(' '):
                    text = text[:-1]
                filepath = text
                with open(filepath, 'r') as file:
                    macro_actions = yaml.safe_load(file)
                macro_name = filepath.split(sep)[-1]
        else:
            text = custom_input('Write the name of the predefined macro: ',
                                'Macro (Ctrl+M)')
            if text != '':
                macro_name = text
                macro_actions = yaml.safe_load(predefined_macros[macro_name])
        if text != '':
            print()
            print(f'Starting macro {name}.')
            print('Please, go back to the plot window and click on it.'
                  ' Press Enter to apply next action, Space to skip or'
                  ' Escape to exit the macro.')
            k = 0
            in_macro = True
    elif event.key in ('ctrl+z', 'cmd+z', 'ctrl+Z', 'cmd+Z','ctrl+<', 'cmd+<'):
        if 'z' in event.key and ilog == 0:
            print('Error: Cannot undone.')
        elif 'z' not in event.key and ilog == len(data_log)-1:
            print('Error: Cannot redo.')
        else:
            prev_variable_y = copy.copy(variable_y)
            in_macro_prev = data_log[ilog]['in_macro']
            ilog = (max(0, ilog-1) if 'z' in event.key
                    else min(len(data_log)-1, ilog+1))
            data = copy.deepcopy(data_log[ilog])
            spectra = data['spectra']
            spectra_old = data['spectra_old']
            spectra_colors = data['spectra_colors']
            spectra_colors_old = data['spectra_colors_old']
            all_indices = data['all_indices']
            active_indices = data['active_indices']
            if idx not in all_indices:
                idx = data['idx']
            spectra_names = data['spectra_names']
            copied_data = data['copied_data']
            variable_y = data['variable_y']
            if in_macro or in_macro_prev:
                selected_points = data['selected_points']
            in_macro = data['in_macro']
            jlog = (max(0, jlog-1) if 'z' in event.key
                    else min(len(action_log)-1, jlog+1))
            action_info = copy.deepcopy(action_log[jlog-1])
            if prev_variable_y != variable_y:
                x_lims_ = [1e4/x_lims[1], 1e4/x_lims[0]] if use_microns else x_lims
                y_lims = calculate_ylims(spectra, x_lims_, 1., 99., rel_margin_y)
            if (prev_variable_y == 'transmittance' and variable_y == 'absorbance'
                    or prev_variable_y == 'absorbance' and variable_y == 'transmittance'):
                invert_yaxis = not invert_yaxis
            if 'flux' in variable_y:
                invert_yaxis = False
            if 'z' in event.key:
                print('Action undone.')
            else:
                print('Action redone.')
    if in_macro and k < len(macro_actions):
        action_text = (macro_actions[k] if type(macro_actions[k]) is str
                       else str(macro_actions[k])[1:-1].replace("'",""))
        print(f'Next action: {action_text}')
    if event.key in ('b', 'B', 'ctrl+b', 'ctrl+B',
                     's', 'S', 'ctrl+s', 'ctrl+S',
                     'x', 'X', 'ctrl+x', 'ctrl+X',
                     'alt+x', 'alt+X', 'cmd+x', 'cmd+X',
                     'r', 'R', 'a', 't', 'd', 'u', 'U', 'e', 'E', ',', ';',
                     'n', 'N', 'o', 'O', 'f', 'ctrl+f', 'F', '0', '=', '1', '!',
                     'ctrl+c', 'cmd+c', 'ctrl+C', 'cmd+C', 'ctrl+v', 'cmd+v',
                     'w', 'W', 'ctrl+W', 'g', 'G', 'ctrl+g', 'ctrl+G', '.', 'j',
                     'up', 'down', 'tab', 'alt+tab', 'backspace'):
        if action_info is None:
            pass
        elif event.key not in ('.', 'j', 'up', 'down', 'w', 'W', 'ctrl+W',
                               'ctrl+c', 'cmd+c', 'ctrl+C', 'cmd+C', 'tab'):
            data = {'spectra': spectra,  'spectra_old': spectra_old,
                    'spectra_colors': spectra_colors,
                    'spectra_colors_old': spectra_colors_old,
                    'spectra_names': spectra_names, 'all_indices': all_indices,
                    'idx': idx, 'active_indices': active_indices,
                    'selected_points': selected_points, 'variable_y': variable_y,
                    'copied_data': copied_data, 'in_macro': in_macro, }
            data_log = data_log[:ilog+1] + [copy.deepcopy(data)]
            ilog += 1
        if using_joint_editing_mode and event.key in ('up', 'down'):
            pass
        else:
            action_log = action_log[:jlog+1] + [copy.deepcopy(action_info)]
            jlog += 1
    plot_data(spectra, spectra_old, active_indices, idx, using_manual_baseline_mode)
    if not using_manual_baseline_mode:
        plot_windows(selected_points, use_microns)
    else:
        plot_baseline_points(selected_points)
    if (click_action_options != {}
            and click_action_options['clicked_borders'] != []):
        plot_baseline_points(click_action_options['clicked_borders'])
    if in_macro and k == len(macro_actions):
        k = 0
        in_macro = False
        print('Macro finished.\n')
    num_actions_stored = len(data_log)
    if num_actions_stored > max_actions_stored:
        prev_num = num_actions_stored
        data_log = data_log[-max_actions_stored:]
        new_num = len(data_log)
        ilog -= (prev_num - new_num)
    plt.draw()

#%% Initialization.

print()
print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Interactive Toolkit')
print('-------------------')
print()

# Weights of AICE.
try:
    weights = np.load(weights_path, allow_pickle=True)
except:
    weights = None

# Reading of the arguments.
variable_y = 'absorbance'
use_optical_depth = False
input_in_microns = False
args = copy.copy(sys.argv)
i = 0
while i < len(args):
    arg = args[i]
    if arg == '-od':
        variable_y = 'absorbance'
        use_optical_depth = True
        del args[i]
    elif arg == '-T':
        variable_y = 'transmittance'
        del args[i]
    elif arg.startswith('-F'):
        units = 'a.u.' if arg == '-F' else arg.split('_')[1]
        variable_y = f'spectral flux density ({units})'
        del args[i]
    else:
        i += 1
filepaths = []
if len(args) > 1:
    for path in args[1:]:
        filepaths += [path]
else:
    text = input('Drag the input file(s): ')
    if text.endswith(' '):
        text = text[:-1]
    args = text.replace(r'\ ', r'\_').split(' ')
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '-od':
            variable_y = 'absorbance'
            use_optical_depth = True
            del args[i]
        if arg == '-T':
            variable_y = 'transmittance'
            del args[i]
        elif arg.startswith('-F'):
            units = 'a.u.' if arg == '-F' else arg.split('_')[1]
            variable_y = f'spectral flux density ({units})'
            del args[i]
        else:
            i += 1
    paths = args
    for path in args:
        path = path.replace(r'\_', r'\ ')
        filepaths += [path]
input_variable_y = copy.copy(variable_y)

#%% Reading of the data files.

# Identification of files.
filenames = np.array([path.split(sep)[-1] for path in filepaths])
folders = np.unique([sep.join(path.split(sep)[:-1]) for path in filepaths])
working_folder = folders[0] if folders[0] != '' else os.getcwd()
os.chdir(working_folder)
filepaths = []
for filename in filenames:
    filepaths += list([str(pp) for pp in pathlib.Path('.').glob(filename)])
name_sizes = np.array([len(name) for name in filenames])
filenames_sorted = []
for size in np.unique(name_sizes):
    mask = name_sizes == size
    filenames_sorted += sorted(filenames[mask])
filenames = [str(name) for name in filenames_sorted]
if len(filenames) == 0:
    print('Error: No files found.\n')
    sys.exit()
original_filenames = copy.copy(filenames)
use_csv_format_for_input = any([name.endswith('.csv') for name in filenames])

# Reading of files.
spectra, spectra_names = [], []
if use_csv_format_for_input:
    for filename in filenames:
        df = pd.read_csv(filename)
        data = df.values
        columns = list(df.columns)[1:]
        x, y = data[:,[0,1]].T
        if np.median(x) < 200.:
            input_in_microns = True
        if np.nanmedian(y) < 1e-10:
            variable_y = 'absorption coefficient (cm2)'
        if input_in_microns:
            x = 1e4 / x
        x, inds = np.unique(x, return_index=True)
        num_spectra = data.shape[1] - 1
        for i in range(num_spectra):
            name = filename.split(sep)[-1] + ' - ' + columns[i]
            y = data[inds,i+1]
            x_, y_, _ = fill_spectrum(x, y)
            spectrum = {'x': x_, 'y': y_, 'y-unc': None, 'edited': False}
            spectra += [spectrum]
            spectra_names += [name]
else:
    i = 0
    while i < len(filenames):
        filename = filenames[i]
        name = filename.split(sep)[-1]
        data = np.loadtxt(filename, comments=['#','%','!',';'])
        try:
            data = np.loadtxt(filename, comments=['#','%','!',';'])
        except:
            print(f'Error: File {filename} could not be opened.')
            del filenames[i]
            continue
        x, y = data[:,[0,1]].T
        y_unc = data[:,2] if data.shape[1] >= 3 else None
        if np.median(x) < 200.:
            input_in_microns = True
        if np.nanmedian(y) < 1e-10:
            variable_y = 'absorption coefficient (cm2)'
        if input_in_microns:
            x = 1e4 / x
        x, inds = np.unique(x, return_index=True)
        y = y[inds]
        if y_unc is not None:
            y_unc = y_unc[inds]
        x, y, y_unc = fill_spectrum(x, y, y_unc)
        spectrum = {'x': x, 'y': y, 'y-unc': y_unc, 'edited': False}
        spectra += [spectrum]
        spectra_names += [name]
        i += 1
    columns = []
    num_spectra = len(filenames)
    if num_spectra == 0:
        print()
        sys.exit()
        
# Setting of colors.
cmap_old = plt.colormaps[colormaps['original']['name']]
offset_old = colormaps['original']['offset']
scale_old = colormaps['original']['scale']
cmap_new = plt.colormaps[colormaps['edited']['name']]
offset_new = colormaps['edited']['offset']
scale_new = colormaps['edited']['scale']
spectra_colors, spectra_colors_old = [], []
for (i, name) in enumerate(spectra_names):
    if 'K' in name:
        text = name.replace(' K', 'K').split('K')[-2]
        temp = ''
        for (j, char) in enumerate(reversed(text)):
            if char.isnumeric():
                temp += char
            else:
                break
        temp = float(temp[::-1])
        value = max(0., temp - 10.) / 180.
    else:
        value = i / num_spectra
    spectra_colors += [cmap_new(offset_new + scale_new * value)]
    spectra_colors_old += [cmap_old(offset_old + scale_old * value)]

# Ranges and limits for plots.
yy = np.concatenate(tuple([spectrum['y'] for spectrum in spectra]))
xx = np.concatenate(tuple([spectrum['x'] for spectrum in spectra]))
mask = np.isfinite(yy)
x_mask = xx[mask]
x_min = x_mask.min()
x_max = x_mask.max()
xrange = x_max - x_min
margin = rel_margin_x * xrange
x_lims = [x_min - margin, x_max + margin]
y_lims = calculate_robust_ylims(spectra, x_lims, perc1=0.1, perc2=99.5,
                                rel_margin=1.5*rel_margin_y)

# Default noise level.
noise_level = 0.5 * compute_noise(spectra)

# Info file.
variable_x = 'wavelength' if input_in_microns else 'wavenumber' 
if use_csv_format_for_input:
    input_data_info = {'file': filename, 'columns': columns,
                       'x': variable_x, 'y': variable_y,
                       'number of spectra': num_spectra}
else:
    input_data_info = {'files': spectra_names, 'number of spectra': num_spectra,
                       'x': variable_x, 'y': variable_y}

#%% Loading the interactive mode.
    
instructions = \
"""
Instructions
------------
- Press Z or '+' to zoom, Left/Right to move through the spectrum, and Shift+Z
  or '<' or '-' to unzoom. Press Y to adapt the vertical range to display the
  spectra, or Ctrl+Y to adapt only to the selected edited spectrum. Press
  Up/Down to switch spectrum in case there is more than one.
- Left click to select a window edge, right click to undo or remove the window
  over the pointer, or Tab to remove all the windows.
- Press B to estimate the a baseline for the selected spectra in the current
  windows; alternatively, press '.' to manually select the baseline points
  clicking with the cursor and then press B. If you press Shift+B, you can
  write the smoothing parameter for the baseline estimation.
- Press R to reduce the spectra using the current baseline. If spectra are in
  flux, they will be divided by the baseline, converting them to transmittance
  (same as pressing T). In transmittance, spectra will also be divided by the
  baseline. In absorbance, the baseline will be subtracted from the spectra.
  In flux or transmittance, Shift+R will subtract the baseline instead of
  dividing by it, while in absorbance it will subtract the previously fitted
  lines in the current windows from the spectra.
- If spectra are in transmission, press A or D to convert to absorbance or
  optical depth. In absorbance or optical depth, press D or A to switch the
  scale, or press T to convert to transmission.
- Press M to switch between wavenumber and wavelength. Press L to switch
  between linear and logarithmic scale. If spectra are not in flux, press
  Ctrl+'-' to invert the vertical axis.
- Press S to smooth the data in the selected windows. If your press Shift+S,
  you can write the smoothing factor in the terminal.
- Press X to remove and interpolate the selected regions, or Shift+X to specify
  a smoothing factor the interpolation and apply it.
- Press N to add Gaussian noise in the selected windows, or Shift+N to specify
  the standard deviation and add it.
- Press 0 to set the selected windows to zero, or Shift+0 (or '=') to only do
  so for negative absorbance values; if using manual selection of points, this
  will automatically select a set of uniform zero points.
- If spectra are in transmission, press 1 to set the selected windows to one,
  or Shift+1 (or '!') to only do so for values greater than one.
- Press O to enter and add an offset to the spectra in the selected regions or
  Shift+O to add the current baseline as an offset.
- Press U to perform the union of all spectra into a single spectrum. Press
  Shift+U to perform the sum of all the spectra into a single spectrum.
- Press E to import a baseline from an external file.
- Press P to use AICE to predict the composition of the ice (in transmittance
  or absorbance / optical depth). If you press Shift+P, AICE will consider the
  previously fitted Gaussians instead of the spectrum where possible.
- Press G to fit a Gaussian in the selected windows, or Shift+G to do so fixing
  the line center. Adding Ctrl will allow for fitting extra Gaussian lines.
  Press Alt+Tab to remove the Gaussians in the current windows. Press I to
  integrate the selected spectrum in the current window, or Shift+I to
  introduce a band strength and integrate the column density. If you also press
  Ctrl, the integration will be performed over the previously fitted Gaussians.
- Press F to multiply the selected regions by the specified factor. Press
  Shift+F to convert from absorbance to absorption coefficient or viceversa
  (using the input column density), or from transmittance to spectral flux
  density (using the current baseline).
- Press ',' or Shift+',' (or ';') to resample the spectra to the given array
  (simple or precise method, respectively).
- If there is more than one spectrum, press J to activate the individual
  editing mode or to restore the joint editing mode.
- Press Ctrl+C or Cmd+C to copy the current spectrum in the selected region,
  and Ctrl+V or Cmd+V to paste and replace the copied data in the current
  selected spectrum. Press Shift+Ctrl+C or Shift+Cmc+C to copy the previously
  fitted Gaussians in the current windows instead, and then use Ctrl+V or Cmd+V
  to paste them to the current spectrum.
- Press Backspace to remove the selected regions from the current spectrum/
  spectra, or the whole selected spectrum if there are no selected regions.
- Press Shift+M to load a macro/algorithm to apply, or Control+M to use one of
  the predefined ones.
- Press Ctrl+Z to undo, or Ctrl+Shift+Z to redo.
- To save the files and exit, press Ctrl+Enter. To also save a record of the
  actions performed, press Shift+Enter. To just save the current baseline,
  press Alt+Enter or Cmd+Enter.
- To cancel and exit, press Esc or close the plot window.
- Before pressing any key to perform actions, make sure that the plot window
  is active (click on it if necessary).
"""

print(instructions)

spectra_old = copy.deepcopy(spectra)
num_spectra = len(spectra)
active_indices = list(range(num_spectra))
all_indices = copy.deepcopy(active_indices)
selected_points = []
using_joint_editing_mode = True
using_manual_baseline_mode = False
invert_yaxis = False
use_logscale = False
use_microns = False
in_macro = False
copied_data = None
waiting_for_click = False
click_action_options = {}
ilog, jlog, idx, k = 0, 0, 0, 0
save_action_log = True
old_x_lims = copy.copy(x_lims)
x_lims = [1e4/x_lims[1], 1e4/x_lims[0]] if use_microns else x_lims
data = {'spectra': spectra,  'spectra_old': spectra_old,
        'spectra_colors': spectra_colors, 'spectra_colors_old': spectra_colors_old,
        'spectra_names': spectra_names, 'all_indices': all_indices,
        'active_indices': active_indices, 'idx': idx,
        'selected_points': selected_points, 'variable_y': variable_y,
        'copied_data': copied_data, 'in_macro': False}
data_log = [copy.deepcopy(data)]
action_log = [{'action': 'start'}]
macro_actions = []

plt.figure('AICE Interactive Toolkit', figsize=(9.,5.))
plot_data(spectra, spectra_old, active_indices, idx, using_manual_baseline_mode)
fig = plt.gcf()
fig.canvas.mpl_connect('button_press_event', click1)
fig.canvas.mpl_connect('button_release_event', click2)
fig.canvas.mpl_connect('key_press_event', press_key)

plt.show()
print()
sys.exit()
