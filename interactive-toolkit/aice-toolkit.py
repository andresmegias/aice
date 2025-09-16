#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.0
------------------------------------------
Interactive Toolkit

Andrés Megías.
"""

# Files for AICE.
aice_labels = ['temp. (K)', 'H2O', 'CO', 'CO2', 'CH3OH', 'NH3', 'CH4']
weights_path = '/Users/andres/Proyectos/AICE/neural-networks/training/models/aice-weights.npy'
# Matplotlib backend.
backend = 'qtagg'
# possible values: ['gtk3agg', 'gtk3cairo', 'gtk4agg', 'gtk4cairo', 'qtagg',
#                   'qtcairo', 'qt5agg', 'qt5cairo', 'tkagg', 'tkcairo',
#                   'webagg', 'wx', 'wxagg', 'wxcairo', 'macosx']

# Libraries.
import os
import sys
import copy
import time
import yaml
import pathlib
import platform
import numpy as np
import pandas as pd
import richvalues as rv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
from matplotlib.backend_bases import MouseEvent, KeyEvent
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy.stats import median_abs_deviation
plt.matplotlib.use(backend)

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
    if interpolation == 'pchip':
        y = rv.rolling_function(np.mean, y, smooth_size)
    if windows is None:
        x_ = copy.copy(x)
        y_ = copy.copy(y)
    else:
        mask = np.zeros(len(x), bool)
        for (x1,x2) in windows:
            mask |= (x >= x1) & (x <= x2)
        x_ = x[mask]
        y_ = y[mask]
    y_ = np.nan_to_num(y_, nan=0.)
    
    if interpolation == 'pchip':
        spl = PchipInterpolator(x_, y_)
        yb = spl(x)
    elif interpolation == 'spline':
        y_s = rv.rolling_function(np.median, y_, smooth_size)
        s = sum((y_s-y_)**2)
        k = len(y_)-1 if len(y_) <= 3 else 3
        spl = UnivariateSpline(x_, y_, s=s, k=k)
        yb = spl(x)
    else:
        raise Exception('Wrong interpolation type.')
    return yb

def create_baseline(x, p):
    """
    Create a baseline from the input points.

    Parameters
    ----------
    x : array
        Data where to apply the baseline fit.
    p : list / array (2, N)
        Reference points for the baseline.

    Returns
    -------
    yb : array
        Resulting baseline.
    """ 
    x_, y_ = np.array(p).T
    inds = np.argsort(x_)
    x_ = x_[inds]
    y_ = y_[inds]
    spl = PchipInterpolator(x_, y_)
    yb = spl(x)
    return yb

def axis_conversion(x):
    """Axis conversion from wavenumber to wavelength and viceversa."""
    with np.errstate(divide='ignore'):
        y = 1e4 / x
    return y

def format_windows(selected_points):
    """Format the selected points into windows."""
    are_points_even = len(selected_points) % 2 == 0
    windows = selected_points[:] if are_points_even else selected_points[:-1]
    windows = np.array(windows).reshape(-1,2)
    for (i, x1x2) in enumerate(windows):
        x1, x2 = min(x1x2), max(x1x2)
        windows[i,:] = [x1, x2]
    return windows

def get_windows(mask, x):
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

def get_mask(windows, x):
    """Obtain a mask corresponding to the input windows on the array x"""
    mask = np.zeros(len(x), bool)
    for x1x2 in windows:
        x1, x2 = min(x1x2), max(x1x2)
        mask |= (x >= x1) & (x <= x2)
    return mask

def invert_windows(windows, x):
    """Obtain the complementary of the input windows for the array x."""
    mask = get_mask(windows, x)
    windows = get_windows(~mask, x)
    return windows

def calculate_ylims(spectra, x_lims, perc1=0., perc2=100., rel_margin=0.):
    """Calculate vertical limits for the given spectra."""
    yy = np.array([], float)
    for spectrum in spectra:
        x = spectrum['x']
        y = spectrum['y']
        x1, x2 = x_lims
        mask = (x >= x1) & (x <= x2) & np.isfinite(y)
        y = y[mask]
        yy = np.append(yy, y)
    yy = np.unique(yy)
    y1 = np.percentile(yy, perc1)
    y2 = np.percentile(yy, perc2)
    if rel_margin > 0.:
        yrange = y2 - y1
        margin = rel_margin * yrange
    else:
        margin = 0.
    y_lims = [y1 - margin, y2 + margin]
    return y_lims 

def calculate_robust_ylims(spectra, x_lims, perc1=1., perc2=99., rel_margin=0.):
    """Compute robust vertical limits for the given spectra."""
    yy = np.array([], float)
    for spectrum in spectra:
        x = spectrum['x']
        y = spectrum['y']
        x1, x2 = x_lims
        mask = (x >= x1) & (x <= x2) & np.isfinite(y)
        yy = np.append(yy, y[mask])
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

def aice_model(absorbance, weights):
    """
    Neural network model of AICE.
    
    Predict the composition and temperature of the input spectrum.
    The composition is given in terms of H2O, CO, CO2, CH3OH, NH3 and CH4.
    
    Parameters
    ----------
    absorbance : array (float)
        Absorbance points of the spectrum.
    weights : array (float)
        Weights of the neural network ensemble.
    
    Returns
    -------
    prediction_df : dataframe (float)
        Predictions for the temperature and molecular fractions-
    """
    relu = lambda x: np.maximum(0, x)
    def sigmoid(x):
        with np.errstate(all='ignore'):
            y = 1 / (1 + np.exp(-x))
        return y
    def nn_model(x, weights, end_act=relu):
        """Multi-layer perceptron."""
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
    results = []
    for j in range(weights.shape[0]):
        yj = np.zeros(weights.shape[1])
        for i in range(len(yj)):
            end_act = relu if i == 0 else sigmoid
            yj[i] = nn_model(absorbance, weights[j,i], end_act)[0]
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

# Predefined spectral windows. (Feel free to edit this.)
species_windows = {
    'H2O': [[3780, 2805], [2570, 2010], [1850, 1200], [1030, 480]],
    'CO': [[2165, 2120]],
    '13CO': [[2110,2075]],
    'CO2': [[3723, 3695], [3612, 3588], [2396, 2316], [687, 636]],
    'CH3OH': [[3560, 2712], [1178, 982]],
    'CH4': [[3103, 2968], [1334, 1274]],
    'NH3': [[3638, 2928], [1223, 976]],
    'H2CO': [[3040, 2780], [1780, 1690], [1530, 1470], [1280, 1150]],
    'C2H5NH2': [[3450, 3040], [3010, 2750], [1670, 1220], [1180, 800]]
}
species_windows['12CO'] = species_windows['CO']

# Predefined macros. (Feel free to edit this.)
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
            smoothing factor : 15
        - modify windows :
            windows: 'auto'
        - estimate baseline :
            smoothing factor: 31
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
            smoothing factor: 25
        """,
    'M3':
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
    
# Graphical options.
colors = {'edited': 'cornflowerblue', 'baseline': 'darkorange',
          'baseline-reference': 'chocolate'}
colormaps = {'original': {'name': 'brg', 'offset': 0.0, 'scale': 0.5},
             'edited': {'name': 'viridis', 'offset': 0.4, 'scale': 0.6}}
dlw = default_linewidth = 2.0
rel_margin_y = 0.04

# Default options.
baseline_smooth_size = 25
interp_smooth_size = 9
smooth_size = 5
sigma_threshold = 8.

# Removing default keymaps for interactive plot.
keymaps = ('back', 'copy', 'forward', 'fullscreen', 'grid', 'grid_minor',
           'help', 'home', 'pan', 'quit', 'quit_all', 'save', 'xscale',
           'yscale', 'zoom')
for keymap in keymaps:            
    plt.rcParams.update({'keymap.' + keymap: []})
    
# Folder separator.
sep = '\\' if platform.system() == 'Windows' else '/'

#%% Functions used in interactive mode.

def plot_data(spectra, spectra_old, idx, manual_mode=False):
    """Plot the input spectra."""
    plt.clf()
    for (i,spectrum_old) in enumerate(spectra_old):
        plt.plot(spectrum_old['x'], spectrum_old['y'], color='gray',
                 lw=dlw, alpha=0.1, zorder=2.4)
    plt.plot(spectra_old[idx]['x'], spectra_old[idx]['y'], color='black',
             lw=dlw, label='original spectrum', zorder=2.6) 
    spectrum = spectra[idx]
    if 'y-base' in spectrum:
        plt.plot(spectrum['x'], spectrum['y-base'], linestyle='--', lw=0.9*dlw,
                 zorder=2.8,  color=colors['baseline'], label='fitted baseline')
        if not np.array_equal(spectra_old[idx]['y'], spectrum['y-ref']):
            plt.plot(spectrum['x-ref'], spectrum['y-ref'], lw=0.8*dlw,
                     zorder=2.7, alpha=0.8,
                     color=colors['baseline-reference'] , label='baseline reference')
    if spectrum['edited']:
        plt.plot(spectrum['x'], spectrum['y'], color=colors['edited'],
                 lw=0.8*dlw, zorder=2.7, label='edited spectrum')
    cmap_old = plt.colormaps[colormaps['original']['name']]
    offset_old = colormaps['original']['offset']
    scale_old = colormaps['original']['scale']
    cmap_new = plt.colormaps[colormaps['edited']['name']]
    offset_new = colormaps['edited']['offset']
    scale_new = colormaps['edited']['scale']
    num_spectra = len(spectra)
    for (i,spectrum) in enumerate(spectra):
        if i == idx:
            continue
        color = (cmap_old(offset_old + scale_old * i/num_spectra)
                 if not spectrum['edited']
                 else cmap_new(offset_new + scale_new * i/num_spectra))
        plt.plot(spectrum['x'], spectrum['y'], color=color,
                 lw=0.8*dlw, alpha=0.2, zorder=2.5)
        if 'y-base' in spectrum:
            plt.plot(spectrum['x'], spectrum['y-base'], '--', lw=0.8*dlw,
                     color='darkorange', alpha=0.2, zorder=2.5)
    global x_min, x_max, x_lims, y_lims, logscale, invert_absorbance_yaxis
    ylabel = copy.copy(variable_y).replace('abs. coeff. (cm2)',
                                           'absorption coefficient (cm$^2$)')
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    if 'abs' in variable_y and invert_absorbance_yaxis:
        plt.gca().invert_yaxis()
    plt.axhline(y=0., color='black', lw=0.5)
    if variable_y == 'transmittance':
        plt.axhline(y=1., color='black', lw=0.5)
    plt.axvline(x=0., color='black', lw=0.5)
    plt.xlabel('wavenumber (cm$^{-1}$)', labelpad=6.)
    plt.ylabel(ylabel, labelpad=12.)
    plt.margins(x=0.01)
    if not manual_mode:
        plt.axvspan(0., 0., 0., 0., edgecolor='lightgray', facecolor='white',
                    alpha=1., label='windows')
    else:
        plt.plot([], '.', color=colors['baseline-reference'],
                 label='baseline reference')
    ax = plt.gca()
    if logscale:
        yy = np.concatenate([spectrum['y'] for spectrum in spectra])
        mask = yy > 0.
        linthresh = 10*np.min(np.abs(yy[mask]))
        plt.yscale('symlog', linthresh=linthresh)
        plt.ylim(bottom=max(-linthresh, np.min(yy)-0.2*linthresh))
        y1, y2 = plt.ylim()
        log_locator = LogLocator(base=10.0, subs='auto')
        minor_ticks = log_locator.tick_values(2*10*linthresh, y2)
        if y1 < -linthresh:
            minor_ticks_neg = -log_locator.tick_values(2*10*linthresh, abs(y1))
            minor_ticks = np.append(minor_ticks, minor_ticks_neg)
        minor_ticks_neg = -log_locator.tick_values(2*10*linthresh, -y1)
        minor_ticks = minor_ticks[(minor_ticks > y1) & (minor_ticks < y2)]
        ax.set_yticks(minor_ticks, minor=True)
    else:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    loc = ('upper left' if 'abs' in variable_y and not invert_absorbance_yaxis
           or 'flux' in variable_y and not logscale else 'lower left')
    plt.legend(loc=loc)
    plt.title(spectra_names[idx], fontweight='bold', pad=12.)
    ax = plt.gca()
    ax.invert_xaxis()
    ax2 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
    wavelength_ticks = [1.5, 1.7, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8,
                        10, 12, 15, 20, 30, 50, 300] # μm
    ax2.set_xticks(wavelength_ticks, wavelength_ticks)
    ax2.set_xlabel('wavelength (μm)', labelpad=6., fontsize=9.)
    plt.tight_layout()
    loc = ((0.98, 0.96) if 'abs' in variable_y and not invert_absorbance_yaxis
           or 'flux' in variable_y and not logscale else (0.98, 0.125))
    plt.text(*loc, '      AICE Interactive Toolkit' '\n ' ,
             ha='right', va='top', fontweight='bold', transform=ax.transAxes,
             bbox=dict(edgecolor=[0.8]*3, facecolor='white', alpha=0.7))
    plt.text(*loc, ' \n' 'check terminal for instructions',
             ha='right', va='top', transform=ax.transAxes)
    plt.gca().set_facecolor([0.9]*3)

def plot_windows(selected_points):
    """Plot the current selected windows."""
    for x in selected_points:
        plt.axvline(x, color='lightgray', alpha=1.)
    windows = format_windows(selected_points)
    for (x1,x2) in windows:
        plt.axvspan(x1, x2, transform=plt.gca().transAxes,
                    color='white', alpha=1.)

def plot_baseline_points(selected_points):
    """Plot the current selected baseline points."""
    if len(selected_points) > 0:
        x, y = np.array(selected_points).T
        plt.plot(x, y, '.', color=colors['baseline-reference'], zorder=2.8)

def estimate_baseline(spectra, indices, selected_points, smooth_size,
                      manual_mode, interpolation='spline'):
    """Estimate a baseline for the input spectra and the given parameters."""
    global x_min, x_max
    windows = format_windows(selected_points)
    if not manual_mode:
        if list(windows) == [] or list(windows) != [] and np.min(windows) > x_min:
            x1 = np.max([spectrum['x'][0] for spectrum in spectra])
            windows = np.append(windows, [[x1-1e-5, x1]], axis=0)
        if np.max(windows) < x_max:
            x2 = np.min([spectrum['x'][-1] for spectrum in spectra])
            windows = np.append(windows, [[x2, x2+1e-5]], axis=0)
        for i in indices:
            spectrum = spectra[i] 
            x = spectrum['x']
            y = spectrum['y']
            y_base = fit_baseline(x, y, smooth_size, windows, interpolation)
            spectra[i]['y-base'] = y_base
            spectra[i]['x-ref'] = copy.copy(x)
            spectra[i]['y-ref'] = copy.copy(y)
    else:
        for i in indices:
            spectrum = spectra[i] 
            x = spectrum['x']
            y = spectrum['y']
            y_base = create_baseline(x, selected_points)
            spectra[i]['y-base'] = y_base
            spectra[i]['x-ref'] = copy.copy(x)
            spectra[i]['y-ref'] = copy.copy(y)

def do_reduction(spectra, indices, variable_y):
    """Reduce the spectra subtracting the pre-computed existing baselines."""
    for i in indices:
        spectrum = spectra[i]
        if 'y-base' in spectrum:
            y = spectrum['y']
            y_base = spectrum['y-base']
            y_red = y - y_base if 'abs' in variable_y else y / y_base
            spectra[i]['edited'] = True
            spectra[i]['y'] = y_red
            if 'flux' in variable_y:
                del spectra[i]['y-base']
                del spectra[i]['x-ref']
                del spectra[i]['y-ref']

def do_removal(spectra, indices, selected_points, smooth_size):
    """Remove the selected regions of the data."""
    if selected_points == []:
        return
    windows = format_windows(selected_points)
    for i in indices: 
        spectrum = spectra[i]
        x = spectrum['x']
        y = spectrum['y']
        if 'y-base' in spectrum:
            y_base = spectrum['y-base']
            x_ref = spectrum['x-ref']
            y_ref = spectrum['y-ref']
        interpolate = False
        for (x1,x2) in windows:
            is_inferior_edge = x1 <= np.min(x)
            is_superior_edge = x2 >= np.max(x)
            if not (is_inferior_edge or is_superior_edge):
                interpolate = True
        if interpolate:
            windows_ = invert_windows(windows, x)
            y_base = fit_baseline(x, y, smooth_size, windows_,
                                  interpolation='pchip')
        for (x1,x2) in windows:
            is_inferior_edge = x1 <= np.min(x)
            is_superior_edge = x2 >= np.max(x)
            if not (is_inferior_edge or is_superior_edge):
                mask = (x >= x1) & (x <= x2)
                y[mask] = y_base[mask]
        for (x1,x2) in windows:
            is_inferior_edge = x1 <= np.min(x)
            is_superior_edge = x2 >= np.max(x)
            if is_inferior_edge or is_superior_edge:
                mask = (x >= x2 if is_inferior_edge else x <= x1)
                x = x[mask]
                y = y[mask]
                if 'y-base' in spectrum:
                    y_base = y_base[mask]
                    x_ref = x_ref[mask]
                    y_ref = y_ref[mask]
        spectra[i]['edited'] = True
        spectra[i]['x'] = x
        spectra[i]['y'] = y
        if 'y-base' in spectrum:
            spectra[i]['y-base'] = y_base
            spectra[i]['x-ref'] = x_ref
            spectra[i]['y-ref'] = y_ref

def do_smoothing(spectra, indices, selected_points, smooth_size,
                 function=np.mean):
    """Smooth the data in the selected regions.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    windows = format_windows(selected_points)
    for i in indices:
        spectrum = spectra[i] 
        x = spectrum['x']
        y = spectrum['y']
        for (x1,x2) in windows:
            mask = (x >= x1) & (x <= x2)
            y[mask] = rv.rolling_function(function, y[mask], smooth_size)
        spectra[i]['edited'] = True
        spectra[i]['y'] = y
        
def do_multiplication(spectra, indices, selected_points, factor):
    """Multiply the data in the selected regions with the input factor.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    windows = format_windows(selected_points)
    for i in indices:
        spectrum = spectra[i] 
        x = spectrum['x']
        y = spectrum['y']
        for (x1,x2) in windows:
            mask = (x >= x1) & (x <= x2)
            y[mask] = factor * y[mask]
        spectra[i]['edited'] = True
        spectra[i]['y'] = y
        
def set_zero(spectra, indices, selected_points, only_negatives=False):
    """Set selected region of the data to zero."""
    if selected_points == []:
        if only_negatives:
            global x_min, x_max
            selected_points = [x_max, x_min]
        else:
            print('Error: No points selected.')
            return
    windows = format_windows(selected_points)
    for i in indices:
        spectrum = spectra[i]
        x = spectrum['x']
        y = spectrum['y']
        for (x1,x2) in windows:
            mask = (x >= x1) & (x <= x2)
            if only_negatives:
                mask &= y < 0.
            y[mask] = 0.
        spectra[i]['edited'] = True
        spectra[i]['y'] = y
        
def set_one(spectra, indices, selected_points, only_gtr1=False):
    """Set selected region of the data to one."""
    if selected_points == []:
        if only_gtr1:
            global x_min, x_max
            selected_points = [x_max, x_min]
        else:
            print('Error: No points selected.')
            return
    windows = format_windows(selected_points)
    for i in indices:
        spectrum = spectra[i]
        x = spectrum['x']
        y = spectrum['y']
        for (x1,x2) in windows:
            mask = (x >= x1) & (x <= x2)
            if only_gtr1:
                mask &= y > 1.
            y[mask] = 1.
        spectra[i]['edited'] = True
        spectra[i]['y'] = y

def add_noise(spectra, indices, selected_points, noise_level):
    """Add noise to the selected regions.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    for i in indices:
        spectrum = spectra[i] 
        x = spectrum['x']
        y = spectrum['y']
        windows = format_windows(selected_points)
        for (x1,x2) in windows:
            mask = (x >= x1) & (x <= x2)
            noise = np.random.normal(0., scale=noise_level, size=sum(mask))
            y[mask] = y[mask] + noise
        spectra[i]['edited'] = True
        spectra[i]['y'] = y

def do_sigma_clip(spectra, indices, selected_points, threshold,
                  smooth_size=1, iters=3):
    """Apply a sigma clip in the selected regions.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    for i in indices:
        spectrum = spectra[i] 
        x = spectrum['x']
        y = spectrum['y']
        windows = format_windows(selected_points)
        for (x1,x2) in windows:
            mask = (x >= x1) & (x <= x2)
            for j in range(iters):
                limit = threshold * median_abs_deviation(y[mask])
                mask &= (np.abs(y) > limit)
            y[mask] = np.nan
        spectra[i]['edited'] = True
        spectra[i]['y'] = y
        
def integrate_spectrum(spectrum, selected_points, factor=1.):
    """Integrate the spectrum."""
    if selected_points == []:
        print('Error: No points selected.')
        return
    windows = format_windows(selected_points)
    x = spectrum['x']
    y = spectrum['y']
    area = 0.
    for (x1,x2) in windows:
        mask = (x >= x1) & (x <= x2)
        area += np.trapz(y[mask], x[mask])
    area /= factor
    return area
        
def do_resampling(spectra, indices, x_new):
    """Resample the spectra to the input wavenumber array.""" 
    for i in indices:
        spectrum = spectra[i] 
        x = spectrum['x']
        y = spectrum['y']
        y_new = np.interp(x_new, x, y)
        spectra[i]['edited'] = True
        spectra[i]['x'] = x_new
        spectra[i]['y'] = y_new
        
def merge_spectra(spectra):
    """Perform the union of all the input spectra into a single one."""
    x_new = np.array([], float)
    y_new = np.array([], float)
    for i in indices:
        spectrum = spectra[i]
        x = spectrum['x']
        y = spectrum['y']
        x_new = np.append(x_new, x)
        y_new = np.append(y_new, y)
    inds = np.argsort(x_new)
    x_new = x_new[inds]
    y_new = y_new[inds]
    new_spectrum = {'x': x_new, 'y': y_new, 'edited': True}
    return new_spectrum
        
def compute_absorbance(spectra, indices):
    """Convert the current spectra from transmittance to absorbance."""
    with np.errstate(invalid='ignore'):
        for i in indices:
            spectrum = spectra[i]
            y = spectrum['y']
            y_new = -np.log10(y)
            spectra[i]['edited'] = True
            spectra[i]['y'] = y_new
            if 'y-base' in spectrum:
                del spectra[i]['y-base']
            if 'x-ref' in spectrum:
                del spectra[i]['x-ref']
                del spectra[i]['y-ref']
        
def save_files(spectra, data_log, action_log, files, baseline=False):
    """Save processed spectra."""
    
    global columns
    nonedited_spectra = np.array([not spectrum['edited'] for spectrum in spectra])
    suffix = '-reduced' if not baseline else '-baseline'
    intensity_variable = 'y' if not baseline else 'y-base'
    if all(nonedited_spectra):
        print('\n''No editing of the spectra was performed.\n')
        suffix = ''
    elif any(nonedited_spectra):
        print('\n''Not all spectra have been edited.\n')
    extra_text = ' (baseline)' if baseline else ''
    filename = input(f'Output filename{extra_text}: ')
    ext = filename.split('.')[-1]
    if ext in ('.csv', ''):
        use_table_format_for_output = True
    elif ext in ('.txt', '.dat'):
        use_table_format_for_output = False
    else:
        print('Warning: Not supported extension, will be replaced by .csv.')
        use_table_format_for_output = True
    all_indices = data_log[-1]['all_indices']
    if use_table_format_for_output:
        if use_table_format_for_input:
            columns = np.array(columns)[all_indices].tolist()
            default_filename = files[0].replace('.csv', '')
        else:
            columns = np.array(spectra_names)[all_indices].tolist()
            columns = ['.'.join(column.split('.')[:-1]) for column in columns]
            default_filename = os.path.commonprefix(files)
        if input_variable_y == 'transmittance' and variable_y != 'transmittance':
            for (i,col) in enumerate(columns):
                columns[i] = col.replace('transm.', 'abs.')
                if variable_y == 'abs. coeff. (cm2)':
                    columns[i] = col.replace('abs.', 'abs. coeff. (cm2)')
        if variable_y == 'abs. coeff. (cm2)':
            for (i,col) in enumerate(columns):
                columns[i] = col.replace('abs.', 'abs. coeff. (cm2)')
        filename = filename.replace('*', default_filename)
        if filename == '':
            filename = default_filename + suffix
        else:
            filename = folder + filename
        if not filename.endswith('.csv'):
            filename += '.csv'
        xx = [spectrum['x'] for spectrum in spectra]
        i = np.argmax([len(x) for x in xx])
        x = xx[i]
        new_df = pd.DataFrame({'wavenumber (/cm)': x})
        for (i,spectrum) in enumerate(spectra):
            column = columns[i]
            x_i = spectrum['x']
            y_i = spectrum[intensity_variable]
            y_i = np.interp(x, x_i, y_i)
            new_df[column] = y_i
        nd = max([len(xi.split('.')[-1]) if '.' in xi else 0
                  for xi in x.astype(str)])
        new_df['wavenumber (/cm)'] = new_df['wavenumber (/cm)'].map(
                                                lambda x: '{:.{}f}'.format(x, nd))
        new_df.to_csv(filename, index=False, float_format='%.3e')
        num_spectra = len(spectra)
        saved_var = 'spectrum' if not baseline else 'baseline'
        if num_spectra > 1:
            saved_var = (saved_var.replace('spectrum', 'spectra')
                         .replace('baseline', 'baselines'))
        print('\n''Saved {} {} in {}'.format(num_spectra, saved_var, filename))
        output_data_info = {'file': filename.split(sep)[-1], 'columns': columns,
                            'number of spectra': len(spectra)}
    else: 
        if use_table_format_for_input:
            files = [files[0].replace('.csv','-')
                     + col.replace(' ','').replace('abs.','').replace('.','')
                     + '.txt' for col in columns]
        else:
            files = np.array(files)[all_indices].tolist()
        output_files = ['.'.join(file.split('.')[:-1]) + suffix + ext
                        for file in files]
        prefix = filename
        for (i,file) in enumerate(output_files):
            filename = prefix + file
            spectrum = spectra[i]
            x = spectrum['x']
            y = spectrum['y']
            data = np.array([x, y]).T
            inds = np.argsort(x)
            data = data[inds]
            nd = max([len(xi.split('.')[-1]) if '.' in xi else 0
                      for xi in x.astype(str)])
            np.savetxt(filename, data, fmt='%.{}f %.3e'.format(nd),
                       header=f"wavenumber_(/cm) {variable_y.replace(' ','_')}")
            saved_var = 'spectrum' if not baseline else 'baseline'
            print('Saved {} in {}.'.format(saved_var, filename))
        output_files = [file.split(sep)[-1] for file in output_files]
        output_data_info = {'files': output_files, 'number of spectra': len(spectra)}
            
    return output_data_info, filename

def save_action_record(action_log, files, spectra, filename, output_data_info):
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
    ext = filename.split('.')[-1]
    if len(spectra) > 1 and not filename.endswith('.csv'):
        log_file = os.path.commonprefix(files).replace(ext,'') + '-log.txt'
        if log_file == ' ':
            log_file = '.'.join(files[0].split('.')[:-1]) + '-log.txt'
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
        global click_time
        elapsed_click_time = time.time() - click_time
        x = event.xdata
        if (elapsed_click_time > 0.5  # s
                or x is None or x is not None and not np.isfinite(x)):
            return 
        global spectra, spectra_old, selected_points, manual_mode, idx
        global action_log, jlog
        if manual_mode:
            if button in ('left', '1'):
                y = event.ydata
                selected_points += [[x, y]]
                plt.plot(x, y, '.', color=colors['baseline-reference'], zorder=2.8)
            else:
                if len(selected_points) == 0:
                    return
                xp, yp = np.array(selected_points).T
                i = np.argmin(np.abs(xp - x))
                del selected_points[i]
                plot_data(spectra, spectra_old, idx, manual_mode)
                plot_baseline_points(selected_points)
        else:
            if button in ('left', '1'):
                selected_points += [x]
                plt.axvline(x, color='lightgray', alpha=1.)
                are_points_even = len(selected_points) % 2 == 0
                if are_points_even:
                    x1, x2 = selected_points[-2:]
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
                plot_data(spectra, spectra_old, idx)
                plot_windows(selected_points)
        if not manual_mode:
            windows = []
            for x1x2 in format_windows(selected_points):
                x1, x2 = min(x1x2), max(x1x2)
                windows += ['({:.2f}, {:.2f})'.format(x2, x1)]
            action_info = {'action': 'modify windows', 'windows': windows}
        else:
            points = []
            for (x,y) in selected_points:
                points += ['({:.2f}, {:.4g})'.format(x, y)]
            action_info = {'action': 'modify points', 'points': points}
        action_log = action_log[:jlog+1] + [copy.deepcopy(action_info)]
        jlog += 1
        plt.draw()    

def press_key(event):
    """Interact with the plot when pressing a key."""
    if type(event) is not KeyEvent:
        pass
    global files, spectra, data_log, action_log, ilog, jlog, variable_y
    global spectra_old, indices, idx, all_indices, spectra_names
    global baseline_smooth_size, interp_smooth_size, smooth_size, noise_level
    global individual_mode, manual_mode
    global selected_points, x_lims, y_lims, old_x_lims, rel_margin_y
    global logscale, invert_absorbance_yaxis
    global in_macro, k, macro_actions
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
                elif action == 'remove' or 'interpolate' in action:
                    event.key = 'x'
                    factor = params['smoothing factor']
                elif action in ('smooth', 'smooth (median)'):
                    event.key = 's' if action == 'smooth' else 'ctrl+s'
                    factor = params['smoothing factor']
                elif action == 'add noise':
                    event.key == 'n'
                    factor = params['smoothing factor']
                elif action == 'sigma-clip':
                    event.key = 'c'
                    factor = params['threhsold']
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
                elif action == 'resample':
                    event.key = '-'
                    text = params['wavenumber array (start, end, step)']
                elif action.startswith('change to') and action.endswith('scale'):
                    event.key = 'l'
                else:
                    print(f'Unknown action: {action}.')
            k += 1
    elif event.key in ('shift+enter', 'ctrl+enter', 'escape'):
        if 'enter' in event.key:
            data_log = data_log[:ilog+1]
            action_log = action_log[:jlog+1]
            output_data_info, filename = save_files(spectra, data_log,
                                                    action_log, files)
            if 'ctrl' in event.key:
                save_files(spectra, data_log, action_log, files, baseline=True)
            save_action_record(action_log, files, spectra, filename,
                               output_data_info)
        plt.close(1)
        return
    x_lims = list(reversed(plt.xlim()))
    y_lims = sorted(plt.ylim())
    if individual_mode and len(indices) == 1:
        individual_mode = False
    if individual_mode:
        indices = [idx]
    if event.key in ('y', 'Y', 'ctrl+y'):
        prev_y_lims = copy.copy(y_lims)
        if event.key == 'y':
            ylims1 = calculate_ylims(spectra, x_lims, rel_margin_y)
            ylims2 = calculate_ylims(spectra_old, x_lims, rel_margin_y)
            y_lims = [min(ylims1[0], ylims2[0]), max(ylims1[1], ylims2[1])]
        elif event.key == 'Y':
            ylims1 = calculate_ylims([spectra[idx]], x_lims, rel_margin_y)
            ylims2 = calculate_ylims([spectra_old[idx]], x_lims, rel_margin_y)
            y_lims = [min(ylims1[0], ylims2[0]), max(ylims1[1], ylims2[1])]
        else:
            y_lims = calculate_ylims([spectra[idx]], x_lims, rel_margin_y)
        if y_lims == prev_y_lims:
            y_lims = calculate_robust_ylims(spectra, x_lims, 1., 99., rel_margin_y)
    elif event.key == '-':
        if 'abs' in variable_y:
            invert_absorbance_yaxis = not invert_absorbance_yaxis
    elif event.key in ('z', 'Z', '<', 'left', 'right'):
        x_range = x_lims[1] - x_lims[0]
        if event.key in ('z', 'Z', '<'):
            if event.key == 'z':
                x_lims = [x_lims[0] + x_range/6, x_lims[1] - x_range/6]
            else:
                x_lims = [x_lims[1] - 1.5*x_range, x_lims[1] + 1.5*x_range]
        else:
            if event.key == 'left' and x_lims[1] != old_x_lims[1]:
                x_lims = [x_lims[0] + x_range/2, x_lims[1] + x_range/2]
            elif event.key == 'right' and x_lims[0] != old_x_lims[0]:
                x_lims = [x_lims[0] - x_range/2, x_lims[1] - x_range/2]
        x_lims = [max(old_x_lims[0], x_lims[0]),
                  min(old_x_lims[1], x_lims[1])]
    elif event.key in ('up', 'down'):
        if len(spectra) > 1:
            idx = idx+1 if event.key == 'up' else idx-1
            idx = idx % len(spectra)
        action_info = {'action': 'switch spectrum', 'spectrum': spectra_names[idx]}
    elif event.key in ('tab', '\t'):
        selected_points = []
        if not manual_mode:
            action_info = {'action': 'modify windows', 'windows': []}
        else:
            action_info = {'action': 'modify points', 'points': []}
    elif event.key == 'backspace':
        if len(all_indices) > 1:
            action_info = {'action': 'delete spectrum',
                           'deleted spectrum': spectra_names[idx]}
            del all_indices[idx]
            if not individual_mode:
                del indices[idx]
            del spectra[idx]
            del spectra_old[idx]
            del spectra_names[idx]
            idx = max(0, idx-1)
            indices = list(range(len(spectra)))
            if len(spectra) > 1 and manual_mode:
                action_info['current selected spectrum': spectra_names[idx]]
    elif event.key in ('w', 'W', 'ctrl+w'):
        if event.key == 'ctrl+w':
            text = input('Write new window (/cm): ')
            text = (text.replace('(','').replace(')','')
                    .replace('[','').replace(']','').replace(', ',','))
            x1, x2 = text.split(',')
            x1x2 = [float(x1), float(x2)]
            x1, x2 = min(x1x2), max(x1x2)
            selected_points += [x1, x2]
        else:
            if not in_macro or in_macro and windows == 'auto':
                text = (files[idx] if event.key == 'w'
                        else input('Write species to mask: '))
                species_list = parse_composition(text, folder)
                x = spectra[idx]['x']
                if len(selected_points) == 0:
                    mask = np.zeros(len(x), bool)
                else:
                    previous_windows = format_windows(selected_points)
                    mask = ~get_mask(previous_windows, x)
                for species in species_list:
                    if species in species_windows:
                        for x1x2 in species_windows[species]:
                            x1, x2 = min(x1x2), max(x1x2)
                            mask |= (x >= x1) & (x <= x2)
                windows = get_windows(~mask, x)
                selected_points = list(np.array(windows).flatten()[::-1])
        windows_text = []
        for x1x2 in format_windows(selected_points):
            x1, x2 = min(x1x2), max(x1x2)
            windows_text += ['({:.2f}, {:.2f})'.format(x2, x1)]
        print('Modified windows.')
        action_info = {'action': 'modify windows', 'windows': windows_text}
    elif event.key in ('s', 'S', 'ctrl+s', 'ctrl+S'):
        if not in_macro:
            factor = copy.copy(smooth_size)
        if 'S' in event.key:
            text = input('- Enter smoothing factor: ')
            factor = ''.join([char for char in text if char.isdigit()])
            factor = copy.copy(smooth_size) if factor == '' else int(factor)
        function = np.median if 'ctrl' in event.key else np.mean
        do_smoothing(spectra, indices, selected_points, factor, function)
        print(f'Smoothed regions in current windows with smoothing factor {factor}.')
        action_info = {'action': 'smooth', 'smoothing factor': factor}
        if 'ctrl' in event.key:
            action_info['action'] = 'smooth (median)'
    elif event.key in ('x', 'X'):
        if selected_points == []:
            print('Error: No points selected.')
        else:
            if not in_macro:
                factor = copy.copy(interp_smooth_size)
            if event.key == 'X':
                text = input('- Enter smoothing factor for interpolation: ')
                factor = ''.join([char for char in text if char.isdigit()])
                factor = (copy.copy(interp_smooth_size) if factor == ''
                          else int(factor))
            do_removal(spectra, indices, selected_points, factor)
            print('Interpolated/removed regions in current windows with'
                  f' smoothing factor {factor}.')
            action_info = {'action': 'remove/interpolate',
                           'smoothing factor': factor}
    elif event.key in ('n', 'N'):
        if not in_macro:
            factor = copy.copy(noise_level)
        if event.key == 'N':
            factor = input('- Enter noise level: ')
            try:
                factor = float(text)
            except:
                factor = 0.
        if factor != 0.:
            add_noise(spectra, indices, selected_points, factor)
            action_info = {'action': 'add noise', 'noise level': f'{factor:.3e}'}
    elif event.key in ('b', 'B'):
        if not in_macro:
            factor = copy.copy(baseline_smooth_size)
        if event.key == 'B' and not manual_mode:
            text = input('- Enter smoothing factor for baseline: ')
            factor = ''.join([char for char in text if char.isdigit()])
            factor == 1 if factor == '' else int(factor)
        estimate_baseline(spectra, indices, selected_points, factor,  manual_mode)
        if not manual_mode:
            print(f'Computed baseline from windows with smoothing factor {factor}.')
        else:
            print('Computed baseline from reference points.')
            factor = 1
        action_info = {'action': 'estimate baseline', 'smoothing factor': factor}
    elif event.key == 'r':
        baselines_in_spectra = ['y-base' in spectrum for spectrum in spectra]
        if not all(baselines_in_spectra):
            print('Warning: There is no baseline to subtract.'
                  ' Press B to compute the baseline.')
        elif ('abs' not in variable_y and (not all(baselines_in_spectra)
                                           or individual_mode)):
            print('Warning: If working in flux or transmittance, baselines'
                  ' must be computed for all the spectra and subtracted all at'
                  ' once. Make sure you compute all the baselines (B) and you'
                  ' are working in the joint mode (J) before trying to reduce.')
        else:
            do_reduction(spectra, indices, variable_y)
            print('Reduced selected spectra.')
            if 'flux' in variable_y:
                variable_y = 'transmittance'
                spectra_old = copy.deepcopy(spectra)
                y_lims = calculate_robust_ylims(spectra, x_lims, rel_margin_y)
            action_info = {'action': 'reduce'}
    elif event.key in ('0', '='):
        if not manual_mode:
            if event.key == '0':
                only_negatives = False
                action_info = {'action': 'set to zero'}
            else:
                only_negatives = True
                action_info = {'action': 'set negatives to zero'}
            set_zero(spectra, indices, selected_points, only_negatives)
            if only_negatives:
                print('Negative values of region in current windows set to 0.')
            else:
                print('Region in current windows set to 0.')
        else:
            points = []
            for (x,y) in selected_points:
                points += ['({:.2f}, {:.4g})'.format(x, y)]
            action_info = {'action': 'modify points', 'points': points}
            for x in np.linspace(x_min, x_max, 40):
                selected_points += [[x, 0.]] 
    elif event.key in ('1', '!'):
        if variable_y == 'transmittance':
            if not manual_mode:
                if event.key == '1':
                    only_gtr1 = False
                    action_info = {'action': 'set to one'}
                else:
                    only_gtr1 = True
                    action_info = {'action': 'set one as maximum'}
                set_one(spectra, indices, selected_points, only_gtr1)
                if only_gtr1:
                    print('Values greater than 1 of region in current windows set to 1.')
                else:
                    print('Region in current windows set to 1.')
            else:
                points = []
                for (x,y) in selected_points:
                    points += ['({:.2f}, {:.4g})'.format(x, y)]
                action_info = {'action': 'modify points', 'points': points}
                for x in np.linspace(x_min, x_max, 40):
                    selected_points += [[x, 0.]] 
    elif event.key in ('c', 'C'):
        if not in_macro:
            factor = copy.copy(sigma_threshold)
        if event.key == 'C':
            factor = input('- Enter a sigma threshold: ')
            factor = copy.copy(sigma_threshold) if factor == '' else float(factor)
        do_sigma_clip(spectra, indices, selected_points, factor)
        print(f'Sigma clipping applied in present windows with {factor:.1f}-sigma.')
        action_info = {'action': 'sigma-clip', 'threshold': factor}
    elif event.key in ('i', 'I'):
        if event.key == 'i':
            factor = 1.
            area_variable = 'area'
            area_units = '/cm'
        else:
            text = input('- Introduce the band strength (cm): ')
            try:
               factor =  float(text)
            except:
                factor = None
            area_variable = 'column density'
            area_units = '/cm2'
        if factor is not None:
            area = integrate_spectrum(spectra[idx], selected_points, factor)
            print('Integrated {}: {:.2e} {}'.format(area_variable, area, area_units))
    elif event.key in ('f', 'F'):
        if event.key == 'F' and variable_y != 'absorbance':
            print('Warning: Spectrum must be in absorbance in order to'
                  ' convert to absorption coefficient.')     
        elif event.key == 'F' and individual_mode:
            print('Warning: Change to joint mode (J) to convert to absorption'
                  ' coefficient.')
        if event.key == 'f':
            text = input('- Enter factor to multiply: ')
            if not in_macro:
                factor = 1. if text == '' else float(factor)
            do_multiplication(spectra, indices, selected_points, factor)
            if factor != 1.:
                action_info = {'action': 'multiply', 'factor': factor}
        else:
            if not in_macro:
                text = input('- Enter ice column density (/cm2) to convert'
                             ' to absorption coefficient: ')
                try:
                    coldens = float(text)
                except:
                    coldens = None
            if coldens is not None:
                factor = 1 / coldens
                variable_y = 'abs. coeff. (cm2)'
                do_multiplication(spectra, indices, [], factor)
                spectra_old = copy.deepcopy(spectra)
                y_lims = calculate_ylims(spectra, x_lims, rel_margin_y)
                action_info = {'action': 'convert to absorption coefficient',
                               'column density (/cm2)': coldens}
    elif event.key in ('u', 'U'):
        if event.key == 'u':
            new_name = os.path.commonprefix(spectra_names)
            if new_name[-1] in ('-', '_', ' '):
                new_name = new_name[:-1]
        if event.key == 'U' or new_name == '':
            new_name = input('Write a new name for the joint spectrum: ')
        if len(indices) > 1:
            new_spectrum = merge_spectra(spectra)
            spectra = [new_spectrum]
            spectra_old = copy.deepcopy(spectra)
            spectra_names = [new_name]
            indices = [0]
            all_indices = [0]
            idx = 0
            print('Performed union of all spectra.')
            action_info = {'action': 'union of spectra'}
    elif event.key == 'p':
        global weights, weights_path
        if weights is None:
            print('Could not find AICE weights in {}.'.format(weights_path))
        elif variable_y == 'flux':
            print('Cannot use AICE with flux. Convert first to transmittance'
                  ' or absorbance.')
        else:
            spectrum = spectra[idx]
            x_ = np.arange(980., 4001., 1.)
            x = spectrum['x']
            if 'abs' in variable_y:   
                y = spectrum['y']
            else:
                y = 10**spectrum['y']
            y_ = np.interp(x_, x, y)
            y_ = np.nan_to_num(y_, nan=0.)
            y_ /= np.nanmean(y_)
            predictions_df = aice_model(y_, weights)
            print(predictions_df)
    elif event.key == ',':
        if not in_macro:
            text = input('- Enter new wavenumber array to resample'
                         ' (initial wavenumber, final wavenumber, step): ')
        if text != '':
            text = (text.replace('(','').replace(')','')
                    .replace('[','').replace(']','').replace(' ',''))
            params = text.split(',')
            x1x2 = np.array(params[:2], float)
            step = float(params[2])
            x1, x2 = min(x1x2), max(x1x2)
            new_wavenumber = np.arange(x1, x2, step)
            do_resampling(spectra, indices, new_wavenumber)
            action_info = {'action': 'resample',
                'wavenumber array (start, end, step)': list(params)}
    elif event.key == 'a':
        if variable_y != 'transmittance':
            print('Warning: Spectra must be in transmittance to convert to'
                  ' absorbance.')
        elif individual_mode:
            print('Warning: Change to joint mode (J) to convert to absorbance.')
        else:
            compute_absorbance(spectra, indices)
            variable_y = 'absorbance'
            logscale = False
            y_lims = calculate_ylims(spectra, x_lims, rel_margin_y)
            spectra_old = copy.deepcopy(spectra)
            y_lims = calculate_ylims(spectra, x_lims, rel_margin_y)
            print('Converted to absorbance.')
            action_info = {'action': 'convert to absorbance'}
    elif event.key in ('j'):
        individual_mode = not individual_mode
        if individual_mode:
            indices = [idx]
            print('Processing mode has changed to individual.')
            action_info = {'action': 'activate individual processing mode',
                           'spectrum': spectra_names[idx]}
        else:
            indices = list(range(len(spectra)))
            print('Processing mode has changed to joint.')
            action_info = {'action': 'activate joint processing mode'}
    elif event.key == '.':
        manual_mode = not manual_mode
        if len(spectra) > 1:
            if indices != [idx]:
                indices = [idx]
                extra_msg = ' (only for selected spectrum)' 
            else:
                indices = list(range(len(spectra)))
                extra_msg = ' (for all spectra)'
        else:
            extra_msg = ''
        selected_points = []
        if manual_mode:
            individual_mode = True
            print('Using manual baseline mode.' + extra_msg)
            action_info = {'action': 'activate manual baseline mode',
                           'spectrum': spectra_names[idx]}
        else:
            print('Using windows baseline mode.' + extra_msg)
            action_info = {'action': 'activate windows baseline mode'}
    elif event.key == 'l':
        if 'abs' not in variable_y:
            if not logscale:
                logscale = True
                action_info = {'action': 'change to logarithmic scale'}
            else:
                logscale = False
                action_info = {'action': 'change to linear scale'}
    elif event.key in ('m', 'M') and not in_macro:
        if event.key == 'm':
            name = input('Drag the macro file: ')
            if name != '':
                if name.endswith(' '):
                    name = name[:-1]
                with open(name, 'r') as file:
                    macro_actions = yaml.safe_load(file)
                name = name.split(sep)[-1]
        else:
            name = input('Write the name of the predefined macro: ')
            if name != '':
                macro_actions = yaml.safe_load(predefined_macros[name])
        if name != '':
            print()
            print(f'Starting macro {name}.')
            print('Please, go back to the plot window and click on it.'
                  ' Press Enter to apply next action, Space to skip or'
                  ' Escape to exit the macro.')
            k = 0
            in_macro = True
    elif event.key in ('ctrl+z', 'cmd+z', 'ctrl+Z', 'cmd+Z','ctrl+<', 'cmd+<'):
        if ('z' in event.key and ilog == 0 
                or 'z' not in event.key and ilog == len(data_log)-1):
            pass
        else:
            recalculate_ylims = False
            prev_variable_y = copy.copy(variable_y)
            if 'z' in event.key and prev_variable_y == 'abs. coeff. (cm2)':
                recalculate_ylims = True
            in_macro_prev = data_log[ilog]['in_macro']
            ilog = (max(0, ilog-1) if 'z' in event.key
                    else min(len(data_log)-1, ilog+1))
            data = copy.deepcopy(data_log[ilog])
            spectra = data['spectra']
            spectra_old = data['spectra_old']
            idx = data['idx']
            indices = data['indices']
            all_indices = data['all_indices']
            spectra_names = data['spectra_names']
            variable_y = data['variable_y']
            if in_macro or in_macro_prev:
                selected_points = data['selected_points']
            in_macro = data['in_macro']
            jlog = (max(0, jlog-1) if 'z' in event.key
                    else min(len(action_log)-1, jlog+1))
            action_info = copy.deepcopy(action_log[jlog-1])
            if prev_variable_y == 'transmittance' and 'flux' in variable_y:
                recalculate_ylims = True
            elif 'flux' in prev_variable_y and variable_y == 'transmittance':
                recalculate_ylims = True
            if 'z' not in event.key and variable_y == 'abs. coeff. (cm2)':
                recalculate_ylims = True
            if recalculate_ylims:
                y_lims = calculate_robust_ylims(spectra, x_lims, rel_margin_y)
            if 'z' in event.key:
                print('Action undone.')
            else:
                print('Action redone.')
    if in_macro and k < len(macro_actions):
        action_text = (macro_actions[k] if type(macro_actions[k]) is str
                       else str(macro_actions[k])[1:-1].replace("'",""))
        print(f'Next action: {action_text}')
    if event.key in ('s', 'S', 'ctrl+s', 'ctrl+S', 'x', 'X', 'n', 'N', 'c', 'C',
                     'f', 'F', '0', '=', '1', '!', 'a', 'w' 'W', 'ctrl+W', 'l',
                     'b', 'B', 'r', 'u', 'U', ',', '.', 'J', 'up', 'down', 'tab',
                     'backspace'):
        if action_info is None:
            pass
        elif event.key not in ('.', 'i', 'J', 'l', 'up', 'down', 'w', 'tab'):
            data = {'spectra': spectra, 'idx': idx, 'indices': indices,
                    'spectra_old': spectra_old,
                    'all_indices': all_indices, 'variable_y': variable_y,
                    'spectra_names': spectra_names, 'in_macro': in_macro,
                    'selected_points': selected_points}
            data_log = data_log[:ilog+1] + [copy.deepcopy(data)]
            ilog += 1
        if (not individual_mode and event.key in ('up', 'down')
            or action_info is None):
            pass
        else:
            action_log = action_log[:jlog+1] + [copy.deepcopy(action_info)]
            jlog += 1
    plot_data(spectra, spectra_old, idx, manual_mode)
    if not manual_mode:
        plot_windows(selected_points)
    else:
        plot_baseline_points(selected_points)
    if in_macro and k == len(macro_actions):
        k = 0
        in_macro = False
        print('Macro finished.\n')
    plt.draw()

#%% Initialization.

print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('Interactive Toolkit')
print()

# Weights of AICE.
try:
    weights = np.load(weights_path, allow_pickle=True)
except:
    weights = None

# Reading of the arguments.
variable_y = 'absorbance'
input_in_microns = False
args = copy.copy(sys.argv)
i = 0
while i < len(args):
    arg = args[i]
    if arg == '-T':
        variable_y = 'transmittance'
        del args[i]
    elif arg.startswith('-F') or arg.startswith('-sF'):
        units = 'a.u.' if arg == '-F' else arg.split('_')[1]
        variable_y = f'flux ({units})'
        if arg.startswith('sF') or 'Jy' in arg:
            variable_y = 'spectral ' + variable_y
        del args[i]
    elif arg == '-m':
        input_in_microns = True
        del args[i]
    else:
        i += 1
file_paths = []
if len(args) > 1:
    for arg in args[1:]:
        file_paths += [arg]
else:
    text = input('Drag the input file(s): ')
    print()
    if text.endswith(' '):
        text = text[:-1]
    text_ = text.replace(r'\ ', r'\_')
    paths_ = text_.split(' ')
    for path_ in paths_:
        path = path_.replace(r'\_', r'\ ')
        folder = sep.join(path.split(sep)[:-1]).replace('\\', '')
        if os.getcwd() != folder:
            os.chdir(folder)
        filename = path.split(sep)[-1].replace('\\', '')
        file_paths += [filename]
input_variable_y = copy.copy(variable_y)
invert_absorbance_yaxis = True if 'abs' not in variable_y else False

#%% Reading of the data files.

# Identification of files.
files = []
for path in file_paths:
    files += list([str(pp) for pp in pathlib.Path('.').glob(path)])
files = np.array(files)
name_sizes = np.array([len(file) for file in files])
files_sorted = []
for size in np.unique(name_sizes):
    mask = name_sizes == size
    files_sorted += sorted(files[mask])
files = [str(file) for file in files_sorted]
if len(files) == 0:
    print('No files found.\n')
    sys.exit()
use_table_format_for_input = any([file.endswith('.csv') for file in files])

# Reading of files.
spectra, spectra_names = [], []
if use_table_format_for_input:
    if len(files) > 1:
        raise Exception('Too many files. Only one file can be read at a time'
                        ' if using table format.')
    file = files[0]
    df = pd.read_csv(file)
    data = df.values
    columns = list(df.columns)[1:]
    x = data[:,0]
    if np.median(x) < 200.:
        input_in_microns = True
    if input_in_microns:
        x = 1e4 / x
    inds = np.argsort(x)
    num_spectra = data.shape[1] - 1
    for i in range(num_spectra):
        name = file.split(sep)[-1] + ' - ' + columns[i]
        y = data[inds,i+1]
        spectrum = {'x': x, 'y': y, 'edited': False}
        spectra += [spectrum]
        spectra_names += [name]
else:
    i = 0
    while i < len(files):
        file = files[i]
        name = file.split(sep)[-1]
        try:
            data = np.loadtxt(file, comments=['#','%','!',';'])
        except:
            print('Warning: File {} could not be opened.'.format(file))
            del files[i]
            continue
        x, y = data[:,[0,1]].T
        if np.median(x) < 200.:
            input_in_microns = True
        if input_in_microns:
            x = 1e4 / x
        inds = np.argsort(x)
        x = x[inds]
        y = y[inds]
        spectrum = {'x': x, 'y': y, 'edited': False}
        spectra += [spectrum]
        spectra_names += [name]
        i += 1
    columns = []
    num_spectra = len(files)
    if num_spectra == 0:
        sys.exit()
folder = sep.join(file.split(sep)[:-1]) + sep
if folder == '/':
    folder = ''

# Ranges and limits for plots.
yy = np.concatenate(tuple([spectrum['y'] for spectrum in spectra]))
xx = np.concatenate(tuple([spectrum['x'] for spectrum in spectra]))
mask = np.isfinite(yy)
x_mask = xx[mask]
x_min = x_mask.min()
x_max = x_mask.max()
xrange = x_max - x_min
margin = 0.015 * xrange
x_lims = [x_min - margin, x_max + margin]
y_lims = calculate_robust_ylims(spectra, x_lims, perc1=0.1, perc2=99.5,
                                rel_margin=0.06)
old_x_lims = copy.copy(x_lims)

# Default noise level.
residuals = []
for spectrum in spectra:
    y = spectrum['y']
    y_smoothed = rv.rolling_function(np.mean, y, size=7)
    residuals += [np.abs(y - y_smoothed)]
residuals = np.concatenate(tuple(residuals))
noise_level = 0.5 * median_abs_deviation(residuals, scale='normal')

# Info file.
variable_x = 'wavelength' if input_in_microns else 'wavenumber' 
if use_table_format_for_input:
    input_data_info = {'file': file, 'columns': columns,
                       'x': variable_x, 'y': variable_y,
                       'number of spectra': num_spectra}
else:
    input_data_info = {'files': spectra_names, 'number of spectra': num_spectra,
                       'x': variable_x, 'y': variable_y}


#%% Loading the interactive mode.
    
print('- Press Z to zoom, Right/Left to move through the spectrum,'
      ' and Shift+Z or < to unzoom. '
      'Press Y to adapt the vertical range to display the spectra, or'
      ' Control+Y to adapt only to the selected spectrum, or Shit+Y to'
      ' restore the original limits. '
      'Press Up/Down to switch spectrum in case there is more than one.\n'
      '- Left click to select a window edge, right click to undo or remove the'
      ' window over the pointer, or Tab to remove all the windows.\n'
      '- Press B to estimate the a baseline for the selected spectra in the'
      ' current windows; alternatively, press . to manually select the'
      ' baseline points and then press B. If you press Shift+B, you will be'
      ' able to write the smoothing parameter for the baseline estimation'
      ' in the terminal.\n'
      '- Press S to smooth the data in the selected windows. If your press'
      ' Shift+S, you can write the smoothing factor in the terminal.\n'
      '- Press X to remove the selected windows and interpolate if possible, or'
      ' Shift+X to specify a smoothing factor the interpolation and apply it.\n'
      '- Press N to add Gaussian noise in the selected windows, or Shift+N to'
      ' specify the standard deviation and add it.\n'
      '- Press C to apply a sigma clip on the selected windows, or Shift+C'
      ' to specify the threshold and apply the sigma clip.\n'
      '- Press 0 to set the selected windows to zero, or Shift+0 (or =) to only'
      ' do so for negative absorbance values; if using manual selection of'
      ' points, this will automatically select a set of uniform zero points.\n'
      '- If spectra are in transmission, press A to convert to absorbance.\n'
      '- If spectra are in transmission, press 1 to set the selected'
      ' windows to one, or Shift+1 (or !) to only do so for values greater'
      ' than one.\n'
      '- If spectra are in transmisson, press L to switch between linear and'
      ' logarithmic scale.'
      '- Press W to automatically add windows depending on the molecules present'
      ' in the file name, or Shift+W to add a molecule manually.\n'
      '- Press P to use AICE to predict the composition of the ice.\n'
      '- Press I to integrate the selected spectrum in the current window, or'
      ' Shift+I to introduce a band strength and integrate the column density.\n'
      '- Press F to multiply the selected regions by the specified factor,'
      ' or Shift+F to convert to absorption coefficient by dividing by the'
      ' input column density.\n'
      '- If spectra are in absorbance, press - to invert the vertical axis.\n'
      '- Press , to resample the spectra to the given wavenumber array.\n'
      '- Press J to activate the individual processing mode or restore'
      ' the joint mode.\n'
      '- Press Delete to remove the selected spectrum from the file.\n'
      '- Press M to load a macro/algorithm to apply, or Shift+M to use one of'
      ' the default ones.\n'
      '- Press Control+Z to undo, or Control+Shift+Z to redo.\n'
      '- To save the files and exit, press Shift+Enter, or Ctrl+Enter if you'
      ' also want to save a file containing the computed baselines.\n'
      '- To cancel and exit, press Escape or close the plot window.\n'
      '- If you write anything on the terminal, you should then click on the'
      ' plot window before pressing any key.\n')

spectra_old = copy.deepcopy(spectra)
num_spectra = len(spectra)
indices = list(range(num_spectra))
all_indices = copy.deepcopy(indices)
selected_points = []
individual_mode = False
manual_mode = False
logscale = False
in_macro = False
ilog, jlog, idx, k = 0, 0, 0, 0
save_action_log = True
spectra_old = copy.deepcopy(spectra)
data = {'spectra': spectra, 'idx': idx, 'indices': indices,
        'spectra_old': spectra,'all_indices': all_indices,
        'spectra_names': spectra_names, 'variable_y': variable_y, 
        'selected_points': selected_points, 'in_macro': False}
data_log = [copy.deepcopy(data)]
action_log = [{'action': 'start'}]
macro_actions = []

plt.figure(1, figsize=(9.,5.))
plot_data(spectra, spectra_old, idx)
fig = plt.gcf()
fig.canvas.mpl_connect('button_press_event', click1)
fig.canvas.mpl_connect('button_release_event', click2)
fig.canvas.mpl_connect('key_press_event', press_key)

plt.show()
print()
sys.exit()