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
weights_path = '/Users/andresmegias/Documents/Astro/Proyectos/AICE/neural-network/training/aice-weights.npy'
errors_path = '/Users/andresmegias/Documents/Astro/Proyectos/AICE/neural-network/training/aice-errors.csv'

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
from matplotlib.ticker import ScalarFormatter
from matplotlib.backend_bases import MouseEvent, KeyEvent
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy.stats import median_abs_deviation

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
        By default, 
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
        spl = UnivariateSpline(x_, y_, s=s)
        yb = spl(x)
    else:
        raise Exception('Wrong interpolation type.')
        
    return yb

def create_baseline(x, p, smooth_size=1):
    """
    Create a baseline from the input points.

    Parameters
    ----------
    x : array
        Data where to apply the baseline fit.
    p : list / array (2, N)
        Reference points for the baseline.
    smooth_size : int, optional
        Smooth factor for the baseline. By default it is 1 (no smoothing).

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
    yb = rv.rolling_function(np.mean, yb, smooth_size)   
    return yb

def sigma_clip_mask(y, sigmas=6.0, iters=2):
    """
    Apply a sigma clip and return a mask of the remaining data.

    Parameters
    ----------
    y : array
        Input data.
    sigmas : float, optional
        Number of deviations used as threshold. The default is 6.0.
    iters : int, optional
        Number of iterations performed. The default is 3.

    Returns
    -------
    mask : array (bool)
        Mask of the remaining data after applying the sigma clip.
    """
    mask = np.ones(len(y), dtype=bool)
    abs_y = abs(y)
    for i in range(iters):
        mask *= abs_y < sigmas*median_abs_deviation(abs_y[mask], scale='normal')
    return mask

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

def calculate_xlims(x_min, x_max, zoom_level, x_zone, rel_margin=0.01,
                    overlap=0.2):
    """Calculate horizontal limits for the given zoom level and zone."""
    num_divisions = 2**zoom_level
    x_range = x_max - x_min
    x_step = x_range / num_divisions
    x_edges = [x_min + i*x_step for i in range(num_divisions+1)]
    idx = len(x_edges) - x_zone - 1
    lim1 = x_min - rel_margin*x_step
    lim2 = x_max + rel_margin*x_step
    margin = (rel_margin * x_step if zoom_level == 0
              else (rel_margin + overlap) * x_step)
    x_lims = [max(lim1, x_edges[idx] - margin),
              min(lim2, x_edges[idx+1] + margin)]
    return x_lims

def calculate_ylims(spectra, x_lims=None, perc1=0., perc2=100.,
                    rel_margin_y=0.04):
    """Calculate vertical limits for the given spectra."""
    absorbances = []
    for spectrum in spectra:
        wavenumber = spectrum['wavenumber']
        absorbance = spectrum['absorbance']
        if x_lims is not None:
            x1, x2 = x_lims
            mask = (wavenumber >= x1) & (wavenumber <= x2)
            absorbance = absorbance[mask]
        absorbances = np.concatenate((absorbances, absorbance))
    absorbances = np.unique(absorbances)
    y1 = np.nanpercentile(absorbances, perc1)
    y2 = np.nanpercentile(absorbances, perc2)
    margin = rel_margin_y * (y2 - y1)
    y_lims = [y1 - margin, y2 + margin]
    return y_lims 

def parse_float(text):
    """Parse input text as a float."""
    factor = input('- Enter smoothing factor for baseline: ')
    factor = int(''.join([char for char in factor if char.isdigit()]))
    factor = int(factor)
    
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
    return species_list

def aice_model(absorbance, weights, errors):
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
    errors : array (float)
        Intrinsec errors of the neural network model.
    
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
            fit_temp = True if i == 0 else False
            end_act = relu if fit_temp else sigmoid
            yj[i] = nn_model(absorbance, weights[j,i], end_act)[0]
        results += [yj]
    results = np.array(results)
    stdevs = np.std(results, axis=0)
    predictions = np.mean(results, axis=0)
    # uncs = np.maximum(errors, stdevs)
    uncs = stdevs
    predictions = rv.RichArray(predictions, uncs)
    labels = aice_labels + ['all molecules']
    sum_predictions = predictions[1:].sum()
    predictions = list(predictions) + [sum_predictions]
    predictions_df = rv.rich_dataframe({'AICE prediction': predictions},
                                       index=labels)
    return predictions_df

# Predefined parameters.

species_windows = {
    'H2O': [[3720, 2805], [2600, 2010], [1800, 1300], [1000, 500]],
    'CO': [[2165, 2120]],
    '13CO': [[2110,2075]],
    'CO2': [[3723, 3695], [3612, 3588], [2396, 2316], [687, 636]],
    'CH3OH': [[3560, 2712], [1178, 982]],
    'CH4': [[3103, 2968], [1334, 1274]],
    'NH3': [[3638, 2928], [1223, 976]],
    'H2CO': [[3040, 2780], [1760, 1690], [1530, 1470], [1280, 1150]],
    'C2H5NH2': [[3450, 3040], [3010, 2750], [1670, 1220], [1180, 800]]
}
species_windows['12CO'] = species_windows['CO']

predefined_macros = {
    'M1':
        """
        - median smooth:
            smooth factor: 15
        - reduce :
            smooth factor: 45
        - modify windows :
            windows:
            - (2992.03, 2942.30)
            - (1280.70, 1237.19)
            - (899.49, 686.09)
        - remove/interpolate :
            smooth factor: 1
        - modify windows :
            windows:
            - (1235.12, 891.20)
        - smooth :
            smooth factor: 15
        """,
    'M2':
        """
        - median smooth:
            smooth factor: 15
        - reduce :
            smooth factor: 31
        - modify windows :
            windows:
            - (2980.00, 2945.30)
            - (1300.70, 1200.19)
            - (899.49, 600.09)
        - remove/interpolate :
            smooth factor: 1
        - modify windows :
            windows:
            - (1235.12, 891.20)
        - smooth :
            smooth factor: 15
        - modify windows :
            windows:
            - (6000.00, 3500.00)
        - smooth :
            smooth factor: 25
        """,
    'M3':
        """
        - median smooth:
            smooth factor: 75
        - reduce :
            smooth factor: 45
        - modify windows :
            windows:
            - (2980.00, 2945.30)
            - (1300.70, 1200.19)
            - (899.49, 600.09)
        - remove/interpolate :
            smooth factor: 1
        - modify windows :
            windows:
            - (1235.12, 891.20)
        - smooth :
            smooth factor: 15
        - modify windows :
            windows:
            - (6000.00, 3500.00)
        - smooth :
            smooth factor: 45
        """
    }

#%% Functions used in interactive mode.

def plot_data(spectra, spectra_old, idx, manual_mode=False):
    """Plot the input spectra."""
    
    fig = plt.figure(1, figsize=(9,5))
    plt.clf()
    
    for spectrum_old in spectra_old:
        plt.plot(spectrum_old['wavenumber'], spectrum_old['absorbance'],
                 color='gray', alpha=0.1, zorder=2.4)
    plt.plot(spectra_old[idx]['wavenumber'], spectra_old[idx]['absorbance'],
             label='original spectrum', color='black', zorder=2.6) 
    spectrum = spectra[idx]
    if 'baseline' in spectrum:
        if not np.array_equal(spectra_old[idx]['absorbance'],
                              spectrum['absorbance-fit']):
            plt.plot(spectrum['wavenumber-fit'], spectrum['absorbance-fit'],
                     color='tab:red', zorder=2.5, label='baseline reference')
        plt.plot(spectrum['wavenumber'], spectrum['baseline'], zorder=2.8,
                 color='darkorange', linestyle='--', label='baseline fit')
    if spectrum['modified']:
        plt.plot(spectrum['wavenumber'], spectrum['absorbance'],
                 color='darkblue', zorder=2.7, label='edited spectrum')
    
    cmap_old = plt.colormaps['brg']
    cmap_new = plt.colormaps['viridis']
    num_spectra = len(spectra)
    for (i,spectrum) in enumerate(spectra):
        if i == idx:
            continue
        color = (cmap_old(0.5 * i / num_spectra) if 'baseline' not in spectrum
                 else cmap_new(0.4 + 0.6 * i / num_spectra))
        plt.plot(spectrum['wavenumber'], spectrum['absorbance'], color=color,
                 alpha=0.2, zorder=2.5)
    
    global x_min, x_max, x_lims, y_lims
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.axhline(y=0, color='black', lw=0.5)
    plt.xlabel('wavenumber (cm$^{-1}$)', labelpad=6)
    plt.ylabel('absorbance', labelpad=12)
    plt.margins(x=0.01)
    if not manual_mode:
        plt.axvspan(0., 0., 0., 0., edgecolor='lightgray', facecolor='white',
                    alpha=1., label='windows')
    else:
        plt.plot([], '.', color='chocolate', label='baseline reference')
    plt.legend(loc='upper left')
    plt.title(spectra_names[idx], fontweight='bold', pad=12.)
    
    ax = plt.gca()
    ax.invert_xaxis()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
    wavelength_ticks = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8,
                        10, 12, 15, 20, 40] # μm
    ax2.set_xticks(wavelength_ticks, wavelength_ticks)
    ax2.set_xlabel('wavelength (μm)', labelpad=6., fontsize=9.)
    
    plt.tight_layout()
    plt.text(0.98, 0.96, '      AICE Interactive Toolkit' '\n ' ,
             ha='right', va='top', fontweight='bold', transform=ax.transAxes,
             bbox=dict(edgecolor=[0.8]*3, facecolor='white'))
    plt.text(0.98, 0.96, ' \n' 'check terminal for instructions',
             ha='right', va='top', transform=ax.transAxes)
    
    plt.gca().set_facecolor([0.9]*3)
    
    return fig

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
        plt.plot(x, y, '.', color='chocolate', zorder=2.8)

def do_reduction(spectra, indices, selected_points, smooth_size, manual_mode,
                 interpolation):
    """Reduce the data."""
    if len(selected_points) == 0:
        return
    windows = format_windows(selected_points)
    for i in indices:
        spectrum = spectra[i] 
        wavenumber = spectrum['wavenumber']
        absorbance = spectrum['absorbance']
        windows = np.append(windows, [wavenumber[:2]], axis=0)
        windows = np.append(windows, [wavenumber[-2:]], axis=0)
        if not manual_mode:
            baseline = fit_baseline(wavenumber, absorbance, smooth_size,
                                    windows, interpolation)
        else:
            baseline = create_baseline(wavenumber, selected_points,
                                       smooth_size=3)
        reduced_absorbance = absorbance - baseline  
        spectra[i]['modified'] = True
        spectra[i]['absorbance'] = reduced_absorbance
        spectra[i]['baseline'] = baseline
        spectra[i]['wavenumber-fit'] = wavenumber
        spectra[i]['absorbance-fit'] = absorbance

def do_removal(spectra, indices, selected_points, smooth_size):
    """Remove the selected regions of the data."""
    if selected_points == []:
        return
    windows = format_windows(selected_points)
    for i in indices: 
        spectrum = spectra[i]
        wavenumber = spectrum['wavenumber']
        absorbance = spectrum['absorbance']
        baseline = spectrum['baseline'] if 'baseline' in spectrum else None
        for (x1,x2) in windows:
            is_inferior_edge = x1 <= np.min(wavenumber)
            is_superior_edge = x2 >= np.max(wavenumber)
            if is_inferior_edge or is_superior_edge:
                mask = (wavenumber >= x2 if is_inferior_edge
                        else wavenumber <= x1)
                wavenumber = wavenumber[mask]
                absorbance = absorbance[mask]
                baseline = baseline[mask] if 'baseline' in spectrum else None
            else:
                mask = (wavenumber >= x1) & (wavenumber <= x2)
                windows_ = invert_windows(windows, wavenumber)
                absorbance[mask] = fit_baseline(wavenumber, absorbance,
                            smooth_size, windows_, interpolation='pchip')[mask]
        spectra[i]['modified'] = True
        spectra[i]['wavenumber'] = wavenumber
        spectra[i]['absorbance'] = absorbance
        if 'baseline' in spectrum:
            spectra[i]['baseline'] = baseline

def do_smoothing(spectra, indices, selected_points, smooth_size,
                 function=np.mean):
    """Smooth the data in the selected regions.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    windows = format_windows(selected_points)
    for i in indices:
        spectrum = spectra[i] 
        wavenumber = spectrum['wavenumber']
        absorbance = spectrum['absorbance']
        for (x1,x2) in windows:
            mask = (wavenumber >= x1) & (wavenumber <= x2)
            absorbance[mask] = rv.rolling_function(function, absorbance[mask],
                                                   smooth_size)
        spectra[i]['modified'] = True
        spectra[i]['absorbance'] = absorbance
        
def do_multiplication(spectra, indices, selected_points, factor):
    """Multiply the data in the selected regions with the input factor.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    windows = format_windows(selected_points)
    for i in indices:
        spectrum = spectra[i] 
        wavenumber = spectrum['wavenumber']
        absorbance = spectrum['absorbance']
        for (x1,x2) in windows:
            mask = (wavenumber >= x1) & (wavenumber <= x2)
            absorbance[mask] = factor * absorbance[mask]
        spectra[i]['modified'] = True
        spectra[i]['absorbance'] = absorbance
        
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
        wavenumber = spectrum['wavenumber']
        absorbance = spectrum['absorbance']
        for (x1,x2) in windows:
            mask = (wavenumber >= x1) & (wavenumber <= x2)
            if only_negatives:
                mask &= absorbance < 0.
            absorbance[mask] = 0.
        spectra[i]['modified'] = True
        spectra[i]['absorbance'] = absorbance

def add_noise(spectra, indices, selected_points, noise_level):
    """Add noise to the selected regions.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    for i in indices:
        spectrum = spectra[i] 
        wavenumber = spectrum['wavenumber']
        absorbance = spectrum['absorbance']
        windows = format_windows(selected_points)
        for (x1,x2) in windows:
            mask = (wavenumber >= x1) & (wavenumber <= x2)
            noise = np.random.normal(0., scale=noise_level, size=sum(mask))
            absorbance[mask] = absorbance[mask] + noise
        spectra[i]['modified'] = True
        spectra[i]['absorbance'] = absorbance

def do_sigma_clip(spectra, indices, selected_points, threshold,
                  smooth_size=1, iters=3):
    """Apply a sigma clip in the selected regions.""" 
    if selected_points == []:
        global x_min, x_max
        selected_points = [x_max, x_min]
    for i in indices:
        spectrum = spectra[i] 
        wavenumber = spectrum['wavenumber']
        absorbance = spectrum['absorbance']
        windows = format_windows(selected_points)
        for (x1,x2) in windows:
            mask = (wavenumber >= x1) & (wavenumber <= x2)
            for j in range(iters):
                limit = threshold * median_abs_deviation(absorbance[mask])
                mask &= (np.abs(absorbance) > limit)
            absorbance[mask] = np.nan
        spectra[i]['modified'] = True
        spectra[i]['absorbance'] = absorbance
        
def integrate_spectrum(spectrum, selected_points, factor=1.):
    """Integrate the spectrum."""
    if selected_points == []:
        print('Error: No points selected.')
        return
    windows = format_windows(selected_points)
    wavenumber = spectrum['wavenumber']
    absorbance = spectrum['absorbance']
    area = 0.
    for (x1,x2) in windows:
        mask = (wavenumber >= x1) & (wavenumber <= x2)
        area += np.trapz(absorbance[mask], wavenumber[mask])
    area /= factor
    return area
        
def do_resampling(spectra, indices, new_wavenumber):
    """Resample the spectra to the input wavenumber array.""" 
    for i in indices:
        spectrum = spectra[i] 
        wavenumber = spectrum['wavenumber']
        absorbance = spectrum['absorbance']
        new_absorbance = np.interp(new_wavenumber, wavenumber, absorbance)
        spectra[i]['modified'] = True
        spectra[i]['wavenumber'] = new_wavenumber
        spectra[i]['absorbance'] = new_absorbance
        
def save_files(spectra, data_log, action_log, files, baseline=False):
    """Save processed spectra."""
    
    global columns
    nondone_fits = np.array([not spectrum['modified'] for spectrum in spectra])
    suffix = '-reduced' if not baseline else '-baseline'
    intensity_variable = 'absorbance' if not baseline else 'baseline'
    if all(nondone_fits):
        print('\nNo processing was performed.\n')
        suffix = ''
    elif any(nondone_fits):
        print('\nProcessing was not performed in all spectra.\n')

    all_indices = data_log[-1]['all_indices']
    if use_table_format_for_output:
        if use_table_format_for_input:
            columns = np.array(columns)[all_indices].tolist()
            default_filename = files[0].replace('.csv', '')
        else:
            columns = np.array(spectra_names)[all_indices].tolist()
            columns = ['.'.join(column.split('.')[:-1]) for column in columns]
            default_filename = os.path.commonprefix(files)
        filename = input('Output filename: ')
        filename = filename.replace('*', default_filename)
        if filename == '':
            filename = default_filename + suffix
        else:
            filename = folder + filename
        if not filename.endswith('.csv'):
            filename += '.csv'
        wavenumbers = [spectrum['wavenumber'] for spectrum in spectra]
        i = np.argmax([len(wavenumber) for wavenumber in wavenumbers])
        wavenumber = wavenumbers[i]
        new_df = pd.DataFrame({'wavenumber (/cm)': wavenumber})
        for (i,spectrum) in enumerate(spectra):
            column = columns[i]
            wavenumber_i = spectrum['wavenumber']
            absorbance_i = spectrum[intensity_variable]
            absorbance_i = np.interp(wavenumber, wavenumber_i, absorbance_i)
            new_df[column] = absorbance_i
            nd = max([len(x.split('.')[-1]) if '.' in x else 0
                      for x in wavenumber.astype(str)])
        new_df['wavenumber (/cm)'] = new_df['wavenumber (/cm)'].map(
                                                lambda x: '{:.{}f}'.format(x, nd))
        new_df.to_csv(filename, index=False, float_format='%.3e')
        num_spectra = len(spectra)
        saved_var = 'spectrum' if not baseline else 'baseline'
        if num_spectra > 1:
            saved_var = (saved_var.replace('spectrum', 'spectra')
                         .replace('baseline', 'baselines'))
        print('\nSaved {} {} in {}'.format(num_spectra, saved_var, filename))
        output_data_info = {'file': filename.split(sep)[-1], 'columns': columns,
                            'number of spectra': len(spectra)}
    else: 
        if use_table_format_for_input:
            files = [files[0].replace('.csv','-')
                     + col.replace(' ','').replace('abs.','').replace('.','')
                     + '.txt' for col in columns]
        else:
            files = np.array(files)[all_indices].tolist()
        output_files = ['.'.join(file.split('.')[:-1]) + suffix + '.txt'
                        for file in files]
        for (i,filename) in enumerate(output_files):
            spectrum = spectra[i]
            wavenumber = spectrum['wavenumber']
            absorbance = spectrum[intensity_variable]
            data = np.array([wavenumber, absorbance]).transpose()
            inds = np.argsort(wavenumber)
            data = data[inds]
            nd = max([len(x.split('.')[-1]) if '.' in x else 0
                      for x in wavenumber.astype(str)])
            np.savetxt(filename, data, fmt='%.{}f %.3e'.format(nd),
                       header='wavenumber (/cm)   absorbance')
            saved_var = 'spectrum' if not baseline else 'baseline'
            print('Saved {} in {}.'.format(saved_var, filename))
        output_files = [file.split(sep)[-1] for file in output_files]
        output_data_info = {'files': output_files, 'number of spectra': len(spectra)}
            
    return output_data_info, filename

def save_action_record(action_log, files, spectra, filename, output_data_info):
    """Write and save action record file."""
    
    global use_table_format_for_output
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
            if action in ('smooth', 'median smooth', 'reduce',
                          'remove/interpolate', 'add noise', 'sigma clip',
                          'multiply'):
                if action in ('smooth', 'median smooth',
                              'reduce', 'remove/interpolate'):
                    factor_name = 'smooth factor'
                elif action == 'add noise':
                    factor_name = 'noise level'
                elif action == 'sigma clip':
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
    info_dic = {'input data': input_data_info,
                'output data': output_data_info,
                'action record': action_record}
    if use_table_format_for_output:
        log_file = filename.replace('.csv', '-log.txt')
    else:
        if len(spectra) == 1:
            log_file = filename.replace('.txt', '-log.txt')
        else:
            log_file = os.path.commonprefix(files).replace('.csv','') + '-log.txt'
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
                plt.plot(x, y, '.', color='chocolate', zorder=2.8)
            elif button in ('right', '3'):
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
            for (x, y) in selected_points:
                points += ['({:.2f}, {:.4g})'.format(x, y)]
            action_info = {'action': 'modify points', 'points': points}
        action_log = action_log[:jlog+1] + [copy.deepcopy(action_info)]
        jlog += 1
        plt.draw()    

def press_key(event):
    """Interact with the plot when pressing a key."""
    if type(event) is not KeyEvent:
        pass
    global files, spectra, data_log, action_log, ilog, jlog
    global spectra_old, idx, indices, all_indices, spectra_names
    global baseline_smooth_size, interp_smooth_size, smooth_size
    global original_spectra, individual_mode, manual_mode
    global selected_points, x_lims, y_lims, old_x_lims, old_y_lims
    global in_macro, k, macro_actions, use_table_format_for_output
    macro_event = False
    if in_macro and event.key in ('enter', ' ', 'escape'):
        if event.key == 'escape':
            k = 0
            in_macro = False
            print('Macro exited.')
        elif event.key in ('enter', ' '):
            if event.key == 'enter':
                macro_event = True
                action = list(macro_actions[k].keys())[0]
                params = macro_actions[k][action]
                if action.startswith('modify windows'):
                    windows = params['windows']
                    if type(windows[0]) is str:
                        windows = [list(np.array(text[1:-1].split(', '),float))
                                   for text in windows]
                    selected_points = list(np.array(windows).flatten())
                    event.key = 'w'
                elif action.startswith('reduce'):
                    event.key = 'r'
                    factor = params['smooth factor']
                elif action.startswith('remove') or 'interpolate' in action:
                    event.key = 'x'
                    factor = params['smooth factor']
                elif (action.startswith('smooth')
                      or action.startswith('median smooth')):
                    event.key = 's' if action.startswith('smooth') else 'ctrl+s'
                    factor = params['smooth factor']
                elif action.startswith('add noise'):
                    event.key == 'n'
                    factor = params['smooth factor']
                elif action.startswith('sigma clip'):
                    event.key = 'c'
                    factor = params['threhsold']
                else:
                    print('Unknown action: {}.')
            k += 1
            if k < len(macro_actions):
                action_text = str(macro_actions[k])[1:-1].replace("'","")
                print('Next action: {}'.format(action_text))
    elif event.key in ('shift+enter', 'ctrl+enter', 'escape'):
        if 'enter' in event.key:
            data_log = data_log[:ilog+1]
            action_log = action_log[:jlog+1]
            (output_data_info, filename) = \
                save_files(spectra, data_log, action_log, files)
            save_action_record(action_log, files, spectra, filename,
                               output_data_info)
        print()
        sys.exit()
    x_lims = list(reversed(plt.xlim()))
    y_lims = plt.ylim()
    if individual_mode:
        indices = [idx]
    if event.key in ('y', 'Y', 'ctrl+y', 'ctrl+Y'):
        if event.key == 'Y':
            y_lims = copy.copy(old_y_lims)
        elif event.key == 'y':
            ylims1 = calculate_ylims(spectra, x_lims, perc1=0.5)
            ylims2 = calculate_ylims(original_spectra, x_lims, perc1=0.5)
            y_lims = [min(ylims1[0], ylims2[0]), max(ylims1[1], ylims2[1])]
        elif event.key == 'ctrl+y':
            ylims1 = calculate_ylims([spectra[idx]], x_lims, perc1=0.5)
            ylims2 = calculate_ylims([original_spectra[idx]], x_lims, perc1=0.5)
            y_lims = [min(ylims1[0], ylims2[0]), max(ylims1[1], ylims2[1])]
        else:
            y_lims = calculate_ylims([spectra[idx]], x_lims, perc1=0.5)
            
    elif event.key in ('z', 'Z', '<', 'left', 'right'):
        x_range = x_lims[1] - x_lims[0]
        if event.key in ('z', 'Z', '<'):
            if event.key == 'z':
                x_lims = [x_lims[0] + x_range/4, x_lims[1] - x_range/4]
            else:
                x_lims = [x_lims[1] - 2*x_range, x_lims[1] + 2*x_range]
                ylims1 = calculate_ylims(spectra, x_lims, perc1=0.5)
                ylims2 = calculate_ylims(original_spectra, x_lims,
                                         perc1=0.5)
                y_lims = [min(ylims1[0], ylims2[0]),  max(ylims1[1], ylims2[1])]
        else:
            if event.key == 'left' and x_lims[1] != old_x_lims[1]:
                x_lims = [x_lims[0] + x_range, x_lims[1] + x_range]
            elif event.key == 'right' and x_lims[0] != old_x_lims[0]:
                x_lims = [x_lims[0] - x_range, x_lims[1] - x_range]
        x_lims = [max(old_x_lims[0], x_lims[0]),
                  min(old_x_lims[1], x_lims[1])]
        if x_lims == old_x_lims:
            y_lims = old_y_lims
    elif event.key in ('up', 'down'):
        if len(spectra) > 1:
            idx = idx+1 if event.key == 'up' else idx-1
            idx = idx % len(spectra)
        action_info = {'action': 'switch spectrum',
                       'spectrum': spectra_names[idx]}
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
            if not macro_event:
                text = (files[idx] if event.key == 'w'
                        else input('Write species to mask: '))
                species_list = parse_composition(text, folder)
                wavenumber = spectra[idx]['wavenumber']
                if len(selected_points) == 0:
                    mask = np.zeros(len(wavenumber), bool)
                else:
                    previous_windows = format_windows(selected_points)
                    mask = ~get_mask(previous_windows, wavenumber)
                for species in species_list:
                    if species in species_windows:
                        for x1x2 in species_windows[species]:
                            x1, x2 = min(x1x2), max(x1x2)
                            mask |= (wavenumber >= x1) & (wavenumber <= x2)
                windows = get_windows(~mask, wavenumber)
                selected_points = list(np.array(windows).flatten()[::-1])
            windows_text = []
            for x1x2 in format_windows(selected_points):
                x1, x2 = min(x1x2), max(x1x2)
                windows_text += ['({:.2f}, {:.2f})'.format(x2, x1)]
            action_info = {'action': 'modify windows', 'windows': windows_text}
    elif event.key in ('r', 'R', 'ctrl+r', 'ctrl+R'):
        if not macro_event:
            factor = copy.copy(baseline_smooth_size)
        interpolation = 'spline' if 'c' in event.key else 'pchip'
        if 'R' in event.key:
            text = input('- Enter smoothing factor for baseline: ')
            factor = int(''.join([char for char in text if char.isdigit()]))
        do_reduction(spectra, indices, selected_points, factor,
                     manual_mode, interpolation)
        if selected_points == []:
            print('Error: No points selected.')
        else:
            print('Reduction performed with smooth factor {}.'.format(factor))
        action_info = {'action': 'reduce', 'smooth factor': factor}
    elif event.key in ('s', 'S', 'ctrl+s', 'ctrl+S'):
        if not macro_event:
            factor = copy.copy(smooth_size)
        if 'S' in event.key:
            text = input('- Enter smoothing factor: ')
            factor = int(''.join([char for char in text if char.isdigit()]))
        function = np.median if 'ctrl' in event.key else np.mean
        do_smoothing(spectra, indices, selected_points, factor, function)
        action_info = {'action': 'smooth', 'smooth factor': factor}
        if 'ctrl' in event.key:
            action_info['action'] = 'median smooth'
    elif event.key in ('x', 'X'):
        if not macro_event:
            factor = copy.copy(interp_smooth_size)
        if event.key == 'X':
            text = input('- Enter smoothing factor for interpolation: ')
            factor = int(''.join([char for char in text if char.isdigit()]))
        do_removal(spectra, indices, selected_points, factor)
        if selected_points == []:
            print('Error: No points selected.')
        else:
            print('Interpolated/removed areas in present windows.')
        action_info = {'action': 'remove/interpolate', 'smooth factor': factor}
    elif event.key in ('n', 'N'):
        if not macro_event:
            factor = copy.copy(noise_level)
        if event.key == 'N':
            factor = float(input('- Enter noise level: '))
        add_noise(spectra, indices, selected_points, factor)
        action_info = {'action': 'add noise', 'noise level': '{:.3e}'
                       .format(factor)}
    elif event.key in ('c', 'C'):
        if not macro_event:
            factor = copy.copy(sigma_threshold)
        if event.key == 'C':
            text = float(input('- Enter a sigma threshold: '))
        do_sigma_clip(spectra, indices, selected_points, factor)
        print('Sigma clipping applied in present windows.')
        action_info = {'action': 'sigma clip', 'threshold': factor}
    elif event.key == 'f':
        text = input('- Enter factor to multiply: ')
        factor = float(''.join([char for char in text
                                if char.isdigit() or char == '.']))
        do_multiplication(spectra, indices, selected_points, factor)
        action_info = {'action': 'multiply', 'factor': factor}
    elif event.key in ('i', 'I'):
        if event.key == 'I':
            factor = float(input('- Introduce the band strength (cm): '))
            area_variable = 'column density'
            area_units = '/cm2'
        else:
            factor = 1.
            area_variable = 'area'
            area_units = '/cm'
        area = integrate_spectrum(spectra[idx], selected_points, factor)
        print('Integrated {}: {:.2e} {}'.format(area_variable, area, area_units))
    elif event.key == 'p':
        global weights, weights_path
        if weights is None:
            print('Could not find AICE weights file in {}.'.format(weights_path))
        else:
            spectrum = spectra[idx]
            wavenumber = np.arange(980., 4001., 1.)
            absorbance = np.interp(wavenumber,
                                spectrum['wavenumber'], spectrum['absorbance'])
            absorbance = np.nan_to_num(absorbance, nan=0.)
            absorbance /= np.nanmean(absorbance)
            predictions_df = aice_model(absorbance, weights, errors)
            print(predictions_df)
    elif event.key in ('B', 'ctrl+b'):
        data_log = data_log[:ilog+1]
        action_log = action_log[:jlog+1]
        save_files(spectra, data_log, action_log, files, baseline=True)
    elif event.key == '-':
        params = input('- Enter new wavenumber array to resample'
                       ' (initial wavenumber, final wavenumber, step): ')
        params = params.replace('(','').replace(')','').replace(' ','')
        params = params.split(',')
        x1x2 = np.array(params[:2], float)
        step = float(params[2])
        x1, x2 = min(x1x2), max(x1x2)
        new_wavenumber = np.arange(x1, x2, step)
        do_resampling(spectra, indices, new_wavenumber)
        action_info = {'action': 'resample',
            'wavenumber array (start, end, step)': list(params)}
    elif event.key in ('0', '='):
        if not manual_mode:
            if event.key == '0':
                only_negatives = False
                action_info = {'action': 'set to zero'}
            else:
                only_negatives = True
                action_info = {'action': 'set negatives to zero'}
            set_zero(spectra, indices, selected_points, only_negatives)
        else:
            action_info = {'action': 'modify points'}
            for x in np.linspace(x_min, x_max, 40):
                selected_points += [[x, 0.]] 
    elif event.key in ('1') and not manual_mode:
        individual_mode = not individual_mode
        if individual_mode:
            indices = [idx]
            print('- Processing mode has changed to individual.')
            action_info = {'action': 'activate individual processing mode',
                           'spectrum': spectra_names[idx]}
        else:
            indices = list(range(len(spectra)))
            print('- Processing mode has changed to joint.')
            action_info = {'action': 'activate joint processing mode'}
    elif event.key == '.':
        print(selected_points)
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
            print('- Using manual baseline mode.' + extra_msg)
            action_info = {'action': 'activate manual baseline mode',
                           'spectrum': spectra_names[idx]}
        else:
            print('- Using windows baseline mode.' + extra_msg)
            action_info = {'action': 'activate windows baseline mode'}
    elif event.key == 't':
        use_table_format_for_output = not use_table_format_for_output
        if use_table_format_for_output:
            print('- Using table format (.csv) when exporting.')
        else:
            print('- Using individual files (.txt) for exporting.')
        action_info = {'action': 'activate table format for output'}
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
    elif event.key in ('m', 'M'):
        if event.key == 'm':
            name = input('Write the name of the macro file: ')
            with open(name, 'r') as file:
                macro_actions = yaml.safe_load(file)
            name = name.split(sep)[-1]
        else:
            name = input('Write the name of the predefined macro: ')
            macro_actions = yaml.safe_load(predefined_macros[name])
        print('\nStarting macro {}.'.format(name))
        print('Please, go back to the plot window and click on it.'
              ' Press Enter to apply next action, Space to skip or'
              ' Escape to exit the macro.')
        action_text = str(macro_actions[k])[1:-1].replace("'","")
        print('Next action: {}'.format(action_text))
        k = 0
        in_macro = True
    elif event.key == 'a':
        if k < len(macro_actions):
            action_text = str(macro_actions[k])[1:-1].replace("'","")
            print('Next action: {}'.format(action_text))
    elif event.key in ('ctrl+z', 'cmd+z', 'ctrl+Z', 'cmd+Z','ctrl+<', 'cmd+<'):
        macro_event_prev = data_log[ilog]['macro_event']
        ilog = (max(0, ilog-1) if 'z' in event.key
                else min(len(data_log)-1, ilog+1))
        data = copy.deepcopy(data_log[ilog])
        spectra = data['spectra']
        idx = data['idx']
        indices = data['indices']
        all_indices = data['all_indices']
        spectra_names = data['spectra_names']
        if macro_event or macro_event_prev:
            selected_points = data['selected_points']
        spectra_old = [original_spectra[i] for i in all_indices]
        macro_event = data['macro_event']
        jlog = (max(0, jlog-1) if 'z' in event.key
                else min(len(action_log)-1, jlog+1))
        action_info = copy.deepcopy(action_log[jlog-1])
        k_ = copy.copy(k)
        if 'z' in event.key and macro_event_prev:
            k = max(0, k-1)
        elif ('Z' in event.key or '<' in event.key) and macro_event:
            k = min(len(macro_actions)-1, k+1)
        if k_ != k:
            action_text = str(macro_actions[k])[1:-1].replace("'","")
            print('Next action: {}'.format(action_text))
    if event.key in ('r', 'R', 'ctrl+r', 'ctrl+R', 's', 'S', 'ctrl+s', 'ctrl+S',
                'x', 'X', 'n', 'N', 'c', 'C', 'f', '0', '=', 'w', 'W', 'ctl+W',
                '-', 'backspace', '.', '1', 'up', 'down', 'tab'):
        if event.key not in ('.', 'i', '1', 'up', 'down', 'tab'):
            data = {'spectra': spectra, 'idx': idx, 'indices': indices,
                'all_indices': all_indices, 'spectra_names': spectra_names,
                'selected_points': selected_points, 'macro_event': macro_event}
            data_log = data_log[:ilog+1] + [copy.deepcopy(data)]
            ilog += 1
        if not individual_mode and event.key in ('up', 'down'):
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

# Default options.
baseline_smooth_size = 25
interp_smooth_size = 1
smooth_size = 5
sigma_threshold = 5.
keymaps = ('back', 'copy', 'forward', 'fullscreen', 'grid', 'grid_minor',
           'help', 'home', 'pan', 'quit', 'quit_all', 'save', 'xscale',
           'yscale', 'zoom')
for keymap in keymaps:            
    plt.rcParams.update({'keymap.' + keymap: []})
sep = '\\' if platform.system() == 'Windows' else '/'  # folder separator

# Weights and errors for AICE.
try:
    weights = np.load(weights_path, allow_pickle=True)
    errors_df = pd.read_csv(errors_path, index_col=0)
    errors = []
    for name in errors_df:
        for label in aice_labels:
            label = label.split(' ')[0].replace('.', '')
            if label == name:
                errors += [errors_df[name].values.mean()]
except:
    weights = None
    errors = None

# Reading of the arguments.
if len(sys.argv) == 1:
    path = input('Write the name of the input file: ')
    print()
    # folder = sep.join(sys.argv[0].split(sep)[:-1])
    folder = sep.join(path.split(sep)[:-1]).replace('\\','')
    if os.getcwd() != folder:
        os.chdir(folder)
    path = path.split(sep)[-1].replace('\\','')
    if path.endswith(' '):
        path = path[:-1]
    file_paths = [path]
else:
    file_paths = []
    for (i,arg) in enumerate(sys.argv[1:]):
        if arg == '--flux':
            using_flux = True
        else:
            file_paths += [arg]

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
    num_spectra = data.shape[1] - 1
    revert_spectra = x[0] > x[1]
    if revert_spectra:
        x = x[::-1]
    for i in range(num_spectra):
        name = file.split(sep)[-1] + ' - ' + columns[i]
        y = data[:,i+1]
        if revert_spectra:
            y = y[::-1]
        spectrum = {'wavenumber': x, 'absorbance': y, 'modified': False}
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
        x, y = data[:,[0,1]].transpose()
        dy = data[:,2] if data.shape[1] > 2 else np.zeros(len(x))
        if np.median(x) < 80.:
            x = 1e4 / x
        inds = np.argsort(x)
        x = x[inds]
        y = y[inds]
        dy = dy[inds]
        spectrum = {'wavenumber': x, 'absorbance': y, 'modified': False}
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
absorbances = np.concatenate(tuple([spectrum['absorbance'] for spectrum in spectra]))
wavenumbers = np.concatenate(tuple([spectrum['wavenumber'] for spectrum in spectra]))
mask = np.isfinite(absorbances)
wavenumbers_mask = wavenumbers[mask]
x_min = wavenumbers_mask.min()
x_max = wavenumbers_mask.max()
x_range = x_max - x_min
margin = 0.015 * x_range
x_lims = [x_min - margin, x_max + margin]
y_lims1 = calculate_ylims(spectra, perc1=0., perc2=100.)
y_lims2 = calculate_ylims(spectra, perc1=0.1, perc2=99.5)
yrange = y_lims2[1] - y_lims2[0]
y_lims = [max(y_lims1[0], y_lims2[0] - 0.05*yrange),
          min(y_lims1[1], y_lims2[1] + 3.0*yrange)]
old_y_lims = copy.copy(y_lims)
old_x_lims = copy.copy(x_lims)
residuals = []
for spectrum in spectra:
    absorbance = spectrum['absorbance']
    absorbance_smoothed = rv.rolling_function(np.mean, absorbance, size=7)
    residuals += [np.abs(absorbance - absorbance_smoothed)]
residuals = np.concatenate(tuple(residuals))
noise_level = 0.5 * median_abs_deviation(residuals, scale='normal')

# Info file.
if use_table_format_for_input:
    input_data_info = {'file': file, 'columns': columns,
                        'number of spectra': num_spectra}
else:
    input_data_info = {'files': spectra_names, 'number of spectra': num_spectra}
use_table_format_for_output = True


#%% Loading the interactive mode.
    
print('Press Z to zoom, Right/Left to move through the spectrum,'
      ' and Shift+Z or < to unzoom. '
      'Press Y to adapt the vertical range to display the spectra, or'
      ' Control+Y to adapt only to the selected spectrum, or Shit+Y to'
      ' restore the original limits. '
      'Press Up/Down to switch spectrum in case there is more than one. '
      'Left click to select a window edge, right click to undo or remove the'
      ' window over the pointer, or Tab to remove all the windows. '
      'Press R to reduce the data fitting a baseline in the selected windows,'
      ' or Shift+R to specify the smoothing factor for the baseline'
      ' and reduce; alternatively, press . to manually select the baseline'
      ' points. '
      'Press S to smooth the data in the selected windows, or Shift+S to'
      ' to specify the smoothing factor and smooth. '
      'Press X to remove the selected windows and interpolate if possible, or'
      ' Shift+X to specify a smoothing factor the interpolation and apply it. '
      'Press N to add gaussian noise in the selected windows, or Shift+N to'
      ' specify the standard deviation and add it. '
      'Press C to apply a sigma clipping on the selected windows, or Shift+C'
      ' to specify the threshold and apply the sigma clip. '
      'Press F to multiply the selected regions by the specified factor. '
      'Press W to automatically add windows depending on the molecules present'
      ' in the file name, or Shift+W to add a molecule manually. '
      'Press P to use AICE to predict the composition of the ice.'
      'Press Ctrl+B to save the current baseline(s). '
      'Press I to integrate the selected spectrum in the current window, or'
      ' Shift+I to introduce a band strength and integrate the column density.'
      'Press 0 to set the selected windows to zero, or Shift+0 (or =) to only'
      ' do so for negative absorbance values; if using manual selection of'
      ' points, this will automatically select a set of uniform zero points. '
      'Press - to resample the spectra to the given wavenumber array. '
      'Press 1 to activate individual processing mode or restore joint mode. '
      'Press T to use a table format when exporting the spectra. '
      'Press Delete to remove selected spectrum from the file (if possible). '
      'Press M to load a macro/algorithm to apply, or Shift+M to use one of'
      ' the default ones. '
      'Press Control+Z to undo, or Control+Shift+Z to redo. '
      'To save the files and exit, press Ctrl+Enter. '
      'To cancel and exit, press Escape. '
      'If you write anything on the terminal, you should then click on the'
      ' plot window before pressing any key.\n')

num_spectra = len(spectra)
indices = list(range(num_spectra))
all_indices = copy.deepcopy(indices)
original_spectra = copy.deepcopy(spectra)
selected_points = []
individual_mode = False
manual_mode = False
in_macro = False
ilog, jlog, idx, k = 0, 0, 0, 0
save_action_log = True
spectra_old = copy.deepcopy(spectra)
data = {'spectra': spectra, 'idx': idx, 'indices': indices,
        'all_indices': all_indices, 'spectra_names': spectra_names,
        'selected_points': selected_points, 'macro_event': False}
data_log = [copy.deepcopy(data)]
action_log = [{'action': 'start'}]
macro_actions = []

fig = plot_data(spectra, spectra_old, idx)
fig.canvas.mpl_connect('button_press_event', click1)
fig.canvas.mpl_connect('button_release_event', click2)
fig.canvas.mpl_connect('key_press_event', press_key)

plt.show()
