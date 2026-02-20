#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Automatic Ice Composition Estimator (AICE)  v 1.1
------------------------------------------
Inference module

Andrés Megías
"""

config_file = 'J110621.yaml'

# Libraries.
import os
import sys
import copy
import pickle
import warnings
import yaml
import numpy as np
import pandas as pd
import richvalues as rv
import scipy.interpolate
import matplotlib.pyplot as plt

# Functions.

relu = lambda x: np.maximum(0, x)
softplus = lambda x: np.log(1 + np.exp(x))
sigmoid = lambda x: 1 / (1 + np.exp(-x))

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

    # Make arrays of edge positions and widths for the old and new bins
    old_edges, old_widths = make_bins(old_wavs)
    new_edges, new_widths = make_bins(new_wavs)

    # Generate output arrays to be populated
    new_fluxes = np.zeros(old_fluxes[...,0].shape + new_wavs.shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape "
                             "as spec_fluxes.")
        else:
            new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs
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

        # Find first old bin which is partially covered by the new bin
        while old_edges[start+1] <= new_edges[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_edges[stop+1] < new_edges[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
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

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_edges[start+1] - new_edges[j])
                            / (old_edges[start+1] - old_edges[start]))

            end_factor = ((new_edges[j+1] - old_edges[stop])
                          / (old_edges[stop+1] - old_edges[stop]))
            old_widths_local = old_widths[start:stop+1].copy()
            if not supersample_linearly or supersample_linearly and stop != start+1:
                old_widths_local[0] *= start_factor
                old_widths_local[stop-start] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths_local * old_fluxes[...,start:stop+1]
            new_fluxes[...,j] = np.sum(f_widths, axis=-1)
            new_fluxes[...,j] /= np.sum(old_widths_local)

            if old_errs is not None:
                # Case of new bin partially overlapping only one old bin.
                if stop == start+1:
                    # Including old flux value at new left edge.
                    if old_wavs[...,start] == new_edges[...,j]:
                        start = max(0, start-1)
                    # Artificially enlarge uncertainties to be consistent.
                    factor = np.sqrt(np.sum(old_widths[start:stop+1])/new_widths[j])
                else:
                    factor = 1.
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
    """Use SpectRes to resample the input spectrum."""
    ssl = supersample_linearly
    result = spectres(x_new, x, y, y_unc, fill=np.nan,
                      supersample_linearly=ssl, verbose=False)
    return result

def fill_spectrum(x, y, y_unc=None, threshold=1.5):
    """Fill the gaps in the input spectrum with NaNs."""
    x_ = np.array([], float)
    y_ = np.array([], float)
    dx = np.median(np.diff(x))
    for i in range(len(x)-1):
        if x[i+1] - x[i] > threshold * dx:
            x_range = x[i+1] - x[i]
            num_points = round(x_range / dx)
            dx_ = x_range / num_points
            x_i = np.arange(x[i]+dx, x[i+1]-dx, dx_)
            y_i = np.nan * np.ones(len(x_i))
            x_ = np.append(x_, x_i)
            y_ = np.append(y_, y_i)
    x_new = np.append(x, x_)
    y_new = np.append(y, y_)
    inds = np.argsort(x_new)
    if y_unc is not None:
        y_unc_new = np.append(y_unc, y_)
        y_unc_new = y_unc_new[inds]
    x_new = x_new[inds]
    y_new = y_new[inds]
    return x_new, y_new, y_unc_new

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
        spl = scipy.interpolate.PchipInterpolator(x_, y_)
    elif interpolation == 'spline':
        y_s = rv.rolling_function(np.median, y_, smooth_size)
        s = np.nansum((y_s-y_)**2)
        k = len(y_)-1 if len(y_) <= 3 else 3
        spl = scipy.interpolate.UnivariateSpline(x_, y_, s=s, k=k)
    yb = spl(x)
    return yb

def gaussian(x, x0, s, h):
    """Gaussian with given center (x0), width (s) and height (h)."""
    y = h * np.exp(-0.5*((x-x0)/s)**2)
    return y

def correct_saturation(x, y, windows='auto'):
    """Correct saturation in input spectrum for H2O, CO and CO2 bands."""
    if windows == 'auto':
        windows = [[2367., 2328.], [2146., 2135.]]
    ys = rv.rolling_function(np.mean, y, size=3)
    y_new = copy.copy(y)
    for x1x2 in windows:
        x1, x2 = min(x1x2), max(x1x2)
        mask = (x >= x1) & (x <= x2)
        x_ = x[mask]
        y_ = ys[mask]
        center = 0.5 * (x1 + x2)
        width = 0.25 * (x2 - x1)
        height = np.nanmax(y_)
        x1 = None
        for i in range(len(x_)-1):
            if y_[i+1] < y_[i] or np.isnan(y_[i+1]):
                x1 = x_[i]
                break
        x2 = None
        for i in range(len(x_)-1):
            if y_[-i-2] < y_[-i-1] or np.isnan(y_[-i-2]):
                x2 = x_[-i]
                break
        if x1 is not None and x2 is not None:
            mask_ = (x_ <= x1) | (x_ >= x2) 
        else:
            mask_ = np.ones(len(x_), bool)
        mask_ &= np.isfinite(y_)
        y_ = y[mask]
        guess = [center, width, height]
        try:
            params = scipy.optimize.curve_fit(gaussian, x_[mask_], y_[mask_],
                                              p0=guess)[0]
        except:
            return y
        if x1 is not None and x2 is not None:
            mask = (x >= x1) & (x <= x2)
            x_ = x[mask]
        y_curve = gaussian(x_, *params)
        y_new[mask] = np.maximum(y_new[mask], y_curve)
    return y_new

def interpolate_nans(x, y, smooth_size, interpolation='pchip'):
    """Fill NaNs in input spectrum by interpolating the surrounding points."""
    y_new = copy.copy(y)
    mask = np.isnan(y)
    if any(mask):
        baseline = fit_baseline(x, y, smooth_size, interpolation=interpolation)
        y_new[mask] = baseline[mask]
    return y_new

def extract_spectrum(x, y, windows):
    """Extract input spectrum in given windows, deleting the rest."""
    y_new = np.zeros(len(x), float)
    mask = np.zeros(len(x), bool)
    for x1x2 in windows:
        x1, x2 = min(x1x2), max(x1x2)
        mask |= (x >= x1) & (x <= x2)
    y_new[mask] = y[mask]
    y_new /= np.mean(y_new)
    return y_new

def aice_model(wavenumber, absorbance, weights, model_info,
               correct_co=True, return_extra_info=False):
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
    def nn_model(x, weights, end_act):
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
    wavenumber_aice = model_info['wavenumber']
    aice_resolution = model_info['resolution']
    aice_spacing = model_info['spacing']
    absorbance_ = correct_saturation(wavenumber, absorbance)
    absorbance_ = interpolate_nans(wavenumber, absorbance_, smooth_size=15,
                                   interpolation='pchip')
    absorbance_aice = resample_spectrum(wavenumber_aice, wavenumber, absorbance_,
                                        supersample_linearly=True)
    size = round(aice_resolution / aice_spacing)
    absorbance_aice = pd.Series(absorbance_aice).rolling(size, min_periods=1,
                                                     center=True).mean().values
    absorbance_aice = np.nan_to_num(absorbance_aice, nan=0.)
    absorbance_aice /= np.mean(absorbance_aice)
    results = []
    for j in range(weights.shape[0]):
        yj = np.zeros(weights.shape[1])
        for i in range(len(yj)):
            end_act = softplus if i == 0 else sigmoid
            yj[i] = nn_model(absorbance_aice, weights[j,i], end_act)[0]
        results += [yj]
    results = np.array(results)
    stdevs = np.std(results, axis=0)
    predictions = np.mean(results, axis=0)
    if correct_co and 'CO' in variables and 'CO2' in variables:
        idx_co2 = np.argwhere(np.equal(variables, 'CO2'))[0,0]
        preds_co2 = results[:,idx_co2]
        pred_co2 = np.mean(results[:,idx_co2])
        pred_co2_unc = np.std(results[:,idx_co2])
        if pred_co2 > 0.03 and pred_co2_unc / pred_co2 < 0.5:
            windows = [[2375., 2310.], [2150., 2120.]]
            idx_co = np.argwhere(np.equal(variables, 'CO'))[0,0]
            absorbance_aice_ = extract_spectrum(wavenumber_aice, absorbance_aice,
                                                windows)
            preds_co_co2 = []
            for j in range(weights.shape[0]):
                pred_co = nn_model(absorbance_aice_, weights[j,idx_co], sigmoid)[0]
                pred_co2 = nn_model(absorbance_aice_, weights[j,idx_co2], sigmoid)[0]
                preds_co_co2 += [pred_co / pred_co2]
            preds_co = np.array(preds_co_co2) * preds_co2
            results[:,idx_co] = preds_co
    predictions = np.mean(results, axis=0)
    stdevs = np.std(results, axis=0)
    if return_extra_info:
        return predictions, stdevs, results
    else:
        return predictions

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

def axis_conversion(x):
    """Axis conversion from wavenumber to wavelength and viceversa"""
    with np.errstate(divide='ignore'):
        y = 1e4 / x
    return y

#%% Initial options.

print('------------------------------------------')
print('Automatic Ice Composition Estimator (AICE)')
print('------------------------------------------')
print('')

# Default options.
default_options = {
    'input spectrum': '',
    'column indices': {'x': 1, 'y': 2, 'y unc.': 3},
    'figure size': [12., 5.],
    'save results': True,
    'output file': 'auto',
    'model file': os.path.join('..', 'training', 'models', 'aice-model.pkl'),
    'spectral variable': 'wavenumber (/cm)',
    'intensity variable': 'absorbance',
    'saturated regions (/cm)': [],
    'spectrum vertical range': None,
    'molecular fraction range': [0., 1.0],
    'temperature range (K)': [0., 100.],
    'show predictions for all submodels': False,
    'propagate observational uncertainties': False,
    'normalization': 'total',
    'reference values': {},
    'used reference values': [],
    'reference colors': ['orchid', 'orange', 'palevioletred'],
    'AICE label': 'AICE'
    }
formatted_names = {'H2O': 'H$_2$O', 'CO': 'CO', 'CO2': 'CO$_2$',
                   'CH3OH': 'CH$_3$OH', 'NH3': 'NH$_3$', 'CH4': 'CH$_4$'}

# Configuration file.
spectrum_name = config_file.replace('.yaml', '').replace('.', '')
config_file = (os.path.realpath(config_file) if len(sys.argv) == 1
               else os.path.realpath(sys.argv[1]))
with open(config_file) as file:
    config = yaml.safe_load(file)
config = {**default_options, **config}

# Options.
filename = config['input spectrum']
column_inds = config['column indices']
figsize = config['figure size']
model_path = config['model file']
spectral_variable = config['spectral variable']
intensity_variable = config['intensity variable']
saturated_regions = config['saturated regions (/cm)']
frac_range = config['molecular fraction range']
absorb_range = config['spectrum vertical range']
temp_range = config['temperature range (K)']
normalization = config['normalization']
all_references = config['reference values']
used_references = config['used reference values']
aice_label = config['AICE label']
show_individual_predictions = config['show predictions for all submodels']
propagate_obs_uncs = config['propagate observational uncertainties']
ref_colors = config['reference colors']
save_results = config['save results']
output_file = config['output file']
# Some checks.
show_references = used_references != []
idx_x, idx_y = column_inds['x'] - 1, column_inds['y'] - 1
idx_y_unc = column_inds['y unc.'] - 1 if 'y unc.' in column_inds else None

# Model info and weights.
with open(model_path, 'rb') as file:
   weights, model_info = pickle.load(file)
if len(weights.shape) == 2:
    weights = weights.reshape(1,*weights.shape)
x1x2 = model_info['wavenumber_range']
x1, x2 = min(x1x2), max(x1x2)
dx = model_info['spacing']
wavenumber = wavenumber_aice = np.arange(x1, x2, dx)
wavenumber_range = (x2, x1)
model_info['wavenumber'] = wavenumber_aice
variables = model_info['variables']
species = [var for var in variables if 'temp' not in var]

#%% Loading of files.

# Reading of input spectrum.
if '.csv' in filename:
    data = pd.read_csv(file).values
    x, y = data[:,[idx_x, idx_y]]
    data[:,idx_y_unc] if idx_y_unc is not None else np.zeros(len(y))
else:
    data = np.loadtxt(filename)
x = data[:,idx_x]
y = data[:,idx_y]
y_unc = data[:,idx_y_unc] if idx_y_unc is not None else np.zeros(len(y))
if spectral_variable == 'wavelength (μm)':
    x = 1e4 / x
if intensity_variable == 'optical depth':
    y /= np.log(10)
    y_unc /= np.log(10)
inds = np.argsort(x)
x = x[inds]
y = y[inds]
y_unc = y_unc[inds]
# x, y, y_unc = fill_spectrum(x, y, y_unc)

# Resampling.
wavenumber_orig = copy.copy(x)
absorbance_orig = copy.copy(y)
absorbance_orig_unc = copy.copy(y_unc)
smooth_size = int(round(model_info['resolution'] / model_info['spacing']))
y_ = interpolate_nans(x, y, smooth_size=5, interpolation='pchip')
y_res, y_res_unc = resample_spectrum(wavenumber, x, y_, y_unc,
                                     supersample_linearly=True)
y_res = np.nan_to_num(y_res, nan=0.)
y_res = rv.rolling_function(np.mean, y_res, smooth_size)
desaturate_regions = saturated_regions is not None
if desaturate_regions:
    y_desat = correct_saturation(x, y_, saturated_regions)
    y_desat = interpolate_nans(x, y_desat, smooth_size=5, interpolation='pchip')
    y_desat_res, y_res_unc = resample_spectrum(wavenumber, x, y_desat, y_unc,
                                                     supersample_linearly=True)
    y_desat_res = np.nan_to_num(y_desat_res, nan=0.)
    y_desat_res = rv.rolling_function(np.mean, y_desat_res, smooth_size)
    absorbance = y_desat_res
    absorbance_sat = y_res
else:
    absorbance = y_res
absorbance_unc = y_res_unc
# y_aice = np.nan_to_num(y_aice, nan=0.)

# Normalization.
norm = np.mean(absorbance)
absorbance /= norm
absorbance_unc /= norm
absorbance_orig /= norm
absorbance_orig_unc /= norm
if desaturate_regions:
    absorbance_sat /= norm

print(f'Read spectrum in file {filename}.')
print()

#%% Calculations of AICE.

# Predictions.
predictions, stdevs, predictions_all = aice_model(wavenumber, absorbance,
                                    weights, model_info, return_extra_info=True)
if propagate_obs_uncs:
    print('Propagating observational uncertainties...\n')
    aice_model_ = lambda absorbance_orig: aice_model(wavenumber_orig, absorbance_orig,
                                                     weights, model_info)
    absorbance_orig_rv = rv.RichArray(absorbance_orig, absorbance_orig_unc)
    predictions_rv = rv.array_function(aice_model_, absorbance_orig_rv,
                                       domain=[0,np.inf], len_samples=400,
                                       consider_intervs=False)
    obs_uncs = predictions_rv.uncs
if (propagate_obs_uncs and any(rv.isnan(predictions).flatten())
        or not propagate_obs_uncs):
    obs_uncs = 0.

# Uncertainty estimation.
stdevs = np.array([stdevs, stdevs]).T  # deviation of nn-predictions
uncs = (obs_uncs**2 + stdevs**2)**0.5
predictions = rv.RichArray(predictions, uncs, domains=[0.,np.inf])

# Preparation of results dataframe.
variables = [var.replace('temp', 'temp. (K)') for var in variables]
# if normalization == 'total':
#     variables += ['all']
#     predictions = np.append(predictions, predictions[1:].sum())
results_df = rv.rich_dataframe({aice_label: predictions}, index=variables).T

# Normalization.
if normalization != 'total':
    norm_results = {aice_label: results_df[normalization].values[0]}
    other_species = [name for name in species if name != normalization]
    results_df[other_species] = (results_df[other_species].values
                                 / results_df[normalization].values)

# Adjustment of results dataframe.
results_df = results_df.T

# Implementation of input references in results dataframe.
if show_references:
    references = {}
    for key1 in all_references:
        if any([key1.startswith(key2) for key2 in used_references]):
            references[key1] = all_references[key1]
    for key in references:
        reference_df = rv.rich_dataframe(references[key], index=[0],
                                         domains=[0,np.inf])
        norm = 0.
        for name in reference_df:
            if 'temp' not in name:
                norm += reference_df[name].values[0]
        for name in reference_df:
            if 'temp' not in name:
                if name not in species:
                    reference_df = reference_df.drop(columns=name)
                    continue
                reference_df[name] /= norm
        for name in results_df[aice_label].keys():
            if name not in reference_df.columns:
                reference_df[name] = rv.rval('nan')
        if normalization != 'total':
            norm_results[key] = reference_df[normalization].values[0]
            with np.errstate(divide='ignore', invalid='ignore'):
                reference_df[other_species] = (reference_df[other_species].values
                                               / reference_df[normalization].values)
        reference_df = reference_df.transpose()
        reference_df.columns = [key]
        results_df = pd.concat((results_df, reference_df), axis=1)
        
# Change of names for normalized fractions.
if normalization != 'total':
    old_vars = list(results_df.index)
    new_vars = []
    for name in old_vars:
        if name == normalization or 'temp' in name:
            new_vars += [name]
        else:
            new_vars += [f'{name}/{normalization}']
    results_df.index = new_vars
        
#%% Displaying predictions and plots.

plt.close('all')    
print('Predictions for the ice composition:')
print()

# Result dataframe.
with pd.option_context('display.max_columns', 4):
    print(results_df, '\n')
if normalization == 'total':
    print(f'Sum of predicted molecules: {results_df.values[1:,0].sum()}')

# Graphic options.
if show_references:  # offsets for predictions
    refs_names = list(references.keys())
    num_refs = len(refs_names)
    if num_refs == 1:
        offsets = {aice_label: -0.15, refs_names[0]: 0.15}
    elif num_refs == 2:
        offsets = {aice_label: 0, refs_names[0]: -0.15, refs_names[1]: 0.15}
    elif num_refs == 3:
        offsets = {aice_label: -0.10, refs_names[0]: 0.10,
                   refs_names[1]: -0.30, refs_names[2]: 0.30}
else:
    offsets = {aice_label: 0}
fig = plt.figure('AICE', figsize=figsize)   # figure dimensions and ratios
plt.clf()
width_ratios = [2.,1.] if normalization == 'total' else [3.,2.]
width_ratios[-1] = width_ratios[-1] - (6. - len(species))/6.
gs = plt.GridSpec(1, 2, width_ratios=width_ratios, wspace=0.18,
                  left=0.07, right=0.93, bottom=0.15, top=0.82)

# Plot of the input spectrum.
fig.add_subplot(gs[0,0])
plt.errorbar(wavenumber_orig, absorbance_orig, absorbance_orig_unc,
             linewidth=1., drawstyle='steps-mid', color='dimgray',
             ecolor=[0.75]*3, label='original spectrum')
plt.fill_between(wavenumber_orig, absorbance_orig-absorbance_orig_unc,
                 absorbance_orig+absorbance_orig_unc, color=[0.75]*3, step='mid')
plt.errorbar(wavenumber, absorbance, absorbance_unc, linewidth=1.,
             drawstyle='steps-mid', color='black', ecolor=[0.20]*3,
             label='resampled spectrum')
if desaturate_regions:
    mask = absorbance_sat != absorbance
    x_ = wavenumber[mask]
    y_ = absorbance[mask]
    y_unc_ = absorbance_unc[mask]
    x_, y_, y_unc_ = fill_spectrum(x_, y_, y_unc_)
    plt.errorbar(x_, y_, y_unc_, linewidth=1., drawstyle='steps-mid',
                 color='darkred', ecolor=[0.20]*3, label='desaturated parts')
plt.axhline(y=0, color='k', ls='--', lw=0.6)
if absorb_range is not None:
    plt.ylim(absorb_range)
plt.xlim(wavenumber_range[0], wavenumber_range[1])
plt.xlabel('wavenumber (cm$^{-1}$)', labelpad=8,)
plt.ylabel('normalised absorbance', labelpad=10)
plt.legend(loc='upper right')
ax = plt.gca()  
ax2 = ax.secondary_xaxis('top', functions=(axis_conversion, axis_conversion))
ax2.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
               [2, 3, 4, 5, '', 7, '', '', 10, 20, 30])
ax2.set_xlabel('wavelength (μm)', labelpad=10, fontsize=9)

# Predicted fraction for normalization molecule.
if normalization != 'total':
    norm_results = pd.DataFrame(norm_results, index=[normalization])
    gs = plt.matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, gs[1],
                                               width_ratios=[1.,5.], wspace=0.5)
    fig.add_subplot(gs[0,0])
    rv.errorbar([0. + offsets[aice_label]], norm_results[aice_label],
                fmt='.', color='black')
    plt.bar([0], norm_results[aice_label].values[0].main,
            edgecolor='black', color='gray')
    if show_references:
        for (i, key) in enumerate(references):
            reference = norm_results[key].values
            rv.errorbar([0. + offsets[key]], reference[0], color=ref_colors[i])
    plt.margins(x=0.7)
    plt.ylim(frac_range)
    plt.xticks([0], [formatted_names[normalization]], rotation=45.,
               fontsize=10.)
    plt.ylabel('molecular fraction')
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
gs = plt.matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, gs[1],
                                            width_ratios=[8.,1.], wspace=0.15)
# Predictions bar plot.
fig.add_subplot(gs[0,0])
predictions = []
for name in results_df[aice_label].index:
    if name != normalization:
        predictions += [results_df[aice_label][name]]
predictions = rv.rarray(predictions)
species = [name for name in species if name != normalization]
labels = [formatted_names[name] for name in species]
positions = np.arange(len(labels))
plt.bar(positions, predictions.mains[1:], width=0.6,  # ice composition
        edgecolor='black', color='gray')
rv.errorbar(positions + offsets[aice_label], predictions[1:], fmt='.',
            color='black')
plt.plot([], [], '.', color='black', label='AICE (this work)')
if show_individual_predictions:
    num_seeds = len(predictions_all)
    for (i, predictions_i) in enumerate(predictions_all):
        offset_i = (i / (num_seeds-1) - 0.5) * 0.6
        plt.plot(positions + offset_i, predictions_i[1:], '.',
                 color='black', alpha=0.3)
if show_references:  # reference
    for (i,key) in enumerate(references):
        reference = []
        for name in (results_df[key].index):
            if name != normalization and 'temp' not in name:
                reference += [results_df[key][name]]
        rv.errorbar(positions + offsets[key], reference, color=ref_colors[i])
        plt.plot([], [], '.', color=ref_colors[i], label=key)
plt.xticks(positions, labels, rotation=45., fontsize=10.)
plt.xlim(positions[0]-0.7, positions[-1]+0.7)
for tick in plt.gca().xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
plt.ylim(frac_range)
ylabel = ('molecular fraction' if normalization == 'total' else
          'abundance with respect to '+ formatted_names[normalization])
plt.ylabel(ylabel, labelpad=4)
plt.legend(fontsize=8)
title = 'AICE predictions  '
if normalization != 'total':
    title += ' '*26
plt.title(title, x=0.55, pad=16.)
fig.add_subplot(gs[0,1])  # temperature
plt.bar([1.], predictions[0].main, edgecolor='black', color='tab:gray')
rv.errorbar([1. + offsets[aice_label]], predictions[0], color='black')
if show_individual_predictions:
    num_seeds = len(predictions_all)
    for (i, predictions_i) in enumerate(predictions_all):
        offset_i = (i / (num_seeds-1) - 0.5) * 0.6
        plt.plot(1. + offset_i, predictions_i[0], '.',
                 color='black', alpha=0.3)
if show_references:  # reference
    for (i,key) in enumerate(references):
        reference = results_df[key].values
        if rv.isfinite(reference[0]):
            rv.errorbar([1. + offsets[key]], reference[0], color=ref_colors[i])
plt.ylim(temp_range)
plt.locator_params(axis='y', nbins=5)
plt.xticks([1], ['$T_\\mathrm{ice}$'], rotation=0., fontsize=10.)
plt.tick_params(axis='x', which='major', pad=8.)
for tick in plt.gca().xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position('right')
plt.ylabel('temperature (K)')
plt.margins(x=0.9)

plt.suptitle(spectrum_name, y=0.94, fontweight='bold')

# plt.tight_layout()

plt.show()

#%% Save output.

if save_results:
    if output_file == 'auto':
        output_file = config_file.replace('.yaml', '-aice-results.csv')
    results_df.to_csv(output_file)
    print(f'\nSaved results in {output_file}.')