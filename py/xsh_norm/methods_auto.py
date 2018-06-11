# -*- coding: utf-8 -*-

__all__ = ["insert_points", "llr", "spline_interpolation", "filtering", "mask"]

import numpy as np
import types

# =======================================----------------------------------------
def f8(seq):
    """Returns array with unique values"""
    seen = set()
    return np.array([i for i, x in enumerate(seq) if x not in seen and not seen.add(x)])
# =======================================----------------------------------------


# =======================================----------------------------------------
def insert_points(pointx, pointy, pointyerr, wave, wave_temp, flux, flux_temp, fluxerror,
                  window_width = 5, spacing = 100,  pick_epsilon = 6):
    """Insert evenly spaced points"""
    if isinstance(pointx, np.ndarray):
        pointx = pointx.tolist()
        pointy = pointy.tolist()
        pointyerr = pointyerr.tolist()

    for i in wave_temp[30+np.random.randint(20):-30+np.random.randint(20):spacing + np.random.randint(spacing/10.0)]:
        try:
            window = np.arange(np.where(wave_temp == i)[0] - window_width, np.where(wave_temp == i)[0]+window_width)
        except TypeError:
            print("TypeError in 'insert_points' when pruning closely spaced points.")
            continue
        # print(window)
        window2 = np.arange(np.where(wave == i)[0]-window_width,np.where(wave == i)[0]+window_width)
        if max(wave_temp[window]) - min(wave_temp[window]) == max(wave[window2]) - min(wave[window2]):
            y_val = np.median(flux_temp[window]).astype(np.float64)
            yerr_val = np.sqrt(np.sum(fluxerror[window] ** 2.)).astype(np.float64)
            pointx.append(i)
            pointy.append(y_val)
            pointyerr.append(yerr_val)

    return pointx, pointy, pointyerr
# =======================================----------------------------------------


# =======================================----------------------------------------        
def spline_interpolation(pointx, pointy, wave, wave_temp, flux, flux_temp, chebfitval, linewidth= 2.0, endpoints = 'y',
                         endpoint_order = 4):
    """Sort spline points and interpolate between marked continuum points"""
    from numpy.polynomial import chebyshev
    from scipy.interpolate import splrep, splev

    # Insert endpoints
    if endpoints == "y" or endpoints == "t":
        sort_array = np.argsort(pointx)
        x = np.array(pointx)[sort_array]
        y = np.array(pointy)[sort_array]
        chebfit = chebyshev.chebfit(x, y, deg = endpoint_order)
        chebfitval = chebyshev.chebval(wave, chebfit)
        i = wave[150], wave[-150]
        window1, window2 = ((i[0]-70) <= wave) & (wave <= (i[0]+70)), ((i[1]-70) <= wave) & (wave <= (i[1]+70))
        y_val = np.median(chebfitval[window1]).astype(np.float64), np.median(chebfitval[window2]).astype(np.float64)
        pointx = np.concatenate([pointx, i])
        pointy = np.concatenate([pointy, y_val])
        ind_uni = f8(pointx)
        pointx = np.array(pointx)[ind_uni]
        pointy = np.array(pointy)[ind_uni]

    # Sort numerically
    # print(pointx)
    sort_array = np.argsort(pointx)
    # print(sort_array, pointx)
    # x, y = pointx[sort_array], pointy[sort_array]
    x = np.array(pointx)[sort_array]
    y = np.array(pointy)[sort_array]
    # Interpolate
    spline = splrep(x, y, k=3)

    def tap(s, x_temp):
        # offset wavelength array to ensure that wavelength does not become negative in the mirror
        diff = max(x_temp) - min(x_temp)
        k = np.tanh((x_temp - min(x_temp)) / (diff/10.0)   )
        return k

    if endpoints == "t":
        continuum = splev(wave, spline) * (tap(10, wave)) * (tap(10, 2 * np.mean(wave) - wave)) + chebfitval * (1 - tap(10, wave)) + chebfitval * (1 - tap(10, 2 * np.mean(wave) - wave))
    else:
        continuum = splev(wave, spline)

    return continuum
# =======================================----------------------------------------


# =======================================----------------------------------------
def filtering(pointx, pointy, pointyerr, wave, wave_temp, flux, flux_temp, linewidth= 2.0, 
                         pick_epsilon = 6, tolerance =0.05, leg_order = 1, division = 50):
    'Filter points by  low-order legendre fitting and clipping values of high sigma iteratively until continuum is found'

    #Ensure uniqueness
    ind_uni = f8(pointx)
    pointx = np.array(pointx)[ind_uni]
    pointy = np.array(pointy)[ind_uni]
    pointyerr = np.array(pointyerr)[ind_uni]
    #Fit with chebyshev polynomial and clip point furthest from the fit to remove points not varying smoothly
    from numpy.polynomial import chebyshev
    from pylab import pause
    # import matplotlib.pyplot as plt

    sigma = np.average(pointy)*tolerance
    chebfitval = 0
    while max(np.sqrt((pointy - chebfitval)**2)) >= abs(sigma):
        sort_array = np.argsort(pointx)
        x = np.array(pointx)[sort_array]
        y = np.array(pointy)[sort_array]
        yerr = np.array(pointyerr)[sort_array]
        try:
            chebfit = chebyshev.chebfit(x, y, deg = leg_order, w=1/yerr)
        except:
            chebfit = chebyshev.chebfit(x, y, deg = leg_order)
        chebfitval = chebyshev.chebval(x, chebfit)
        ind = [i for i, j in enumerate(np.sqrt((pointy - chebfitval)**2)) if j == max(np.sqrt((pointy - chebfitval)**2))]
        pointx = np.delete(pointx, [ind[0]])
        pointy = np.delete(pointy, [ind[0]])
        pointyerr = np.delete(pointyerr, [ind[0]])
        chebfitval = np.delete(chebfitval, [ind[0]])


    min_sep = [min(abs(np.array(pointx)[np.nonzero(abs(np.array(pointx) - j))] - j)) for i, j in enumerate(pointx)]

    while min(min_sep) <= (max(wave) - min(wave))/(division):
        ind_min = np.where(min(min_sep) == min_sep)
        p = np.random.randint(len(ind_min[0]))
        pointx = np.delete(pointx,[(ind_min[0])[p]])
        pointy = np.delete(pointy,[(ind_min[0])[p]])
        pointyerr = np.delete(pointyerr,[(ind_min[0])[p]])
        min_sep = [min(abs(np.array(pointx)[np.nonzero(abs(np.array(pointx) - j))] - j)) for i, j in enumerate(pointx)]

    if len(pointx) >= 6:
        sort_array = np.argsort(pointx)
        x = np.array(pointx)[sort_array]
        y = np.array(pointy)[sort_array]
        yerr = np.array(pointyerr)[sort_array]
        try:
            chebfit = chebyshev.chebfit(x, y, deg = leg_order, w=1/yerr)
        except:
            chebfit = chebyshev.chebfit(x, y, deg = leg_order)
        chebfitval = chebyshev.chebval(x, chebfit)
        ind = [i for i, j in enumerate(np.sqrt((pointy - chebfitval)**2)) if j == max(np.sqrt((pointy - chebfitval)**2))]
        pointx = np.delete(pointx,[ind[0]])
        pointy = np.delete(pointy,[ind[0]])
        pointyerr = np.delete(pointyerr,[ind[0]])

    return pointx, pointy, pointyerr, chebyshev.chebval(wave, chebfit)
# =======================================----------------------------------------


# =======================================----------------------------------------
def mask(pointx, pointy, wave, wave_temp, flux, fluxerror, flux_temp, continuum, chebfitval, linewidth= 0.2,
         exclude_width = 20, sigma_mask = 3, lower_mask_bound = 0):
    """Mask areas where signal is present"""
    from scipy.signal import medfilt as smooth
    difference = smooth(abs(chebfitval - flux), 9)
    ind_err = np.array([i for i, j in enumerate(fluxerror) if difference[i] < sigma_mask*j and flux[i] >= lower_mask_bound])# and 6*fluxerror[i] < flux[i]]) #and np.average(self.difference[i-100:i+100]) <= j] )
    b = exclude_width
    ind_err = np.array([j for i, j in enumerate(ind_err[:-b]) if j + b == ind_err[i+b] and j - b == ind_err[i-b]])
    wave_temp = wave[ind_err]
    flux_temp = flux[ind_err]

    return wave_temp, flux_temp
# =======================================----------------------------------------
