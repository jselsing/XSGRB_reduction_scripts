#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from scipy.interpolate import interp1d
from util import *
from scipy import optimize
import lmfit
# Plotting
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')


def load_array(data_array):
    return data_array[:, 1], data_array[:, 2], data_array[:, 3], data_array[:, 4]


def get_overlap(wave1, wave2):
    # Get values at ends
    bounds1 = [min(wave1), max(wave1)]
    bounds2 = [min(wave2), max(wave2)]
    # Find overlapping indices
    wave1_out = np.where((wave1 < bounds2[0]))
    overlap1 = np.where((wave1 >= bounds2[0]) & (wave1 <= bounds2[1]))
    overlap2 = np.where((wave2 >= bounds1[0]) & (wave2 <= bounds1[1]))
    wave2_out = np.where((wave2 > bounds1[1]))
    return wave1_out, (overlap1, overlap2), wave2_out


def interpspec(wave_input, spectrum_input, error_input, wave_target):
    # Interpolate input onto new grid
    f = interp1d(wave_input, spectrum_input)
    g = interp1d(wave_input, error_input)
    return f(wave_target), g(wave_target)


def stitch_XSH_spectra(waves, fluxs, errors, scale=None):

    # Find overlaps
    UVB, overlapUVBVIS, VIS1 = get_overlap(waves[0], waves[1])
    VIS2, overlapVISNIR, NIR = get_overlap(waves[1], waves[2])
    VIS = np.arange(min(*VIS1), max(*VIS2) + 1)

    # Cut flux and error in uvb overlap
    f_vis = fluxs[1][overlapUVBVIS[1]][~np.isnan(fluxs[1][overlapUVBVIS[1]])]
    e_vis = errors[1][overlapUVBVIS[1]][~np.isnan(errors[1][overlapUVBVIS[1]])]
    f_uvb = fluxs[0][overlapUVBVIS[0]][~np.isnan(fluxs[0][overlapUVBVIS[0]])][:len(f_vis)]
    e_uvb = errors[0][overlapUVBVIS[0]][~np.isnan(errors[0][overlapUVBVIS[0]])][:len(f_vis)]

    # Offset range
    look_range = np.arange(0, 3, 0.001)

    # Get uvb offset
    uvbres = [np.sum(((f_uvb*ii - f_vis)**2)/(e_uvb**2 + e_vis**2)) for ii in look_range]
    uvb_offset = abs(look_range[np.where(uvbres == min(uvbres))[0]])
    offset_UVB = np.average(f_uvb, weights=1/(e_uvb)**2)/np.average(f_vis, weights=1/(e_vis)**2)

    # Cut flux and error in nir overlap
    f_nir = fluxs[2][overlapVISNIR[1]][~np.isnan(fluxs[2][overlapVISNIR[1]])]
    e_nir = errors[2][overlapVISNIR[1]][~np.isnan(errors[2][overlapVISNIR[1]])]
    f_vis = fluxs[1][overlapVISNIR[0]][~np.isnan(fluxs[1][overlapVISNIR[0]])][::3][:len(f_nir)]
    e_vis = errors[1][overlapVISNIR[0]][~np.isnan(errors[1][overlapVISNIR[0]])][::3][:len(f_nir)]

    # Get nir offset
    nirres = [np.sum(((f_vis*ii - f_nir)**2)/(e_vis**2 + e_nir**2)) for ii in look_range]
    nir_offset = abs(look_range[np.where(nirres == min(nirres))[0]])
    offset_NIR = np.average(f_vis, weights=1/(e_vis)**2)/np.average(f_nir, weights=1/(e_nir)**2)

    print(1/offset_UVB, 1/offset_NIR)
    print(1/uvb_offset, 1/nir_offset)

    # Apply correction
    if scale:
        fluxs[0] /= offset_UVB
        fluxs[2] *= offset_NIR


    # Get wl, flux  and error in overlaps
    UVB_overlap_flux, UVB_overlap_error = interpspec(waves[0], fluxs[0], errors[0], waves[1][overlapUVBVIS[1]])
    VIS1_overlap_flux, VIS1_overlap_error = fluxs[1][overlapUVBVIS[1]], errors[1][overlapUVBVIS[1]]
    VIS2_overlap_flux, VIS2_overlap_error = interpspec(waves[1], fluxs[1], errors[1], waves[2][overlapVISNIR[1]])
    NIR_overlap_flux, NIR_overlap_error = fluxs[2][overlapVISNIR[1]], errors[2][overlapVISNIR[1]]

    # Get weighted average in overlap between UVB and VIS, using the VIS sampling
    UVBVIS_wl = waves[1][overlapUVBVIS[1]]
    UVBVIS_flux, UVBVIS_error = np.zeros_like(UVBVIS_wl), np.zeros_like(UVBVIS_wl)
    for ii, (kk, ll) in enumerate(zip(VIS1_overlap_flux, VIS1_overlap_error)):
        UVBVIS_flux[ii], UVBVIS_error[ii] = weighted_avg(np.array([UVB_overlap_flux[ii], kk]), np.array([UVB_overlap_error[ii], ll]), axis = 0)

    # Get weighted average in overlap between VIS and NIR, using the NIR sampling
    VISNIR_wl = waves[2][overlapVISNIR[1]]
    VISNIR_flux, VISNIR_error = np.zeros_like(VISNIR_wl), np.zeros_like(VISNIR_wl)
    for ii, (kk, ll) in enumerate(zip(NIR_overlap_flux, NIR_overlap_error)):
        VISNIR_flux[ii], VISNIR_error[ii] = weighted_avg(np.array([VIS2_overlap_flux[ii], kk]), np.array([VIS2_overlap_error[ii], ll]), axis = 0)

    wl = np.concatenate((waves[0][UVB], UVBVIS_wl, waves[1][VIS], VISNIR_wl, waves[2][NIR]))
    flux = np.concatenate((fluxs[0][UVB], UVBVIS_flux, fluxs[1][VIS], VISNIR_flux, fluxs[2][NIR]))
    error = np.concatenate((errors[0][UVB], UVBVIS_error, errors[1][VIS], VISNIR_error, errors[2][NIR]))

    return wl, flux, error


def main():
    # Small script to stitch X-shooter arms. Inspired by https://github.com/skylergrammer/Astro-Python/blob/master/stitch_spec.py

    # Load data from individual files
    data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/GRB130408A/"
    scale = True

    data = np.genfromtxt(data_dir + "UVBOB1skysuboptext.dat")
    UVB_wl, UVB_flux, UVB_error, UVB_bp = load_array(data)
    data = np.genfromtxt(data_dir + "VISOB1skysuboptext.dat")
    VIS_wl, VIS_flux, VIS_error, VIS_bp = load_array(data)
    data = np.genfromtxt(data_dir + "NIROB1skysuboptext.dat")
    NIR_wl, NIR_flux, NIR_error, NIR_bp = load_array(data)

    # Construct lists
    waves = [UVB_wl[UVB_wl > 3200], VIS_wl, NIR_wl]
    flux = [UVB_flux[UVB_wl > 3200], VIS_flux, NIR_flux]
    error = [UVB_error[UVB_wl > 3200], VIS_error, NIR_error/10]
    bpmap = [UVB_bp[UVB_wl > 3200], VIS_bp, NIR_bp]

    # Stitch!
    wl, flux, error = stitch_XSH_spectra(waves, flux, error, scale = scale)
    np.savetxt(data_dir + "stitched_spectrum.dat", zip(wl, flux, error), fmt = ['%10.6e', '%10.6e', '%10.6e'], header=" wl flux error")

    hbin = 10
    wl_bin, flux_bin, error_bin = bin_spectrum(wl, flux, error, hbin)
    np.savetxt(data_dir + "stitched_spectrum_bin"+str(hbin)+".dat", zip(wl, flux, error), fmt = ['%10.6e', '%10.6e', '%10.6e'], header=" wl flux error")
    pl.errorbar(wl_bin[::1], flux_bin[::1], yerr=error_bin[::1], fmt=".k", capsize=0, elinewidth=0.5, ms=3, alpha=0.3)
    pl.plot(wl_bin, flux_bin, linestyle="steps-mid", lw=0.3, alpha=0.5)

    def pow(x, y, z):
        return y * x ** z

    popt, pcov = optimize.curve_fit(pow, wl[~np.isnan(flux)][wl < 9000], flux[~np.isnan(flux)][wl < 9000], sigma=error[~np.isnan(flux)][wl < 9000], maxfev=5000, p0 = [5e-11, -1.6])
    pl.plot(wl_bin, popt[0] * wl_bin ** (popt[1]))

    scale = np.median(flux_bin[~np.isnan(flux_bin)])
    pl.xlim(2500, 25500)
    pl.ylim(-1 * scale, 7.5 * scale)
    pl.xlabel(r"Wavelength / [$\mathrm{\AA}$]")
    pl.ylabel(r'Flux density [erg s$^{-1}$ cm$^{-1}$ $\AA^{-1}$]')
    pl.savefig(data_dir + "stitched_spectrum_bin"+str(hbin)+".pdf")
    pl.show()


if __name__ == '__main__':
    main()