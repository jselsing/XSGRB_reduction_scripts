#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function


# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import numpy as np
from scipy.interpolate import interp1d
from util import *
from scipy import optimize
from scipy.signal import medfilt
import lmfit
# Plotting
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')
from numpy.polynomial import chebyshev
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

def pow(x, y, z):
    return y * x ** z


def load_array(data_array, tell = None):

    # Check is reposnse has been written based on value in sixt column
    has_resp = False
    # if np.nanmean(data_array[:, 6]) < 1e-10:
    #     has_resp = True

    if has_resp:
        try:
            tellcorr = data_array[:, 8]
            tellcorr[np.isnan(tellcorr)] = 1
            slitcorr = np.median(data_array[:, 7])
            return data_array[:, 1], data_array[:, 2]*tellcorr*slitcorr, data_array[:, 3]*tellcorr*slitcorr, data_array[:, 4]
        except:
            slitcorr = np.median(data_array[:, 7])
            return data_array[:, 1], data_array[:, 2]*slitcorr, data_array[:, 3]*slitcorr, data_array[:, 4]
    elif not has_resp:
        try:
            tellcorr = tell#data_array[:, 7]
            # print(tellcorr)
            tellcorr[np.isnan(tellcorr)] = 1
            slitcorr = 1#np.median(data_array[:, 6])
            return data_array[:, 1], data_array[:, 2]*tellcorr*slitcorr, data_array[:, 3]*tellcorr*slitcorr, data_array[:, 4]
        except:
            slitcorr = 1#np.median(data_array[:, 6])
            return data_array[:, 1], data_array[:, 2]*slitcorr, data_array[:, 3]*slitcorr, data_array[:, 4]


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


def interpspec(wave_input, spectrum_input, error_input, bpmap_input, wave_target):
    # Interpolate input onto new grid
    f = interp1d(wave_input, spectrum_input)
    g = interp1d(wave_input, error_input)
    h = interp1d(wave_input, bpmap_input)
    return f(wave_target), g(wave_target), h(wave_target)


def stitch_XSH_spectra(waves, fluxs, errors, bpmaps, scale=None):

    # Find overlaps
    UVB, overlapUVBVIS, VIS1 = get_overlap(waves[0], waves[1])
    VIS2, overlapVISNIR, NIR = get_overlap(waves[1], waves[2])
    VIS = np.arange(min(*VIS1), max(*VIS2) + 1)

    # # Get uvb offset
    # p0 = [1e-15, -0.5]
    # UVB_mask = ((waves[0] > 5000) & (waves[0] < 5500))
    # epsilon, diff, msk = np.median(errors[0][UVB_mask]), 2*np.median(errors[0][UVB_mask]), np.ones_like(errors[0][UVB_mask]).astype("bool")
    # ii = 0
    # try:
    #     while diff > epsilon and ii < 5:
    #         popt_UVB, pcov = optimize.curve_fit(pow, waves[0][UVB_mask][msk], fluxs[0][UVB_mask][msk], sigma=errors[0][UVB_mask][msk], p0=p0, maxfev=5000)
    #         msk = abs(pow(waves[0][UVB_mask], *popt_UVB) - fluxs[0][UVB_mask]) < 10*errors[0][UVB_mask]
    #         diff = np.median(abs(pow(waves[0][UVB_mask], *popt_UVB) - fluxs[0][UVB_mask]))
    #         ii += 1


    #     VIS_mask = (waves[1] > 6000) & (waves[1] < 7500)
    #     epsilon, diff, msk = np.median(errors[1][VIS_mask]), 2*np.median(errors[1][VIS_mask]), np.ones_like(errors[1][VIS_mask]).astype("bool")
    #     ii = 0
    #     while diff > epsilon and ii < 5:
    #         popt_VIS, pcov = optimize.curve_fit(pow, waves[1][VIS_mask][msk], fluxs[1][VIS_mask][msk], sigma=errors[1][VIS_mask][msk], p0=p0, maxfev=5000)
    #         msk = abs(pow(waves[1][VIS_mask], *popt_VIS) - fluxs[1][VIS_mask]) < 10*errors[1][VIS_mask]
    #         diff = np.median(abs(pow(waves[1][VIS_mask], *popt_VIS) - fluxs[1][VIS_mask]))
    #         ii += 1

    #     off_UVB = np.median(pow(waves[1][overlapUVBVIS[1]], *popt_VIS)) / np.median(pow(waves[0][overlapUVBVIS[0]], *popt_UVB))
    # except:
    #     print("Fitting failed ... Skipping UVB scaling")
    #     off_UVB = 1

    # # Get nir offset
    # VIS_mask = (waves[1] > 7700) & (waves[1] < 8700)
    # epsilon, diff, msk = np.median(errors[1][VIS_mask]), 2*np.median(errors[1][VIS_mask]), np.ones_like(errors[1][VIS_mask]).astype("bool")
    # ii = 0
    # while diff > epsilon and ii < 5:
    #     popt_VIS, pcov = optimize.curve_fit(pow, waves[1][VIS_mask][msk], fluxs[1][VIS_mask][msk], sigma=errors[1][VIS_mask][msk], p0=p0, maxfev=5000)
    #     msk = abs(pow(waves[1][VIS_mask], *popt_VIS) - fluxs[1][VIS_mask]) < 10*errors[1][VIS_mask]
    #     diff = np.median(abs(pow(waves[1][VIS_mask], *popt_VIS) - fluxs[1][VIS_mask]))
    #     ii += 1

    # NIR_mask = ((waves[2] > 10300) & (waves[2] < 10900)) | ((waves[2] > 15000) & (waves[2] < 17000))

    # epsilon, diff, msk = np.median(errors[2][NIR_mask]), 2*np.median(errors[2][NIR_mask]), np.ones_like(errors[2][NIR_mask]).astype("bool")
    # ii = 0
    # while diff > epsilon and ii < 5:
    #     popt_NIR, pcov = optimize.curve_fit(pow, waves[2][NIR_mask][msk], fluxs[2][NIR_mask][msk], sigma=errors[2][NIR_mask][msk], p0=p0, maxfev=5000)
    #     msk = abs(pow(waves[2][NIR_mask], *popt_NIR) - fluxs[2][NIR_mask]) < (10 - ii)*errors[2][NIR_mask]
    #     diff = np.median(abs(pow(waves[2][NIR_mask], *popt_NIR) - fluxs[2][NIR_mask]))
    #     ii += 1

    # # pl.plot(waves[1][overlapVISNIR[0]], pow(waves[1][overlapVISNIR[0]], *popt_VIS))
    # # pl.plot(waves[2][overlapVISNIR[1]], pow(waves[2][overlapVISNIR[1]], *popt_NIR))

    # off_NIR =  np.median(pow(waves[1][overlapVISNIR[0]], *popt_VIS))/np.median(pow(waves[2][overlapVISNIR[1]], *popt_NIR))

    # print("UVB scaling: "+str(off_UVB))
    # print("NIR scaling: "+str(off_NIR))
    # # Apply correction
    # if scale:
    #     fluxs[0] *= off_UVB
    #     fluxs[2] *= off_NIR

    # Get wl, flux  and error in overlaps
    UVB_overlap_flux, UVB_overlap_error, UVB_overlap_bpmap = interpspec(waves[0], fluxs[0], errors[0], bpmaps[0], waves[1][overlapUVBVIS[1]])
    VIS1_overlap_flux, VIS1_overlap_error, VIS1_overlap_bpmap = fluxs[1][overlapUVBVIS[1]], errors[1][overlapUVBVIS[1]], bpmaps[1][overlapUVBVIS[1]]
    VIS2_overlap_flux, VIS2_overlap_error, VIS2_overlap_bpmap = interpspec(waves[1], fluxs[1], errors[1], bpmaps[1], waves[2][overlapVISNIR[1]])
    NIR_overlap_flux, NIR_overlap_error, NIR_overlap_bpmap = fluxs[2][overlapVISNIR[1]], errors[2][overlapVISNIR[1]], bpmaps[2][overlapVISNIR[1]]

    # Get weighted average in overlap between UVB and VIS, using the VIS sampling
    UVBVIS_wl = waves[1][overlapUVBVIS[1]]
    UVBVIS_flux, UVBVIS_error, UVBVIS_bpmap = avg(np.array([UVB_overlap_flux, VIS1_overlap_flux]), np.array([UVB_overlap_error, VIS1_overlap_error]), mask = np.array([UVB_overlap_bpmap, VIS1_overlap_bpmap]).astype("bool"), axis=0, weight = True)

    # Get weighted average in overlap between VIS and NIR, using the NIR sampling
    VISNIR_wl = waves[2][overlapVISNIR[1]]
    VISNIR_flux, VISNIR_error, VISNIR_bpmap = avg(np.array([VIS2_overlap_flux, NIR_overlap_flux]), np.array([VIS2_overlap_error, NIR_overlap_error]), mask = np.array([VIS2_overlap_bpmap, NIR_overlap_bpmap]).astype("bool"), axis = 0, weight = True)

    wl = np.concatenate((waves[0][UVB], UVBVIS_wl, waves[1][VIS], VISNIR_wl, waves[2][NIR]))
    flux = np.concatenate((fluxs[0][UVB], UVBVIS_flux, fluxs[1][VIS], VISNIR_flux, fluxs[2][NIR]))
    error = np.concatenate((errors[0][UVB], UVBVIS_error, errors[1][VIS], VISNIR_error, errors[2][NIR]))
    bpmap = np.concatenate((bpmaps[0][UVB], UVBVIS_bpmap, bpmaps[1][VIS], VISNIR_bpmap, bpmaps[2][NIR]))

    return wl, flux, error, bpmap

def main():
    # Small script to stitch X-shooter arms. Inspired by https://github.com/skylergrammer/Astro-Python/blob/master/stitch_spec.py

    # Load data from individual files
    data_dir = "/Users/jselsing/Work/work_rawDATA/STARGATE/GRB171010A/"
    # data_dir = "/Users/jselsing/Work/work_rawDATA/XSGW/MYST/"
    # data_dir = "/Users/jselsing/Work/etc/Kasper/GRB161014A/"

    scale = False # True, False

    data = np.genfromtxt(data_dir + "UVBOB1skysubstdext_SN.dat")
    UVB_wl, UVB_flux, UVB_error, UVB_bp = load_array(data)
    data = np.genfromtxt(data_dir + "VISOB1skysubstdext_SN.dat")
    # tell = np.genfromtxt(data_dir + "VIS_tell.dat")

    VIS_wl, VIS_flux, VIS_error, VIS_bp = load_array(data)
    data = np.genfromtxt(data_dir + "NIROB1skysubstdext_SN.dat")
    # tell = np.genfromtxt(data_dir + "NIR_tell.dat")
    NIR_wl, NIR_flux, NIR_error, NIR_bp = load_array(data)

    UVB_wl, UVB_flux, UVB_error, UVB_bp = bin_spectrum(UVB_wl, UVB_flux, UVB_error, UVB_bp.astype("bool"), 3, weight=True)
    VIS_wl, VIS_flux, VIS_error, VIS_bp = bin_spectrum(VIS_wl, VIS_flux, VIS_error, VIS_bp.astype("bool"), 3, weight=True)

    # Construct lists
    UVB_mask = (UVB_wl > 3300) & (UVB_wl < 5600)
    VIS_mask = (VIS_wl < 9800)
    NIR_mask = (NIR_wl > 9800) & (NIR_wl < 25000)
    waves = [UVB_wl[UVB_mask], VIS_wl[VIS_mask], NIR_wl[NIR_mask]]
    flux = [UVB_flux[UVB_mask], VIS_flux[VIS_mask], NIR_flux[NIR_mask]]
    error = [UVB_error[UVB_mask], VIS_error[VIS_mask], NIR_error[NIR_mask]]
    bpmap = [UVB_bp[UVB_mask], VIS_bp[VIS_mask], NIR_bp[NIR_mask]]

    # Stitch!
    wl, flux, error, bpmap = stitch_XSH_spectra(waves, flux, error, bpmap, scale = scale)

    np.savetxt(data_dir + "stitched_spectrum.dat", zip(wl, flux, error, bpmap), fmt = ['%10.6e', '%10.6e', '%10.6e', '%10.6f'], header=" wl flux error bpmap")

    hbin = 100
    wl_bin, flux_bin, error_bin, bp_bin = bin_spectrum(wl, flux, error, bpmap.astype("bool"), hbin, weight=True)





    pl.errorbar(wl_bin[::1], flux_bin[::1], yerr=error_bin[::1], fmt=".k", capsize=0, elinewidth=0.5, ms=3, alpha=0.3)
    pl.plot(wl_bin, medfilt(flux_bin, 1), linestyle="steps-mid", lw=0.5, alpha=0.7)
    # pl.plot(wl_bin, error_bin, linestyle="steps-mid", lw=1.0, alpha=0.5, color = "grey")
    pl.axhline(0, linestyle="dashed", color = "black", lw = 0.4)



    # nu = 3e5/(wl_bin*1e-13) # nu
    # f_nu = (flux_bin*(1e-13*(wl_bin**2.))/3e5)/1e-29
    # f_nu[f_nu <= 0] = 0.1
    # f_nu_err = (error_bin*(1e-13*(wl_bin**2.))/3e5)/1e-29
    # f_nu_err[f_nu_err <= 0] = 1
    # # print(wl_bin)
    # # exit()


    np.savetxt(data_dir + "stitched_spectrum_bin"+str(hbin)+".dat", zip(wl_bin, flux_bin, error_bin, bp_bin), fmt = ['%10.6e', '%10.6e', '%10.6e', '%10.6f'], header=" wl flux error bpmap")
    # pl.errorbar(wl_bin[::1], f_nu[::1], yerr=f_nu_err[::1], fmt=".k", capsize=0, elinewidth=0.5, ms=3, alpha=0.3)
    # pl.plot(wl_bin, medfilt(f_nu, 1), linestyle="steps-mid", lw=0.5, alpha=0.7)
    # # pl.plot(wl_bin, f_nu_err, linestyle="steps-mid", lw=1.0, alpha=0.5, color = "grey")
    # pl.axhline(0, linestyle="dashed", color = "black", lw = 0.4)

    # def pow(x, y, z):
    #     return y * x ** z

    # mask = (wl > 4500) & (wl < 15000)
    # epsilon, diff, msk = np.median(error[~np.isnan(flux)][mask]), 2*np.median(error[~np.isnan(flux)][mask]), np.ones_like(error[~np.isnan(flux)][mask]).astype("bool")
    # ii = 0
    # while diff > epsilon and ii < 5:
    #     popt, pcov = optimize.curve_fit(pow, wl[~np.isnan(flux)][mask][msk], flux[~np.isnan(flux)][mask][msk], sigma=error[~np.isnan(flux)][mask][msk], p0 = [5e-11, -1.6], maxfev=5000)
    #     msk = abs(pow(wl[~np.isnan(flux)][mask], *popt) - flux[~np.isnan(flux)][mask]) < (10 - ii)*error[~np.isnan(flux)][mask]
    #     diff = np.median(abs(pow(wl[~np.isnan(flux)][mask], *popt) - flux[~np.isnan(flux)][mask]))
    #     ii += 1
    # # popt, pcov = optimize.curve_fit(pow, wl[~np.isnan(flux)][mask], flux[~np.isnan(flux)][mask], sigma=error[~np.isnan(flux)][mask], maxfev=5000, p0 = [5e-11, -1.6])
    # print(popt, np.sqrt(np.diag(pcov)))
    # pow_fit = popt[0] * wl_bin ** (popt[1])
    # pl.plot(wl_bin, pow_fit)
    # pl.plot(wl_bin, flux_bin/(pow_fit))
    # pl.errorbar(wl_bin[::1], flux_bin[::1]/(pow_fit), yerr=error_bin[::1]/(pow_fit), fmt=".k", capsize=0, elinewidth=0.5, ms=3, alpha=0.3)

    # scale = np.median(f_nu[~np.isnan(f_nu)])
    pl.xlim(3200, 24000)
    # pl.ylim(-1 * scale, 10 * scale)
    pl.xlabel(r"Wavelength / [$\mathrm{\AA}$]")
    pl.ylabel(r'F$_\nu$ [$\mu$Jy]')
    ax = pl.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # pl.loglog()
    pl.savefig(data_dir + "stitched_spectrum_bin"+str(hbin)+".pdf")
    pl.show()


if __name__ == '__main__':
    main()