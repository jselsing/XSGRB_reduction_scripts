#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/local/anaconda3/envs/py36 python
# -*- coding: utf-8 -*-

# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')

# Imports
import numpy as np
from astropy.io import fits
from scipy import interpolate
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from scipy.signal import medfilt
import astropy.units as u
from astropy.time import Time
from util import *


import pyphot
lib = pyphot.get_library()

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
params = {
   'axes.labelsize': 12,
   'font.size': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'text.usetex': True,
   'figure.figsize': [8, 8/1.61]
   }
mpl.rcParams.update(params)

def filter_bad_values(wl, flux, error):
    medfilter = medfilt(flux, 501)
    mask = np.logical_and(abs(flux - medfilter) < 3*error, ~np.isnan(flux))
    f = interpolate.interp1d(wl[mask], flux[mask], bounds_error=False, fill_value = np.nan)
    g = interpolate.interp1d(wl[mask], error[mask], bounds_error=False, fill_value = np.nan)
    return wl, f(wl), g(wl)


def synpassflux(wl, flux, band):
    # Calculate synthetic magnitudes
    n_bands = len(band)
    synmags = [0]*n_bands
    cwav = [0]*n_bands
    cwave = [0]*n_bands
    for ii, kk in enumerate(band):
        filt = np.genfromtxt("/Users/jselsing/github/iPTF16geu/data/passbands/%s.dat"%kk)
        lamb_T, T = filt[:,0], filt[:,1]

        f = pyphot.Filter(lamb_T, T, name=kk, dtype='photon', unit='Angstrom')
        fluxes = f.get_flux(wl, flux, axis=0)
        synmags[ii] = -2.5 * np.log10(fluxes) - f.AB_zero_mag
        cwav[ii] = np.mean(lamb_T)
        cwave[ii] = (max(lamb_T[T > 0.2] - cwav[ii]), (cwav[ii] - min(lamb_T[T > 0.2])))
    synmag_flux = ((synmags*u.ABmag).to((u.erg/(u.s * u.cm**2 * u.AA)), u.spectral_density(cwav * u.AA))).value
    return synmag_flux, cwav, cwave



def main():

    root_dir = "/Users/jselsing/Work/work_rawDATA/Francesco/final/"
    # root_dir = "/Users/jselsing/Work/work_rawDATA/SLSN/SN2018bsz/final/"

    # root_dir = "/Users/jselsing/Work/work_rawDATA/XSGW/AT2017GFO/final/"
    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9", "OB10", "OB11", "OB12", "OB13", "OB14", "OB15", "OB16", "OB17", "OB18"]
    OBs = ["OB1", "OB2"]#, "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9"]
    arms = ["UVB", "VIS", "NIR"]# UVB, VIS, NIR, ["UVB", "VIS", "NIR"]

    nx = [0]*len(arms)
    for pp, ll in enumerate(arms):
        f = fits.open(root_dir + "%s%s.fits"%(ll, OBs[0]))
        nx[pp] = f[1].header["NELEM"]


    block_UVB = np.zeros((3, len(OBs), nx[0]))
    block_VIS = np.zeros((3, len(OBs), nx[1]))
    block_NIR = np.zeros((3, len(OBs), nx[2]))

    # block = [block_UVB, block_VIS, block_NIR]
    for pp, ll in enumerate(arms):
        for ii, kk in enumerate(OBs):
            f = fits.open(root_dir + "%s%s.fits"%(ll, kk))

            wl = 10.*f[1].data.field("WAVE").flatten()
            q = f[1].data.field("QUAL").flatten()
            t = f[1].data.field("TRANS").flatten()
            mask_qual = ~q.astype("bool")
            flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
            error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
            wl_plot = wl
            flux = flux(wl_plot)/ t
            error = error(wl_plot)/ t
            wl, flux, error = filter_bad_values(wl_plot, flux, error)

            if ll == "UVB":
                block_UVB[0, ii, :] = wl
                block_UVB[1, ii, :] = flux
                block_UVB[2, ii, :] = error
            elif ll == "VIS":
                block_VIS[0, ii, :] = wl
                block_VIS[1, ii, :] = flux
                block_VIS[2, ii, :] = error
            elif ll == "NIR":
                block_NIR[0, ii, :] = wl
                block_NIR[1, ii, :] = flux
                block_NIR[2, ii, :] = error



    block = [block_UVB, block_VIS, block_NIR]




    for ii, kk in enumerate(block):




        combined_flux, combined_error, __ = avg(kk[1], kk[2], axis=0)
        # print(out_arr_sav[:, 3])
        wl = np.median(kk[0], axis=0)
        # pl.plot(wl, combined_flux, color="black", alpha=1, linestyle="dashed")
        b_wl, b_f, b_e, b_q = bin_spectrum(wl, combined_flux, combined_error, np.zeros_like(combined_error).astype("bool"), 200)

        pl.plot(b_wl, b_f, color="firebrick", linestyle="steps-mid")
        pl.plot(b_wl, b_e, color="black", alpha=1, linestyle="dashed")
        # pl.errorbar(b_wl, b_f, yerr=b_e, fmt=".k", capsize=0, elinewidth=1.0, ms=3, alpha=0.8)
        # pl.plot(out_arr_sav[:, 1], medfilt(out_arr_sav[:, 2], 21), color="firebrick")
        # pl.plot(out_arr_sav[:, 1], abs(out_arr_sav[:, 2]/out_arr_sav[:, 3]))
        pl.axhline(0, linestyle="dashed")



        # if len(files) > 1:
        #     if optimal:
        #         # print(object_name+arm+"optext_combined.dat")
        #         np.savetxt(object_name+arm+"optext_combined.dat", out_arr_sav, header="# air_wave      vacuum_wave      flux           error           bpmap           E(B-V)      slitloss     tell_corr", fmt="%10.6e", delimiter="\t")
        #     elif not optimal:
        #         np.savetxt(object_name+arm+"stdext_combined.dat", out_arr_sav)

        pl.ylim(-1e-17, 5e-17)
        # pl.semilogy()
        # pl.xlim(3200, 6000)
        pl.show()



if __name__ == '__main__':
    main()