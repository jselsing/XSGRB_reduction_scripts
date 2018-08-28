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






def main():

    root_dir = "/Users/jselsing/Work/work_rawDATA/STARGATE/GRB180728A/final/"
    # root_dir = "/Users/jselsing/Work/work_rawDATA/SLSN/SN2018bsz/final/"

    # root_dir = "/Users/jselsing/Work/work_rawDATA/XSGW/AT2017GFO/final/"
    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9", "OB10", "OB11", "OB12", "OB13", "OB14", "OB15", "OB16", "OB17", "OB18"]
    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5"]
    OBs = ["OB8"]


    for ii, kk in enumerate(OBs):
        wl_out, flux_out, error_out = [0]*3, [0]*3, [0]*3

        off = 0#(len(OBs) - ii) * 2e-17
        mult = 1.0

        ############################## OB ##############################
        f = fits.open(root_dir + "UVB%s.fits"%kk)
        wl = 10.*f[1].data.field("WAVE").flatten()


        q = f[1].data.field("QUAL").flatten()
        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(q)
        mask_wl = (wl > 3200) & (wl < 5550)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)

        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot)
        error = error(wl_plot)
        print(flux/error)


        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), 50)
        wl_out[0], flux_out[0], error_out[0] = b_wl, b_f, b_e





        f = fits.open(root_dir + "VIS%s.fits"%kk)

        wl = 10.*f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()

        t = f[1].data.field("TRANS").flatten()
        # t = np.ones_like(f[1].data.field("TRANS").flatten())
        mask_wl = (wl > 5650) & (wl < 10000)
        mask_qual = ~q.astype("bool")

        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]

        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), 50)
        wl_out[1], flux_out[1], error_out[1] = b_wl, b_f, b_e



        f = fits.open(root_dir + "NIR%s.fits"%kk)
        wl = 10.*f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()

        t = f[1].data.field("TRANS").flatten()
        # t = np.ones_like(f[1].data.field("TRANS").flatten())
        mask_wl = (wl > 10500) & (wl < 25000) & (t > 0.3)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]

        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), 50)
        wl_out[2], flux_out[2], error_out[2] = b_wl, b_f, b_e



        conc_wl = np.concatenate(wl_out)
        f = interpolate.interp1d(conc_wl, np.concatenate(flux_out))
        g = interpolate.interp1d(conc_wl, np.concatenate(error_out))
        new_wl = np.arange(min(conc_wl), max(conc_wl), np.median(np.diff(conc_wl)))
        new_flux = f(new_wl)
        new_error = g(new_wl)
        np.savetxt(root_dir+"stictsh%s_bin50.dat"%kk, list(zip(new_wl, new_flux, new_error)))


if __name__ == '__main__':
    main()