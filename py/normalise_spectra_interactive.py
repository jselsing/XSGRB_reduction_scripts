#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys

import stitch_arms, util

import glob
import numpy as np
import matplotlib.pyplot as pl
from scipy import optimize
import os
import xsh_norm.interactive
from scipy.signal import medfilt
from util import *

from numpy.polynomial import chebyshev
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from astropy.io import fits

def main():
    """
    Script to loop though bursts and normalise arms
    """

    root_dir = "/Users/jonatanselsing/Work/work_rawDATA/STARGATE/GRB181020A/"
    bursts = glob.glob(root_dir+"final/*.fits")

    # Loop through burst names
    for ii, kk in enumerate(bursts):
        # List of extractions in burst dir


        print("Normalizing file:")
        print(kk)
        burst_name = kk.split("/")[-1].split(".")[0]

        outname = kk[:-5]


        if not os.path.exists(outname+"_norm.fits"):
            f = fits.open(kk)
            wl = 10.*f[1].data.field("WAVE").flatten()
            flux = f[1].data.field("FLUX").flatten()
            error = f[1].data.field("ERR").flatten()
            q = f[1].data.field("QUAL").flatten()
            t = f[1].data.field("TRANS").flatten()
            # t = np.ones_like(f[1].data.field("TRANS").flatten())
            if "UVB" in kk:
                mask_wl = (wl > 3200) & (wl < 5550)
            elif "VIS" in kk:
                mask_wl = (wl > 5650) & (wl < 10000)
            elif "NIR" in kk:
                mask_wl = (wl > 10200) & (t > 0.3) & (wl < 22500)
            mask_qual = ~q.astype("bool")
            flux_n = interp1d(wl[mask_qual], flux[mask_qual], bounds_error=False, fill_value=0)
            error_n = interp1d(wl[mask_qual], error[mask_qual], bounds_error=False, fill_value=0)
            wl_plot = wl[mask_wl]
            flux_f = flux_n(wl_plot) / t[mask_wl]
            error_f = error_n(wl_plot) / t[mask_wl]
            # wl_i, flux_i, error_i = filter_bad_values(wl_plot, flux_f, error_f)
            wl_i, flux_i, error_i = wl_plot, flux_f, error_f


            normalise = xsh_norm.interactive.xsh_norm(wl_i, flux_i, error_i, mask_qual, wl, flux, error, outname)



            # normalise.leg_order = 5 #float(raw_input('Filtering Chebyshev Order (default = 3) = '))
            # normalise.endpoint_order = 5#float(raw_input('Order of polynomial used for placing endpoints (default = 3) = '))
            normalise.exclude_width = 3#float(raw_input('Exclude_width (default = 5) = '))
            normalise.sigma_mask = 5#float(raw_input('Sigma_mask (default = 5) = '))
            normalise.lover_mask = -1e-15#float(raw_input('Lover bound mask (default = -1e-17) = '))
            # normalise.tolerance = 0.5#float(raw_input('Filtering tolerance (default = 0.25) = '))
            # normalise.leg_order = #float(raw_input('Filtering Chebyshev Order (default = 3) = '))
            normalise.spacing = 400#float(raw_input('Spacing in pixels between spline points (default = 150) = '))
            normalise.division = 20#float(raw_input('Maximum number of allowed points (default = 300) = '))
            normalise.endpoint_order = 2#float(raw_input('Order of polynomial used for placing endpoints (default = 3) = '))
            normalise.endpoint = "t"#str(raw_input('Insert endpoint before interpolation(y/n)? '))



            try:
                normalise.run()
            except:
                pass
            pl.show()

if __name__ == '__main__':
    main()


