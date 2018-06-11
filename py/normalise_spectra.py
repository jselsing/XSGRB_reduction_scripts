#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import matplotlib; matplotlib.use('TkAgg')
import sys

import stitch_arms, util

from glob import glob
import numpy as np
import matplotlib.pyplot as pl
from scipy import optimize
import os
import xsh_norm.interactive
from scipy.signal import medfilt

from numpy.polynomial import chebyshev
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev


def main():
    """
    Script to loop though bursts and normalise arms
    """
    data_dir = "../data/"
    bursts = glob(data_dir+"*")

    # Loop through burst names
    for ii, kk in enumerate(bursts):
        # List of extractions in burst dir
        extractions = [ll for ll in glob(kk+"/*") if ".dat" in ll or ".spec" in ll]
        # print(extractions)

        # Loop through extractions and normalize
        for ll in extractions:

            if "GRB" in ll:
                print("Normalizing file:")
                print(ll)
                burst_name = ll.split("/")[2]
                if "spec" in ll:
                    outname = (ll.split("/")[-1]).replace("skysub", "")[:-5]
                else:
                    outname = (ll.split("/")[-1]).replace("skysub", "")[:-4]
                print(kk+"/"+outname+"_norm.npy")

                if not os.path.exists(kk+"/"+outname+"_norm.npy"):
                    wl, flux, error, bpmap = stitch_arms.load_array(np.genfromtxt(ll))
                    mask = ~bpmap.astype("bool")
                    normalise = xsh_norm.interactive.xsh_norm(wl[mask], flux[mask], error[mask], bpmap, wl, flux, error, kk+"/"+outname)
                    # change = raw_input('Change default maskparameters(y/n)? ')
                    # if change == 'y':

                    if burst_name == "GRB090809A":
                        normalise.leg_order = 4
                        normalise.endpoint_order = 4
                    elif burst_name == "GRB091018A":
                        normalise.leg_order = 4
                        normalise.lover_mask = -5e-17
                        normalise.exclude_width = 2
                        normalise.sigma_mask = 10
                    elif burst_name == "GRB100205A":
                        normalise.lover_mask = -1e-16
                    elif burst_name == "GRB100219A":
                        normalise.leg_order = 1
                        normalise.endpoint_order = 1
                        normalise.exclude_width = 5
                        normalise.lover_mask = 0
                    elif burst_name == "GRB100316B":
                        normalise.fluxerror[normalise.wave > 10000] *= 2
                    elif burst_name == "GRB100418A1":
                        normalise.fluxerror[normalise.wave > 10000] *= 2
                    elif burst_name == "GRB100724A":
                        normalise.leg_order = 6
                        normalise.endpoint_order = 6
                    elif burst_name == "GRB110715A":
                        normalise.leg_order = 2
                        normalise.endpoint_order = 2
                    elif burst_name == "GRB100814A2":
                        normalise.fluxerror[normalise.wave > 10000] *= 2
                        normalise.leg_order = 5
                        normalise.endpoint_order = 5
                    elif burst_name == "GRB110818A":
                        normalise.leg_order = 2
                        normalise.endpoint_order = 2
                        normalise.lover_mask = -4e-17
                    elif burst_name == "GRB111008A1":
                        normalise.leg_order = 1
                        normalise.endpoint_order = 1
                    elif burst_name == "GRB111008A2":
                        normalise.leg_order = 1
                        normalise.endpoint_order = 1
                    elif burst_name == "GRB111211A":
                        normalise.tolerance = 0.50
                    elif burst_name == "GRB111211A":
                        normalise.tolerance = 0.50
                    elif burst_name == "GRB111228A":
                        normalise.leg_order = 2
                        normalise.endpoint_order = 2
                        # normalise.endpoint = "n"
                        normalise.lover_mask = 0
                    elif burst_name == "GRB120119A1":
                        normalise.leg_order = 3
                        normalise.endpoint_order = 3
                        normalise.endpoint = "t"
                        normalise.sigma_mask = 10
                    elif burst_name == "GRB120327A1":
                        normalise.leg_order = 7
                        normalise.endpoint_order = 7
                        normalise.tolerance = 1.0
                        normalise.endpoint = "n"
                    elif burst_name == "GRB120404A":
                        normalise.lover_mask = -0.5e-17
                    elif burst_name == "GRB130427A":
                        normalise.endpoint = "n"

                    elif burst_name == "GRB130606A0":
                        normalise.leg_order = 1
                        normalise.endpoint_order = 1
                    elif burst_name == "GRB130606A1":
                        normalise.leg_order = 1
                        normalise.endpoint_order = 1
                    elif burst_name == "GRB131030A":
                        normalise.endpoint = "n"
                    # normalise.leg_order = 5 #float(raw_input('Filtering Chebyshev Order (default = 3) = '))
                    # normalise.endpoint_order = 5#float(raw_input('Order of polynomial used for placing endpoints (default = 3) = '))
                    normalise.exclude_width = 1#float(raw_input('Exclude_width (default = 5) = '))
                    normalise.sigma_mask = 10#float(raw_input('Sigma_mask (default = 5) = '))
                    normalise.lover_mask = -1e-17#float(raw_input('Lover bound mask (default = -1e-17) = '))
                    # normalise.tolerance = 0.5#float(raw_input('Filtering tolerance (default = 0.25) = '))
                    # normalise.leg_order = #float(raw_input('Filtering Chebyshev Order (default = 3) = '))
                    normalise.spacing = 500#float(raw_input('Spacing in pixels between spline points (default = 150) = '))
                    normalise.division = 20#float(raw_input('Maximum number of allowed points (default = 300) = '))
                    normalise.endpoint_order = 1#float(raw_input('Order of polynomial used for placing endpoints (default = 3) = '))
                    normalise.endpoint = "n"#str(raw_input('Insert endpoint before interpolation(y/n)? '))
                    try:
                        normalise.run()
                    except:
                        pass
                    pl.title(burst_name)
                    pl.xlabel(r'Observed Wavelength  [$\mathrm{\AA}$]')
                    pl.ylabel(r'Flux [erg/cm$^2$/s/$\mathrm{\AA}$]')
                    # print(np.median(flux_arr[~np.isnan(flux_arr)]))
                    # sn = np.median(flux_arr[~np.isnan(flux_arr)]) / np.median(error_arr[~np.isnan(error_arr)]) 
                    l, h = np.percentile(flux[mask], (1, 99))
                    pl.ylim(l, h)
                    pl.show()
                    # answer = "n"
                    # answer = raw_input('Re-run normalisation(y/n)? ')
        #            normalise.clear()







                    # edge_mask = np.ones_like(wl)
                    # edge_mask[:500] = 0
                    # edge_mask[-500:] = 0
                    # mask = ~bpmap.astype("bool") & ~np.isnan(flux) & edge_mask.astype("bool")

                    # dn_reject = 1
                    # while dn_reject != 0:
                    #     n_reject = np.sum(mask.astype("int"))
                    #     f = interp1d(wl[mask], flux[mask], bounds_error=False, fill_value=np.median(flux[mask]))
                    #     filt = medfilt(f(wl), 501)
                    #     outlier = (abs(flux - filt) < 3 * error)
                    #     mask = mask & outlier
                    #     dn_reject =  n_reject - np.sum(mask.astype("int"))
                    #     print(dn_reject)

                    # dn_reject = 1
                    # while dn_reject != 0:
                    #     n_reject = np.sum(mask.astype("int"))
                    #     chebfit = chebyshev.chebfit(wl[mask], flux[mask], deg = 3)
                    #     outlier = (abs(flux - chebyshev.chebval(wl, chebfit)) < 3 * error)
                    #     mask = mask & outlier
                    #     dn_reject =  n_reject - np.sum(mask.astype("int"))
                    #     print(dn_reject)

                    # continuum = []
                    # for tt in range(1000):
                    #     print(tt)
                    #     indices = np.random.randint(0, len(wl[mask]), 200)
                    #     # spline = splrep(wl[mask][indices], flux[mask][indices], k=3)
                    #     points = np.array([np.median(flux[mask][xx-25:xx+25]) for xx in indices])
                    #     sort = np.argsort(wl[mask][indices])
                    #     spline = splrep(wl[mask][indices][sort], points[sort], k=3)
                    #     spl = splev(wl, spline)
                    #     if np.sum(np.isnan(spl).astype("int")) == 0:
                    #         continuum.append(spl)



if __name__ == '__main__':
    main()


