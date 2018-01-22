#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')

import glob
import numpy as np

from util import *
from scipy.signal import medfilt

def main():

    # data_dir = "/Users/jselsing/Work/work_rawDATA/STARGATE/"
    data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/"
    object_name = data_dir + "GRB121229A/"
    # object_name = "/Users/jselsing/Work/work_rawDATA/XSGW/AT2017GFO/"

    arms = ["UVB", "VIS", "NIR"] # # UVB, VIS, NIR, ["UVB", "VIS", "NIR"]
    optimal = False # True, False

    for ii in arms:
        arm = ii
        # Get files
        if optimal:
            files = glob.glob(object_name+arm+"*optext.dat")
            if len(files) == 0:
                files = glob.glob(object_name+arm+"*stdext.dat")
        elif not optimal:
            files = glob.glob(object_name+arm+"*stdext.dat")

        # Remove duplicates
        for ii, kk in enumerate(files):
            if "combined" in kk and "ext.dat" in kk:
                del files[ii]


        xlen, ylen = 0, 0
        for ii, kk in enumerate(files):
            shp = np.shape(np.genfromtxt(kk))
            xlen = max(shp[0], xlen)
            ylen = max(shp[1], ylen)
        zlen = len(files)

        out_arr = np.zeros((xlen, ylen, zlen))*np.nan
        for ii, kk in enumerate(files):
            dat = np.genfromtxt(kk)
            out_arr[:dat.shape[0], :, ii] = dat


        out_arr_sav = np.nanmean(out_arr, axis=2)
        rescale = np.median(out_arr[int(xlen/4):-int(xlen/4), 2, :], axis=0)
        f = out_arr[:, 2, :]/(rescale/rescale[0])
        pl.plot(out_arr_sav[:, 1], medfilt(f[:, 0], 21), lw=0.5)
        pl.plot(out_arr_sav[:, 1], medfilt(f[:, 1], 21), lw=0.5)
        # pl.plot(out_arr_sav[:, 1], medfilt(f[:, 2], 51))
        # pl.plot(out_arr_sav[:, 1], medfilt(f[:, 3], 51))
        # pl.plot(out_arr_sav[:, 1], medfilt(f[:, 4], 51))
        e = out_arr[:, 3, :]/(rescale/rescale[0])
        # pl.plot(out_arr_sav[:, 1], medfilt(e[:, 0], 51), color="black", alpha=0.5, linestyle="dashed")
        # pl.plot(out_arr_sav[:, 1], medfilt(e[:, 1], 21), color="black", alpha=0.5, linestyle="dashed")
        # pl.plot(out_arr_sav[:, 1], medfilt(e[:, 2], 51))
        # pl.plot(out_arr_sav[:, 1], medfilt(e[:, 3], 51))
        # pl.plot(out_arr_sav[:, 1], medfilt(e[:, 4], 51))


        xs, ys = np.shape(e)
        # Weight by S/N
        # print(np.nanmedian(1./e**2., axis=0))
        # exit()
        weight = np.tile(np.nanmedian(1./e**2., axis=0), (xs, 1))
        bp = out_arr[:, 4, :].astype("bool")

        # out_arr_sav[:, 2], out_arr_sav[:, 3], out_arr_sav[:, 4] = avg(f, e, mask = bp, axis=1, weight=False)
        out_arr_sav[:, 2], out_arr_sav[:, 3], out_arr_sav[:, 4] = avg(f, e, mask = bp, axis=1, weight_map=weight)
        # print(out_arr_sav[:, 3])
        # pl.plot(out_arr_sav[:, 1], medfilt(out_arr_sav[:, 3], 51), color="black", alpha=1, linestyle="dashed")
        b_wl, b_f, b_e, b_q = bin_spectrum(out_arr_sav[:, 1], out_arr_sav[:, 2], out_arr_sav[:, 3], out_arr_sav[:, 4].astype("bool"), 50 )
        pl.plot(b_wl, b_f, color="firebrick", linestyle="steps-mid")
        pl.plot(b_wl, b_e, color="black", alpha=1, linestyle="dashed")
        # pl.errorbar(b_wl, b_f, yerr=b_e, fmt=".k", capsize=0, elinewidth=1.0, ms=3, alpha=0.8)
        # pl.plot(out_arr_sav[:, 1], medfilt(out_arr_sav[:, 2], 21), color="firebrick")
        # pl.plot(out_arr_sav[:, 1], abs(out_arr_sav[:, 2]/out_arr_sav[:, 3]))
        pl.axhline(0, linestyle="dashed")
 


        if len(files) > 1:
            if optimal:
                # print(object_name+arm+"optext_combined.dat")
                np.savetxt(object_name+arm+"optext_combined.dat", out_arr_sav, header="# air_wave      vacuum_wave      flux           error           bpmap           E(B-V)      slitloss     tell_corr", fmt="%10.6e", delimiter="\t")
            elif not optimal:
                np.savetxt(object_name+arm+"stdext_combined.dat", out_arr_sav)

    pl.ylim(-1e-17, 5e-17)
    # pl.semilogy()
    # pl.xlim(3200, 6000)
    pl.show()



if __name__ == '__main__':
    main()