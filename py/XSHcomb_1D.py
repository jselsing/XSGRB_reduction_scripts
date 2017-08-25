#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import glob
import numpy as np
import matplotlib.pyplot as pl
from util import *


def main():

    # data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/"
    # object_name = data_dir + "GRB111117A/"
    object_name = "/Users/jselsing/Work/work_rawDATA/SN2005ip/"

    arms = ["UVB", "VIS", "NIR"] # # UVB, VIS, NIR, ["UVB", "VIS", "NIR"]
    optimal = True # True, False

    for ii in arms:
        arm = ii
        # Get files
        if optimal:
            files = glob.glob(object_name+arm+"*optext.dat")
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
        f = out_arr[:, 2, :]
        e = out_arr[:, 3, :]
        bp =out_arr[:, 4, :].astype("bool")
        out_arr_sav[:, 2], out_arr_sav[:, 3], out_arr_sav[:, 4] = avg(f, e, mask = bp, axis=1, weight=True)

        # out_arr_sav[:, 0] = np.ma.mean(out_arr[:, 0, :], )

        if optimal:
            # print(object_name+arm+"optext_combined.dat")
            np.savetxt(object_name+arm+"optext_combined.dat", out_arr_sav, header="# air_wave      vacuum_wave      flux           error           bpmap           E(B-V)      slitloss     tell_corr", fmt="%10.6e", delimiter="\t")
        elif not optimal:
            np.savetxt(object_name+arm+"stdext_combined.dat", out_arr_sav)

if __name__ == '__main__':
    main()