#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import glob
import numpy as np
import matplotlib.pyplot as pl
from util import *


def main():

    data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/"
    object_name = data_dir + "GRB121229A/"
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
            if "combined_" in kk and "ext.dat" in kk:
                del files[ii]

        out_arr = []
        for ii, kk in enumerate(files):
            out_arr.append(np.genfromtxt(kk))
        out_arr_sav = np.nanmean(np.array(out_arr), axis=0)
        f = np.array(out_arr)[:, :, 2]
        e = np.array(out_arr)[:, :, 3]
        bp = np.array(out_arr)[:, :, 4].astype("bool")
        out_arr_sav[:, 2], out_arr_sav[:, 3], out_arr_sav[:, 4] = avg(f, e, mask = bp, axis=0, weight=True)

        if optimal:
            np.savetxt(object_name+arm+"combined_optext.dat", out_arr_sav, header="# air_wave      vacuum_wave      flux           error           bpmap           E(B-V)      slitloss     tell_corr", fmt="%10.6e", delimiter="\t")
        elif not optimal:
            np.savetxt(object_name+arm+"combined_stdext.dat", out_arr_sav)

if __name__ == '__main__':
    main()