#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import glob
import numpy as np
import matplotlib.pyplot as pl
from util import *


def main():

    data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/"
    object_name = data_dir + "GRB100425A/"
    arm = "UVB" # UVB, VIS, NIR
    optimal = False # True, False

    # Get files
    if optimal:
        files = glob.glob(object_name+arm+"*optext.dat")
    elif not optimal:
        files = glob.glob(object_name+arm+"*stdext.dat")

    # Remove duplicates
    for ii, kk in enumerate(files):
        if "combined_" in kk and "ext.dat" in kk:
            del files[ii]


    print(files)

    out_arr = []
    for ii, kk in enumerate(files):
        out_arr.append(np.genfromtxt(kk))
    out_arr_sav = np.nanmean(np.array(out_arr), axis=0)
    f = np.ma.array(np.array(out_arr)[:, :, 2], mask = np.array(out_arr)[:, :, 4].astype("bool"))
    e = np.ma.array(np.array(out_arr)[:, :, 3], mask = np.array(out_arr)[:, :, 4].astype("bool"))
    out_arr_sav[:, 2], out_arr_sav[:, 3] = weighted_avg(f, e, axis=0)

    if optimal:
        np.savetxt(object_name+arm+"combined_optext.dat", out_arr_sav)
    elif not optimal:
        np.savetxt(object_name+arm+"combined_stdext.dat", out_arr_sav)

if __name__ == '__main__':
    main()