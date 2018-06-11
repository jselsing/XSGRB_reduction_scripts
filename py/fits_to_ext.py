#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import glob
from astropy.io import fits
import matplotlib.pyplot as pl


def main():
    """
    Small script to convert BinTableHDU to extensions to use with IRAF
    """
    root_dir = "/Users/jselsing/Work/work_rawDATA/STARGATE/GRB180205A/"
    files = glob.glob(root_dir+"final/*.fits")
    cont_files = glob.glob(root_dir+"final/*.npy")

    # Loop through files
    for kk, ii in enumerate(files):
        # outname = ii[:3] + "fits" + ii[9:]
        outname = ii[:-5] + "ext.fits"
        bintab = fits.open(ii)
        header = bintab[0].header

        wl, flux, error, bpmap, tell = bintab[1].data.field("WAVE").flatten(), bintab[1].data.field("FLUX").flatten(), bintab[1].data.field("ERR").flatten(), bintab[1].data.field("QUAL").flatten(), bintab[1].data.field("TRANS").flatten()

        cont = np.load(cont_files[kk])
        # print(cont_files[kk])
        # print(outname)
        # exit()


        minlam, maxlam, dlam = min(wl), max(wl), np.median(np.diff(wl))
        header["CUNIT1"] = "angstrom"
        header["CRVAL1"] = minlam
        header["CDELT"] = dlam
        header["CD1_1"] = dlam

        f = fits.PrimaryHDU(flux, header= header)
        e = fits.ImageHDU(error, name ="ERR", header= header)
        bp = fits.ImageHDU(bpmap, name ="QUAL", header= header)
        cont = fits.ImageHDU(cont[0, :], name ="CONTINUUM", header= header)
        tell = fits.ImageHDU(tell, name ="TELL_CORR", header= header)
        hdulist = fits.HDUList([f, e, bp, cont, tell])

        for kk in xrange(1, 5):
            hdulist[kk].header["XTENSION"]
            hdulist[kk].header["PCOUNT"]
            hdulist[kk].header["GCOUNT"]
            hdulist[kk].header["PCOUNT"]

        hdulist.writeto(outname, overwrite=True)
        # exit()




    pass




if __name__ == '__main__':
    main()