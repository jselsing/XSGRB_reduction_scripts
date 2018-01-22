#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
from astropy.io import fits
import numpy as np
import glob
import matplotlib.pyplot as pl
from scipy.interpolate import interp1d
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve


def main():

    root_dir = "/Users/jselsing/Work/work_rawDATA/STARGATE/GRB171205A/"
    # root_dir = "/Users/jselsing/Work/work_rawDATA/XSGW/AT2017GFO/"

    arms = ["UVB", "VIS", "NIR"]

    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9", "OB10", "OB11", "OB12"]
    OBs = ["OB5"]

    ext_name = "skysuboptext.dat" # None

    tell_file = 2

    for ii, arm in enumerate(arms):
        for kk, OB in enumerate(OBs):

            if OBs is None:
                try:
                    dat = np.genfromtxt(root_dir+arm+OB+"skysuboptext.dat")
                except:
                    dat = np.genfromtxt(root_dir+arm+OB+"skysubstdext.dat")
            else:
                dat = np.genfromtxt(root_dir+arm+OB+ext_name)

            try:
                wl, f, e, bpmap, dust, resp, slitcorr = dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], dat[:, 5], dat[:, 6], dat[:, 7]
            except:
                wl, f, e, bpmap, dust, resp, slitcorr = dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], dat[:, 5], dat[:, 6], dat[:, 7]



            file = glob.glob(root_dir + "reduced_data/"+OB+"/"+arm+"/*/*_IDP_"+arm+".fits")[0]
            n_files = len(glob.glob(root_dir + "reduced_data/"+OB+"/"+arm+"/*/*_IDP_"+arm+".fits"))



            fitsfile = fits.open(file)

            # if arm == "UVB" or arm == "VIS":
            #     fitsfile[0].header["NCOMBINE"] = 1
            # elif arm == "NIR":
            #     fitsfile[0].header["NCOMBINE"] = 1
            # Read in telluric correction
            # print(root_dir +"telluric/"+  arm + OB + "_tell"+str(tell_file)+"_TAC.fits")
            try:
                # Get telluric correction spectrum
                t_file = fits.open(root_dir +"telluric/"+ arm + OB + "_tell"+str(tell_file)+"_TAC.fits")
                print(root_dir + arm + OB + "_tell"+str(tell_file)+"_TAC.fits")
                # Get spectral resolution
                t_res = root_dir +"telluric/" + arm + OB + "_tell"+str(tell_file)+"_"+arm.lower()+"_molecfit_fit.res"
                for line in open(t_res):
                    if "Gaussian" in line:
                        f_G = float(line.split()[-3])
                        f_Gerr = float(line.split()[-1])
                    elif "Lorentzian" in line:
                        f_L = float(line.split()[-3])
                        f_Lerr = float(line.split()[-1])
                # print(np.median(wl))
                FWHM = (0.5346 * f_L + np.sqrt(0.2166*f_L**2 + f_G**2)) * fitsfile[0].header["spec_bin"] * 10
                print("FWHM: "+str(FWHM)+" at " + str(np.median(wl)) + " Å")
                R = np.median(wl) / FWHM
                print("R: "+str(R))

                t = t_file[1].data.field("mtrans").flatten()
            except:
                t = np.ones_like(wl)

            # Update data columns
            c = fitsfile[1].columns["WAVE"]
            c.data = wl/10 #* (1 - fitsfile[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
            # c.unit = "angstrom"
            fitsfile[1].data["WAVE"] = c.data

            c = fitsfile[1].columns["FLUX"]
            c.data = f*slitcorr
            fitsfile[1].data["FLUX"] = c.data

            c = fitsfile[1].columns["ERR"]
            c.data = e*slitcorr
            fitsfile[1].data["ERR"] = c.data

            c = fitsfile[1].columns["QUAL"]
            c.data = bpmap
            fitsfile[1].data["QUAL"] = c.data

            c = fitsfile[1].columns["SNR"]
            c.data = t
            c.name = "TRANS"
            fitsfile[1].data["TRANS"] = c.data

            c = fitsfile[1].columns["FLUX_REDUCED"]
            c.data = f/resp
            fitsfile[1].data["FLUX_REDUCED"] = c.data

            c = fitsfile[1].columns["ERR_REDUCED"]
            c.data = e/resp
            fitsfile[1].data["ERR_REDUCED"] = c.data


            # Update header values
            fitsfile[1].header["TELAPSE"] = fitsfile[1].header["TELAPSE"] * n_files
            fitsfile[0].header["EXPTIME"] = fitsfile[0].header["EXPTIME"] * n_files
            fitsfile[0].header["TEXPTIME"] = fitsfile[0].header["TEXPTIME"] * n_files

            # fitsfile.writeto(root_dir+"final/"+arm+OB+"_STARE.fits", overwrite=True)

            if arm == "UVB" or arm == "VIS":
                fitsfile.writeto(root_dir+"final/"+arm+OB+".fits", overwrite=True)
            elif arm == "NIR":
                fitsfile.writeto(root_dir+"final/"+arm+OB+".fits", overwrite=True)


if __name__ == '__main__':
    main()