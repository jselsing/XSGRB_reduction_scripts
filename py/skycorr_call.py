# -*- coding: utf-8 -*-
# Adding ppxf path
# import sys
# sys.path.append('/Users/jonatanselsing/github/XSspec/')
import skycorr
import glob
import numpy as np
from astropy.io import fits


def main():
    # root_dir = "/Users/jonatanselsing/Work/work_rawDATA/SLSN/SN2018bsz/"
    # root_dir = "/Users/jonatanselsing/Work/work_rawDATA/STARGATE/GRB181010A/"
    root_dir = "/Users/jonatanselsing/Work/work_rawDATA/Crab_Pulsar/"
    # allfiles = glob.glob(root_dir+"reduced_data/OB9_TELL/*/*/*")

    # files = glob.glob(root_dir+"NIR*.fits")

    # files = [ii for ii in allfiles if "NIR" in ii]
    # files = [ii for ii in allfiles if "IDP" in ii]



    arms = ["UVB", "VIS", "NIR"] # "VIS", "NIR"
    OBs = ["OB1"]
    for kk in arms:
        for ll in OBs:
            # files = glob.glob(root_dir+"final/"+kk+ll+".fits")
            allfiles = glob.glob(root_dir+"reduced_data/"+ll+"/"+kk+"/*/*/*")
            files = [ii for ii in allfiles if "SKY_SLIT_MERGE1D" in ii]

            outpath =  root_dir+"reduced_data/"+ll+"/"+kk+"/"
            for ii in files:
                fitsfile = fits.open(ii)

                obj_name = kk+ll


                skycorr_fit = skycorr.Skycorr(ii, obj_name, outpath)
                skycorr_fit.setParams()
                skycorr_fit.runSkycorr()
                # skycorr_fit.updateSpec()




if __name__ == '__main__':
    main()
