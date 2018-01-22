# -*- coding: utf-8 -*-
# Adding ppxf path
import sys
sys.path.append('/Users/jselsing/github/XSspec/')
import molec
import glob
import numpy as np
from astropy.io import fits


def main():
    root_dir = "/Users/jselsing/Work/work_rawDATA/STARGATE/GRB171205A/"
    # root_dir = "/Users/jselsing/Work/work_rawDATA/XSGW/AT2017GFO/"
    allfiles = glob.glob(root_dir+"reduced_data/OB5_tell/NIR/*/*")

    # allfiles = glob.glob("/Users/jselsing/Work/work_rawDATA/SN2013l/extfits/*.fits")

    # files = [ii for ii in allfiles if "NIR" in ii and "OB3" in ii]
    files = [ii for ii in allfiles if "IDP" in ii]

    outpath =  root_dir + "telluric/"
    n = 1
    counter = []
    for ii in files:
        fitsfile = fits.open(ii)
        # sn = np.median(fitsfile[1].data.field("FLUX"))/np.median(fitsfile[1].data.field("ERR"))
        # obj_name = ii.split("/")[-1]
        obj_name = ii.split("/")[-3]+ii.split("/")[-4]+str(n)
        if obj_name in counter:
            n += 1
        elif obj_name not in counter:
            n = 1
        obj_name = ii.split("/")[-3]+ii.split("/")[-4]+str(n)
        counter.append(obj_name)

        # print(obj_name)
        # continue

        m_fit = molec.molecFit(ii, obj_name, outpath)
        m_fit.setParams()
        m_fit.runMolec()
        m_fit.updateSpec()




if __name__ == '__main__':
    main()