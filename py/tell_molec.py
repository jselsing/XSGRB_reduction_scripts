# -*- coding: utf-8 -*-
# Adding ppxf path
# import sys
# sys.path.append('/Users/jonatanselsing/github/XSspec/')
import molec
import glob
import numpy as np
from astropy.io import fits


def main():
    # root_dir = "/Users/jonatanselsing/Work/work_rawDATA/SLSN/SN2018bsz/"
    root_dir = "/Users/jonatanselsing/Work/work_rawDATA/STARGATE/GRB180325A/"
    root_dir = "/Users/jonatanselsing/Work/work_rawDATA/Crab_Pulsar/"
    # allfiles = glob.glob(root_dir+"reduced_data/OB9_TELL/*/*/*")

    # files = glob.glob(root_dir+"NIR*.fits")

    # files = [ii for ii in allfiles if "NIR" in ii]
    # files = [ii for ii in allfiles if "IDP" in ii]

    outpath = root_dir + "telluric/"

    arms = ["NIR"]  # "VIS", "NIR"
    OBs = ["OB9"]
    for kk in arms:
        for ll in OBs:
            files = glob.glob(root_dir+"final/"+kk+ll+".fits")
            # allfiles = glob.glob(root_dir+"reduced_data/"+ll+"_TELL/"+kk+"/*/*")
            # files = [ii for ii in allfiles if "IDP" in ii]
            n = 1
            counter = []
            for ii in files:
                fitsfile = fits.open(ii)
                print(ii)
                sn = np.median(fitsfile[1].data.field("FLUX"))/np.median(fitsfile[1].data.field("ERR"))
                obj_name = kk+ll+"_"+str(n)
                # obj_name = ii.split("/")[-3]+ii.split("/")[-4]+str(n)
                if obj_name in counter:
                    n += 1
                elif obj_name not in counter:
                    n = 1
                # obj_name = ii.split("/")[-3]+ii.split("/")[-4]+str(n)
                counter.append(obj_name)

                # print(obj_name, sn)
                # continue

                m_fit = molec.molecFit(ii, obj_name, outpath)
                m_fit.setParams()
                m_fit.runMolec()
                m_fit.updateSpec()




if __name__ == '__main__':
    main()
