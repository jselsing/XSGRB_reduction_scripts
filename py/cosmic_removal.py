# -*- coding: utf-8 -*-

# import cosmics
import astroscrappy
import glob
from astropy.io import fits
import os
import matplotlib.pyplot as pl


data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/"
objectname = data_dir + "GRB100418A"
# objectname = "/Users/jselsing/Work/etc/GB_IDL_XSH_test/Q0157/J_red/"
# objectname = "/Volumes/io.erda.dk/XSGRB/GRB100621A"

for nn in glob.glob(objectname+"/data_with_raw_calibs/*cosmicced*"):
    os.remove(nn)


files = glob.glob(objectname+"/data_with_raw_calibs/*.fits")
for n in files:
    fitsfile = fits.open(str(n))
    try:
        fitsfile[0].header['HIERARCH ESO DPR CATG'] = fitsfile[0].header['HIERARCH ESO DPR CATG']
    except:
        continue
    # try:
    #     print n, fitsfile[0].header['HIERARCH ESO DPR CATG'], fitsfile[0].header['HIERARCH ESO SEQ ARM'], fitsfile[0].header['OBJECT']
    # except:
    #     pass

    if fitsfile[0].header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and (fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'NIR' or fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'UVB' or fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'VIS'):
    # if fitsfile[0].header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and (fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'NIR'):
        print 'Removing cosmics from file: '+n+'...'

        if fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'UVB' or fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'VIS':
            gain = fitsfile[0].header['HIERARCH ESO DET OUT1 GAIN']
            ron = fitsfile[0].header['HIERARCH ESO DET OUT1 RON']
            frac = 0.01
            if fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'UVB':
                objlim = 3
                sigclip = 1
            elif fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'VIS':
                objlim = 7
                sigclip = 3
            niter = 4
        elif fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'NIR':
            gain = 2.12
            ron = 8
            frac = 0.0001
            objlim = 45
            sigclip = 30
            niter = 10
        # try:
        #     crmask, clean_arr = astroscrappy.detect_cosmics(fitsfile[0].data, inmask = fitsfile[2].data.astype("bool"))
        # except:
        crmask, clean_arr = astroscrappy.detect_cosmics(fitsfile[0].data, sigclip=sigclip, sigfrac=frac, objlim=objlim, cleantype='medmask', niter=niter, sepmed=True, verbose=True)

        # Replace data array with cleaned image
        fitsfile[0].data = clean_arr

        # Try to retain info of corrected pixel if extension is present.
        try:
            fitsfile[2].data[crmask] = 16 #Flag value for removed cosmic ray
        except:
            print("No bad-pixel extension present. No flag set for corrected pixels")

        # Update file
        fitsfile.writeto(n[:-5]+"cosmicced.fits", output_verify='fix')

        # Moving original file
        dirname = objectname+"/backup"
        try:
            os.mkdir(dirname)
        except:
            pass
        os.rename(n, dirname+'/'+fitsfile[0].header['ARCFILE'])
