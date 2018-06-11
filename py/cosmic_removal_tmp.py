# -*- coding: utf-8 -*-

# import cosmics
import astroscrappy
import glob
from astropy.io import fits
import os
import matplotlib.pyplot as pl


# object_name = "/Users/jselsing/Work/work_rawDATA/FRB/FRB121102"
# object_name = "/Users/jselsing/Work/work_rawDATA/XSGRB/GRB170214A"
# object_name = "/Users/jselsing/Work/work_rawDATA/STARGATE/GRB180325A"

object_names =  ["/Volumes/io.erda.dk/XSGRB/GRB120909A", "/Volumes/io.erda.dk/XSGRB/GRB120119A", "/Volumes/io.erda.dk/XSGRB/GRB161014A", "/Volumes/io.erda.dk/XSGRB/GRB111209A", "/Volumes/io.erda.dk/XSGRB/GRB110128A", "/Volumes/io.erda.dk/XSGRB/GRB121024A", "/Volumes/io.erda.dk/XSGRB/GRB160804A", "/Volumes/io.erda.dk/XSGRB/GRB111211A", "/Volumes/io.erda.dk/XSGRB/GRB111117A", "/Volumes/io.erda.dk/XSGRB/GRB161007A", "/Volumes/io.erda.dk/XSGRB/GRB111228A", "/Volumes/io.erda.dk/XSGRB/GRB111107A", "/Volumes/io.erda.dk/XSGRB/GRB110721A", "/Volumes/io.erda.dk/XSGRB/GRB111008A", "/Volumes/io.erda.dk/XSGRB/GRB110715A", "/Volumes/io.erda.dk/XSGRB/GRB110709B", "/Volumes/io.erda.dk/XSGRB/GRB120716A", "/Volumes/io.erda.dk/XSGRB/GRB111129A", "/Volumes/io.erda.dk/XSGRB/GRB110407A", "/Volumes/io.erda.dk/XSGRB/GRB110818A", "/Volumes/io.erda.dk/XSGRB/GRB111005A", "/Volumes/io.erda.dk/XSGRB/GRB120311A", "/Volumes/io.erda.dk/XSGRB/GRB111123A", "/Volumes/io.erda.dk/XSGRB/GRB100316D", "/Volumes/io.erda.dk/XSGRB/GRB110808A", "/Volumes/io.erda.dk/XSGRB/GRB120815A", "/Volumes/io.erda.dk/XSGRB/GRB121229A"]


# for nn in glob.glob(object_name+"/data_with_raw_calibs/*cosmicced*"):
#     os.remove(nn)
for object_name in object_names:
    files = glob.glob(object_name+"/data_with_raw_calibs/*.fits")
    for n in files:
        try:
            fitsfile = fits.open(str(n))
        except:
            continue
        try:
            fitsfile[0].header['HIERARCH ESO DPR CATG'] = fitsfile[0].header['HIERARCH ESO DPR CATG']
        except:
            continue
        # try:
        #     print n, fitsfile[0].header['HIERARCH ESO DPR CATG'], fitsfile[0].header['HIERARCH ESO SEQ ARM'], fitsfile[0].header['OBJECT']
        # except:
        #     pass

        if fitsfile[0].header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and (fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'UVB' or fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'VIS' or fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'NIR'):
        # if fitsfile[0].header['HIERARCH ESO DPR CATG'] == 'SCIENCE' and (fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'NIR'):
            print('Removing cosmics from file: '+n+'...')

            # if fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'UVB' or fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'VIS':
            #     gain = fitsfile[0].header['HIERARCH ESO DET OUT1 GAIN']
            #     ron = fitsfile[0].header['HIERARCH ESO DET OUT1 RON']
            #     frac = 0.01
            #     if fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'UVB':
            #         objlim = 3
            #         sigclip = 2
            #     elif fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'VIS':
            #         objlim = 7
            #         sigclip = 3
            #     niter = 10
            # elif fitsfile[0].header['HIERARCH ESO SEQ ARM'] == 'NIR':
            #     gain = 2.12
            #     ron = 8
            #     frac = 0.0001
            #     objlim = 45
            #     sigclip = 20
            #     niter = 10
            # # try:
            # #     crmask, clean_arr = astroscrappy.detect_cosmics(fitsfile[0].data, inmask = fitsfile[2].data.astype("bool"))
            # # except:
            # crmask, clean_arr = astroscrappy.detect_cosmics(fitsfile[0].data, sigclip=sigclip, sigfrac=frac, objlim=objlim, cleantype='medmask', niter=niter, sepmed=True, verbose=True)

            # # Replace data array with cleaned image
            # fitsfile[0].data = clean_arr

            # # Try to retain info of corrected pixel if extension is present.
            # try:
            #     fitsfile[2].data[crmask] = 16 #Flag value for removed cosmic ray
            # except:
            #     print("No bad-pixel extension present. No flag set for corrected pixels")

            # # Update file
            # fitsfile.writeto(n[:-5]+"cosmicced.fits", output_verify='fix')

            # Moving original file
            dirname = object_name+"/backup"
            try:
                os.mkdir(dirname)
            except:
                pass
            os.rename(n, dirname+'/'+fitsfile[0].header['ARCFILE'])
