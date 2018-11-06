#!/usr/bin/env py36

"""
Estimate of sky for spectra wit

"""

import os
import subprocess
import time
import numpy as np

# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('ticks')
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits

skycorrBase = os.path.expanduser('/usr/local/astrosoft/ESO/skycorr_1.1.2')
skycorrCall = os.path.join(skycorrBase, 'bin/skycorr')

class Skycorr():

    ''' Sets up and runs a skycorr to spectral data - requires skycorr in its version 1.1.2 or greater.
    See http://www.eso.org/sci/software/pipelines/skytools/skycorr
    '''

    def __init__(self, fitsfile_path, object_name, output_path):
        ''' Reads in the required information to set up the parameter files'''
        self.input_file_path = fitsfile_path
        self.fits = fits.open(self.input_file_path)
        self.arm = self.fits[0].header["HIERARCH ESO SEQ ARM"].lower()
        self.name = object_name
        self.output_path = output_path
        self.skycorrparfile = []
        self.params = {'basedir': skycorrBase,
            'listname': 'none', 'trans' : 1,
            'columns': 'WAVE FLUX ERR QUAL',
            'default_error': 0.01,
            'wlgtomicron' : 0.001,
            'vac_air': 'air',
            'output_dir': os.path.abspath(os.path.expanduser(output_path))}

        self.headpars = {'utc': 'UTC', 'telalt': 'HIERARCH ESO TEL ALT',
            'rhum' : 'HIERARCH ESO TEL AMBI RHUM',
            'obsdate' : 'MJD-OBS',
            'temp' : 'HIERARCH ESO TEL AMBI TEMP',
            'm1temp' : 'HIERARCH ESO TEL TH M1 TEMP',
            'geoelev': 'HIERARCH ESO TEL GEOELEV',
            'longitude': 'HIERARCH ESO TEL GEOLON',
            'latitude': 'HIERARCH ESO TEL GEOLAT',
            'pixsc' : 'SPEC_BIN' }#CD2_2

        self.atmpars = {'ref_atm': 'equ.atm',
            'gdas_dir': os.path.join(skycorrBase, 'data/profiles/grib'),
            'gdas_prof': 'auto', 'layers': 1, 'emix': 5.0, 'pwv': -1.}

    def updateParams(self, paramdic, headpars):
        ''' Sets Parameters for skycorr execution'''

        for key in paramdic.keys():
            print('\t\tskycorr: Setting parameter %s to %s' %(key,  paramdic[key]))
            if key in self.params.keys():
                self.params[key] = paramdic[key]
            elif key in self.headpars.keys():
                self.headpars[key] = paramdic[key]
            elif key in self.atmpars.keys():
                self.atmpars[key] = paramdic[key]
            else:
                print('\t\tWarning: Parameter %s not known to skycorr')
                self.params[key] = paramdic[key]


    def setParams(self):
        ''' Writes skycorr parameters into file '''



        print('\tPreparing skycorr params file')
        self.skycorrparfile = os.path.abspath(self.output_path+'/%s_skycorr_%s.par' % (self.name, self.arm))
        f = open(self.skycorrparfile, 'w')

        f.write('# ----------------------------------------------------------------------------\n')
        f.write('# -------------------- INPUT PARAMETER FILE FOR SKYCORR ----------------------\n')
        f.write('# ----------------------------------------------------------------------------\n')
        f.write('\n')
        f.write('# ---------------------------DIRECTORIES + FILES------------------------------\n')
        f.write('\n')
        f.write('# Absolute path of skycorr installation directory\n')
        f.write('INST_DIR=%s\n'%self.params['basedir'])
        f.write('\n')
        f.write('# Absolute or relative (with respect to INST_DIR) path and filename of input\n')
        f.write('# object spectrum\n')
        f.write('INPUT_OBJECT_SPECTRUM=%s\n'%self.input_file_path)
        f.write('\n')
        f.write('# Absolute or relative (with respect to INST_DIR) path and filename of input\n')
        f.write('# sky spectrum\n')
        f.write('INPUT_SKY_SPECTRUM=%s\n'%self.input_file_path)
        f.write('\n')
        f.write('# Absolute or relative (with respect to INST_DIR) path and filename of output\n')
        f.write('# directory (will be created if not present; default: <INST_DIR>/output/)\n')
        f.write('OUTPUT_DIR=%s\n'%self.params['output_dir'])
        f.write('\n')
        f.write('# Main name of diagnostic output files, extensions will be added\n')
        f.write('OUTPUT_NAME=%s\n'%self.name)
        f.write('\n')
        f.write('#------------------------------INPUT STRUCTURE--------------------------------\n')
        f.write('\n')
        f.write('# Names of file columns (table) or extensions (image)\n')
        f.write('# A list of 4 labels has to be provided:\n')
        f.write('# 1: wavelength [image: NONE if dedicated extension does not exist]\n')
        f.write('# 2: flux [image: NONE if in zeroth, unnamed extension]\n')
        f.write('# 3: flux error [NONE if not present]\n')
        f.write('# 4: mask (integer: 1 = selected, 0 = rejected;\n')
        f.write('#          float:   0. = selected, otherwise rejected) [NONE if not present]\n')
        f.write('COL_NAMES=%s %s %s %s\n'%('WAVE', 'FLUX', 'ERR', 'QUAL'))
        f.write('\n')
        f.write('# Error relative to mean if no error column is provided (default: 0.01)\n')
        f.write('DEFAULT_ERROR=0.01\n')
        f.write('\n')
        f.write('# Multiplicative factor to convert wavelength to micron\n')
        f.write('# e.g.: wavelength unit = A -> WLG_TO_MICRON = 1e-4\n')
        f.write('WLG_TO_MICRON=%s\n'%self.params['wlgtomicron'])
        f.write('\n')
        f.write('# Wavelengths in vacuum (= vac) or air (= air)\n')
        f.write('VAC_AIR=%s\n'%self.params['vac_air'])
        f.write('\n')
        f.write('\n')
        f.write('# ----------------------------------------------------------------------------\n')
        f.write('# ------------------------- EXPERT MODE PARAMETERS ---------------------------\n')
        f.write('# ----------------------------------------------------------------------------\n')
        f.write('\n')
        f.write('# ------------------------------FITS KEYWORDS---------------------------------\n')
        f.write('\n')
        f.write('# FITS keyword of sky spectrum for Modified Julian Day (MJD) or date in years\n')
        f.write('# (default: MJD-OBS; optional parameter for value: DATE_VAL)\n')
        f.write('DATE_KEY=MJD-OBS\n')
        f.write('\n')
        f.write('# FITS keyword of sky spectrum for UTC time in s\n')
        f.write('# (default: TM-START; optional parameter for value: TIME_VAL)\n')
        f.write('TIME_KEY=TM-START\n')
        f.write('\n')
        f.write('# FITS keyword of sky spectrum for telescope altitude angle in deg\n')
        f.write('# (default: ESO TEL ALT; optional parameter for value: TELALT_VAL)\n')
        f.write('TELALT_KEY=ESO TEL ALT\n')
        f.write('\n')
        f.write('# ---------------------------REQUIRED INPUT DATA------------------------------\n')
        f.write('\n')
        f.write('# Airglow line list\n')
        f.write('# Required directory: <INST_DIR>/sysdata/\n')
        f.write('LINETABNAME=airglow_groups.dat\n')
        f.write('\n')
        f.write('# File for airglow scaling parameters\n')
        f.write('# Required directory: <INST_DIR>/sysdata/\n')
        f.write('VARDATNAME=airglow_var.dat\n')
        f.write('\n')
        f.write('# FTP address (supplemented by "ftp://") for folder with monthly averages of\n')
        f.write('# solar radio flux at 10.7 cm\n')
        f.write('SOLDATURL=ftp.geolab.nrcan.gc.ca/data/solar_flux/monthly_averages\n')
        f.write('\n')
        f.write('# File with monthly averages of solar radio flux at 10.7 cm\n')
        f.write('# Required directory: SOLDATURL or <INST_DIR>/sysdata/\n')
        f.write('SOLDATNAME=solflux_monthly_average.txt\n')
        f.write('\n')
        f.write('# Solar radio flux at 10.7 cm:\n')
        f.write('# Positive value in sfu (= 0.01 MJy) or -1 [default] for corresponding monthly\n')
        f.write('# average from http://www.spaceweather.gc.ca. Download only if local file in\n')
        f.write('# <INST_DIR>/sysdata/ does not contain required month.\n')
        f.write('SOLFLUX=-1\n')
        f.write('\n')
        f.write('# ---------------------------LINE IDENTIFICATION------------------------------\n')
        f.write('\n')
        f.write('# Initial estimate of line FWHM [pixel]\n')
        f.write('FWHM=5.0\n')
        f.write('\n')
        f.write('# Variable line width (linear increase with wavelength)? -- 1 = yes; 0 = no\n')
        f.write('VARFWHM=0\n')
        f.write('\n')
        f.write('# Relative FWHM convergence criterion (default: 1e-2)\n')
        f.write('LTOL=1e-2\n')
        f.write('\n')
        f.write('# Minimum distance to neighbouring lines for classification as isolated line:\n')
        f.write('# <MIN_LINE_DIST> * <FWHM> [pixel]\n')
        f.write('MIN_LINE_DIST=2.5\n')
        f.write('\n')
        f.write('# Minimum line peak flux for consideration of lines from airglow line list:\n')
        f.write('# <FLUXLIM> * <median flux of identified lines>\n')
        f.write('# Automatic search -> FLUXLIM = -1 (default)\n')
        f.write('FLUXLIM=-1\n')
        f.write('\n')
        f.write('# ---------------------------FITTING OF SKY LINES-----------------------------\n')
        f.write('\n')
        f.write('# Relative chi^2 MPFIT convergence criterion (default: 1e-3)\n')
        f.write('FTOL=1e-3\n')
        f.write('\n')
        f.write('# Relative parameter MPFIT convergence criterion (default: 1e-3)\n')
        f.write('XTOL=1e-3\n')
        f.write('\n')
        f.write('# Relative chi^2 convergence criterion for iterative improvement of\n')
        f.write('# wavelength grid (default: 1e-3)\n')
        f.write('WTOL=1e-3\n')
        f.write('\n')
        f.write('# Maximum degree of Chebyshev polynomial for wavelength grid correction:\n')
        f.write('# -1 = no correction\n')
        f.write('#  0 = linear term (coef. = 1) is also considered but not fitted\n')
        f.write('#  7 = default\n')
        f.write('CHEBY_MAX=7\n')
        f.write('\n')
        f.write('# Minimum degree of Chebyshev polynomial for wavelength grid correction.\n')
        f.write('# CHEBY_MIN <= CHEBY_MAX:\n')
        f.write('# - Iterative increase of polynomial degree at least until CHEBY_MIN\n')
        f.write('#   (default: 3).\n')
        f.write('# - Procedure stops if chi^2 gets worse or CHEBY_MAX is reached.\n')
        f.write('# - Results of degree with best chi^2 are taken.\n')
        f.write('# CHEBY_MIN > CHEBY_MAX:\n')
        f.write('# - Iterative increase of polynomial degree until CHEBY_MAX is reached.\n')
        f.write('# - Results of degree CHEBY_MAX are taken.\n')
        f.write('CHEBY_MIN=3\n')
        f.write('\n')
        f.write('# Initial constant term for wavelength grid correction (shift relative to half\n')
        f.write('# wavelength range)\n')
        f.write('CHEBY_CONST=0.\n')
        f.write('\n')
        f.write('# Type of rebinning:\n')
        f.write('# 0 = simple rebinning (summation of pixel fractions)\n')
        f.write('# 1 = convolution with asymmetric, damped sinc kernel [default]\n')
        f.write('REBINTYPE=1\n')
        f.write('\n')
        f.write('# Minimum relative weight of the strongest line group of a pixel for\n')
        f.write('# including a pixel in the line fitting procedure (default: 0.67)\n')
        f.write('WEIGHTLIM=0.67\n')
        f.write('\n')
        f.write('# Sigma limit for excluding outliers (e.g. object emission lines) from\n')
        f.write('# estimate of group flux correction factors (default: 15.)\n')
        f.write('SIGLIM=15.\n')
        f.write('\n')
        f.write('# Lower relative uncertainty limit for the consideration of a line group for\n')
        f.write('# the fitting procedure. The value is compared to the sigma-to-mean ratio of\n')
        f.write('# the group-specific flux correction factors of the initial estimate\n')
        f.write('# (default: 0. -> include all fittable line groups).\n')
        f.write('FITLIM=0.\n')
        f.write('\n')
        f.write('# ---------------------------------PLOTTING-----------------------------------\n')
        f.write('\n')
        f.write('# Diagnostic gnuplot plots:\n')
        f.write('# Options for output on screen:\n')
        f.write('# W - wxt terminal\n')
        f.write('# X - x11 terminal\n')
        f.write('# N - no screen output [default]\n')
        f.write('# NOTE: An illustration of the sky subtraction quality is plotted into a PS\n')
        f.write('#       file in the OUTPUT_DIR folder in any case.\n')
        f.write('PLOT_TYPE=N\n')

        f.close()

    def runSkycorr(self):
        t1 = time.time()
        print('\tRunning skycorr')
        print('\t%s' %(' '.join([skycorrCall, self.skycorrparfile])))
        skycorrpar = os.path.abspath(self.output_path+'/%s_skycorr_%s.output' % (self.name, self.arm))
        with open(skycorrpar, 'w') as f:
            runskycorr = subprocess.run([skycorrCall, self.skycorrparfile],
                                        stdout=f)
        f.close()
        runskycorrRes = subprocess.check_output(['tail', '-1', skycorrpar]).decode("utf-8")


        if '[ INFO  ] No errors occurred' in runskycorrRes:
            print('\tskycorr sucessful in %.0f s' % (time.time()-t1))
        else:
            print(runskycorrRes)


