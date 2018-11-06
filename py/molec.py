#!/usr/bin/env python

""" Correcting for telluric absorption. Primarily used for X-shooter spectra.
20.08.2017 - Ported to Python 3 by Jonatan Selsing

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

molecBase = os.path.expanduser('/usr/local/astrosoft/ESO/molecfit_1.4')
molecCall = os.path.join(molecBase, 'bin/molecfit')
transCall = os.path.join(molecBase, 'bin/calctrans')


class molecFit():

    ''' Sets up and runs a molecfit to spectral data - requires molecfit in its
    version 1.2.0 or greater.
    See http://www.eso.org/sci/software/pipelines/skytools/molecfit
    '''

    def __init__(self, fitsfile_path, object_name, output_path):
        ''' Reads in the required information to set up the parameter files'''
        self.input_file_path = fitsfile_path
        self.fits = fits.open(self.input_file_path)
        self.arm = self.fits[0].header["HIERARCH ESO SEQ ARM"].lower()
        self.name = object_name
        self.output_path = output_path
        self.molecparfile = []
        self.params = {'basedir': molecBase,
            'listname': 'none', 'trans' : 1,
            'columns': 'WAVE FLUX ERR QUAL',
            'default_error': 0.01, 'wlgtomicron' : 0.001,
            'vac_air': 'vac', 'list_molec': [], 'fit_molec': [],
            'wrange_exclude': 'none',
            'output_dir': os.path.abspath(os.path.expanduser(output_path)),
            'plot_creation' : '', 'plot_range': 1,
            'ftol': 0.01, 'xtol': 0.01,
            'relcol': [], 'flux_unit': 2,
            'fit_back': 0, 'telback': 0.1, 'fit_cont': 1, 'cont_n': 4,
            'cont_const': 1.0, 'fit_wlc': 1, 'wlc_n': 1, 'wlc_const': 0.0,
            'fit_res_box': 0, 'relres_box': 0.0, 'kernmode': 1,
            'fit_res_gauss': 1, 'res_gauss': 1.5,
            'fit_res_lorentz': 1, 'res_lorentz': 0.1,
            'kernfac': 30.0, 'varkern': 1, 'kernel_file': 'none'}

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
            'gdas_dir': os.path.join(molecBase, 'data/profiles/grib'),
            'gdas_prof': 'auto', 'layers': 1, 'emix': 5.0, 'pwv': -1.}

    def updateParams(self, paramdic, headpars):
        ''' Sets Parameters for molecfit execution'''

        for key in paramdic.keys():
            print('\t\tMolecfit: Setting parameter %s to %s' %(key,  paramdic[key]))
            if key in self.params.keys():
                self.params[key] = paramdic[key]
            elif key in self.headpars.keys():
                self.headpars[key] = paramdic[key]
            elif key in self.atmpars.keys():
                self.atmpars[key] = paramdic[key]
            else:
                print('\t\tWarning: Parameter %s not known to molecfit')
                self.params[key] = paramdic[key]


    def setParams(self):
        ''' Writes Molecfit parameters into file '''

        wlinc = {
            # 'vis': [[0.686, 0.694] , [0.762, 0.770], [0.940, 0.951], [0.964, 0.974]],
            'vis': [[0.686, 0.694], [0.715, 0.730], [0.758, 0.762], [0.762, 0.770], [0.940, 0.951], [0.964, 0.974]],

            # 'nir': [[1.12, 1.13], [1.14, 1.15], [1.23, 1.26], [1.47, 1.48], [1.48, 1.49], [1.80, 1.81]]}
            'nir': [[1.12, 1.13], [1.14, 1.15], [1.47, 1.48], [1.48, 1.49], [1.80, 1.81]]}


        if self.arm == 'vis':
            slitkey = 'HIERARCH ESO INS OPTI4 NAME'
        elif self.arm == 'nir':
            slitkey = 'HIERARCH ESO INS OPTI5 NAME'
        elif self.arm == 'uvb':
            slitkey = 'HIERARCH ESO INS OPTI3 NAME'

        if not self.fits[0].header[slitkey].endswith('JH'):
            wlinc['nir'].append([1.96, 1.97])
            wlinc['nir'].append([2.06, 2.07])
            wlinc['nir'].append([2.35, 2.36])


        wrange = self.output_path+'/%s_molecfit_%s_inc.dat' %(self.name, self.arm)
        f = open(wrange, 'w')
        for wls in wlinc[self.arm]:
            f.write('%.4f %.4f\n' %(wls[0], wls[1]))
        f.close

        self.params['wrange_include'] =  os.path.abspath(wrange)
        # wrange = os.path.join(molecBase, 'examples/config/include_xshoo_%s.dat' % self.arm)
        # self.params['wrange_include'] =  wrange
        prange = os.path.join(molecBase, 'examples/XSHOOTER/exclude_xshoo_%s.dat' % self.arm)
        self.params['prange_exclude'] =  prange

        if self.arm == 'vis':
            self.params['list_molec'] = ['H2O', 'CO2', 'CO', 'CH4', 'O2']
            self.params['fit_molec'] = [1, 1, 1, 1, 1]
            self.params['relcol'] = [1.0, 1.0, 1.0, 1.0, 1.0]
            self.params['res_gauss'] /= 3
            self.params['res_lorentz'] /= 3
            # self.params['wlc_n'] = 0

        elif self.arm == 'nir':
            self.params['list_molec'] = ['H2O', 'CO2', 'CO', 'CH4', 'O2']
            self.params['fit_molec'] = [1, 1, 1, 1, 1]
            self.params['relcol'] = [1.0, 1.06, 1.0, 1.0, 1.0]
            # self.params['wlc_n'] = 0

        print('\tPreparing Molecfit params file')
        self.molecparfile = os.path.abspath(self.output_path+'/%s_molecfit_%s.par' % (self.name, self.arm))
        f = open(self.molecparfile, 'w')
        f.write('## INPUT DATA\n')
        f.write('filename: %s\n' % (os.path.abspath(self.input_file_path)))
        f.write('output_name: %s_%s_molecfit\n' % (self.name, self.arm))

        for param in self.params.keys():
            if isinstance(self.params[param], list):
                f.write('%s: %s \n' % (param, ' '.join([str(a) for a in self.params[param]]) ))
            else:
                f.write('%s: %s \n' % (param, self.params[param]))

        f.write('\n## HEADER PARAMETERS\n')
        slitw = self.fits[0].header[slitkey].split('x')[0]

        f.write('slitw: %s\n' %(slitw))
        for headpar in self.headpars.keys():
            f.write('%s: %s \n' % (headpar,  self.fits[0].header[self.headpars[headpar]]))

        f.write('\n## ATMOSPHERIC PROFILES\n')
        for atmpar in self.atmpars.keys():
            f.write('%s: %s \n' % (atmpar, self.atmpars[atmpar]))

        f.write('\nend\n')
        f.close()

    def runMolec(self):
        t1 = time.time()
        print('\tRunning molecfit')
        print('\t%s' %(' '.join([molecCall, self.molecparfile])))
        molecpar = os.path.abspath(self.output_path+'/%s_molecfit_%s.output' % (self.name, self.arm))
        with open(molecpar, 'w') as f:
            runMolec = subprocess.run([molecCall, self.molecparfile],
                                        stdout=f)
        f.close()
        runMolecRes = subprocess.check_output(['tail', '-1', molecpar]).decode("utf-8")


        if '[ INFO  ] No errors occurred' in runMolecRes:
            print('\tMolecfit sucessful in %.0f s' % (time.time()-t1))
        else:
            print(runMolecRes)

        t1 = time.time()
        print('\tRunning calctrans')
        print('\t%s' %(' '.join([transCall, self.molecparfile])))
        runtranspar = os.path.abspath(self.output_path+'/%s_calctrans_%s.output' % (self.name, self.arm))
        with open(runtranspar, 'w') as f:
            runMolec = subprocess.run([transCall, self.molecparfile],
                                        stdout=f)
        f.close()
        runTransRes = subprocess.check_output(['tail', '-1', runtranspar]).decode("utf-8")

        if '[ INFO  ] No errors occurred' in runTransRes:
            print('\tCalctrans sucessful in %.0f s' % (time.time()-t1))
        else:
            print(runTransRes)


    def updateSpec(self, tacfile = ''):

        ''' Read in Calctrans output und update spectrum2d class with telluric
        correction spectrum '''
        print('\tUpdating the spectra with telluric-corrected data')
        if tacfile == '':
            os.rename(self.output_path + self.input_file_path.split("/")[-1].split(".")[0] + '_TAC.fits', self.output_path + self.name + '_TAC.fits')
            tacfilearm = self.output_path + self.name + '_TAC.fits'
            tacfile = fits.open(tacfilearm)

        if os.path.isfile(tacfilearm):
            f = fits.open(tacfilearm)

            wl = f[1].data.field("WAVE")
            rawspec = f[1].data.field("FLUX")*1e17
            rawspece = f[1].data.field("ERR")*1e17
            transm = f[1].data.field("mtrans")
            tcspec = f[1].data.field("tacflux")*1e17
            tcspece = f[1].data.field("tacdflux")*1e17
            self.wave = wl
        # plt.plot(wl, rawspec)
        # plt.plot(wl, tcspec)
        # plt.show()
        # Plot the fit regions
        # wrange = os.path.join(molecBase, 'examples/config/include_xshoo_%s.dat' % self.arm)
        wrange = os.path.abspath(self.output_path+'/%s_molecfit_%s_inc.dat' %(self.name, self.arm))
        g = open(wrange, 'r')
        fitregs = [reg for reg in g.readlines() if not reg.startswith('#')]

        pp = PdfPages(self.output_path+'/%s_tellcor_%s.pdf' % (self.name, self.arm) )
        print('\tPlotting telluric-corrected data for arm %s' %self.arm)

        for fitreg in fitregs:
            mictowl = 1./self.params['wlgtomicron']
            if float(fitreg.split()[1])*mictowl < self.wave[-1]:
                x1 = self.wltopix(float(fitreg.split()[0])*mictowl)
                x2 = self.wltopix(float(fitreg.split()[1])*mictowl)

                fig = plt.figure(figsize = (9.5, 7.5))
                fig.subplots_adjust(hspace=0.05, wspace=0.0, right=0.97)

                ax1 = fig.add_subplot(2, 1, 1)
                wlp, tcspecp, transmp = wl[x1:x2], tcspec[x1:x2], transm[x1:x2]

                # if len(tcspecp[transmp>0.90]) > 3:
                cont = np.median(tcspecp[transmp>0.90])/np.median(transmp[transmp>0.90])
                ax1.plot(wlp, transmp * cont, '-', color = 'firebrick', lw = 2)
                # else:
                ax1.errorbar(wlp, tcspec[x1:x2], tcspece[x1:x2],
                capsize = 0, color = 'firebrick', fmt = 'o', ms = 4,
                mec = 'grey', lw = 0.8, mew = 0.5)

                ax1.errorbar(wlp, rawspec[x1:x2], rawspece[x1:x2],
                    capsize = 0, color = 'black', fmt = 'o', ms = 4,
                    mec = 'grey', lw = 0.8, mew = 0.5)

                ax2 = fig.add_subplot(2, 1, 2)
                if len(tcspecp[transmp>0.90]) > 3:
                    ax2.errorbar(wlp, rawspec[x1:x2] / (transmp * cont),
                        yerr = rawspece[x1:x2] / (transmp * cont),
                        capsize = 0, color = 'black', fmt = 'o',  ms = 4,
                        mec = 'grey', lw = 0.8, mew = 0.5)
                else:
                    ax2.plot(wlp, transmp, '-' ,color = 'firebrick', lw = 2)


                for ax in [ax1, ax2]:
                    ax.yaxis.set_major_formatter(plt.FormatStrFormatter(r'$%s$'))
                    ax.set_xlim(float(fitreg.split()[0])*mictowl, float(fitreg.split()[1])*mictowl)
                    ax.set_ylim(ymin=-0.05)

                ax1.xaxis.set_major_formatter(plt.NullFormatter())
                ax2.set_xlabel(r'$\rm{Observed\,wavelength\, (\AA)}$')
                ax1.set_ylabel(r'$F_{\lambda}\,\rm{(10^{-%s}\,erg\,s^{-1}\,cm^{-2}\, \AA^{-1})}$' \
                             %(str(1e17))[-2:])
                if len(tcspecp[transmp>0.90]) > 3:
                    ax2.set_ylim(ymax = max(rawspec[x1:x2] / (transmp * cont))*1.05)
                    ax2.set_ylabel(r'Ratio')
                else:
                    ax2.set_ylim(ymax=1.05)
                    ax2.set_ylabel(r'Transmission')

                ax1.set_ylim(ymax = max(tcspec[x1:x2])*1.05)
                pp.savefig(fig)
                plt.close(fig)
        pp.close()


    def wltopix(self, wl):
        dl = (self.wave[-1]-self.wave[0]) / (len(self.wave) - 1)
        pix = ((wl - self.wave[0]) / dl) + 1
        return max(0, int(round(pix)))
