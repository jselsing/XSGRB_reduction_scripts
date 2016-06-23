#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Importing manupulation packages
from astropy.io import fits
import numpy as np
import glob
from numpy.polynomial import chebyshev
from scipy import interpolate
from scipy import optimize

# Plotting
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')
import copy

from util import *
from XSHcomb import XSHcomb, weighted_avg

class XSHextract(XSHcomb):
    """
    Class to contain XSH spectrscopy extraction.
    """
    def __init__(self, input_file, base_name, resp):
        """
        Instantiate fitsfiles. Input list of file-names to be combined.
        """

        if len(input_file) == 0:
            raise ValueError("Input file list empty")

        self.input_file = input_file
        self.fitsfile = fits.open(self.input_file)
        self.header = self.fitsfile[0].header
        try:
            self.flux = self.fitsfile[0].data
        except:
            raise ValueError("No flux-array. Aborpting")
        try:
            self.error = self.fitsfile[1].data
        except IndexError:
            print("Empty error extension. Inserting placeholder of ones")
            self.error = np.ones_like(self.flux)
        try:
            self.bpmap = self.fitsfile[2].data
        except IndexError:
            print("Empty bad-pixel bap extension. Inserting placeholder of zeroes")
            self.bpmap = np.zeros_like(self.flux)

        self.flux = np.ma.array(self.flux, mask=self.bpmap.astype("bool"))
        self.error = np.ma.array(self.error, mask=self.bpmap.astype("bool"))

        self.base_name = base_name

        # Apply flux calibration from master response file
        resp = fits.open(resp)
        self.wl_response, self.response = resp[1].data.field('LAMBDA'), resp[1].data.field('RESPONSE')
        # pl.plot(self.wl_response, self.response)
        # pl.show()
        f = interpolate.interp1d(10 * self.wl_response, self.response, bounds_error=False)
        self.response = f(10.*((np.arange(self.header['NAXIS1']) - self.header['CRPIX1'])*self.header['CD1_1']+self.header['CRVAL1'])/(1 + self.header['WAVECORR']))

        try:
            gain = self.header['HIERARCH ESO DET OUT1 GAIN']
        except:
            gain = 2.12
        # Applyu atmospheric extinciton correction
        atmpath = "/opt/local/share/esopipes/datastatic/xshoo-2.7.1/xsh_paranal_extinct_model_"+self.header['HIERARCH ESO SEQ ARM'].lower()+".fits"
        ext_atm = fits.open(atmpath)
        self.wl_ext_atm, self.ext_atm = ext_atm[1].data.field('LAMBDA'), ext_atm[1].data.field('EXTINCTION')
        f = interpolate.interp1d(10 * self.wl_ext_atm, self.ext_atm, bounds_error=False)
        self.ext_atm = f(10.*((np.arange(self.header['NAXIS1']) - self.header['CRPIX1'])*self.header['CD1_1']+self.header['CRVAL1'])/(1 + self.header['WAVECORR']))

        self.response = (10 * self.header['CD1_1'] * self.response * (10**(0.4*self.header['HIERARCH ESO TEL AIRM START'] * self.ext_atm))) / ( gain * self.header['EXPTIME'])


    def get_trace_profile(self, seeing_pix):

        # Get binned spectrum
        bin_length = int(len(self.haxis) / 200)
        bin_flux, bin_error = bin_image(self.flux, self.error, bin_length)
        bin_haxis = 10.*(((np.arange(self.header['NAXIS1']/bin_length)) - self.header['CRPIX1'])*self.header['CD1_1']*bin_length+self.header['CRVAL1'])

        # Save binned image for quality control
        self.fitsfile[0].data = bin_flux.data
        self.fitsfile[1].data = bin_error.data
        self.fitsfile[0].header["CD1_1"] = self.fitsfile[0].header["CD1_1"] * bin_length
        self.fitsfile.writeto(self.base_name+"_binned.fits", clobber=True)
        self.fitsfile[0].header["CD1_1"] = self.fitsfile[0].header["CD1_1"] / bin_length

        # Inital parameter guess
        p0 = [1, np.median(self.vaxis), 0.1, 0.1]
        # Parameter containers
        amp, cen, sig, gam = np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis)
        eamp, ecen, esig, egam = np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis)

        # Loop though along dispersion axis in the binned image and fit a Voigt
        for ii, kk in enumerate(bin_haxis):
            try:
                popt, pcov = optimize.curve_fit(voigt, self.vaxis, bin_flux[:, ii], p0 = p0, maxfev = 5000)
                # pl.errorbar(self.vaxis, bin_flux[:, ii], yerr=bin_error[:, ii], fmt=".k", capsize=0, elinewidth=0.5, ms=3)
                # pl.plot(self.vaxis, voigt(self.vaxis, *popt))
                # pl.title(ii)
                # pl.show()
            except:
                print("Fitting error at index: "+str(ii)+". Replacing fit value with guess and set fit error to 10^10")
                popt, pcov = p0, np.diag(1e10*np.ones_like(p0))
            amp[ii], cen[ii], sig[ii], gam[ii] = popt[0], popt[1], popt[2], popt[3]
            eamp[ii], ecen[ii], esig[ii], egam[ii] = np.diag(pcov)[0], np.diag(pcov)[1], np.diag(pcov)[2], np.diag(pcov)[3]

        # ecen[:55] = 1e10
        # ecen[-25:] = 1e10
        ecen[abs(amp - p0[0]) < p0[0]/1000] = 1e10
        ecen[abs(cen - p0[1]) < p0[1]/1000] = 1e10
        ecen[abs(sig - p0[2]) < p0[2]/1000] = 1e10
        # ecen[gam < 0] = 1e10
        # ecen[sig < 0] = 1e10
        ecen[gam/egam > 50] = 1e10

        ecen[abs(gam - p0[3]) < p0[3]/1000] = 1e10
        # Fit polynomial for center and iteratively reject outliers
        std_resid = 5
        while std_resid > 0.5:
            fitcen = chebyshev.chebfit(bin_haxis, cen, deg=3, w=1/ecen)
            resid = cen - chebyshev.chebval(bin_haxis, fitcen)
            avd_resid, std_resid = np.median(resid[ecen != 1e10]), np.std(resid[ecen != 1e10])
            mask = (resid < avd_resid - std_resid) | (resid > avd_resid + std_resid)
            ecen[mask] = 1e10
        fitcenval = chebyshev.chebval(self.haxis, fitcen)
        # Plotting for quality control
        fig, (ax1, ax2, ax3, ax4) = pl.subplots(4,1, figsize=(14, 14), sharex=True)

        ax1.errorbar(bin_haxis, cen, yerr=ecen, fmt=".k", capsize=0, elinewidth=0.5, ms=7)
        ax1.plot(self.haxis, fitcenval)
        vaxis_range = max(self.vaxis) - min(self.vaxis)
        ax1.set_ylim((min(self.vaxis) + 0.33 * vaxis_range, max(self.vaxis) - 0.33 * vaxis_range))
        ax1.set_ylabel("Profile center / [arcsec]")
        ax1.set_title("Quality test: Center estimate")
        # Sigma-clip outliers in S/N-space
        esig[ecen == 1e10] = 1e10
        # snsig = sig/esig
        # esig[snsig > 500 ] = 1e10
        # esig[snsig < 10 ] = 1e10
        fitsig = chebyshev.chebfit(bin_haxis, sig, deg=2, w=1/esig**2)
        fitsigval = chebyshev.chebval(self.haxis, fitsig)

        # Plotting for quality control
        ax2.errorbar(bin_haxis, sig, yerr=esig, fmt=".k", capsize=0, elinewidth=0.5, ms=7)
        ax2.plot(self.haxis, fitsigval)
        ax2.set_ylim((0, 1))
        ax2.set_ylabel("Profile sigma width / [arcsec]")
        ax2.set_title("Quality test: Profile Gaussian width estimate")

        # Sigma-clip outliers in S/N-space
        egam[ecen == 1e10] = 1e10
        fitgam = chebyshev.chebfit(bin_haxis, gam, deg=2, w=1/egam**2)
        fitgamval = chebyshev.chebval(self.haxis, fitgam)
        # Ensure positivity
        fitgamval[fitgamval < 0] = 0

        # Plotting for quality control
        ax3.errorbar(bin_haxis, gam, yerr=egam, fmt=".k", capsize=0, elinewidth=0.5, ms=7)
        ax3.plot(self.haxis, fitgamval)
        ax3.set_ylim((-0.1, 1.0))
        ax3.set_ylabel("Profile gamma width / [arcsec]")
        ax3.set_title("Quality test: Profile Lorentzian width estimate")

        # Amplitude replaced with ones
        from scipy import interpolate, signal
        # amp[amp <= 0] = 0
        eamp[ecen == 1e10] = 1e10
        # eamp[esig == 1e10] = 1e10
        amp = signal.medfilt(amp, 5)
        mask = ~(eamp == 1e10)
        f = interpolate.interp1d(bin_haxis[mask], amp[mask], bounds_error=False, fill_value="extrapolate")
        fitampval = f(self.haxis)
        fitampval[fitampval < 0] = 0

        # Plotting for quality control
        ax4.errorbar(bin_haxis, amp, fmt=".k", capsize=0, elinewidth=0.5, ms=5)
        ax4.plot(self.haxis, fitampval)
        # ax4.set_ylim((0, 1))
        ax4.set_ylabel("Profile amplitude / [counts/s]")
        ax4.set_title("Quality test: Profile amplitude estimate")
        ax4.set_xlabel(r"Spectral index / [$\mathrm{\AA}$]")
        fig.subplots_adjust(hspace=0)
        fig.savefig(self.base_name + "PSF_quality_control.pdf")
        pl.close(fig)

        # Calculating slitt-losses based on fit-width
        if self.header['HIERARCH ESO SEQ ARM'] == "UVB":
            slit_width = float(self.header['HIERARCH ESO INS OPTI3 NAME'].split("x")[0])
        elif self.header['HIERARCH ESO SEQ ARM'] == "VIS":
            slit_width = float(self.header['HIERARCH ESO INS OPTI4 NAME'].split("x")[0])
        elif self.header['HIERARCH ESO SEQ ARM'] == "NIR":
            slit_width = float(self.header['HIERARCH ESO INS OPTI5 NAME'].split("x")[0])
        self.slitcorr = slit_loss(fitsigval, slit_width)

        self.full_profile, self.trace_model = np.zeros_like(self.flux), np.zeros_like(self.flux)
        for ii, kk in enumerate(self.haxis):
            self.trace_model[:, ii] = voigt(self.vaxis, fitampval[ii], fitcenval[ii], fitsigval[ii], fitgamval[ii])
            self.full_profile[:, ii] = self.trace_model[:, ii] / abs(np.trapz(self.trace_model[:, ii]))

    def extract_spectrum(self, seeing, optimal=True, slitcorr=True):

        """Optimally extracts a spectrum from sky-subtracted X-shooter image.

        Function to extract spectra from X-shooter images. Either sums the flux in a central aperture or uses a profile-weighted extraction.

        fitsfile : fitsfile
            Input sky-subtracted image with flux, error and bad-pixel map in extensions 0, 1, 2 respectively.
        seeing : float
            Seeing of observation used to find extraction width
        outname : str
            Name of saved spectrum
        Returns
        -------
        Wavelength, Extracted spectrum, Associated error array : np.array, np.array, np.array

        Notes
        -----
        na
        """

        if optimal:
            print("Optimally extracting spectrum....")
        elif not optimal:
            print("Extracting spectrum by summing over 3* seeingFWHM....")
        # Making barycentric correction to wavlength solution.
        self.haxis = 10.*(((np.arange(self.header['NAXIS1'])) - self.header['CRPIX1'])*self.header['CDELT1']+self.header['CRVAL1'])
        self.haxis = self.haxis + self.haxis*self.header['HIERARCH ESO QC VRAD BARYCOR']/3e5
        self.vaxis =  (((np.arange(self.header['NAXIS2'])) - self.header['CRPIX2'])*self.header['CD2_2']+self.header['CRVAL2'])
        # Finding extraction radius
        seeing_pix = seeing / self.header['CD2_2']
        # Construct spatial PSF to be used as weight in extraction
        if optimal:
            print("Fitting for the full spectral extraction profile")
            XSHextract.get_trace_profile(self, seeing_pix)
            self.fitsfile[0].data = (self.flux - self.trace_model).data
            self.fitsfile[1].data = self.error.data
            self.fitsfile.writeto(self.base_name + "Profile_subtracted_image.fits", clobber=True)

        elif not optimal:
            print("Using simplified extraction profile")
            profile = np.ma.median(self.flux[:, self.header['NAXIS1']/10:-self.header['NAXIS1']/10], axis = 1)
            self.full_profile = np.tile(profile/np.sum(profile) , (self.header['NAXIS1'], 1)).T
            # Calculating slit-loss based on specified seeing.
            if self.header['HIERARCH ESO SEQ ARM'] == "UVB":
                slit_width = float(self.header['HIERARCH ESO INS OPTI3 NAME'].split("x")[0])
            elif self.header['HIERARCH ESO SEQ ARM'] == "VIS":
                slit_width = float(self.header['HIERARCH ESO INS OPTI4 NAME'].split("x")[0])
            elif self.header['HIERARCH ESO SEQ ARM'] == "NIR":
                slit_width = float(self.header['HIERARCH ESO INS OPTI5 NAME'].split("x")[0])
            self.slitcorr = slit_loss(seeing/2.35, slit_width)

            # Masking pixels not in trace.
            trace_mask = np.zeros(self.header['NAXIS2']).astype("bool")
            trace_mask[int(self.header['NAXIS2']/2 - 3*seeing_pix): int(self.header['NAXIS2']/2 + 3*seeing_pix)] = True
            full_trace_mask = ~np.tile(trace_mask , (self.header['NAXIS1'], 1)).T
            self.flux.mask, self.error.mask = self.flux.mask | ~full_trace_mask, self.error.mask | ~full_trace_mask
        print("Correcting for slitloss. Estimated correction factor is:"+str(self.slitcorr))

        if optimal:
            # Do optimal extraction
            denom = np.ma.sum((self.full_profile**2. / self.error**2.), axis=0)
            spectrum = np.ma.sum(self.full_profile * self.flux.data / self.error.data**2., axis=0) / denom*self.slitcorr
            errorspectrum = np.sqrt(1 / denom)*self.slitcorr
            extname = "optext.dat"
        elif not optimal:
            # Do normal sum
            spectrum, errorspectrum = np.ma.sum(self.flux, axis=0)*self.slitcorr, np.sqrt(np.ma.sum(self.error**2.0, axis=0))*self.slitcorr
            extname = "stdext.dat"
        else:
            print("Optimal argument need to be boolean")

        extinc_corr, ebv = correct_for_dust(self.haxis, self.header["RA"], self.header["DEC"])
        print("Applying the following extinction correction for queried E(B-V):"+str(ebv))
        print(extinc_corr)
        spectrum *= extinc_corr
        errorspectrum *= extinc_corr
        print("Applying the master response function")
        spectrum *= self.response
        errorspectrum *= self.response


        dt = [("wl", np.float64), ("flux", np.float64), ("error", np.float64), ("response", np.float64), ("slitcorr", np.float64), ("extinc", np.float64) ]
        data = np.array(zip(self.haxis, spectrum, errorspectrum, self.response, self.slitcorr, extinc_corr), dtype=dt)
        head = "wavelength flux error response_function slitloss_correction E(B-V) = "+str(ebv)
        np.savetxt(self.base_name + extname, data, header=head, fmt = ['%1.5e', '%1.5e', '%1.5e', '%1.5e', '%1.5e', '%1.5e'] )

        return self.haxis, spectrum, errorspectrum


def main():
    """
    Central scipt to extract spectra from X-shooter for the X-shooter GRB sample.
    """
    data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/"
    object_name = data_dir + "GRB100316D/"
    arm = "UVB" # UVB, VIS, NIR

    # Load in file
    files = glob.glob(object_name+arm+"_combined.fits")

    for ii, kk in enumerate(glob.glob(object_name+"data_with_raw_calibs/M*.fits")):
        try:
            filetype = fits.open(kk)[0].header["CDBFILE"]
            if "GRSF" in filetype and arm in filetype:
                response_file = kk
        except:
            pass

    spec = XSHextract(files[0], object_name+arm, resp=response_file)
    # Optimal extraction
    wl, flux, error = spec.extract_spectrum(seeing=1.5, optimal=True, slitcorr=True)

    # fig, ax = pl.subplots()
    # SN = True
    # if SN:
    #     ax.plot(wl, flux/error, lw = 1, linestyle="steps-mid", alpha=0.7, label="Optimally extracted")
    # if not SN:
    #     ax.errorbar(wl, flux, yerr=error, fmt=".k", capsize=0, elinewidth=0.5, ms=3)
    #     ax.plot(wl, flux, lw = 1, linestyle="steps-mid", alpha=0.7)
    # ax.set_ylabel("Flux density")
    # ax.set_xlabel("Wavelength")
    # pl.show()


if __name__ == '__main__':
    main()









