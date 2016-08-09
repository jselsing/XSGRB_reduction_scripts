#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Import parser
import sys
import argparse
import os

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
    def __init__(self, input_file, resp=None):
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

        self.base_name = "/".join(input_file.split("/")[:-1]) + "/" + self.header['HIERARCH ESO SEQ ARM']

        if resp:
            # Apply flux calibration from master response file
            resp = fits.open(resp)
            self.wl_response, self.response = resp[1].data.field('LAMBDA'), resp[1].data.field('RESPONSE')

            f = interpolate.interp1d(10 * self.wl_response, self.response, bounds_error=False)
            self.response = f(10.*((np.arange(self.header['NAXIS1']) - self.header['CRPIX1'])*self.header['CD1_1']+self.header['CRVAL1'])/(1 + self.header['WAVECORR']))

            try:
                gain = self.header['CONAD']
            except:
                gain = 2.12
            # Applyu atmospheric extinciton correction
            atmpath = "/opt/local/share/esopipes/datastatic/xshoo-2.8.0/xsh_paranal_extinct_model_"+self.header['HIERARCH ESO SEQ ARM'].lower()+".fits"
            ext_atm = fits.open(atmpath)
            self.wl_ext_atm, self.ext_atm = ext_atm[1].data.field('LAMBDA'), ext_atm[1].data.field('EXTINCTION')
            f = interpolate.interp1d(10 * self.wl_ext_atm, self.ext_atm, bounds_error=False)
            self.ext_atm = f(10.*((np.arange(self.header['NAXIS1']) - self.header['CRPIX1'])*self.header['CD1_1']+self.header['CRVAL1'])/(1. + self.header['WAVECORR']))

            self.response = (10. * self.header['CD1_1'] * self.response * (10.**(0.4*self.header['HIERARCH ESO TEL AIRM START'] * self.ext_atm))) / ( gain * self.header['EXPTIME']) 

        # Get slit width
        if self.header['HIERARCH ESO SEQ ARM'] == "UVB":
            self.slit_width = float(self.header['HIERARCH ESO INS OPTI3 NAME'].split("x")[0])
        elif self.header['HIERARCH ESO SEQ ARM'] == "VIS":
            self.slit_width = float(self.header['HIERARCH ESO INS OPTI4 NAME'].split("x")[0])
        elif self.header['HIERARCH ESO SEQ ARM'] == "NIR":
            self.slit_width = float(self.header['HIERARCH ESO INS OPTI5 NAME'].split("x")[0])


    def get_trace_profile(self, n_fit_elements = 200, lower_element_nr = 1, upper_element_nr = 1, pol_degree = [3, 2, 2]):

        # Get binned spectrum
        bin_length = int(len(self.haxis) / n_fit_elements)
        bin_flux, bin_error = bin_image(self.flux, self.error, bin_length)
        bin_haxis = 10.*(((np.arange(self.header['NAXIS1']/bin_length)) - self.header['CRPIX1'])*self.header['CD1_1']*bin_length+self.header['CRVAL1'])

        # Save binned image for quality control
        self.fitsfile[0].data = bin_flux.data
        self.fitsfile[1].data = bin_error.data
        self.fitsfile[0].header["CD1_1"] = self.fitsfile[0].header["CD1_1"] * bin_length
        self.fitsfile.writeto(self.base_name+"_binned.fits", clobber=True)
        self.fitsfile[0].header["CD1_1"] = self.fitsfile[0].header["CD1_1"] / bin_length

        # Inital parameter guess
        p0 = [50 * np.mean(bin_flux), np.median(self.vaxis), 0.2, 0.2]
        # Parameter containers
        amp, cen, sig, gam = np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis)
        eamp, ecen, esig, egam = np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis)

        # Loop though along dispersion axis in the binned image and fit a Voigt
        for ii, kk in enumerate(bin_haxis):
            try:
                width = int(len(self.vaxis)/4)
                popt, pcov = optimize.curve_fit(voigt, self.vaxis[width:-width], bin_flux[:, ii][width:-width], p0 = p0, maxfev = 5000)
                # pl.errorbar(self.vaxis[width:-width], bin_flux[:, ii][width:-width], yerr=bin_error[:, ii][width:-width], fmt=".k", capsize=0, elinewidth=0.5, ms=3)
                # pl.plot(self.vaxis[width:-width], voigt(self.vaxis, *popt)[width:-width])
                # pl.title(ii)
                # pl.show()
            except:
                print("Fitting error at binned image index: "+str(ii)+". Replacing fit value with guess and set fit error to 10^10")
                popt, pcov = p0, np.diag(1e10*np.ones_like(p0))
            amp[ii], cen[ii], sig[ii], gam[ii] = popt[0], popt[1], popt[2], popt[3]
            eamp[ii], ecen[ii], esig[ii], egam[ii] = np.diag(pcov)[0], np.diag(pcov)[1], np.diag(pcov)[2], np.diag(pcov)[3]

        # Mask elements too close to guess, indicating a bad fit.
        ecen[:lower_element_nr] = 1e10
        ecen[-upper_element_nr:] = 1e10
        ecen[abs(amp - p0[0]) < p0[0]/100] = 1e10
        ecen[abs(cen - p0[1]) < p0[1]/100] = 1e10
        ecen[abs(sig - p0[2]) < p0[2]/100] = 1e10
        ecen[abs(gam - p0[3]) < p0[3]/100] = 1e10

        # Remove the 5 highest S/N pixels
        ecen[np.argsort(sig/esig)[-5:]] = 1e10
        ecen[np.argsort(gam/egam)[-5:]] = 1e10

        # Fit polynomial for center and iteratively reject outliers
        std_resid = 5
        while std_resid > 0.5:
            fitcen = chebyshev.chebfit(bin_haxis, cen, deg=pol_degree[0], w=1/ecen)
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
        # esig[snsig > 100 ] = 1e10

        fitsig = chebyshev.chebfit(bin_haxis, sig, deg=pol_degree[1], w=1/esig**2)
        fitsigval = chebyshev.chebval(self.haxis, fitsig)
        # Ensure positivity
        fitsigval[fitsigval < 0.1] = 0.1

        # Plotting for quality control
        ax2.errorbar(bin_haxis, sig, yerr=esig, fmt=".k", capsize=0, elinewidth=0.5, ms=7)
        ax2.plot(self.haxis, fitsigval)
        ax2.set_ylim((0, 1))
        ax2.set_ylabel("Profile sigma width / [arcsec]")
        ax2.set_title("Quality test: Profile Gaussian width estimate")

        # Sigma-clip outliers in S/N-space
        egam[ecen == 1e10] = 1e10
        # sngam = gam/egam
        # egam[sngam > 100 ] = 1e10
        fitgam = chebyshev.chebfit(bin_haxis, gam, deg=pol_degree[2], w=1/egam**2)
        fitgamval = chebyshev.chebval(self.haxis, fitgam)
        # Ensure positivity
        fitgamval[fitgamval < 0] = 0.0001

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
        fitampval[fitampval <= 0] = 0.0001

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
        if hasattr(self, 'slitcorr'):
            self.slitcorr = slit_loss(fitsigval, self.slit_width)

        self.full_profile, self.trace_model = np.zeros_like(self.flux), np.zeros_like(self.flux)
        for ii, kk in enumerate(self.haxis):
            self.trace_model[:, ii] = voigt(self.vaxis, fitampval[ii], fitcenval[ii], fitsigval[ii], fitgamval[ii])
            self.full_profile[:, ii] = self.trace_model[:, ii] / abs(np.trapz(self.trace_model[:, ii]))

    def extract_spectrum(self, seeing, optimal=None, slitcorr=None, edge_mask=None, pol_degree=None):

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

        if slitcorr:
            self.slitcorr = slitcorr

        # Making barycentric correction to wavlength solution.
        self.haxis = 10.*(((np.arange(self.header['NAXIS1'])) - self.header['CRPIX1'])*self.header['CDELT1']+self.header['CRVAL1'])
        self.haxis = self.haxis + self.haxis*self.header['HIERARCH ESO QC VRAD BARYCOR']/3e5
        self.vaxis =  (((np.arange(self.header['NAXIS2'])) - self.header['CRPIX2'])*self.header['CD2_2']+self.header['CRVAL2'])

        # Finding extraction radius
        seeing_pix = seeing / (2.35*self.header['CD2_2'])

        # Construct spatial PSF to be used as weight in extraction
        if optimal:
            print("Fitting for the full spectral extraction profile")
            XSHextract.get_trace_profile(self, lower_element_nr = int(tuple(edge_mask)[0]), upper_element_nr = int(tuple(edge_mask)[1]), pol_degree=pol_degree)
            self.fitsfile[0].data = (self.flux - self.trace_model).data
            self.fitsfile[1].data = self.error.data
            self.fitsfile.writeto(self.base_name + "Profile_subtracted_image.fits", clobber=True)

        elif not optimal:
            print("Using simplified extraction. Extracting spectrum by summing a 3 seeing-sigma aperture. Seeing FWHM is: " + str(seeing) + " arcsec.")
            print("Extracting spectrum between pixel " +str(int(round(self.header['NAXIS2']/2 - 3*seeing_pix)))+ " and " +str(int(round(self.header['NAXIS2']/2 + 3*seeing_pix))))

            # Calculating slit-loss based on specified seeing.
            if hasattr(self, 'slitcorr'):
                self.slitcorr = slit_loss(seeing/2.35, self.slit_width)

            # Defining extraction aperture
            ext_aper = slice(int(round(self.header['NAXIS2']/2 - 3*seeing_pix)), int(round(self.header['NAXIS2']/2 + 3*seeing_pix)))

        # Interpolate over bad pixel map
        self.flux.data[self.flux.mask] = np.nan
        self.error.data[self.flux.mask] = 1e2
        self.error = self.error.data
        self.bpmap = self.flux.mask.astype("int")
        self.flux = inpaint_nans(self.flux.data, kernel_size=5)

        # Save interpolated image for quality control
        self.fitsfile[0].data = self.flux
        self.fitsfile[1].data = self.error
        self.fitsfile.writeto(self.base_name+"_interpolated.fits", clobber=True)

        if optimal:
            # Do optimal extraction
            denom = np.sum((self.full_profile**2. / self.error**2.), axis=0)
            spectrum = np.sum(self.full_profile * self.flux / self.error**2., axis=0) / denom
            errorspectrum = np.sqrt(1 / denom)

            # Sum up bpvalues to find interpoalted values in 2-sigma width
            self.bpmap[self.full_profile/np.max(self.full_profile) < 0.05] = 0
            bpmap = np.sum(self.bpmap, axis=0)
            extname = "optext.dat"
            # Unpack masked array
            spectrum = spectrum.data
            errorspectrum = errorspectrum.data
        elif not optimal:
            # Do normal sum
            spectrum, errorspectrum = np.sum(self.flux[ext_aper, :], axis=0), np.sqrt(np.sum(self.error[ext_aper, :]**2.0, axis=0))
            bpmap = np.sum(self.bpmap[ext_aper, :], axis=0)
            extname = "stdext.dat"
        else:
            print("Optimal argument need to be boolean")

        # Boost error in noisy pixels, where noisy pixels are more than 50-sigma pixel-to-pixel variation based on error map
        mask = (abs(np.diff(spectrum)) > 50 * errorspectrum[1:])
        errorspectrum[1:][mask] = 10*max(errorspectrum)
        bpmap[1:][mask] = 1

        extinc_corr, ebv = correct_for_dust(self.haxis, self.header["RA"], self.header["DEC"])
        print("Applying the following extinction correction for queried E(B-V):"+str(ebv))
        print(extinc_corr)
        spectrum *= extinc_corr
        errorspectrum *= extinc_corr

        dt = [("wl_air", np.float64), ("wl_vac", np.float64), ("flux", np.float64), ("error", np.float64), ("bpmap", np.float64), ("extinc", np.float64)]
        out_data = [self.haxis, convert_air_to_vacuum(self.haxis), spectrum, errorspectrum, bpmap, extinc_corr]
        formatt = ['%10.6e', '%10.6e', '%10.6e', '%10.6e', '%10.6e', '%10.6e']
        head = "air_wavelength vacuum_wavelength flux error bpmap E(B-V) = "+str(ebv)

        if hasattr(self, 'response'):
            print("Applying the master response function")
            spectrum *= self.response
            errorspectrum *= self.response
            dt.append(("response", np.float64))
            out_data.append(self.response)
            formatt.append('%10.6e')
            head = head + " reponse"

        if hasattr(self, 'slitcorr'):
            print("Correcting for slitloss. Estimated correction factor is:"+str(self.slitcorr))
            if type(self.slitcorr) == np.float64:
                self.slitcorr = np.ones_like(spectrum) * self.slitcorr
            spectrum *= self.slitcorr
            errorspectrum *= self.slitcorr
            dt.append(("slitcorr", np.float64))
            out_data.append(self.slitcorr)
            formatt.append('%10.6e')
            head = head + " slitloss_correction_factor"


        data = np.array(zip(*out_data), dtype=dt)
        # head = "wavelength flux error response_function slitloss_correction E(B-V) = "+str(ebv)
        # formatt = ['%1.5e', '%1.5e', '%1.5e', '%1.5e', '%1.5e', '%1.5e']
        np.savetxt(self.base_name + extname, data, header=head, fmt = formatt, delimiter="\t")

        return self.haxis, spectrum, errorspectrum


def run_extraction(args):
    spec = XSHextract(args.filepath, resp = args.response_path)
    # Optimal extraction
    wl, flux, error = spec.extract_spectrum(seeing=args.seeing, optimal=args.optimal, slitcorr=args.slitcorr, edge_mask=args.edge_mask, pol_degree=args.pol_degree)

    if args.plot_ext:
        fig, ax = pl.subplots()
        ax.errorbar(wl[::5], flux[::5], yerr=error[::5], fmt=".k", capsize=0, elinewidth=0.5, ms=3, alpha=0.5)
        ax.plot(wl[::5], flux[::5], lw = 0.2, linestyle="steps-mid", alpha=0.5, rasterized=True)
        # ax.plot(wl, flux/error, lw = 1, linestyle="steps-mid", alpha=0.7)

        m = np.average(flux[~np.isnan(flux)], weights=1/error[~np.isnan(flux)])
        pl.xlim(min(wl), max(wl))
        pl.ylim(-0.2*m, 5*m)
        pl.xlabel(r"Wavelength / [$\mathrm{\AA}$]")
        pl.ylabel(r'Flux density [erg s$^{-1}$ cm$^{-1}$ $\AA^{-1}$]')
        pl.savefig(args.filepath[:-13] + "extraction.pdf")


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str, help='Path to file on which to run extraction')
    parser.add_argument('-response_path', type=str, help='Response function to apply. Can either be a path to file or path to directory. If directory, will look for correct file.')
    parser.add_argument('-seeing', type=float, default=1, help='Estimated seeing of observations. Used for standard extraction width')
    parser.add_argument('-edge_mask', type=str, default="1, 1", help='Tuple containing the edge masks. (10, 10) means that 10 pixels are masked at each edge.')
    parser.add_argument('-pol_degree', type=str, default=[3, 2, 2], help='List containing the edge masks. Each number specify the degree of the polynomial used for the fit in central prosition, Gaussian width and Lorentzian width, respectively')
    parser.add_argument('--optimal', action="store_true" , help = 'Enable optimal extraction')
    parser.add_argument('--slitcorr', action="store_true" , help = 'Apply slitloss correction based on profile width')
    parser.add_argument('--plot_ext', action="store_true" , help = 'Plot extracted spectrum')

    args = parser.parse_args(argv)

    if not args.filepath:
        print('When using arguments, you need to supply a filepath. Stopping execution')
        exit()

    if args.response_path:
        # Look for response function at file dir
        if os.path.isdir(args.response_path):
            for ii, kk in enumerate(glob.glob(args.response_path+"/M*.fits")):

                try:
                    filetype = fits.open(kk)[0].header["CDBFILE"]
                    arm = fits.open(args.filepath)[0].header["HIERARCH ESO SEQ ARM"]
                    if "GRSF" in filetype and arm in filetype:
                        response_file = kk
                except:
                    pass
            args.response_path = response_file

    if args.edge_mask:
        args.edge_mask = [ int(x) for x in args.edge_mask.split(",")]

    if args.pol_degree:
        args.pol_degree = [ int(x) for x in args.pol_degree.split(",")]


    print("Running extraction on file: " + args.filepath)
    print("with options: ")
    print("optimal = " + str(args.optimal))
    print("slitcorr = " + str(args.slitcorr))
    print("plot_ext = " + str(args.plot_ext))
    print("")

    run_extraction(args)


if __name__ == '__main__':
    # If script is run from editor or without arguments, run using this:
    if len(sys.argv) == 1:
        """
        Central scipt to extract spectra from X-shooter for the X-shooter GRB sample.
        """
        data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/"
        object_name = data_dir + "GRB160804A/"
        # object_name = "/Users/jselsing/Work/etc/GB_IDL_XSH_test/Q0157/J_red/"
        arm = "UVB" # UVB, VIS, NIR
        # Construct filepath
        file_path = object_name+arm+"_combined.fits"

        # Load in file
        files = glob.glob(file_path)

        # Look for response function at file dir

        for ii, kk in enumerate(glob.glob(object_name+"data_with_raw_calibs/M*.fits")):
            try:
                filetype = fits.open(kk)[0].header["CDBFILE"]

                # print(filetype, kk)
                if "GRSF" in filetype and arm in filetype:
                    # print(filetype, kk, fits.open(kk)[0].header)
                    response_file = kk
            except:
                pass
        # exit()
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.filepath = files[0]
        # args.response_path = response_file
        args.response_path = None
        args.seeing = 0.8
        args.optimal = True
        args.slitcorr = True
        args.plot_ext = True
        args.edge_mask = (15, 15)
        args.pol_degree = [4, 1, 1]
        print('Running extraction')
        run_extraction(args)

    else:
        main(argv = sys.argv[1:])









