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
        # print("".join(input_file.split("/")[-1])[:-5])
        # exit()
        self.base_name = "/".join(input_file.split("/")[:-1]) + "/" + "".join(input_file.split("/")[-1])[:-5]

        if resp:
            # Apply flux calibration from master response file
            resp = fits.open(resp)
            self.wl_response, self.response = resp[1].data.field('LAMBDA'), resp[1].data.field('RESPONSE')

            f = interpolate.interp1d(10 * self.wl_response, self.response, bounds_error=False)
            self.response = f(10.*((np.arange(self.header['NAXIS1']) - self.header['CRPIX1'])*self.header['CD1_1']+self.header['CRVAL1'])/(self.header['WAVECORR']))

            if self.header['HIERARCH ESO SEQ ARM'] == "UVB" or self.header['HIERARCH ESO SEQ ARM'] == "VIS":
                gain = self.header["HIERARCH ESO DET OUT1 GAIN"]
            elif self.header['HIERARCH ESO SEQ ARM'] == "NIR":
                gain = 1.0/2.12
            else:
                print("Missing arm keyword in header. Stopping.")
                exit()

            # Apply atmospheric extinciton correction
            atmpath = "/opt/local/share/esopipes/datastatic/xshoo-2.8.3/xsh_paranal_extinct_model_"+self.header['HIERARCH ESO SEQ ARM'].lower()+".fits"
            ext_atm = fits.open(atmpath)
            self.wl_ext_atm, self.ext_atm = ext_atm[1].data.field('LAMBDA'), ext_atm[1].data.field('EXTINCTION')

            f = interpolate.interp1d(10. * self.wl_ext_atm, self.ext_atm, bounds_error=False)
            self.ext_atm = f(10.*(((np.arange(self.header['NAXIS1'])) - self.header['CRPIX1'])*self.header['CDELT1']+self.header['CRVAL1']) * self.header['WAVECORR'])
            self.response = (10. * self.header['CD1_1'] * self.response * (10.**(0.4*self.header['HIERARCH ESO TEL AIRM START'] * self.ext_atm))) / ( gain * self.header['EXPTIME'])

        # Get slit width
        if self.header['HIERARCH ESO SEQ ARM'] == "UVB":
            self.slit_width = float(self.header['HIERARCH ESO INS OPTI3 NAME'].split("x")[0])
        elif self.header['HIERARCH ESO SEQ ARM'] == "VIS":
            self.slit_width = float(self.header['HIERARCH ESO INS OPTI4 NAME'].split("x")[0])
        elif self.header['HIERARCH ESO SEQ ARM'] == "NIR":
            self.slit_width = float(self.header['HIERARCH ESO INS OPTI5 NAME'].split("x")[0])


    def get_trace_profile(self, lower_element_nr = 1, upper_element_nr = 1, pol_degree = [3, 2, 2], bin_elements=100):

        # Get binned spectrum
        bin_length = int(len(self.haxis) / bin_elements)
        bin_flux, bin_error = bin_image(self.flux, self.error, bin_length)
        bin_haxis = 10.*(((np.arange(self.header['NAXIS1']/bin_length)) - self.header['CRPIX1'])*self.header['CD1_1']*bin_length+self.header['CRVAL1'])
        width = int(len(self.vaxis)/5)

        # Save binned image for quality control
        self.fitsfile[0].data = bin_flux.data
        self.fitsfile[1].data = bin_error.data
        self.fitsfile[0].header["CD1_1"] = self.fitsfile[0].header["CD1_1"] * bin_length
        self.fitsfile.writeto(self.base_name+"_binned.fits", clobber=True)
        self.fitsfile[0].header["CD1_1"] = self.fitsfile[0].header["CD1_1"] / bin_length

        # Inital parameter guess
        p0 = [np.nanmean(bin_flux[bin_flux > 0]), np.median(self.vaxis), 0.3, 0.3, 0]
        # Corrections to slit position from broken ADC, taken DOI: 10.1086/131052
        # Pressure in hPa, Temperature in Celcius
        p, T = self.header['HIERARCH ESO TEL AMBI PRES END'], self.header['HIERARCH ESO TEL AMBI TEMP']
        # Convert hPa to mmHg
        p = p * 0.7501
        # Wavelength in microns
        wl_m = bin_haxis/1e4
        # Refractive index in dry air (n - 1)1e6
        eq_1 = 64.328 + (29498.1/(146 - wl_m**-2)) + (255.4/(41 - wl_m**-2))
        # Corrections for ambient temperature and pressure
        eq_2 = eq_1*((p*(1. + (1.049 - 0.0157*T)*1e-6*p)) / (720.883*(1. + 0.003661*T)))
        # Correction from water vapor. Water vapor obtained from the Antione equation, https://en.wikipedia.org/wiki/Antoine_equation
        eq_3 = eq_2 - ((0.0624 - 0.000680*wl_m**-2) / (1. + 0.003661*T)) * 10**(8.07131 - (1730.63/(233.426 + T)))
        # Isolate n
        n = eq_3 / 1e6 + 1
        # Angle relative to zenith
        z = np.arccos(1/self.header['HIERARCH ESO TEL AIRM START'])

        # Zero-deviation wavelength of arms, from http://www.eso.org/sci/facilities/paranal/instruments/xshooter/doc/VLT-MAN-ESO-14650-4942_v87.pdf
        if self.header['HIERARCH ESO SEQ ARM'] == "UVB":
            zdwl = 0.405
        elif self.header['HIERARCH ESO SEQ ARM'] == "VIS":
            zdwl = 0.633
        elif self.header['HIERARCH ESO SEQ ARM'] == "NIR":
            zdwl = 1.31
        else:
            raise ValueError("Input image does not contain header keyword 'HIERARCH ESO SEQ ARM'. Cannot determine ADC correction.")
        zdwl_inx = find_nearest(wl_m, zdwl)
        # Correction of position on slit, relative to Zero-deviation wavelength
        dR = (206265*(n - n[zdwl_inx])*np.tan(z))

        # Parameter containers
        amp, cen, sig, gam = np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis)
        eamp, ecen, esig, egam = np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis)

        # Loop though along dispersion axis in the binned image and fit a Voigt
        for ii, kk in enumerate(bin_haxis):
            try:
                # Edit trace position by analytic ADC-amount
                p0[1] = np.median(self.vaxis) + dR[ii]
                # Fit SPSF
                popt, pcov = optimize.curve_fit(voigt, self.vaxis[width:-width], bin_flux[:, ii][width:-width], p0 = p0, maxfev = 5000)
                # pl.errorbar(self.vaxis[width:-width], bin_flux[:, ii][width:-width], yerr=bin_error[:, ii][width:-width], fmt=".k", capsize=0, elinewidth=0.5, ms=3)
                # pl.plot(self.vaxis[width:-width], voigt(self.vaxis, *popt)[width:-width])
                # pl.title(ii)
                # pl.show()
            except:
                print("Fitting error at binned image index: "+str(ii)+". Replacing fit value with guess and set fit error to 10^10")
                popt, pcov = p0, np.diag(1e10*np.ones_like(p0))
            amp[ii], cen[ii], sig[ii], gam[ii] = popt[0], popt[1], popt[2], popt[3]
            eamp[ii], ecen[ii], esig[ii], egam[ii] = np.sqrt(np.diag(pcov)[0]), np.sqrt(np.diag(pcov)[1]), np.sqrt(np.diag(pcov)[2]), np.sqrt(np.diag(pcov)[3])

        # Mask elements too close to guess, indicating a bad fit.
        ecen[:lower_element_nr] = 1e10
        ecen[-upper_element_nr:] = 1e10


        ecen[abs(cen/ecen) > abs(np.nanmean(cen/ecen)) + 5*np.nanstd(cen/ecen)] = 1e10
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
        ax1.set_ylim((min(self.vaxis[width:-width]), max(self.vaxis[width:-width])))
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

        eamp[ecen == 1e10] = 1e10
        # eamp[esig == 1e10] = 1e10
        amp[amp < 0] = 1e-20
        amp = signal.medfilt(amp, 5)
        mask = ~(eamp == 1e10)
        f = interpolate.interp1d(bin_haxis[mask], amp[mask], bounds_error=False, fill_value="extrapolate")
        fitampval = f(self.haxis)
        fitampval[fitampval <= 0] = 1e-20#np.nanmean(fitampval[fitampval > 0])

        # Plotting for quality control
        ax4.errorbar(bin_haxis, amp, fmt=".k", capsize=0, elinewidth=0.5, ms=5)
        ax4.plot(self.haxis, fitampval)
        # ax4.set_ylim((0, 1))
        ax4.set_ylabel("Profile amplitude / [counts/s]")
        ax4.set_title("Quality test: Profile amplitude estimate")
        ax4.set_xlabel(r"Wavelength / [$\mathrm{\AA}$]")
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

    def extract_spectrum(self, extraction_bounds, optimal=None, slitcorr=None, edge_mask=None, pol_degree=None, bin_elements=None, plot_ext=None):

        """Optimally extracts a spectrum from sky-subtracted X-shooter image.

        Function to extract spectra from X-shooter images. Either sums the flux in a central aperture or uses a profile-weighted extraction.

        fitsfile : fitsfile
            Input sky-subtracted image with flux, error and bad-pixel map in extensions 0, 1, 2 respectively.
        extraction_bounds : tuple
            Tuple containing extraction bounds for the standard extraction.
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

        # Applying updated wavelength solution. (This also includes barycentric correction etc.)
        self.haxis = 10.*(((np.arange(self.header['NAXIS1'])) - self.header['CRPIX1'])*self.header['CDELT1']+self.header['CRVAL1']) * self.header['WAVECORR'] #* (1 + self.header['HIERARCH ESO QC VRAD BARYCOR']/3e5)**-1
        self.vaxis =  (((np.arange(self.header['NAXIS2'])) - self.header['CRPIX2'])*self.header['CD2_2']+self.header['CRVAL2'])

        # Finding extraction radius
        seeing = (extraction_bounds[1] - extraction_bounds[0])*self.header['CD2_2']

        # Construct spatial PSF to be used as weight in extraction
        if optimal:
            print("Fitting for the full spectral extraction profile")
            XSHextract.get_trace_profile(self, lower_element_nr = int(tuple(edge_mask)[0]), upper_element_nr = int(tuple(edge_mask)[1]), pol_degree=pol_degree, bin_elements=bin_elements)
            self.fitsfile[0].data = (self.flux - self.trace_model).data
            self.fitsfile[1].data = self.error.data
            self.fitsfile.writeto(self.base_name + "Profile_subtracted_image.fits", clobber=True)

        elif not optimal:
            print("Using simplified extraction. Extracting spectrum by summing aperture. Aperture width is: " + str(seeing) + " arcsec.")
            print("Extracting spectrum between pixel " +str(extraction_bounds[0])+ " and " +str(extraction_bounds[1]))

            # Calculating slit-loss based on specified seeing.
            if hasattr(self, 'slitcorr'):
                self.slitcorr = slit_loss(seeing/10., self.slit_width)

            # Defining extraction aperture
            ext_aper = slice(extraction_bounds[0], extraction_bounds[1])

        # Interpolate over bad pixel map
        self.flux.data[self.flux.mask] = np.nan
        self.error.data[self.flux.mask] = np.nanmax(self.error.data[~self.flux.mask])
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
        errorspectrum[1:][mask] = np.nanmax(errorspectrum)
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
            print("Estimated slitloss correction factor is:"+str(self.slitcorr))
            if type(self.slitcorr) == np.float64:
                self.slitcorr = np.ones_like(spectrum) * self.slitcorr
            # spectrum *= self.slitcorr
            # errorspectrum *= self.slitcorr
            dt.append(("slitcorr", np.float64))
            out_data.append(self.slitcorr)
            formatt.append('%10.6e')
            head = head + " slitloss_correction_factor"

        data = np.array(zip(*out_data), dtype=dt)
        np.savetxt(self.base_name + extname, data, header=head, fmt = formatt, delimiter="\t")

        if plot_ext:
            fig, ax = pl.subplots()

            ax.errorbar(self.haxis[::5], spectrum[::5], yerr=errorspectrum[::5], fmt=".k", capsize=0, elinewidth=0.5, ms=3, alpha=0.5)
            ax.plot(self.haxis[::5], spectrum[::5], lw = 0.2, linestyle="steps-mid", alpha=0.5, rasterized=True)
            m = np.average(spectrum[~np.isnan(spectrum)], weights=1/errorspectrum[~np.isnan(spectrum)])
            s = np.nanstd(spectrum[abs(spectrum - m) < 3 * np.nanstd(spectrum) ][int(len(spectrum)/10):int(-len(spectrum)/10)])
            pl.xlim(min(self.haxis), max(self.haxis))
            pl.ylim(m - 10 * s, m + 10 * s)
            pl.xlabel(r"Wavelength / [$\mathrm{\AA}$]")
            pl.ylabel(r'Flux density [erg s$^{-1}$ cm$^{-1}$ $\AA^{-1}$]')
            pl.savefig(self.base_name + "Extraction"+str(extname.split(".")[0])+".pdf")
            # pl.show()

        return self.haxis, spectrum, errorspectrum


def run_extraction(args):

    print("Running extraction on file: " + args.filepath)
    print("with options:")
    print("optimal = " + str(args.optimal))
    print("slitcorr = " + str(args.slitcorr))
    print("plot_ext = " + str(args.plot_ext))
    print("use_master_response = " + str(args.use_master_response))
    print("")

    # Look for response function at file dir
    if not args.response_path and args.use_master_response:
        print("--use_master_reponse is set, but no -response_path is. I will try to guess where the master reponse file is located.")
        for ii, kk in enumerate(glob.glob("/".join(args.filepath.split("/")[:-1])+"/data_with_raw_calibs/M*.fits")):
            try:
                filetype = fits.open(kk)[0].header["CDBFILE"]
                arm = fits.open(args.filepath)[0].header["HIERARCH ESO SEQ ARM"]
                if "GRSF" in filetype and arm in filetype:
                    args.response_path = kk
            except:
                pass
        if args.response_path:
            print("Found master response at: "+str(args.response_path))
        elif not args.response_path:
            print("None found. Skipping flux calibration.")
    if args.response_path and args.use_master_response:
        # Look for response function at file dir
        if os.path.isdir(args.response_path):
            print("Path to response file supplied. Looking for response function.")
            for ii, kk in enumerate(glob.glob(args.response_path+"/M*.fits")):
                try:
                    filetype = fits.open(kk)[0].header["CDBFILE"]
                    arm = fits.open(args.filepath)[0].header["HIERARCH ESO SEQ ARM"]
                    if "GRSF" in filetype and arm in filetype:
                        args.response_path = kk
                except:
                    pass
            # args.response_path = response_file
            if not os.path.isdir(args.response_path):
                print("Found master response at: "+str(args.response_path))
            elif os.path.isdir(args.response_path):
                print("None found. Skipping flux calibration.")
                args.response_path = None
            # args.response_path = response_file
    if not args.use_master_response:
        args.response_path = None

    spec = XSHextract(args.filepath, resp = args.response_path)
    # Optimal extraction
    wl, flux, error = spec.extract_spectrum(extraction_bounds=args.extraction_bounds, optimal=args.optimal, slitcorr=args.slitcorr, edge_mask=args.edge_mask, pol_degree=args.pol_degree, bin_elements=args.bin_elements, plot_ext=args.plot_ext)


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str, help='Path to file on which to run extraction')
    parser.add_argument('-response_path', type=str, default=None, help='Response function to apply. Can either be a path to file or path to directory containing file. If directory, will look for correct file.')
    parser.add_argument('-extraction_bounds', type=str, default="30, 60", help='Bounds in which to do the standard extraction. Must be indices over which to do the extraction. Example -extraction_bounds 30,60')
    parser.add_argument('-edge_mask', type=str, default="1, 1", help='Tuple containing the edge masks. (10,10) means that 10 pixels are masked at each edge.')
    parser.add_argument('-pol_degree', type=str, default="3,2,2", help='List containing the edge masks. Each number specify the degree of the polynomial used for the fit in central prosition, Gaussian width and Lorentzian width, respectively. Must be specified as 3,2,2 without the backets.')
    parser.add_argument('-bin_elements', type=int, default=100, help='Integer specifying the number of elements to bin down to for tracing. A higher value will allow for a more precise tracing, but is only suitable for very high S/N objects')
    parser.add_argument('--use_master_response', action="store_true" , help = 'Set this optional keyword if input file is not flux-calibrated. The master response function is applied to the extracted spectrum.')
    parser.add_argument('--optimal', action="store_true" , help = 'Enable optimal extraction')
    parser.add_argument('--slitcorr', action="store_true" , help = 'Apply slitloss correction based on profile width')
    parser.add_argument('--plot_ext', action="store_true" , help = 'Plot extracted spectrum')

    args = parser.parse_args(argv)

    if not args.filepath:
        print('When using arguments, you need to supply a filepath. Stopping execution')
        exit()

    if args.edge_mask:
        args.edge_mask = [int(x) for x in args.edge_mask.split(",")]

    if args.extraction_bounds:
        args.extraction_bounds = [int(x) for x in args.extraction_bounds.split(",")]

    if args.pol_degree:
        args.pol_degree = [int(x) for x in args.pol_degree.split(",")]

    run_extraction(args)


if __name__ == '__main__':
    # If script is run from editor or without arguments, run using this:
    if len(sys.argv) == 1:
        """
        Central scipt to extract spectra from X-shooter for the X-shooter GRB sample.
        """
        data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/"
        object_name = data_dir + "GRB100724A*/"

        arm = "UVB" # UVB, VIS, NIR
        OB = "OB4"
        # Construct filepath
        file_path = object_name+arm+OB+"skysub.fits"
        # file_path = object_name+arm+"_combined.fits"

        # Load in file
        files = glob.glob(file_path)

        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.filepath = files[0]
        args.response_path = None
        args.use_master_response = True

        args.optimal = True # True, False
        args.extraction_bounds = (17, 50) # UVB, VIS = (40, 60), NIR (30, 50)
        args.slitcorr = True
        args.plot_ext = True
        args.edge_mask = (1, 1)
        args.pol_degree = [3, 2, 2]
        args.bin_elements = 100
        run_extraction(args)

    else:
        main(argv = sys.argv[1:])









