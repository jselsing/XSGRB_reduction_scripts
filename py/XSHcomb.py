#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# Importing manupulation packages
from astropy.io import fits
import numpy as np
import glob
from numpy.polynomial import chebyshev
from scipy import ndimage
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve

# Import parser
import sys
import argparse
import os

# Plotting
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')
import copy

from util import *


class XSHcomb:
    """
    Class to contain XSH spectrscopy combinations.
    """

    def __init__(self, list_of_files, base_name, sky, synth_sky, sky2d=None):
        """
        Instantiate fitsfiles. Input list of file-names to be combined.
        """

        if len(list_of_files) == 0:
            raise ValueError("Input file list empty")

        self.list_of_files = list_of_files
        self.list_of_skyfiles = sky

        fitsfile, header = {}, {}
        flux, error, bpmap = {}, {}, {}
        for ii, kk in enumerate(self.list_of_files):
            fitsfile[ii] = fits.open(kk)
            header[ii] = fitsfile[ii][0].header
            flux[ii] = fitsfile[ii][0].data
            error[ii] = fitsfile[ii][1].data
            bpmap[ii] = fitsfile[ii][2].data
            if sky2d is not None:
                flux[ii] += sky2d[ii]

        em_sky = []
        for ii, kk in enumerate(self.list_of_skyfiles):
            # fitsfile[ii] = fits.open(kk)
            em_sky.append(fits.open(kk)[0].data)

        self.fitsfile = fitsfile
        self.header = header
        self.flux = flux
        self.error = error
        self.bpmap = bpmap

        # Construcs WCS
        self.haxis = convert_air_to_vacuum(10.*((np.arange(self.header[0]['NAXIS1']) - self.header[0]['CRPIX1'])*self.header[0]['CD1_1']+self.header[0]['CRVAL1']))
        self.vaxis = (np.arange(self.header[0]['NAXIS2']) - self.header[0]['CRPIX2'])*self.header[0]['CD2_2']+self.header[0]['CRVAL2']

        self.em_sky = em_sky
        self.base_name = base_name
        self.synth_sky = synth_sky

    def combine_imgs(self, NOD=False, same=False, repeats=1):
        """
        Combines X-shooter images.

        Function to inverse weighting combine multiple exposures from X-shooter. Tailored for combination of STARE-mode reduced images in NOD-sequence ABBA.

        Args:
            arg1 (bool): True to form nodding pairs before combination.

        Returns:
            fitsfile: fitsfile containing the combined flux, error and bad-pixel maps in consequtive extensions.

        """
        print("Combining to files "+str(self.list_of_files)+" to file: "+self.base_name+".fits....")
        img_nr = len(self.fitsfile)
        img_nr_list = np.arange(img_nr)

        pix_offsetx, pix_offsety = np.ones_like(img_nr_list), np.ones_like(img_nr_list)
        naxis1, naxis2 = np.ones_like(img_nr_list), np.ones_like(img_nr_list)
        full_edge_mask = self.bpmap.copy()
        for ii, kk in enumerate(self.fitsfile):
            # Arrays to contain axis indices
            naxis1[ii] = self.header[ii]['NAXIS1']
            naxis2[ii] = self.header[ii]['NAXIS2']

        for ii, kk in enumerate(self.fitsfile):
            try:
                pix_offsetx[ii] = int(round(self.header[ii]['HIERARCH ESO SEQ CUMOFF X'] / self.header[ii]['CD1_1']))
                pix_offsety[ii] = int(round(self.header[ii]['HIERARCH ESO SEQ CUMOFF Y'] / self.header[ii]['CD2_2']))
            except KeyError:
                print("No header keyword: HIERARCH ESO SEQ CUMOFF X or HIERARCH ESO SEQ CUMOFF Y")
                pix_offsetx[ii] = 0
                pix_offsety[ii] = 0
            if same:
                pix_offsetx[ii] = (max(naxis1) - naxis1[ii])/2
                pix_offsety[ii] = (max(naxis2) - naxis2[ii])/2

            # Pixel numbers in x- and y-direction
            xs = np.arange(naxis1[ii]) + 1
            ys = np.arange(naxis2[ii]) + 1

            # Masking 1 pixel edges in frames.
            edge_len = 3
            if NOD:
                edge_len = 0
            edge_mask = (ys > max(ys) - edge_len) | (ys < min(ys) + edge_len)
            full_edge_mask[ii] = np.tile(edge_mask , (len(xs), 1)).T

        pix_offsetxmax = abs(max(pix_offsetx) - min(pix_offsetx))
        pix_offsetymax = abs(max(pix_offsety) - min(pix_offsety))
        # Defining size of out-array
        v_size = max(naxis1) + pix_offsetxmax
        h_size = max(naxis2) + pix_offsetymax

        # Data storage
        flux_cube = np.ma.zeros((h_size, v_size, img_nr))
        error_cube = np.ma.zeros((h_size, v_size, img_nr))
        bpmap_cube = np.ma.zeros((h_size, v_size, img_nr))

        # Manually mask bad region in VIS arm
        if self.header[ii]['HIERARCH ESO SEQ ARM'] == "VIS":
            for ii, kk in enumerate(img_nr_list):
                for xx, pp in enumerate(np.arange(11220, 11340, 1)):
                    self.bpmap[ii][int(round(26 - 0.2 * xx)):int(round(33 - 0.2 * xx)), pp] = 543

        for ii, kk in enumerate(img_nr_list):

            self.bpmap[ii] += full_edge_mask[ii].astype("bool")*100

            # Defining positional offset between the frames.
            pos_v, pos_h = pix_offsety[kk], pix_offsetx[kk]  # offset

            # Finding the indices of the container in which to put image.
            offv = pix_offsety[kk] - min(pix_offsety)
            offh = pix_offsetx[kk] - min(pix_offsetx)

            # Define slices where to put image
            v_range1 = slice(offv, naxis2[ii] + offv)
            h_range1 = slice(offh, naxis1[ii] + offh)

            # b1 is full-size container with all values masked and b2 is input image with edge-mask + bad pixel mask.
            b1 = np.array(np.zeros((h_size, v_size)))
            b2 = np.array(self.flux[ii])

            # Insert smaller (b3, input image) frame into larger frame (container)
            b1[v_range1, h_range1] = b2

            # Append to list containing flux images
            flux_cube[:, :, ii] = b1

            # Repeat for error extension
            b3 = np.array(np.zeros((h_size, v_size)))
            b4 = np.array(self.error[ii])
            b3[v_range1, h_range1] = b4
            error_cube[:, :, ii] = b3

            # Repeat for bad pixel map
            b5 = np.array(np.zeros((h_size, v_size)))
            # b6 = np.array(self.bpmap[ii])
            # Grow bap pixel regions
            b6 = np.rint(convolve(np.array(self.bpmap[ii]), Gaussian2DKernel(0.2)))
            b5[v_range1, h_range1] = b6
            bpmap_cube[:, :, ii] = b5

        if NOD:
            if not repeats == 1:
                # Smaller container
                flux_cube_tmp = np.ma.zeros((h_size, v_size, np.ceil(img_nr / repeats)))
                error_cube_tmp = np.ma.zeros((h_size, v_size, np.ceil(img_nr / repeats)))
                bpmap_cube_tmp = np.ma.zeros((h_size, v_size, np.ceil(img_nr / repeats)))
                # Collapse in repeats
                for ii, kk in enumerate(np.arange(repeats)):
                    # Make lower an upper index of files, which is averaged over. If all NOD positions does not have the same number og repeats, assume the last position is cut.
                    low, up = ii*repeats, min(img_nr, (ii+1)*repeats)
                    # Slice structure
                    subset = slice(low, up)
                    # Average over subset
                    flux_cube_tmp[:, :, ii], error_cube_tmp[:, :, ii] = weighted_avg(flux_cube[:, :, subset], error_cube[:, :, subset], axis=2)
                    # Sum corresponding bpmap
                    bpmap_cube_tmp[:, :, ii] = np.sum(bpmap_cube[:, :, subset], axis=2)
                # Update number holders
                img_nr_list = np.arange(img_nr/repeats)
                pix_offsety = pix_offsety[::repeats]
                flux_cube, error_cube, bpmap_cube = flux_cube_tmp, error_cube_tmp, bpmap_cube_tmp

            # Form the pairs [(A1-B1) - shifted(B1-A1)] and [(B2-A2) - shifted(A2-B2)] at positions 0, 2. Sets the other images to np.nan.
            flux_cube, error_cube, bpmap_cube, self.em_sky = form_nodding_pairs(flux_cube, error_cube,  bpmap_cube, max(naxis2), pix_offsety)
            # Calibrate wavlength solution
            XSHcomb.finetune_wavlength_solution(self)
            self.sky_mask = np.tile(np.tile(self.sky_mask, (h_size, 1)).astype("int").T, (np.ceil(img_nr / repeats), 1, 1)).T
            bpmap_cube += self.sky_mask

        # Mask 3-sigma outliers in the direction of the stack
        clip_mask = np.zeros_like(flux_cube).astype("bool")
        low, high = np.ma.mean(flux_cube, axis=2) - 3*np.ma.std(flux_cube, axis=2), np.ma.mean(flux_cube, axis=2) + 3*np.ma.std(flux_cube, axis=2)
        for ii, kk in enumerate(img_nr_list):
            clip_mask[:, :, ii]  = (flux_cube[:, :, ii] < low) | (flux_cube[:, :, ii] > high)
        bpmap_cube[clip_mask] += 666


        # Update mask based on the bad-pixel map, the edge mask and the sigma-clipped mask
        mask_cube = (bpmap_cube.data != 0)

        # Update the mask for the data-cubes
        flux_cube.mask, error_cube.mask = mask_cube, mask_cube

        # Calculate weighted average and variance
        w_avg, ew_avg  = weighted_avg(flux_cube, error_cube, axis=2)
        w_avg.data[w_avg.mask] = np.nan
        ew_avg.data[ew_avg.mask] = np.nan

        # Sum over bad pixels where nan and from original mask
        self.bpmap = np.isnan(w_avg.data).astype("int") + (w_avg.mask).astype("int")

        # Assign new flux and error
        self.flux = w_avg.data
        self.error = ew_avg.data

        if same:
            self.flux[np.isnan(self.flux)] = np.median(self.flux[~np.isnan(self.flux)])
            self.error[np.isnan(self.error)] = 10. * max(self.error[~np.isnan(self.error)])

        # Write to file
        wrf = np.where(pix_offsety == min(pix_offsety))[0][0]

        self.fitsfile = self.fitsfile[wrf]
        self.header = self.header[wrf]

        self.fitsfile[0].data, self.fitsfile[1].data = self.flux, self.error
        self.fitsfile[2].data = self.bpmap

        self.header["CRVAL2"] = self.header["CRVAL2"] - (max(pix_offsety - min(pix_offsety)))  * self.header["CD2_2"]

        if not same:
            # Simply combined image
            if not NOD:
                # Updating file header
                self.fitsfile.header = self.header
                # Updating extention header keywords
                self.fitsfile[1].header["CRVAL2"], self.fitsfile[2].header["CRVAL2"] = self.fitsfile[0].header["CRVAL2"], self.fitsfile[0].header["CRVAL2"]
                self.fitsfile.writeto(self.base_name+".fits", clobber =True)
            # If nodded
            elif NOD:
                self.header["WAVECORR"] = self.correction_factor
                self.fitsfile.header = self.header
                self.fitsfile[1].header["CRVAL2"], self.fitsfile[2].header["CRVAL2"] = self.fitsfile[0].header["CRVAL2"], self.fitsfile[0].header["CRVAL2"]
                self.fitsfile.writeto(self.base_name+"skysub.fits", clobber =True)
        elif same:
            self.fitsfile.header = self.header
            self.fitsfile[1].header["CRVAL2"], self.fitsfile[2].header["CRVAL2"] = self.fitsfile[0].header["CRVAL2"], self.fitsfile[0].header["CRVAL2"]
            self.fitsfile.writeto(self.base_name[:-3]+"_combined.fits", clobber =True)

        # Update WCS
        self.haxis = convert_air_to_vacuum(10.*((np.arange(self.header['NAXIS1']) - self.header['CRPIX1'])*self.header['CD1_1']+self.header['CRVAL1']))
        self.vaxis = (np.arange(self.header['NAXIS2']) - self.header['CRPIX2'])*self.header['CD2_2']+self.header['CRVAL2']
        print("Combined")

    def sky_subtract(self, seeing, additional_masks, sky_check=False):

        """Sky-subtracts X-shooter images.

        Function to subtract sky off a rectifed x-shooter image. Fits a low-order polynomial in the spatial direction at each pixel and subtracts this off the same pixel. Assumes the object is centrered and masks the trace according to the seeing. Additionally accepts a list of arcsecond offsets with potential additional objects in the slit.

        fitsfile : fitsfile
            Input image to be sky-subtracted
        seeing : float
            Seeing of observation used to mask trace. Uses this as the width from the center which is masked. Must be wide enouch to completely mask any signal.
        additional_masks : list
            List of floats to additionally mask. This list contains the offset in arcsec relative to the center of the slit in which additional sources appear in the slit and which must be masked for sky-subtraction purposes. Positive values is defined relative to the WCS.

        Returns
        -------
        sky-subtracted image : fitsfile

        Notes
        -----
        na
        """
        print('Subtracting sky....')
        # Make trace mask
        seeing_pix = seeing / self.header['CD2_2']
        trace_offsets = np.append(np.array([0]), np.array(additional_masks) / self.header['CD2_2'])
        traces = []
        for ii in trace_offsets:
            traces.append(self.header['NAXIS2']/2 + ii)

        # Masking pixels in frame.
        trace_mask = np.zeros(self.header['NAXIS2']).astype("bool")
        for ii, kk in enumerate(traces):
            trace_mask[int(kk - seeing_pix):int(kk + seeing_pix)] = True
        full_trace_mask = np.tile(trace_mask , (self.header['NAXIS1'], 1)).T
        full_mask = self.bpmap.astype("bool") | full_trace_mask

        sky_background = np.zeros_like(self.flux)
        sky_background_error = np.zeros_like(self.flux)
        for ii, kk in enumerate(self.haxis):
            # Pick mask slice
            mask = full_mask[:, ii]

            # Sigma clip before sky-estimate to remove noisy pixels with bad error estimate.
            clip_mask = (self.flux[:, ii] < np.median(self.flux[:, ii][~mask]) - np.std(self.flux[:, ii][~mask])) | (self.flux[:, ii] > np.median(self.flux[:, ii][~mask]) + np.std(self.flux[:, ii][~mask]))

            # Combine masks
            mask = mask | clip_mask

            # Subtract simple median sky
            # self.flux[:, ii] -= np.ma.median(self.flux[:, ii][~mask])

            # Subtract polynomial estiamte of sky
            vals = self.flux[:, ii][~mask]
            errs = self.error[:, ii][~mask]

            try:
                chebfit = chebyshev.chebfit(self.vaxis[~mask], vals, deg = 1, w=1/errs)
                chebfitval = chebyshev.chebval(self.vaxis, chebfit)
                # chebfitval[chebfitval <= 0] = 0
            except TypeError:
                print("Empty array for sky-estimate at index "+str(ii)+". Sky estimate replaced with zeroes.")
                chebfitval = np.zeros_like(self.vaxis)
            except:
                print("Polynomial fit did not converge at index "+str(ii)+". Sky estimate replaced with median value.")
                chebfitval = np.ones_like(self.vaxis)*np.ma.median(self.vaxis[~mask])
            if ii % int(self.header['NAXIS1']/5) == 0 and sky_check and ii != 0:
                # Plotting for quality control
                pl.errorbar(self.vaxis[~mask], vals, yerr=errs, fmt=".k", capsize=0, elinewidth=0.5, ms=3)
                pl.plot(self.vaxis, chebfitval)
                pl.xlabel("Spatial index")
                pl.ylabel("Flux density")
                pl.title("Quality test: Sky estimate at index: "+str(ii) )
                # pl.savefig("Sky_estimate.pdf")
                pl.show()

            # self.flux[:, ii] -= chebfitval
            # self.error[:, ii] = (self.error[:, ii] + np.tile(np.std(vals - chebfitval[~mask]),  (1, self.header['NAXIS2'])))/2
            sky_background[:, ii] = chebfitval
            sky_background_error[:, ii] = np.tile(np.std(vals - chebfitval[~mask]),  (1, self.header['NAXIS2']))

        # Subtract sky and average error
        self.flux = self.flux - convolve(sky_background, Gaussian2DKernel(1.0))
        self.error = np.sqrt(self.error**2. + convolve(sky_background_error, Gaussian2DKernel(1.0))**2.)/2.

        self.em_sky = np.sum(self.em_sky, axis=0)
        # Calibrate wavlength solution
        XSHcomb.finetune_wavlength_solution(self)
        self.sky_mask = np.tile(self.sky_mask, (self.header["NAXIS2"], 1)).astype("int")
        self.bpmap += self.sky_mask
        self.flux[self.bpmap.astype("bool")] = np.nan
        self.header["WAVECORR"] = self.correction_factor
        self.fitsfile.header = self.header
        # self.fitsfile[1].header["CRVAL2"], self.fitsfile[2].header["CRVAL2"] = self.fitsfile[0].header["CRVAL2"], self.fitsfile[0].header["CRVAL2"]
        self.fitsfile[0].data, self.fitsfile[1].data = self.flux, self.error
        self.fitsfile[2].data = self.bpmap
        self.fitsfile.writeto(self.base_name+"skysub.fits", clobber =True)

        print('Writing sky subtracted image to '+self.base_name+"skysub.fits")

    def finetune_wavlength_solution(self):
        sky_model = fits.open(self.synth_sky[0])
        wl_sky = 10.*(sky_model[1].data.field('lam'))
        flux_sky = (sky_model[1].data.field('flux'))
        from scipy.interpolate import interp1d
        f = interp1d(wl_sky, flux_sky, bounds_error=False, fill_value=np.nan)
        synth_sky = f(self.haxis)

        # Cross correlate with redshifted spectrum and find velocity offset
        correlation = []
        offsets = np.arange(-0.0005, 0.0005, 0.00001)
        for ii in offsets:
            synth_sky = f(self.haxis*(1+ii))
            mask = ~np.isnan(synth_sky)
            corr = np.correlate(self.em_sky[mask]/np.median(self.em_sky[mask]), synth_sky[mask]/np.median(synth_sky[mask]))
            correlation.append(corr)

        # Smooth cross-correlation
        correlation = list(convolve(correlation, Gaussian2DKernel(stddev=10)))
        # Index with maximal value
        max_index = correlation.index(max(correlation))

        # Mask flux with extreme sky brightness
        self.sky_mask = convolve(f(self.haxis*(1+offsets[max_index])), Gaussian1DKernel(stddev=3)) > 250000

        pl.errorbar(offsets[max_index]*3e5, max(correlation), fmt=".k", capsize=0, elinewidth=0.5, ms=13, label="Found offset:" + str(offsets[max_index]*3e5) +" km/s")
        pl.plot(offsets*3e5, correlation)
        pl.xlabel("Offset velocity / [km/s]")
        pl.ylabel("Cross correlation")
        pl.title("Quality test: Wavelength calibration")
        pl.legend(loc=2)
        pl.savefig(self.base_name+"Crosscorrelated_Sky.pdf")
        self.correction_factor = 1. + offsets[max_index]

def run_combination(args):
    # Load in files
    sky2d = None
    response_2d = 1
    if args.mode == "STARE" or args.mode == "NODSTARE":
        files = glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SCI_SLIT_MERGE2D_*.fits")
        sky_files = glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SKY_SLIT_MERGE1D_*.fits")
        n_flux_files = len(glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SCI_SLIT_FLUX_MERGE2D_*.fits"))
        if n_flux_files == 0 and not args.use_master_response:
            print("Option to use master response function has not been set and no flux calibrated data exists. You should probably set the optional argument, --use_master_response.")
            print("Press \"enter\" to continue anyway and use the non-flux calibrated images.")
            raw_input()
        if not args.use_master_response and n_flux_files != 0:
            response_2d = [fits.open(ii)[0].data for ii in files]
            files = glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SCI_SLIT_FLUX_MERGE2D_*.fits")
            response_2d = [fits.open(kk)[0].data/response_2d[ii] for ii, kk in enumerate(files)]
        if args.mode == "NODSTARE":
            sky2d = glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SKY_SLIT_MERGE2D_*.fits")
            sky2d = np.array([fits.open(ii)[0].data for ii in sky2d]) * np.array(response_2d)
    elif args.mode == "COMBINE":
        files = glob.glob(args.filepath+args.arm+"*skysub.fits")
        sky_files = glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SKY_SLIT_MERGE1D_*.fits")

    skyfile = glob.glob("data/static_sky/"+args.arm+"skytable.fits")

    img = XSHcomb(files, args.filepath+args.arm+args.OB, sky=sky_files, synth_sky=skyfile, sky2d=sky2d)
    # Combine nodding observed pairs.
    if args.mode == "STARE":
        img.combine_imgs(NOD=False)
        img.sky_subtract(seeing=args.seeing, additional_masks=args.additional_masks, sky_check=False)
    elif args.mode == "NODSTARE":
        img.combine_imgs(NOD=True, repeats=args.repeats)
    elif args.mode == "COMBINE":
        img.combine_imgs(same=True)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str, default="/Users/jselsing/Work/work_rawDATA/XSGRB/GRB120327A/", help='Path to burst directory on which to run combination. Directory must contain a centain directory structure, similar to /Users/jselsing/Work/work_rawDATA/XSGRB/GRB121027A/reduced_data/OB1/UVB/XSHOO.2012-10-30T05:03:26.098cosmicced/ToO_GRBtrigger_4x600_SCI_SLIT_FLUX_MERGE2D_UVB.fits.')
    parser.add_argument('arm', type=str, default="UVB", help='X-shooter arm to combine. Used to find files')
    parser.add_argument('mode', type=str, default="STARE", help='MODE in which to run combinations. Can either be STARE, NODSTARE or COMBINE')
    parser.add_argument('OB', type=str, default="OB1", help='OB number. Used to look for files.')
    parser.add_argument('-repeats', type=int, default=1, help='Number of times nodding position has been repeated')
    parser.add_argument('-seeing', type=float, default=1.0, help='Estimated seeing. Used to mask trace for sky-subtraction.')
    parser.add_argument('-additional_masks', type=list, default=list(), help='List of offsets relative to center of additional masks for sky-subtraction.')
    parser.add_argument('--use_master_response', action="store_true" , help = 'Set this optional keyword if input files are not flux-calibrated. Used in sky-subtraction.')

    args = parser.parse_args(argv)

    if not args.filepath:
        print('When using arguments, you need to supply a filepath. Stopping execution')
        exit()


    print("Running combination on files: " + args.filepath)
    # print("with options: ")
    # # print("bin_elements = " + str(args.bin_elements))
    # print("")

    run_combination(args)


if __name__ == '__main__':
    # If script is run from editor or without arguments, run using this:
    if len(sys.argv) == 1:
        """
        Central scipt to combine images from X-shooter for the X-shooter GRB sample.
        """

        # Load in files
        parser = argparse.ArgumentParser()
        args = parser.parse_args()

        data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/"
        object_name = data_dir + "GRB120119A/"
        args.filepath = object_name

        args.arm = "NIR" # UVB, VIS, NIR

        args.mode = "NODSTARE" # STARE, NODSTARE, COMBINE

        args.OB = "OB4"

        args.use_master_response = False # True False

        args.additional_masks = []
        args.seeing = 1.0
        args.repeats = 1

        run_combination(args)

    else:
        main(argv = sys.argv[1:])
