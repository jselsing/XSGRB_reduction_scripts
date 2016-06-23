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

# Plotting
import matplotlib.pyplot as pl
import seaborn; seaborn.set_style('ticks')
import copy

from util import *


def form_nodding_pairs(flux_cube, error_cube,  bpmap_cube, naxis2, pix_offsety):

    flux_cube_out = np.ma.zeros(flux_cube.shape)
    error_cube_out = np.ma.zeros(error_cube.shape)
    bpmap_cube_out = np.ma.ones(bpmap_cube.shape)*3
    em_sky = np.ma.sum(np.ma.median(flux_cube, axis = 2), axis=0)

    # Finding the indices of the container in which to put image.
    offv1 = pix_offsety[0] - min(pix_offsety)
    offv2 = pix_offsety[1] - min(pix_offsety)
    try:
        offv3 = pix_offsety[2] - min(pix_offsety)
        offv4 = pix_offsety[3] - min(pix_offsety)
    except:
        pass

    # Define slices where to put image
    v_range1 = slice(offv1, naxis2 + offv1)
    v_range2 = slice(offv2, naxis2 + offv2)
    try:
        v_range3 = slice(offv3, naxis2 + offv3)
        v_range4 = slice(offv4, naxis2 + offv4)
    except:
        pass

    # Make mask based on the bad-pixel map, the edge mask and the sigma-clipped mask
    mask_cube = (bpmap_cube.data != 0)

    # Replacing masked values with zeroes. This will make then disappear in addition and subtraciton.
    flux_cube[mask_cube] = 0
    error_cube[mask_cube] = 0
    bpmap_cube[mask_cube] = 1

    # From A-B and B-A pairs
    flux_cube_out[v_range1, :, 0] = flux_cube[v_range1, :, 0] - flux_cube[v_range2, :, 1]
    flux_cube_out[v_range2, :, 1] = flux_cube[v_range2, :, 1] - flux_cube[v_range1, :, 0]
    try:
        flux_cube_out[v_range3, :, 2] = flux_cube[v_range3, :, 2] - flux_cube[v_range4, :, 3]
        flux_cube_out[v_range4, :, 3] = flux_cube[v_range4, :, 3] - flux_cube[v_range3, :, 2]
    except:
        pass

    # Subtract residiual sky due to varying sky-brightness over obserations
    flux_cube_out[v_range1, :, 0] -= np.ma.median(flux_cube_out[v_range1, :, 0], axis=0)
    flux_cube_out[v_range2, :, 1] -= np.ma.median(flux_cube_out[v_range2, :, 1], axis=0)
    try:
        flux_cube_out[v_range3, :, 2] -= np.ma.median(flux_cube_out[v_range3, :, 2], axis=0)
        flux_cube_out[v_range4, :, 3] -= np.ma.median(flux_cube_out[v_range4, :, 3], axis=0)
    except:
        pass

    # # From A-B and B-A error pairs
    error_cube_out[v_range1, :, 0] = np.sqrt(error_cube[v_range1, :, 0]**2. + error_cube[v_range2, :, 1]**2.)
    error_cube_out[v_range2, :, 1] = np.sqrt(error_cube[v_range2, :, 1]**2. + error_cube[v_range1, :, 0]**2.)
    try:
        error_cube_out[v_range3, :, 2] = np.sqrt(error_cube[v_range3, :, 2]**2. + error_cube[v_range4, :, 3]**2.)
        error_cube_out[v_range4, :, 3] = np.sqrt(error_cube[v_range4, :, 3]**2. + error_cube[v_range3, :, 2]**2.)
    except:
        pass

    # From A-B and B-A  bpmap pairs
    bpmap_cube_out[v_range1, :, 0] = bpmap_cube[v_range1, :, 0] + bpmap_cube[v_range2, :, 1]
    bpmap_cube_out[v_range2, :, 1] = bpmap_cube[v_range2, :, 1] + bpmap_cube[v_range1, :, 0]
    try:
        bpmap_cube_out[v_range3, :, 2] = bpmap_cube[v_range3, :, 2] + bpmap_cube[v_range4, :, 3]
        bpmap_cube_out[v_range4, :, 3] = bpmap_cube[v_range4, :, 3] + bpmap_cube[v_range3, :, 2]
    except:
        pass

    # Form A-B - shifted(B-A) pairs
    flux_cube_out[:, :, 0] = 0.5*(flux_cube_out[:, :, 0] + flux_cube_out[:, :, 1])
    flux_cube_out[:, :, 1] = np.nan
    try:
        flux_cube_out[:, :, 2] = 0.5*(flux_cube_out[:, :, 2] + flux_cube_out[:, :, 3])
        flux_cube_out[:, :, 3] = np.nan
    except:
        pass

    error_cube_out[:, :, 0] = 0.5*np.sqrt(error_cube_out[:, :, 0]**2. + error_cube_out[:, :, 1]**2.)
    error_cube_out[:, :, 1] = np.nan
    try:
        error_cube_out[:, :, 2] = 0.5*np.sqrt(error_cube_out[:, :, 2]**2. + error_cube_out[:, :, 3]**2.)
        error_cube_out[:, :, 3] = np.nan
    except:
        pass

    bpmap_cube_out[:, :, 0] = bpmap_cube_out[:, :, 0] + bpmap_cube_out[:, :, 1]
    bpmap_cube_out[:, :, 1] = np.ones_like(bpmap_cube_out[:, :, 0])
    try:
        bpmap_cube_out[:, :, 2] = bpmap_cube_out[:, :, 2] + bpmap_cube_out[:, :, 3]
        bpmap_cube_out[:, :, 3] = np.ones_like(bpmap_cube_out[:, :, 0])
    except:
        pass

    good_mask = (bpmap_cube_out == 0) | (bpmap_cube_out == 2) | (bpmap_cube_out == 3)
    bpmap_cube_out[good_mask] = 0

    return flux_cube_out, error_cube_out, bpmap_cube_out, em_sky


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
            if sky2d:
                flux[ii] += fits.open(sky2d[ii])[0].data

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
        print('Combining to file: '+self.base_name+'.fits....')
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
                    # print(self.bpmap[ii].shape)
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
            b6 = np.array(self.bpmap[ii])
            b5[v_range1, h_range1] = b6
            bpmap_cube[:, :, ii] = b5

        if NOD:
            if not repeats == 1:
                # Smaller container
                flux_cube_tmp = np.ma.zeros((h_size, v_size, img_nr / repeats))
                error_cube_tmp = np.ma.zeros((h_size, v_size, img_nr / repeats))
                bpmap_cube_tmp = np.ma.zeros((h_size, v_size, img_nr / repeats))
                # Collapse in repeats
                for ii, kk in enumerate(np.arange(repeats)):
                    subset = slice(ii*repeats, (ii+1)*repeats)
                    flux_cube_tmp[:, :, ii], error_cube_tmp[:, :, ii] = weighted_avg(flux_cube[:, :, subset], error_cube[:, :, subset], axis=2)
                    bpmap_cube_tmp[:, :, ii] = np.sum(bpmap_cube[:, :, subset], axis=2)
                # Update number holders
                img_nr_list = np.arange(img_nr/repeats)
                pix_offsety = pix_offsety[::repeats]
                flux_cube, error_cube, bpmap_cube = flux_cube_tmp, error_cube_tmp, bpmap_cube_tmp

            # Form the pairs [(A1-B1) - shifted(B1-A1)] and [(B2-A2) - shifted(A2-B2)] at positions 0, 2. Sets the other images to np.nan.
            flux_cube, error_cube, bpmap_cube, self.em_sky = form_nodding_pairs(flux_cube, error_cube,  bpmap_cube, max(naxis2), pix_offsety)
            # self.haxis = 10.*((np.arange(self.header[0]['NAXIS1']) - self.header[0]['CRPIX1'])*self.header[0]['CD1_1']+self.header[0]['CRVAL1'])
            # Calibrate wavlength solution
            XSHcomb.finetune_wavlength_solution(self)
            self.header['CD1_1'] *= 1+self.correction_factor
            self.header['CDELT1'] *= 1+self.correction_factor
            self.header['CRVAL1'] *= 1+self.correction_factor
            self.header["WAVECORR"] = self.correction_factor


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

        # if not same:
        #     laplace = ndimage.laplace(w_avg.data)
        #     print(np.mean(abs(laplace[~np.isnan(laplace)])), np.std(abs(laplace[~np.isnan(laplace)])))

        #     self.bpmap[abs(laplace) > np.median(abs(laplace[~np.isnan(laplace)])) + np.std(abs(laplace[~np.isnan(laplace)]))] = 66

        #     # Grow bad pixel regions
        #     self.bpmap = convolve(self.bpmap, Gaussian2DKernel(0.15))
        #     w_avg.data[self.bpmap.astype("bool")] = np.nan
        #     ew_avg.data[self.bpmap.astype("bool")] = np.nan

        self.flux = w_avg.data
        self.error = ew_avg.data

        if same:
            self.flux[np.isnan(self.flux)] = np.median(self.flux[~np.isnan(self.flux)])
            self.error[np.isnan(self.error)] *= 1e4

        # Write to file
        wrf = np.where(pix_offsety == min(pix_offsety))[0][0]

        self.fitsfile = self.fitsfile[wrf]
        self.header = self.header[wrf]

        self.fitsfile[0].data, self.fitsfile[1].data = self.flux, self.error
        self.fitsfile[2].data = self.bpmap
        self.header["CRVAL2"] = self.header["CRVAL2"] - (max(pix_offsety - min(pix_offsety)))  * self.header["CD2_2"]

        if not same:
            if not NOD:
                # Updating file header
                self.fitsfile.header = self.header
                # Updating extention header keywords
                self.fitsfile[1].header["CRVAL2"], self.fitsfile[2].header["CRVAL2"] = self.fitsfile[0].header["CRVAL2"], self.fitsfile[0].header["CRVAL2"]
                self.fitsfile[1].header["CD1_1"], self.fitsfile[2].header["CD1_1"] = self.fitsfile[0].header["CD1_1"], self.fitsfile[0].header["CD1_1"]
                self.fitsfile[1].header["CDELT1"], self.fitsfile[2].header["CDELT1"] = self.fitsfile[0].header["CDELT1"], self.fitsfile[0].header["CDELT1"]
                self.fitsfile[1].header["CRVAL1"], self.fitsfile[2].header["CRVAL1"] = self.fitsfile[0].header["CRVAL1"], self.fitsfile[0].header["CRVAL1"]
                self.fitsfile.writeto(self.base_name+".fits", clobber =True)
            elif NOD:
                self.fitsfile.header = self.header
                self.fitsfile[1].header["CRVAL2"], self.fitsfile[2].header["CRVAL2"] = self.fitsfile[0].header["CRVAL2"], self.fitsfile[0].header["CRVAL2"]
                self.fitsfile[1].header["CD1_1"], self.fitsfile[2].header["CD1_1"] = self.fitsfile[0].header["CD1_1"], self.fitsfile[0].header["CD1_1"]
                self.fitsfile[1].header["CDELT1"], self.fitsfile[2].header["CDELT1"] = self.fitsfile[0].header["CDELT1"], self.fitsfile[0].header["CDELT1"]
                self.fitsfile[1].header["CRVAL1"], self.fitsfile[2].header["CRVAL1"] = self.fitsfile[0].header["CRVAL1"], self.fitsfile[0].header["CRVAL1"]
                self.fitsfile.writeto(self.base_name+"skysub.fits", clobber =True)
        elif same:
            self.fitsfile.header = self.header
            self.fitsfile[1].header["CRVAL2"], self.fitsfile[2].header["CRVAL2"] = self.fitsfile[0].header["CRVAL2"], self.fitsfile[0].header["CRVAL2"]
            self.fitsfile[1].header["CD1_1"], self.fitsfile[2].header["CD1_1"] = self.fitsfile[0].header["CD1_1"], self.fitsfile[0].header["CD1_1"]
            self.fitsfile[1].header["CDELT1"], self.fitsfile[2].header["CDELT1"] = self.fitsfile[0].header["CDELT1"], self.fitsfile[0].header["CDELT1"]
            self.fitsfile[1].header["CRVAL1"], self.fitsfile[2].header["CRVAL1"] = self.fitsfile[0].header["CRVAL1"], self.fitsfile[0].header["CRVAL1"]
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

                chebfit = chebyshev.chebfit(self.vaxis[~mask], vals, deg = 2, w=1/errs)
                chebfitval = chebyshev.chebval(self.vaxis, chebfit)
                # chebfitval[chebfitval <= 0] = 0
            except TypeError:
                print("Empty array for sky-estimate at index "+str(ii)+". Sky estimate replaced with zeroes.")
                chebfitval = np.zeros_like(self.vaxis)
            except:
                print("Polynomial fit did not converge at index "+str(ii)+". Sky estimate replaced with median value.")
                chebfitval = np.ones_like(self.vaxis)*np.ma.median(self.vaxis[~mask])
            if ii % int(self.header['NAXIS1']/10) == 0 and sky_check and ii != 0:
                # Plotting for quality control
                pl.errorbar(self.vaxis[~mask], vals, yerr=errs, fmt=".k", capsize=0, elinewidth=0.5, ms=3)
                pl.plot(self.vaxis, chebfitval)
                pl.xlabel("Spatial index")
                pl.ylabel("Flux density")
                pl.title("Quality test: Sky estimate at index: "+str(ii) )
                pl.savefig("Sky_estimate.pdf")
                pl.show()

            self.flux[:, ii] -= chebfitval
            # self.error[:, ii] = np.tile(np.std(vals - chebfitval[~mask]),  (1, self.header['NAXIS2']))

        self.em_sky = np.sum(self.em_sky, axis=0)
        # Calibrate wavlength solution
        XSHcomb.finetune_wavlength_solution(self)
        self.fitsfile[0].header['CD1_1'] *= 1+self.correction_factor
        self.fitsfile[0].header['CDELT1'] *= 1+self.correction_factor
        self.fitsfile[0].header['CRVAL1'] *= 1+self.correction_factor
        self.header["WAVECORR"] = self.correction_factor
        self.fitsfile.header = self.header
        self.fitsfile[1].header["CRVAL2"], self.fitsfile[2].header["CRVAL2"] = self.fitsfile[0].header["CRVAL2"], self.fitsfile[0].header["CRVAL2"]
        self.fitsfile[1].header["CD1_1"], self.fitsfile[2].header["CD1_1"] = self.fitsfile[0].header["CD1_1"], self.fitsfile[0].header["CD1_1"]
        self.fitsfile[1].header["CDELT1"], self.fitsfile[2].header["CDELT1"] = self.fitsfile[0].header["CDELT1"], self.fitsfile[0].header["CDELT1"]
        self.fitsfile[1].header["CRVAL1"], self.fitsfile[2].header["CRVAL1"] = self.fitsfile[0].header["CRVAL1"], self.fitsfile[0].header["CRVAL1"]
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


        # Fit polynomial for continuum
        # mask = ~np.isnan(synth_sky) | (synth_sky > 10000)
        # chebfit = chebyshev.chebfit(self.haxis[mask], synth_sky[mask], deg=2)
        # synth_sky -= chebyshev.chebval(self.haxis, chebfit)

        # pl.plot(self.haxis, synth_sky)
        # pl.plot(self.haxis, self.em_sky)
        # pl.show()

        correlation = []
        offsets = np.arange(-0.0005, 0.0005, 0.00001)
        for ii in offsets:
            synth_sky = f(self.haxis*(1+ii))
            mask = ~np.isnan(synth_sky)
            corr = np.correlate(self.em_sky[mask]/np.median(self.em_sky[mask]), synth_sky[mask]/np.median(synth_sky[mask]))
            correlation.append(corr)

        max_index = correlation.index(max(correlation))
        pl.errorbar(offsets[max_index]*3e5, max(correlation), fmt=".k", capsize=0, elinewidth=0.5, ms=13, label="Found offset:" + str(offsets[max_index]*3e5) +" km/s")
        pl.plot(offsets*3e5, correlation)
        pl.xlabel("Offset velocity / [km/s]")
        pl.ylabel("Cross correlation")
        pl.title("Quality test: Wavelength calibration")
        pl.legend(loc=2)
        pl.savefig(self.base_name+"Crosscorrelated_Sky.pdf")
        self.correction_factor = offsets[max_index]

def main():
    """
    Central scipt to combine images from X-shooter for the X-shooter GRB sample.
    """
    data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/"
    object_name = data_dir + "GRB100316D/"
    arm = "UVB" # UVB, VIS, NIR
    mode = "STARE" # STARE, NODSTARE, COMBINE
    OB = "OB1"

    # Load in files
    sky2d_files = None
    if mode == "STARE" or mode == "NODSTARE":
        files = glob.glob(object_name+"reduced_data/"+OB+"/"+arm+"/*/*SCI_SLIT_MERGE2D_*.fits")
        sky_files = glob.glob(object_name+"reduced_data/"+OB+"/"+arm+"/*/*SKY_SLIT_MERGE1D_*.fits")
        if mode == "NODSTARE":
            sky2d_files = glob.glob(object_name+"reduced_data/"+OB+"/"+arm+"/*/*SKY_SLIT_MERGE2D_*.fits")
    elif mode == "COMBINE":
        files = glob.glob(object_name+arm+"*skysub*.fits")
        sky_files = glob.glob(object_name+"reduced_data/"+OB+"/"+arm+"/*/*SKY_SLIT_MERGE1D_*.fits")

    skyfile = glob.glob(data_dir +"static_sky/"+arm+"skytable.fits")

    img = XSHcomb(files, object_name+arm+OB, sky=sky_files, synth_sky=skyfile, sky2d=sky2d_files)
    # Combine nodding observed pairs.
    if mode == "STARE":
        img.combine_imgs(NOD=False)
        img.sky_subtract(seeing=2.5, additional_masks=[], sky_check=False)
    elif mode == "NODSTARE":
        img.combine_imgs(NOD=True, repeats=1)
        # img.sky_subtract(seeing=1.0, additional_masks=[], sky_check=False)
    elif mode == "COMBINE":
        img.combine_imgs(same=True)


if __name__ == '__main__':
    main()