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
import astroscrappy
# Import parser
import sys
import argparse
import os

# Plotting
import matplotlib; matplotlib.use('TkAgg')
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
        seeing = {}
        for ii, kk in enumerate(self.list_of_files):
            fitsfile[ii] = fits.open(kk)
            header[ii] = fitsfile[ii][0].header
            flux[ii] = fitsfile[ii][0].data


            error[ii] = fitsfile[ii][1].data
            bpmap[ii] = fitsfile[ii][2].data
            seeing[ii] = np.mean([header[ii]["HIERARCH ESO TEL AMBI FWHM START"], header[ii]["HIERARCH ESO TEL AMBI FWHM END"]])
            if sky2d is not None:
                flux[ii] += sky2d[ii]


        self.FWHM = np.median(list(seeing.values()))

        em_sky = []
        for ii, kk in enumerate(self.list_of_skyfiles):
            # fitsfile[ii] = fits.open(kk)
            em_sky.append(np.median(fits.open(kk)[0].data, axis = 0))

        self.fitsfile = fitsfile
        self.header = header
        self.flux = flux
        self.error = error
        self.bpmap = bpmap

        # Constructs WCS
        self.haxis = convert_air_to_vacuum(10.*(((np.arange(self.header[0]['NAXIS1'])) + 1 - self.header[0]['CRPIX1'])*self.header[0]['CDELT1']+self.header[0]['CRVAL1']))
        self.vaxis = (np.arange(self.header[0]['NAXIS2']) - self.header[0]['CRPIX2'])*self.header[0]['CDELT2']+self.header[0]['CRVAL2']

        if len(em_sky) == 0:
            print("No sky-frame given ... Using science image collapsed in the spatial direction ...")
            try:
                em_sky = np.sum(np.array(flux.values()), axis = 1)
            except:
                em_sky = None
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
        print("Combining "+str(len(self.list_of_files))+" files,\n"+str(self.list_of_files)+"\nto file:\n"+self.base_name+".fits....")
        img_nr = len(self.fitsfile)
        img_nr_list = np.arange(img_nr)

        pix_offsetx, pix_offsety = np.ones_like(img_nr_list), np.ones_like(img_nr_list)
        naxis1, naxis2 = np.ones_like(img_nr_list), np.ones_like(img_nr_list)
        exptimes = np.ones_like(img_nr_list)
        ra, dec = [0]*len(img_nr_list), [0]*len(img_nr_list)
        full_edge_mask = self.bpmap.copy()
        for ii, kk in enumerate(self.fitsfile):
            # Arrays to contain axis indices
            naxis1[ii] = self.header[ii]['NAXIS1']
            naxis2[ii] = self.header[ii]['NAXIS2']
            exptimes[ii] = self.header[ii]['EXPTIME']
            ra[ii], dec[ii] = float(self.header[ii]['RA']), float(self.header[ii]['DEC'])
        ref_ra, ref_dec = ra[0], dec[0]

        for ii, kk in enumerate(self.fitsfile):
            try:
                pix_offsetx[ii] = int(round(self.header[ii]['HIERARCH ESO SEQ CUMOFF X'] / self.header[ii]['CDELT1']))
                pix_offsety[ii] = int(round(self.header[ii]['HIERARCH ESO SEQ CUMOFF Y'] / self.header[ii]['CDELT2']))
            except KeyError:
                try:
                    # Offset mode along slit for older observations
                    from astropy.coordinates import SkyCoord, SkyOffsetFrame
                    import astropy.units as u
                    point_ra = self.header[ii]['RA']*u.deg
                    point_dec = self.header[ii]['DEC']*u.deg
                    offset_ra = self.header[ii]['HIERARCH ESO SEQ CUMOFF RA']*u.arcsec
                    offset_dec = self.header[ii]['HIERARCH ESO SEQ CUMOFF DEC']*u.arcsec
                    center = SkyCoord(ra=point_ra, dec=point_dec, frame = self.header[ii]['RADECSYS'].lower())
                    off_ra = point_ra + offset_ra
                    off_dec = point_dec + offset_dec
                    other = SkyCoord(ra=off_ra, dec=off_dec, frame = self.header[ii]['RADECSYS'].lower())
                    offset = center.separation(other).arcsecond
                    # Assume offset is along slit axis
                    pix_offsetx[ii] = int(round(0 / self.header[ii]['CDELT1']))
                    pix_offsety[ii] = int(round(offset / self.header[ii]['CDELT2']))

                except KeyError:
                    print("No header keyword: HIERARCH ESO SEQ CUMOFF X or HIERARCH ESO SEQ CUMOFF Y")
                    pix_offsetx[ii] = 0
                    pix_offsety[ii] = 0

            if same:

                # Wavelength step in velocity
                midwl = (max(self.haxis) - min(self.haxis))/2
                dv = 3e5*10*self.header[ii]['CDELT1']/midwl
                pix_offsetx[ii] = int(round((self.header[ii]['HIERARCH ESO QC VRAD BARYCOR']  + (self.header[ii]['WAVECORR']-1)*3e5)  / dv))
                # # Assume object is centered
                pix_offsety[ii] = int(round((max(naxis2)/2 - naxis2[ii]/2)))

            # Pixel numbers in x- and y-direction
            xs = np.arange(naxis1[ii]) + 1
            ys = np.arange(naxis2[ii]) + 1

            # Masking 1 pixel edges in frames.
            edge_len = 2
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
        flux_cube = np.zeros((h_size, v_size, img_nr))
        error_cube = np.zeros((h_size, v_size, img_nr))
        bpmap_cube = np.ones((h_size, v_size, img_nr))

        # Manually mask bad region in VIS arm
        if self.header[ii]['HIERARCH ESO SEQ ARM'] == "VIS":
            for ii, kk in enumerate(img_nr_list):
                for xx, pp in enumerate(np.arange(11220, 11340, 1)):
                    self.bpmap[ii][int(round(26 - 0.2 * xx)):int(round(33 - 0.2 * xx)), pp] = 543


        for ii, kk in enumerate(img_nr_list):

            self.bpmap[ii] = self.bpmap[ii] + full_edge_mask[ii].astype("bool")*100

            # Defining positional offset between the frames.
            pos_v, pos_h = pix_offsety[kk], pix_offsetx[kk]  # offset

            # Finding the indices of the container in which to put image.
            offv = pix_offsety[kk] - min(pix_offsety)
            offh = pix_offsetx[kk] - min(pix_offsetx)

            # Define slices where to put image
            v_range1 = slice(offv, naxis2[ii] + offv)
            h_range1 = slice(offh, naxis1[ii] + offh)

            # b1 is full-size container with all values masked and b2 is input image with edge-mask + bad pixel mask.
            b1 = np.zeros((h_size, v_size))
            b2 = self.flux[ii]

            # Insert smaller (b3, input image) frame into larger frame (container)
            b1[v_range1, h_range1] = b2


            # Append to list containing flux images
            flux_cube[:, :, ii] = b1


            # Repeat for error extension
            b3 = np.zeros((h_size, v_size))
            b4 = self.error[ii]
            b3[v_range1, h_range1] = b4
            error_cube[:, :, ii] = b3

            # Repeat for bad pixel map
            b5 = np.ones((h_size, v_size))
            b6 = self.bpmap[ii]
            # Grow bap pixel regions !! Deprecated after update to pipeline version 2.8.3
            # b6 = np.rint(convolve(np.array(self.bpmap[ii]), Gaussian2DKernel(0.3)))
            b5[v_range1, h_range1] = b6
            bpmap_cube[:, :, ii] = b5

        # Mask 3-sigma outliers in the direction of the stack
        m, s = np.ma.median(np.ma.array(flux_cube, mask=bpmap_cube), axis = 2).data,  np.std(np.ma.array(flux_cube, mask=bpmap_cube), axis = 2).data
        if self.header[ii]['HIERARCH ESO SEQ ARM'] == "NIR":
            sigma_mask = 3
        else:
            sigma_mask = 3
        l, h = np.tile((m - sigma_mask*s).T, (img_nr, 1, 1)).T, np.tile((m + sigma_mask*s).T, (img_nr, 1, 1)).T
        bpmap_cube[(flux_cube < l) | (flux_cube > h)] = 666

        # Form nodding pairs
        if NOD:
            if not repeats == 1:
                # Smaller container
                flux_cube_tmp = np.zeros((h_size, v_size, int(np.ceil(img_nr / repeats))))
                error_cube_tmp = np.zeros((h_size, v_size, int(np.ceil(img_nr / repeats))))
                bpmap_cube_tmp = np.zeros((h_size, v_size, int(np.ceil(img_nr / repeats))))
                # Collapse in repeats
                for ii, kk in enumerate(np.arange(int(np.ceil(img_nr / repeats)))):
                    # Make lower an upper index of files, which is averaged over. If all NOD positions does not have the same number of repeats, assume the last position is cut.
                    low, up = ii*repeats, min(img_nr, (ii+1)*repeats)
                    # Slice structure
                    subset = slice(low, up)
                    # Average over subset
                    flux_cube_tmp[:, :, ii], error_cube_tmp[:, :, ii], bpmap_cube_tmp[:, :, ii] = avg(flux_cube[:, :, subset], error_cube[:, :, subset], bpmap_cube[:, :, subset].astype("bool"), axis=2)
                # Update number holders
                img_nr_list = np.arange(img_nr/repeats)
                pix_offsety = pix_offsety[::repeats]
                flux_cube, error_cube, bpmap_cube = flux_cube_tmp, error_cube_tmp, bpmap_cube_tmp

            # Form the pairs [(A1-B1) - shifted(B1-A1)] and [(B2-A2) - shifted(A2-B2)] at positions 0, 2. Sets the other images to np.nan.
            flux_cube, error_cube, bpmap_cube = form_nodding_pairs(flux_cube, error_cube,  bpmap_cube, max(naxis2), pix_offsety)

        # Mask outliers
        bpmap_cube[(flux_cube == 0) | (flux_cube == 1) | (flux_cube == np.inf) | (flux_cube == np.nan)] = 666
        bpmap_cube[(error_cube == 0) | (error_cube == 1) | (error_cube == np.inf) | (error_cube == np.nan)] = 666

        # Boolean mask based on the bad-pixel map, the edge mask and the sigma-clipped mask
        mask_cube = (bpmap_cube != 0)

        # Make weight map based on background variance in boxcar window
        shp = error_cube.shape
        weight_cube = np.zeros_like(error_cube)
        for ii in range(shp[2]):
            run_var = np.ones(shp[1])
            for kk in np.arange(shp[1]):
                err_bin = 1/(error_cube[:, kk-4:kk+4, ii][~(bpmap_cube[:, kk-4:kk+4, ii].astype("bool"))])**2
                if len(err_bin) != 0:
                    run_var[kk] = np.median(err_bin.flatten())
            weight_cube[:, :, ii] = np.tile(run_var, (shp[0], 1))

        # Normlize weights
        weight_cube[mask_cube] = 0
        weight_cube = weight_cube/np.tile(np.sum(weight_cube, axis=2).T, (shp[2], 1, 1)).T

        # Calculate mean and error
        mean, error, bpmap = avg(flux_cube, error_cube, mask_cube, axis=2, weight_map = weight_cube)

        # Assign new flux and error
        mean[np.isnan(mean)] = 0
        self.flux = mean
        self.error = error
        # 3-sigma percentiles
        mi, ma = np.percentile(self.flux.flatten() , (0.1349898032, 99.8650101968))
        outlier_map = ((self.flux < mi) | (self.flux > ma)).astype("int")
        outlier_map = 1000 * np.rint(convolve(outlier_map, Gaussian2DKernel(0.3)))

        self.bpmap = bpmap + outlier_map

        if same:
            self.flux[np.isnan(self.flux)] = np.median(self.flux[~np.isnan(self.flux)])
            self.error[np.isnan(self.error)] = 10. * max(self.error[~np.isnan(self.error)])

        # Write to file
        wrf = np.where(pix_offsety == min(pix_offsety))[0][0]

        self.fitsfile = self.fitsfile[wrf]
        self.header = self.header[wrf]
        self.fitsfile[0].data = self.flux
        self.fitsfile[1].data = self.error

        # Mask outliers
        # l, m, h = np.percentile(self.flux[np.isfinite(self.flux)].flatten(), (16, 50, 84))
        # l_s, h_s = m - l, h - m
        # mask = (self.flux < 100*l_s) & (self.flux > 100*h_s)
        # self.bpmap[mask] = 555

        self.fitsfile[2].data = self.bpmap

        # Update WCS
        self.header["CRVAL2"] = self.header["CRVAL2"] - (max(pix_offsety - min(pix_offsety)))  * self.header["CDELT2"]

        # Set header keyword based on median seeing
        self.header["SEEING"] = self.FWHM

        if not same:
            # Simply combined image
            if not NOD:
                # Updating file header
                self.fitsfile.header = self.header
                # Updating extention header keywords
                self.fitsfile[1].header["CRVAL2"], self.fitsfile[2].header["CRVAL2"] = self.fitsfile[0].header["CRVAL2"], self.fitsfile[0].header["CRVAL2"]
                self.fitsfile.writeto(self.base_name+".fits", overwrite =True)
            # If nodded
            elif NOD:
                # self.header["WAVECORR"] = self.correction_factor
                self.fitsfile.header = self.header
                self.fitsfile[1].header["CRVAL2"], self.fitsfile[2].header["CRVAL2"] = self.fitsfile[0].header["CRVAL2"], self.fitsfile[0].header["CRVAL2"]
                self.fitsfile.writeto(self.base_name+"skysub.fits", overwrite =True)
        elif same:
            self.fitsfile.header = self.header
            self.fitsfile[1].header["CRVAL2"], self.fitsfile[2].header["CRVAL2"] = self.fitsfile[0].header["CRVAL2"], self.fitsfile[0].header["CRVAL2"]
            self.base_name = self.base_name[:-3]+"_combined"
            self.fitsfile.writeto(self.base_name+".fits", overwrite =True)

        # Update WCS axis
        self.vaxis = (np.arange(self.header['NAXIS2']) - self.header['CRPIX2'])*self.header['CDELT2']+self.header['CRVAL2']

        mask = (self.flux > -1e-17) & (self.flux < 1e-17)
        hs, ed = np.histogram(self.flux[mask], bins=1000000)
        print("")
        print("Mode of flux values (should be close to zero):")
        print(ed[find_nearest(hs, max(hs))], ed[find_nearest(hs, max(hs))+ 1])
        print("")
        print("Finished combining files ...")

        # Get binned spectrum
        bin_length = int(len(self.haxis) / 300)
        bin_flux, bin_error = bin_image(self.flux, self.error, self.bpmap, bin_length, weight = True)

        # Save binned image for quality control
        self.fitsfile[0].data = bin_flux
        self.fitsfile[1].data = bin_error
        self.fitsfile[0].header["CD1_1"] = self.fitsfile[0].header["CD1_1"] * bin_length
        self.fitsfile.writeto(self.base_name+"_binned.fits", overwrite=True)
        self.fitsfile[0].data = self.flux
        self.fitsfile[1].data = self.error
        self.fitsfile[0].header["CD1_1"] = self.fitsfile[0].header["CD1_1"] / bin_length


    def sky_subtract(self, seeing, masks, sky_check=False, nod=False):

        """Sky-subtracts X-shooter images.

        Function to subtract sky off a rectifed x-shooter image. Fits a low-order polynomial in the spatial direction at each pixel and subtracts this off the same pixel. Assumes the object is centrered and masks the trace according to the seeing. Additionally accepts a list of arcsecond offsets with potential additional objects in the slit.

        fitsfile : fitsfile
            Input image to be sky-subtracted
        seeing : float
            Seeing of observation used to mask trace. Uses this as the width from the center which is masked. Must be wide enouch to completely mask any signal.
        masks : list
            List of floats to additionally mask. This list contains the offset in arcsec relative to the center of the slit in which additional sources appear in the slit and which must be masked for sky-subtraction purposes. Positive values is defined relative to the WCS.

        Returns
        -------
        sky-subtracted image : fitsfile

        Notes
        -----
        na
        """
        if not nod:
            print("")
            print('Subtracting sky....')
            # Make trace mask
            seeing_pix = seeing / self.header['CDELT2']
            trace_offsets = np.array(masks) / self.header['CDELT2']
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
                m, s = np.nanmean(self.flux[:, ii]), np.nanstd(self.flux[:, ii])
                clip_mask = (self.flux[:, ii] < m - s) | (self.flux[:, ii] > m + s)

                # Combine masks
                mask = mask | clip_mask

                # Subtract polynomial estiamte of sky
                vals = self.flux[:, ii][~mask]
                errs = self.error[:, ii][~mask]

                try:
                    chebfit = chebyshev.chebfit(self.vaxis[~mask], vals, deg = 2, w=1/errs)
                    chebfitval = chebyshev.chebval(self.vaxis, chebfit)
                    chebfitval[chebfitval <= 0] = 0
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
                    pl.show()

                sky_background[:, ii] = chebfitval

            # Subtract sky
            self.flux = self.flux - convolve(sky_background, Gaussian2DKernel(1.0))


        # 3-sigma percentiles
        mi, ma = np.percentile(self.flux.flatten() , (0.1349898032, 99.8650101968))
        # mi, ma = np.percentile(self.flux.flatten() , (0.02275013195, 0.9772498681))
        print(mi, ma)
        outlier_map = ((self.flux < mi) | (self.flux > ma)).astype("int")
        outlier_map = 1000 * np.rint(convolve(outlier_map, Gaussian2DKernel(0.3)))

        self.em_sky = np.sum(self.em_sky, axis=0)
        # Calibrate wavlength solution
        XSHcomb.finetune_wavlength_solution(self)
        self.sky_mask = np.tile(self.sky_mask, (self.header["NAXIS2"], 1)).astype("int")

        self.bpmap += self.sky_mask + outlier_map
        self.flux[self.bpmap.astype("bool")] = 0
        self.fitsfile.header = self.header
        self.fitsfile[0].data, self.fitsfile[1].data = self.flux, self.error
        self.fitsfile[2].data = self.bpmap
        self.fitsfile.writeto(self.base_name+"skysub.fits", overwrite =True)

        print('Writing sky subtracted image to '+self.base_name+"skysub.fits")

    def finetune_wavlength_solution(self):
        print("")
        print("Cross correlating with synthetic sky to obtain refinement to wavlength solution ...")
        print("")

        # Remove continuum
        mask = ~np.isnan(self.em_sky) & (self.haxis < 18000) & (self.haxis > 3500)
        hist, edges = np.histogram(self.em_sky[mask], bins="auto")
        max_idx = find_nearest(hist, max(hist))
        sky = self.em_sky - edges[max_idx]
        mask = ~np.isnan(sky) & (sky > 0) & (self.haxis < 18000) & (self.haxis > 3500)

        # Load synthetic sky
        sky_model = fits.open("data/static_sky/skytable_hres.fits")
        wl_sky = 1e4*(sky_model[1].data.field('lam')) # In micron
        flux_sky = sky_model[1].data.field('flux')

        # Convolve to observed grid
        from scipy.interpolate import interp1d
        f = interp1d(wl_sky, convolve(sky_model[1].data.field('flux'), Gaussian1DKernel(stddev=10)), bounds_error=False, fill_value=np.nan)
        synth_sky = f(self.haxis)

        # Cross correlate with redshifted spectrum and find velocity offset
        offsets = np.arange(-0.0005, 0.0005, 0.00001)
        correlation = np.zeros(offsets.shape)
        for ii, kk in enumerate(offsets):
            synth_sky = f(self.haxis * (1. + kk))
            correlation[ii] = np.correlate(sky[mask]*(np.nanmax(synth_sky)/np.nanmax(sky)), synth_sky[mask])

        # Index with maximal value
        max_idx = find_nearest(correlation, max(correlation))
        # Corrections to apply to original spectrum, which maximizes correlation.
        self.correction_factor = 1. + offsets[max_idx]
        print("Found preliminary velocity offset: "+str((self.correction_factor - 1.)*3e5)+" km/s")
        print("")
        print("Minimising residuals between observed sky and convolved synthetic sky to obtain the sky PSF ...")
        print("")

        # Zero-deviation wavelength of arms, from http://www.eso.org/sci/facilities/paranal/instruments/xshooter/doc/VLT-MAN-ESO-14650-4942_v87.pdf
        if self.header['HIERARCH ESO SEQ ARM'] == "UVB":
            zdwl = 4050
            pixel_width = 50
        elif self.header['HIERARCH ESO SEQ ARM'] == "VIS":
            zdwl = 6330
            pixel_width = 50
        elif self.header['HIERARCH ESO SEQ ARM'] == "NIR":
            zdwl = 13100
            pixel_width = 50

        # Get seeing PSF by minimizing the residuals with a theoretical sky model, convovled with an increasing psf.
        psf_width = np.arange(1, pixel_width, 1)
        res = np.zeros(psf_width.shape)
        for ii, kk in enumerate(psf_width):
            # Convolve sythetic sky with Gaussian psf
            convolution = convolve(flux_sky, Gaussian1DKernel(stddev=kk))
            # Interpolate high-res syntheric sky onto observed wavelength grid.
            f = interp1d(wl_sky, convolution, bounds_error=False, fill_value=np.nan)
            synth_sky = f(self.haxis*self.correction_factor)
            # Calculate squared residuals
            residual = np.nansum((synth_sky[mask]*(np.nanmax(sky[mask])/np.nanmax(synth_sky[mask])) - sky[mask])**2.)
            res[ii] = residual

        # Index of minimal residual
        min_idx = find_nearest(res, min(res))

        # Wavelegth step corresponding psf width in FWHM
        R, seeing = np.zeros(psf_width.shape), np.zeros(psf_width.shape)
        for ii, kk in enumerate(psf_width):
            dlambda = np.diff(wl_sky[::kk])*2*np.sqrt(2*np.log(2))
            # Interpolate to wavelegth grid
            f = interp1d(wl_sky[::kk][:-1], dlambda, bounds_error=False, fill_value=np.nan)
            dlambda = f(self.haxis)

            # PSF FWHM in pixels
            d_pix = dlambda/(10*self.header['CDELT1'])
            # Corresponding seeing PSF FWHM in arcsec
            spatial_psf = d_pix*self.header['CDELT2']

            # Index of zero-deviation
            zd_idx = find_nearest(self.haxis, zdwl)

            # Resolution at zero-deviation wavelength
            R[ii] = (self.haxis/dlambda)[zd_idx]
            # Seeing at zero-deviation wavelength
            seeing[ii] = spatial_psf[zd_idx]

        # self.header['R'] = R[min_idx]
        # self.header['PSFFWHM'] = seeing[min_idx]

        fig, ax1 = pl.subplots()
        # ax1.errorbar(R[min_idx], min(res), fmt=".k", capsize=0, elinewidth=0.5, ms=13, label=r"R$_{sky}$ = " + str(int(R[min_idx])), color="#4682b4")
        # ax1.errorbar(seeing[min_idx], min(res), fmt=".k", capsize=0, elinewidth=0.5, ms=13, label="Seeing PSF FWHM = " + str(np.around(seeing[min_idx], decimals = 2)) + " arcsec", color="#4682b4")
        # ax1.plot(R, res, color="#4682b4")
        # ax1.set_ylabel("Residual", color="#4682b4")
        ax1.yaxis.set_major_formatter(pl.NullFormatter())

        # ax1.set_xlabel("Resolution", color="#4682b4")
        ax1.legend()

        convolution = convolve(flux_sky, Gaussian1DKernel(stddev=psf_width[min_idx]))
        f = interp1d(wl_sky, convolution, bounds_error=False, fill_value=np.nan)
        synth_sky = f(self.haxis)

        # Cross correlate with redshifted spectrum and find velocity offset
        offsets = np.arange(-0.0005, 0.0005, 0.000001)
        correlation = np.zeros_like(offsets)
        for ii, kk in enumerate(offsets):
            synth_sky = f(self.haxis * (1. + kk))
            correlation[ii] = np.correlate(sky[mask]*(np.nanmax(synth_sky[mask])/np.nanmax(sky[mask])), synth_sky[mask])

        # Smooth cross-correlation
        correlation = convolve(correlation, Gaussian1DKernel(stddev=20))

        # Index with maximum correlation
        max_idx = find_nearest(correlation, max(correlation))
        self.correction_factor = (1. + offsets[max_idx])
        self.header["WAVECORR"] = self.correction_factor
        print("Found refined velocity offset: "+str((self.correction_factor - 1.)*3e5)+" km/s")
        print("")

        # Mask flux with > 3-sigma sky brightness
        self.sky_mask = f(self.haxis*self.correction_factor) > np.percentile(f(self.haxis*self.correction_factor), 99)
        ax2 = ax1.twiny()

        ax2.errorbar(offsets[max_idx]*3e5, max(correlation)*(max(res)/max(correlation)), fmt=".k", capsize=0, elinewidth=0.5, ms=13, label="Wavelength correction:" + str(np.around((self.correction_factor - 1.)*3e5, decimals = 1)) +" km/s", color="r")
        ax2.plot(offsets*3e5, correlation*(max(res)/max(correlation)), color="r")
        ax2.set_xlabel("Offset velocity / [km/s]", color="r")
        ax2.set_ylabel("Cross correlation", color="r")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_major_formatter(pl.NullFormatter())
        ax2.legend(loc=2)
        pl.savefig(self.base_name+"Wavelength_cal.pdf")
        pl.clf()

def run_combination(args):
    # Load in files
    sky2d = None
    response_2d = 1
    if args.mode == "STARE" or args.mode == "NODSTARE":
        files = glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SCI_SLIT_MERGE2D_*.fits")
        if len(files) == 0:
            print("No files found... Exitting..")
            exit()
        sky_files = glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SKY_SLIT_MERGE2D_*.fits")
        n_flux_files = len(glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SCI_SLIT_FLUX_MERGE2D_*.fits"))
        if n_flux_files == 0 and not args.use_master_response:
            print("Option to use master response function has not been set and no flux calibrated data exists. You should probably set the optional argument, --use_master_response.")
            print("Press \"enter\" to continue anyway and use the non-flux calibrated images.")
            raw_input()
        if not args.use_master_response and n_flux_files != 0:
            response_2d = [fits.open(ii)[0].data for ii in files]

            files = glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*MANMERGE_*.fits")


            if len(files) == 0:
                files = glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SCI_SLIT_FLUX_MERGE2D_*.fits")

            response_2d = [np.tile(np.nanmedian(fits.open(kk)[0].data/response_2d[ii], axis=0), (np.shape(response_2d[ii])[0], 1)) for ii, kk in enumerate(files)]

            np.savetxt(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/response_function.dat", np.nanmean(np.nanmean(response_2d, axis=1), axis=0))

        if args.mode == "NODSTARE":
            sky2d = glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SKY_SLIT_MERGE2D_*.fits")
            sky2d = np.array([fits.open(ii)[0].data for ii in sky2d]) * np.array(response_2d)
    elif args.mode == "COMBINE":

        files = glob.glob(args.filepath+args.arm+"*skysub.fits")
        if len(files) == 0:
            print("No files found... Exitting..")
            exit()
        sky_files = glob.glob(args.filepath+"reduced_data/"+args.OB+"/"+args.arm+"/*/*SKY_SLIT_MERGE2D_*.fits")

    skyfile = glob.glob("data/static_sky/"+args.arm+"skytable.fits")

    img = XSHcomb(files, args.filepath+args.arm+args.OB, sky=sky_files, synth_sky=skyfile, sky2d=sky2d)
    # Combine nodding observed pairs.
    if args.mode == "STARE":
        img.combine_imgs(NOD=False)
        img.sky_subtract(seeing=args.seeing, masks=args.masks, sky_check=False)
    elif args.mode == "NODSTARE":
        img.combine_imgs(NOD=True, repeats=args.repeats)
        img.sky_subtract(seeing=args.seeing, masks=args.masks, sky_check=False, nod=True)
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
    parser.add_argument('-masks', type=list, default=list(), help='List of offsets relative to center of additional masks for sky-subtraction.')
    parser.add_argument('--use_master_response', action="store_true" , help = 'Set this optional keyword if input files are not flux-calibrated. Used in sky-subtraction.')

    args = parser.parse_args(argv)

    if not args.filepath:
        print('When using arguments, you need to supply a filepath. Stopping execution')
        exit()

    print("Running combination on files: " + args.filepath)

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
        # data_dir = "/Users/jselsing/Work/work_rawDATA/XSGRB/"
        # object_name = data_dir + "GRB121229A/"
        data_dir = "/Users/jselsing/Work/work_rawDATA/STARGATE/"
        object_name = data_dir + "GRB171205A/"

        # object_name = "/Users/jselsing/Work/work_rawDATA/XSGW/AT2017GFO/"
        # object_name = "/Users/jselsing/Work/work_rawDATA/HZSN/RLC16Nim/"

        args.filepath = object_name

        arms = ["UVB", "VIS", "NIR"] # # UVB, VIS, NIR, ["UVB", "VIS", "NIR"]

        combine = False # True False

        OBs = ["OB5"] # ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9", "OB10", "OB11", "OB12", "OB13", "OB14"]
        for ll in OBs:
            args.OB = ll
            # print(ll)
            for ii in arms:
                args.arm = ii # UVB, VIS, NIR
                args.mode = "STARE"
                # args.mode = "NODSTARE"
                # if ii == "NIR":
                #     args.mode = "NODSTARE"
                # if combine:
                #     args.mode = "COMBINE"


                args.use_master_response = False # True False
                args.masks = []
                args.seeing = 1.0
                args.repeats = 1

                run_combination(args)

    else:
        main(argv = sys.argv[1:])
