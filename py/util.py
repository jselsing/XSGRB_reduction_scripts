#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from astropy.stats import sigma_clip
from scipy import signal

__all__ = ["correct_for_dust", "bin_image", "weighted_avg", "gaussian", "voigt", "slit_loss", "convert_air_to_vacuum", "convert_vacuum_to_air", "inpaint_nans"]


def gaussian(x, amp, cen, sigma):
    # Simple Gaussian
    return amp * np.exp(-(x - cen)**2 / sigma**2)


def voigt(x, amp=1, cen=0, sigma=1, gamma=0):
    """1 dimensional voigt function.
    see http://en.wikipedia.org/wiki/Voigt_profile
    """
    from scipy.special import wofz
    z = (x-cen + 1j*gamma)/ (sigma*np.sqrt(2.0))
    return amp * wofz(z).real / (sigma*np.sqrt(2*np.pi))


def slit_loss(seeing, slit_width):
    """
    Calculates the slit-loss based on the seeing sigma and slit width in arcsec
    """
    from scipy.special import erf
    # FWHM = 2 * np.sqrt(2 * np.log(2))
    return 1/erf((slit_width/2) / (np.sqrt(2) * (seeing)))


def weighted_avg(flux, error, axis=2):

    """Calculate the weighted average with errors
    ----------
    flux : masked array-like
        Values to take average of
    error : masked array-like
        Errors associated with values, assumed to be standard deviations.
    mask : masked array-like
        Errors associated with values, assumed to be standard deviations.
    axis : int, default 0
        axis argument passed to numpy.ma.average

    Returns
    -------
    average, error : tuple

    Notes
    -----
    Functionality similar to np.ma.average, only also returns the associated error
    """

    # Normalize to avoid numerical issues in flux-calibrated data
    norm = abs(np.ma.mean(flux))
    flux_func = flux.copy() / norm
    error_func = error.copy() / norm

    weight = 1.0 / (error_func ** 2.0)

    average, sow = np.ma.average(flux_func, weights = weight, axis = axis, returned = True)
    variance = 1.0 / sow

    return (average * norm, np.sqrt(variance)*norm)


def correct_for_dust(wavelength, ra, dec):
    """Query IRSA dust map for E(B-V) value and returns reddening array
    ----------
    wavelength : numpy array-like
        Wavelength values for which to return reddening
    ra : float
        Right Ascencion in degrees
    dec : float
        Declination in degrees

    Returns
    -------
    reddening : numpy array

    Notes
    -----
    For info on the dust maps, see http://irsa.ipac.caltech.edu/applications/DUST/
    """

    from astroquery.irsa_dust import IrsaDust
    import astropy.coordinates as coord
    import astropy.units as u
    C = coord.SkyCoord(ra*u.deg, dec*u.deg, frame='fk5')
    dust_image = IrsaDust.get_images(C, radius=2 *u.deg, image_type='ebv')[0]
    ebv = np.mean(dust_image[0].data[40:42,40:42])
    # print(ebv)
    r_v = 3.1
    av =  r_v * ebv
    from specutils.extinction import reddening
    return reddening(wavelength* u.angstrom, av, r_v=r_v, model='ccm89'), ebv


def bin_image(flux, error, binh):

    """Bin low S/N 2D data from xshooter
    ----------
    flux : np.array containing 2D-image flux
        Flux in input image
    error : np.array containing 2D-image error
        Error in input image
    binh : int
        binning along x-axis

    Returns
    -------
    binned fits image
    """

    print("Binning image by a factor: "+str(binh))
    if binh == 1:
        return flux, error

    # Outsize
    v_size, h_size = flux.shape
    outsizeh = int(h_size/binh)

    # Containers
    res = np.ma.zeros((v_size, outsizeh))
    reserr = np.ma.zeros((v_size, outsizeh))

    flux_tmp = flux.copy()
    for ii in np.arange(0, h_size - binh, binh):
        # Find psotions in new array
        h_slice = slice(ii, ii + binh)
        h_index = int((ii + binh)/binh - 1)

        # Sigma clip before binning to remove noisy pixels with bad error estimate.
        clip_mask = sigma_clip(flux[:, ii:ii + binh], axis=1)

        # Combine masks
        mask = flux[:, ii:ii + binh].mask | clip_mask.mask
        flux_tmp[:, ii:ii + binh].mask = mask

        # Construct weighted average and weighted std along binning axis
        res[:, h_index], reserr[:, h_index] = weighted_avg(flux_tmp[:, ii:ii + binh], error[:, ii:ii + binh], axis=1)

    return res, reserr


def convert_air_to_vacuum(air_wave) :
    # convert air to vacuum. Based onhttp://idlastro.gsfc.nasa.gov/ftp/pro/astro/airtovac.pro
    # taken from https://github.com/desihub/specex/blob/master/python/specex_air_to_vacuum.py

    sigma2 = (1e4/air_wave)**2
    fact = 1. +  5.792105e-2/(238.0185 - sigma2) +  1.67917e-3/( 57.362 - sigma2)
    vacuum_wave = air_wave*fact

    # comparison with http://www.sdss.org/dr7/products/spectra/vacwavelength.html
    # where : AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)
    # air_wave=numpy.array([4861.363,4958.911,5006.843,6548.05,6562.801,6583.45,6716.44,6730.82])
    # expected_vacuum_wave=numpy.array([4862.721,4960.295,5008.239,6549.86,6564.614,6585.27,6718.29,6732.68])
    return vacuum_wave


def convert_vacuum_to_air(vac_wave) :
    # convert vacuum to air
    # taken from http://www.sdss.org/dr7/products/spectra/vacwavelength.html

    air_wave = vac_wave / (1.0 + 2.735182e-4 + 131.4182 / vac_wave**2 + 2.76249e8 / vac_wave**4)
    return air_wave


def inpaint_nans(im, kernel_size=5):
    # Taken from http://stackoverflow.com/a/21859317/6519723
    ipn_kernel = np.ones((kernel_size, kernel_size)) # kernel for inpaint_nans
    ipn_kernel[int(kernel_size/2), int(kernel_size/2)] = 0

    nans = np.isnan(im)
    while np.sum(nans)>0:
        im[nans] = 0
        vNeighbors = signal.convolve2d((nans == False), ipn_kernel, mode='same', boundary='symm')
        im2 = signal.convolve2d(im, ipn_kernel, mode='same', boundary='symm')
        im2[vNeighbors > 0] = im2[vNeighbors > 0]/vNeighbors[vNeighbors > 0]
        im2[vNeighbors == 0] = np.nan
        im2[(nans == False)] = im[(nans == False)]
        im = im2
        nans = np.isnan(im)
    return im


