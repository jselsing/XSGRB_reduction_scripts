#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import numpy as np
from astropy.stats import sigma_clip
from scipy import signal
import matplotlib.pyplot as pl
from scipy.special import wofz, erf

__all__ = ["correct_for_dust", "bin_image", "avg", "gaussian", "voigt", "two_voigt", "slit_loss", "convert_air_to_vacuum", "convert_vacuum_to_air", "inpaint_nans", "bin_spectrum", "form_nodding_pairs", "find_nearest"]


def find_nearest(array, value):
        idx = (np.abs(array-value)).argmin()
        return idx


def gaussian(x, amp, cen, sigma):
    # Simple Gaussian
    return amp * np.exp(-(x - cen)**2 / sigma**2)


def voigt(x, amp=1, cen=0, sigma=1, gamma=0, c=0, a=0):
    """1 dimensional voigt function.
    see http://en.wikipedia.org/wiki/Voigt_profile
    """
    # Penalize negative values
    if sigma <= 0:
        amp = 1e10
    if gamma <= 0:
        amp = 1e10
    if amp <= 0:
        amp = 1e10
    z = (x-cen + 1j*gamma)/ (sigma*np.sqrt(2.0))

    return amp * wofz(z).real / (sigma*np.sqrt(2*np.pi)) + c + a * x

def two_voigt(x, amp=1, cen=0, sigma=1, gamma=0, c=0, a=0, amp2=0.0, cen2=-1, sig2=0.5, gam2=0):
    """1 dimensional voigt function.
    see http://en.wikipedia.org/wiki/Voigt_profile
    """
    # Penalize negative values
    if sigma <= 0:
        amp = 1e10
    if sig2 <= 0:
        amp = 1e10
    if gamma <= 0:
        amp = 1e10
    if gam2 <= 0:
        amp = 1e10
    if amp <= 0:
        amp = 1e10
    if amp2 <= 0:
        amp2 = 1e10
    z = (x-cen + 1j*gamma)/ (sigma*np.sqrt(2.0))
    z2 = (x-cen2 + 1j*gam2)/ (sig2*np.sqrt(2.0))
    return amp * wofz(z).real / (sigma*np.sqrt(2*np.pi)) + c + a * x + amp2 * wofz(z2).real / (sig2*np.sqrt(2*np.pi))


def slit_loss(g_sigma, slit_width, l_sigma=False):
    """
    Calculates the slit-loss based on the seeing sigma and slit width in arcsec
    """
    # With pure Gaussian, do the analytical solution
    try:
        if not l_sigma:
            # FWHM = 2 * np.sqrt(2 * np.log(2))
            return 1/erf((slit_width/2) / (np.sqrt(2) * (g_sigma)))
    except:
        pass
    # For the voigt, calculate the integral numerically.
    x = np.arange(-10, 10, 0.01)
    v = [voigt(x, sigma = kk, gamma = l_sigma[ii]) for ii, kk in enumerate(g_sigma)]
    mask = (x > -slit_width/2) & (x < slit_width/2)
    sl = np.zeros_like(g_sigma)
    for ii, kk in enumerate(g_sigma):
        sl[ii] = np.trapz(v[ii], x) / np.trapz(v[ii][mask], x[mask])
    return sl


def avg(flux, error, mask=None, axis=2, weight=False, weight_map=None):

    """Calculate the weighted average with errors
    ----------
    flux : array-like
        Values to take average of
    error : array-like
        Errors associated with values, assumed to be standard deviations.
    mask : array-like
        Array of bools, where true means a masked value.
    axis : int, default 0
        axis argument passed to numpy

    Returns
    -------
    average, error : tuple

    Notes
    -----
    """
    try:
        if not mask:
            mask = np.zeros_like(flux).astype("bool")
    except:
        pass
        # print("All values are masked... Returning nan")
        # if np.sum(mask.astype("int")) == 0:
        #     return np.nan, np.nan, np.nan


    # Normalize to avoid numerical issues in flux-calibrated data
    norm = abs(np.median(flux[flux > 0]))
    if norm == np.nan or norm == np.inf or norm == 0:
        print("Nomalization factor in avg has got a bad value. It's "+str(norm)+" ... Replacing with 1")

    flux_func = flux.copy() / norm
    error_func = error.copy() / norm

    # Calculate average based on supplied weight map
    if weight_map is not None:
        # Remove non-contributing pixels
        flux_func[mask] = 0
        error_func[mask] = 0
        average = np.sum(weight_map * flux_func, axis = axis)
        variance = np.sum(weight_map ** 2 * error_func ** 2.0, axis = axis)

    # Inverse variance weighted average
    elif weight:
        ma_flux_func = np.ma.array(flux_func, mask=mask)
        ma_error_func = np.ma.array(error_func, mask=mask)
        w = 1.0 / (ma_error_func ** 2.0)
        average = np.ma.sum(ma_flux_func * w, axis = axis) / np.ma.sum(w, axis = axis)
        variance = 1. / np.ma.sum(w, axis = axis)
        if not isinstance(average, float):
            # average[average.mask] = np.nan
            average = average.data
            # variance[variance.mask] = np.nan
            variance = variance.data

    # Normal average
    elif not weight:
        # Number of pixels in the mean
        n = np.sum((~mask).astype("int"), axis = axis)
        # Remove non-contributing pixels
        flux_func[mask] = 0
        error_func[mask] = 0
        # mean
        average = (1 / n) * np.sum(flux_func, axis = axis)
        # probagate errors
        variance = (1 / n**2) * np.sum(error_func ** 2.0, axis = axis)

    mask = (np.sum((~mask).astype("int"), axis = axis) == 0).astype("int")
    return (average * norm, np.sqrt(variance)*norm, mask)


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
    dust_image = IrsaDust.get_images(C, radius=2 *u.deg, image_type='ebv', timeout=60)[0]
    ebv = np.mean(dust_image[0].data[40:42, 40:42])
    r_v = 3.1
    av =  r_v * ebv
    from specutils.extinction import reddening
    return reddening(wavelength* u.angstrom, av, r_v=r_v, model='ccm89'), ebv


def bin_spectrum(wl, flux, error, mask, binh, weight=False):

    """Bin low S/N 1D data from xshooter
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
        return wl, flux, error, mask

    # Outsize
    size = flux.shape[0]
    outsize = int(np.round(size/binh))

    # Containers
    wl_out = np.zeros((outsize))
    res = np.zeros((outsize))
    reserr = np.zeros((outsize))
    resbp = np.zeros((outsize))

    for ii in np.arange(0, size - binh, binh):
        # Find psotions in new array
        h_slice = slice(ii, ii + binh)
        h_index = int((ii + binh)/binh) - 1
        # Construct weighted average and weighted std along binning axis
        res[h_index], reserr[h_index], resbp[h_index] = avg(flux[ii:ii + binh], error[ii:ii + binh], mask = mask[ii:ii + binh], axis=0, weight=weight)
        wl_out[h_index] = np.median(wl[ii:ii + binh], axis=0)

    return wl_out[1:-1], res[1:-1], reserr[1:-1], resbp[1:-1]


def bin_image(flux, error, mask, binh, weight=False):

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
    res = np.zeros((v_size, outsizeh))
    reserr = np.zeros((v_size, outsizeh))

    flux_tmp = flux.copy()
    for ii in np.arange(0, h_size - binh, binh):
        # Find psotions in new array
        h_slice = slice(ii, ii + binh)
        h_index = int((ii + binh)/binh - 1)

        # Sigma clip before binning to remove noisy pixels with bad error estimate.
        clip_mask = sigma_clip(flux[:, ii:ii + binh], axis=1)

        # Combine masks
        mask_comb = mask[:, ii:ii + binh].astype("bool") | clip_mask.mask

        # Construct weighted average and weighted std along binning axis
        res[:, h_index], reserr[:, h_index], __ = avg(flux_tmp[:, ii:ii + binh], error[:, ii:ii + binh], mask=mask_comb, axis=1, weight=weight)

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


def form_nodding_pairs(flux_cube, error_cube, bpmap_cube, naxis2, pix_offsety):

    if not len(pix_offsety) % 2 == 0:
        print("")
        print("Attempting to form nodding pairs out of an uneven number of images ...")
        print("Discarding last image ...")
        print("")
        pix_offsety = pix_offsety[:-1]

    flux_cube_out = np.zeros(flux_cube.shape)
    error_cube_out = np.zeros(error_cube.shape)
    bpmap_cube_out = np.ones(bpmap_cube.shape)*10

    # Make mask based on the bad-pixel map, the edge mask and the sigma-clipped mask
    mask_cube = (bpmap_cube != 0)

    # Setting masks
    flux_cube[mask_cube] = 0
    error_cube[mask_cube] = 0
    bpmap_cube[mask_cube] = 1

    # Finding the indices of the container in which to put image.
    offv = np.zeros_like(pix_offsety)
    for ii, kk in enumerate(pix_offsety):
        offv[ii] = kk - min(pix_offsety)

    # Define slices where to put image
    v_range = []
    for ii, kk in enumerate(offv):
        v_range.append(slice(kk, naxis2 + kk))

    # From A-B and B-A pairs
    alter = 1
    for ii, kk in enumerate(v_range):
        flux_cube_out[kk, :, ii] =  flux_cube[kk, :, ii] - flux_cube[v_range[ii + alter], :, ii + alter]
        error_cube_out[kk, :, ii] = np.sqrt(error_cube[kk, :, ii]**2. + error_cube[v_range[ii + alter], :, ii + alter]**2.)
        bpmap_cube_out[kk, :, ii] = bpmap_cube[kk, :, ii] + bpmap_cube[v_range[ii + alter], :, ii + alter]
        alter *= -1

        # Subtract residiual sky due to varying sky-brightness over obserations
        median = np.tile(np.nanmedian(flux_cube_out[kk, :, ii], axis=0), (flux_cube_out[kk, :, ii].shape[0], 1))
        median[bpmap_cube_out[kk, :, ii].astype("bool")] = 0
        flux_cube_out[kk, :, ii] = flux_cube_out[kk, :, ii] - median

    # Form A-B - shifted(B-A) pairs
    alter = 1
    for ii, kk in enumerate(v_range):
        if alter == 1:
            flux_cube_out[:, :, ii] = flux_cube_out[:, :, ii] + flux_cube_out[:, :, ii + 1]
            error_cube_out[:, :, ii] = np.sqrt(error_cube_out[:, :, ii]**2. + error_cube_out[:, :, ii + 1]**2.)
            bpmap_cube_out[:, :, ii] = bpmap_cube_out[:, :, ii] + bpmap_cube_out[:, :, ii + 1]
        elif alter == -1:
            flux_cube_out[:, :, ii] = np.nan
            error_cube_out[:, :, ii] = np.nan
            bpmap_cube_out[:, :, ii] = np.ones_like(bpmap_cube_out[:, :, ii])*666
        alter *= -1

    n_pix = np.ones_like(bpmap_cube_out) + (~(bpmap_cube_out.astype("bool"))).astype("int")
    flux_cube_out = flux_cube_out/n_pix
    error_cube_out = error_cube_out/(n_pix)

    good_mask = (bpmap_cube_out == 0) | (bpmap_cube_out == 10)| (bpmap_cube_out == 2)
    bpmap_cube_out[good_mask] = 0

    return flux_cube_out, error_cube_out, bpmap_cube_out

