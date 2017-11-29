#!/usr/local/anaconda3/envs/py36 python
# -*- coding: utf-8 -*-

# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')

# Imports
import numpy as np
from astropy.io import fits
from scipy import interpolate
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from scipy.signal import medfilt
import astropy.units as u
from astropy.time import Time

import pyphot
lib = pyphot.get_library()

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
params = {
   'axes.labelsize': 12,
   'font.size': 12,
   'legend.fontsize': 12,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'text.usetex': True,
   'figure.figsize': [8, 8/1.61]
   }
mpl.rcParams.update(params)




def synpassflux(wl, flux, band):
    # Calculate synthetic magnitudes
    n_bands = len(band)
    synmags = [0]*n_bands
    cwav = [0]*n_bands
    cwave = [0]*n_bands
    for ii, kk in enumerate(band):
        filt = np.genfromtxt("/Users/jselsing/github/iPTF16geu/data/passbands/%s.dat"%kk)
        lamb_T, T = filt[:,0], filt[:,1]

        f = pyphot.Filter(lamb_T, T, name=kk, dtype='photon', unit='Angstrom')
        fluxes = f.get_flux(wl, flux, axis=0)
        synmags[ii] = -2.5 * np.log10(fluxes) - f.AB_zero_mag
        cwav[ii] = np.mean(lamb_T)
        cwave[ii] = (max(lamb_T[T > 0.2] - cwav[ii]), (cwav[ii] - min(lamb_T[T > 0.2])))
    synmag_flux = ((synmags*u.ABmag).to((u.erg/(u.s * u.cm**2 * u.AA)), u.spectral_density(cwav * u.AA))).value
    return synmag_flux, cwav, cwave



def main():

    root_dir = "/Users/jselsing/Work/work_rawDATA/STARGATE/GRB171010A/final/"
    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9", "OB10", "OB11", "OB12", "OB13", "OB14"]
    OBs = ["OB1_SN"]
    # OBs = ["OB1", "OB2", "OB3"]
    colors = sns.color_palette("Blues_r", len(OBs))
    for ii, kk in enumerate(OBs):

        off = 0#(len(OBs) - ii) * 2e-17
        mult = 1.0

        ############################## OB ##############################
        f = fits.open(root_dir + "UVB%s.fits"%kk)
        # print(f[1].header)
        wl = f[1].data.field("WAVE").flatten()

        q = f[1].data.field("QUAL").flatten()
        t = f[1].data.field("TRANS").flatten()
        mask_wl = (wl > 320) & (wl < 540)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)

        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot)
        error = error(wl_plot)
        # pl.plot(wl_plot, flux, lw=0.3, color = "black", alpha = 0.4)
        pl.plot(wl_plot, off + medfilt(error, 101), lw=1.5, color = "black", alpha = 0.4, linestyle = "dashed")
        pl.plot(wl_plot, off + mult*medfilt(flux, 51), color = colors[ii], lw=1.5, rasterized = True)
        # pl.fill_between(wl_plot, off + mult*medfilt(flux, 51) - mult*medfilt(error, 51), off + mult*medfilt(flux, 51) + mult*medfilt(error, 51), color = "grey", alpha = 0.5, rasterized = True)
        f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        # f.writeto(root_dir + "final/UVB%s.fits"%kk, clobber = True)
        max_v, min_v = max(medfilt(flux, 101)), min(medfilt(flux, 101))

        f = fits.open(root_dir + "VIS%s.fits"%kk)

        wl = f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()
        t = f[1].data.field("TRANS").flatten()
        mask_wl = (wl > 590) & (wl < 1000)
        mask_qual = ~q.astype("bool")

        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]
        # pl.plot(wl_plot, flux, lw=0.3, color = "black", alpha = 0.4)
        pl.plot(wl_plot, off + medfilt(error, 101), lw=1.5, color = "black", alpha = 0.4, linestyle = "dashed")
        pl.plot(wl_plot, off + mult*medfilt(flux, 51), color = colors[ii], lw=1.5, rasterized = True)
        # pl.fill_between(wl_plot, off + mult*medfilt(flux, 51) - mult*medfilt(error, 51), off + mult*medfilt(flux, 51) + mult*medfilt(error, 51), color = "grey", alpha = 0.5, rasterized = True)
        f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        f[1].data["FLUX"] = f[1].data["FLUX"] / t
        f[1].data["ERR"] = f[1].data["ERR"] / t
        # f.writeto(root_dir + "final/VIS%s.fits"%kk, clobber = True)
        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(min(medfilt(flux, 101)), min_v)

        f = fits.open(root_dir + "NIR%s.fits"%kk)
        wl = f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()
        t = f[1].data.field("TRANS").flatten()
        mask_wl = (wl > 1040) & (t > 0.3)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]
        # pl.plot(wl_plot, flux, lw=0.3, color = "black", alpha = 0.4)
        pl.plot(wl_plot, off + medfilt(error, 101), lw=1.5, color = "black", alpha = 0.4, linestyle = "dashed")
        pl.plot(wl_plot, off + mult*medfilt(flux, 51), color = colors[ii], lw=1.5, label = "XSH: %s"%str(f[0].header['DATE-OBS']), rasterized = True)
        # pl.fill_between(wl_plot, off + mult*medfilt(flux, 51) - mult*medfilt(error, 51), off + mult*medfilt(flux, 51) + mult*medfilt(error, 51), color = "grey", alpha = 0.5, rasterized = True)
        f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        f[1].data["FLUX"] = f[1].data["FLUX"] / t
        f[1].data["ERR"] = f[1].data["ERR"] / t
        # f.writeto(root_dir + "final/NIR%s.fits"%kk, clobber = True)
        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(min(medfilt(flux, 101)), min_v)
        pl.axvspan(1260, 1280, color = "grey", alpha = 0.2)
        pl.axvspan(1350, 1450, color = "grey", alpha = 0.2)
        pl.axvspan(1800, 1950, color = "grey", alpha = 0.2)
        print(max(f[0].header['HIERARCH ESO TEL AMBI FWHM END'], f[0].header['HIERARCH ESO TEL AMBI FWHM START']))


    pl.ylim(1.1 * min_v, 1.25 * max_v)
    pl.xlim(250, 2500)
    pl.xlabel(r"Observed wavelength  [$\mathrm{\AA}$]")
    pl.ylabel(r'Flux density [$\mathrm{erg} \mathrm{s}^{-1} \mathrm{cm}^{-1} \mathrm{\AA}^{-1}$]')
    leg = pl.legend(loc=1)

    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    pl.savefig(root_dir + "spectra_SN.pdf")
    pl.show()


if __name__ == '__main__':
    main()