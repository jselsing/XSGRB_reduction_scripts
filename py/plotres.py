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
from util import *


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

    root_dir = "/Users/jselsing/Work/work_rawDATA/STARGATE/GRB171205A/final/"
    # root_dir = "/Users/jselsing/Work/work_rawDATA/XSGW/AT2017GFO/final/"
    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9", "OB10", "OB11", "OB12", "OB13", "OB14"]
    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5"]
    OBs = ["OB5"]
    z = 0.0368
    colors = sns.color_palette("Blues_r", len(OBs))
    wl_out, flux_out, error_out = 0, 0, 0
    for ii, kk in enumerate(OBs):

        off = 0#(len(OBs) - ii) * 2e-17
        mult = 1.0

        ############################## OB ##############################
        f = fits.open(root_dir + "UVB%s.fits"%kk)
        # print(f[1].header)
        wl = 10.*f[1].data.field("WAVE").flatten()/(1+z)

        q = f[1].data.field("QUAL").flatten()
        t = f[1].data.field("TRANS").flatten()
        mask_wl = (wl > 3200) & (wl < 5550)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)

        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot)
        error = error(wl_plot)
        wl_plot, flux, error = filter_bad_values(wl_plot, flux, error)
        # pl.plot(wl_plot, flux, lw=0.3, color = "black", alpha = 0.4)
        pl.plot(wl_plot, off + medfilt(error, 101), lw=1.5, color = "black", alpha = 0.4, linestyle = "dashed")
        pl.plot(wl_plot, off + mult*medfilt(flux, 1), color = colors[0], lw=0.5, alpha=0.5, rasterized = True)
        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), 30)
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        pl.plot(b_wl, off + mult*b_f, color="black", linestyle="steps-mid")
        pl.plot(wl_plot, off + mult*medfilt(flux, 51), color = "firebrick", rasterized = True)
        # pl.fill_between(wl_plot, off + mult*medfilt(flux, 51) - mult*medfilt(error, 51), off + mult*medfilt(flux, 51) + mult*medfilt(error, 51), color = "grey", alpha = 0.5, rasterized = True)
        f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        # f.writeto(root_dir + "final/UVB%s.fits"%kk, clobber = True)
        max_v, min_v = max(medfilt(flux, 101)), min(medfilt(flux, 101))

        f = fits.open(root_dir + "VIS%s.fits"%kk)

        wl = 10.*f[1].data.field("WAVE").flatten()/(1+z)
        q = f[1].data.field("QUAL").flatten()
        t = f[1].data.field("TRANS").flatten()
        mask_wl = (wl > 5650) & (wl < 10000)
        mask_qual = ~q.astype("bool")

        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]
        wl_plot, flux, error = filter_bad_values(wl_plot, flux, error)
        # pl.plot(wl_plot, flux, lw=0.3, color = "black", alpha = 0.4)
        pl.plot(wl_plot, off + medfilt(error, 101), lw=1.5, color = "black", alpha = 0.4, linestyle = "dashed")

        pl.plot(wl_plot, off + mult*medfilt(flux, 1), color = colors[0], lw=0.5, alpha=0.5, rasterized = True)
        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), 30)
        pl.plot(b_wl, off + mult*b_f, color="black", linestyle="steps-mid")
        pl.plot(wl_plot, off + mult*medfilt(flux, 51), color = "firebrick", rasterized = True)
        wl_out, flux_out, error_out = np.concatenate((wl_out, b_wl)), np.concatenate((flux_out, b_f)), np.concatenate((error_out, b_e))
        f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        f[1].data["FLUX"] = f[1].data["FLUX"] / t
        f[1].data["ERR"] = f[1].data["ERR"] / t
        # f.writeto(root_dir + "final/VIS%s.fits"%kk, clobber = True)
        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(min(medfilt(flux, 101)), min_v)

        f = fits.open(root_dir + "NIR%s.fits"%kk)
        wl = 10.*f[1].data.field("WAVE").flatten()/(1+z)
        q = f[1].data.field("QUAL").flatten()
        t = f[1].data.field("TRANS").flatten()
        mask_wl = (wl > 10200) & (t > 0.3) & (wl < 22500)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]
        wl_plot, flux, error = filter_bad_values(wl_plot, flux, error)
        # pl.plot(wl_plot, flux, lw=0.3, color = "black", alpha = 0.4)
        pl.plot(wl_plot, off + medfilt(error, 101), lw=1.5, color = "black", alpha = 0.4, linestyle = "dashed")


        pl.plot(wl_plot, off + mult*medfilt(flux, 1), color = colors[0], lw=0.5, alpha=0.5, rasterized = True, label = "XSH: %s"%str(f[0].header['DATE-OBS']))
        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), 30)
        pl.plot(b_wl, off + mult*b_f, color="black", linestyle="steps-mid")
        pl.plot(wl_plot, off + mult*medfilt(flux, 51), color = "firebrick", rasterized = True)

        # pl.plot(wl_plot, off + mult*medfilt(flux, 51), color = colors[0], lw=1.5, label = "XSH: %s"%str(f[0].header['DATE-OBS']), rasterized = True)
        # pl.fill_between(wl_plot, off + mult*medfilt(flux, 51) - mult*medfilt(error, 51), off + mult*medfilt(flux, 51) + mult*medfilt(error, 51), color = "grey", alpha = 0.5, rasterized = True)
        f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        f[1].data["FLUX"] = f[1].data["FLUX"] / t
        f[1].data["ERR"] = f[1].data["ERR"] / t
        # f.writeto(root_dir + "final/NIR%s.fits"%kk, clobber = True)
        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(min(medfilt(flux, 101)), min_v)
        pl.axvspan(5550, 5650, color = "grey", alpha = 0.2)
        pl.axvspan(10000, 10200, color = "grey", alpha = 0.2)
        pl.axvspan(12600, 12800, color = "grey", alpha = 0.2)
        pl.axvspan(13500, 14500, color = "grey", alpha = 0.2)
        pl.axvspan(18000, 19500, color = "grey", alpha = 0.2)
        # print(max(f[0].header['HIERARCH ESO TEL AMBI FWHM END'], f[0].header['HIERARCH ESO TEL AMBI FWHM START']))
        # print(np.diff(wl_out), np.median(np.diff(wl_out)))
        # np.savetxt("UVBVIS_bin30OB3.dat", list(zip(wl_out, flux_out, error_out)))

        # h = 6.626e-34
        # c = 3.0e+8
        # k = 1.38e-23
        # def planck(wav, T):
        #     a = 2.0*h*c**2
        #     b = h*c/(wav*k*T)
        #     intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
        #     return intensity
        # x = np.arange(0, 30000, 1)
        # pl.plot(x, 0.8e-8*x**(-2.1))
        # pl.plot(x, 5e-31*planck(x*1e-10, 13000))
        pl.ylim(1.1 * min_v, 1.25 * max_v)
        # pl.ylim(5e-19, 5.25 * max_v)
        pl.xlim(2500, 23000)
        pl.xlabel(r"Observed wavelength  [$\mathrm{\AA}$]")
        pl.ylabel(r'Flux density [$\mathrm{erg} \mathrm{s}^{-1} \mathrm{cm}^{-1} \mathrm{\AA}^{-1}$]')
        leg = pl.legend(loc=1)

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)
        # pl.loglog()
        pl.savefig(root_dir + "GRB171205A%s.pdf"%kk)
        # pl.clf()
        pl.show()


if __name__ == '__main__':
    main()