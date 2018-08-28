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

    root_dir = "/Users/jselsing/Work/work_rawDATA/STARGATE/GRB180728A/final/"
    # root_dir = "/Users/jselsing/Work/work_rawDATA/SLSN/SN2018bsz/final/"

    # root_dir = "/Users/jselsing/Work/work_rawDATA/XSGW/AT2017GFO/final/"
    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9", "OB10", "OB11", "OB12", "OB13", "OB14", "OB15", "OB16", "OB17", "OB18"]
    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5"]
    OBs = ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8"]
    z = 0#3.07
    colors = sns.color_palette("Blues_r", 2*len(OBs))
    # colors = sns.color_palette("Blues_r", len(OBs))


    wl_out, flux_out, error_out = 0, 0, 0
    for ii, kk in enumerate(OBs):

        off = 0#(len(OBs) - ii) * 2e-17
        mult = 1.0

        ############################## OB ##############################
        f = fits.open(root_dir + "UVB%s.fits"%kk)
        wl = 10.*f[1].data.field("WAVE").flatten()


        q = f[1].data.field("QUAL").flatten()
        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(q)
        mask_wl = (wl > 3200) & (wl < 5550)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)

        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot)
        error = error(wl_plot)
        print(flux/error)

        pl.plot(wl_plot[::5]/(1+z), off + mult*flux[::5], lw=0.3, color = "black", alpha = 0.2, rasterized = True)

        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), 50)
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        np.savetxt(root_dir+"UVB%s_bin50.dat"%kk, list(zip(wl_out, flux_out, error_out)))

        # pl.plot(b_wl, off + mult*b_f, color="firebrick", linestyle="steps-mid")
        # pl.plot(b_wl/(1+z), off + mult*b_e, color="black", linestyle="dashed")

        # pl.plot(wl_plot/(1+z), off + mult*medfilt(flux, 251), color = colors[ii], rasterized = True)
        pl.plot(b_wl/(1+z), off + mult*b_f, color= colors[ii], linestyle="steps-mid", rasterized = True)

        # pl.fill_between(wl_plot, off + mult*medfilt(flux, 1) - mult*medfilt(error, 1), off + mult*medfilt(flux, 1) + mult*medfilt(error, 1), color = "grey", alpha = 0.5, rasterized = True)
        # f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        # f.writeto(root_dir + "final/UVB%s.fits"%kk, clobber = True)
        max_v, min_v = max(medfilt(flux, 101)), min(medfilt(flux, 101))
        f = fits.open(root_dir + "VIS%s.fits"%kk)

        wl = 10.*f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()

        t = f[1].data.field("TRANS").flatten()
        # t = np.ones_like(f[1].data.field("TRANS").flatten())
        mask_wl = (wl > 5650) & (wl < 10000)
        mask_qual = ~q.astype("bool")

        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]

        pl.plot(wl_plot[::5]/(1+z), off + mult*flux[::5], lw=0.3, color = "black", alpha = 0.2, rasterized = True)

        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), 50)
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        np.savetxt(root_dir+"VIS%s_bin50.dat"%kk, list(zip(wl_out, flux_out, error_out)))
        # pl.plot(b_wl/(1+z), off + mult*b_e, color="black", linestyle="dashed")

        # pl.plot(wl_plot/(1+z), off + mult*medfilt(flux, 251), color = colors[ii], rasterized = True)
        pl.plot(b_wl/(1+z), off + mult*b_f, color= colors[ii], linestyle="steps-mid", rasterized = True)

        # pl.fill_between(wl_plot, off + mult*medfilt(flux, 1) - mult*medfilt(error, 1), off + mult*medfilt(flux, 1) + mult*medfilt(error, 1), color = "grey", alpha = 0.5, rasterized = True)

        # wl_out, flux_out, error_out = np.concatenate((wl_out, b_wl)), np.concatenate((flux_out, b_f)), np.concatenate((error_out, b_e))
        # f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        # f[1].data["FLUX"] = f[1].data["FLUX"] / t
        # f[1].data["ERR"] = f[1].data["ERR"] / t
        # f.writeto(root_dir + "final/VIS%s.fits"%kk, clobber = True)
        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(min(medfilt(flux, 101)), min_v)

        f = fits.open(root_dir + "NIR%s.fits"%kk)
        wl = 10.*f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()

        t = f[1].data.field("TRANS").flatten()
        # t = np.ones_like(f[1].data.field("TRANS").flatten())
        mask_wl = (wl > 10500) & (wl < 25000) & (t > 0.3)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]


        pl.plot(wl_plot[::5]/(1+z), off + mult*flux[::5], lw=0.3, color = "black", alpha = 0.2, rasterized = True)

        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), 50)
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        np.savetxt(root_dir+"NIR%s_bin50.dat"%kk, list(zip(wl_out, flux_out, error_out)))

        # pl.plot(b_wl, off + mult*b_f, color="firebrick", linestyle="steps-mid")
        # pl.plot(b_wl/(1+z), off + mult*b_e, color="black", linestyle="dashed")

        # pl.plot(wl_plot/(1+z), off + mult*medfilt(flux, 251), color = colors[ii], lw=1.5, label = "XSH: %s"%str(f[0].header['DATE-OBS']), rasterized = True)
        pl.plot(b_wl/(1+z), off + mult*b_f, color= colors[ii], linestyle="steps-mid", rasterized = True, label = "XSH: %s"%str(f[0].header['DATE-OBS']))

        # pl.fill_between(wl_plot, off + mult*medfilt(flux, 51) - mult*medfilt(error, 51), off + mult*medfilt(flux, 51) + mult*medfilt(error, 51), color = "grey", alpha = 0.5, rasterized = True)
        # f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        # f[1].data["FLUX"] = f[1].data["FLUX"] / t
        # f[1].data["ERR"] = f[1].data["ERR"] / t
        # f.writeto(root_dir + "final/NIR%s.fits"%kk, clobber = True)
        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(min(medfilt(flux, 101)), min_v)
        max_std = np.nanstd(flux)
        # print(max_std)
        pl.axvspan(5550, 5650, color = "grey", alpha = 0.2)
        pl.axvspan(10000, 10500, color = "grey", alpha = 0.2)
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
        # pl.ylim(1.1 * min_v, 1.5 * max_v)
        pl.ylim(-max_std, 1.6e-16)
        pl.xlim(2500, 23000)
        pl.xlabel(r"Observed wavelength  [$\mathrm{\AA}$]")
        pl.ylabel(r'Flux density [$\mathrm{erg} \mathrm{s}^{-1} \mathrm{cm}^{-1} \mathrm{\AA}^{-1}$]')
        leg = pl.legend(loc=1)

        # set the linewidth of each legend object
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)
        pl.semilogy()
        # pl.loglog()
        # pl.axhline(0, linestyle="dashed", color="black")
        # pl.show()
    pl.savefig(root_dir + "%s_all.pdf"%kk)
        # pl.clf()
    pl.show()
    # pl.ylim(7e-17, 4e-14)
    # pl.axhline(0, linestyle="dashed", color="black")
    # pl.savefig(root_dir + "all.pdf")
    # pl.show()

if __name__ == '__main__':
    main()