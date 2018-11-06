#!/usr/local/anaconda3/envs/py36 python
# -*- coding: utf-8 -*-

# Plotting
import matplotlib; matplotlib.use('Qt5Agg')
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')

# Imports
import numpy as np
import pandas as pd
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
    'figure.figsize': [8, 8 / 1.61]
}
mpl.rcParams.update(params)


def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f

def synpassflux(wl, flux, band):
    # Calculate synthetic magnitudes
    n_bands = len(band)

    filt = np.genfromtxt("/Users/jonatanselsing/github/iPTF16geu/data/passbands/%s"%band)
    lamb_T, T = filt[:,0], filt[:,1]
    f = pyphot.Filter(lamb_T, T, name=band, dtype='photon', unit='Angstrom')
    fluxes = f.get_flux(wl, flux, axis=0)
    synmags = -2.5 * np.log10(fluxes) - f.AB_zero_mag
    cwav = np.mean(lamb_T)
    cwave = (float(max(lamb_T[T > np.percentile(T, 10)] - cwav)), float(cwav - min(lamb_T[T > np.percentile(T, 10)])))
    synmag_flux = ((synmags*u.ABmag).to((u.erg/(u.s * u.cm**2 * u.AA)), u.spectral_density(cwav * u.AA))).value
    return synmag_flux, cwav, cwave, synmags



def main():

    # root_dir = "/Users/jonatanselsing/Work/work_rawDATA/FRB/FRB180930/final/"
    # root_dir = "/Users/jonatanselsing/Work/work_rawDATA/XSGRB/GRB161023A/final/"
    root_dir = "/Users/jonatanselsing/Work/work_rawDATA/Crab_Pulsar/final/"

    # root_dir = "/Users/jonatanselsing/Work/work_rawDATA/STARGATE/GRB171205A/final/"
    bin_f = 10
    # root_dir = "/Users/jonatanselsing/Work/work_rawDATA/SLSN/SN2018bsz/final/"

    # root_dir = "/Users/jonatanselsing/Work/work_rawDATA/XSGW/AT2017GFO/final/"
    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9", "OB10", "OB11", "OB12", "OB13", "OB14", "OB15", "OB16", "OB17", "OB18"]
    # OBs = ["OB1", "OB2", "OB3", "OB4", "OB5"]
    OBs = ["OB9", "OB9_pip"]
    z = 0#3.07
    colors = sns.color_palette("viridis", len(OBs))
    # colors = sns.color_palette("Blues_r", len(OBs))


    wl_out, flux_out, error_out = 0, 0, 0
    for ii, kk in enumerate(OBs):

        off = 0#(len(OBs) - ii) * 2e-17
        mult = 1.0
        # if kk == "OB9_pip":

        ############################## OB ##############################
        f = fits.open(root_dir + "UVB%s.fits"%kk)
        wl = 10. * f[1].data.field("WAVE").flatten()


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

        pl.plot(wl_plot[::1]/(1+z), off + mult*flux[::1], lw=0.3, color = "black", alpha = 0.2, rasterized = True)
        wl_stitch, flux_stitch, error_stitch = wl_plot, flux, error
        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), round_up_to_odd(bin_f))
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        # np.savetxt(root_dir+"UVB%s_bin150.dat"%kk, list(zip(wl_out, flux_out, error_out)))

        # pl.plot(b_wl, off + mult*b_f, color="firebrick", linestyle="steps-mid")
        # pl.plot(b_wl/(1+z), off + mult*b_e, color="black", linestyle="dashed")

        # pl.plot(wl_plot/(1+z), off + mult*medfilt(error , 251), linestyle="dashed", color="black", rasterized = True)
        pl.plot(b_wl/(1+z), off + mult*b_f, color= colors[ii], linestyle="steps-mid", rasterized = True)

        # pl.fill_between(wl_plot, off + mult*medfilt(flux, 1) - mult*medfilt(error, 1), off + mult*medfilt(flux, 1) + mult*medfilt(error, 1), color = "grey", alpha = 0.5, rasterized = True)
        # f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        # f.writeto(root_dir + "final/UVB%s.fits"%kk, clobber = True)
        max_v, min_v = max(medfilt(flux, 101)), min(medfilt(flux, 101))
        f = fits.open(root_dir + "VIS%s.fits"%kk)

        wl = 10. * f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()

        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(wl)


        mask_wl = (wl > 5650) & (wl < 10000)
        mask_qual = ~q.astype("bool")

        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]

        pl.plot(wl_plot[::1]/(1+z), off + mult*flux[::1], lw=0.3, color = "black", alpha = 0.2, rasterized = True)
        wl_stitch, flux_stitch, error_stitch = np.concatenate([wl_stitch,wl_plot]), np.concatenate([flux_stitch,flux]), np.concatenate([error_stitch, error])
        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), round_up_to_odd(bin_f))
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        # np.savetxt(root_dir+"VIS%s_bin150.dat"%kk, list(zip(wl_out, flux_out, error_out)))
        # pl.plot(b_wl/(1+z), off + mult*b_e, color="black", linestyle="dashed")

        # pl.plot(wl_plot/(1+z), off + mult*medfilt(error , 251), linestyle="dashed", color="black", rasterized = True)
        pl.plot(b_wl/(1+z), off + mult*b_f, color= colors[ii], linestyle="steps-mid", rasterized = True)

        # pl.fill_between(wl_plot, off + mult*medfilt(flux, 1) - mult*medfilt(error, 1), off + mult*medfilt(flux, 1) + mult*medfilt(error, 1), color = "grey", alpha = 0.5, rasterized = True)

        # wl_out, flux_out, error_out = np.concatenate((wl_out, b_wl)), np.concatenate((flux_out, b_f)), np.concatenate((error_out, b_e))
        # f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        # f[1].data["FLUX"] = f[1].data["FLUX"] / t
        # f[1].data["ERR"] = f[1].data["ERR"] / t
        # f.writeto(root_dir + "final/VIS%s.fits"%kk, clobber = True)
        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(min(medfilt(flux, 101)), min_v)

        f = fits.open(root_dir + "NIR%s.fits"%kk)
        wl = 10. * f[1].data.field("WAVE").flatten()
        q = f[1].data.field("QUAL").flatten()

        try:
            t = f[1].data.field("TRANS").flatten()
        except:
            t = np.ones_like(wl)


        mask_wl = (wl > 10500) & (wl < 25000) & (t > 0.3)
        mask_qual = ~q.astype("bool")
        flux = interpolate.interp1d(wl[mask_qual], f[1].data.field("FLUX").flatten()[mask_qual], bounds_error=False, fill_value=0)
        error = interpolate.interp1d(wl[mask_qual], f[1].data.field("ERR").flatten()[mask_qual], bounds_error=False, fill_value=0)
        wl_plot = wl[mask_wl]
        flux = flux(wl_plot) / t[mask_wl]
        error = error(wl_plot) / t[mask_wl]


        pl.plot(wl_plot[::1]/(1+z), off + mult*flux[::1], lw=0.3, color = "black", alpha = 0.2, rasterized = True)
        wl_stitch, flux_stitch, error_stitch = np.concatenate([wl_stitch, wl_plot]), np.concatenate([flux_stitch,flux]), np.concatenate([error_stitch, error])

        b_wl, b_f, b_e, b_q = bin_spectrum(wl_plot, flux, error, np.zeros_like(flux).astype("bool"), round_up_to_odd(bin_f/3))
        wl_out, flux_out, error_out = b_wl, b_f, b_e
        # np.savetxt(root_dir+"NIR%s_bin150.dat"%kk, list(zip(wl_out, flux_out, error_out)))

        # pl.plot(b_wl, off + mult*b_f, color="firebrick", linestyle="steps-mid")
        # pl.plot(b_wl/(1+z), off + mult*b_e, color="black", linestyle="dashed")

        # pl.plot(wl_plot/(1+z), off + mult*medfilt(error , 251), linestyle="dashed", color="black")
        pl.plot(b_wl/(1+z), off + mult*b_f, color= colors[ii], linestyle="steps-mid", rasterized = True, label = "XSH: %s"%str(f[0].header['DATE-OBS']))

        # pl.fill_between(wl_plot, off + mult*medfilt(flux, 51) - mult*medfilt(error, 51), off + mult*medfilt(flux, 51) + mult*medfilt(error, 51), color = "grey", alpha = 0.5, rasterized = True)
        # f[1].data["WAVE"] = wl * (1 + f[0].header['HIERARCH ESO QC VRAD BARYCOR']/3e5)
        # f[1].data["FLUX"] = f[1].data["FLUX"] / t
        # f[1].data["ERR"] = f[1].data["ERR"] / t
        # f.writeto(root_dir + "final/NIR%s.fits"%kk, clobber = True)
        max_v, min_v = max(max(medfilt(flux, 101)), max_v), min(min(medfilt(flux, 101)), min_v)
        max_std = np.nanstd(flux)
        # print(max_std)
        pl.axvspan(5550, 5650, color="grey", alpha=0.2)
        pl.axvspan(10000, 10500, color="grey", alpha=0.2)
        pl.axvspan(12600, 12800, color="grey", alpha=0.2)
        pl.axvspan(13500, 14500, color="grey", alpha=0.2)
        pl.axvspan(18000, 19500, color="grey", alpha=0.2)
        # print(max(f[0].header['HIERARCH ESO TEL AMBI FWHM END'], f[0].header['HIERARCH ESO TEL AMBI FWHM START']))
        # print(np.diff(wl_out), np.median(np.diff(wl_out)))
        # np.savetxt("UVBVIS_bin30OB3.dat", list(zip(wl_out, flux_out, error_out)))






        # pl.ylim(1.1 * min_v, 1.2 * max_v)
        # pl.ylim(-1e-18, 1.2 * max_v)
        wl = np.arange(min(wl_stitch), max(wl_stitch), np.median(np.diff(wl_stitch)))
        f = interpolate.interp1d(wl_stitch, flux_stitch)
        g = interpolate.interp1d(wl_stitch, error_stitch)

        np.savetxt(root_dir+"%s_stitched.dat"%kk, list(zip(wl, f(wl), g(wl))), fmt='%1.2f %1.4e %1.4e')

        # pl.ylim(-1e-19, 5e-16)
        # pl.ylim(1e-16, 1e-11)

        pl.xlim(2500, 20000)
        pl.xlabel(r"Observed wavelength  [$\mathrm{\AA}$]")
        pl.ylabel(r'Flux density [$\mathrm{erg} \mathrm{s}^{-1} \mathrm{cm}^{-1} \mathrm{\AA}^{-1}$]')
        # leg = pl.legend(loc=1)

        # set the linewidth of each legend object
        # for legobj in leg.legendHandles:
        #     legobj.set_linewidth(2.0)
        # pl.semilogy()
        # pl.loglog()




        pl.axhline(0, linestyle="dashed", color="black")
        # pl.legend()
        # pl.savefig(root_dir + "%s.pdf"%kk)
        # pl.show()

        # pl.clf()


    from dust_extinction.parameter_averages import F04
    # initialize the model
    ext = F04(Rv=3.1)
    import astropy.units as u
    gd71 = np.genfromtxt("/Users/jonatanselsing/Work/work_rawDATA/Crab_Pulsar/fGD71.dat")
    redd = ext.extinguish(gd71[:, 0]*u.angstrom, Ebv=0.2580)
    ext_corr = gd71[:, 1]#/redd
    pl.plot(gd71[:, 0], ext_corr)




    # if True == True:
    #     OBs = ["OB1"]
    #     for ii, OB in enumerate(OBs):


    #         # passbands = [
    #             # "FORS2_U.dat", "FORS2_B.dat", "FORS2_V.dat", "FORS2_R.dat", "FORS2_I.dat", "2MASS_J.dat", "2MASS_H.dat", "2MASS_Ks.dat"
    #             # ]
    #         passbands = ["FORS2_V.dat", "FORS2_R.dat", "FORS2_I.dat", "2MASS_J.dat", "2MASS_H.dat"]
    #         photometry = pd.read_csv("/Users/jonatanselsing/Work/work_rawDATA/Crab_Pulsar/Crab_phot.csv")

    #         for pp, ss in enumerate(passbands):
    #             pass_name = ss.split("_")[-1].split(".")[0]
    #             meas_mag = photometry.loc[photometry['Bandpass'] == pass_name]


    #             # print(meas_mag.keys())

    #             # exit()

    #             wl, flux, error = wl, f(wl), g(wl)


    #             synmag_flux, cwav, cwave, synmag = synpassflux(wl, flux, ss)
    #             synmag_error, cwav, cwave, synmag_err = synpassflux(wl, error, ss)
    #             _, _, _, synmag_err_up = synpassflux(wl, flux + error, ss)
    #             _, _, _, synmag_err_low = synpassflux(wl, flux - error, ss)
    #             e_u = (synmag - synmag_err_up)
    #             e_l = (synmag_err_low - synmag)
    #             pl.errorbar(cwav, synmag_flux, xerr = [[cwave]], yerr = synmag_error,  fmt = 'o', zorder = 10, ms = 5, elinewidth=1.7, label = "%s = %s$^{+%s}_{-%s}$"%(pass_name, np.around(synmag, 1), np.around(e_u, 1), np.around(e_l, 1)))

    #             if meas_mag.shape[0] > 0:
    #                 meas_flux = ((meas_mag["CrabPulsar+Knot"].values*u.ABmag).to((u.erg/(u.s * u.cm**2 * u.AA)), u.spectral_density(cwav * u.AA))).value
    #                 meas_flux_up =  (((meas_mag["CrabPulsar+Knot"].values + meas_mag["CrabPulsar+Knot_e"].values)*u.ABmag).to((u.erg/(u.s * u.cm**2 * u.AA)), u.spectral_density(cwav * u.AA))).value
    #                 meas_flux_do =  (((meas_mag["CrabPulsar+Knot"].values - meas_mag["CrabPulsar+Knot_e"].values)*u.ABmag).to((u.erg/(u.s * u.cm**2 * u.AA)), u.spectral_density(cwav * u.AA))).value

    #                 pl.errorbar(cwav, meas_flux, xerr = [[cwave]], yerr = [[(meas_flux_do, meas_flux_up)]],  fmt = 'o', ms = 15)
    # except:
        # pass
    # pl.loglog()
    # pl.legend()

    # pl.savefig(root_dir + "%s_all.pdf"%kk)

    # pl.ylim(-1e-18, 0.25e-16)
    # pl.savefig(root_dir + "%s_all.pdf"%kk)
    pl.semilogy()

    pl.show()


if __name__ == '__main__':
    main()
