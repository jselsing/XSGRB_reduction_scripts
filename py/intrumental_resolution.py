# -*- coding: utf-8 -*-



import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import seaborn as sns; sns.set_style('ticks')

# Adding ppxf path
import sys
sys.path.append('/Users/jselsing/Work/Pythonlibs/ppxf/')
import ppxf
import ppxf_util as util
from scipy.special import wofz, erf
from scipy.optimize import curve_fit


# import seaborn; seaborn.set_style('ticks')
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import splrep,splev
from time import clock
from scipy import interpolate


def voigt_base(x, amp=1, cen=0, sigma=1, gamma=0):
    """1 dimensional voigt function.
    see http://en.wikipedia.org/wiki/Voigt_profile
    """
    z = (x-cen + 1j*gamma)/ (sigma*np.sqrt(2.0))
    return amp * wofz(z).real / (sigma*np.sqrt(2*np.pi))


def multi_voigt(x, *params):
    # Multiple voigt-profiles for telluric resolution estimate
    sigma = params[0]
    gamma = params[1]
    c = params[2]
    a = params[3]

    multivoigt = 0
    for ii in range(4, int(4 + len(params[4:])/2)):
        # print(params[ii])
        if gamma < 0:
            sigma = 1e9

        multivoigt += voigt_base(x, params[ii], params[int(len(params[4:])/2 + ii)], sigma, gamma)
    return multivoigt + c + a * x


if __name__ == '__main__':
    from astropy.io import fits
    import glob
    # import matplotlib.pyplot as pl
    import numpy as np
    from scipy.interpolate import splrep, splev, interp1d

    #Files
    root_dir = "/Users/jselsing/Work/work_rawDATA/STARGATE/GRB180205A/"
    # root_dir = "/Users/jselsing/Work/work_rawDATA/XSGW/AT2017GFO/"
    xsgrbobject = glob.glob(root_dir+"reduced_data/*_TELL/*/*/*.fits")

    tell_file = [kk for kk in xsgrbobject if "IDP" in kk]
    # print(tell_file)
    tell_file_2D = [kk for kk in xsgrbobject if "SLIT_FLUX_MERGE2D" in kk]
    # print(tell_file_2D)
    Rs = []
    fwhms = []
    ambifwhms = []
    Rse = []
    fwhmse = []
    arms = []
    grbs = []
    for kk, ii in enumerate(tell_file):

        print('Working on object: '+ii)
        tell_file = fits.open(ii)


        arm = tell_file[0].header["HIERARCH ESO SEQ ARM"]
        ob = ii.split("/")[-4]
        # print(ob)
        # exit()

        grb = ii.split("/")[-2]

        # wl = 10.*((np.arange(tell_file[0].header['NAXIS1']) - tell_file[0].header['CRPIX1'])*tell_file[0].header['CDELT1']+tell_file[0].header['CRVAL1'])
        # IDP files
        wl = 10*tell_file[1].data.field("WAVE").flatten()



        flux = tell_file[1].data.field("FLUX").flatten()
        err = tell_file[1].data.field("ERR").flatten()
        bpmap = tell_file[1].data.field("QUAL").flatten()
        # bpmap = np.zeros_like(bpmap) 
        # flux = tell_file[0].data#*response
        # err = tell_file[1].data#*response
        # bpmap = tell_file[2].data
        # bpmap = np.zeros_like(tell_file[2].data)


        wl_temp = wl[~(bpmap.astype("bool"))][200:-100]
        flux_temp = flux[~(bpmap.astype("bool"))][200:-100]
        err_temp = err[~(bpmap.astype("bool"))][200:-100]
        f = interpolate.interp1d(wl_temp, flux_temp, kind="nearest", fill_value="extrapolate")
        flux = f(wl)
        f = interpolate.interp1d(wl_temp, err_temp, kind="nearest", fill_value="extrapolate")
        err = f(wl)

        if arm == "UVB":

            mask2 = (wl > wl[int(len(wl)/2)] - 100) & (wl < wl[int(len(wl)/2)] + 100)

        elif arm == "VIS":
            # mask = (wl > 6881) & (wl < 6922)
            mask = (wl > 8241) & (wl < 8287)
            mask2 = (wl > wl[int(len(wl)/2)] - 100) & (wl < wl[int(len(wl)/2)] + 100)
            cens = [8.24347741e+03, 8.25277232e+03, 8.25654085e+03, 8.25972744e+03, 8.26346878e+03, 8.27206819e+03, 8.27438225e+03, 8.27665085e+03, 8.27963344e+03, 8.28206007e+03]
            amps = len(cens) * [-0.1]
            # cens = [6.88392564e+03, 6.88585686e+03, 6.88683982e+03, 6.88904597e+03, 6.89000132e+03, 6.89247384e+03, 6.89341075e+03, 6.89613979e+03, 6.89706733e+03, 6.90005809e+03, 6.90097910e+03, 6.90422797e+03, 6.90513024e+03, 6.90863564e+03, 6.90954103e+03, 6.91332001e+03, 6.91421144e+03, 6.91822776e+03, 6.91913232e+03]

        elif arm == "NIR":
            mask = (wl > 17475) & (wl < 17645)
            mask2 = (wl > wl[int(len(wl)/2)] - 500) & (wl < wl[int(len(wl)/2)]+ 500)
            cens = [17510, 17546, 17563, 17569, 17604, 17620, 17625]
            amps = len(cens)  * [-0.3]
            cens = [17510, 17546, 17563, 17569, 17604, 17620, 17625]
             # cens = [17510, 17546, 17550, 17563, 17569, 17604, 17620, 17625, 17654, 17676, 17691, 17702]

        tell_file2D = fits.open(tell_file_2D[kk])



        min_idx, max_idx = min(*np.where(mask)), max(*np.where(mask))
        v_len = np.shape(tell_file2D[0].data)[0]




        # profile = np.median(tell_file2D[0].data[int(v_len/3):int(-v_len/3), min_idx:max_idx], axis= 1)
        profile = np.nanmedian(tell_file2D[0].data[:, min_idx:max_idx], axis= 1)
        profile[profile < 0] = 0

        # xarr = np.arange(len(profile))

        xarr = (np.arange(tell_file2D[0].header['NAXIS2']) - tell_file2D[0].header['CRPIX2'])*tell_file2D[0].header['CDELT2']+tell_file2D[0].header['CRVAL2']


        fig, ax1 = pl.subplots()
        ax2 = ax1.twinx().twiny()
        p0 = [max(profile), np.median(xarr), 5, 0]
        # popt, pcuv = curve_fit(voigt_base, xarr, profile, p0=p0)
        try:
            popt, pcuv = curve_fit(voigt_base, xarr, profile, p0=p0)
        except:
            continue

        x = np.arange(min(xarr), max(xarr), 0.01)

        ax2.plot(xarr, profile)
        ax2.plot(x, voigt_base(x, *popt))
        ax2.set_xlabel(r"Extent (\")")
        ax2.set_ylabel(r'Flux density [erg s$^{-1}$ cm$^{-1}$ $\AA^{-1}$]')
        # pl.ylim(0, 5e-14)
        # pl.show()
        fwhm_g, fwhm_l = 2.35 * popt[2], 2*popt[3]
        fwhm_g_var, fwhm_l_var = 2.35 * pcuv[2, 2], 2*pcuv[3, 3]
        fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)
        dfdl = 0.5346 - 0.5 * ((0.2166 * fwhm_l**2 + fwhm_g**2) ** (-3/2)) *(2 * 0.2166 * fwhm_l)
        dfdg = - 0.5 * ((0.2166 * fwhm_l**2 + fwhm_g**2) ** (-3/2)) *(2 * fwhm_g)
        fwhm_err = np.sqrt((dfdl**2) * (fwhm_g_var) + (dfdg**2) * (fwhm_l_var))
        seeing_fwhm = fwhm#*tell_file2D[0].header["CD2_2"]
        # print(ii, tell_file_2D[kk], seeing_fwhm)
        seeing_fwhm_err = fwhm_err#*tell_file2D[0].header["CD2_2"]


        # Get first extractions
        tell_file2D[1].data[tell_file2D[0].data == 0] = 1e50
        profile_ext = np.tile(profile, (np.shape(tell_file2D[0].data[:, min_idx:max_idx])[1], 1)).T
        denom = np.sum((profile_ext**2. / tell_file2D[1].data[:, min_idx:max_idx]), axis=0)
        spectrum = np.sum(profile_ext * tell_file2D[0].data[:, min_idx:max_idx] / tell_file2D[1].data[:, min_idx:max_idx], axis=0) / denom
        errorspectrum = np.sqrt(1 / denom)
        # spectrum, errorspectrum = flux[min_idx:max_idx], err[min_idx:max_idx]
        # print(np.shape(tell_file2D[0].data[:, min_idx:max_idx]))

        # ax1.plot(wl[min_idx:max_idx], spectrum)
        # ax1.plot(wl[min_idx:max_idx], errorspectrum)
        # pl.show()

        # exit()

        p0 =  [0.2, 0.01, max(spectrum), -0.001] + amps + cens
        # popt, pcuv = curve_fit(multi_voigt, wl[min_idx:max_idx], spectrum, p0=p0)
        # try:
        popt, pcuv = curve_fit(multi_voigt, wl[min_idx:max_idx], spectrum, p0=p0)
        # except:
            # continue
        midwl = np.median(wl[min_idx:max_idx])


        fwhm_g, fwhm_l = 2.35 * popt[0], 2*popt[1]
        fwhm_g_var, fwhm_l_var = 2.35 * pcuv[0, 0], 2*pcuv[1, 1]
        fwhm = 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)
        dfdl = 0.5346 - 0.5 * ((0.2166 * fwhm_l**2 + fwhm_g**2) ** (-3/2)) *(2 * 0.2166 * fwhm_l)
        dfdg = - 0.5 * ((0.2166 * fwhm_l**2 + fwhm_g**2) ** (-3/2)) *(2 * fwhm_g)
        fwhm_err = np.sqrt((dfdl**2) * (fwhm_g_var) + (dfdg**2) * (fwhm_l_var))


        R = midwl / (fwhm)
        Rerr =   R - midwl /((fwhm + fwhm_err))
        x = np.arange(min(wl[min_idx:max_idx]), max(wl[min_idx:max_idx]), 0.01)
        ax1.plot(x, multi_voigt(x, *popt), label="R = "+str(int(np.around(R, decimals = -2))) + " +- " + str(int(np.around(Rerr, decimals = -2))))
        # ax1.plot(x, multi_voigt(x, *p0), label="Guess")
        # print(multi_voigt(x, *p0))
        # print(wl[min_idx:max_idx], spectrum)
        ax1.plot(wl[min_idx:max_idx], spectrum, label="Seeing FWHM = "+str(np.around(seeing_fwhm, decimals = 2)) + " +- " + str(np.around(seeing_fwhm_err, decimals = 2)))
        ax1.set_xlabel(r"Wavelength / [$\mathrm{\AA}$]")
        ax1.set_ylabel(r'Flux density [erg s$^{-1}$ cm$^{-1}$ $\AA^{-1}$]')
        ax1.set_ylim((min(spectrum), max(spectrum)*1.10))
        ax1.legend(loc=1)
        outname = "/".join(ii.split("/")[:-1])
        # print(outname)
        print(root_dir+ob+arm+"_resolution_all.pdf")
        pl.savefig(root_dir+ob+arm+"_resolution_all.pdf")
        pl.close()
        # pl.show()

        Rs.append(int(np.around(R, decimals = -2)))
        fwhms.append(np.around(seeing_fwhm, decimals = 2))

        Rse.append(int(np.around(Rerr, decimals = -2)))
        fwhmse.append(np.around(seeing_fwhm_err, decimals = 2))
        arms.append(arm)
        grbs.append(grb)


        # Theoretical slitloss based on DIMM seeing
        # try:
        #     seeing = abs(tell_file2D[0].header["SEEING"])
        # except:
        seeing = max(abs(tell_file2D[0].header["HIERARCH ESO TEL AMBI FWHM START"]), abs(tell_file2D[0].header["HIERARCH ESO TEL AMBI FWHM END"]))

        # Correct seeing for airmass
        airmass = np.average([tell_file2D[0].header["HIERARCH ESO TEL AIRM START"], tell_file2D[0].header["HIERARCH ESO TEL AIRM END"]])
        seeing_airmass_corr = seeing * (airmass)**(3/5)
        # print(seeing_airmass_corr)

        # Theoretical wavelength dependence
        # haxis_0 = 5000 # Å, DIMM center
        # S0 = seeing_airmass_corr / haxis_0**(-1/5)
        # seeing_theo = S0 * midwl**(-1/5)
        # print(seeing_theo)
        ambifwhms.append(seeing_airmass_corr)


    np.savetxt(root_dir+"Rfwhm.dat", list(zip(fwhms, fwhmse, Rs, Rse, arms, grbs, ambifwhms)), fmt=('%s %s %s %s %s %s %s'), header="fwhms fwhmse Rs Rse arms grbs ambifwhms")
