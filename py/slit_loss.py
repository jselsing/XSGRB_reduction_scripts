import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
import time
from astropy.modeling import models, fitting
from astropy.io import fits
from util import *
from matplotlib.backends.backend_pdf import PdfPages
import glob
from scipy import optimize
from numpy.polynomial import chebyshev
# beta = 4.765
# slit_corr = []
# seeings = np.arange(0.5, 3, 0.01)
# slit_width = 0.9 # arcsec



def get_slitloss(seeing_fwhm, slit_width):
    # Generate image parameters
    img_size = 100

    arcsec_to_pix = img_size/5 # Assumes a 5 arcsec image
    slit_width_pix = arcsec_to_pix * slit_width
    seeing_pix = arcsec_to_pix * seeing_fwhm

    x, y = np.mgrid[:img_size, :img_size]
    source_pos = [int(img_size/2), int(img_size/2)]


    # Simulate source Moffat
    beta = 4.765
    gamma = seeing_pix / (2 * np.sqrt(2**(1/beta) - 1))
    source = models.Moffat2D.evaluate(x, y, 1, source_pos[0], source_pos[1], gamma, beta)

    # Define slit mask
    mask = slice(int(source_pos[1] - slit_width_pix/2),int(source_pos[1] + slit_width_pix / 2))
    sl = np.trapz(np.trapz(source)) / np.trapz(np.trapz(source[:, mask]))

    return sl



def Moffat1D(x, amplitude, x_0, fwhm):
    beta = 4.765
    gamma = fwhm / (2 * np.sqrt(2**(1/beta) - 1))
    return models.Moffat1D.evaluate(x, amplitude, x_0, gamma, 4.765)




if __name__ == '__main__':

    #Files
    root_dir = "/Users/jselsing/Work/work_rawDATA/XSGW/SSS17a/"
    xsgrbobject = glob.glob(root_dir+'reduced_data/OB*_tell/*/*/*.fits')
    xsgrbobject = glob.glob(root_dir+'*.fits')
    # Use number
    idx = 0
    arms = ["UVB", "VIS", "NIR"]
    # arms = ["NIR"]
    OBs = ["OB1", "OB2", "OB3", "OB4", "OB5", "OB6", "OB7", "OB8", "OB9", "OB10"]
    OBs = ["OB10"]
    idx_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for ll in arms:
        for xx, pp in enumerate(OBs):

            tell_file_2D = [kk for kk in xsgrbobject if ll+pp+"skysub.fits" in kk][idx_t[xx]]
            print('Working on object: '+tell_file_2D)

            tell_file = fits.open(tell_file_2D)
            header = tell_file[0].header
            namesplit = tell_file_2D.split("/")

            OB = pp#namesplit[-4][:3]

            arm = tell_file[0].header["HIERARCH ESO SEQ ARM"]
            filenam = "".join(("".join(namesplit[-2].split(":")).split(".")))
            file_name = arm + OB + "TELL" #+ filenam
            # print(file_name)
            # exit()


            haxis = 10.*(((np.arange(header['NAXIS1'])) - header['CRPIX1'])*header['CDELT1']+header['CRVAL1'])
            vaxis =  ((np.arange(header['NAXIS2'])) - header['CRPIX2'])*header['CDELT2'] + header['CRVAL2']
            if arm == "UVB":
                slit_width = 1.0
            elif arm == "VIS" or arm == "NIR":
                slit_width = 0.9

            # flux = tell_file[0].data
            # error = tell_file[1].data
            # bpmap = tell_file[2].data


            # # # Bin spectrum
            # bin_length = int(len(haxis) / 100)
            # bin_flux, bin_error = bin_image(flux, error, bpmap, bin_length, weight = False)
            # bin_haxis = 10.*(((np.arange(header['NAXIS1']/bin_length)) - header['CRPIX1'])*header['CDELT1']*bin_length+header['CRVAL1'])
            # # Cutting edges of image. Especially importnant for nodding combinations, due to the negative signals
            # width = int(len(vaxis)/4)

            # # Inital parameter guess
            # p0 = [5e-13, -2.5, 0.9]
            # # Parameter containers
            # amp, cen, fwhm = np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis)
            # eamp, ecen, efwhm = np.zeros_like(bin_haxis), np.zeros_like(bin_haxis), np.zeros_like(bin_haxis)

            # # Loop though along dispersion axis in the binned image and fit a Moffat
            # x = np.arange(min(vaxis[width:-width]), max(vaxis[width:-width]), 0.01)
            # for ii, kk in enumerate(bin_haxis):
            #     # print(ii)
            #     try:
            #         popt, pcov = optimize.curve_fit(Moffat1D, vaxis[width:-width], bin_flux[:, ii][width:-width], p0 = p0, maxfev = 5000)
            #         guess_par = [popt[0]] + p0[1:]
            #         # pl.plot(x, Moffat1D(x, *popt), label="Best-fit")
            #         # pl.plot(x, Moffat1D(x, *guess_par), label="Fit guess parameters")
            #         # pl.errorbar(vaxis[width:-width], bin_flux[:, ii][width:-width], yerr=bin_error[:, ii][width:-width], fmt=".k", capsize=0, elinewidth=0.5, ms=3)
            #         # pl.show()
            #     except:
            #         print("Fitting error at binned image index: "+str(ii)+". Replacing fit value with guess and set fit error to 10^10")
            #         popt, pcov = p0, np.diag(1e10*np.ones_like(p0))

            #     amp[ii], cen[ii], fwhm[ii] = popt[0], popt[1], popt[2]
            #     eamp[ii], ecen[ii], efwhm[ii] = np.sqrt(np.diag(pcov)[0]), np.sqrt(np.diag(pcov)[1]), np.sqrt(np.diag(pcov)[2])

            # # Mask elements too close to guess, indicating a bad fit.
            # ecen[:1] = 1e10
            # ecen[-1:] = 1e10

            # ecen[abs(cen/ecen) > abs(np.nanmean(cen/ecen)) + 5*np.nanstd(cen/ecen)] = 1e10
            # ecen[abs(amp - p0[0]) < p0[0]/100] = 1e10
            # ecen[abs(cen - p0[1]) < p0[1]/100] = 1e10
            # ecen[abs(fwhm - p0[2]) < p0[2]/100] = 1e10


            # # Remove the 5 highest S/N pixels
            # ecen[np.argsort(fwhm/efwhm)[-5:]] = 1e10

            # # Fit polynomial for center and iteratively reject outliers
            # pol_degree = [3, 2, 2]
            # std_resid = 5
            # while std_resid > 0.5:
            #     idx = np.isfinite(cen) & np.isfinite(ecen)
            #     fitcen = chebyshev.chebfit(bin_haxis[idx], cen[idx], deg=pol_degree[0], w=1/ecen[idx])
            #     resid = cen - chebyshev.chebval(bin_haxis, fitcen)
            #     avd_resid, std_resid = np.median(resid[ecen != 1e10]), np.std(resid[ecen != 1e10])
            #     mask = (resid < avd_resid - std_resid) | (resid > avd_resid + std_resid)
            #     ecen[mask] = 1e10
            # fitcenval = chebyshev.chebval(haxis, fitcen)
            # # Plotting for quality control
            # fig, (ax1, ax2, ax3) = pl.subplots(3,1, figsize=(14, 14), sharex=True)

            # ax1.errorbar(bin_haxis, cen, yerr=ecen, fmt=".k", capsize=0, elinewidth=0.5, ms=7)
            # ax1.plot(haxis, fitcenval)
            # vaxis_range = max(vaxis) - min(vaxis)
            # ax1.set_ylim((min(vaxis[width:-width]), max(vaxis[width:-width])))
            # ax1.set_ylabel("Profile center / [arcsec]")
            # ax1.set_title("Quality test: Center estimate")
            # # fwhmma-clip outliers in S/N-space
            # efwhm[ecen == 1e10] = 1e10
            # efwhm[fwhm < 0.01] = 1e10

            # fitfwhm = chebyshev.chebfit(bin_haxis, fwhm, deg=pol_degree[1], w=1/efwhm)
            # fitfwhmval = chebyshev.chebval(haxis, fitfwhm)
            # # Ensure positivity
            # fitfwhmval[fitfwhmval < 0.1] = 0.1

            # # Plotting for quality control
            # ax2.errorbar(bin_haxis, fwhm, yerr=efwhm, fmt=".k", capsize=0, elinewidth=0.5, ms=7)
            # ax2.plot(haxis, fitfwhmval)
            # ax2.set_ylim((0, 2))
            # ax2.set_ylabel("Profile FWHM / [arcsec]")
            # ax2.set_title("Quality test: Profile Moffat width estimate")


            # # Amplitude replaced with ones
            # from scipy import interpolate, signal

            # eamp[ecen == 1e10] = 1e10
            # amp[amp < 0] = 1e-20
            # amp = signal.medfilt(amp, 5)
            # mask = ~(eamp == 1e10)
            # f = interpolate.interp1d(bin_haxis[mask], amp[mask], bounds_error=False, fill_value="extrapolate")
            # fitampval = f(haxis)
            # fitampval[fitampval <= 0] = 1e-20#np.nanmean(fitampval[fitampval > 0])

            # # Plotting for quality control
            # ax3.errorbar(bin_haxis, amp, fmt=".k", capsize=0, elinewidth=0.5, ms=5)
            # ax3.plot(haxis, fitampval)
            # ax3.set_ylabel(r"Profile amplitude / [erg s$^{-1}$ cm$^{-1}$ $\AA^{-1}$]")
            # ax3.set_title("Quality test: Profile amplitude estimate")
            # ax3.set_xlabel(r"Wavelength / [$\mathrm{\AA}$]")
            # fig.subplots_adjust(hspace=0)
            # fig.savefig(root_dir+file_name + "_fitpars.pdf")
            # pl.close(fig)









            # Theoretical seeing
            # seeing = fitfwhmval[int(len(fitfwhmval)/2)]
            seeing = max(header["HIERARCH ESO TEL AMBI FWHM START"], header["HIERARCH ESO TEL AMBI FWHM END"])
            # print(seeing)
            # haxis_0 = haxis[int(len(fitfwhmval)/2)]
            # print(haxis_0)
            haxis_0 = 5000
            S0 = seeing / haxis_0**(-1/5)
            seeing_theo = S0 * haxis**(-1/5)
            # pl.plot(haxis, seeing_theo)
            # pl.show()

            # exit()
            # # Calculating slit-losses based on fit-width
            sl = [0]*len(seeing_theo)
            for ii, kk in enumerate(seeing_theo):
                sl[ii] = get_slitloss(kk, slit_width)

            # # Saving to .dat file
            dt = [("Wavelength", np.float64), ("SeeingFWHM", np.float64), ("slitcorr", np.float64)]
            data = np.array(list(zip(haxis, seeing_theo, sl)), dtype=dt)
            np.savetxt(root_dir+file_name + "_SLITCORR.dat", data, header="Wavlength/[Å] seeing/[FWHM] slitloss_correction", fmt = ['%1.2f', '%1.2f', '%1.3f'])
            # exit()






