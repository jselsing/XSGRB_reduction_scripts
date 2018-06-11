# -*- coding: utf-8 -*-


# from matplotlib import rc_file
# rc_file('/Users/jselsing/Pythonlibs/plotting/matplotlibstyle.rc')
import matplotlib.pyplot as plt

from xsh_norm import methods_auto
from pylab import pause
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns; sns.set_style('ticks')
cmap = [sns.xkcd_rgb["denim blue"], sns.xkcd_rgb["medium green"], sns.xkcd_rgb["pale red"]]


__all__ = ["xsh_norm"]


def binning1d(array,bin):
    """
    Used to bin low S/N 1D  response data from xshooter.
    Calculates the biweighted mean (a la done in Kriek'10). 
    Returns binned 1dimage
    """
#    ;--------------
    s=len((array))
    outsize=s/bin
    res = np.zeros((outsize))
    for i in np.arange(0,s-(bin+1),bin):
             res[((i+bin)/bin-1)] = np.sum(array[i:i+bin])/bin
    return res



def f8(seq):
    seen = set()
    return np.array([i for i, x in enumerate(seq) if x not in seen and not seen.add(x)])


class xsh_norm(object):
    """
    A normalisation suite.

    Key-bindings
      'left-mouse-button'
              place spline point

      'enter'
              1. sort the continuum-point-array according to the x-values
              2. fit a spline and evaluate it in the wavelength points
              3. plot the continuum
      'n'

              Apply normalisation
      'w'

              Write to file

      'y'

              Insert evenly spaced points - every 100'th pixel

      'u'
              Insert evenly spaced points - every 50'th pixel

      'd'

              Delete point under mouse

      't'

              Filter points by low-order chebyshev fitting and clipping values of high sigma iteratively until continuum is found

      'm'

              Mask values (spectrum - continuum fit) > 2 sigma

      'i'
              Run through the process y-t-enter-mask until the median value of the realisations converge
    """


    def __init__(self, w, f, e, bpmap, w_ori, f_ori, e_ori, filename):
        self.showverts = True
        self.epsilon = 10  # max pixel distance to count as a vertex hit
        self.pointx = []
        self.pointy = []
        self.pointyerr = []
        self.linewidth = 0.5
        self.linewidth_over = 1.5
        self.exclude_width = 5
        self.sigma_mask = 5
        self.lover_mask = -1e-17
        self.tolerance = 0.25
        self.leg_order = 4
        self.division = 300
        self.spacing = 150
        self.endpoint = 't'
        self.endpoint_order = 4

        self.filename = filename

        # self.fitsfile = fitsfile
        # self.wave = 10.*(np.arange((np.shape(fitsfile[0].data)[0]))*fitsfile[0].header['CDELT1']+fitsfile[0].header['CRVAL1'])
        from scipy.signal import medfilt
        self.wave = w
        self.flux = medfilt(f, 1)
        self.fluxerror = e

        self.wave_ori = w_ori
        self.flux_ori = f_ori
        self.fluxerror_ori = e_ori
        self.bpmap = bpmap
        # self.bp = bp
        self.wave = self.wave[~(np.isnan(self.flux))]
        self.flux = self.flux[~(np.isnan(self.flux))]
        self.fluxerror = self.fluxerror[~(np.isnan(self.flux))]
        self.ind_err = np.arange(len(self.wave))
        self.wave_temp = self.wave[self.ind_err]
        self.flux_temp = self.flux[self.ind_err]
        self.fluxerror_temp = self.fluxerror[self.ind_err]
        self.fig = plt.figure()
        canvas = self.fig.canvas
        self.ax = self.fig.add_subplot(111)
        self.line = Line2D(self.wave, self.flux, color = cmap[0], drawstyle='steps-mid',lw=self.linewidth,label='spectrum', zorder = 1, alpha = 0.5, rasterized=True)
        self.ax.add_line(self.line)
        self.point = None
        self.pointerror = None
        self.con = None
        self.leg = None
        self.llr = None
        self.diff = None
        self.error = None
        self.over = None
        self.chebfitval = None
        self.spline = None
        self.canvas = canvas

    def run(self):
        'Iterate over points, filter, spline, mask'
        iterations = 500
        self.con_err = [0]*iterations
        sigma, val, i = np.std(self.flux), np.median(self.flux), 0

        # while sigma > val / 1e3 and i < 150:
        while i < iterations:
            try:
              self.pointx, self.pointy, self.pointyerr = methods_auto.insert_points(self.pointx, self.pointy, self.pointyerr, self.wave, self.wave_temp, self.flux, self.flux_temp, self.fluxerror, spacing = self.spacing, pick_epsilon = self.epsilon)

              self.pointx, self.pointy, self.pointyerr, self.chebfitval = methods_auto.filtering(self.pointx, self.pointy, self.pointyerr, self.wave, self.wave_temp, self.flux, self.flux_temp, tolerance=self.tolerance, leg_order=self.leg_order,
                         division=self.division, pick_epsilon = self.epsilon)

              self.continuum = methods_auto.spline_interpolation(self.pointx, self.pointy, self.wave,
                         self.wave_temp, self.flux, self.flux_temp, self.chebfitval, endpoints = self.endpoint,
                         endpoint_order = self.endpoint_order)

              self.con_err[i] = self.continuum
              i += 1
              self.wave_temp, self.flux_temp = methods_auto.mask(self.pointx, self.pointy, self.wave,
                                     self.wave_temp, self.flux, self.fluxerror,
                                     self.flux_temp, self.continuum, self.chebfitval,
                                     exclude_width=self.exclude_width,
                                     sigma_mask=self.sigma_mask, lower_mask_bound = self.lover_mask  )
            except:
                self.wave_temp, self.flux_temp = self.wave, self.flux
                print("Bad points ... Resetting")
                continue
            print(i)


        self.con_err = np.array(self.con_err)
        self.continuum = np.nanmean(self.con_err, axis= 0)
        self.stderror = np.nanstd(self.con_err, axis=0)#/np.sqrt(i)



        self.con = None
        xsh_norm.clear(self)

        self.canvas.draw()

        self.con, = self.ax.plot(self.wave, self.continuum, lw=self.linewidth_over, label='continuum', zorder = 10, color = cmap[2], alpha = 0.8, rasterized=True)

        for n in [1]:
            self.ax.plot(self.wave, self.continuum+n*self.stderror, lw=self.linewidth_over/2., label='continuum', zorder = 10, color = cmap[2], alpha = 0.8, rasterized=True)
            self.ax.plot(self.wave, self.continuum-n*self.stderror, lw=self.linewidth_over/2., label='continuum', zorder = 10, color = cmap[2], alpha = 0.8, rasterized=True)

        l, h = np.percentile(self.flux[300:-300], (5, 95))
        plt.ylim((l, h))
        plt.xlabel(r"Wavelength / [$\mathrm{\AA}$]")
        plt.ylabel(r'Flux density [erg s$^{-1}$ cm$^{-1}$ $\AA^{-1}$]')
        plt.savefig(self.filename + "_norm.pdf")
        plt.close()
        'Apply xsh_norm'
        from scipy.interpolate import interp1d
        self.spline = interp1d(self.wave, self.continuum, bounds_error=False)


        self.flux /= self.continuum
        self.flux_ori /= self.spline(self.wave_ori)
        self.fluxerror /= self.continuum
        self.fluxerror_ori /= self.spline(self.wave_ori)
        if hasattr(self, 'stderror'):
            self.stderror /= self.continuum
        if  not hasattr(self, 'stderror'):
            self.stderror = 0.1*np.ones_like(self.continuum)
        stderror = interp1d(self.wave, self.stderror, bounds_error=False)
        self.stderror = stderror(self.wave_ori)

        'Write to file'
        print 'Writing to file '+self.filename+'continuum.npy'
        data_array = np.array([self.spline(self.wave_ori), stderror(self.wave_ori)])
        np.save(self.filename+"continuum", data_array)
        # self.fitsfile[0].data = self.flux
        # self.fitsfile[1].data = self.fluxerror
        # self.fitsfile.writeto(self.filename+'_norm.fits', clobber = True)
        # if hasattr(self, 'stderror'):
        #     from astropy.io import fits
        #     fits.append(self.filename+'_norm.fits', self.stderror, self.fitsfile[1].header)


        self.ax.relim()
        self.ax.autoscale_view(True,True,True)
        self.canvas.draw()

    def clear(self):
        if self.point != None:
            self.point.remove()
        self.point = None

        if self.pointerror != None:
            self.pointerror[0].remove()
        self.pointerror = None

               #if self.con != None:
        #    self.con.remove()
        #self.con = None

        if self.leg != None:
            self.leg.remove()
        self.leg = None

        if self.diff != None:
            self.diff.remove()
        self.diff = None

        if self.error != None:
            self.error.remove()
        self.error = None

        if self.over != None:
            self.over.remove()
        self.over = None

        self.pointx = []
        self.pointy = []
        self.pointyerr = []
        self.wave_temp = []
        self.flux_temp = []

