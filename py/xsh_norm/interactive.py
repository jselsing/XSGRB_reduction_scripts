# -*- coding: utf-8 -*-


# from matplotlib import rc_file
# rc_file('/Users/jselsing/Pythonlibs/plotting/matplotlibstyle.rc')
# Plotting
import matplotlib.pyplot as plt

from xsh_norm import methods
from pylab import pause
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns; sns.set_style('ticks')
from matplotlib.backends.backend_pdf import PdfPages
# cmap = sns.color_palette("muted")
from scipy.signal import medfilt

# plt.plot([0, 1], [0, 1], sns.xkcd_rgb["pale red"], lw=3)
# plt.plot([0, 1], [0, 2], sns.xkcd_rgb["medium green"], lw=3)
# plt.plot([0, 1], [0, 3], sns.xkcd_rgb["denim blue"], lw=3);
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
        self.line = Line2D(self.wave, medfilt(self.flux, 51), color= "black", drawstyle='steps-mid', lw=2*self.linewidth, zorder = 2, rasterized=True)
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

        self._ind = None # the active vert
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.left_button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.left_button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas

    def run(self):
        self.ax.relim()
        self.ax.autoscale_view(True,True,True)
        self.fig.canvas.manager.window.raise_()
        self.fig.canvas.draw()

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.line)
        if self.point is not None:
            self.ax.draw_artist(self.point)
        if self.pointerror is not None:
            self.ax.draw_artist(self.pointerror[0])
        self.canvas.blit(self.fig.bbox)

    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        if self.point == None:
            ind = None
            return ind
        elif self.point != None:
            xyt = self.ax.transData.transform(self.point.get_xydata())
            xt, yt = xyt[:, 0], xyt[:, 1]
            d = np.sqrt((xt-event.x)**2 + (yt-event.y)**2)
#            dx = abs(xt-event.x)
            indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
            ind = indseq[0]
            if (d[ind]>=self.epsilon):# and dx[ind]>=self.epsilon:
                ind = None
            return ind

    def mask_manual(self, mask):
        self.mask = mask

    def left_button_press_callback(self, event):
        'whenever a left mouse button is pressed'
        if isinstance(self.pointx, np.ndarray):
            self.pointx = self.pointx.tolist()
            self.pointy = self.pointy.tolist()
            self.pointyerr = self.pointyerr.tolist()
        if event.inaxes is None:
            return
        if event.button != 1: return
        ind = self.get_ind_under_point(event)
        if ind is None:
            toolbar = plt.get_current_fig_manager().toolbar
            if event.button==1 and toolbar.mode=='':
                window = ((event.xdata-5)<=self.wave) & (self.wave<=(event.xdata+5))
                y = np.median(self.flux[window]).astype(np.float64)
                yerr = np.sqrt(np.sum(self.fluxerror[window] ** 2.)).astype(np.float64)
                self.pointx.append(event.xdata)
                self.pointy.append(y)
                self.pointyerr.append(yerr)
                if self.point is None:
                    self.point, _, self.pointerror = self.ax.errorbar(self.pointx ,self.pointy, yerr = self.pointyerr, fmt=".r", capsize=0, elinewidth=0.5, color = cmap[2], ms=self.epsilon, picker=self.epsilon, label='cont_pnt')
                else:
                    try:
                        self.point.remove()
                        self.pointerror[0].remove()
                        self.canvas.draw()
                    except AttributeError:
                        pass
                    except ValueError:
                        pass
                    finally:
                        self.point, _, self.pointerror = self.ax.errorbar(self.pointx, self.pointy, yerr = self.pointyerr, fmt=".r", capsize=0, elinewidth=0.5, color = cmap[2], ms=self.epsilon, picker=self.epsilon, label='cont_pnt')

        self._ind = self.get_ind_under_point(event)
        self.canvas.restore_region(self.background)
        self.canvas.draw()
        self.canvas.blit(self.fig.bbox)

    def left_button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts: return
        if event.button != 1: return
        self._ind = None
        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
#        if not self.showverts: return
        self._ind = self.get_ind_under_point(event)
        if self._ind is None: return
        if event.inaxes is None: return
        if event.button != 1: return
        toolbar = plt.get_current_fig_manager().toolbar
        if event.button == 1 and toolbar.mode == "":
            x, y = event.xdata, event.ydata
            self.point.get_xydata()[self._ind] = x, y
            self.point.set_data(zip(*self.point.get_xydata()))
            self.pointx[self._ind] = x
            self.pointy[self._ind] = y
            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.point)
            self.canvas.blit(self.fig.bbox)

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return

        if event.key == "d":
            'Delete point under mouse'
            self._ind = self.get_ind_under_point(event)
            if self._ind == None: return
            del self.pointx[self._ind]
            del self.pointy[self._ind]
            del self.pointyerr[self._ind]
            self.point.remove()
            self.pointerror[0].remove()
            self.canvas.draw()
            self.point, _, self.pointerror = self.ax.errorbar(self.pointx, self.pointy, yerr = self.pointyerr, fmt=".r", capsize=0, elinewidth=0.5, color = cmap[2], ms=self.epsilon, picker=self.epsilon, label='cont_pnt')

        elif event.key == 'y':
            "Insert evenly spaced points - every 100'th pixel"
            self.pointx, self.pointy, self.pointyerr, self.point, self.pointerror = methods.insert_points(self.pointx, self.pointy, self.pointyerr, self.wave, self.wave_temp, self.flux, self.flux_temp, self.fluxerror, self.ax, self.point, self.pointerror, spacing = self.spacing / 2.0, pick_epsilon = self.epsilon)
            self.canvas.draw()

        if event.key == 'u':
            "Insert evenly spaced points - every 50'th pixel"
            self.pointx, self.pointy, self.pointyerr, self.point, self.pointerror = methods.insert_points(self.pointx,
                                       self.pointy, self.pointyerr, self.wave, self.wave_temp,
                                       self.flux, self.flux_temp, self.fluxerror, self.ax,
                                       self.point, self.pointerror, spacing = self.spacing / 4.0)
            self.canvas.draw()

        elif event.key == 't':
             'Filter points by low-order legendre fitting and clipping values of highest sigma iteratively until continuum is found'
             self.pointx, self.pointy, self.pointyerr, self.point, self.pointerror, self.chebfitval, self.leg  = methods.filtering(self.pointx, self.pointy, self.pointyerr, self.wave,
                                       self.wave_temp, self.flux, self.flux_temp,
                                       self.ax, self.point, self.pointerror, self.leg, tolerance=self.tolerance, leg_order=self.leg_order,
                                       division=self.division)
             self.canvas.draw()

        elif event.key == 'enter':
            'Sort spline points and interpolate between marked continuum points'
            self.continuum, self.leg, self.con = methods.spline_interpolation(self.pointx, self.pointy, self.wave,
                               self.wave_temp, self.flux, self.flux_temp,
                               self.ax, self.leg, self.con, self.chebfitval, endpoints = self.endpoint,
                               endpoint_order = self.endpoint_order)
            self.canvas.draw()

        elif event.key=='m':
            'Mask areas where signal is present'

            self.wave_temp, self.flux_temp, self.diff, self.error, self.over = \
                methods.mask(self.pointx, self.pointy, self.wave,
                                       self.wave_temp, self.flux, self.fluxerror,
                                       self.flux_temp, self.continuum, self.ax,
                                       self.diff, self.error, self.over, self.chebfitval,
                                       exclude_width=self.exclude_width,
                                       sigma_mask=self.sigma_mask, lower_mask_bound = self.lover_mask )
            self.canvas.draw()

        elif event.key == 'a':
            'Local linear regression'
            self.continuum, self.llr = methods.llr(self.wave, self.wave_temp, self.flux,
                                           self.flux_temp, self.ax, self.llr)
            self.canvas.draw()

        elif event.key ==  'i':
            'Iterate over points, filter, spline, mask'
            iterations = 100
            self.con_err = [0]*iterations
            sigma, val, i = np.std(self.flux), np.median(self.flux), 0

            # while sigma > val / 1e3 and i < 150:
            pp = PdfPages("../figures/Continuum.pdf")
            while i < iterations:
                i += 1
                if i <= 100:

                    pause(0.001)
                    self.pointx, self.pointy, self.pointyerr, self.point, self.pointerror = methods.insert_points(self.pointx,
                               self.pointy, self.pointyerr, self.wave, self.wave_temp,
                               self.flux, self.flux_temp, self.fluxerror, self.ax,
                               self.point, self.pointerror, spacing = self.spacing, pick_epsilon = self.epsilon)
                    self.canvas.draw()
                    pp.savefig()
                    pause(0.001)
                    self.pointx, self.pointy, self.pointyerr, self.point, self.pointerror, self.chebfitval, self.leg  = methods.filtering(self.pointx, self.pointy, self.pointyerr, self.wave,
                               self.wave_temp, self.flux, self.flux_temp,
                               self.ax, self.point, self.pointerror, self.leg, tolerance=self.tolerance, leg_order=self.leg_order,
                               division=self.division, pick_epsilon = self.epsilon)
                    self.canvas.draw()
                    pp.savefig()
                    pause(0.001)
                    self.continuum, self.leg, self.con = methods.spline_interpolation(self.pointx, self.pointy, self.wave,
                               self.wave_temp, self.flux, self.flux_temp,
                               self.ax, self.leg, self.con, self.chebfitval, endpoints = self.endpoint,
                               endpoint_order = self.endpoint_order)
                    self.canvas.draw()
                    pp.savefig()
                    self.con_err[i - 1] = self.continuum

                    pause(0.001)
                    self.wave_temp, self.flux_temp, self.diff, self.error, self.over = \
                        methods.mask(self.pointx, self.pointy, self.wave,
                                           self.wave_temp, self.flux, self.fluxerror,
                                           self.flux_temp, self.continuum, self.ax,
                                           self.diff, self.error, self.over, self.chebfitval,
                                           exclude_width=self.exclude_width,
                                           sigma_mask=self.sigma_mask, lower_mask_bound = self.lover_mask  )
                    pp.savefig()

                else:
                    self.pointx, self.pointy, self.pointyerr, self.point, self.pointerror = methods.insert_points(self.pointx,
                               self.pointy, self.pointyerr, self.wave, self.wave_temp,
                               self.flux, self.flux_temp, self.fluxerror, self.ax,
                               self.point, self.pointerror, spacing = self.spacing, pick_epsilon = self.epsilon)

                    self.pointx, self.pointy, self.pointyerr, self.point, self.pointerror, self.chebfitval, self.leg  = methods.filtering(self.pointx, self.pointy, self.pointyerr, self.wave,
                               self.wave_temp, self.flux, self.flux_temp,
                               self.ax, self.point, self.pointerror, self.leg, tolerance=self.tolerance, leg_order=self.leg_order,
                               division=self.division, pick_epsilon = self.epsilon)

                    self.continuum, self.leg, self.con = methods.spline_interpolation(self.pointx, self.pointy, self.wave,
                               self.wave_temp, self.flux, self.flux_temp,
                               self.ax, self.leg, self.con, self.chebfitval, endpoints = self.endpoint,
                               endpoint_order = self.endpoint_order)

                    self.con_err[i - 1] = self.continuum
                    self.wave_temp, self.flux_temp, self.diff, self.error, self.over = \
                        methods.mask(self.pointx, self.pointy, self.wave,
                                           self.wave_temp, self.flux, self.fluxerror,
                                           self.flux_temp, self.continuum, self.ax,
                                           self.diff, self.error, self.over, self.chebfitval,
                                           exclude_width=self.exclude_width,
                                           sigma_mask=self.sigma_mask, lower_mask_bound = self.lover_mask  )

                print(i)
                # pp.savefig()

            self.con_err = np.array(self.con_err)
            self.continuum = np.mean(self.con_err, axis= 0)
            self.stderror = np.std(self.con_err,axis=0)#/np.sqrt(i)


            self.con.remove()
            self.con = None
            xsh_norm.clear(self)

            self.canvas.draw()

            self.con, = self.ax.plot(self.wave, self.continuum, lw=self.linewidth_over, label='continuum', zorder = 10, color = cmap[2], alpha = 0.8, rasterized=True)

            for n in [1]:
                self.ax.plot(self.wave, self.continuum+n*self.stderror, lw=self.linewidth_over/2., label='continuum', zorder = 10, color = cmap[2], alpha = 0.8, rasterized=True)
                self.ax.plot(self.wave, self.continuum-n*self.stderror, lw=self.linewidth_over/2., label='continuum', zorder = 10, color = cmap[2], alpha = 0.8, rasterized=True)
            pp.savefig()
            pp.close()

        elif event.key == 'n':
            'Apply xsh_norm'
            from scipy.interpolate import interp1d
            self.spline = interp1d(self.wave, self.continuum, bounds_error=False)
            xsh_norm.clear(self)
            self.canvas.draw()

            # self.fig.set_size_inches(14,10)
            plt.savefig(self.filename + "_norm.pdf")

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
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_ylim((-0.5,1.5))
            y1 = np.ones(np.shape(self.wave))
            self.line, = self.ax.plot(self.wave_ori, self.flux_ori, color= cmap[0], drawstyle='steps-mid', lw=self.linewidth, label='normalised spectrum', zorder = 1, rasterized=True)
            self.line, = self.ax.plot(self.wave_ori, medfilt(self.flux_ori, 51), color= "black", drawstyle='steps-mid', lw=2*self.linewidth, zorder = 2, rasterized=True)
            self.line1, = self.ax.plot(self.wave, y1, color=cmap[2], drawstyle='steps-mid', lw=self.linewidth_over,label='1', zorder = 10, alpha=1.0, rasterized=True)

        elif event.key == 'w':


          'Write to file'
          print('Writing to file '+self.filename+'continuum.npy')
          data_array = np.array([self.spline(self.wave_ori), 0.1*np.ones_like(self.wave_ori)])
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

