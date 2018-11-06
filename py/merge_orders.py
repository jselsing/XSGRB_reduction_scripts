#!/usr/local/anaconda3/envs/py36 python
# -*- coding: utf-8 -*-

# Plotting
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as pl
# import seaborn; seaborn.set_style('ticks')

# Imports
import numpy as np
from astropy.io import fits
from astropy import wcs
import glob

class XshOrder2D(object):
    # Thanks, Johannes!
    def __init__(self, fname):

        self.fname = fname
        self.hdul =  fits.open(fname)

        self.get_order_information()

    def get_order_information(self):
        from astropy import wcs

        self.norders = int(len(self.hdul) / 3)

        header_list = [self.hdul[i*3].header for i in range(self.norders)]
        wcs_list = [wcs.WCS(header) for header in header_list]

        self.cdelt1 = header_list[0]['CDELT1']
        self.naxis1_list = [header['NAXIS1'] for header in header_list]
        self.naxis2 = header_list[0]['NAXIS2']

        # print(wcs_list[0].all_pix2world(1,1,1))

        self.start_wave_list = [wcs.wcs_pix2world(1, 1, 1)[0] for wcs in wcs_list]

        self.end_wave_list = [wcs.wcs_pix2world(naxis1,1,1)[0] for wcs, naxis1 in zip(wcs_list, self.naxis1_list)]
        self.wcs_list, self.header_list = wcs_list, header_list

    def create_out_frame_empty(self):

        self.npixel = int((self.end_wave_list[0] - self.start_wave_list[-1]) / self.cdelt1) + 1
        # print(self.cdelt1)
        self.ref_wcs = self.wcs_list[-1]
        self.data_new = np.zeros((self.naxis2, self.npixel))
        self.err_new = self.data_new.copy()
        self.qual_new = self.data_new.copy()




    def fill_data(self):

        weight_new = self.data_new.copy().astype(np.float64)
        pixels_new = np.zeros_like(weight_new)

        for i in range(self.norders):
            # In zero based coordinates
            start_pix = int(np.floor(self.ref_wcs.wcs_world2pix(self.start_wave_list[i],0, 0)[0]))
            end_pix = int(np.floor(self.ref_wcs.wcs_world2pix(self.end_wave_list[i], 0, 0)[0]))


            # TODO; better concept fo error propagation

            data_inter = self.hdul[i*3].data.astype(np.float64)
            err_inter = self.hdul[i*3+1].data.astype(np.float64)
            qual_inter = self.hdul[i*3+2].data
            mask = qual_inter > 0
            err_inter[mask] = 1E15

            # TODO: Implement here the masked array, as in combine_xshooter_script.py
            err_cols = err_inter.copy()*0. + np.median(err_inter, axis=0)
            err_cols[mask] = 1E15

            weight_inter = 1./np.square(err_cols)


            self.err_new[:, start_pix:end_pix+1] += weight_inter**2. * err_inter**2.
            self.data_new[:, start_pix:end_pix+1] += weight_inter * data_inter
            weight_new[:, start_pix:end_pix+1]  += weight_inter
            pixels_new[:, start_pix:end_pix+1]  += (~mask).astype("int")

        self.data_new /= weight_new
        self.err_new = np.sqrt(self.err_new / weight_new**2.)
        self.qual_new = (self.err_new > 1E10).astype(int) # FIXME; this needs to be implemente in a better way; using the actual masks

        self.data_new = self.data_new.astype(np.float32)
        self.err_new = self.err_new.astype(np.float32)




        #     data_inter = self.hdul[i*3].data.astype(np.float64)
        #     err_inter = self.hdul[i*3+1].data.astype(np.float64)
        #     qual_inter = self.hdul[i*3+2].data
        #     mask = qual_inter > 0

        #     data_inter[mask] = 0
        #     err_inter[mask] = 0


        #     self.err_new[:, start_pix:end_pix+1] += err_inter**2.
        #     self.data_new[:, start_pix:end_pix+1] += data_inter
        #     self.qual_new[:, start_pix:end_pix+1] += qual_inter
        #     weight_new[:, start_pix:end_pix+1]  += (~mask).astype("int")


        # self.data_new /= weight_new
        # self.err_new = np.sqrt(self.err_new / weight_new**2.)
        # self.qual_new = (self.data_new == 0).astype(int)



    def create_final_hdul(self):

        hdul_new = fits.HDUList()
        hdu_data = fits.PrimaryHDU(self.data_new, self.hdul[-3].header)
        hdu_err = fits.PrimaryHDU(self.err_new, self.hdul[-2].header)
        hdu_qual = fits.PrimaryHDU(self.qual_new, self.hdul[-1].header)




        hdu_data.header['EXTNAME'] = 'FLUX'
        hdu_data.header['PIPEFILE'] = 'MERGe_PYTHON_JZ'
        hdu_err.header['EXTNAME'] = 'ERRS'
        hdu_err.header['SCIDATA'] = 'FLUX'
        hdu_err.header['QUALDATA'] = 'QUAL'
        hdu_qual.header['EXTNAME'] = 'QUAL'
        hdu_qual.header['SCIDATA'] = 'FLUX'
        hdu_qual.header['ERRDATA'] = 'ERRS'


        hdul_new.append(hdu_data)
        hdul_new.append(hdu_err)
        hdul_new.append(hdu_qual)

        self.hdul_new = hdul_new


    def write_result(self, fname_out, clobber=False):

        self.hdul_new.writeto(fname_out, overwrite=True)



    def do_all(self, fname_out, clobber=False):
        self.create_out_frame_empty()
        self.fill_data()
        self.create_final_hdul()
        self.write_result(fname_out, clobber=clobber)


def main():
    input_dir = "/Users/jonatanselsing/Work/work_rawDATA/Crab_Pulsar"
    # input_dir = "/Users/jonatanselsing/Work/work_rawDATA/FRB/FRB180930"

    # input_dir = "/Users/jonatanselsing/Work/work_rawDATA/STARGATE/GRB181010A"
    # input_dir = "/Users/jonatanselsing/Work/work_rawDATA/STARGATE/GRB181020A"

    # input_dir = "/Users/jonatanselsing/Work/work_rawDATA/SLSN/SN2018bsz"
    # input_dir = "/Users/jonatanselsing/Work/work_rawDATA/XSGW/AT2017GFO"
    merge_files = glob.glob(input_dir+"/reduced_data/OB9/*/*/*/*FLUX_ORDER2D*")
    target_files = glob.glob(input_dir+"/reduced_data/OB9/*/*/*FLUX_MERGE2D*")
    target_files = [ii for ii in target_files if "MANMERGE" not in ii and "TELL" not in ii]


    for kk, ll in list(zip(merge_files, target_files)):
        print(ll)
        insorder2d = XshOrder2D(kk)
        fname_out = ll.replace('FLUX_MERGE2D', 'FLUX_MERGE2D_MANMERGE')
        insorder2d.do_all(fname_out)


if __name__ == '__main__':
    main()
