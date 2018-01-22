#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from astropy.io import fits
from astropy.time import Time, TimeDelta
import pandas as pd
import datetime
import numpy as np

instrument = "X-SHOOTER"
uname = "jselsing"
passw = "spEkUxu2"



def uniques(seq):
    """
    From https://www.peterbe.com/plog/uniqifiers-benchmark
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def main():

    # Get all files in parent dir
    filepath = "/Users/jselsing/Work/work_rawDATA/XSGW/get_data_test/"
    filelist = glob.glob(filepath+"*")


    arms = [fits.open(x)[0].header["HIERARCH ESO SEQ ARM"] for x in filelist]
    obsids = [fits.open(x)[0].header["HIERARCH ESO OBS ID"] for x in filelist]
    n_sets = uniques(obsids)
    progids = [fits.open(x)[0].header["HIERARCH ESO OBS PROG ID"] for x in filelist]
    propid = uniques(progids)
    obs_dates = [fits.open(x)[0].header["DATE-OBS"] for x in filelist]
    counter = np.ones(len(obs_dates))

    # Create an empty dataframe
    df = pd.DataFrame()
    # Create a column from the observed dates
    df['datetime'] = obs_dates
    # Convert that column into a datetime datatype
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Set the datetime column as the index
    df.index = df['datetime']
    # Create a column to count the number of observations taken each night
    df['n_files'] = counter
    # Get number of exposures ceneted around each night.
    nights = df.resample('D', base = 0.5, label = 'left', loffset = "12H").sum().dropna(axis=0).index
    print(nights)
    n_night = len(nights)
    for ii, kk in enumerate(nights):

        t_night = Time(str(kk))
        print(t_night)
        print(abs(t_night - Time(obs_dates)).value < 0.5 )
    # print(n_night, nights)
    # print(Time(nights) - Time(obs_dates))
    # print(nights['datetime'] - obs_dates)
    # print(obs_dates.ptp())
    # t_start = min(obs_dates)
    # t_end = max(obs_dates)
    # print(min(obs_dates), max(obs_dates))

    # Sort in datasets:
    # for ii, kk in filelist:
    #     obsids = headers[ii]["HIERARCH ESO SEQ ARM"]
    #     print(obsids)
    # #     fitsfile =
    # print(filelist)
    # pass

    # run_call = ["./prog_access.sh", "-User", uname, "-Passw", passw, "RunId", runid, "-Inst", instrument, "-StartDate", startdate, "-EndDate", enddate, "-FileCat", category, "-MaxRows", 50]
    # t1 = time.time()
    # print('\tGetting calibrations')
    # print('\t%s' %(' '.join([run_call])))
    # progress_log = os.path.abspath(output_path+'/prog_access.log' % (self.name, self.arm))
    # with open(molecpar, 'w') as f:
    #     runMolec = subprocess.run([molecCall, self.molecparfile],
    #                                 stdout=f)
    # f.close()
    # with open(molecpar, 'r') as f:
    #     lines = f.read().splitlines()
    #     runMolecRes = lines[-1].strip()

    # if runMolecRes == '[ INFO  ] No errors occurred':
    #     print('\tMolecfit sucessful in %.0f s' % (time.time()-t1))
    # else:
    #     print(runMolecRes)




if __name__ == '__main__':
    main()


# wget -O output_query_$$.csv "http://archive.eso.org/wdb/wdb/eso/eso_archive_main/query?tab_object=on&target=$targetname&resolver=simbad&tab_target_coord=on&ra=$ra&dec=$dec&box=00+10+00&deg_or_hour=hours&format=SexaHours&tab_prog_id=on&prog_id=$pid&tab_instrument=on&instrument=$inst&stime=$date1&starttime=12&etime=$date2&endtime=12&tab_dp_cat=true&dp_cat=$cat&top=$maxrows&wdbo=csv"




# echo "  ./prog_access.sh -User <your username> -Passw <your password> -RunId '090.C-0733(A)' -Inst FORS2 -StartDate '2013 01 01' -EndDate '2013 04 01' -FileCat SCIENCE -MaxRows 30" 

# http://archive.eso.org/wdb/wdb/eso/xshooter/query?wdbo=html%2fdisplay&max_rows_returned=1000&target=&resolver=simbad&coord_sys=eq&coord1=&coord2=&box=00%2010%2000&format=sexagesimal&tab_wdb_input_file=on&wdb_input_file=&tab_night=on&night=2017%2008%2027&stime=&starttime=12&etime=&endtime=12&tab_prog_id=on&prog_id=&prog_type=%25&obs_mode=%25&pi_coi=&pi_coi_name=PI_only&prog_title=&tab_dp_id=on&dp_id=&tab_ob_id=on&ob_id=&tab_obs_targ_name=on&obs_targ_name=&tab_exptime=on&exptime=&tab_dp_cat=on&dp_cat=%25&tab_dp_type=on&dp_type=%25&dp_type_user=&tab_dp_tech=on&dp_tech=%25&dp_tech_user=&tab_ins_mode=on&ins_mode=%25&tel_id=&tpl_id=&tpl_start=&tpl_nexp=&tpl_expno=&tab_seq_arm=on&seq_arm=%25&det_dit=&det_ndit=&tab_det_read_clock=on&det_read_clock=%25&binx=%25&biny=%25&tab_ins_filt1_name=on&ins_filt1_name=%25&tab_ins_opti2_name=on&ins_opti2_name=%25&tab_ins_opti3_name=on&ins_opti3_name=%25&tab_ins_opti4_name=on&ins_opti4_name=%25&tab_ins_opti5_name=on&ins_opti5_name=%25&seq_jitter_width=&seq_nod_throw=&tel_ambi_fwhm_start=&tel_ambi_fwhm_end=&tel_ia_fwhm=&tab_fwhm_avg=on&fwhm_avg=&airmass_range=&night_flag=%25&moon_illu=&order=&


# wdbo = html/display
# top = 1000
# resolver = simbad
# coord_sys = eq
# box = 00 10 00
# format = sexagesimal
# night = 2017 08 27
# starttime = 12
# endtime = 12
# prog_type = %
# obs_mode = %
# pi_coi_name = PI_only
# dp_cat = %
# dp_type = %
# dp_tech = %
# ins_mode = %
# seq_arm = %
# det_read_clock = %
# binx = %
# biny = %
# ins_filt1_name = %
# ins_opti2_name = %
# ins_opti3_name = %
# ins_opti4_name = %
# ins_opti5_name = %
# night_flag = %

