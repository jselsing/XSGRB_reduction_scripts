# XSGRB_reduction_scrips
Scripts created for the uniform reduction of GRB afterglows as part of the XSGRB collaboration

If you want to make use of this material, please contact me at jselsing@dark-cosmology.dk. If you find any bugs or missing features, please let me know.

-------
## Usage


The two main scripts in the package are XSHcomp.py and XSHextract.py. These take care of combinations of individual exposures and 1D-extractions respectively. The idea is that the ESO X-shooter pipeline, http://www.eso.org/sci/software/pipelines/, is used to reduce all observations in STARE-mode, and then the scripts provided here, do combinations and extractions where the X-shooter pipeline behaves sub-optimally. 

The scripts can be run from the commandline using:

$
python XSHcomb.py -h
$

example usage

$
python XSHcomb.py /Users/jselsing/Work/work_rawDATA/XSGRB/GRB120327A/ UVB STARE OB1 --use_master_response
$

and 

$
python XSHextract.py -h
$

$
python XSHextract.py /Users/jselsing/Work/work_rawDATA/XSGRB/GRB101219A/UVBOB2skysub.fits  --optimal --slit_corr --plot_ext --adc_corr_guess
$

## License
-------

Copyright 2016-2020 Jonatan Selsing and contributors.

These scripts are free software made available under the GNU License. For details see
the LICENSE file.
