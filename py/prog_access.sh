#!/bin/sh
#
# Script to browse, query, and download data from the Archive via the Raw data query form, with as input parameters: 
# User Portal login and password, and as options: target name, right ascension, declination, run id, instrument, Start date, 
# End date, file Category (CALIB or SCIENCE or ACQUISITION) and maximum number of rows to be returned (this is the number 
# of fits files to get returned when querying the archive. This number may be lower than the actual number of eventually 
# downloaded files as there may be added files like night logs or associated files to calibrate the data).
#
#
#
#           EXAMPLES: 
#           ./prog_acc.sh -User <your username> -Passw <your password> -RunId '090.C-0733(A)' -Inst FORS2 
#              -StartDate '2013 01 01'-EndDate '2013 04 01' -FileCat SCIENCE -MaxRows 30
#           ./prog_acc.sh -User <your username> -Passw <your password> -RunId '090.C-0733(A)' 
#           ./prog_acc.sh -User <your username> -Passw <your password> -Inst FORS2 -FileCat SCIENCE
#           ./prog_acc.sh -User <your username> -Passw <your password> -StartDate '2013 01 01' -EndDate '2013 04 01' 
#           ./prog_acc.sh -User <your username> -Passw <your password> -FileCat FORS2 -MaxRows 200
#           ./prog_acc.sh -User <your username> -Passw <your password> -MaxRows 20 -RA '08 36 15.1' -DEC '-26 24 33.6'
#           ./prog_acc.sh -User <your username> -Passw <your password> -MaxRows 20 -TargetName 'Hen2-10'
#
#
uname=`uname`
us_agent="ESO_RAW_DATA_PROGRAMMATIC_SCRIPT($uname)"

#####################################
# Get the command line option values
#####################################
while [ $# -gt 0 ]; do
        case $1 in
                -User )                 shift
                user=$1
		;;
	        -Passw )                shift
                pw=$1
		;;
                -TargetName )           shift
		targetname=$1
		;;
	        -RA )                   shift
		ra=$1
		;;
	        -DEC )                  shift
		dec=$1
		;;
	        -RunId )		shift
		pid=$1
		;;
		-Inst )	                shift
		inst=$1
		;;
		-StartDate )		shift
		date1=$1
		;;
		-EndDate )		shift
	        date2=$1
		;;
		-FileCat )		shift
		cat=$1
		;;
                -MaxRows )              shift
                maxrows=$1
                ;;
	esac
		shift
done


################################################################
# Display the usage, if credentials are missing exit gracefully
################################################################
if [ "$user" = "" ] || [ "$pw" = "" ]; then
echo "  USAGE: ./prog_access.sh -User <your username> -Passw <your password> -TargetName 'Target name' -RA 'Right Ascension' -DEC 'Declination' -RunId 'Run ID' -Inst <INSTRUMENT> -StartDate 'Start-date' -EndDate 'End-date' -FileCat <CATEGORY> -MaxRows <Maximum number of rows returned>"
echo "                          "
echo "  - Be aware that the maximum number of rows returned by default is 100"
echo "  - The search radius around the target name and/or coordinates is by default 10 minutes."
echo "  - wget must be installed on your machine as well as GNU awk version 4 or higher."
echo "  - If you run this script on a MAC, part of the last command line 'xargs -l1 -P 2 wget -c -nv' must be replaced by 'xargs -L1 -P 2 wget -c -nv' " 
echo "  - The following example will download directly 32 files (16 fits, 16 nightlog files) onto your disk" 
echo "  ./prog_access.sh -User <your username> -Passw <your password> -RunId '090.C-0733(A)' -Inst FORS2 -StartDate '2013 01 01' -EndDate '2013 04 01' -FileCat SCIENCE -MaxRows 30" 
exit
fi

########################################################################
# Check if the correct gawk version is installed (version 4 and higher)
########################################################################
check_gawk() {

   gawk_status=1
   # check if gawk is in the path
   which gawk
   which gawk >& /dev/null
   if [ "$?" -eq "0" ]; then
      # Check if this version of gawk allows FPAT
      tmpfile=/tmp/test_gawk_$$
      echo '"A,SeparatedField",Another' > $tmpfile
      numfields=`cat $tmpfile | gawk '{print NF}' FPAT='([^,]+)|("[^"]+")'`
      if [ "$numfields" -eq "2" ]; then
         gawk_status=0
      fi
   fi
}

check_gawk
if [ "$gawk_status" -eq "1" ]; then
   echo "Sorry, you either don't have gawk installed, or it is installed but it does not support the FPAT construct (version 4 and higher)."
   exit $gawk_status
fi

####################################################################
# Before to query the archive force capitalise the instrument value
####################################################################
inst=`echo $inst | tr '[a-z]' '[A-Z]'`
echo $inst
#exit

#########################################################################################################
# Query to the archive with Target Coordinates, Taget name, RunID, Inst, StartDate, EndDate, FileCat and 
# MaxRows as (empty or not) entries 
#########################################################################################################
wget -O output_query_$$.csv "http://archive.eso.org/wdb/wdb/eso/eso_archive_main/query?tab_object=on&target=$targetname&resolver=simbad&tab_target_coord=on&ra=$ra&dec=$dec&box=00+10+00&deg_or_hour=hours&format=SexaHours&tab_prog_id=on&prog_id=$pid&tab_instrument=on&instrument=$inst&stime=$date1&starttime=12&etime=$date2&endtime=12&tab_dp_cat=true&dp_cat=$cat&top=$maxrows&wdbo=csv"
#exit

######################################################################################
# Check if there is any record matching the provided criteria, if not exit gracefully
######################################################################################
checkifempty=`cat output_query_$$.csv | grep "A total of 0"`
if [ -n "$checkifempty" ]; then
echo "A total of 0 records were found matching the provided criteria. Exiting."
exit
fi

####################################################################################################
# Create a list of file_ids out of the output csv file to get submitted as a request to the Archive.
# The list of files to get submitted must be in the right format. 
#################################################################################################### 

filelist=`cat output_query_$$.csv | grep ":" | grep -v "SIMBAD" | gawk  '{print $9}' FPAT="([^,]*)|(\"[^\"]*\")" | gawk -F "." '{print "SAF%2B"$1"."$2"."$3","}'`
echo $filelist
#exit

#################################################
# Submit the request using the file list created
#################################################
wget -O submission_$$ --user-agent="${user_agent}" --auth-no-challenge --post-data="requestDescription=script&dataset=$filelist" --header="Accept:text/plain" --http-user=$user --http-password=$pw https://dataportal.eso.org/rh/api/requests/$user/submission

#####################################################################
# Get the request number of the request that has just been submitted
#####################################################################
reqnum=`cat submission_$$ | gawk '{print$1}'`
echo "reqnum=" $reqnum

#####################################################################
# Before downloading the data make sure that the request is complete 
# so that also the download.sh script is complete
#####################################################################

wget -O state_$$ --auth-no-challenge --user-agent="${user_agent}" --http-user=$user --http-password=$pw https://dataportal.eso.org/rh/api/requests/$user/$reqnum/state

requeststate=`tail -1 state_$$ | cut -c1-9`
echo "request state is" $requeststate

while [ $requeststate != "COMPLETE" ]; do
   \rm state_$$
   wget -O state_$$ --auth-no-challenge --user-agent="${user_agent}" --http-user=$user --http-password=$pw https://dataportal.eso.org/rh/api/requests/$user/$reqnum/state
   requeststate=`tail -1 state_$$| cut -c1-9`
   echo "request state is now" $requeststate
   sleep 5
done

###############################
# Download the download script
###############################
wget -O downloadRequest_$reqnum.sh --user-agent="${user_agent}" --auth-no-challenge --http-user=$user --http-password=$pw https://dataportal.eso.org/rh/api/requests/$user/$reqnum/script


#########################################
# Check for the presence of a .netrc file
#########################################
check_netrc() {

   # check if netrc exists
   ls -la ~/.netrc
   ls -la ~/.netrc >& /dev/null
   if [ "$?" -eq "0" ]; then
      # Add a line to the already existing .netrc
      echo "machine dataportal.eso.org login" $user "password" $pw >> ~/.netrc
   elif [ "$?" -eq "1" ]; then
      # Create a .netrc
      echo "machine dataportal.eso.org login" $user "password" $pw > ~/.netrc
   fi
}

check_netrc


###################################################################################
# Download the data, first by making the download script executable.
# Files will be downloaded with 2 parallel threads, 5 being the maximum allowed, 
# just like with the download manager applet.
###################################################################################
chmod 777 downloadRequest_$reqnum.sh
./downloadRequest_$reqnum.sh -d "--user-agent=$us_agent" -X "-L 1 -P 2"


#######################################
# clean ancillary files before leaving
#######################################
#\rm -rf submission_$$* state_$$* 
#
#
#exit
#
