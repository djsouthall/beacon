#!/bin/bash
# This will execute all of the current analysis scripts in turn for a given run number.
echo 'Executing analysis scripts for Run' $1  || exit

# echo 'Attempting to prepare file by executing data_handler.py' || exit
# python3 ${BEACON_ANALYSIS_DIR}tools/data_handler.py $1 || exit

# Sine subtract cache should be run seperately, but it SHOULD be run.  
# echo 'Attempting to prepare sine subtraction by executing sine_subtract_cache.py' || exit
# python3 ${BEACON_ANALYSIS_DIR}tools/sine_subtract_cache.py $1 || exit

echo 'Attempting to prepare time delays by executing save_time_delays.py align_method = 0' || exit
python3 ${BEACON_ANALYSIS_DIR}analysis/save_time_delays.py $1 0 || exit

# echo 'Attempting to identify cw by executing flag_cw.py' || exit
# python3 ${BEACON_ANALYSIS_DIR}analysis/flag_cw.py $1 || exit

# # These need to be run after time delays. 

# # Depends on time delays to similar timings
echo 'Attempting to prepare counts for similarity between events by executing similarity.py' || exit
python3 ${BEACON_ANALYSIS_DIR}analysis/similarity.py $1 || exit

# Depends on time delays to align signals
echo 'Attempting to prepare impulsivity metric by executing impulsivity.py' || exit
python3 ${BEACON_ANALYSIS_DIR}analysis/impulsivity.py $1 || exit


echo 'Attempting to prepare simple CR template search executing simple_cr_template_search.py' || exit
python3 ${BEACON_ANALYSIS_DIR}analysis/cr_search/simple_cr_template_search.py $1 1 || exit

echo 'Completed Analysis Part 1' || exit