#!/bin/bash
# This will execute all of the current analysis scripts in turn for a given run number. 
echo 'Executing analysis scripts for Run' $1  || exit

echo 'Attempting to prepare file by executing data_handler.py' || exit
python3 ${BEACON_ANALYSIS_DIR}tools/data_handler.py $1 || exit

echo 'Attempting to prepare time delays by executing save_time_delays.py align_method = 0' || exit
python3 ${BEACON_ANALYSIS_DIR}analysis/save_time_delays.py $1 0 || exit

echo 'Attempting to prepare current pointing directions by executing rf_bg_search.py' || exit
python3 ${BEACON_ANALYSIS_DIR}analysis/rf_bg_search.py $1 || exit

# These need to be run after time delays. 
echo 'Attempting to prepare counts for similarity between events by executing similarity.py' || exit
python3 ${BEACON_ANALYSIS_DIR}analysis/similarity.py $1 || exit

echo 'Attempting to prepare impulsivity metric by executing impulsivity.py' || exit
python3 ${BEACON_ANALYSIS_DIR}analysis/impulsivity.py $1 || exit

echo 'Attempting to prepare impulsivity metric by executing correlate_with_background_templates.py' || exit
python3 ${BEACON_ANALYSIS_DIR}analysis/correlate_with_background_templates.py $1 || exit

echo 'Attempting to prepare simple CR template search executing simple_cr_template_search.py' || exit
python3 ${BEACON_ANALYSIS_DIR}analysis/cr_search/simple_cr_template_search.py $1 1 || exit

echo 'Attempting to prepare time averaged spectrum tracking executing time_averaged_spectrum.py' || exit
python3 ${BEACON_ANALYSIS_DIR}analysis/time_averaged_spectrum.py $1 || exit