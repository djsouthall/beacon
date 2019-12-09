#!/bin/bash
# This will execute all of the current analysis scripts in turn for a given run number.
echo 'Executing analysis scripts for Run' $1

echo 'Attempting to prepare file by executing data_handler.py'
python3 ${BEACON_ANALYSIS_DIR}tools/data_handler.py $1

echo 'Attempting to prepare time delays by executing save_time_delays.py'
python3 ${BEACON_ANALYSIS_DIR}analysis/save_time_delays.py $1

echo 'Attempting to prepare current pointing directions by executing rf_bg_search.py'
python3 ${BEACON_ANALYSIS_DIR}analysis/rf_bg_search.py $1

#Similarity must be run after save_time_delays.py
echo 'Attempting to prepare counts for similarity between events by executing similarity.py'
python3 ${BEACON_ANALYSIS_DIR}analysis/similarity.py $1

#Impulsivity must be run after save_time_delays.py
echo 'Attempting to prepare impulsivity metric by executing impulsivity.py'
python3 ${BEACON_ANALYSIS_DIR}analysis/impulsivity.py $1