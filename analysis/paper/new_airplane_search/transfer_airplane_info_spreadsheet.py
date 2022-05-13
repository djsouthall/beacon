#!/usr/bin/env python3
'''
Given 2 spreadsheets this will copy the column ICAO24 from one to the other.
'''

import sys
import os
import inspect
import h5py
import copy
from pprint import pprint
import time
import glob

import numpy
import scipy
import scipy.signal
import pandas as pd

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import FFTPrepper, TemplateCompareTool
# from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.write_event_dict_to_excel import writeDataFrameToExcel
# from beacon.tools.line_of_sight import circleSource
# from beacon.tools.flipbook_reader import flipbookToDict


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)


if __name__ == '__main__':
    plt.close('all')
    start_time = time.time()
    infile_has_airplanes = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'paper', 'data', 'event_info_collated.xlsx')
    sheet_with_planes = 'good-with-airplane'
    sheet_with_sorting = 'stage_2_pass'
    infile_needs_airplanes = '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/new-cut-event-info_master_missing_keys.xlsx'
    outfile = infile_needs_airplanes.replace('.xlsx','_%i.xlsx'%start_time)


    with pd.ExcelFile(infile_has_airplanes) as xls:
        for sheet_index, sheet_name in enumerate(xls.sheet_names):
            if sheet_name == sheet_with_planes:
                df_has_airplanes = xls.parse(sheet_name)
                df_has_airplanes = df_has_airplanes.sort_values(by = ['run', 'eventid'], ascending = [True, True], na_position = 'last')
                df_has_airplanes = df_has_airplanes.copy(deep=True)
            elif sheet_name == sheet_with_sorting:
                df_has_sorting = xls.parse(sheet_name)
                df_has_sorting = df_has_sorting.sort_values(by = ['run', 'eventid'], ascending = [True, True], na_position = 'last')
                df_has_sorting = df_has_sorting.copy(deep=True)
            else:
                continue


    with pd.ExcelFile(infile_needs_airplanes) as xls:
        for sheet_index, sheet_name in enumerate(xls.sheet_names):
            df_needs_airplanes = xls.parse(sheet_name)
            df_needs_airplanes = df_needs_airplanes.sort_values(by = ['run', 'eventid'], ascending = [True, True], na_position = 'last')

            new_df = df_needs_airplanes.copy(deep=True)

            if not numpy.isin('suspected_airplane_icao24', new_df.columns):
                new_df.insert(list(df_has_airplanes.columns).index('suspected_airplane_icao24'), 'suspected_airplane_icao24' , pd.Series(dtype=df_has_airplanes.dtypes['suspected_airplane_icao24']))

            #COpy from master airplane sheet
            for row_index, row in new_df.iterrows():
                airplane = ''
                try:
                    airplane = str(numpy.asarray(df_has_airplanes.query('run == %i & eventid == %i'%(row['run'],row['eventid'])).fillna('')['suspected_airplane_icao24'])[0])
                except:
                    airplane = ''

                try:
                    airplane_notes = str(df_has_airplanes.query('run == %i & eventid == %i'%(row['run'],row['eventid'])).fillna('')['notes'].to_numpy()[0])
                except:
                    airplane_notes = ''

                new_df['suspected_airplane_icao24'][row_index] = airplane

                try:
                    sort_key = str(df_has_sorting.query('run == %i & eventid == %i'%(row['run'],row['eventid'])).fillna('unsorted')['key'].to_numpy()[0])
                except:
                    sort_key = 'unsorted'

                try:
                    sort_key_notes = str(df_has_sorting.query('run == %i & eventid == %i'%(row['run'],row['eventid'])).fillna('')['notes'].to_numpy()[0])
                except:
                    sort_key_notes = ''

                if new_df['key'][row_index] == 'unsorted':
                    new_df['key'][row_index] = sort_key
                else:
                    if new_df['key'][row_index] != sort_key:
                        print("new_df['key'][%i] already sorted by eye as %s so not set to %s"%(row_index, new_df['key'][row_index] , sort_key))


                if airplane_notes != '' and sort_key_notes != '':
                    note = airplane_notes + ' - ' + sort_key_notes
                else:
                    note = airplane_notes + sort_key_notes

                if new_df['notes'].fillna('').to_numpy()[row_index] != '':
                    note = new_df['notes'].fillna('').to_numpy()[row_index] + ' - ' + note

                new_df['notes'][row_index] = note


            writeDataFrameToExcel(new_df, outfile, sheet_name, format_string="%0.10f")
