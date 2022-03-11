#!/usr/bin/env python3
'''
This takes the collated excel file and recreates it while maintain certain columns that were typed in.  This was originally
written because the data was not stored with high precision. 
'''

import sys
import os
import inspect
import h5py
import copy
from pprint import pprint
import time

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

impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'

if __name__ == '__main__':
    plt.close('all')
    start_time = time.time()
    infile = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'sept2021-week1-analysis', 'event_info_collated.xlsx')
    outfile = infile.replace('.xlsx','_%i.xlsx'%start_time)
    #Load in current excel sheet
    #df = pd.read_excel(os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'sept2021-week1-analysis', 'event_info_collated.xlsx'))

    with pd.ExcelFile(infile) as xls:
        for sheet_index, sheet_name in enumerate(xls.sheet_names):
            df = xls.parse(sheet_name)
            if sheet_index == 0:
                runs = numpy.unique(df['run'])
                ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, verbose_setup=False)

            new_df = df.copy(deep=True)
            #Get eventid dict:
            eventids_dict = {}
            for run in numpy.unique(df['run']):
                eventids_dict[run] = numpy.asarray(df.query('run == %i'%run)['eventid'],dtype=int)

            for col in df.columns:
                print('Processing Column %s'%col)
                if col in ['run','eventid','key','monutau','notes','suspected_airplane_icao24']:
                    continue
                else:
                    new_df[col] = ds.getDataArrayFromParam(col, trigger_types=None, eventids_dict=eventids_dict)

            writeDataFrameToExcel(new_df, outfile, sheet_name, format_string="%0.10f")







