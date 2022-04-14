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

impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'

template_dict = {}

# Load cranberry waveforms
cranberry_waveform_paths = glob.glob(os.path.join(os.environ['BEACON_ANALYSIS_DIR'] , 'analysis', 'cr_search', 'cranberry_analysis', '2-11-2022') + '/*.npy')
for path in cranberry_waveform_paths:
    event_name = os.path.split(path)[-1].replace('.npy','')
    template_dict['cranberry_%s'%event_name] = {}
    with open(path, 'rb') as f:
        template_dict['cranberry_%s'%event_name]['args'] = numpy.load(f, allow_pickle=True)
        channel0 = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['t']   = 2.0*numpy.arange(len(channel0))
        template_dict['cranberry_%s'%event_name]['wfs'] = numpy.zeros((8, len(channel0)))
        template_dict['cranberry_%s'%event_name]['wfs'][0] = channel0
        template_dict['cranberry_%s'%event_name]['wfs'][1] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][2] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][3] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][4] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][5] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][6] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][7] = numpy.load(f)

# Load the strong cr candidate 5911 73399
for run, eventid in [[5911,73399]]:
    template_dict['%i-%i'%(run,eventid)] = {}
    reader = Reader(raw_datapath, run)
    reader.setEntry(eventid)
    template_dict['%i-%i'%(run,eventid)]['t']   = reader.t()
    template_dict['%i-%i'%(run,eventid)]['wfs'] = numpy.zeros((8, len(template_dict['%i-%i'%(run,eventid)]['t'])))
    template_dict['%i-%i'%(run,eventid)]['wfs'][0] = reader.wf(0) - numpy.mean(reader.wf(0))
    template_dict['%i-%i'%(run,eventid)]['wfs'][1] = reader.wf(1) - numpy.mean(reader.wf(1))
    template_dict['%i-%i'%(run,eventid)]['wfs'][2] = reader.wf(2) - numpy.mean(reader.wf(2))
    template_dict['%i-%i'%(run,eventid)]['wfs'][3] = reader.wf(3) - numpy.mean(reader.wf(3))
    template_dict['%i-%i'%(run,eventid)]['wfs'][4] = reader.wf(4) - numpy.mean(reader.wf(4))
    template_dict['%i-%i'%(run,eventid)]['wfs'][5] = reader.wf(5) - numpy.mean(reader.wf(5))
    template_dict['%i-%i'%(run,eventid)]['wfs'][6] = reader.wf(6) - numpy.mean(reader.wf(6))
    template_dict['%i-%i'%(run,eventid)]['wfs'][7] = reader.wf(7) - numpy.mean(reader.wf(7))



if __name__ == '__main__':
    plt.close('all')
    start_time = time.time()
    infile = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated.xlsx')#event_info_collated_1647378994.xlsx
    outfile = infile.replace('.xlsx','_%i.xlsx'%start_time)
    #Load in current excel sheet
    #df = pd.read_excel(os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'sept2021-week1-analysis', 'event_info_collated.xlsx'))


    if False:
        #This section increases the resolution of dataslicer friendly variables.
        with pd.ExcelFile(infile) as xls:
            for sheet_index, sheet_name in enumerate(xls.sheet_names):
                df = xls.parse(sheet_name)
                df = df.sort_values(by = ['run', 'eventid'], ascending = [True, True], na_position = 'last')
                if sheet_index == 0:
                    runs = numpy.unique(df['run'][df['run'].notnull()]).astype(int)
                    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, verbose_setup=False)

                new_df = df.copy(deep=True)
                #Get eventid dict:
                eventids_dict = {}
                for run in numpy.unique(df['run'][df['run'].notnull()]).astype(int):
                    eventids_dict[run] = numpy.asarray(df.query('run == %i'%run)['eventid'],dtype=int)

                for col in df.columns:
                    print('Processing Column %s'%col)
                    if col in ['run','eventid','key','monutau','notes','suspected_airplane_icao24'] or 'template_' in col:
                        continue
                    else:
                        new_df[col] = ds.getDataArrayFromParam(col, trigger_types=None, eventids_dict=eventids_dict)

                writeDataFrameToExcel(new_df, outfile, sheet_name, format_string="%0.10f")
    if True:
        #Similar to the above but only reprocesses the specified columns, and will add new data as well.

        reprocess = ['std_h', 'std_v', 'snr_h', 'snr_v', 'p2p_h', 'p2p_v', 'p2p_gap_h', 'p2p_gap_v', 'csnr_h', 'csnr_v']
        with pd.ExcelFile(infile) as xls:
            for sheet_index, sheet_name in enumerate(xls.sheet_names):
                df = xls.parse(sheet_name)
                df = df.sort_values(by = ['run', 'eventid'], ascending = [True, True], na_position = 'last')
                df = df[df.isnull().sum(axis=1) < 10]
                if sheet_index == 0:
                    runs = numpy.unique(df['run'][df['run'].notnull()]).astype(int)
                    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, verbose_setup=False)

                new_df = df.copy(deep=True)
                #Get eventid dict:
                eventids_dict = {}
                for run in numpy.unique(df['run'][df['run'].notnull()]).astype(int):
                    eventids_dict[run] = numpy.asarray(df.query('run == %i'%run)['eventid'],dtype=int)

                for col in df.columns:
                    print('Processing Column %s'%col)
                    if col in ['run','eventid','key','monutau','notes','suspected_airplane_icao24'] or 'template_' in col:
                        continue
                    elif col not in reprocess:
                        continue
                    elif col not in ds.data_slicers[0].known_param_keys:
                        continue
                    else:
                        new_df[col] = ds.getDataArrayFromParam(col, trigger_types=None, eventids_dict=eventids_dict)

                #Add new data.
                add_data = ['filtered_std_h', 'filtered_std_v', 'filtered_snr_h', 'filtered_snr_v', 'filtered_p2p_h', 'filtered_p2p_v', 'filtered_p2p_gap_h', 'filtered_p2p_gap_v', 'filtered_csnr_h', 'filtered_csnr_v']
                insert_at = list(new_df.columns).index('std_h')
                for col in add_data:
                    if not numpy.isin(col, new_df.columns):
                        print('Adding %s'%col)
                        new_df.insert(insert_at, col, ds.getDataArrayFromParam(col, trigger_types=None, eventids_dict=eventids_dict))
                writeDataFrameToExcel(new_df, outfile, sheet_name, format_string="%0.10f")

    elif False:
        #This adds the airplane column to other sheets, and does some extra calculations on the cross correlations. 
        with pd.ExcelFile(infile) as xls:
            good_with_airplanes = xls.parse('good-with-airplane')
            good_with_airplanes['suspected_airplane_icao24'] = good_with_airplanes['suspected_airplane_icao24'].fillna(value='')

            for sheet_index, sheet_name in enumerate(xls.sheet_names):
                df = xls.parse(sheet_name)
                new_df = df.copy(deep=True)
                #Get eventid dict:
                eventids_dict = {}
                for run in numpy.unique(df['run'][df['run'].notnull()]).astype(int):
                    eventids_dict[run] = numpy.asarray(df.query('run == %i'%run)['eventid'],dtype=int)

                if not numpy.isin('suspected_airplane_icao24', new_df.columns):
                    new_df.insert(list(good_with_airplanes.columns).index('suspected_airplane_icao24'), 'suspected_airplane_icao24' , pd.Series(dtype=good_with_airplanes.dtypes['suspected_airplane_icao24']))

                #COpy from master airplane sheet
                for row_index, row in new_df.iterrows():
                    airplane = ''
                    try:
                        airplane = str(numpy.asarray(good_with_airplanes.query('run == %i & eventid == %i'%(row['run'],row['eventid']))['suspected_airplane_icao24'])[0])
                    except:
                        airplane = ''

                    new_df['suspected_airplane_icao24'][row_index] = airplane

                for key in list(template_dict.keys()):
                    key = 'template_' + key
                    if not numpy.isin(key+'_all_normalized', new_df.columns):
                        vals = (new_df[key+'_hpol'] + new_df[key+'_vpol'])/2
                        norm_vals = (vals - min(vals)) / (max(vals) - min(vals))

                        new_df.insert(list(new_df.columns).index(key+'_vpol') + 1, key+'_all_normalized' , norm_vals)
                        new_df.insert(list(new_df.columns).index(key+'_vpol') + 1, key+'_all' , vals)

                writeDataFrameToExcel(new_df, outfile, sheet_name, format_string="%0.10f")
    elif False:
        #This section will add a new column if it does not already exist.
        with pd.ExcelFile(infile) as xls:
            for sheet_index, sheet_name in enumerate(xls.sheet_names):
                if sheet_name == 'stage_2_pass':
                    df = xls.parse(sheet_name)
                    runs = numpy.unique(df['run'][df['run'].notnull()]).astype(int)
                    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, verbose_setup=False)

            for sheet_index, sheet_name in enumerate(xls.sheet_names):
                print('On %s'%sheet_name)
                df = xls.parse(sheet_name)
                new_df = df.copy(deep=True)
                insert_at = list(df.columns).index('phi_best_choice')

                #Get eventid dict:
                eventids_dict = {}
                for run in numpy.unique(df['run'][df['run'].notnull()]).astype(int):
                    eventids_dict[run] = numpy.asarray(df.query('run == %i'%run)['eventid'],dtype=int)

                add_data = ['calibrated_trigtime']

                for col in add_data:
                    if not numpy.isin(col, new_df.columns):
                        print('Adding %s'%col)
                        new_df.insert(insert_at, col, ds.getDataArrayFromParam(col, trigger_types=None, eventids_dict=eventids_dict))

                writeDataFrameToExcel(new_df, outfile, sheet_name, format_string="%0.10f")






