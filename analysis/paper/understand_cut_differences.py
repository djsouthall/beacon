#!/usr/bin/env python3
'''
This script is intended to look at the events that construct best above horizon in allsky maps.  I want to view the
peak to sidelobe values and other parameters for the belowhorizon and abovehorizon maps and determine if there is
an obvious cut for which sidelobed above horizon events can be discriminated. 
'''

import sys
import os
import gc
import inspect
import h5py
import copy
from pprint import pprint
import textwrap
from multiprocessing import Process

import numpy
import scipy
import scipy.signal
import scipy.signal

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.line_of_sight import circleSource
from beacon.tools.get_sun_coords_from_timestamp import getSunElWeightsFromRunDict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, FuncFormatter
import matplotlib.dates as mdates
import time
from datetime import datetime
import pytz
import pandas as pd

plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

def saveprint(*args, outfile=None):
    print(*args) #Print once to command line
    if outfile is not None:
        print(*args, file=open(outfile,'a')) #Print once to file


start_time = time.time()

raw_datapath = os.environ['BEACON_DATA']

gc.enable()

processed_datapath = os.environ['BEACON_PROCESSED_DATA']

if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length
    
    run_batches = {}
    run_batches['batch_0'] = numpy.arange(5733,5974) # September data, should setup to auto add info to the "notes" section based off of existing sorting, and run this one those events for consistency
    run_batches['batch_1'] = numpy.arange(5974,6073)
    run_batches['batch_2'] = numpy.arange(6074,6173)
    run_batches['batch_3'] = numpy.arange(6174,6273)
    run_batches['batch_4'] = numpy.arange(6274,6373)
    run_batches['batch_5'] = numpy.arange(6374,6473)
    run_batches['batch_6'] = numpy.arange(6474,6573)
    run_batches['batch_7'] = numpy.arange(6574,6641)
    runs = numpy.array([])
    for k in (run_batches.keys()):
        runs = numpy.append(runs,run_batches[k])

    runs = runs.astype(int)

    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, \
                    low_ram_mode=True,\
                    analysis_data_dir=processed_datapath, trigger_types=[2], remove_incomplete_runs=True)


    total_time = 0
    for run in runs:
        sys.stdout.write('Run %i\r'%(run))
        sys.stdout.flush()
        reader = Reader(raw_datapath, int(run))
        t = ds.getDataArrayFromParam('calibrated_trigtime', eventids_dict={run:[0,reader.N()-1]})
        total_time += t[1] - t[0]

    print('Total Time : %0.2f s'%total_time)
    print('Minutes:\t', total_time/60)
    print('Hours:\t', total_time/3600)
    print('Days:\t', total_time/(24*3600))

    ds.addROI('cuts', {   'elevation_best_choice':[10,90],
                                'phi_best_choice':[-90,90],
                                'similarity_count_h':[-0.1,10],
                                'similarity_count_v':[-0.1,10],
                                'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':[1.2,1e10],
                                'impulsivity_hSLICERADDimpulsivity_v':[0.3,1e10],
                                'cr_template_search_hSLICERMAXcr_template_search_v':[0.4,1e10],
                                'in_targeted_box':[-0.5, 0.5],
                                'p2p_gap_h':[-1e10, 95],
                                'above_normalized_map_max_line':[0,1e10],
                                'above_snr_line':[0,1e10]})

    plot_params = [['phi_best_choice', 'elevation_best_choice'], ['similarity_count_h', 'similarity_count_v'], ['hpol_peak_to_sidelobe','vpol_peak_to_sidelobe'], ['impulsivity_h','impulsivity_v'], ['cr_template_search_h','cr_template_search_v'], ['above_normalized_map_max_line','above_snr_line']]

    if True:
        new_cut_dict = numpy.load( os.path.join( '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/cuts_run5733-run6640_1652152119' , 'pass_all_cuts_eventids_dict.npy')  , allow_pickle=True)[()]
    else:
        new_cut_dict = numpy.load( os.path.join( '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/cuts_run5733-run6640_1652133421' , 'pass_all_cuts_eventids_dict.npy')  , allow_pickle=True)[()]


    old_cut_dicts = {}
    for i in range(8):
        f = os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/eventid_dicts', 'stage_2_eventids_dict_batch_%i.npy'%i)
        out = numpy.load( f , allow_pickle=True)[()]
        for run in list(out.keys()):
            old_cut_dicts[run] = out[run]

    new_cut_array = ds.organizeEventDict(new_cut_dict)
    old_cut_array = ds.organizeEventDict(old_cut_dicts)

    matching                        = ds.organizeEventDict(ds.returnCommonEvents(new_cut_dict, old_cut_dicts))
    matching_dict                   = ds.returnCommonEvents(new_cut_dict, old_cut_dicts)
    
    events_in_new_not_old         = ds.organizeEventDict(ds.returnEventsAWithoutB(new_cut_dict, old_cut_dicts))
    events_in_new_not_old_dict    = ds.returnEventsAWithoutB(new_cut_dict, old_cut_dicts)


    if False:
        print('Getting combined impulsivity')
        new_impulsivity = ds.getDataArrayFromParam('impulsivity_hSLICERADDimpulsivity_v', eventids_dict=new_cut_dict)
        events_by_interest = new_cut_array[numpy.argsort(new_impulsivity)[::-1]]
        ds.eventInspector({6027:[53175], 6303:[102086], 5911:[73399]})
        # 6027:[53175] flagged as part of a cluster of events potentially an airplane but not sure
        # 6303:[102086] believed to be a below horizon source, reconstructed above horizon due to coincident event
    
    events_in_old_not_new         = ds.organizeEventDict(ds.returnEventsAWithoutB(old_cut_dicts, new_cut_dict))
    events_in_old_not_new_dict    = ds.returnEventsAWithoutB(old_cut_dicts, new_cut_dict)

    if False:
        for param_key in list(ds.roi['cuts'].keys()):
            fig = plt.figure()
            ax = plt.gca()
            plt.title(param_key)
            for index, (d, name) in enumerate([[matching_dict, 'overlap'], [events_in_new_not_old_dict, 'gained'], [events_in_old_not_new_dict, 'lost']]):
                if index == 0:
                    fig, ax = ds.plot1dHist(param_key, d, cumulative=None, pdf=False, title=None, lognorm=True, return_counts=False, return_only_count_details=False, label=name, alpha=0.7)
                else:
                    fig, ax = ds.plot1dHist(param_key, d, cumulative=None, pdf=False, title=None, lognorm=True, return_counts=False, return_only_count_details=False, ax=ax, label=name, alpha=0.7)
            plt.legend()


    print('matching\t',len(matching))
    print('events_in_new_not_old\t',len(events_in_new_not_old))
    print('events_in_old_not_new\t',len(events_in_old_not_new))

    df_good = pd.read_excel(os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated.xlsx'), sheet_name='good-with-airplane')
    df_ambiguous = pd.read_excel(os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated.xlsx'), sheet_name='ambiguous')
    df_bad = pd.read_excel(os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated.xlsx'), sheet_name='bad')

    good_cut = df_good[~numpy.logical_and(numpy.isin(df_good['run'], matching['run']), numpy.isin(df_good['eventid'], matching['eventid']))]
    ambiguous_cut = df_ambiguous[~numpy.logical_and(numpy.isin(df_ambiguous['run'], matching['run']), numpy.isin(df_ambiguous['eventid'], matching['eventid']))]
    bad_cut = df_bad[~numpy.logical_and(numpy.isin(df_bad['run'], matching['run']), numpy.isin(df_bad['eventid'], matching['eventid']))]


    would_need_to_reexamine = new_cut_array[numpy.isin(new_cut_array['run'], exclude_runs)]


    print('If excluding runs that are not backed up, the final statistical breakdown becomes:')

    df_good_reduced = df_good[~numpy.isin(df_good['run'], exclude_runs)]
    print('len(df_good) %i, len(df_good_reduced) %i:  %i -- > %i'%(len(df_good), len(df_good_reduced), len(df_good), len(df_good_reduced)))
    print('%0.2f lost by excluding runs not backed up'%(100 - 100*len(df_good_reduced)/len(df_good)))


    df_ambiguous_reduced = df_ambiguous[~numpy.isin(df_ambiguous['run'], exclude_runs)]
    print('len(df_ambiguous) %i, len(df_ambiguous_reduced) %i:  %i -- > %i'%(len(df_ambiguous), len(df_ambiguous_reduced), len(df_ambiguous), len(df_ambiguous_reduced)))
    print('%0.2f lost by excluding runs not backed up'%(100 - 100*len(df_ambiguous_reduced)/len(df_ambiguous)))


    df_bad_reduced = df_bad[~numpy.isin(df_bad['run'], exclude_runs)]
    print('len(df_bad) %i, len(df_bad_reduced) %i:  %i -- > %i'%(len(df_bad), len(df_bad_reduced), len(df_bad), len(df_bad_reduced)))
    print('%0.2f lost by excluding runs not backed up'%(100 - 100*len(df_bad_reduced)/len(df_bad)))



    # print('good_cut: ', len(good_cut))
    # print('ambiguous_cut: ', len(ambiguous_cut))
    # print('bad_cut: ', len(bad_cut))

    # for param_x, param_y in plot_params:
    #     for index, (d, name) in enumerate([[matching_dict, 'overlap'], [events_in_new_not_old_dict, 'gained'], [events_in_old_not_new_dict, 'lost']]):
    #         ds.plot2dHist(param_x, param_y, d, title=name, cmap='coolwarm', return_counts=False,lognorm=True, fig=None, ax=None)