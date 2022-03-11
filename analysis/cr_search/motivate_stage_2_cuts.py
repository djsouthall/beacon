#!/usr/bin/env python3
'''
This is meant to take the events that were sorted by eye in batch 0, and see if there are obvious cuts that can be
applied ot auto sort stage 1 events into stage 2: good, maybe, bad.

This is a scrappy script that depends on highly specific file sortings I have done by hand.
'''

import sys
import os
import inspect
import h5py
import copy
from pprint import pprint

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
from beacon.tools.flipbook_reader import flipbookToDict, concatenateFlipbookToArray, concatenateFlipbookToDict

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


'''
'impulsivity_h','impulsivity_v', 'cr_template_search_h', 'cr_template_search_v', 'std_h', 'std_v', 'p2p_h', 'p2p_v', 'snr_h', 'snr_v',\
'time_delay_0subtract1_h','time_delay_0subtract2_h','time_delay_0subtract3_h','time_delay_1subtract2_h','time_delay_1subtract3_h','time_delay_2subtract3_h',\
'time_delay_0subtract1_v','time_delay_0subtract2_v','time_delay_0subtract3_v','time_delay_1subtract2_v','time_delay_1subtract3_v','time_delay_2subtract3_v',
'cw_present','cw_freq_Mhz','cw_linear_magnitude','cw_dbish','theta_best_h','theta_best_v','elevation_best_h','elevation_best_v','phi_best_h','phi_best_v',\
'calibrated_trigtime','triggered_beams','beam_power','hpol_peak_to_sidelobe','vpol_peak_to_sidelobe','hpol_max_possible_map_value','vpol_max_possible_map_value',\
'map_max_time_delay_0subtract1_h','map_max_time_delay_0subtract2_h','map_max_time_delay_0subtract3_h',\
'map_max_time_delay_1subtract2_h','map_max_time_delay_1subtract3_h','map_max_time_delay_2subtract3_h',\
'map_max_time_delay_0subtract1_v','map_max_time_delay_0subtract2_v','map_max_time_delay_0subtract3_v',\
'map_max_time_delay_1subtract2_v','map_max_time_delay_1subtract3_v','map_max_time_delay_2subtract3_v'
'''




if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length

    runs = numpy.arange(5733,5974) # September data
    bad_runs = numpy.array([])

    flipbook_path = '/home/dsouthall/scratch-midway2/event_flipbook_1643154940'
    sorted_dict = flipbookToDict(flipbook_path)

    sorted_array = concatenateFlipbookToArray(sorted_dict)
    flipbook_path = '/home/dsouthall/Projects/Beacon/beacon/analysis/sept2021-week1-analysis/airplane_event_flipbook_1643947072'
    sorted_dict2 = flipbookToDict(flipbook_path)
    sorted_array2 = concatenateFlipbookToArray(sorted_dict2)

    #These are ones that fell into good, but not "very-good", which means they aren't actually good but were suspected to be associated with events like lightning.  Bad for sorting purposes here.
    cut = sorted_array['key'] == 'good'
    sorted_array[cut] = sorted_array[cut]
    sorted_array['key'][cut] = 'bad'

    for event_index, event in enumerate(sorted_array):
        run = event['run']
        eventid = event['eventid']
        if numpy.logical_and(numpy.isin(run,sorted_array2['run']), numpy.isin(eventid,sorted_array2['eventid'])):
            # If was further sorted, use new info.
            sorted_array[event_index] = sorted_array2[numpy.logical_and(sorted_array2['run'] == run, sorted_array2['eventid'] == eventid)]

    simplified_sorted_array = numpy.zeros_like(sorted_array)

    #Lump airplanes and stuff together.
    for key in numpy.unique(sorted_array['key']):
        cut = sorted_array['key'] == key
        if key in ['maybe','bad','lightning']:
            simplified_sorted_array[cut] = sorted_array[cut]
        elif len(key.replace('maybe-','')) == 6 or key == 'maybe-airplanes':
            #Probably an airplane
            simplified_sorted_array[cut] = sorted_array[cut]
            simplified_sorted_array['key'][cut] = 'airplane'
        elif 'no-obvious-airpl' in key:
            simplified_sorted_array[cut] = sorted_array[cut]
            simplified_sorted_array['key'][cut] = 'good'


    runs = runs[~numpy.isin(runs,bad_runs)]
    runs = runs[numpy.isin(runs,numpy.unique(sorted_array['run']))]

    # runs = runs[0:10] #for testing

    print("Preparing dataSlicer")

    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath)

    if False:
        ds.addROI('interest',{'cr_template_search_h':[0.8,0.94],'cr_template_search_v':[0.42,0.52]})
        roi_eventid_dict = ds.getCutsFromROI('interest',eventids_dict=concatenateFlipbookToDict(sorted_dict),load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
        ds.eventInspector(roi_eventid_dict)
        ds.eventInspector({5911:[73399]})
    if True:
        try:
            #cut_roi = {'hpol_normalized_map_value_abovehorizon':[0.75,10],'vpol_normalized_map_value_abovehorizon':[0.55,10], 'cr_template_search_h':[0.5,10], 'cr_template_search_v':[0.3,10]}
            cut_roi = {'above_normalized_map_max_line':[0,10], 'above_snr_line':[0,10000], 'p2p_gap_h':[-1, 95]}

            #plot_params = [['hpol_peak_to_sidelobe','vpol_peak_to_sidelobe'],['cr_template_search_h', 'cr_template_search_v'], ['hpol_normalized_map_value_abovehorizon','vpol_normalized_map_value_abovehorizon'], ['snr_h', 'snr_v']]
            plot_params = [['p2p_gap_h','p2p_gap_v']]
            #plot_params = [['min_csnr_h','min_csnr_v'], ['csnr_h','csnr_v'], ['snr_h', 'snr_v'], ['min_std_h','min_std_v'],['hpol_peak_to_sidelobe','vpol_peak_to_sidelobe'],['cr_template_search_h', 'cr_template_search_v'], ['impulsivity_h','impulsivity_v'], ['hpol_normalized_map_value_abovehorizon','vpol_normalized_map_value_abovehorizon'], ['hpol_peak_to_sidelobe','cr_template_search_h'],['vpol_peak_to_sidelobe','cr_template_search_v']]
            for apply_cuts in [False,True]:

                for key_x, key_y in plot_params:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    
                    ds.setCurrentPlotBins(key_x, key_y, None)
                    fig, ax = plt.subplots()
                    ax.set_xlim(min(ds.current_bin_edges_x), max(ds.current_bin_edges_x))
                    ax.set_xlabel(ds.current_label_x)
                    ax.set_ylim(min(ds.current_bin_edges_y), max(ds.current_bin_edges_y))
                    ax.set_ylabel(ds.current_label_y)

                    plt.grid(which='both', axis='both')
                    ax.minorticks_on()
                    ax.grid(b=True, which='major', color='k', linestyle='-')
                    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                    kept_events = 0
                    removed_events = numpy.array([],dtype=simplified_sorted_array.dtype)

                    for key in numpy.unique(simplified_sorted_array['key']):
                        events = simplified_sorted_array[simplified_sorted_array['key'] == key]
                        eventids_dict = {}


                        _kept_events = 0
                        for run in numpy.unique(events['run']):
                            if run in runs:
                                keep_cut = numpy.ones(len(events[events['run'] == run]), dtype=bool)
                                if apply_cuts:
                                    for cut_key, cut_range in cut_roi.items():
                                        param = ds.getDataArrayFromParam(cut_key, eventids_dict={run:events[events['run'] == run]['eventid']})    
                                        keep_cut = numpy.logical_and(keep_cut, numpy.logical_and(param >= min(cut_range), param <= max(cut_range)))                    

                                kept_events += numpy.sum(keep_cut)
                                _kept_events += numpy.sum(keep_cut)
                                eventids_dict[run] = events[events['run'] == run]['eventid'][keep_cut]
                                removed_events = numpy.append(removed_events,events[events['run'] == run][~keep_cut])

                        print('Events in category %s passing cuts is: %i/%i'%(key, _kept_events, len(events)))

                        x = ds.getDataArrayFromParam(key_x, eventids_dict=eventids_dict)
                        y = ds.getDataArrayFromParam(key_y, eventids_dict=eventids_dict)
                        plt.scatter(x,y, label=key, alpha=0.7)
                    plt.legend(loc = 'best')

                    print('Events passing cuts is: %i/%i = %0.2f%%'%(kept_events, len(simplified_sorted_array), 100*kept_events/len(simplified_sorted_array)))

                if apply_cuts == True:
                    #Inspect any events being cut that maybe shouldn't be
                    interesting_events = removed_events[removed_events['key'] != 'bad']
                    interesting_dict = {}
                    for run in numpy.unique(interesting_events['run']):
                        interesting_dict[run] = interesting_events[interesting_events['run'] == run]['eventid']

                    ds.eventInspector(interesting_dict)



            for event in simplified_sorted_array[simplified_sorted_array['key'] == 'good']:
                if event['run'] in runs:
                    t = ds.getDataArrayFromParam('calibrated_trigtime', eventids_dict={event['run']:[event['eventid']]})[0]
                    print(event, ' calibrated trigtime: ', t)


            # if False:
            #     for key in numpy.unique(simplified_sorted_array['key']):
            #         events = simplified_sorted_array[simplified_sorted_array['key'] == key]
            #         eventids_dict = {}
            #         for run in numpy.unique(events['run']):
            #             if run in runs:
            #                 eventids_dict[run] = events[events['run'] == run]['eventid'][keep_cut]


            #         ds.addROI(key,{'cr_template_search_h':[0.8,0.94],'cr_template_search_v':[0.42,0.52]})
            #         roi_eventid_dict = ds.getCutsFromROI(key,eventids_dict=eventids_dict,load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
            #         ds.eventInspector(roi_eventid_dict)


        except Exception as e:

            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

