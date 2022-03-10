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
import pandas as pd

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
    notes = numpy.empty(simplified_sorted_array.shape[0], dtype=numpy.dtype(('U',100)))
    #simplified_sorted_array = numpy.lib.recfunctions.append_fields(simplified_sorted_array, 'notes', numpy.empty(simplified_sorted_array.shape[0], dtype=numpy.dtype(('U',100))), dtypes=numpy.dtype(('U',100)))
    #Lump airplanes and stuff together.
    for key in numpy.unique(sorted_array['key']):
        cut = sorted_array['key'] == key
        notes[cut] = key
        if key == 'maybe':
            simplified_sorted_array[cut] = sorted_array[cut]
            simplified_sorted_array['key'][cut] = 'ambiguous'
        elif key in 'bad':
            simplified_sorted_array[cut] = sorted_array[cut]
        elif key in 'lightning':
            simplified_sorted_array[cut] = sorted_array[cut]
            simplified_sorted_array['key'][cut] = 'bad'
        elif len(key.replace('maybe-','')) == 6 or key == 'maybe-airplanes':
            #Probably an airplane
            simplified_sorted_array[cut] = sorted_array[cut]
            simplified_sorted_array['key'][cut] = 'good'
        elif 'no-obvious-airpl' in key:
            simplified_sorted_array[cut] = sorted_array[cut]
            simplified_sorted_array['key'][cut] = 'good'

    
    stage2_pass = numpy.load('/home/dsouthall/Projects/Beacon/beacon/analysis/sept2021-week1-analysis/batch_0_1646861435/stage_2_eventids_dict_batch_0.npy',allow_pickle=True).flatten()[0]
    stage2_partial = numpy.load('/home/dsouthall/Projects/Beacon/beacon/analysis/sept2021-week1-analysis/batch_0_1646861435/partial_pass_stage_2_eventids_dict_batch_0.npy',allow_pickle=True).flatten()[0]
    stage2_failed = numpy.load('/home/dsouthall/Projects/Beacon/beacon/analysis/sept2021-week1-analysis/batch_0_1646861435/failed_stage_2_eventids_dict_batch_0.npy',allow_pickle=True).flatten()[0]
    
    pass_key = []
    monutau_links = []
    for event in simplified_sorted_array:
        if event['run'] in stage2_pass.keys() and event['eventid'] in stage2_pass[event['run']]:
            pass_key.append('pass')
        elif  event['run'] in stage2_partial.keys() and event['eventid'] in stage2_partial[event['run']]:
            pass_key.append('partial')
        elif  event['run'] in stage2_failed.keys() and event['eventid'] in stage2_failed[event['run']]:
            pass_key.append('failed')
        else:
            pass_key.append('unlabeled')

        url = "https://users.rcc.uchicago.edu/~cozzyd/monutau/#event&run=%i&entry=%i"%(event['run'], event['eventid'])
        monutau_links.append('=HYPERLINK("%s", "link")'%url)


    df = pd.DataFrame(simplified_sorted_array)
    df['monutau'] = monutau_links
    df['notes'] = notes
    df['pass_key'] = pass_key
    filename = 'quick_batch_0_excel.xlsx'
    with pd.ExcelWriter(filename, engine="openpyxl", mode='w') as writer:
        df.to_excel(writer,float_format="%0.2f", index=False)

    # pprint(df)