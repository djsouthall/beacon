#!/usr/bin/env python3
'''
This will cross correlate the events found in the event_info_collated.xlsx spreadsheet against several templates,
storing the results.  
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
from examples.beacon_data_reader import Reader as RawReader #Without sine subtraction
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TimeDelayCalculator
# from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
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
filtered_template_dict = {} #will be changed throughout the script
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

# Load cranberry waveforms
cranberry_waveform_paths = glob.glob(os.path.join(os.environ['BEACON_ANALYSIS_DIR'] , 'analysis', 'cr_search', 'cranberry_analysis', '2-11-2022') + '/*.npy')
cranberry_waveform_paths = numpy.array(cranberry_waveform_paths)[numpy.argsort([int(i.split('/')[-1].replace('.npy','').replace('event','')) for i in cranberry_waveform_paths])]
for path in cranberry_waveform_paths:
    event_name = os.path.split(path)[-1].replace('.npy','')
    template_dict['cranberry_%s'%event_name] = {}
    with open(path, 'rb') as f:
        template_dict['cranberry_%s'%event_name]['args'] = numpy.load(f, allow_pickle=True).flatten()[0]
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



crit_freq_low_pass_MHz = 80
low_pass_filter_order = 14

crit_freq_high_pass_MHz = 20
high_pass_filter_order = 4

plot_filter=False
sine_subtract = True
sine_subtract_min_freq_GHz = 0.00
sine_subtract_max_freq_GHz = 0.25
sine_subtract_percent = 0.03
max_failed_iterations = 5

apply_phase_response = True

shorten_signals = False
shorten_thresh = 0.7
shorten_delay = 10.0
shorten_length = 90.0

notch_tv = True
misc_notches = True
# , notch_tv=notch_tv, misc_notches=misc_notches

align_method = None

hilbert=False
final_corr_length = 2**17

def loadTriggerTypes(reader):
    '''
    Will get a list of trigger types corresponding to all eventids for the given reader
    trigger_type:
    1 Software
    2 RF
    3 GPS
    '''
    #trigger_type = numpy.zeros(reader.N())
    '''
    N = reader.head_tree.Draw("raw_approx_trigger_time_nsecs:raw_approx_trigger_time:trig_time:Entry$","","goff") 
    #ROOT.gSystem.ProcessEvents()
    subtimes = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N)
    times = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N) 
    trigtimes = numpy.frombuffer(reader.head_tree.GetV3(), numpy.dtype('float64'), N)
    eventids = numpy.frombuffer(reader.head_tree.GetV4(), numpy.dtype('float64'), N).astype(int)
    '''


    try:
        N = reader.head_tree.Draw("trigger_type","","goff") 
        trigger_type = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N).astype(int)
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print('Error while trying to copy header elements to attrs.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return e
    return trigger_type

all_runs = numpy.arange(5733,6641,dtype=int)
def getTriggerStatistics(runs):
    count_1 = 0
    count_2 = 0
    count_3 = 0
    for run in runs:
        reader = RawReader(raw_datapath, run)
        trigger_type = loadTriggerTypes(reader)
        count_1 += numpy.sum(trigger_type == 1)
        count_2 += numpy.sum(trigger_type == 2)
        count_3 += numpy.sum(trigger_type == 3)
    return count_1, count_2, count_3


if __name__ == '__main__':
    plt.close('all')
    start_time = time.time()
    infile = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated.xlsx')
    outfile = infile.replace('.xlsx','_%i.xlsx'%start_time)
    main_sheet_name = 'good-with-airplane'

    plot = True

    #Create initial tdc
    reader = Reader(raw_datapath, 5911)
    tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=(None,None),plot_filters=plot_filter,apply_phase_response=apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches)
    if sine_subtract:
        tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=max_failed_iterations, verbose=False, plot=False)

    for key in template_dict.keys():
        filtered_template_dict[key] = {}
        filtered_template_dict[key]['t'] = tdc.t()
        filtered_template_dict[key]['wfs'] = numpy.zeros((8, len(filtered_template_dict[key]['t'])))
        filtered_template_dict[key]['std'] = numpy.zeros(8)
        for channel in range(8):
            filtered_template_dict[key]['wfs'][channel] = tdc.applyFilterToGivenWF(template_dict[key]['wfs'][channel], channel, apply_filter=True, hilbert=False, tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True) #this doesn't need to be done per event for the template, but I am doing it for consistency as this isn't a super intensive stage of the analysis
            filtered_template_dict[key]['std'][channel] = filtered_template_dict[key]['wfs'][channel].std()

    align = True
    for channel in range(8):
        if channel != 4:
            continue
        plt.figure()
        plt.suptitle('Channel %i'%channel)
        for key in template_dict.keys():
            if 'args' in list(template_dict[key].keys()):
                if template_dict[key]['args']['energy'] < 18.5:
                    continue
                label = 'Sample Cranberry Event at $10^{%i}$ eV'%template_dict[key]['args']['energy']
            else:
                label = 'Candidate Event %s'%key
            if channel == 0:
                print('%s len(t) = %i, max(t) = %i'%(key, len(template_dict[key]['t']), template_dict[key]['t'][-1]))
            plt.subplot(2,1,1)
            plt.ylabel('Amplitude (adu)', fontsize=16)
            if align == True:
                if '5911' in key:
                    offset = template_dict[key]['t'][numpy.argmax(template_dict[key]['wfs'][channel])]
                else:
                    offset = template_dict[key]['t'][numpy.argmax(template_dict[key]['wfs'][channel])+1]
            else:
                offset = 0
            plt.plot(template_dict[key]['t']  - offset, template_dict[key]['wfs'][channel], label=label, alpha=0.7)
            plt.legend(loc='upper right', fontsize = 12)
            plt.ylim(-90,80)
            if align == True:
                plt.xlim(-200,500)
                plt.xlabel('Time (ns)\nAligned to Max Amplitude', fontsize=16)
            else:
                plt.xlabel('Time (ns)', fontsize=16)
                plt.xlim(100,1000)
            plt.subplot(2,1,2)
            plt.ylabel('Filtered Amplitude (adu)', fontsize=16)
            if align == True:
                offset = filtered_template_dict[key]['t'][numpy.argmax(filtered_template_dict[key]['wfs'][channel])]
            else:
                offset = 0
            plt.plot(filtered_template_dict[key]['t'] - offset, filtered_template_dict[key]['wfs'][channel], label=None, alpha=0.7)
            plt.ylim(-90,80)
            if align == True:
                plt.xlim(-200,500)
                plt.xlabel('Time (ns)\nAligned to Max Amplitude', fontsize=16)
            else:
                plt.xlabel('Time (ns)', fontsize=16)
                plt.xlim(100,1000)

    if False:
        count_1, count_2, count_3 = getTriggerStatistics(all_runs)





