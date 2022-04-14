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

    for channel in range(8):
        plt.figure()
        plt.title('Channel %i'%channel)
        for key in template_dict.keys():
            if channel == 0:
                print('%s len(t) = %i, max(t) = %i'%(key, len(template_dict[key]['t']), template_dict[key]['t'][-1]))
            plt.subplot(2,1,1)
            plt.ylabel('Templates')
            plt.plot(template_dict[key]['t'],template_dict[key]['wfs'][channel], label=key, alpha=0.7)
            plt.legend()
            plt.subplot(2,1,2)
            plt.ylabel('Filtered Templates')
            plt.plot(filtered_template_dict[key]['t'],filtered_template_dict[key]['wfs'][channel], label=key, alpha=0.7)
            plt.legend()






