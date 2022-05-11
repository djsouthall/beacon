#!/usr/bin/env python3
'''
This will plot the polarization for a given run and eventid.
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
import textwrap

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



def maximizeAllFigures(x_ratio=1.0, y_ratio=1.0):
    '''
    Maximizes all matplotlib plots.
    '''
    for i in plt.get_fignums():
        plt.figure(i)
        plt.tight_layout()
        fm = plt.get_current_fig_manager()
        fm.resize(x_ratio*fm.window.maxsize()[0], y_ratio*fm.window.maxsize()[1])
        plt.tight_layout()

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
    if event_name != 'event000088':
        print('skipping ' , event_name)
        continue
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





if False:
    #Filters used in dataslicer correlator
    crit_freq_low_pass_MHz = 85
    low_pass_filter_order = 6

    crit_freq_high_pass_MHz = 25
    high_pass_filter_order = 8

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

    upsample = 2**14 #Just upsample in this case, Reduced to 2**14 when the waveform length was reduced, to maintain same time precision with faster execution.
    notch_tv = True
    misc_notches = True
else:
    #Standard values used for waveform analysis
    crit_freq_low_pass_MHz = 80
    low_pass_filter_order = 14

    crit_freq_high_pass_MHz = 20
    high_pass_filter_order = 4

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

    upsample = 2**17
    notch_tv = True
    misc_notches = True

# , notch_tv=notch_tv, misc_notches=misc_notches

align_method = None

hilbert=False
final_corr_length = upsample


mpl_colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

if __name__ == '__main__':
    if len(sys.argv) == 3:
        run = int(sys.argv[1])
        eventid = int(sys.argv[2])
    else:
        run = 5911
        eventid = 73399

    plt.close('all')
    start_time = time.time()
    infile = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated.xlsx')
    outfile = infile.replace('.xlsx','_%i.xlsx'%start_time)
    main_sheet_name = 'good-with-airplane'

    plot = True

    #Create initial tdc
    reader = Reader(raw_datapath, run)
    polarization_degs = []
    for notch_tv in [False, True]:
        tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=(None,None),plot_filters=False,apply_phase_response=apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches)
        if sine_subtract:
            tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=max_failed_iterations, verbose=False, plot=False)

        polarization_deg = tdc.calculatePolarizationFromTimeDelays(eventid, apply_filter=True, waveforms=None, plot=True, sine_subtract=True)
        polarization_degs.append(polarization_deg)
    polarization_deg = tdc.calculatePolarizationFromTimeDelays(eventid, apply_filter=False, waveforms=None, plot=True, sine_subtract=True)
    polarization_degs.append(polarization_deg)
    '''
    This data is a placeholder of the histogram data for polarization
    '''
    bin_centers = [2.5,7.5,12.5,17.5,22.5,27.5,32.5,37.5,42.5,47.5,52.5,57.5,62.5,67.5,72.5,77.5,82.5,87.5]
    vals = [0.000132, 0.000132, 0.0021, 0.0338, 0.268, 0.59, 0.109, 0.00145, 0.00145, 0.00145, 0.00145, 0.00145, 0.00145, 0.00145, 0.00145, 0.00145, 0.00145, 0.00145]
    bin_width = 5 #degrees
    fig = plt.figure(figsize=(25,10))

    ax = plt.subplot(1,3,1)
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='tab:gray', linestyle='-',alpha=0.4)
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)

    plt.bar(bin_centers, vals, bin_width, edgecolor='k', facecolor="dodgerblue", label='Observable Cosmic Ray Polarization\nDistribution From Cranberry')

    # plt.axvline(polarization_degs[0], c='gold', label=textwrap.fill('5911-73399 with Symettric Filtering',width=25))
    # plt.axvline(polarization_degs[1], c='g', label='5911-73399 with Asymettric Filtering')
    # plt.axvline(polarization_degs[2], c='m', label='5911-73399 with Only Sine Subtraction')
    plt.axvline(polarization_degs[0], c='gold', label=textwrap.fill('5911-73399 Polarization',width=25))

    plt.legend(loc='center right', fontsize=12)
    plt.ylabel('Frequency of Occurance Per Bin From Cranberry', fontsize=18)
    plt.xlabel('Polarization Angle (deg)', fontsize=18)

    ax.text(0.5, 0.5, 'Preliminary and Placeholder', transform=ax.transAxes,
        fontsize=40, color='gray', alpha=0.5,
        ha='center', va='center', rotation='30')



    #Expected Elevation Direction
    bin_centers = 90.0 - numpy.array([2.5,7.5,12.5,17.5,22.5,27.5,32.5,37.5,42.5,47.5,52.5,57.5,62.5,67.5,72.5,77.5,82.5,87.5])
    vals = numpy.array([0, 0, 0, 0, 0.001257, 0.0014675, 0.0039832, 0.012159, 0.029350, 0.0597, 0.0920, 0.12997, 0.1775, 0.20041, 0.15681, 0.10230, 0.032704, 0.0014675])
    vals = vals/sum(vals)
    bin_width = 5 #degrees

    ax = plt.subplot(1,3,2)
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='tab:gray', linestyle='-',alpha=0.4)
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)

    plt.bar(bin_centers, vals, bin_width, edgecolor='k', facecolor="dodgerblue", label=textwrap.fill('Observable Cosmic Ray Elevation Distribution From Cranberry',width=25))

    plt.axvline(18.852, c='gold', label=textwrap.fill('5911-73399 Elevation',width=25))

    plt.legend(loc='center right', fontsize=12)
    plt.ylabel('Frequency of Occurance Per Bin From Cranberry', fontsize=18)
    plt.xlabel('Elevation Angle (deg)', fontsize=18)

    ax.text(0.5, 0.5, 'Preliminary and Placeholder', transform=ax.transAxes,
        fontsize=40, color='gray', alpha=0.5,
        ha='center', va='center', rotation='30')

    #Expected Azimuth Direction
    bin_centers = numpy.arange(-175, 185, 10.0)
    vals = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.034482758620689696, 0.057650862068965525, 0.07564655172413794, 0.1, 0.10301724137931034, 0.10172413793103448, 0.10064655172413793, 0.09213362068965518, 0.08502155172413794, 0.09137931034482759, 0.0851293103448276, 0.07295258620689656, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    vals = vals/sum(vals)
    bin_width = 10 #degrees

    ax = plt.subplot(1,3,3)
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='tab:gray', linestyle='-',alpha=0.4)
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)

    plt.bar(bin_centers, vals, bin_width, edgecolor='k', facecolor="dodgerblue", label=textwrap.fill('Observable Cosmic Ray Azimuth Distribution From Cranberry',width=25))

    plt.axvline(-33.959, c='gold', label=textwrap.fill('5911-73399 Azimuth',width=25))

    plt.legend(loc='center right', fontsize=12)
    plt.ylabel('Frequency of Occurance Per Bin From Cranberry', fontsize=18)
    plt.xlabel('Azimuth Angle (deg)', fontsize=18)

    ax.text(0.5, 0.5, 'Preliminary and Placeholder', transform=ax.transAxes,
        fontsize=40, color='gray', alpha=0.5,
        ha='center', va='center', rotation='30')


    #plt.tight_layout(True)
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.95, wspace=0.20, hspace=None)
    fig.savefig('./angles_histogram_placeholder.pdf', dpi=300)
    fig.savefig('./angles_histogram_placeholder.png', dpi=300)