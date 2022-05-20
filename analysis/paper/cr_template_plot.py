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


start_time = time.time()
infile = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated.xlsx')
outfile = infile.replace('.xlsx','_%i.xlsx'%start_time)
main_sheet_name = 'good-with-airplane'

plot = True

#Create initial tdc
reader = Reader(raw_datapath, 5911)
tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=(None,None),plot_filters=False,apply_phase_response=apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches)
if sine_subtract:
    tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=max_failed_iterations, verbose=False, plot=False)

polarization_deg = tdc.calculatePolarizationFromTimeDelays(73399, apply_filter=False, waveforms=None, plot=True, sine_subtract=True)
polarization_deg = tdc.calculatePolarizationFromTimeDelays(73399, apply_filter=True, waveforms=None, plot=True, sine_subtract=True)

for key in template_dict.keys():
    filtered_template_dict[key] = {}
    filtered_template_dict[key]['t'] = tdc.t()
    filtered_template_dict[key]['wfs'] = numpy.zeros((8, len(filtered_template_dict[key]['t'])))
    filtered_template_dict[key]['std'] = numpy.zeros(8)
    filtered_template_dict[key]['resampled_std'] = numpy.zeros(8)
    for channel in range(8):
        filtered_template_dict[key]['wfs'][channel] = tdc.applyFilterToGivenWF(template_dict[key]['wfs'][channel], channel, apply_filter=True, hilbert=False, tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True) #this doesn't need to be done per event for the template, but I am doing it for consistency as this isn't a super intensive stage of the analysis
        filtered_template_dict[key]['std'][channel] = filtered_template_dict[key]['wfs'][channel].std()

        resampled_wf, resampled_t = scipy.signal.resample(filtered_template_dict[key]['wfs'][channel], upsample, t=filtered_template_dict[key]['t'])
        if channel == 0:
            filtered_template_dict[key]['resampled_wfs'] = numpy.zeros((8, len(resampled_wf)))
            filtered_template_dict[key]['resampled_t'] = resampled_t

        filtered_template_dict[key]['resampled_wfs'][channel] = resampled_wf
        filtered_template_dict[key]['resampled_std'][channel] = filtered_template_dict[key]['resampled_wfs'][channel].std()



base_key = '5911-73399'
base_offset = filtered_template_dict[base_key]['t'][int(numpy.argmax(filtered_template_dict[base_key]['wfs'][4]))] - 250
base_waveforms = filtered_template_dict[base_key]['resampled_wfs']

normalize = False


def makeTemplateFigure(fig, ax, channel, minor_fontsize=18, major_fontsize=24):
    '''
    Will do what this script does but for a single channel so this function can be imported. 
    '''
    for key in template_dict.keys():
        print(key)
        if 'cranberry' in key:
            c = 'k'
            key_label = 'Sample Simulated Cosmic Ray Event'
            if '88' in key:
                linestyle = '-'
            elif '421' in key:
                linestyle = '-.'
            else:
                linestyle = '--'
        else:
            c = mpl_colors[channel]
            key_label = 'Run ' + key.split('-')[0] + ' Event ' + key.split('-')[1]
            linestyle = '-'
        if channel == 0:
            print('%s len(t) = %i, max(t) = %i'%(key, len(template_dict[key]['t']), template_dict[key]['t'][-1]))

        plt.xlabel('Time (ns)', fontsize=major_fontsize)

        correlation = scipy.signal.correlate(filtered_template_dict[key]['resampled_wfs'][channel], base_waveforms[channel])
        roll = -(numpy.argmax(correlation) - len(correlation)//2)

        if normalize:
            plt.plot(filtered_template_dict[key]['resampled_t'] - 350, numpy.roll(filtered_template_dict[key]['resampled_wfs'][channel], int(roll))/filtered_template_dict[key]['resampled_wfs'][channel].max(), c=c, label=key_label, alpha=1.0, linewidth=3, linestyle=linestyle)
            plt.ylabel('Filtered Waveform\n' 'Antenna %i%s (Normalized)'%(channel//2, ['H','V'][channel%2]), fontsize=major_fontsize)
        else:
            plt.plot(filtered_template_dict[key]['resampled_t'] - 350, numpy.roll(filtered_template_dict[key]['resampled_wfs'][channel], int(roll)), c=c, label=key_label, alpha=1.0, linewidth=3, linestyle=linestyle)
            plt.ylabel('Filtered Waveform\n' 'Antenna %i%s (adu)'%(channel//2, ['H','V'][channel%2]), fontsize=major_fontsize)
            plt.ylim(-100,100)
        plt.xlim(0,500)
        # plt.plot(filtered_template_dict[key]['resampled_t'], numpy.roll(filtered_template_dict[key]['resampled_wfs'][channel], int(roll)), c=c, label= str(roll) + '   ' + 'Antenna %i%s - '%(channel//2, ['H','V'][channel%2]) + key, alpha=1.0, linewidth=3, linestyle='--')
        plt.legend(fontsize=minor_fontsize)
        ax.minorticks_on()
        ax.grid(b=True, which='major', color='k', linestyle='-')
        ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


if __name__ == '__main__':
    plt.close('all')

    if False:
        for antenna in range(4):
            fig = plt.figure(figsize=(9,9))
            for channel_index, channel in enumerate([antenna*2, antenna*2+1]):
                ax = plt.subplot(2,1,channel_index+1)

                for key in template_dict.keys():
                    print(key)
                    if 'cranberry' in key:
                        c = 'k'
                        key_label = 'Sample Simulated Cosmic Ray Event'
                        if '88' in key:
                            linestyle = '-'
                        elif '421' in key:
                            linestyle = '-.'
                        else:
                            linestyle = '--'
                    else:
                        c = mpl_colors[channel]
                        key_label = 'Run ' + key.split('-')[0] + ' Event ' + key.split('-')[1]
                        linestyle = '-'
                    if channel == 0:
                        print('%s len(t) = %i, max(t) = %i'%(key, len(template_dict[key]['t']), template_dict[key]['t'][-1]))

                    if channel_index == 1:
                        plt.xlabel('Time (ns)', fontsize=18)

                    correlation = scipy.signal.correlate(filtered_template_dict[key]['resampled_wfs'][channel], base_waveforms[channel])
                    roll = -(numpy.argmax(correlation) - len(correlation)//2)

                    if normalize:
                        plt.plot(filtered_template_dict[key]['resampled_t'] - 350, numpy.roll(filtered_template_dict[key]['resampled_wfs'][channel], int(roll))/filtered_template_dict[key]['resampled_wfs'][channel].max(), c=c, label=key_label, alpha=1.0, linewidth=3, linestyle=linestyle)
                        plt.ylabel('Filtered Waveform\n' 'Antenna %i%s (Normalized)'%(channel//2, ['H','V'][channel%2]), fontsize=18)
                    else:
                        plt.plot(filtered_template_dict[key]['resampled_t'] - 350, numpy.roll(filtered_template_dict[key]['resampled_wfs'][channel], int(roll)), c=c, label=key_label, alpha=1.0, linewidth=3, linestyle=linestyle)
                        plt.ylabel('Filtered Waveform\n' 'Antenna %i%s (adu)'%(channel//2, ['H','V'][channel%2]), fontsize=18)
                        plt.ylim(-100,100)
                    plt.xlim(0,500)
                    # plt.plot(filtered_template_dict[key]['resampled_t'], numpy.roll(filtered_template_dict[key]['resampled_wfs'][channel], int(roll)), c=c, label= str(roll) + '   ' + 'Antenna %i%s - '%(channel//2, ['H','V'][channel%2]) + key, alpha=1.0, linewidth=3, linestyle='--')
                    plt.legend(fontsize=18)
                    ax.minorticks_on()
                    ax.grid(b=True, which='major', color='k', linestyle='-')
                    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.95, wspace=None, hspace=None)

            #     plt.tight_layout()
            #     plt.gca().set_position([0, 0, 1, 1])
            # maximizeAllFigures(x_ratio=1, y_ratio=1)

            if False:
                if normalize:
                    fig.savefig('./normalized_template_event000088_comparison_wf_ch%i.pdf'%channel, dpi=300 )
                else:
                    fig.savefig('./template_event000088_comparison_wf_ch%i.pdf'%channel, dpi=300 )

    else:
        for channel in range(8):
            fig = plt.figure(figsize=(16,4))
            for key in template_dict.keys():
                print(key)
                if 'cranberry' in key:
                    c = 'k'
                    key_label = 'Sample Simulated Cosmic Ray Event'
                    if '88' in key:
                        linestyle = '-'
                    elif '421' in key:
                        linestyle = '-.'
                    else:
                        linestyle = '--'
                else:
                    c = mpl_colors[channel]
                    key_label = 'Run ' + key.split('-')[0] + ' Event ' + key.split('-')[1]
                    linestyle = '-'
                if channel == 0:
                    print('%s len(t) = %i, max(t) = %i'%(key, len(template_dict[key]['t']), template_dict[key]['t'][-1]))

                if False:
                    ax = plt.subplot(2,1,1)
                    plt.ylabel('Raw Waveform (adu)', fontsize=18)
                    plt.xlabel('Time (ns)', fontsize=18)
                    # plt.xlim(0,500)
                    t_offset = template_dict[key]['t'][int(numpy.argmax(template_dict[key]['wfs'][channel]))]
                    plt.plot(template_dict[key]['t'] - t_offset, template_dict[key]['wfs'][channel], label=key_label, c=c, alpha=1.0, linewidth=3)
                    plt.legend()
                    ax.minorticks_on()
                    ax.grid(b=True, which='major', color='k', linestyle='-')
                    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                    ax = plt.subplot(2,1,2)
                else:
                    ax = plt.gca()

                plt.xlabel('Time (ns)', fontsize=18)


                correlation = scipy.signal.correlate(filtered_template_dict[key]['resampled_wfs'][channel], base_waveforms[channel])
                roll = -(numpy.argmax(correlation) - len(correlation)//2)

                if normalize:
                    plt.plot(filtered_template_dict[key]['resampled_t'] - 350, numpy.roll(filtered_template_dict[key]['resampled_wfs'][channel], int(roll))/filtered_template_dict[key]['resampled_wfs'][channel].max(), c=c, label=key_label, alpha=1.0, linewidth=3, linestyle=linestyle)
                    plt.ylabel('Filtered Waveform\n' 'Antenna %i%s (Normalized)'%(channel//2, ['H','V'][channel%2]), fontsize=18)
                else:
                    plt.plot(filtered_template_dict[key]['resampled_t'] - 350, numpy.roll(filtered_template_dict[key]['resampled_wfs'][channel], int(roll)), c=c, label=key_label, alpha=1.0, linewidth=3, linestyle=linestyle)
                    plt.ylabel('Filtered Waveform\n' 'Antenna %i%s (adu)'%(channel//2, ['H','V'][channel%2]), fontsize=18)
                    plt.ylim(-100,100)
                plt.xlim(0,500)
                # plt.plot(filtered_template_dict[key]['resampled_t'], numpy.roll(filtered_template_dict[key]['resampled_wfs'][channel], int(roll)), c=c, label= str(roll) + '   ' + 'Antenna %i%s - '%(channel//2, ['H','V'][channel%2]) + key, alpha=1.0, linewidth=3, linestyle='--')
                plt.legend(fontsize=18)
                ax.minorticks_on()
                ax.grid(b=True, which='major', color='k', linestyle='-')
                ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.95, wspace=None, hspace=None)

            #     plt.tight_layout()
            #     plt.gca().set_position([0, 0, 1, 1])
            # maximizeAllFigures(x_ratio=1, y_ratio=1)

            if True:
                if normalize:
                    fig.savefig('./figures/normalized_template_event000088_comparison_wf_ch%i.pdf'%channel, dpi=300 )
                else:
                    fig.savefig('./figures/template_event000088_comparison_wf_ch%i.pdf'%channel, dpi=300 )





