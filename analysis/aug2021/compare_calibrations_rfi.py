#!/usr/bin/env python3
'''
This will take rfi events and compare the peak to sidelobe ratio of their maps etc. using different calibrations
in an attempt to determine if improvements are seen with the new calibration.
'''
import os
import sys
import numpy
import pymap3d as pm

from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
import beacon.tools.interpret #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_slicer import dataSlicerSingleRun,dataSlicer
from beacon.tools.correlator import Correlator
from beacon.tools.fftmath import TemplateCompareTool
import beacon.tools.info as info

import csv
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
from pprint import pprint
import itertools
import warnings
import h5py
import inspect
import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path
import matplotlib.patches as mpatches
from matplotlib.widgets import LassoSelector
import matplotlib.colors as mpl_colors
import pandas as pd

plt.ion()

# import beacon.tools.info as info
from beacon.tools.data_handler import createFile, getTimes, loadTriggerTypes, getEventTimes
# from beacon.tools.fftmath import FFTPrepper
# from beacon.analysis.background_identify_60hz import plotSubVSec, plotSubVSecHist, alg1, diffFromPeriodic, get60HzEvents2, get60HzEvents3, get60HzEvents
from beacon.tools.fftmath import TimeDelayCalculator

from beacon.analysis.find_pulsing_signals_june2021_deployment import Selector

from beacon.analysis.aug2021.parse_pulsing_runs import PulserInfo, predictAlignment

colors_hex = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] #To distinguish calibrations

colors = []

for c in colors_hex:
    rgb = mpl_colors.ColorConverter.to_rgb(c)
    rgba = (rgb[0],rgb[1],rgb[2],0.7)
    colors.append(rgba)

#Filter settings
max_method = 0
final_corr_length = 2**18
apply_phase_response = True

crit_freq_low_pass_MHz = 80
low_pass_filter_order = 14

crit_freq_high_pass_MHz = 20
high_pass_filter_order = 4

sine_subtract = False
sine_subtract_min_freq_GHz = 0.02
sine_subtract_max_freq_GHz = 0.15
sine_subtract_percent = 0.01
max_failed_iterations = 3

plot_filters = False
plot_multiple = False

hilbert = False #Apply hilbert envelope to wf before correlating
align_method = 1 #0,1,2 


shorten_signals = False
shorten_thresh = 0.7
shorten_delay = 10.0
shorten_length = 40.0
shorten_keep_leading = 10.0

map_resolution = 0.25 #degrees
range_phi_deg = (-90, 90)
range_theta_deg = (80,120)
n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
thetas_deg = numpy.linspace(min(range_theta_deg),max(range_theta_deg),n_theta) #Zenith angle
phis_deg = numpy.linspace(min(range_phi_deg),max(range_phi_deg),n_phi) #Azimuth angle


def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))


def prepPlots(pol, normalize_map_peaks, runs, calibrations, calibrations_shorthand,  additional_window_title_text=''):
    '''
    This will generate the axis used for plotting in this script.
    '''
    fig = plt.figure(figsize=(16,9))
    fig.canvas.set_window_title(pol + '_' + additional_window_title_text)
    if numpy.all(numpy.diff(runs) == 1):
        run_string = '%s - %s'%(runs[0], runs[-1])
    else:
        run_string = str(runs)
    plt.suptitle('Comparing Event Reconstruction of %s Channels\nfor Runs %s and Trigtypes %s\n%s'%(pol, run_string, str(trigger_types),additional_window_title_text))
    ax1 = plt.subplot(2,2,1)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylabel('Counts')
    plt.xlabel('Peak To Sidelobe')

    ax3 = plt.subplot(2,2,3)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylabel('Counts')
    if normalize_map_peaks == True:
        plt.xlabel('Map Peak Value/Max Possible Per Event')
    else:
        plt.xlabel('Map Peak Value')

    ax2 = plt.subplot(2,2,2)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylabel('Counts')
    if len(calibrations_shorthand) == len(calibrations):
        plt.xlabel('Offset in Zenith from %s (deg)'%(calibrations_shorthand[0]))
    else:
        plt.xlabel('Offset in Zenith from Cal 0 (deg)')

    ax4 = plt.subplot(2,2,4)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylabel('Counts')
    if len(calibrations_shorthand) == len(calibrations):
        plt.xlabel('Offset in Azimuth from %s (deg)'%(calibrations_shorthand[0]))
    else:
        plt.xlabel('Offset in Azimuth from Cal 0 (deg)')

    axs = [ax1,ax2,ax3,ax4]

    return fig, axs


if __name__ == '__main__':
    plt.close('all')
    try:
        calibrations = calibrations = ['/home/dsouthall/Projects/Beacon/beacon/config/for_comparison_9_15_2021/rtk-gps-day3-june22-2021.json','/home/dsouthall/Projects/Beacon/beacon/config/for_comparison_9_15_2021/theodolite-day3-june22-2021_only_enu.json','/home/dsouthall/Projects/Beacon/beacon/config/for_comparison_9_15_2021/rtk-gps-day3-june22-2021_2021-09-15_cables.json','/home/dsouthall/Projects/Beacon/beacon/config/for_comparison_9_15_2021/rtk-gps-day3-june22-2021_2021-09-15_nolim.json','/home/dsouthall/Projects/Beacon/beacon/config/for_comparison_9_15_2021/rtk-gps-day3-june22-2021_2021-09-15_joint_ants_nolim.json']#['rtk-gps-day3-june22-2021.json' , 'rtk-gps-day3-2021-09-13_both_minimized_wide_range_3_antennas_move.json']#['rtk-gps-day3-june22-2021.json' , 'rtk-gps-day3-2021-09-13_both_minimized_wide_range_3_antennas_move.json', 'theodolite-day3-june22-2021_only_enu.json']
        calibrations_shorthand = ['GPS', 'Theodolite', 'Cable Float', 'Cable + Ant Float', 'Cable + Locked Pols Float']
        runs = numpy.arange(5800,5841)
        cors_list = [] #To keep interactive live
        lassos = []
        normalize_map_peaks = True #This will use the maximum possible map value for a given signal to normalize the map peak value.

        #Set Baseline
        pols = ['hpol','vpol']
        limit_eventids = 1000
        trigger_types = [2]

        include_p2p = False

        # Prepare the folder to save files in
        time_string = str(datetime.datetime.now()).replace(' ', '_').replace('.','p').replace(':','-') #Used in file names to keep them unique
        original_out_dir = os.path.join(os.getcwd(),'compare_calibrations_rfi_%s'%time_string)
        os.mkdir(original_out_dir)

        for pol in pols:
            out_dir = os.path.join(original_out_dir,pol)
            os.mkdir(out_dir)
            #Prepare Figures
            fig_total, axs_total = fig, axs = prepPlots(pol, normalize_map_peaks, runs, calibrations, calibrations_shorthand,  additional_window_title_text='')

            if True:
                patches = []
                if len(calibrations_shorthand) == len(calibrations):
                    for calibration_index_value, deploy_index in enumerate(calibrations):
                        print(calibrations_shorthand[calibration_index_value] + ': ' + deploy_index)
                        patches.append(mpatches.Patch(edgecolor='k',facecolor=colors[calibration_index_value],alpha=0.7,label=calibrations_shorthand[calibration_index_value]))
                else:
                    for calibration_index_value, deploy_index in enumerate(calibrations):
                        print(str(calibration_index_value) + ': ' + deploy_index)
                        patches.append(mpatches.Patch(edgecolor='k',facecolor=colors[calibration_index_value],alpha=0.7,label='Cal %i'%calibration_index_value))

                legend_patches = patches
                plt.sca(axs_total[0])
                plt.legend(handles = legend_patches,loc='upper right')



            # Prepare events before specific calibration chosen such that same events compared for each type. 
            all_event_info = numpy.array([],dtype={'names':['run','eventid','trigtype','p2p'], 'formats':['<i4','<i4','<i4','f']})
            for run in runs:
                cor_reader = Reader(os.environ['BEACON_DATA'],run)
                _trigger_types = loadTriggerTypes(cor_reader)
                trigtype_cut = numpy.where(numpy.isin(_trigger_types , trigger_types))[0]

                event_info = numpy.zeros(len(trigtype_cut),dtype=all_event_info.dtype)
                event_info['eventid'] = trigtype_cut
                event_info['run'] = run
                event_info['trigtype'] = _trigger_types[trigtype_cut]

                if include_p2p:
                    channels = numpy.arange(4,dtype=int)*2 + int(pol == 'vpol')
                    print('Calculting p2p for run %i'%run)
                    for event_index, eventid in enumerate(event_info['eventid']):
                        if event_index%1000 == 0:
                            print(event_index, '/' , len(event_info))
                        cor_reader.setEntry(eventid)
                        max_p2p = 0
                        for ch in channels:
                            wf = cor_reader.wf(int(ch))
                            max_p2p = max( max_p2p , max(wf) - min(wf) )
                        event_info['p2p'][event_index] = max_p2p                    
                    
                all_event_info = numpy.append(all_event_info,event_info)

            if limit_eventids is not None:
                cut = numpy.sort(numpy.random.choice(numpy.arange(len(all_event_info)), numpy.min([limit_eventids,len(all_event_info)]),replace=False))
            else:
                cut = numpy.arange(len(all_event_info))

            all_max_possible_map_values = {}
            all_peak_to_sidelobes = {}
            all_map_peaks = {}
            all_theta_best = {}
            all_phi_best = {}

            all_figs = []
            all_axs = []

            for calibration_index_value, deploy_index in enumerate(calibrations):
                fig_calib, axs_calib = fig, axs = prepPlots(pol, normalize_map_peaks, runs, calibrations, calibrations_shorthand,  additional_window_title_text=calibrations_shorthand[calibration_index_value])

                origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
                pulser_info = PulserInfo()


                #SHould store these per calibration in a dictionary so they can be compared.  Because the expeceted direction makes no sense here I may just compare the shift.
                
                    
                # Prepare correlators for later calculations.
                cors = {}
                
                for run in runs:
                    cor_reader = Reader(os.environ['BEACON_DATA'],run)
                    cors[run] = Correlator(cor_reader,upsample=len(cor_reader.t())*8, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True, deploy_index=deploy_index, map_source_distance_m = 5e5)
                    if sine_subtract:
                        cors[run].prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                
                #Cut is index array after here

                #Calculate Peak To Sidelobe, Peak Values, and Reconstruction directions
                for run_index, run in enumerate(runs):
                    print('Cal %i / %i , run %i / %i'%(calibration_index_value+1,len(calibrations),run_index + 1, len(runs)))
                    eventids = all_event_info[cut]
                    eventids = eventids[eventids['run'] == run]['eventid']
                    if run_index == 0:
                        if len(runs) == 1:
                            hist, phi_best, theta_best, max_possible_map_values, map_peaks, peak_to_sidelobes, fig_map = cors[run].histMapPeak(eventids, pol, initial_hist=None, initial_thetas=None, initial_phis=None, plot_map=True, return_fig=True, hilbert=hilbert, max_method=max_method, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[90,120], zenith_cut_array_plane=[0,90], circle_zenith=None, circle_az=None, window_title='%s %s'%(calibrations_shorthand[calibration_index_value] , site), radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False, return_map_peaks=True, return_peak_to_sidelobe=True, initial_peaks=None, initial_peak_to_sidelobes=None, return_max_possible_map_values=True, initial_max_possible_map_values=None)
                        else:
                            hist, phi_best, theta_best, max_possible_map_values, map_peaks, peak_to_sidelobes = cors[run].histMapPeak(eventids, pol, initial_hist=None, initial_thetas=None, initial_phis=None, plot_map=False, hilbert=hilbert, max_method=max_method, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[90,120], zenith_cut_array_plane=[0,90], circle_zenith=None, circle_az=None, window_title=None,radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False, return_map_peaks=True, return_peak_to_sidelobe=True, initial_peaks=None, initial_peak_to_sidelobes=None, return_max_possible_map_values=True, initial_max_possible_map_values=None)
                    elif run_index != len(runs) - 1:
                        hist, phi_best, theta_best, max_possible_map_values, map_peaks, peak_to_sidelobes = cors[run].histMapPeak(eventids, pol, initial_hist=hist, initial_thetas=theta_best, initial_phis=phi_best, plot_map=False, hilbert=hilbert, max_method=max_method, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[90,120], zenith_cut_array_plane=[0,90], circle_zenith=None, circle_az=None, window_title=None,radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False, return_map_peaks=True, return_peak_to_sidelobe=True, initial_peaks=map_peaks, initial_peak_to_sidelobes=peak_to_sidelobes, return_max_possible_map_values=True, initial_max_possible_map_values=max_possible_map_values)
                    else:
                        hist, phi_best, theta_best, max_possible_map_values, map_peaks, peak_to_sidelobes, fig_map = cors[run].histMapPeak(eventids, pol, initial_hist=hist, initial_thetas=theta_best, initial_phis=phi_best, plot_map=True, return_fig=True, hilbert=hilbert, max_method=max_method, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[90,120], zenith_cut_array_plane=[0,90], circle_zenith=None, circle_az=None, window_title='%s %s'%(calibrations_shorthand[calibration_index_value] , site), radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False, return_map_peaks=True, return_peak_to_sidelobe=True, initial_peaks=map_peaks, initial_peak_to_sidelobes=peak_to_sidelobes, return_max_possible_map_values=True, initial_max_possible_map_values=max_possible_map_values)

                all_max_possible_map_values[deploy_index] = max_possible_map_values
                all_peak_to_sidelobes[deploy_index] = peak_to_sidelobes
                all_map_peaks[deploy_index] = map_peaks
                all_theta_best[deploy_index] = theta_best
                all_phi_best[deploy_index] = phi_best

                fig_map.set_size_inches(16,9)
                plt.sca(fig_map.axes[0])
                if fig_map._suptitle is not None:
                    plt.suptitle(fig_map._suptitle.get_text() + '\n%s %s'%(calibrations_shorthand[calibration_index_value] , site))
                else:
                    plt.suptitle('%s %s'%(calibrations_shorthand[calibration_index_value] , site))
                fig_map.savefig(os.path.join(calib_sub_dirs[calibration_index_value],'map_%s'%(site) + '.png'), dpi=90)

                # Populate Plots
                plt.sca(axs_total[0])
                plt.hist(peak_to_sidelobes,bins = numpy.arange(0.5,3.05,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0)

                plt.sca(axs_total[2])
                if normalize_map_peaks == True:
                    plt.hist(numpy.divide(map_peaks,max_possible_map_values) ,bins = numpy.arange(0,1.1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0)
                else:
                    plt.hist(map_peaks,bins = numpy.arange(0,1.1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0)


                plt.sca(axs_calib[0])
                plt.hist(peak_to_sidelobes,bins = numpy.arange(0.5,3.05,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0)

                plt.sca(axs_calib[2])
                if normalize_map_peaks == True:
                    plt.hist(numpy.divide(map_peaks,max_possible_map_values) ,bins = numpy.arange(0,1.1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0)
                else:
                    plt.hist(map_peaks,bins = numpy.arange(0,1.1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0)

                all_figs.append(fig_calib)
                all_axs.append(axs_calib)

                print('\n\n\n!!!!\n\n\n\n')
                print(deploy_index)
                print(len(numpy.divide(map_peaks,max_possible_map_values)))
                print('\n\n\n!!!!\n\n\n\n')


            # Must be done after calculations made for all calibrations
            for calibration_index_value, deploy_index in enumerate(calibrations):
                if calibration_index_value == 0:
                    reference_deploy_index = deploy_index
                    continue
                else:
                    plt.sca(axs_total[1])
                    plt.hist(all_theta_best[deploy_index] - all_theta_best[reference_deploy_index],bins = numpy.arange(-10,10,map_resolution) , facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0)

                    plt.sca(axs_total[3])
                    plt.hist(all_phi_best[deploy_index] - all_phi_best[reference_deploy_index],bins = numpy.arange(-10,10,map_resolution) , facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0)

                    # Per calibration site plots
                    plt.sca(all_axs[calibration_index_value][1])
                    plt.hist(all_theta_best[deploy_index] - all_theta_best[reference_deploy_index],bins = numpy.arange(-10,10,map_resolution) , facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0)

                    plt.sca(all_axs[calibration_index_value][3])
                    plt.hist(all_phi_best[deploy_index] - all_phi_best[reference_deploy_index],bins = numpy.arange(-10,10,map_resolution) , facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0)

                plt.sca(all_axs[calibration_index_value][0])
                plt.legend(handles = legend_patches,loc='upper right')
                fig_total.savefig(os.path.join(out_dir,'comparing_calibrations_rfi_events_%s_%s'%(pol,calibrations_shorthand[calibration_index_value].replace(' ','').replace('+','and')) + '.png'), dpi=90)

            fig_total.savefig(os.path.join(out_dir,'comparing_calibrations_rfi_events_%s'%pol + '.png'), dpi=90)


    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

