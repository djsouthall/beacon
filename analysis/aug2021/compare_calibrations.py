#!/usr/bin/env python3
'''
This will take the pulsing events and compare the peak to sidelobe ratio of their maps etc. using different calibrations
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
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path
import matplotlib.patches as mpatches
from matplotlib.widgets import LassoSelector
import matplotlib.colors as mpl_colors
import pandas as pd
import datetime

plt.ion()

# import beacon.tools.info as info
from beacon.tools.data_handler import createFile, getTimes, loadTriggerTypes, getEventTimes
# from beacon.tools.fftmath import FFTPrepper
# from beacon.analysis.background_identify_60hz import plotSubVSec, plotSubVSecHist, alg1, diffFromPeriodic, get60HzEvents2, get60HzEvents3, get60HzEvents
from beacon.tools.fftmath import TimeDelayCalculator

from beacon.analysis.find_pulsing_signals_june2021_deployment import Selector

from beacon.analysis.aug2021.parse_pulsing_runs import PulserInfo, predictAlignment

hatches = [ 'o', 'O', '.', '*','/', '\\', '-', '+', 'x', '|'] #To destinguish pulsing sites
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


def prepPlots(pol, normalize_map_peaks, additional_window_title_text=''):
    '''
    This will generate the axis used for plotting in this script.
    '''
    fig = plt.figure(figsize=(16,9))
    fig.canvas.set_window_title('%s %s'%(pol, additional_window_title_text))
    plt.suptitle(pol + '_' + additional_window_title_text)
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
    plt.xlabel('Residual From Expected Zenith (deg)')

    ax4 = plt.subplot(2,2,4)
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylabel('Counts')
    plt.xlabel('Residual From Expected Azimuth (deg)')

    axs = [ax1,ax2,ax3,ax4]

    return fig, axs



def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))


if __name__ == '__main__':
    plt.close('all')
    try:
        initial_state = numpy.random.get_state() #Such that each site will look at the same subset of randomly selected events
        calibrations = ['/home/dsouthall/Projects/Beacon/beacon/config/for_comparison_9_15_2021/rtk-gps-day3-june22-2021.json','/home/dsouthall/Projects/Beacon/beacon/config/for_comparison_9_15_2021/rtk-gps-day3-june22-2021_2021-09-15_nolim.json','/home/dsouthall/Projects/Beacon/beacon/config/for_comparison_9_15_2021/rtk-gps-day3-june22-2021_2021-09-15_joint_ants_nolim.json']#['rtk-gps-day3-june22-2021.json' , 'rtk-gps-day3-2021-09-13_both_minimized_wide_range_3_antennas_move.json']#['rtk-gps-day3-june22-2021.json' , 'rtk-gps-day3-2021-09-13_both_minimized_wide_range_3_antennas_move.json', 'theodolite-day3-june22-2021_only_enu.json'] # ,'/home/dsouthall/Projects/Beacon/beacon/config/for_comparison_9_15_2021/theodolite-day3-june22-2021_only_enu.json','/home/dsouthall/Projects/Beacon/beacon/config/for_comparison_9_15_2021/rtk-gps-day3-june22-2021_2021-09-15_cables.json'
        calibrations_shorthand = ['GPS',  'Cable + Ant Float', 'Cable + Locked Pols Float']#'Theodolite', 'Cable Float',
        stacked = False
        sites_day2 = ['d2sa']
        sites_day3 = ['d3sa','d3sb','d3sc']
        sites_day4 = ['d4sa','d4sb']
        all_sites = ['d2sa','d3sa','d3sb','d3sc','d4sa','d4sb']
        cors_list = [] #To keep interactive live
        lassos = []
        normalize_map_peaks = True #This will use the maximum possible map value for a given signal to normalize the map peak value.
        #Set Baseline
        sites = all_sites#[sites_day3[1]]
        pols = ['hpol','vpol']
        limit_eventids = 10000
        use_hatches = False

        #The attenuations to include for each site and polarization
        attenuations_dict = {'hpol':{   'd2sa' : [20],
                                        'd3sa' : [10],
                                        'd3sb' : [6],
                                        'd3sc' : [20],
                                        'd4sa' : [20],
                                        'd4sb' : [6]
                                    },
                             'vpol':{   'd2sa' : [10],
                                        'd3sa' : [6],
                                        'd3sb' : [20],
                                        'd3sc' : [20],
                                        'd4sa' : [10],
                                        'd4sb' : [6]
                                    }
                            }

        known_pulser_ids = info.load2021PulserEventids()
        
        # Prepare the folder to save files in
        plot_per_site = True
        time_string = str(datetime.datetime.now()).replace(' ', '_').replace('.','p').replace(':','-') #Used in file names to keep them unique
        original_out_dir = os.path.join(os.getcwd(),'compare_calibrations_%s'%time_string)
        os.mkdir(original_out_dir)

        for pol in pols:
            out_dir = os.path.join(original_out_dir,pol)
            os.mkdir(out_dir)

            if plot_per_site:
                calib_sub_dirs = []
                for calibration_index_value, deploy_index in enumerate(calibrations):
                    calib_sub_dirs.append(os.path.join(out_dir, '%s'%(calibrations_shorthand[calibration_index_value].replace(' ','').replace('+','and'))))
                    os.mkdir(calib_sub_dirs[calibration_index_value])
            #Prepare Figures
            fig_total, axs_total =  prepPlots(pol, normalize_map_peaks,additional_window_title_text='Total')

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


            if use_hatches:
                patches = []
                for site_index, site in enumerate(sites):
                    patches.append(mpatches.Patch(edgecolor='k',facecolor='w', hatch=hatches[site_index],alpha=0.7,label='Pulser ' + site))

                hatch_patches = patches
                plt.sca(axs_total[1])
                plt.legend(handles = hatch_patches,loc='lower right')


            for calibration_index_value, deploy_index in enumerate(calibrations):

                origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
                pulser_info = PulserInfo()

                all_max_possible_map_values = []
                all_peak_to_sidelobes = []
                all_map_peaks = []
                all_theta_best = []
                all_phi_best = []
                all_facecolors = []
                if use_hatches:
                    all_hatches = []

                for site_index, site in enumerate(sites):
                    all_event_info = known_pulser_ids[site][pol]
                    runs = numpy.sort(numpy.unique(all_event_info['run']))

                    #Prepare correlators for future use on a per event basis
                    source_latlonel = pulser_info.getPulserLatLonEl(site)
                    
                    # Prepare expected angle and arrival times
                    enu = pm.geodetic2enu(source_latlonel[0],source_latlonel[1],source_latlonel[2],origin[0],origin[1],origin[2])
                    source_distance_m = numpy.sqrt(enu[0]**2 + enu[1]**2 + enu[2]**2)
                    azimuth_deg = numpy.rad2deg(numpy.arctan2(enu[1],enu[0]))
                    zenith_deg = numpy.rad2deg(numpy.arccos(enu[2]/source_distance_m))
                    
                    # Prepare correlators for later calculations for this site.
                    cors = {}
                    for run in runs:
                        cor_reader = Reader(os.environ['BEACON_DATA'],run)
                        cors[run] = Correlator(cor_reader,upsample=len(cor_reader.t())*8, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True, deploy_index=deploy_index, map_source_distance_m = source_distance_m)
                        if sine_subtract:
                            cors[run].prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)


                    #Calculate Expected Time Delays

                    if False:
                        expected_time_delays = predictAlignment(azimuth_deg, zenith_deg, cors[runs[0]], pol=pol)
                        align_method_10_window_ns = 20

                        time_delay_dict = {pol:{'[0, 1]' : [expected_time_delays[0]], '[0, 2]': [expected_time_delays[1]], '[0, 3]': [expected_time_delays[2]], '[1, 2]': [expected_time_delays[3]], '[1, 3]': [expected_time_delays[4]], '[2, 3]': [expected_time_delays[5]]}}
                        align_method_10_estimates = numpy.append(expected_time_delays,expected_time_delays_vpol)
                    

                    cut = numpy.isin(all_event_info['attenuation_dB'],attenuations_dict[pol][site])

                    #Cut is boolean array to here
                    if limit_eventids is not None:
                        cut[numpy.cumsum(cut) > limit_eventids] = False #will always choose the first 100
                        numpy.random.set_state(initial_state)
                        cut = numpy.sort(numpy.random.choice(numpy.where(cut)[0], numpy.min([limit_eventids,sum(cut)]),replace=False))
                    else:
                        cut = numpy.where(cut)[0]

                    #Cut is index array after here

                    #Calculate Peak To Sidelobe, Peak Values, and Reconstruction directions
                    for run_index, run in enumerate(runs):
                        eventids = all_event_info[cut]
                        eventids = eventids[eventids['run'] == run]['eventid']
                        if run_index == 0:
                            if len(runs) == 1:
                                hist, phi_best, theta_best, max_possible_map_values, map_peaks, peak_to_sidelobes, fig_map = cors[run].histMapPeak(eventids, pol, initial_hist=None, initial_thetas=None, initial_phis=None, plot_map=True, return_fig=True, hilbert=hilbert, max_method=max_method, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[90,120], zenith_cut_array_plane=[0,90], circle_zenith=zenith_deg, circle_az=azimuth_deg, window_title='%s %s'%(calibrations_shorthand[calibration_index_value] , site),radius=1.0,iterate_sub_baselines=None, shift_1d_hists=True, return_map_peaks=True, return_peak_to_sidelobe=True, initial_peaks=None, initial_peak_to_sidelobes=None, return_max_possible_map_values=True, initial_max_possible_map_values=None)
                            else:
                                hist, phi_best, theta_best, max_possible_map_values, map_peaks, peak_to_sidelobes = cors[run].histMapPeak(eventids, pol, initial_hist=None, initial_thetas=None, initial_phis=None, plot_map=False, hilbert=hilbert, max_method=max_method, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[90,120], zenith_cut_array_plane=[0,90], circle_zenith=None, circle_az=None, window_title=None,radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False, return_map_peaks=True, return_peak_to_sidelobe=True, initial_peaks=None, initial_peak_to_sidelobes=None, return_max_possible_map_values=True, initial_max_possible_map_values=None)
                        elif run_index != len(runs) - 1:
                            hist, phi_best, theta_best, max_possible_map_values, map_peaks, peak_to_sidelobes = cors[run].histMapPeak(eventids, pol, initial_hist=hist, initial_thetas=theta_best, initial_phis=phi_best, plot_map=False, hilbert=hilbert, max_method=max_method, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[90,120], zenith_cut_array_plane=[0,90], circle_zenith=None, circle_az=None, window_title=None,radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False, return_map_peaks=True, return_peak_to_sidelobe=True, initial_peaks=map_peaks, initial_peak_to_sidelobes=peak_to_sidelobes, return_max_possible_map_values=True, initial_max_possible_map_values=max_possible_map_values)
                        else:
                            hist, phi_best, theta_best, max_possible_map_values, map_peaks, peak_to_sidelobes, fig_map = cors[run].histMapPeak(eventids, pol, initial_hist=hist, initial_thetas=theta_best, initial_phis=phi_best, plot_map=True, return_fig=True, hilbert=hilbert, max_method=max_method, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[90,120], zenith_cut_array_plane=[0,90], circle_zenith=zenith_deg, circle_az=azimuth_deg, window_title='%s %s'%(calibrations_shorthand[calibration_index_value] , site),radius=1.0,iterate_sub_baselines=None, shift_1d_hists=True, return_map_peaks=True, return_peak_to_sidelobe=True, initial_peaks=map_peaks, initial_peak_to_sidelobes=peak_to_sidelobes, return_max_possible_map_values=True, initial_max_possible_map_values=max_possible_map_values)

                    fig_map.set_size_inches(16,9)
                    plt.sca(fig_map.axes[0])
                    if fig_map._suptitle is not None:
                        plt.suptitle(fig_map._suptitle.get_text() + '\n%s %s'%(calibrations_shorthand[calibration_index_value] , site))
                    else:
                        plt.suptitle('%s %s'%(calibrations_shorthand[calibration_index_value] , site))
                    fig_map.savefig(os.path.join(calib_sub_dirs[calibration_index_value],'map_%s'%(site) + '.png'), dpi=90)

                    all_max_possible_map_values.append(max_possible_map_values)
                    all_peak_to_sidelobes.append(peak_to_sidelobes)
                    all_map_peaks.append(map_peaks)
                    all_theta_best.append(theta_best - zenith_deg)
                    all_phi_best.append(phi_best - azimuth_deg)
                    #all_facecolors.append(colors[calibration_index_value])
                    if use_hatches:
                        all_hatches.append(hatches[site_index])

                    if plot_per_site == True:
                        fig_single_site, axs_single_site =  prepPlots(pol, normalize_map_peaks, additional_window_title_text=site) #Just for the calibration
                        plt.sca(axs_single_site[0])
                        plt.legend(handles = legend_patches,loc='upper right')
                        if use_hatches:
                            plt.sca(axs_single_site[1])
                            plt.legend(handles = hatch_patches,loc='lower right')

                        # Populate Plots
                        plt.sca(axs_single_site[0])
                        n, bins, patches = plt.hist(peak_to_sidelobes,bins = numpy.arange(0.5,3.05,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=False)
                        plt.xlim(1,3)

                        plt.sca(axs_single_site[2])
                        if normalize_map_peaks == True:
                            n, bins, patches = plt.hist(numpy.divide(map_peaks, max_possible_map_values),bins = numpy.arange(0,1.1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=False)
                        else:
                            n, bins, patches = plt.hist(map_peaks,bins = numpy.arange(0,1.1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=False)

                        plt.sca(axs_single_site[1])
                        n, bins, patches = plt.hist(theta_best - zenith_deg,bins = numpy.arange(-15,15,map_resolution),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=False)

                        plt.sca(axs_single_site[3])
                        n, bins, patches = plt.hist(phi_best - azimuth_deg,bins = numpy.arange(-10,10,map_resolution),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=False)

                        fig_single_site.savefig(os.path.join(calib_sub_dirs[calibration_index_value],'%s'%(site) + '.png'), dpi=90)

                    if False:
                        if site_index == 0:
                            print('\n\n\n!!!!\n\n\n\n')
                            print(deploy_index)
                            print(cut)
                            print('\n\n\n!!!!\n\n\n\n')


                fig_calib, axs_calib =  prepPlots(pol, normalize_map_peaks, additional_window_title_text=calibrations_shorthand[calibration_index_value]) #Just for the calibration
                plt.sca(axs_calib[0])
                plt.legend(handles = legend_patches,loc='upper right')
                if use_hatches:
                    plt.sca(axs_calib[1])
                    plt.legend(handles = hatch_patches,loc='lower right')


                for axs in [axs_total, axs_calib]:
                    if stacked == True:
                        # Populate Plots
                        plt.sca(axs[0])
                        n, bins, patches = plt.hist(all_peak_to_sidelobes,bins = numpy.arange(0.5,3.05,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=True)
                        plt.xlim(1,3)
                        if use_hatches:
                            for patch_set, hatch in zip(patches, all_hatches):
                                for patch in patch_set.patches:
                                    patch.set_hatch(hatch)

                        plt.sca(axs[2])
                        
                        if normalize_map_peaks == True:
                            n, bins, patches = plt.hist(numpy.divide(all_map_peaks,all_max_possible_map_values),bins = numpy.arange(0,1.1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=True)
                        else:
                            n, bins, patches = plt.hist(all_map_peaks,bins = numpy.arange(0,1.1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=True)
                        
                        if use_hatches:
                            for patch_set, hatch in zip(patches, all_hatches):
                                for patch in patch_set.patches:
                                    patch.set_hatch(hatch)

                        plt.sca(axs[1])
                        n, bins, patches = plt.hist(all_theta_best,bins = numpy.arange(-15,15,map_resolution),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=True)
                        if use_hatches:
                            for patch_set, hatch in zip(patches, all_hatches):
                                for patch in patch_set.patches:
                                    patch.set_hatch(hatch)

                        plt.sca(axs[3])
                        n, bins, patches = plt.hist(all_phi_best,bins = numpy.arange(-10,10,map_resolution),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=True)
                        if use_hatches:
                            for patch_set, hatch in zip(patches, all_hatches):
                                for patch in patch_set.patches:
                                    patch.set_hatch(hatch)
                    else:
                        # Populate Plots
                        plt.sca(axs[0])
                        n, bins, patches = plt.hist(numpy.concatenate(all_peak_to_sidelobes),bins = numpy.arange(0.5,3.05,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=False)
                        plt.xlim(1,3)
                        if use_hatches:
                            for patch_set, hatch in zip(patches, all_hatches):
                                for patch in patch_set.patches:
                                    patch.set_hatch(hatch)

                        plt.sca(axs[2])
                        if normalize_map_peaks == True:
                            n, bins, patches = plt.hist(numpy.divide(numpy.concatenate(all_map_peaks), numpy.concatenate(all_max_possible_map_values)),bins = numpy.arange(0,1.1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=False)
                        else:
                            n, bins, patches = plt.hist(numpy.concatenate(all_map_peaks),bins = numpy.arange(0,1.1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=False)
                        if use_hatches:
                            for patch_set, hatch in zip(patches, all_hatches):
                                for patch in patch_set.patches:
                                    patch.set_hatch(hatch)

                        plt.sca(axs[1])
                        n, bins, patches = plt.hist(numpy.concatenate(all_theta_best),bins = numpy.arange(-15,15,map_resolution),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=False)
                        if use_hatches:
                            for patch_set, hatch in zip(patches, all_hatches):
                                for patch in patch_set.patches:
                                    patch.set_hatch(hatch)

                        plt.sca(axs[3])
                        n, bins, patches = plt.hist(numpy.concatenate(all_phi_best),bins = numpy.arange(-10,10,map_resolution),facecolor=colors[calibration_index_value] , alpha = 0.5, edgecolor='black', linewidth=1.0, stacked=False)
                        if use_hatches:
                            for patch_set, hatch in zip(patches, all_hatches):
                                for patch in patch_set.patches:
                                    patch.set_hatch(hatch)
                fig_calib.savefig(os.path.join(out_dir,'comparing_calibrations_pulsing_events_%s_%s_allsites'%(pol,calibrations_shorthand[calibration_index_value].replace(' ','').replace('+','and')) + '.png'), dpi=90)
                fig_calib.savefig(os.path.join(calib_sub_dirs[calibration_index_value],'allsites.png'), dpi=90)
            fig_total.savefig(os.path.join(out_dir,'comparing_calibrations_pulsing_events_%s'%pol + '.png'), dpi=90)
                    



    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

