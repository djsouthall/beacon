#!/usr/bin/env python3
'''
This is intended to calculate time delays after first averaging each waveform from a given set/channel.  The goal here 
is to generate the cleanest possibly waveform for each channel to remove any time delay ambiguity, and then to gain 
statistical error based on how each waveform aligns with that template.
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



def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))


if __name__ == '__main__':
    plt.close('all')
    try:
        calibrations = ['rtk-gps-day3-june22-2021.json' , 'rtk-gps-day3-2021-09-13_both_minimized_wide_range_3_antennas_move.json']
        stacked = True
        sites_day2 = ['d2sa']
        sites_day3 = ['d3sa','d3sb','d3sc']
        sites_day4 = ['d4sa','d4sb']
        all_sites = ['d2sa','d3sa','d3sb','d3sc','d4sa','d4sb']
        cors_list = [] #To keep interactive live
        lassos = []

        #Set Baseline
        sites = all_sites#[sites_day3[1]]
        pols = ['hpol']#,'vpol']
        limit_eventids = 10000

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
                                        'd4sb' : [10]
                                    }
                            }

        known_pulser_ids = info.load2021PulserEventids()

        for pol in pols:
            #Prepare Figures
            fig_peaks = plt.figure(figsize=(16,9))
            fig_peaks.canvas.set_window_title('%s'%(pol))
            ax1 = plt.subplot(2,2,1)
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            plt.ylabel('Counts')
            plt.xlabel('Peak To Sidelobe')

            ax3 = plt.subplot(2,2,3)
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            plt.ylabel('Counts')
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

            if True:
                patches = []
                for calibration_index_value, deploy_index in enumerate(calibrations):
                    print(str(calibration_index_value) + ': ' + deploy_index)
                    patches.append(mpatches.Patch(edgecolor='k',facecolor=colors[calibration_index_value],alpha=0.7,label='Cal %i'%calibration_index_value))

                plt.sca(ax4)
                plt.legend(handles = patches,loc='upper right')

            if True:
                patches = []
                for site_index, site in enumerate(sites):
                    patches.append(mpatches.Patch(edgecolor='k',facecolor='w', hatch=hatches[site_index],alpha=0.7,label='Pulser ' + site))

                plt.sca(ax2)
                plt.legend(handles = patches,loc='lower right')


            for calibration_index_value, deploy_index in enumerate(calibrations):
                origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
                pulser_info = PulserInfo()

                all_peak_to_sidelobes = []
                all_map_peaks = []
                all_theta_best = []
                all_phi_best = []
                all_facecolors = []
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
                        cut = numpy.sort(numpy.random.choice(numpy.where(cut)[0], numpy.min([limit_eventids,sum(cut)]),replace=False))
                    else:
                        cut = numpy.where(cut)[0]
                    #Cut is index array after here

                    #Calculate Peak To Sidelobe, Peak Values, and Reconstruction directions
                    for run_index, run in enumerate(runs):
                        eventids = all_event_info[cut]
                        eventids = eventids[eventids['run'] == run]['eventid']
                        if run_index == 0:
                            hist, phi_best, theta_best, map_peaks, peak_to_sidelobes = cors[run].histMapPeak(eventids, pol, initial_hist=None, initial_thetas=None, initial_phis=None, plot_map=False, hilbert=hilbert, max_method=max_method, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[90,120], zenith_cut_array_plane=[0,90], circle_zenith=None, circle_az=None, window_title=None,radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False, return_map_peaks=True, return_peak_to_sidelobe=True, initial_peaks=None, initial_peak_to_sidelobes=None)
                        elif run_index != len(runs) - 1:
                            hist, phi_best, theta_best, map_peaks, peak_to_sidelobes = cors[run].histMapPeak(eventids, pol, initial_hist=hist, initial_thetas=theta_best, initial_phis=phi_best, plot_map=False, hilbert=hilbert, max_method=max_method, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[90,120], zenith_cut_array_plane=[0,90], circle_zenith=None, circle_az=None, window_title=None,radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False, return_map_peaks=True, return_peak_to_sidelobe=True, initial_peaks=map_peaks, initial_peak_to_sidelobes=peak_to_sidelobes)
                        else:
                            hist, phi_best, theta_best, map_peaks, peak_to_sidelobes = cors[run].histMapPeak(eventids, pol, initial_hist=hist, initial_thetas=theta_best, initial_phis=phi_best, plot_map=False, hilbert=hilbert, max_method=max_method, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[90,120], zenith_cut_array_plane=[0,90], circle_zenith=None, circle_az=None, window_title=None,radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False, return_map_peaks=True, return_peak_to_sidelobe=True, initial_peaks=map_peaks, initial_peak_to_sidelobes=peak_to_sidelobes)


                    if stacked == True:
                        all_peak_to_sidelobes.append(peak_to_sidelobes)
                        all_map_peaks.append(map_peaks)
                        all_theta_best.append(theta_best - zenith_deg)
                        all_phi_best.append(phi_best - azimuth_deg)
                        #all_facecolors.append(colors[calibration_index_value])
                        all_hatches.append(hatches[site_index])
                    else:
                        # Populate Plots
                        plt.sca(ax1)
                        plt.hist(peak_to_sidelobes,bins = numpy.arange(0.5,3.05,0.05),facecolor=colors[calibration_index_value] , alpha = 0.7, hatch=hatches[site_index], edgecolor='black', linewidth=1.0)

                        plt.sca(ax3)
                        plt.hist(map_peaks,bins = numpy.arange(0,1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.7, hatch=hatches[site_index], edgecolor='black', linewidth=1.0)

                        plt.sca(ax2)
                        plt.hist(theta_best - zenith_deg,bins = numpy.arange(-10,10,map_resolution),facecolor=colors[calibration_index_value] , alpha = 0.7, hatch=hatches[site_index], edgecolor='black', linewidth=1.0)

                        plt.sca(ax4)
                        plt.hist(phi_best - azimuth_deg,bins = numpy.arange(-10,10,map_resolution),facecolor=colors[calibration_index_value] , alpha = 0.7, hatch=hatches[site_index], edgecolor='black', linewidth=1.0)
                    
                        
                if stacked == True:
                    # Populate Plots
                    plt.sca(ax1)
                    n, bins, patches = plt.hist(all_peak_to_sidelobes,bins = numpy.arange(0.5,3.05,0.05),facecolor=colors[calibration_index_value] , alpha = 0.7, edgecolor='black', linewidth=1.0, stacked=True)
                    for patch_set, hatch in zip(patches, all_hatches):
                        for patch in patch_set.patches:
                            patch.set_hatch(hatch)

                    plt.sca(ax3)
                    n, bins, patches = plt.hist(all_map_peaks,bins = numpy.arange(0,1,0.05),facecolor=colors[calibration_index_value] , alpha = 0.7, edgecolor='black', linewidth=1.0, stacked=True)
                    for patch_set, hatch in zip(patches, all_hatches):
                        for patch in patch_set.patches:
                            patch.set_hatch(hatch)

                    plt.sca(ax2)
                    n, bins, patches = plt.hist(all_theta_best,bins = numpy.arange(-10,10,map_resolution),facecolor=colors[calibration_index_value] , alpha = 0.7, edgecolor='black', linewidth=1.0, stacked=True)
                    for patch_set, hatch in zip(patches, all_hatches):
                        for patch in patch_set.patches:
                            patch.set_hatch(hatch)

                    plt.sca(ax4)
                    n, bins, patches = plt.hist(all_phi_best,bins = numpy.arange(-10,10,map_resolution),facecolor=colors[calibration_index_value] , alpha = 0.7, edgecolor='black', linewidth=1.0, stacked=True)
                    for patch_set, hatch in zip(patches, all_hatches):
                        for patch in patch_set.patches:
                            patch.set_hatch(hatch)
                    
                    



    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

