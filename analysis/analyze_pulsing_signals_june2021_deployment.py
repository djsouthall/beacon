#!/usr/bin/env python3
'''
Pulsing was conducted over 2 days during the June 2021 deployment.  On the day it was unclear if signals were actually
being seen.  This script is a playground for the work needed to find those signals in the data such that they can be
used in calibration.
'''
import os
import sys
import numpy

from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
import beacon.tools.interpret #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_slicer import dataSlicerSingleRun,dataSlicer
from beacon.tools.correlator import Correlator

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
from matplotlib.widgets import LassoSelector
import pandas as pd
numpy.set_printoptions(threshold=sys.maxsize)

plt.ion()

import beacon.tools.info as info
from beacon.tools.data_handler import createFile, getTimes, loadTriggerTypes, getEventTimes
from beacon.tools.fftmath import TimeDelayCalculator, TemplateCompareTool

from beacon.analysis.find_pulsing_signals_june2021_deployment import Selector

if True:
    pulsing_sites = {}
    pulsing_sites['Site 1 SE'] = ['run5167','run5168','run5169','run5170','run5171','run5172','run5173','run5190','run5191','run5195','run5196','run5198']#,'run5176' <- removed because the map data is not present :(
    pulsing_sites['Close Site N of Ant 1'] = ['run5179','run5180']
    pulsing_sites['Site 2 NE'] = ['run5182','run5183','run5185']
elif False:
    pulsing_sites = {}
    pulsing_sites['Site 1 SE'] = ['run5176']#,'run5176' <- removed because the map data is not present :(
    pulsing_sites['Close Site N of Ant 1'] = ['run5179','run5180']
    pulsing_sites['Site 2 NE'] = ['run5182','run5183','run5185']
else:
    #just to limit runs
    pulsing_sites = {}
    pulsing_sites['Site 1 SE'] = ['run5195']
    pulsing_sites['Close Site N of Ant 1'] = ['run5183']
    pulsing_sites['Site 2 NE'] = ['run5185']


if __name__ == '__main__':
    try:
        plt.close('all')

        #sites = list(pulsing_sites.keys())
        #sites = [list(pulsing_sites.keys())[0]]
        sites = numpy.array(list(pulsing_sites.keys()))[[0,2]]


        deploy_index = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'config/rtk-gps-day3-june22-2021.json')

        # Time Delay and Map Prep
        

        sine_subtract = True
        sine_subtract_min_freq_GHz = 0.03
        sine_subtract_max_freq_GHz = 0.13
        sine_subtract_percent = 0.03

        apply_phase_response = True

        hilbert=False
        final_corr_length = 2**15
        #waveform_index_range = {'Site 1 SE':(0,450),'Close Site N of Ant 1':(None, None), 'Site 2 NE':(0,450)}
        if False:

            crit_freq_low_pass_MHz = None#85
            low_pass_filter_order = None#6

            crit_freq_high_pass_MHz = None#25
            high_pass_filter_order = None#8


            waveform_index_range = {'run5167':(0,275),\
                                    'run5168':(0,280),\
                                    'run5169':(0,280),\
                                    'run5170':(0,280),\
                                    'run5171':(0,280),\
                                    'run5172':(0,280),\
                                    'run5173':(0,300),\
                                    'run5190':(0,150),\
                                    'run5191':(None,None),\
                                    'run5195':(None,None),\
                                    'run5196':(None,None),\
                                    'run5198':(0,450),\
                                    'run5176':(None,None),\
                                    'run5179':(0,280),\
                                    'run5180':(None,None),\
                                    'run5182':(None,None),\
                                    'run5183':(None,None),\
                                    'run5185':(0,500)}
            shorten_signals = False
            shorten_thresh  = 0.7
            shorten_delay   = 10.0
            shorten_length  = 90.0

        elif True:

            crit_freq_low_pass_MHz = 85
            low_pass_filter_order = 6

            crit_freq_high_pass_MHz = 25
            high_pass_filter_order = 8


            waveform_index_range = {'run5167':(0,275),\
                                    'run5168':(0,280),\
                                    'run5169':(0,280),\
                                    'run5170':(0,280),\
                                    'run5171':(0,280),\
                                    'run5172':(0,280),\
                                    'run5173':(0,300),\
                                    'run5190':(0,150),\
                                    'run5191':(None,None),\
                                    'run5195':(None,None),\
                                    'run5196':(None,None),\
                                    'run5198':(0,450),\
                                    'run5176':(None,None),\
                                    'run5179':(0,280),\
                                    'run5180':(None,None),\
                                    'run5182':(None,None),\
                                    'run5183':(None,None),\
                                    'run5185':(0,500)}
            shorten_signals = True
            shorten_thresh  = 0.4
            shorten_delay   = 0.0
            shorten_length  = 150.0

        else:
            crit_freq_low_pass_MHz = None#85
            low_pass_filter_order = None#6

            crit_freq_high_pass_MHz = None#25
            high_pass_filter_order = None#8


            waveform_index_range = {'run5167':(None, None),\
                                    'run5168':(None, None),\
                                    'run5169':(None, None),\
                                    'run5170':(None, None),\
                                    'run5171':(None, None),\
                                    'run5172':(None, None),\
                                    'run5173':(None, None),\
                                    'run5190':(None, None),\
                                    'run5191':(None, None),\
                                    'run5195':(None, None),\
                                    'run5196':(None, None),\
                                    'run5198':(None, None),\
                                    'run5176':(None, None),\
                                    'run5179':(None, None),\
                                    'run5180':(None, None),\
                                    'run5182':(None, None),\
                                    'run5183':(None, None),\
                                    'run5185':(None, None)}

            shorten_signals = True
            shorten_thresh  = 0.7
            shorten_delay   = 10.0
            shorten_length  = 90.0

        limit_eventids_per_run = 10000 #For testing purposes
        align_method = 10
        align_method_10_window_ns = 6

        n_phi = 901 #Used in dataSlicer
        range_phi_deg = (-90,90) #Used in dataSlicer
        n_theta = 901 #Used in dataSlicer
        range_theta_deg = (0,180) #Used in dataSlicer
        cmap = 'YlOrRd'#'binary'#'coolwarm'
        lognorm=True

        # Plotting Flags
        plot_multiple_per_run = False #Will plot time delay A LOT of histograms for every individual run
        plot_time_delays = False
        plot_maps = True
        plot_trig_times = False

        #Load pulser data
        pulser_locations = info.loadPulserLocationsENU(deploy_index=deploy_index)
        origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
        known_pulser_ids = info.loadPulserEventids(remove_ignored=True)

        used_eventids = {}
        for site in sites:
            used_eventids[site] = {}
            pulsing_sites_cut = numpy.ones(len(pulsing_sites[site])).astype(bool)
            for run_index, run_string in enumerate(pulsing_sites[site]):
                if numpy.isin(run_string, list(known_pulser_ids.keys())):
                    eventids = numpy.sort(numpy.unique(numpy.append(known_pulser_ids[run_string]['hpol'],known_pulser_ids[run_string]['vpol'])))
                    eventids = numpy.sort(numpy.random.choice(eventids,size=min(len(eventids),limit_eventids_per_run),replace=False))
                    used_eventids[site][run_string] = eventids
                else:
                    pulsing_sites_cut[run_index] = False
            pulsing_sites[site] = numpy.array(pulsing_sites[site])[pulsing_sites_cut] #Cutting out any runs that have no eventids after ignored events are removed.


        # Make correlators and store them.  The readers are stored in those correlators.
        correlators = {}
        tdcs = {}
        expected_tdcs = {}
        data_slicers = {}
        for site in sites:
            # Calculate expected direction information
            correlators[site] = {}
            tdcs[site] = {}

            runs = numpy.array([int(s.replace('run','')) for s in pulsing_sites[site]])
            #Currently only the maps are preprocessed.
            time_delays_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
            impulsivity_dset_key = time_delays_dset_key
            #map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_65536-maxmethod_0-sinesubtract_1-deploy_calibration_30-scope_allsky'
            map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_rtk-gps-day3-june22-2021.json-n_phi_1440-min_phi_neg180-max_phi_180-n_theta_720-min_theta_0-max_theta_120-scope_abovehorizon'
            map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_rtk-gps-day3-june22-2021.json-n_phi_1440-min_phi_neg180-max_phi_180-n_theta_720-min_theta_0-max_theta_120-scope_allsky'
            map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_rtk-gps-day3-june22-2021.json-n_phi_1440-min_phi_neg180-max_phi_180-n_theta_720-min_theta_0-max_theta_120-scope_belowhorizon'

            data_slicers[site] = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                                            curve_choice=0, trigger_types=[1,2,3],included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                                            cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                                            impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                                            time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                                            std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                                            p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                                            snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,\
                                            n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg,remove_incomplete_runs=False)
            if False:
                if site == 'Site 1 SE':
                    data_slicers[site].addROI('Pulser Direction',{'phi_best_h':[-100,-30],'elevation_best_h':[-15,0]})
                elif site == 'Site 2 NE':
                    data_slicers[site].addROI('Pulser Direction',{'phi_best_h':[40,100],'elevation_best_h':[-15,0]})


            for run_string in pulsing_sites[site]:
                p = pulser_locations[run_string]
                source_distance_m = numpy.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
                azimuth_deg = numpy.rad2deg(numpy.arctan2(p[1],p[0]))
                zenith_deg = numpy.rad2deg(numpy.arccos(p[2]/source_distance_m))
                print(run_string + ' az: %0.2f, zen: %0.2f'%(azimuth_deg,zenith_deg))
                run = int(run_string.replace('run',''))
                reader = Reader(os.environ['BEACON_DATA'],run)
                correlators[site][run_string] = Correlator(reader,waveform_index_range=waveform_index_range[run_string],upsample=len(reader.t())*8, n_phi=180*4 + 1, n_theta=360*4 + 1, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, apply_phase_response=apply_phase_response, tukey=True, sine_subtract=True, map_source_distance_m=source_distance_m, deploy_index=deploy_index)
                tdcs[site][run_string] = TimeDelayCalculator(reader,waveform_index_range=waveform_index_range[run_string], final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,apply_phase_response=apply_phase_response)
                if sine_subtract == True:
                    tdcs[site][run_string].addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
                
                #Calculate expected time delays:
                hpol_expected_delays, vpol_expected_delays = correlators[site][run_string].generateExpectedTimeDelaysFromDir(numpy.deg2rad(azimuth_deg), numpy.deg2rad(zenith_deg), return_indices=False, debug=False)
                expected_tdcs[site] = numpy.append(hpol_expected_delays, vpol_expected_delays)

            if False:
                #Debugging
                print('Plotting debug plot')
                print(site)
                print(run_string)
                correlators[site][run_string].generateExpectedTimeDelaysFromDir(numpy.deg2rad(azimuth_deg), numpy.deg2rad(zenith_deg), return_indices=False, debug=True)

        #Calculations
        if False:
            #Plot waveforms.  Helpful for windowing waveforms.
            for site_index, site in enumerate(sites):
                for channel_set in [[0,2,4,6],[1,3,5,7]]:
                    fig = plt.figure()
                    fig.canvas.set_window_title(site + str(channel_set))
                    plt.title(site + str(channel_set))
                    for channel in channel_set:
                        plt.subplot(4,1,channel//2 + 1)
                        plt.ylabel('adu')
                        plt.xlabel('time index')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                    for run_index, run_string in enumerate(pulsing_sites[site]):
                        eventids = used_eventids[site][run_string]
                        for eventid in eventids:
                            tdcs[site][run_string].reader.setEntry(eventid)
                            for channel in channel_set:
                                plt.subplot(4,1,channel//2 + 1)
                                plt.plot(tdcs[site][run_string].reader.wf(channel),alpha = 0.4)
        elif False:
            #Plot waveforms.  Helpful for windowing waveforms.
            for site_index, site in enumerate(sites):

                for run_index, run_string in enumerate(pulsing_sites[site]):
                    for channel_set in [[0,1,2,3,4,5,6,7]]:#,[1,3,5,7]]:
                        fig = plt.figure()
                        fig.canvas.set_window_title(run_string)
                        plt.title(run_string)
                        for channel in channel_set:
                            plt.subplot(4,1,channel//2 + 1)
                            plt.ylabel('adu')
                            plt.xlabel('time index')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        eventids = used_eventids[site][run_string]
                        for eventid in eventids:
                            tdcs[site][run_string].reader.setEntry(eventid)
                            for channel in channel_set:
                                plt.subplot(4,1,channel//2 + 1)
                                plt.plot(tdcs[site][run_string].reader.wf(channel),alpha = 0.4)
        '''
        Below this point the preparation work has been done to establish both map and time delay calculations.  Now each calculation can be performed below.
        '''
        if plot_time_delays == True:
            tds = {}
            corr_values = {}
            bin_dt = 1e6 
            corr_threshold = 0.5 #Below this won't be plotted in the time delay histograms.
            for site in sites:
                tds[site] = {}
                corr_values[site] = {}
                for run_index, run_string in enumerate(pulsing_sites[site]):
                    eventids = used_eventids[site][run_string]
                    print('%s Time Delay Calculations'%run_string)
                    # if run_index == 0:
                    #     plot_multiple_per_run = True
                    # else:
                    #     plot_multiple_per_run = False
                    if align_method == 10:
                        time_shifts, corrs, pairs = tdcs[site][run_string].calculateMultipleTimeDelays(eventids,align_method=10,hilbert=hilbert,plot=plot_multiple_per_run, sine_subtract=sine_subtract, align_method_10_estimates=expected_tdcs[site], align_method_10_window_ns=align_method_10_window_ns, shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length)
                    else:
                        time_shifts, corrs, pairs = tdcs[site][run_string].calculateMultipleTimeDelays(eventids,align_method=0,hilbert=hilbert,plot=plot_multiple_per_run, sine_subtract=sine_subtract, shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length)
                    bin_dt = min(bin_dt, tdcs[site][run_string].dt_ns_upsampled)

                    tds[site][run_string] = time_shifts
                    corr_values[site][run_string] = corrs

            # for site in sites:
            #     for run_index, run_string in enumerate(pulsing_sites[site]):
            #         if run_index == 0:
            #             _tds = tds[site][run_string]
            #             _corrs = corr_values[site][run_string]
            #         else:
            #             _tds = numpy.hstack((_tds,tds[site][run_string])) #will be 12 x n , with odd rows being vpol and even hpol
            #             _corrs = numpy.hstack((_corrs,corr_values[site][run_string])) #will be 12 x n , with odd rows being vpol and even hpol

            # if True:
            #     for site in sites:
            #         for run_string in pulsing_sites[site]:
            #             eventids = used_eventids[site][run_string]
            #             residuals = tds[site] - expected_tdcs[site]

            if False:
                #Plot correlation values
                for pair_index, pair in enumerate(pairs):
                    fig = plt.figure()
                    if pair[0]%2 == 0:
                        fig.canvas.set_window_title('Corr H %i - %i'%(int(pair[0]/2),int(pair[1]/2)))
                        plt.title('Corr  Hpol Baseline Ant %i - Ant %i'%(int(pair[0]/2),int(pair[1]/2)))
                    else:
                        fig.canvas.set_window_title('Corr  V %i - %i'%(int(pair[0]/2),int(pair[1]/2)))
                        plt.title('Corr Vpol Baseline Ant %i - Ant %i'%(int(pair[0]/2),int(pair[1]/2)))
                    plt.ylabel('Counts')
                    plt.xlabel('Correlation Value')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    for site_index, site in enumerate(sites):
                        for site_index, site in enumerate(sites):                
                            for run_index, run_string in enumerate(pulsing_sites[site]):
                                if run_index == 0:
                                    _corrs = corr_values[site][run_string]

                                else:
                                    _corrs = numpy.hstack((_corrs,corr_values[site][run_string])) #will be 12 x n , with odd rows being vpol and even hpol
                        bins = numpy.arange(-0.1,1.2,0.05)
                        plt.hist(_corrs[pair_index],bins=bins,alpha=0.7,edgecolor='black',linewidth=1.2,color = plt.rcParams['axes.prop_cycle'].by_key()['color'][site_index],label=site)
                    plt.axvline(corr_threshold,c = 'r', label='Plotting Threshold for TD', linestyle='--')
                    plt.legend()

            if False:
                #This will sift out eventids that might be ignored.
                for site_index, site in enumerate(sites):
                    for run_index, run_string in enumerate(pulsing_sites[site]):
                        if len(tds[site][run_string]) == 0:
                            continue
                        residuals = (tds[site][run_string].T - expected_tdcs[site]).T
                        residual_limit = 5
                        necessary_baselines = 3 #Hpol only
                        cut = numpy.sum(numpy.abs(residuals[0:6]) < residual_limit,axis=0) >= necessary_baselines
                        if False:
                            if numpy.sum(cut) > 0:
                                print(run_string)
                                print('Eventids with at least %i hpol baseline residuals < %.1f ns from expected'%(necessary_baselines,residual_limit))
                                pprint(used_eventids[site][run_string][cut])
                        else:
                            if numpy.sum(~cut) > 0:
                                print('Eventids which DONT have %i hpol baseline residuals < %.1f ns from expected'%(necessary_baselines,residual_limit))
                                print('ignore_eventids[\'%s\'] = numpy.'%(run_string))
                                pprint(used_eventids[site][run_string][~cut])


            if True:
                # Plot time delays
                plot_residuals = True
                for pair_index, pair in enumerate(pairs):
                    fig = plt.figure()
                    if plot_residuals:
                        plt.subplot(2,1,1)
                    if pair[0]%2 == 0:
                        fig.canvas.set_window_title('H %i - %i'%(int(pair[0]/2),int(pair[1]/2)))
                        plt.suptitle('Hpol Baseline Ant %i - Ant %i\nOnly max(xc) > %0.1f'%(int(pair[0]/2),int(pair[1]/2),corr_threshold))
                    else:
                        fig.canvas.set_window_title('V %i - %i'%(int(pair[0]/2),int(pair[1]/2)))
                        plt.suptitle('Vpol Baseline Ant %i - Ant %i\nOnly max(xc) > %0.1f'%(int(pair[0]/2),int(pair[1]/2),corr_threshold))
                    plt.ylabel('Counts')
                    plt.xlabel('Time Delay (ns)')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    for site_index, site in enumerate(sites):                
                        for run_index, run_string in enumerate(pulsing_sites[site]):
                            if run_index == 0:
                                _tds = tds[site][run_string]
                                _corrs = corr_values[site][run_string]

                            else:
                                _tds = numpy.hstack((_tds,tds[site][run_string])) #will be 12 x n , with odd rows being vpol and even hpol
                                _corrs = numpy.hstack((_corrs,corr_values[site][run_string])) #will be 12 x n , with odd rows being vpol and even hpol

                        thresh_cut = _corrs[pair_index] >= corr_threshold
                        _tds = _tds[pair_index][thresh_cut]
                        
                        if numpy.sum(thresh_cut) == 0:
                            continue

                        middle = numpy.mean([numpy.median(_tds),numpy.mean(_tds)])
                        bins = numpy.arange(middle - 50, middle + 50, 2*bin_dt)
                        #bins = numpy.arange(-200, 200, bin_dt)
                        #bins = numpy.arange(100)*bin_dt - numpy.median(_tds) - bin_dt/2.0
                        overflow = numpy.sum(numpy.logical_or(_tds < min(bins), _tds > max(bins)))
                        counts, bins, patches = plt.hist(_tds,bins=bins,alpha=0.7,edgecolor='black',linewidth=1.2,color = plt.rcParams['axes.prop_cycle'].by_key()['color'][site_index],label=site + '\nOverflow: %i/%i'%(overflow, len(_tds)))
                        x = 0.5*(bins[1:]+bins[:-1])
                        y = scipy.stats.norm.pdf(x,numpy.mean(_tds),numpy.std(_tds))
                        y = y*(numpy.sum(counts)/numpy.sum(y))
                        plt.plot(x,y,label=site + ' mean = %0.3f, sig = %0.3f'%(numpy.mean(_tds),numpy.std(_tds)))


                        # stacked_variables = numpy.vstack((all_phi_best,all_theta_best)) #x,y
                        # covariance_matrix = numpy.cov(stacked_variables)
                        # sig_phi = numpy.sqrt(covariance_matrix[0,0])
                        # sig_theta = numpy.sqrt(covariance_matrix[1,1])
                        # rho_phi_theta = covariance_matrix[0,1]/(sig_phi*sig_theta)
                        # mean_phi = numpy.mean(all_phi_best)
                        # mean_theta = numpy.mean(all_theta_best)

                        # y = scipy.stats.norm.pdf(x,0,sig_phi)

                    
                    for site_index, site in enumerate(sites):
                        #Seperated so on top
                        plt.axvline(expected_tdcs[site][pair_index],c = plt.rcParams['axes.prop_cycle'].by_key()['color'][site_index], label=site + ' Expected td = %0.2f'%expected_tdcs[site][pair_index], linestyle='--')
                    
                    plt.legend()

                    if plot_residuals:
                        plt.subplot(2,1,2)
                        plt.ylabel('Counts')
                        plt.xlabel('Time Delay - Expected Time Delay (ns)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                        for site_index, site in enumerate(sites):
                            for run_index, run_string in enumerate(pulsing_sites[site]):
                                if run_index == 0:
                                    _tds = tds[site][run_string]
                                    _corrs = corr_values[site][run_string]

                                else:
                                    _tds = numpy.hstack((_tds,tds[site][run_string])) #will be 12 x n , with odd rows being vpol and even hpol
                                    _corrs = numpy.hstack((_corrs,corr_values[site][run_string])) #will be 12 x n , with odd rows being vpol and even hpol

                            thresh_cut = _corrs[pair_index] >= corr_threshold
                            
                            if numpy.sum(thresh_cut) == 0:
                                continue

                            _tds = _tds[pair_index][thresh_cut] - expected_tdcs[site][pair_index]

                            middle = numpy.mean([numpy.median(_tds),numpy.mean(_tds)])
                            bins = numpy.arange(middle - 10, middle + 10, 2*bin_dt)
                            overflow = numpy.sum(numpy.logical_or(_tds < min(bins), _tds > max(bins)))
                            plt.hist(_tds,bins=bins,alpha=0.7,edgecolor='black',linewidth=1.2,color = plt.rcParams['axes.prop_cycle'].by_key()['color'][site_index],label=site + '\nOverflow: %i/%i'%(overflow, len(_tds)))

        if plot_maps == True:
            for site in sites:
                if True:
                    #Do data slicer maps
                    plot_param_pairs = [['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v'],['p2p_h', 'p2p_v']]
                    contour_eventids_dict = {}
                    if len(list(data_slicers[site].roi.keys())) > 0:
                        try:
                            _roi_eventid_dict = data_slicers[site].getCutsFromROI('Pulser Direction',load=False,save=False,verbose=False)
                        except Exception as e:
                            print(e)
                            print('Expected ROI likely was not added.')
                    for run_index, run_string in enumerate(pulsing_sites[site]):
                        eventids = used_eventids[site][run_string]
                        if False:
                            try:
                                eventids = eventids[numpy.isin(eventids, _roi_eventid_dict[int(run_string.replace('run',''))])]
                            except Exception as e:
                                print(e)
                                print('Expected ROI likely was not added.')
                            
                        contour_eventids_dict[int(run_string.replace('run',''))] = eventids


                    for key_x, key_y in plot_param_pairs:
                        print('Generating %s plot'%(key_x + ' vs ' + key_y))
                        fig, ax = data_slicers[site].plotROI2dHist(key_x, key_y, cmap=cmap, include_roi=False, lognorm=lognorm)
                        ax, cs = data_slicers[site].addContour(ax, key_x, key_y, contour_eventids_dict, 'r', n_contour=5, alpha=0.85, log_contour=True, mask=None, fill_value=None)

                        if 'phi' in key_x:
                            p = pulser_locations[pulsing_sites[site][0]]
                            source_distance_m = numpy.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
                            azimuth_deg = numpy.rad2deg(numpy.arctan2(p[1],p[0]))
                            zenith_deg = numpy.rad2deg(numpy.arccos(p[2]/source_distance_m))
                            ax.axvline(azimuth_deg,c='m',linewidth=1,alpha=0.5)
                            ax.axhline(90.0 - zenith_deg,c='m',linewidth=1,alpha=0.5)
                            ax.set_ylim(-30,10)
                            ax.set_xlim(-100,100)

                if True:
                    #Do correlator maps
                    for pol in ['hpol', 'vpol']:
                        for run_index, run_string in enumerate(pulsing_sites[site]):
                            p = pulser_locations[run_string]
                            source_distance_m = numpy.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
                            azimuth_deg = numpy.rad2deg(numpy.arctan2(p[1],p[0]))
                            zenith_deg = numpy.rad2deg(numpy.arccos(p[2]/source_distance_m))

                            eventids = used_eventids[site][run_string]

                            if run_index == 0:
                                hist, all_phi_best, all_theta_best = correlators[site][run_string].histMapPeak(eventids, pol, initial_hist=None, initial_thetas=None, initial_phis=None, plot_map=run_index == len(pulsing_sites[site]) - 1, hilbert=False, max_method=None, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=(90,180), zenith_cut_array_plane=(0,90), circle_zenith=zenith_deg, circle_az=azimuth_deg, window_title=site + ' Map Hist',radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False)
                            else:
                                hist, all_phi_best, all_theta_best = correlators[site][run_string].histMapPeak(eventids, pol, initial_hist=hist, initial_thetas=all_theta_best, initial_phis=all_phi_best, plot_map=run_index == len(pulsing_sites[site]) - 1, hilbert=False, max_method=None, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=(90,180), zenith_cut_array_plane=(0,90), circle_zenith=zenith_deg, circle_az=azimuth_deg, window_title=site + ' Map Hist',radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False)

        if plot_trig_times == True:
            for site in sites:
                for run_index, run_string in enumerate(pulsing_sites[site]):
                    eventids = used_eventids[site][run_string]

                    if run_index == 0:
                        calibrated_trig_time = getEventTimes(correlators[site][run_string].reader)[eventids]
                        c = run_index*numpy.ones_like(eventids)
                    else:
                        calibrated_trig_time = numpy.append(calibrated_trig_time,getEventTimes(correlators[site][run_string].reader)[eventids])
                        c = numpy.append(c,run_index*numpy.ones_like(eventids))
                second_time = numpy.floor(calibrated_trig_time)
                subsecond_time = calibrated_trig_time - second_time

                pfs = [1.0,1/20.0,1/100.0]

                for index, period_factor in enumerate(pfs):
                    fig = plt.figure()
                    if index == 0:
                        ax1 = plt.gca()
                    else:
                        plt.subplot(1,1,1, sharex=ax1)
                    ax = plt.gca()
                    plt.title('%s\n%0.2f Hz Expected'%(site, 1/period_factor))
                    scatter = plt.scatter(calibrated_trig_time, (subsecond_time%period_factor)*1e9, c=c)
                    cbar = fig.colorbar(scatter)
                    #plt.clim(zlim[0],zlim[1])
                    plt.ylabel('Subsecond Time (ns)')
                    plt.xlabel('Trigger Time (s)')
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)





#numpy.sort(numpy.unique(   ))