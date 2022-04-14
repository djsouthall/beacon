#!/usr/bin/env python3
'''
This script is intended to look at the events that construct best above horizon in allsky maps.  I want to view the
peak to sidelobe values and other parameters for the belowhorizon and abovehorizon maps and determine if there is
an obvious cut for which sidelobed above horizon events can be discriminated. 
'''

import sys
import os
import inspect
import h5py
import copy
from pprint import pprint
import textwrap

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
from beacon.tools.get_sun_coords_from_timestamp import getSunElWeightsFromRunDict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import time
from datetime import datetime
import pytz
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
    cmap = 'cool'#'coolwarm'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length

    if False:
        #Quicker for testing
        runs = numpy.arange(5910,5912)
    elif True:
        runs = numpy.arange(5733,5974)
        runs = runs[runs != 5864]
    else:
        runs = numpy.arange(5910,5934)#numpy.arange(5733,5974)#numpy.arange(5910,5912)

    print("Preparing dataSlicer")

    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                    n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)

    if False:
        ds2 = dataSlicer([5911], impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                        n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)

        ds2.addROI('cr_template_search diagonal cut',{'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]})
        ds2.plotROI2dHist('cr_template_search_h', 'cr_template_search_v', cmap='cool', eventids_dict=None, include_roi=False)
        ds2.plotROI2dHist('cr_template_search_h', 'cr_template_search_v', cmap='cool', eventids_dict=None, include_roi=True)

    #ds.printKnownParamKeys()
    ds.addROI('above horizon',{'elevation_best_h':[10,90],'phi_best_h':[-90,90],'elevation_best_v':[10,90],'phi_best_v':[-90,90]})
    ds.addROI('above horizon sim',{'elevation_best_h':[10,90],'phi_best_h':[-90,90],'elevation_best_v':[10,90],'phi_best_v':[-90,90],'similarity_count_h':[0,10]})
    ds.addROI('above horizon full',{'elevation_best_h':[10,90],'phi_best_h':[-90,90],'elevation_best_v':[10,90],'phi_best_v':[-90,90],'similarity_count_h':[0,10],'similarity_count_v':[0,10],'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,10],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]})
    ds.addROI('above horizon full with stage 2',{   'elevation_best_h':[10,90],
                                                    'phi_best_h':[-90,90],
                                                    'elevation_best_v':[10,90],
                                                    'phi_best_v':[-90,90],
                                                    'similarity_count_h':[0,10],
                                                    'similarity_count_v':[0,10],
                                                    'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,10],
                                                    'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],
                                                    'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100],
                                                    'p2p_gap_h':[-1, 95],
                                                    'above_normalized_map_max_line':[0,10000],
                                                    'above_snr_line':[0,10000]})


    above_horizon_eventids_dict = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
    #above_horizon_sim_eventids_dict, successive_cut_counts, total_cut_counts = ds.getCutsFromROI('above horizon sim',load=False,save=False,verbose=True, return_successive_cut_counts=True, return_total_cut_counts=True)
    above_horizon_full_eventids_dict, successive_cut_counts, total_cut_counts = ds.getCutsFromROI('above horizon full',load=False,save=False,verbose=True, return_successive_cut_counts=True, return_total_cut_counts=True)
    del successive_cut_counts
    del total_cut_counts

    #Choose list of parameters you want plotted.  



    #params = ['elevation_best_h', 'phi_best_h', 'elevation_best_v', 'phi_best_v', 'similarity_count_h', 'similarity_count_v', 'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe', 'impulsivity_hSLICERADDimpulsivity_v', 'cr_template_search_hSLICERADDcr_template_search_v']
    #params = ['hpol_peak_to_sidelobe', 'impulsivity_h', 'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe']
    #params = ['impulsivity_hSLICERMAXimpulsivity_v', 'cr_template_search_hSLICERMAXcr_template_search_v', 'std_hSLICERMAXstd_v', 'p2p_hSLICERMAXp2p_v', 'snr_hSLICERMAXsnr_v', 'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe', 'hpol_max_possible_map_valueSLICERMAXvpol_max_possible_map_value']

    if True:
        mode = 1 #Will compare cuts based on the above horizon full, with an all but current param dict being created for each item.
        params = list(ds.roi['above horizon full'].keys())
    else:
        mode = 2 #Will instead just present the cuts for a set of unrelated params that are not currently used in the cut.
        params = ['impulsivity_hSLICERMAXimpulsivity_v', 'cr_template_search_hSLICERMAXcr_template_search_v', 'std_hSLICERMAXstd_v', 'p2p_hSLICERMAXp2p_v', 'snr_hSLICERMAXsnr_v', 'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe', 'hpol_max_possible_map_valueSLICERMAXvpol_max_possible_map_value']

    #Make dict point_of_interest_dict for list of axvlines to plot on the subsequent histograms.
    #{Key :{label:xvalue}}, put value as a dict of {'run-eventid':None} and the value corresponding to those events will be highlighted.
    point_of_interest_dict = {}


    #Populate used cuts for plotting axvlines
    roi_key = 'above horizon full'
    for key in list(ds.roi[roi_key].keys()):
        if key not in list(point_of_interest_dict.keys()):
            point_of_interest_dict[key] = {}
        point_of_interest_dict[key]['%s LL'%roi_key] = min(ds.roi[roi_key][key])
        point_of_interest_dict[key]['%s UL'%roi_key] = max(ds.roi[roi_key][key])

    #Populate same event values for plotting    
    sample_event = '5911-73399'
    #ds.eventInspector({5911:numpy.array([73399])})
    for param in params:
        if param not in list(point_of_interest_dict.keys()):
            point_of_interest_dict[param] = {}
        point_of_interest_dict[param][sample_event] = None
        

    #Plot parameters, will loop over dicts, naming them as listed.  Counts will be deployed in given format.

    dicts = [None, above_horizon_eventids_dict, above_horizon_full_eventids_dict] #last one currently replaced in loop for its partial components - 1 if mode == 1.  It is mostly used to get the set of cuts that would be used by it, and they stats are presented for each cut.  Elif mode == 2 then it is ignored altogether. 
    dict_names = ['All RF Events', 'Forward Above Horizon', 'Quality Cuts']
    log_counts = False
    plot_daynight = True
    daynight_mode = 'time' #'sun' #Time since first event in hours or sun elevation. 
    wrap_daynight = True
    standalone_daynight = False


    for dict_index, eventids_dict in enumerate(dicts):
        fig_map, ax_map = ds.plotROI2dHist('phi_best_h_allsky', 'elevation_best_h_allsky', cmap='cool', eventids_dict=eventids_dict, include_roi=False)
        fig_map.canvas.set_window_title(dict_names[dict_index].replace('\n',''))
    del above_horizon_full_eventids_dict

    cut_dict = copy.deepcopy(ds.roi['above horizon full'])

    for param_key_index, param_key in enumerate(params):
        ds.resetAllROI()
        if '_best_' in param_key:
            continue #skipping directional cuts
        ds.setCurrentPlotBins(param_key, param_key, None)
        current_label = ds.current_label_x.replace('\n','  ')

        if 'similarity_count' in param_key:
            current_bin_edges = numpy.linspace(-0.5,100.5,101)
        elif 'peak_to_sidelobe' in param_key and 'SLICERADD' in param_key:
            current_bin_edges = numpy.linspace(2,4,101)
        elif 'peak_to_sidelobe' in param_key:
            current_bin_edges = numpy.linspace(1,3,101)
        elif 'impulsivity' in param_key and 'SLICERADD' in param_key:
            current_bin_edges = numpy.linspace(-0.25,1.5,101)
        elif 'cr_template_search' in param_key and 'SLICERADD' in param_key:
            current_bin_edges = numpy.linspace(0.25,1.8,101)
        else:
            current_bin_edges = ds.current_bin_edges_x



        if plot_daynight == True:
            #Want 2 plots in left column and N in the right in this case
            fig = plt.figure()
            fig.canvas.set_window_title(current_label)

            ax = plt.subplot(2,2,1)
            plt.grid(which='both', axis='both')
            ax.minorticks_on()
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            plt.ylabel('Normalized Counts')
            plt.xlabel(textwrap.fill(current_label.replace('\n',' '),50))
        else:
            #Want them all stacked here.
            fig = plt.figure()
            fig.canvas.set_window_title(current_label)

            ax = plt.subplot(len(dicts) + 1 + int(plot_daynight),1,1)
            plt.grid(which='both', axis='both')
            ax.minorticks_on()
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            plt.ylabel('Normalized Counts')
            plt.xlabel(textwrap.fill(current_label.replace('\n',' '),80))

        if plot_daynight:
            #Get event trigtimes.
            time_dict = {}

            min_event_time = numpy.inf
            max_event_time = -numpy.inf
            for run_index, run in enumerate(ds.runs):
                max_min_eventids_dict = {run:[0,ds.data_slicers[run_index].reader.N()-1]}
                time_dict[run] = ds.getDataArrayFromParam('calibrated_trigtime', eventids_dict=max_min_eventids_dict)
                min_event_time = min(min_event_time,min(time_dict[run]))
                max_event_time = max(max_event_time,max(time_dict[run]))

            start_reference_time = min_event_time #Reference times will change, min max event times wont.  event times used for weights.
            finish_reference_time = max_event_time


            timezone = pytz.timezone("America/Los_Angeles")
            start_reference_datetime = datetime.fromtimestamp(start_reference_time, tz=timezone) #Get the date and time of first event in Cali
            start_reference_datetime = datetime(start_reference_datetime.year, start_reference_datetime.month, start_reference_datetime.day, tzinfo=timezone) #Midnight of first day so time of day is time since start. 
            start_reference_time = datetime.timestamp(start_reference_datetime) #Rest reference time to the very beginning of the first day.

            finish_reference_datetime = datetime.fromtimestamp(finish_reference_time + 24*60*60, tz=timezone) #Get the date and time of first event in Cali
            finish_reference_datetime = datetime(finish_reference_datetime.year, finish_reference_datetime.month, finish_reference_datetime.day, tzinfo=timezone) #Midnight of first day so time of day is time since finish. 
            finish_reference_time = datetime.timestamp(finish_reference_datetime) #Rest reference time to the very beginning of the first day.

            f_daynight_wrap = lambda t : ((t - start_reference_time)/(60*60))%24


            ax_daynight = plt.subplot(2,2,3)#plt.subplot(len(dicts) + 1 + int(plot_daynight),1,len(dicts) + 1 + int(plot_daynight))
            plt.grid(which='both', axis='both')
            ax_daynight.minorticks_on()
            ax_daynight.grid(b=True, which='major', color='k', linestyle='-')
            ax_daynight.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            log_diff = True
            if log_diff == False:
                if wrap_daynight == True:
                    plt.ylabel('Counts / % Time Spent in Each Bin')
                else:
                    plt.ylabel('Counts')
            else:
                if wrap_daynight == True:
                    
                    plt.ylabel(textwrap.fill('log10(Counts / % Time Spent in Each Bin - Mean)*(Sign from Mean)',50))
                else:
                    plt.ylabel('log10(Counts - Mean)*(Sign from Mean)')


            if daynight_mode == 'sun':
                ds.setCurrentPlotBins('sun_el', 'sun_el', None)
                daynight_edges = ds.current_bin_edges_x
                daynight_centers = 0.5*(daynight_edges[1:]+daynight_edges[:-1])
                daynight_xlabel = 'Sun Elevation when Event Triggered'
                plt.xlabel(daynight_xlabel)
                daynight_time_weights = getSunElWeightsFromRunDict(time_dict, daynight_edges)
            elif daynight_mode == 'time':

                time_bin_h = 15.0/60.0
                if wrap_daynight == True:
                    daynight_edges = numpy.arange(0,24 + time_bin_h,time_bin_h)
                    daynight_xlabel = 'Time of Day at BEACON'#'Wrapped Time of Day Since %f'%start_reference_time
                else:
                    daynight_edges = numpy.arange(0,finish_reference_time - start_reference_time + time_bin_h,time_bin_h)
                    daynight_xlabel = textwrap.fill('Time Since %s'%str(start_reference_datetime),50)
                daynight_centers = 0.5*(daynight_edges[1:]+daynight_edges[:-1])
                plt.xlabel(daynight_xlabel)

                if wrap_daynight == True:
                    daynight_time_weights = numpy.zeros(len(daynight_centers))
                    for run_index, run in enumerate(time_dict.keys()):
                        t = numpy.arange(min(time_dict[run]), max(time_dict[run]) + 1, 1) #1 second interval in the window, to go through the same calculation as the data would to get a sense of the % of time spent at each time.
                        daynight_time_weights = daynight_time_weights + numpy.histogram(f_daynight_wrap(t),bins=daynight_edges)[0]
                    daynight_time_weights = daynight_time_weights/sum(daynight_time_weights)


            if standalone_daynight == True:
                #Do it all again for a new plot.  
                fig_daynight = plt.figure()
                ax_daynight2 = plt.gca()
                plt.grid(which='both', axis='both')
                ax_daynight2.minorticks_on()
                ax_daynight2.grid(b=True, which='major', color='k', linestyle='-')
                ax_daynight2.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.xlabel(ax_daynight.get_xlabel())
                plt.ylabel(ax_daynight.get_ylabel())
                axs_daynight = [ax_daynight, ax_daynight2]
            else:
                axs_daynight = [ax_daynight]

        if mode == 1:
            if 'similarity' in param_key:
                plot_UL = True
                plot_LL = False
            else:
                plot_UL = False
                plot_LL = True

            param_dict = copy.deepcopy(cut_dict)
            del param_dict[param_key]
            ds.addROI('param',param_dict)
            param_eventids_dict, successive_cut_counts = ds.getCutsFromROI('param',load=False,save=False,verbose=False, return_successive_cut_counts=True, return_total_cut_counts=False)

            dicts[-1] = param_eventids_dict
        else:
            dicts = [dicts[0],dicts[1]]
            plot_UL = True
            plot_LL = True

        for dict_index, eventids_dict in enumerate(dicts):
            if dict_names[dict_index] == 'Quality Cuts':
                dict_name = dict_names[dict_index] + ' Except ' + param_key
            else:
                dict_name = dict_names[dict_index]

            data = ds.getDataArrayFromParam(param_key, eventids_dict=eventids_dict)
            max_count = len(data)
            overflow = sum(numpy.logical_or(data < min(current_bin_edges), data > max(current_bin_edges)))
            
            #Plot normal histogram
            plt.sca(ax)
            if plot_daynight:
                plt.hist(data, bins=current_bin_edges, log=True, cumulative=False, density=True, label='%s, Overflow: %i'%(dict_name,overflow),alpha=0.6, edgecolor='black', linewidth=1.2)
            else:
                plt.hist(data, bins=current_bin_edges, log=True, cumulative=False, density=True, label='%s\nOverflow: %i'%(dict_name,overflow),alpha=0.6, edgecolor='black', linewidth=1.2)
            
            #Plot cumulative histogram
            if plot_daynight == True:
                ax1 = plt.subplot(len(dicts),2,2*(dict_index + 1))
            else:
                ax1 = plt.subplot(len(dicts) + 1 + int(plot_daynight),1,dict_index + 2)
            plt.grid(which='both', axis='both')
            ax1.minorticks_on()
            ax1.grid(b=True, which='major', color='k', linestyle='-')
            ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.25)
            if dict_index == len(dicts) - 1:
                #Only label bottom plot.
                if plot_daynight:
                    plt.xlabel(textwrap.fill(current_label.replace('\n',' '),50))
                else:
                    plt.xlabel(textwrap.fill(current_label.replace('\n',' '),80))
            ax1.set_ylabel(r'% Of Events' + '\n%s'%dict_names[dict_index], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][dict_index]) 
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


            if plot_UL:
                plt.hist(data, bins=current_bin_edges, log=log_counts, weights=numpy.ones(max_count)*100/max_count, cumulative=True, label='Pass Upper Bound',alpha=0.6, color='#009DAE')#edgecolor='black', linewidth=1.2,
            if plot_LL:
                plt.hist(data, bins=current_bin_edges, log=log_counts, weights=numpy.ones(max_count)*100/max_count, cumulative=-1, label='Pass Lower Bound',alpha=0.6, color='#F38BA0')#edgecolor='black', linewidth=1.2,

            ax2 = ax1.twinx()
            mn, mx = ax1.get_ylim()
            ax2.set_ylim(mn*max_count/100,mx*max_count/100)
            ax2.set_ylabel('Counts', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][dict_index])
            if log_counts:
                ax2.set_yscale('log')
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%i'))

            if plot_daynight:
                if daynight_mode == 'sun':
                    data = ds.getDataArrayFromParam('sun_el', eventids_dict=eventids_dict)
                    min_daynight = min(min_daynight, min(data))
                    max_daynight = max(max_daynight, max(data))
                    max_count = len(data)
                    overflow = sum(numpy.logical_or(data < min(daynight_edges), data > max(daynight_edges)))
                    
                    counts = numpy.histogram(data,bins=daynight_edges)[0]
                    counts = numpy.divide(counts, daynight_time_weights, out=numpy.zeros(len(counts)), where=daynight_time_weights!=0)
                    #Convert counts to density
                    # counts = counts / (sum(counts) * (daynight_centers[1]-daynight_centers[0]))

                    for _ax_daynight in axs_daynight:
                        #Plot normal histogram
                        plt.sca(_ax_daynight)
                        plt.bar(daynight_centers, counts, width=daynight_centers[1]-daynight_centers[0], label='%s, Overflow: %i'%(dict_name,overflow),alpha=0.6, edgecolor='black', linewidth=1.0)
                        #_ax_daynight.set_yscale('log')

                elif daynight_mode == 'time':
                    if wrap_daynight == True:
                        min_daynight = 0
                        max_daynight = 24

                        data = f_daynight_wrap(ds.getDataArrayFromParam('calibrated_trigtime', eventids_dict=eventids_dict))
                        counts = numpy.histogram(data,bins=daynight_edges)[0]
                        counts = numpy.divide(counts, daynight_time_weights, out=numpy.zeros(len(counts)), where=daynight_time_weights!=0)
                    else:
                        min_daynight = start_reference_time
                        max_daynight = finish_reference_time
                        data = (ds.getDataArrayFromParam('calibrated_trigtime', eventids_dict=eventids_dict) - start_reference_time)/(60*60)
                        counts = numpy.histogram(data,bins=daynight_edges)[0]
                    # counts = counts / (sum(counts) * (daynight_centers[1]-daynight_centers[0])) #Normalize counts?  I don't think I want to do this actuall, not as visually nice. 

                    for _ax_daynight in axs_daynight:
                        #Plot normal histogram
                        plt.sca(_ax_daynight)
                        if log_diff == False:
                            plt.bar(daynight_centers, counts, width=daynight_centers[1]-daynight_centers[0], label='%s'%(dict_name),alpha=0.6, edgecolor='black', linewidth=1.0)
                            _ax_daynight.set_yscale('log')
                        else:
                            plt.bar(daynight_centers, numpy.log10(numpy.abs(counts - numpy.mean(counts)))*numpy.sign(counts-numpy.mean(counts)), width=daynight_centers[1]-daynight_centers[0], label='%s'%(dict_name),alpha=0.6, edgecolor='black', linewidth=1.0)
                        # counts = plt.hist(data, bins=daynight_edges, log=True, cumulative=False, density=True, label='%s, Overflow: %i'%(dict_name,overflow),alpha=0.6, edgecolor='black', linewidth=1.0)[0]


        if param_key in list(point_of_interest_dict.keys()):
            if 'similarity' in param_key:
                plot_UL = True
                plot_LL = False
            else:
                plot_UL = False
                plot_LL = True

            for key in list(point_of_interest_dict[param_key].keys()):
                if point_of_interest_dict[param_key][key] is None:
                    run = int(key.split('-')[0])
                    eventid = int(key.split('-')[1])
                    val = ds.getDataArrayFromParam(param_key, eventids_dict={run:numpy.array([eventid])})[0]
                    linestyle = '--'
                    linecolor = 'g'
                    if plot_daynight:
                        if daynight_mode == 'sun':
                            val_daynight = ds.getDataArrayFromParam('sun_el', eventids_dict={run:numpy.array([eventid])})[0]
                        elif daynight_mode == 'time':
                            if wrap_daynight == True:
                                val_daynight = f_daynight_wrap(ds.getDataArrayFromParam('calibrated_trigtime', eventids_dict={run:numpy.array([eventid])})[0])
                            else:
                                val_daynight = (ds.getDataArrayFromParam('calibrated_trigtime', eventids_dict={run:numpy.array([eventid])})[0] - start_reference_time)/(60*60)
                else:
                    val = point_of_interest_dict[param_key][key]
                    linestyle = '-'
                    if 'UL' in key:
                        if not plot_UL:
                            continue
                        linecolor = 'b'
                    elif 'LL':
                        if not plot_LL:
                            continue
                        linecolor = 'r'
                    else:
                        linecolor = 'k'


                for axis_index, _ax in enumerate(fig.axes):
                    if r'%' in _ax.get_ylabel():
                        continue
                    elif plot_daynight:
                        if _ax.get_xlabel() == daynight_xlabel:
                            if point_of_interest_dict[param_key][key] is None:
                                for _ax_daynight in axs_daynight:
                                    plt.sca(_ax_daynight)
                                    plt.axvline(val_daynight, label=key + ' = %0.3f'%val_daynight, linestyle=linestyle,color=linecolor, linewidth=2)
                                    _ax.set_xlim(min_daynight,max_daynight)
                            continue

                    plt.sca(_ax)
                    plt.axvline(val, label=key + ' = %0.3f'%val, linestyle=linestyle,color=linecolor, linewidth=2)
                    #plt.legend(loc='center right')
                    _ax.set_xlim(min(current_bin_edges),max(current_bin_edges))
                
        # Put a legend to the right of the current axis
        for axis_index, _ax in enumerate(fig.axes):
            if r'%' in _ax.get_ylabel():
                continue
            plt.sca(_ax)
            if plot_daynight == False:
                legend = plt.legend(loc='center left', fontsize=10, bbox_to_anchor=(1.15, 0.5))
                legend.get_frame().set_alpha(1)
            else:
                plt.sca(ax_daynight)
                legend = plt.legend(loc='upper center', fontsize=10, bbox_to_anchor=(0.5, -0.25))
                legend.get_frame().set_alpha(1)

        plt.tight_layout()
        if plot_daynight == False:
            plt.subplots_adjust(left=0.1, right=0.7, hspace=0.7)
        else:
            plt.subplots_adjust(wspace=0.3, left=0.06, right=0.9, hspace=0.5, top=0.95, bottom=0.2)



        if plot_daynight and standalone_daynight:
            plt.sca(ax_daynight2)
            legend = plt.legend(loc='upper right', fontsize=10)
            plt.tight_layout()
        # fig.savefig(os.path.join('./',param_key + '_%i.png'%(time.time())),dpi=300)

