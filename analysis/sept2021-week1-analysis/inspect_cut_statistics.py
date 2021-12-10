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

import numpy
import scipy
import scipy.signal
import scipy.signal

from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.line_of_sight import circleSource

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
#processed_datapath = os.environ['BEACON_PROCESSED_DATA']
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

    runs = numpy.arange(5910,5912)

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

    #ds.printKnownParamKeys()
    ds.addROI('above horizon',{'elevation_best_h':[10,90],'phi_best_h':[-90,90]})
    ds.addROI('above horizon sim',{'elevation_best_h':[10,90],'phi_best_h':[-90,90],'similarity_count_h':[0,10]})
    ds.addROI('above horizon full',{'elevation_best_h':[10,90],'phi_best_h':[-90,90],'elevation_best_v':[10,90],'phi_best_v':[-90,90],'similarity_count_h':[0,10],'similarity_count_v':[0,10],'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,10],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]})

    above_horizon_eventids_dict = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
    #above_horizon_sim_eventids_dict, successive_cut_counts, total_cut_counts = ds.getCutsFromROI('above horizon sim',load=False,save=False,verbose=True, return_successive_cut_counts=True, return_total_cut_counts=True)
    above_horizon_full_eventids_dict, successive_cut_counts, total_cut_counts = ds.getCutsFromROI('above horizon full',load=False,save=False,verbose=True, return_successive_cut_counts=True, return_total_cut_counts=True)

    #Choose list of parameters you want plotted.  

    #params = ['elevation_best_h', 'phi_best_h', 'elevation_best_v', 'phi_best_v', 'similarity_count_h', 'similarity_count_v', 'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe', 'impulsivity_hSLICERADDimpulsivity_v', 'cr_template_search_hSLICERADDcr_template_search_v']
    #params = ['hpol_peak_to_sidelobe', 'impulsivity_h', 'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe']
    params = list(ds.roi['above horizon full'].keys())
    #params = ['impulsivity_hSLICERMAXimpulsivity_v', 'cr_template_search_hSLICERMAXcr_template_search_v', 'std_hSLICERMAXstd_v', 'p2p_hSLICERMAXp2p_v', 'snr_hSLICERMAXsnr_v', 'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe', 'hpol_max_possible_map_valueSLICERMAXvpol_max_possible_map_value']

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

    dicts = [None, above_horizon_eventids_dict, above_horizon_full_eventids_dict]
    dict_names = ['All RF Events', 'Forward Above Horizon', 'Forward Above Horizon\n+ Quality Cuts']
    log_counts = False

    for dict_index, eventids_dict in enumerate(dicts):
        fig_map, ax_map = ds.plotROI2dHist('phi_best_h_allsky', 'elevation_best_h_allsky', cmap='cool', eventids_dict=eventids_dict, include_roi=False)
        fig_map.canvas.set_window_title(dict_names[dict_index].replace('\n',''))

    for param_key_index, param_key in enumerate(params):
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


        fig = plt.figure()
        fig.canvas.set_window_title(current_label)

        ax = plt.subplot(len(dicts)+1,1,1)
        plt.grid(which='both', axis='both')
        ax.minorticks_on()
        ax.grid(b=True, which='major', color='k', linestyle='-')
        ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.ylabel('Normalized Counts')
        plt.xlabel(current_label)


        for dict_index, eventids_dict in enumerate(dicts):

            data = ds.getDataArrayFromParam(param_key, eventids_dict=eventids_dict)
            max_count = len(data)
            overflow = sum(numpy.logical_or(data < min(current_bin_edges), data > max(current_bin_edges)))
            
            # if overflow > 0:
            #     print(param_key, overflow)
            #     import pdb; pdb.set_trace()
            #Plot normal histogram
            plt.sca(ax)
            plt.hist(data, bins=current_bin_edges, log=True, cumulative=False, density=True, label='dict = %s\nOverflow: %i'%(dict_names[dict_index],overflow),alpha=0.6, edgecolor='black', linewidth=1.2)
            #plt.legend(loc='upper right')
            
            #Plot cumulative histogram
            ax1 = plt.subplot(len(dicts)+1,1,dict_index + 2)
            plt.grid(which='both', axis='both')
            ax1.minorticks_on()
            ax1.grid(b=True, which='major', color='k', linestyle='-')
            ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            plt.xlabel(current_label)
            ax1.set_ylabel(r'% Of Events' + '\n%s'%dict_names[dict_index], color=plt.rcParams['axes.prop_cycle'].by_key()['color'][dict_index]) 
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))



            plt.hist(data, bins=current_bin_edges, log=log_counts, weights=numpy.ones(max_count)*100/max_count, cumulative=True, label='Pass Upper Bound',alpha=0.6, color='#009DAE')#edgecolor='black', linewidth=1.2,
            plt.hist(data, bins=current_bin_edges, log=log_counts, weights=numpy.ones(max_count)*100/max_count, cumulative=-1, label='Pass Lower Bound',alpha=0.6, color='#F38BA0')#edgecolor='black', linewidth=1.2,
            # if dict_index == 0:
                #plt.legend(loc='center right')

            ax2 = ax1.twinx()
            mn, mx = ax1.get_ylim()
            ax2.set_ylim(mn*max_count/100,mx*max_count/100)
            ax2.set_ylabel('Counts', color=plt.rcParams['axes.prop_cycle'].by_key()['color'][dict_index])
            if log_counts:
                ax2.set_yscale('log')
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%i'))
            # ax2.tick_params(direction='inout', which='major', color='k')
            # ax2.tick_params(direction='inout', which='minor', color='tab:gray')
            # ax1.tick_params(direction='inout', which='both')

        if param_key in list(point_of_interest_dict.keys()):
            for key in list(point_of_interest_dict[param_key].keys()):
                if point_of_interest_dict[param_key][key] is None:
                    run = int(key.split('-')[0])
                    eventid = int(key.split('-')[1])
                    val = ds.getDataArrayFromParam(param_key, eventids_dict={run:numpy.array([eventid])})[0]
                    linestyle = '--'
                    linecolor = 'g'
                else:
                    val = point_of_interest_dict[param_key][key]
                    linestyle = '-'
                    if 'UL' in key:
                        linecolor = 'b'
                    elif 'LL':
                        linecolor = 'r'
                    else:
                        linecolor = 'k'


                for _ax in fig.axes:
                    if r'%' in _ax.get_ylabel():
                        continue
                    plt.sca(_ax)
                    plt.axvline(val, label=key + ' = %0.3f'%val, linestyle=linestyle,color=linecolor, linewidth=2)
                    #plt.legend(loc='center right')
                    _ax.set_xlim(min(current_bin_edges),max(current_bin_edges))
                
        for _ax in fig.axes:
            if r'%' in _ax.get_ylabel():
                continue
            plt.sca(_ax)
            #box = _ax.get_position()
            #_ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            legend = plt.legend(loc='center left', fontsize=10, bbox_to_anchor=(1.15, 0.5))
            legend.get_frame().set_alpha(1)
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.7, hspace=0.7)
        