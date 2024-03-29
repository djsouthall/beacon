#!/usr/bin/env python3
'''
This script is intended to look at the events that construct best above horizon in allsky maps.  I want to view the
peak to sidelobe values and other parameters for the belowhorizon and abovehorizon maps and determine if there is
an obvious cut for which sidelobed above horizon events can be discriminated. 
'''

import sys
import os
import gc
import inspect
import h5py
import copy
from pprint import pprint
import textwrap
from multiprocessing import Process

import numpy
import scipy
import scipy.signal
import pandas as pd

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
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, FuncFormatter
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

gc.enable()
if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length
    
    runs = numpy.array([5911]) #To make a data slicer with.  In general just loading the data though.

    print("Preparing dataSlicer")

    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, \
                    low_ram_mode=False,\
                    analysis_data_dir=processed_datapath, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                    n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)

    cut_dict_key = 'above horizon full with stage 2'
    # ds.addROI(cut_dict_key, {   'elevation_best_choice':[10,90],
    #                             'phi_best_choice':[-90,90],
    #                             'similarity_count_h':[-0.1,10],
    #                             'similarity_count_v':[-0.1,10],
    #                             'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':[1.2,1e10],
    #                             'impulsivity_hSLICERADDimpulsivity_v':[0.3,1e10],
    #                             'cr_template_search_hSLICERMAXcr_template_search_v':[0.4,1e10],
    #                             'p2p_gap_h':[-1e10, 95],
    #                             'above_normalized_map_max_line':[0,1e10],
    #                             'above_snr_line':[0,1e10],
    #                             'in_targeted_box':[-0.5, 0.5]})

    ds.addROI(cut_dict_key, {   'elevation_best_choice':[10,90],
                                'phi_best_choice':[-90,90],
                                'similarity_count_h':[-0.1,10],
                                'similarity_count_v':[-0.1,10],
                                'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':[1.2,1e10],
                                'impulsivity_hSLICERADDimpulsivity_v':[0.3,1e10],
                                'cr_template_search_hSLICERMAXcr_template_search_v':[0.4,1e10],
                                'p2p_gap_h':[-1e10, 95],
                                'above_normalized_map_max_line':[0,1e10],
                                'above_snr_line':[0,1e10]})



    #Add - 1 ROI:
    for param_key in list(ds.roi[cut_dict_key].keys()):
        copied_dict = copy.deepcopy(ds.roi[cut_dict_key])
        del copied_dict[param_key]
        ds.addROI(cut_dict_key + '-' + param_key, copy.deepcopy(copied_dict)) #Add an ROI containing all cuts BUT the current key (used as ROI label).

    root_dir = '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/cuts_run5733-run6640_1652152119'#'./data/cuts_run5733-run6640_1651969811'
    load_dirs = [os.path.join(root_dir,'no_cuts'), os.path.join(root_dir,'all_but_cuts')]#[os.path.join(root_dir,'all_cuts'), os.path.join(root_dir,'all_but_cuts'), os.path.join(root_dir,'no_cuts')]

    save_dir = './figures/cut_histograms/v2/'


    label_group_0 =  ['elevation_best_choice','phi_best_choice','similarity_count_h','similarity_count_v','hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe']
    label_group_1 =  ['impulsivity_hSLICERADDimpulsivity_v','cr_template_search_hSLICERMAXcr_template_search_v','p2p_gap_h','above_normalized_map_max_line','above_snr_line']
    label_group_2 =  ['elevation_best_choice','phi_best_choice','impulsivity_hSLICERADDimpulsivity_v','cr_template_search_hSLICERMAXcr_template_search_v']
    label_group_3 =  ['impulsivity_hSLICERADDimpulsivity_v','cr_template_search_hSLICERMAXcr_template_search_v']
    label_group_4 =  ['elevation_best_choice','phi_best_choice']
    label_group_5 =  [label_group_0, label_group_1]
    label_group_6 =  [['elevation_best_choice','impulsivity_hSLICERADDimpulsivity_v'],['phi_best_choice','cr_template_search_hSLICERMAXcr_template_search_v']]
    plt.ioff()
    # plt.ion()



    if True:
        excel_filename = '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/new-cut-event-info_master_updated.xlsx'
        with pd.ExcelFile(excel_filename) as xls:
            for sheet_index, sheet_name in enumerate(xls.sheet_names):
                if sheet_name != 'passing all cuts':
                    continue
                else:
                    df = xls.parse(sheet_name)
                    df = df.sort_values(by = ['run', 'eventid'], ascending = [True, True], na_position = 'last')
        filtered_df = df.query('key == "good" and suspected_airplane_icao24 != suspected_airplane_icao24')


    do = [3,4]
    for lg_index, label_group in enumerate([label_group_0, label_group_1, label_group_2, label_group_3, label_group_4, label_group_5, label_group_6]):
        if lg_index not in do:
            continue
        if lg_index > 1 and lg_index < 5:
            fig = plt.figure(figsize=(15,5*len(label_group)))
            major_fontsize = 36
            minor_fontsize = 20
            # major_fontsize = 42
            # minor_fontsize = 28
        elif lg_index >= 5:
            fig = plt.figure(figsize=(12*len(label_group),6*len(label_group[0])))
            major_fontsize = 42
            minor_fontsize = 28
        else:
            fig = plt.figure(figsize=(12,4*len(label_group)))
            major_fontsize = 36
            minor_fontsize = 20

        for param_index, param_key in enumerate(numpy.asarray(label_group).T.flatten()):
            if True:
                if param_key == 'elevation_best_choice':
                    xlabel = 'Elevation'
                    x_units = ' (deg)'
                elif param_key == 'phi_best_choice':
                    xlabel = 'Azimuth'
                    x_units = ' (deg)'
                elif param_key == 'similarity_count_h':
                    xlabel = 'Time Delay Clustering H'
                    x_units = ' (Counts)'
                elif param_key == 'similarity_count_v':
                    xlabel = 'Time Delay Clustering V'
                    x_units = ' (Counts)'
                elif param_key == 'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':
                    xlabel = 'Peak-to-Sidelobe , Max(H,V)'
                    x_units = ''
                elif param_key == 'impulsivity_hSLICERADDimpulsivity_v':
                    xlabel = 'Impulsivity H + V'
                    x_units = ''
                elif param_key == 'cr_template_search_hSLICERMAXcr_template_search_v':
                    xlabel = 'CR Template Correlation, Max(H,V)'
                    x_units = ''
                elif param_key == 'p2p_gap_h':
                    xlabel = 'Signal Amplitude Differences'
                    x_units = ''
                elif param_key == 'above_normalized_map_max_line':
                    xlabel = 'Combined Normalized Map Peak'
                    x_units = ''
                elif param_key == 'above_snr_line':
                    xlabel = 'Combined P2P/(2 STD)'
                    x_units = ''
                elif param_key == 'in_targeted_box':
                    xlabel = 'Target Below Horizon Box'
                    x_units = ' (0 = Not Inside, 1 = Inside)'
            else:
                if param_key == 'elevation_best_choice':
                    xlabel = 'Elevation'
                    x_units = ' (deg)'
                elif param_key == 'phi_best_choice':
                    xlabel = 'Azimuth'
                    x_units = ' (deg)'
                elif param_key == 'similarity_count_h':
                    xlabel = 'Time Delay Similarity H'
                    x_units = ' (Counts)'
                elif param_key == 'similarity_count_v':
                    xlabel = 'Time Delay Similarity V'
                    x_units = ' (Counts)'
                elif param_key == 'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':
                    xlabel = 'Peak to Sidelobe , Max(H,V)'
                    x_units = ''
                elif param_key == 'impulsivity_hSLICERADDimpulsivity_v':
                    xlabel = 'Impulsivity H + V'
                    x_units = ''
                elif param_key == 'cr_template_search_hSLICERMAXcr_template_search_v':
                    xlabel = 'CR Template Correlation, Max(H,V)'
                    x_units = ''
                elif param_key == 'p2p_gap_h':
                    xlabel = 'Max P2P (H) - Min P2P (H)'
                    x_units = ''
                elif param_key == 'above_normalized_map_max_line':
                    xlabel = 'Distance Above Map Max Line'
                    x_units = ''
                elif param_key == 'above_snr_line':
                    xlabel = 'Distance From P2P/(2*STD) Line'
                    x_units = ''
                elif param_key == 'in_targeted_box':
                    xlabel = 'Target Below Horizon Box'
                    x_units = ' (0 = Not Inside, 1 = Inside)'


            # if lg_index > 1:
            #     ylabel = 'Normalized Counts'
            # else:
            ylabel = 'Normalized\nCounts'

            if lg_index >= 5:
                if param_key in label_group[0]:
                    ax1 = plt.subplot(len(label_group[0]), len(label_group), 1 + param_index)
                else:
                    ax1 = plt.subplot(len(label_group[0]), len(label_group), 1 + param_index)
            else:
                ax1 = plt.subplot(len(label_group), 1, 1 + param_index)

            filename = 'hist_for_%s_with_no_cuts.npz'%(param_key)
            # plt.title('No Cuts Applied')

            cut_name = cut_dict_key + '-' + param_key
            loaded_eventids_dict = {}
            hist_data = {}
            with numpy.load(os.path.join(root_dir,'no_cuts', filename)) as data:
                for key in list(data.keys()):
                    if key.isdigit():
                        loaded_eventids_dict[int(key)] = data[key]
                    else:
                        hist_data[key] = data[key]
            print(cut_name)
            print(list(hist_data.keys()))

            if param_key == 'cr_template_search_hSLICERMAXcr_template_search_v':
                xlim = min(hist_data['bin_centers'][hist_data['counts'] > 0]) , 1.03
            else:
                xlim = min(hist_data['bin_centers'][hist_data['counts'] > 0]), max(hist_data['bin_centers'][hist_data['counts'] > 0])

            ax1.minorticks_on()
            ax1.grid(b=True, which='major', color='tab:gray', linestyle='-',alpha=0.4)
            ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)

            labeled = False
            if ds.roi[cut_dict_key][param_key][0] >= min(hist_data['bin_centers']) and ds.roi[cut_dict_key][param_key][0] <= max(hist_data['bin_centers']):
                # plt.axvline(ds.roi[cut_dict_key][param_key][0], c='g', label=textwrap.fill('Cut Lower Limit',width=25))
                ax1.axvspan(xlim[0], ds.roi[cut_dict_key][param_key][0], color='r', label=textwrap.fill('Cut Regions',width=25), alpha=0.3)
                labeled = True
            if ds.roi[cut_dict_key][param_key][1] >= min(hist_data['bin_centers']) and ds.roi[cut_dict_key][param_key][1] <= max(hist_data['bin_centers']):
                # plt.axvline(ds.roi[cut_dict_key][param_key][1], c='b', label=textwrap.fill('Cut Upper Limit',width=25))
                if labeled == False:
                    ax1.axvspan(ds.roi[cut_dict_key][param_key][1], xlim[1], color='r', label=textwrap.fill('Cut Regions',width=25), alpha=0.3)
                else:
                    ax1.axvspan(ds.roi[cut_dict_key][param_key][1], xlim[1], color='r', alpha=0.3)


            normalization_factor = hist_data['bin_width']*sum(hist_data['counts'])


            plt.bar(hist_data['bin_centers'], hist_data['counts']/normalization_factor, hist_data['bin_width'], facecolor='k', label='Before Cuts')




            filename = 'hist_for_%s_with_all_cuts_but_%s.npz'%(param_key, param_key)
            # plt.title('All Cuts Applied Except:\n%s'%param_key.replace('_',' ').title())

            cut_name = cut_dict_key + '-' + param_key
            loaded_eventids_dict = {}
            hist_data = {}
            with numpy.load(os.path.join(root_dir,'all_but_cuts', filename)) as data:
                for key in list(data.keys()):
                    if key.isdigit():
                        loaded_eventids_dict[int(key)] = data[key]
                    else:
                        hist_data[key] = data[key]
            print(cut_name)
            print(list(hist_data.keys()))


            plt.bar(hist_data['bin_centers'], hist_data['counts']/normalization_factor, hist_data['bin_width'], facecolor='dodgerblue', label='If Last Cut')

            if ds.roi[cut_dict_key][param_key][0] >= min(hist_data['bin_centers']) and ds.roi[cut_dict_key][param_key][0] <= max(hist_data['bin_centers']):
                plt.axvline(ds.roi[cut_dict_key][param_key][0], c='r', lw=1, alpha=0.7)
            if ds.roi[cut_dict_key][param_key][1] >= min(hist_data['bin_centers']) and ds.roi[cut_dict_key][param_key][1] <= max(hist_data['bin_centers']):
                plt.axvline(ds.roi[cut_dict_key][param_key][1], c='r', lw=1, alpha=0.7)



            if True:
                cr_val = ds.getDataArrayFromParam(param_key, eventids_dict={5911:[73399]})
                ax1.axvline(cr_val, c='gold', lw=2, label='5911-73399')

            # xy = [hist_data['bin_centers'][numpy.argmax(hist_data['counts'])] , numpy.max(numpy.argmax(hist_data['counts']))/normalization_factor]
            if param_key == 'elevation_best_choice' and lg_index > 1:
                cut_range = numpy.logical_and(hist_data['bin_centers'] > 30, hist_data['bin_centers'] < 40)
                xy = [hist_data['bin_centers'][cut_range][numpy.argmax(hist_data['counts'][cut_range])], numpy.max(hist_data['counts'][cut_range]/normalization_factor)]
                xy_text = [52,0.5e-1]

                # raw_text = "Above horizon clustering corresponds to below horizon events misreconstructing to a prominent sidelobe"
                # text = textwrap.fill(raw_text,width=int(len(raw_text)//3))
                if lg_index >= 5 and False:
                    text = 'Events misreconstructing\nto a prominent sidelobe'
                    ann = ax1.annotate(text,
                          xy=xy, xycoords='data',
                          xytext=xy_text, textcoords='data',
                          size=minor_fontsize, va="center", ha="center",
                          bbox=dict(boxstyle="round", fc="w"),
                          arrowprops=dict(arrowstyle="-|>",
                                          connectionstyle="arc3,rad=+0.2",
                                          fc="r",ec="r"),
                          )
                elif False:
                    text = 'Above horizon clustering corresponds\nto below horizon events misreconstructing\nto a prominent sidelobe'
                    ann = ax1.annotate(text,
                          xy=xy, xycoords='data',
                          xytext=xy_text, textcoords='data',
                          size=16, va="center", ha="center",
                          bbox=dict(boxstyle="round", fc="w"),
                          arrowprops=dict(arrowstyle="-|>",
                                          connectionstyle="arc3,rad=+0.2",
                                          fc="r",ec="r"),
                          )


            if lg_index == 5:
                if param_key == 'similarity_count_h':
                    plt.ylabel('Normalized Counts', fontsize=major_fontsize)
            elif lg_index == 6:
                if param_key in ['elevation_best_choice','impulsivity_hSLICERADDimpulsivity_v']:
                    plt.ylabel('Normalized Counts', fontsize=major_fontsize)
            else:
                plt.ylabel(ylabel, fontsize=major_fontsize)

            


            if True:
                vals = filtered_df[param_key].to_numpy()
                hist_data['bin_centers']
                bin_centers = hist_data['bin_centers']
                bin_width = hist_data['bin_width']
                bin_edges = numpy.zeros(len(bin_centers) + 1)
                bin_edges[0:len(bin_centers)] = bin_centers - bin_width/2
                bin_edges[-1] = bin_edges[-2] + bin_width

                ax1.hist(vals,facecolor='gold', bins=bin_edges, weights=numpy.ones_like(vals)/normalization_factor, label='Remaining %i Events'%len(vals))

            plt.xlabel(xlabel + x_units, fontsize=major_fontsize)
            plt.xticks(fontsize=minor_fontsize)
            plt.yticks(fontsize=minor_fontsize)



            plt.xlim(xlim[0], xlim[1])
            ax1.set_yscale('log')
            ax1.tick_params(axis='both', labelsize=minor_fontsize)

            if lg_index == 3 or lg_index == 4:
                print('Adding custom legend!!!!!!!!!!!!!')
                #Need to change these to be present for each figure appropriately to the paper.  Additionally need to make
                #sure this added horizon line works.
                #Lg index 6 is wrong, I don't want the combined plot I want the 2 seperate plots.  
                if lg_index == 4 and param_index == 0:
                    horizon_angle = -1.5
                    horizon_label = 'Horizon'#'Horizon\nElevation ~ %0.1f deg'%horizon_angle
                    horizon_line = ax1.axvline(horizon_angle,linestyle='-.', alpha=1.0, linewidth=2.0, c='k', label=horizon_label)

                #     #['5911-73399', 'Horizon', 'Cut Regions', 'Remaining 36 Events', 'Before Cuts', 'If Last Cut']
                #     numpy.array([1,0,5])
                #     lines_2_cut = numpy.array([3,4,2])
                # else:
                #     lines_1_cut = numpy.array([1,0])
                #     lines_2_cut = numpy.array([3,4,2])

                lines, labels = plt.gca().get_legend_handles_labels()
                lines = numpy.asarray(lines)
                labels = numpy.asarray(labels)

                if False:
                    if 'Horizon' in labels:
                        labels_1 = ['Cut Regions', '5911-73399', 'Horizon']
                    else:
                        labels_1 = ['Cut Regions', '5911-73399']
                    lines_1_cut = []
                    for l in labels_1:
                        lines_1_cut.append(numpy.where(labels == l)[0][0])
                    lines_1_cut = numpy.asarray(lines_1_cut, dtype=int)
                    
                    labels_2 = ['Before Cuts', 'If Last Cut', 'Remaining 36 Events']
                    lines_2_cut = []
                    for l in labels_2:
                        lines_2_cut.append(numpy.where(labels == l)[0][0])
                    lines_2_cut = numpy.asarray(lines_2_cut, dtype=int)

                    ax_list = fig.axes
                    if param_index == 0:
                        ax_list[0].legend(lines[lines_1_cut], labels[lines_1_cut], loc='upper right')
                    elif param_index == 1:
                        ax_list[1].legend(lines[lines_2_cut], labels[lines_2_cut], loc='upper right')
                else:

                    if 'Horizon' in labels:
                        label_order = ['Before Cuts', 'If Last Cut', 'Remaining 36 Events', 'Cut Regions', '5911-73399', 'Horizon']
                    else:
                        label_order = ['Before Cuts', 'If Last Cut', 'Remaining 36 Events', 'Cut Regions', '5911-73399']
                    label_order_cut = []
                    for l in label_order:
                        label_order_cut.append(numpy.where(labels == l)[0][0])
                    label_order_cut = numpy.asarray(label_order_cut, dtype=int)
                    
                    ax_list = fig.axes
                    if param_index == 0:
                        ax_list[0].legend(lines[label_order_cut], labels[label_order_cut], loc='upper right', ncol=2)

            elif param_index == len(numpy.asarray(label_group).T.flatten())-1:
                plt.legend(loc='upper right', fontsize=minor_fontsize)


            plt.tight_layout()
            plt.subplots_adjust(hspace=0.30)#left=0.06, bottom=0.12,right=0.98, top=0.95, wspace=0.3, 

        fig.savefig(os.path.join(save_dir, 'cuts_same_axis_pg_%i_v2.pdf'%(lg_index+1)))
        fig.savefig(os.path.join(save_dir, 'cuts_same_axis_pg_%i_v2.png'%(lg_index+1)), dpi=300)
        # fig.savefig('./figures/cut_histograms/cuts_same_axis_pg_%i.pdf'%(lg_index+1))
        # fig.savefig('./figures/cut_histograms/cuts_same_axis_pg_%i.png'%(lg_index+1), dpi=300)
