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
    ds.addROI(cut_dict_key, {   'elevation_best_choice':[10,90],
                                'phi_best_choice':[-90,90],
                                'similarity_count_h':[-0.1,10],
                                'similarity_count_v':[-0.1,10],
                                'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':[1.2,1e10],
                                'impulsivity_hSLICERADDimpulsivity_v':[0.3,1e10],
                                'cr_template_search_hSLICERMAXcr_template_search_v':[0.4,1e10],
                                'p2p_gap_h':[-1e10, 95],
                                'above_normalized_map_max_line':[0,1e10],
                                'above_snr_line':[0,1e10],
                                'in_targeted_box':[-0.5, 0.5]})


    #Add - 1 ROI:
    for param_key in list(ds.roi[cut_dict_key].keys()):
        copied_dict = copy.deepcopy(ds.roi[cut_dict_key])
        del copied_dict[param_key]
        ds.addROI(cut_dict_key + '-' + param_key, copy.deepcopy(copied_dict)) #Add an ROI containing all cuts BUT the current key (used as ROI label).

    root_dir = './data/cuts_run5733-run6640_1651969811'
    load_dirs = [os.path.join(root_dir,'no_cuts'), os.path.join(root_dir,'all_but_cuts'), ]#[os.path.join(root_dir,'all_cuts'), os.path.join(root_dir,'all_but_cuts'), os.path.join(root_dir,'no_cuts')]

    save_dir = './figures/cut_histograms'

    for param_key in list(ds.roi[cut_dict_key].keys()):
        if param_key in ['elevation_best_choice', 'phi_best_choice']:
            continue
        else:
            fig = plt.figure(figsize=(25,10))
            axs = []
            for dir_index, load_dir in enumerate(load_dirs):
                if dir_index == 0:
                    axs.append(plt.subplot(1,len(load_dirs),dir_index+1))
                else:
                    axs.append(plt.subplot(1,len(load_dirs),dir_index+1, sharex=axs[0]))

            for dir_index, load_dir in enumerate(load_dirs):
                if 'all_cuts' in load_dir:
                    filename = 'hist_for_%s_with_all_cuts.npz'%(param_key)
                    plt.title('All Cuts Applied')
                elif 'all_but_cuts' in load_dir:
                    filename = 'hist_for_%s_with_all_cuts_but_%s.npz'%(param_key, param_key)
                    plt.title('All Cuts Applied Except:\n%s'%param_key.replace('_',' ').title())
                else:
                    filename = 'hist_for_%s_with_no_cuts.npz'%(param_key)
                    plt.title('No Cuts Applied')


                cut_name = cut_dict_key + '-' + param_key
                loaded_eventids_dict = {}
                hist_data = {}
                with numpy.load(os.path.join(load_dir, filename)) as data:
                    for key in list(data.keys()):
                        if key.isdigit():
                            loaded_eventids_dict[int(key)] = data[key]
                        else:
                            hist_data[key] = data[key]
                print(cut_name)
                print(list(hist_data.keys()))

                ax = axs[dir_index]
                plt.sca(ax)
                ax.minorticks_on()
                ax.grid(b=True, which='major', color='tab:gray', linestyle='-',alpha=0.4)
                ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)

                plt.bar(hist_data['bin_centers'], hist_data['counts'], hist_data['bin_width'], edgecolor='k', facecolor=None)

                # # plt.axvline(polarization_degs[0], c='r', label=textwrap.fill('5911-73399 with Symettric Filtering',width=25))
                # # plt.axvline(polarization_degs[1], c='g', label='5911-73399 with Asymettric Filtering')
                # # plt.axvline(polarization_degs[2], c='m', label='5911-73399 with Only Sine Subtraction')
                # plt.axvline(polarization_degs[0], c='r', label=textwrap.fill('5911-73399 Polarization',width=25))


                if ds.roi[cut_dict_key][param_key][0] >= min(hist_data['bin_centers']) and ds.roi[cut_dict_key][param_key][0] <= max(hist_data['bin_centers']):
                    plt.axvline(ds.roi[cut_dict_key][param_key][0], c='g', label=textwrap.fill('Cut Lower Limit',width=25))
                if ds.roi[cut_dict_key][param_key][1] >= min(hist_data['bin_centers']) and ds.roi[cut_dict_key][param_key][1] <= max(hist_data['bin_centers']):
                    plt.axvline(ds.roi[cut_dict_key][param_key][1], c='b', label=textwrap.fill('Cut Upper Limit',width=25))

                plt.legend(loc='center right', fontsize=12)
                plt.ylabel('Counts\nAll Cuts Applied Except %s'%param_key, fontsize=18)

                if param_key == 'above_snr_line':
                    plt.xlabel('Distance From\nNormalized P2P/(2*STD) Line', fontsize=18)
                else:
                    plt.xlabel(hist_data['label'], fontsize=18)

                # ax.text(0.5, 0.5, 'Preliminary and Placeholder', transform=ax.transAxes,
                #     fontsize=40, color='gray', alpha=0.5,
                #     ha='center', va='center', rotation='30')



