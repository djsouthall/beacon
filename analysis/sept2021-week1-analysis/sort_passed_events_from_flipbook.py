#!/usr/bin/env python3
'''
Given an event flipbook (which serves as a list of events that pass the main cuts), this will take those events
and perform some final cuts.  These per not performed at the original filtering stage for fear of cutting out interesting
events, but not that I have inspected all of these events by eye, it would be good to see if there is an obvious
way to replicate the results of my by-eye sorting.  
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
import time

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.flipbook_reader import flipbookToDict, concatenateFlipbookToArray, concatenateFlipbookToDict
import  beacon.tools.get_plane_tracks as pt
from tools.airplane_traffic_loader import getDataFrames, getFileNamesFromTimestamps

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)


def enu2Spherical(enu):
    '''
    2d array like ((e_0, n_0, u_0), (e_1, n_1, u_1), ... , (e_i, n_i, u_i))

    Return in degrees
    '''
    r = numpy.linalg.norm(enu, axis=1)
    theta = numpy.degrees(numpy.arccos(enu[:,2]/r))
    phi = numpy.degrees(numpy.arctan2(enu[:,1],enu[:,0]))
    # import pdb; pdb.set_trace()
    return numpy.vstack((r,phi,theta)).T

if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length

    # _runs = numpy.arange(5733,5974)#[0:100]
    # bad_runs = numpy.array([5775])
    #_runs = numpy.arange(5733,5800)

    stage1_eye_sorted_sorted_dict = flipbookToDict('/home/dsouthall/scratch-midway2/event_flipbook_1643154940')
    stage1_eye_sorted_eventids_dict = concatenateFlipbookToDict(stage1_eye_sorted_sorted_dict)
    stage1_eye_sorted_eventids_array = concatenateFlipbookToArray(stage1_eye_sorted_sorted_dict)
    stage1_maybe_dict = stage1_eye_sorted_sorted_dict['maybe']['eventids_dict']

    verygood_array  = concatenateFlipbookToArray(stage1_eye_sorted_sorted_dict['very-good'])
    good_array  = concatenateFlipbookToArray(stage1_eye_sorted_sorted_dict['good'])
    maybe_array = concatenateFlipbookToArray(stage1_eye_sorted_sorted_dict['maybe'])
    bad_array   = concatenateFlipbookToArray(stage1_eye_sorted_sorted_dict['bad'])


    runs = list(stage1_eye_sorted_eventids_dict.keys())

    stage2_eye_sorted_dict = flipbookToDict('/home/dsouthall/Projects/Beacon/beacon/analysis/sept2021-week1-analysis/airplane_event_flipbook_1643947072')
    stage2_eye_sorted_array = concatenateFlipbookToArray(stage2_eye_sorted_dict)


    # len(concatenateFlipbookToArray(stage2_eye_sorted_dict['no-obvious-airplane']))
    airplane_eventids_dict = concatenateFlipbookToDict(stage2_eye_sorted_dict['airplanes'])
    maybe_eventids_dict = concatenateFlipbookToDict(stage2_eye_sorted_dict['maybe'])
    lightning_eventids_dict = concatenateFlipbookToDict(stage2_eye_sorted_dict['lightning'])
    good_dict = concatenateFlipbookToDict(stage2_eye_sorted_dict['no-obvious-airplane'])


    force_fit_order = 3 #None to use varying order
    save_figures = False
    if save_figures == True:
        outpath = './airplane_event_flipbook_%i'%time.time() 
        os.mkdir(outpath)
    use_old_code = False
    use_new_code = True

    print("Preparing dataSlicer")

    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

    # import pdb; pdb.set_trace()

    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                    n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)

    ds.prepareCorrelator()

    trigger_time = ds.getDataArrayFromParam('calibrated_trigtime', trigger_types=None, eventids_dict=good_dict)
    stage1_eye_sorted_bad_eventids_dict = ds.returnEventsAWithoutB(stage1_eye_sorted_eventids_dict,stage2_eye_sorted_dict)
                

    print('Generating plots:')

    # plot_params = [['std_h', 'std_v'], ['snr_h', 'snr_v'], ['phi_best_choice','elevation_best_choice'],['hpol_peak_to_sidelobe','vpol_peak_to_sidelobe'],['cr_template_search_h', 'cr_template_search_v'], ['impulsivity_h','impulsivity_v'],['hpol_max_map_value_abovehorizonSLICERDIVIDEhpol_max_possible_map_value','vpol_max_map_value_abovehorizonSLICERDIVIDEvpol_max_possible_map_value']]
    plot_params = [ ['snr_ch0', 'snr_ch1'],
                    ['snr_h', 'snr_v'],
                    ['phi_best_choice','snr_h_over_v']]
    # plot_params = [ ['snr_ch0', 'snr_ch1'],
    #                 ['snr_h', 'snr_v'],
    #                 ['hpol_peak_to_sidelobe','vpol_peak_to_sidelobe'],
    #                 ['cr_template_search_h', 'cr_template_search_v'],
    #                 ['impulsivity_h','impulsivity_v'],
    #                 ['hpol_max_map_value_abovehorizonSLICERDIVIDEhpol_max_possible_map_value','vpol_max_map_value_abovehorizonSLICERDIVIDEvpol_max_possible_map_value'],
    #                 ['hpol_max_map_value_abovehorizonSLICERDIVIDEhpol_max_map_value_belowhorizon','vpol_max_map_value_abovehorizonSLICERDIVIDEvpol_max_map_value_belowhorizon']]
    for key_x, key_y in plot_params:
        print('Generating %s plot'%(key_x + ' vs ' + key_y))
        # ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None, include_roi=False)
        # ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=above_horizon_only_eventids_dict, include_roi=False)
        fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=stage1_eye_sorted_bad_eventids_dict, include_roi=False)
        ax.set_title('All Stage 1 Events')
        fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=stage1_maybe_dict, include_roi=False)
        ax.set_title('Stage 1 Maybes')
        fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=stage1_eye_sorted_eventids_dict, include_roi=False)
        ax.set_title('Stage 2 Events (No Stage 1 Maybes)')

        legend_properties = []
        legend_labels = []


        if True:
            ax, cs = ds.addContour(ax, key_x, key_y, good_dict, 'g', n_contour=5, alpha=0.85, log_contour=True, mask=None, fill_value=None)
            legend_properties.append(cs.legend_elements()[0][0])
            legend_labels.append('Good Events')

            ax, cs = ds.addContour(ax, key_x, key_y, airplane_eventids_dict, 'm', n_contour=5, alpha=0.85, log_contour=True, mask=None, fill_value=None)
            legend_properties.append(cs.legend_elements()[0][0])
            legend_labels.append('Airplane Events')
            
            
            ax, cs = ds.addContour(ax, key_x, key_y, lightning_eventids_dict, 'tab:red', n_contour=5, alpha=0.85, log_contour=True, mask=None, fill_value=None)
            legend_properties.append(cs.legend_elements()[0][0])
            legend_labels.append('Lightning Events')

            ax, cs = ds.addContour(ax, key_x, key_y, maybe_eventids_dict, 'tab:orange', n_contour=5, alpha=0.85, log_contour=True, mask=None, fill_value=None)
            legend_properties.append(cs.legend_elements()[0][0])
            legend_labels.append('Ambiguous Events')

            plt.legend(legend_properties,legend_labels,loc='upper left')

    ds.addROI('High CR Corr',{'cr_template_search_hSLICERADDcr_template_search_v':[1.45,10]})
    high_cr_eventids_dict = ds.getCutsFromROI('High CR Corr',eventids_dict=stage1_eye_sorted_eventids_dict,load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
    high_cr_eventids_dict_no_airplanes = ds.returnEventsAWithoutB(high_cr_eventids_dict,airplane_eventids_dict)

    print('high_cr_eventids_dict_no_airplanes')
    pprint(high_cr_eventids_dict_no_airplanes)
    ds.eventInspector(high_cr_eventids_dict_no_airplanes)

    ds.addROI('High Impulsivity',{'impulsivity_hSLICERADDimpulsivity_v':[1,10]})
    summed_impulsivity = ds.getDataArrayFromParam('impulsivity_hSLICERADDimpulsivity_v', trigger_types=None, eventids_dict=stage1_eye_sorted_eventids_dict)
    high_impulsivity = ds.getCutsFromROI('High Impulsivity',eventids_dict=stage1_eye_sorted_eventids_dict,load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
    high_impulsivity_no_airplanes = ds.returnEventsAWithoutB(high_impulsivity,airplane_eventids_dict)

    print('high_impulsivity_no_airplanes')
    pprint(high_impulsivity_no_airplanes)
    ds.eventInspector(high_impulsivity_no_airplanes)