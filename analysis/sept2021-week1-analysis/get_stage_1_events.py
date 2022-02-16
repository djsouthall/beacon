#!/usr/bin/env python3
'''
This script is the same as inspect_above_horizon_events.py but specifically designed to reduce the clutter and just 
apply the already selected cuts on the data and save the eventids. 

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

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.line_of_sight import circleSource
from beacon.tools.flipbook_reader import flipbookToDict

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

if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'

    batch_number_0 = numpy.arange(5733,5974) # September data
    batch_number_1 = numpy.arange(5974,6073)
    batch_number_2 = numpy.arange(6074,6173)
    batch_number_3 = numpy.arange(6174,6273)
    batch_number_4 = numpy.arange(6274,6373)
    batch_number_5 = numpy.arange(6374,6473)
    batch_number_6 = numpy.arange(6474,6573)
    batch_number_7 = numpy.arange(6574,6673)

    # This is a list of runs that were cancelled or ran out of time for whatever reason.  They should eventually be good 
    # to work with, but in the mean time they are to be ignored so they are not accessed by calculations happen.
    flawed_runs = numpy.array([5775,5981,5993,6033,6090,6520,6537,6538,6539]) 

    # These are runs that were processed correctly, but I choosing to ignore due to abnormal run behaviour.
    ignored_runs = numpy.array([6062,6063,6064]) 

    runs = batch_number_1
    runs = runs[numpy.logical_and(~numpy.isin(runs,flawed_runs),~numpy.isin(runs,ignored_runs))]

    print("Preparing dataSlicer")
    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, verbose_setup=False)

    #This one cuts out ALL events
    ds.addROI('above horizon only',{'elevation_best_choice':[10,90],'phi_best_choice':[-90,90]})
    ds.addROI('above horizon',{'elevation_best_choice':[10,90],'phi_best_choice':[-90,90],'similarity_count_h':[-0.1,10],'similarity_count_v':[-0.1,10],'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':[1.2,10000],'impulsivity_hSLICERADDimpulsivity_v':[0.3,100],'cr_template_search_hSLICERMAXcr_template_search_v':[0.4,100]})
    # ds.addROI('above horizon stripe',{'elevation_best_choice':[22,25],'phi_best_choice':[-43,-40],'similarity_count_h':[-0.1,10],'similarity_count_v':[-0.1,10],'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':[1.2,10000],'impulsivity_hSLICERADDimpulsivity_v':[0.3,100],'cr_template_search_hSLICERMAXcr_template_search_v':[0.4,100]})

    # elevation best choice         :   [10,90]
    # phi best choice               :   [-90,90]
    # similarity count h            :   [-0.1,10]
    # similarity count v            :   [-0.1,10]
    # max(hpol peak to sidelobe , vpol peak to sidelobe)  :   [1.2,10000]
    # impulsivity h + impulsivity v                       :   [0.3,100]
    # max(cr template search h + cr template search v)    :   [0.4,100]

    return_successive_cut_counts = True
    return_total_cut_counts = True

    if return_successive_cut_counts and return_total_cut_counts:
        above_horizon_eventids_dict, successive_cut_counts, total_cut_counts = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
    elif return_successive_cut_counts:
        above_horizon_eventids_dict, successive_cut_counts = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
    else:
        above_horizon_eventids_dict = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
    above_horizon_only_eventids_dict = ds.getCutsFromROI('above horizon only',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)

    for remove_box_az, remove_el in [ [[44.5,50], [-6,0]] , [[-11,-8], [-7,-3]] , [[-3,-1], [-12,0]] , [[24.5,28], [-7.75,-1]] , [[30.75,30], [-6,-1]] , [[6,8.5], [-12,-4]] ]:
        cluster_cut_dict = copy.deepcopy(ds.roi['above horizon'])
        cluster_cut_dict['phi_best_all_belowhorizon'] = remove_box_az
        cluster_cut_dict['elevation_best_all_belowhorizon'] = remove_el
        ds.addROI('below horizon cluster',cluster_cut_dict)
        remove_from_above_horizon_eventids_dict = ds.getCutsFromROI('below horizon cluster',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
        above_horizon_eventids_dict = ds.returnEventsAWithoutB(above_horizon_eventids_dict, remove_from_above_horizon_eventids_dict)

    
    above_horizon_eventids_array = ds.organizeEventDict(above_horizon_eventids_dict)
    remove_from_above_horizon_array = ds.organizeEventDict(remove_from_above_horizon_eventids_dict)

    # ds.plotROI2dHist('phi_best_choice','elevation_best_choice', cmap=cmap, eventids_dict=None, include_roi=False)
    ds.plotROI2dHist('phi_best_h','elevation_best_h', cmap=cmap, eventids_dict=None, include_roi=False)
    ds.plotROI2dHist('phi_best_v','elevation_best_v', cmap=cmap, eventids_dict=None, include_roi=False)
    
    if len(above_horizon_eventids_array) > 0 or False:
        plot_params = [['snr_h', 'snr_v'], ['phi_best_choice','elevation_best_choice'],['hpol_peak_to_sidelobe','vpol_peak_to_sidelobe'],['cr_template_search_h', 'cr_template_search_v'], ['impulsivity_h','impulsivity_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'],['hpol_max_map_value_abovehorizonSLICERDIVIDEhpol_max_possible_map_value','vpol_max_map_value_abovehorizonSLICERDIVIDEvpol_max_possible_map_value']]
        
        print('Generating plots:')

        for key_x, key_y in plot_params:
            print('Generating %s plot'%(key_x + ' vs ' + key_y))
            ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=above_horizon_eventids_dict, include_roi=False)

        ds.eventInspector(above_horizon_eventids_dict)
    if return_successive_cut_counts:
        roi_key = 'above horizon'
        for key in list(successive_cut_counts.keys()):
            if key == 'initial':
                print('Initial Event Count is %i'%(successive_cut_counts[key]))
            else:
                print('%0.3f%% events then cut by %s with bounds %s'%(100*(previous_count-successive_cut_counts[key])/previous_count , key, str(ds.roi[roi_key][key])))
                print('%0.3f%% of initial events would be cut by %s with bounds %s'%(100*(total_cut_counts['initial']-total_cut_counts[key])/total_cut_counts['initial'] , key, str(ds.roi[roi_key][key])))
                print('\nRemaining Events After Step %s is %i'%(key, successive_cut_counts[key]))
            previous_count = successive_cut_counts[key]

        print('')
        print('%0.3f%% events then cut by targeted below horizon box cuts'%(100*(previous_count-len(above_horizon_eventids_array))/previous_count))
        print('Double checking above math, %i events cut by targeted below horizon box cuts'%(len(remove_from_above_horizon_array)))
        print('Final number of events remaining is %i'%len(above_horizon_eventids_array))

    impulsivity_h = ds.getDataArrayFromParam('impulsivity_h', trigger_types=None, eventids_dict=above_horizon_eventids_dict)
    impulsivity_v = ds.getDataArrayFromParam('impulsivity_v', trigger_types=None, eventids_dict=above_horizon_eventids_dict)
    impulsivity = impulsivity_h + impulsivity_v
    most_impulsive_events = numpy.lib.recfunctions.rec_append_fields(above_horizon_eventids_array, ['impulsivity_h', 'impulsivity_v'], [impulsivity_h, impulsivity_v])[[numpy.argsort(impulsivity)[::-1]]]
