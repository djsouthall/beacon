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

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
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

    _runs = numpy.arange(5733,5974)
    # _runs = numpy.arange(5733,5800)

    for runs in [_runs]:

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
            plot_params =  [['similarity_count_h','similarity_count_v'], ['similarity_fraction_h','similarity_fraction_v'],['phi_best_h','event_rate_sigma_60.000Hz_20s']]
            print('Generating plots:')

            #plot_params = [['hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe']]
            for key_x, key_y in plot_params:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                ds.plotROI2dHist(key_x, key_y, cmap=cmap, include_roi=True)

        elif True:
            #This one cuts out ALL events
            # ds.addROI('above horizon only',{'elevation_best_h':[10,90]})
            # ds.addROI('above horizon',{'elevation_best_h':[10,90],'phi_best_h':[-90,90],'elevation_best_v':[10,90],'phi_best_v':[-90,90],'similarity_count_h':[0,10],'similarity_count_v':[0,10],'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,10],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]})
            ds.addROI('above horizon only',{'elevation_best_choice':[10,90],'phi_best_choice':[-90,90]})
            #ds.addROI('above horizon',{'elevation_best_choice':[10,90],'phi_best_choice':[-90,90],'similarity_count_h':[0,1],'similarity_count_v':[0,1],'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,10],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERMAXcr_template_search_v':[0.5,100]})#'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]
            ds.addROI('above horizon',{'elevation_best_choice':[10,90],'phi_best_choice':[-90,90],'similarity_count_h':[0,1],'similarity_count_v':[0,1],'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':[1.2,10000],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERMAXcr_template_search_v':[0.5,100],'min_snr_hSLICERADDmin_snr_v':[20,1000]})#'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]


            return_successive_cut_counts = False
            return_total_cut_counts = False
            if return_successive_cut_counts and return_total_cut_counts:
                above_horizon_eventids_dict, successive_cut_counts, total_cut_counts = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
            elif return_successive_cut_counts:
                above_horizon_eventids_dict, successive_cut_counts = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
            else:
                above_horizon_eventids_dict = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
            above_horizon_only_eventids_dict = ds.getCutsFromROI('above horizon only',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
            
            # ds.addROI('cluster',{'elevation_best_choice':[15,17.5],'phi_best_choice':[36.5,38.25],'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,10],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]})
            # cluster_dict = ds.getCutsFromROI('cluster',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
            # # ds.eventInspector(cluster_dict)


            if 5911 in list(above_horizon_eventids_dict.keys()):
                if ~numpy.isin(73399, above_horizon_eventids_dict[5911]):
                    print('Warning, the neat event isnt included in current cuts!')
                    ds.eventInspector({5911:numpy.array([73399])})

            #Perform temporal coincidence cut on all events that pass initial similarity and direction cuts.

            if False:
                above_horizon_eventids_array = ds.organizeEventDict(above_horizon_eventids_dict)
                time_window = 5*60
                print('Cutting away any events passing cuts with time coincidence < %i s'%time_window)
                time_data = ds.getDataArrayFromParam('calibrated_trigtime', trigger_types=None, eventids_dict=above_horizon_eventids_dict)
                pass_cut = ~numpy.logical_or(numpy.abs(time_data - numpy.roll(time_data,1)) <= time_window/2, numpy.abs(time_data - numpy.roll(time_data,-1)) <= time_window/2)

                above_horizon_eventids_array = above_horizon_eventids_array[pass_cut]

                if return_successive_cut_counts:
                    successive_cut_counts['time_cluster_cut'] = sum(pass_cut)
                    pprint(successive_cut_counts)
                for key in list(above_horizon_eventids_dict.keys()):
                    above_horizon_eventids_dict[key] = above_horizon_eventids_dict[key][numpy.isin(above_horizon_eventids_dict[key], above_horizon_eventids_array[above_horizon_eventids_array['run'] == key]['eventid'])]

                del time_data
                del pass_cut

            if 5911 in list(above_horizon_eventids_dict.keys()):
                if ~numpy.isin(73399, above_horizon_eventids_dict[5911]):
                    print('Warning, the neat event isnt included in current cuts!')
                    ds.eventInspector({5911:numpy.array([73399])})
            
            above_horizon_eventids_array = ds.organizeEventDict(above_horizon_eventids_dict)

            if len(above_horizon_eventids_array) > 0:
                # ds.plotROI2dHist('phi_best_all_belowhorizon','elevation_best_all_belowhorizon', cmap=cmap, eventids_dict=above_horizon_eventids_dict, include_roi=True)

                # plot_params = [['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v'],['phi_best_all','elevation_best_all'],['hpol_peak_to_sidelobe','vpol_peak_to_sidelobe'],['all_peak_to_sidelobe','elevation_best_all'],['cr_template_search_h', 'cr_template_search_v'], ['impulsivity_h','impulsivity_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v'] ,['similarity_count_h','similarity_count_v']]
                plot_params = [['min_snr_h','min_snr_v'],['snr_gap_h','snr_gap_v'], ['snr_h', 'snr_v'], ['phi_best_choice','elevation_best_choice'],['hpol_peak_to_sidelobe','vpol_peak_to_sidelobe'],['cr_template_search_h', 'cr_template_search_v'], ['impulsivity_h','impulsivity_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'],['hpol_max_map_value_abovehorizonSLICERDIVIDEhpol_max_possible_map_value','vpol_max_map_value_abovehorizonSLICERDIVIDEvpol_max_possible_map_value']]
                

                print('Generating plots:')

                # possible_polarizations = ['hpol', 'vpol', 'all'] #used to exclude best case scenario created here
                # possible_mapmax_cut_modes = ['abovehorizon','belowhorizon','allsky']

                # for key_x, key_y in [['phi_best_h_belowhorizon','elevation_best_h_belowhorizon'],['phi_best_h_abovehorizon','elevation_best_h_abovehorizon'],['phi_best_h_allsky','elevation_best_h_allsky']]:
                #     print('Generating %s plot'%(key_x + ' vs ' + key_y))
                #     ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None, include_roi=False)

                # for key_x, key_y in [['phi_best_choice','elevation_best_choice'],['phi_best_all','elevation_best_all'],['phi_best_h','elevation_best_h']]:
                #     print('Generating %s plot'%(key_x + ' vs ' + key_y))
                #     ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None, include_roi=False)

                # ds.plotROI2dHist('time_delay_0subtract1_h','time_delay_0subtract2_h', cmap=cmap, eventids_dict=cluster_dict, include_roi=False)

                ds.plotROI2dHist('phi_best_choice','elevation_best_choice', cmap=cmap, eventids_dict=None, include_roi=False)
                for key_x, key_y in plot_params:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    # ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None, include_roi=False)
                    # ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=above_horizon_only_eventids_dict, include_roi=False)
                    ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=above_horizon_eventids_dict, include_roi=False)



                # if len(above_horizon_eventids_array) < 100:
                #     print(above_horizon_eventids_array)
                #     ds.eventInspector(above_horizon_eventids_dict)
                # else:
                #     print('Number of remaining events above 100, printing them here:')
                #     pprint(above_horizon_eventids_array)


                common, common_eventids_array = ds.calculateTimeDelayCommonality(above_horizon_eventids_dict, verbose=True, return_eventids_array=True)
                common_run, common_run_index, common_eventid = common_eventids_array[numpy.argmax(common)]
                uncommon_run, uncommon_run_index, uncommon_eventid = common_eventids_array[numpy.argmin(common)]


                plt.figure()
                plt.hist(common, bins=200)

                ds.eventInspector({uncommon_run:numpy.array([uncommon_eventid])})

                reduced_events = common_eventids_array[common < 100]
                reduced_common = common[common < 100]
                # sorted_reduced_events = reduced_events[numpy.argsort(reduced_common)]
                reduced_eventid_dict = {}
                for run in numpy.unique(reduced_events['run']):
                    reduced_eventid_dict[run] = numpy.sort(reduced_events[reduced_events['run'] == run]['eventid'])
                ds.eventInspector(reduced_eventid_dict)

                ds.plotROI2dHist('phi_best_choice','elevation_best_choice', cmap=cmap, eventids_dict=above_horizon_eventids_dict, include_roi=False)
                ds.plotROI2dHist('phi_best_choice','elevation_best_choice', cmap=cmap, eventids_dict=reduced_eventid_dict, include_roi=False)



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
