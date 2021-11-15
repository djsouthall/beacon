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
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
processed_datapath = os.environ['BEACON_PROCESSED_DATA']

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
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384#32768
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length
    #numpy.arange(5732,5790)
    #
    
    for runs in [numpy.arange(5733,5974)]:

        print("Preparing dataSlicer")

        map_resolution_theta = 0.25 #degrees
        min_theta   = 0
        max_theta   = 120
        n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

        map_resolution_phi = 0.1 #degrees
        min_phi     = -180
        max_phi     = 180
        n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)



        ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                        n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)

        #ds.addROI('above horizon',{'elevation_best_h':[0,90],'elevation_best_v':[0,90], 'hpol_peak_to_sidelobe':[2.0,300], 'vpol_peak_to_sidelobe':[2.0,300]})#'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.0,300]})
        



#This one cuts out ALL events
#ds.addROI('above horizon',{'elevation_best_h':[5,90], 'hpol_peak_to_sidelobe_abovehorizonSLICERADDvpol_peak_to_sidelobe_abovehorizon':[3,300], 'hpol_peak_to_sidelobe_abovehorizon':[1.5,300], 'impulsivity_hSLICERADDimpulsivity_v':[0.4,300]})
ds.addROI('above horizon only',{'elevation_best_h':[10,90]})
#ds.addROI('above horizon',{'elevation_best_h':[10,90],'phi_best_h':[-90,90],'hpol_max_map_value_abovehorizon':[0.7,10],'vpol_max_map_value_abovehorizon':[0.7,10],'hpol_peak_to_sidelobe_abovehorizonSLICERADDhpol_peak_to_sidelobe_abovehorizon':[2.5,300]})#,'hpol_peak_to_sidelobe_abovehorizonSLICERSUBTRACThpol_peak_to_sidelobe_belowhorizon':[0,1e6], 'impulsivity_hSLICERADDimpulsivity_v':[1.2,1e6]})#, 'hpol_peak_to_sidelobe_abovehorizonSLICERADDvpol_peak_to_sidelobe_abovehorizon':[3,300], 'hpol_peak_to_sidelobe_abovehorizon':[1.5,300], 'impulsivity_hSLICERADDimpulsivity_v':[0.4,300]})
#ds.addROI('above horizon',{'elevation_best_h':[10,90],'phi_best_h':[-90,90]})#,'hpol_peak_to_sidelobe_abovehorizonSLICERSUBTRACThpol_peak_to_sidelobe_belowhorizon':[0,1e6], 'impulsivity_hSLICERADDimpulsivity_v':[1.2,1e6]})#, 'hpol_peak_to_sidelobe_abovehorizonSLICERADDvpol_peak_to_sidelobe_abovehorizon':[3,300], 'hpol_peak_to_sidelobe_abovehorizon':[1.5,300], 'impulsivity_hSLICERADDimpulsivity_v':[0.4,300]})
#, 'hpol_peak_to_sidelobe_abovehorizonSLICERADDvpol_peak_to_sidelobe_abovehorizon':[1.5,300]
ds.addROI('above horizon',{'hpol_max_map_value_abovehorizonSLICERSUBTRACThpol_max_map_value_belowhorizon':[0.25,1000],'elevation_best_h':[10,90],'phi_best_h':[-90,90]})

above_horizon_eventids_only_dict = ds.getCutsFromROI('above horizon only',load=False,save=False,verbose=False)
above_horizon_eventids_only_array = ds.organizeEventDict(above_horizon_eventids_only_dict)

above_horizon_eventids_dict = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False)
# print('Hard bipassing the horizon dict')
# above_horizon_eventids_dict = {5911:numpy.array([73399])}


above_horizon_eventids_array = ds.organizeEventDict(above_horizon_eventids_dict)
if False:
    time_window = 5*60
    print('Cutting away any events passing cuts with time coincidence < %i s'%time_window)
    time_data = ds.getDataArrayFromParam('calibrated_trigtime', trigger_types=None, eventids_dict=above_horizon_eventids_dict)
    pass_cut = ~numpy.logical_or(numpy.abs(time_data - numpy.roll(time_data,1)) <= time_window/2, numpy.abs(time_data - numpy.roll(time_data,-1)) <= time_window/2)

    above_horizon_eventids_array = above_horizon_eventids_array[pass_cut]
    for key in list(above_horizon_eventids_dict.keys()):
        above_horizon_eventids_dict[key] = above_horizon_eventids_dict[key][numpy.isin(above_horizon_eventids_dict[key], above_horizon_eventids_array[above_horizon_eventids_array['run'] == key]['eventid'])]


    

ds.resetAllROI()
#ds.addROI('test',{'hpol_peak_to_sidelobe_abovehorizonSLICERADDvpol_peak_to_sidelobe_abovehorizon':[2.0,300]})
# roi_eventid_dict = ds.getCutsFromROI('test',load=False,save=False,verbose=False)
# roi_eventid_dict = ds.returnCommonEvents(above_horizon_eventids_dict, roi_eventid_dict)
#test_data = ds.getDataArrayFromParam('hpol_peak_to_sidelobe_abovehorizonSLICERADDvpol_peak_to_sidelobe_abovehorizon', trigger_types=None, eventids_dict=roi_eventid_dict)
# ds.addROI('baseline_0subtract1',{'map_max_time_delay_0subtract1_h_belowhorizonSLICERSUBTRACTtime_delay_0subtract1_h':[-5,5]})
# ds.addROI('baseline_0subtract2',{'map_max_time_delay_0subtract2_h_belowhorizonSLICERSUBTRACTtime_delay_0subtract2_h':[-5,5]})
# ds.addROI('baseline_0subtract3',{'map_max_time_delay_0subtract3_h_belowhorizonSLICERSUBTRACTtime_delay_0subtract3_h':[-5,5]})
# ds.addROI('baseline_1subtract2',{'map_max_time_delay_1subtract2_h_belowhorizonSLICERSUBTRACTtime_delay_1subtract2_h':[-5,5]})
# ds.addROI('baseline_1subtract3',{'map_max_time_delay_1subtract3_h_belowhorizonSLICERSUBTRACTtime_delay_1subtract3_h':[-5,5]})
# ds.addROI('baseline_2subtract3',{'map_max_time_delay_2subtract3_h_belowhorizonSLICERSUBTRACTtime_delay_2subtract3_h':[-5,5]})

#ds.addROI('test',{'map_max_time_delay_0subtract1_h_belowhorizonSLICERSUBTRACTtime_delay_0subtract1_h':[-5,5],'map_max_time_delay_0subtract2_h_belowhorizonSLICERSUBTRACTtime_delay_0subtract2_h':[-5,5],'map_max_time_delay_0subtract3_h_belowhorizonSLICERSUBTRACTtime_delay_0subtract3_h':[-5,5],'map_max_time_delay_1subtract2_h_belowhorizonSLICERSUBTRACTtime_delay_1subtract2_h':[-5,5],'map_max_time_delay_1subtract3_h_belowhorizonSLICERSUBTRACTtime_delay_1subtract3_h':[-5,5],'map_max_time_delay_2subtract3_h_belowhorizonSLICERSUBTRACTtime_delay_2subtract3_h':[-5,5]})
# ds.addROI('p2s',{'elevation_best_h':[0,90],'elevation_best_v':[0,90], 'hpol_peak_to_sidelobe':[1.5,10]})
#ds.addROI('Cluster 1',{'phi_best_h':[33.2,35.7],'elevation_best_h':[7.3,9.4]})
# roi_eventid_dict = ds.getCutsFromROI('p2s',load=False,save=False,verbose=False)
#ds.addROI('template_match',{'cr_template_search_h':[0.8,1.0],'cr_template_search_v':[0.8,1.0]})
#ds.addROI('intersting events',{'elevation_best_h':[10,90],'elevation_best_v':[10,90],'cr_template_search_h':[0.6,1.0],'cr_template_search_v':[0.6,1.0]})
#roi_eventid_dict = ds.getCutsFromROI('template_match',load=False,save=False,verbose=False)

#interesting_eventids_dict = ds.getCutsFromROI('intersting events',load=False,save=False,verbose=False)
#plot_params = [['hpol_peak_to_sidelobe','vpol_peak_to_sidelobe'],['hpol_peak_to_sidelobe','elevation_best_h'],['impulsivity_h','impulsivity_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v'],['p2p_h', 'p2p_v'],['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v']]
#plot_params = [['phi_best_h','elevation_best_h'], ['hpol_peak_to_sidelobe_abovehorizon', 'vpol_peak_to_sidelobe_abovehorizon'],['hpol_peak_to_sidelobe_belowhorizon','hpol_peak_to_sidelobe_abovehorizon'], ['map_max_time_delay_0subtract1_h_abovehorizon', 'map_max_time_delay_0subtract1_h_belowhorizon'], ['map_max_time_delay_0subtract1_h_belowhorizonSLICERSUBTRACTtime_delay_0subtract1_h', 'map_max_time_delay_0subtract2_h_belowhorizonSLICERSUBTRACTtime_delay_0subtract2_h'], ['time_delay_0subtract1_h', 'map_max_time_delay_0subtract1_h_belowhorizon'], ['time_delay_0subtract1_h', 'map_max_time_delay_0subtract1_h_belowhorizon'], ['time_delay_0subtract1_h', 'map_max_time_delay_0subtract1_h_abovehorizon'], ['impulsivity_h','impulsivity_v']]
#['hpol_max_map_value_belowhorizon','elevation_best_h']
if False:
    ds.eventInspector(above_horizon_eventids_dict)
else:
    #plot_params = [['phi_best_h','elevation_best_h'],['hpol_max_map_value_abovehorizon','vpol_max_map_value_abovehorizon'], ['hpol_max_map_value_belowhorizon','hpol_max_map_value_abovehorizon'],['vpol_max_map_value_belowhorizon','vpol_max_map_value_abovehorizon'], ['hpol_max_map_value','vpol_max_map_value'], ['hpol_peak_to_sidelobe_abovehorizon', 'vpol_peak_to_sidelobe_abovehorizon'],['hpol_peak_to_sidelobe_belowhorizon','hpol_peak_to_sidelobe_abovehorizon'], ['impulsivity_h','impulsivity_v'], ['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v'], ['hpol_max_possible_map_value','vpol_max_possible_map_value'], ['hpol_peak_to_sidelobe_abovehorizonSLICERDIVIDEhpol_peak_to_sidelobe_belowhorizon', 'vpol_peak_to_sidelobe_abovehorizonSLICERDIVIDEvpol_peak_to_sidelobe_belowhorizon'],['hpol_peak_to_sidelobe_abovehorizonSLICERSUBTRACThpol_peak_to_sidelobe_belowhorizon', 'vpol_peak_to_sidelobe_abovehorizonSLICERSUBTRACTvpol_peak_to_sidelobe_belowhorizon']]
    #plot_params = [['phi_best_h','elevation_best_h'],['hpol_max_map_value_abovehorizon','vpol_max_map_value_abovehorizon'], ['hpol_max_map_value_abovehorizonSLICERDIVIDEhpol_max_possible_map_value','vpol_max_map_value_abovehorizonSLICERDIVIDEvpol_max_possible_map_value'], ['hpol_max_map_value','vpol_max_map_value'], ['hpol_peak_to_sidelobe_abovehorizon', 'vpol_peak_to_sidelobe_abovehorizon'],['hpol_peak_to_sidelobe_belowhorizon','hpol_peak_to_sidelobe_abovehorizon'], ['impulsivity_h','impulsivity_v'], ['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v'], ['hpol_max_possible_map_value','vpol_max_possible_map_value']]
    plot_params = [['phi_best_h','elevation_best_h'],['hpol_max_map_value_belowhorizon','elevation_best_h']]
    plot_params =  [['hpol_max_map_value_belowhorizon','hpol_max_map_value_abovehorizon']]
    print('Generating plots:')

    #plot_params = [['hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe']]
    for key_x, key_y in plot_params:
        print('Generating %s plot'%(key_x + ' vs ' + key_y))
        ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
        #ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', eventids_dict=above_horizon_eventids_only_dict, include_roi=True)
        ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', eventids_dict=above_horizon_eventids_dict, include_roi=True)

    if len(above_horizon_eventids_array) < 100:
        ds.eventInspector(above_horizon_eventids_dict)



