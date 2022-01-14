#!/usr/bin/env python3
'''
The purpose of this script is to look at the results from various maps and find a way to determine how to choose the
best map for reconstruction.
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
processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_jan12_2022')
# processed_datapath = os.environ['BEACON_PROCESSED_DATA']
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
    map_length = 16384#32768
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length
    #numpy.arange(5732,5790)
    #
    #numpy.arange(5910,5920)
    #numpy.arange(5733,5974)
    #numpy.arange(5911,5912)
    # _runs = numpy.arange(5733,5974)
    _runs = numpy.arange(5733,5833)
    #_runs = numpy.array([5733,5734])
    # _runs = [5740]

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

        
        ds.addROI('above horizon',{'elevation_best_p2p':[10,90],'phi_best_p2p':[-90,90],'similarity_count_h':[0,0.5],'similarity_count_v':[0,0.5],'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,10],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]})
        eventids_dict = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
        
        ds.resetAllROI()
        ds.addROI('cluster',{'elevation_best_p2p':[12.5,17.5],'phi_best_p2p':[35,40],'similarity_count_h':[0,0.5],'similarity_count_v':[0,0.5],'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,10],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]})
        cluster_dict = ds.getCutsFromROI('cluster',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
        #ds.eventInspector(cluster_dict)

        #plot_params = ['phi_best_allSLICERSUBTRACTphi_best_h','phi_best_allSLICERSUBTRACTphi_best_v'],['elevation_best_allSLICERSUBTRACTelevation_best_h','elevation_best_allSLICERSUBTRACTelevation_best_v']
        #[['phi_best_all','elevation_best_all'],['hpol_peak_to_sidelobe','vpol_peak_to_sidelobe'],['cr_template_search_h', 'cr_template_search_v'], ['impulsivity_h','impulsivity_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v'] ,['similarity_count_h','similarity_count_v']]
        
        print('Generating plots:')

        #Perhaps some other cut can get rid of these rather than a best map calculation or a multipeaks?
        
        ds.plotROI2dHist('hpol_max_map_value_abovehorizonSLICERDIVIDEhpol_max_map_value_belowhorizon','vpol_max_map_value_abovehorizonSLICERDIVIDEvpol_max_map_value_belowhorizon', cmap=cmap, eventids_dict=None, include_roi=False)
        ds.plotROI2dHist('hpol_max_map_value_abovehorizonSLICERDIVIDEhpol_max_map_value_belowhorizon','vpol_max_map_value_abovehorizonSLICERDIVIDEvpol_max_map_value_belowhorizon', cmap=cmap, eventids_dict=eventids_dict, include_roi=True)


        ds.plotROI2dHist('phi_best_h','elevation_best_h', cmap=cmap, eventids_dict=None, include_roi=False)
        ds.plotROI2dHist('phi_best_all','elevation_best_all', cmap=cmap, eventids_dict=None, include_roi=False)
        ds.plotROI2dHist('phi_best_p2p','elevation_best_p2p', cmap=cmap, eventids_dict=None, include_roi=False)
        ds.plotROI2dHist('phi_best_p2p','elevation_best_p2p', cmap=cmap, eventids_dict=eventids_dict, include_roi=False)

        # for key_x, key_y in plot_params:
        #     print('Generating %s plot'%(key_x + ' vs ' + key_y))
        #     ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=eventids_dict, include_roi=False)

        # # ds.addROI()
        # ds.plotROI2dHist('hpol_peak_to_sidelobe','vpol_peak_to_sidelobe', cmap=cmap, eventids_dict=eventids_dict, include_roi=False)
        # ds.eventInspector(eventids_dict)