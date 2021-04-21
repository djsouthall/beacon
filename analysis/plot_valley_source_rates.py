'''
This will plot the event rates of events in specified ROI using the data slicer tool.
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import inspect
import sys
import csv
import h5py

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info
from tools.data_slicer import dataSlicerSingleRun,dataSlicer
from tools.fftmath import FFTPrepper
from tools.correlator import Correlator
from tools.data_handler import createFile
import tools.get_plane_tracks as pt

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
plt.ion()

datapath = os.environ['BEACON_DATA']

if __name__=="__main__":
    plt.close('all')
    runs = numpy.arange(1650,1760)#numpy.array([1728,1773])#numpy.array([1650,1728,1773,1774,1783,1784])#numpy.arange(1650,2000)#numpy.array([1650,1728,1773,1774,1783,1784])#numpy.arange(1650,1675)#numpy.array([1774])#numpy.array([1650,1728,1773,1774,1783,1784])#numpy.array([1650])#

    include_sources = ['Northern Cell Tower','Tonopah KTPH','Nye County Sherriff','Tonopah AFS GATR Site','Miller Substation','Dyer Cell Tower','West Dyer Substation','East Dyer Substation','Beatty Mountain Cell Tower','Palmetto Cell Tower','Cedar Peak','Silver Peak Town Antenna'  ]

    # List of unique source dictionaries
    # 'Northern Cell Tower'       :{'time_delay_0subtract1_h':[-127,-123],'time_delay_0subtract2_h':[-127,-123.5]},\
    # 'Tonopah KTPH'              :{'time_delay_0subtract1_h':[-140.5,-137],'time_delay_0subtract2_h':[-90,-83.5],'time_delay_0subtract3_h':[-167,-161],'time_delay_1subtract2_h':[46,55]},\
    # 'Nye County Sherriff'       :{'time_delay_0subtract1_h':[-135,-131],'time_delay_0subtract2_h':[-111,-105]},\
    # 'Tonopah AFS GATR Site'     :{'time_delay_0subtract1_h':[-127,-123],'time_delay_0subtract2_h':[-127,-123.5]},\
    # 'Miller Substation'         :{'time_delay_0subtract1_h':[-135,-131],'time_delay_0subtract2_h':[-111,-105]},\
    # 'Dyer Cell Tower'           :{'time_delay_0subtract1_h':[-140.5,-137],'time_delay_0subtract2_h':[-90,-83.5],'time_delay_0subtract3_h':[-167,-161],'time_delay_1subtract2_h':[46,55]},\
    # 'West Dyer Substation'      :{'time_delay_0subtract1_h':[-143,-140],'time_delay_0subtract2_h':[-60.1,-57.4]},\
    # 'East Dyer Substation'      :{'time_delay_0subtract1_h':[-138,-131.7],'time_delay_0subtract2_h':[-7,-1]},\
    # 'Beatty Mountain Cell Tower':{'time_delay_0subtract1_h':[-124.5,-121],'time_delay_0subtract2_h':[22.5,28.5]},\
    # 'Palmetto Cell Tower'       :{'time_delay_0subtract1_h':[-138,-131.7],'time_delay_0subtract2_h':[-7,-1]},\
    # 'Cedar Peak'                :{'time_delay_0subtract1_h':[-143,-140],'time_delay_0subtract2_h':[-60.1,-57.4]},\
    # 'Silver Peak Town Antenna'  :{'time_delay_0subtract1_h':[-140.5,-137],'time_delay_0subtract2_h':[-90,-83.5],'time_delay_0subtract3_h':[-167,-161],'time_delay_1subtract2_h':[46,55]},\


    datapath = os.environ['BEACON_DATA']

    impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_belowhorizon'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'

    trigger_types = [2]#[2]
    plot_maps = True


    sources_ENU, data_slicer_cut_dict = info.loadValleySourcesENU() #Plot all potential sources
    try:
        if False:
            ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-140,max_time_delays_val=-90,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

            for source_key, cut_dict in data_slicer_cut_dict.items():
                if source_key in include_sources:
                    ds.addROI(source_key,cut_dict)

            ds.trackROICounts(roi_keys=None,time_bin_width_s=3600,plot_run_start_times=True)
            ds.trackROICounts(roi_keys=None,time_bin_width_s=3600,plot_run_start_times=False)


            # plot_param_pairs = [['time_delay_0subtract1_h','time_delay_0subtract2_h'],['time_delay_0subtract3_h','time_delay_1subtract2_h'],['time_delay_1subtract3_h','time_delay_2subtract3_h']]#[['time_delay_0subtract1_h','time_delay_0subtract2_h'],['time_delay_0subtract3_h','time_delay_1subtract2_h'],['time_delay_1subtract3_h','time_delay_2subtract3_h']]#[['phi_best_h','elevation_best_h'],['time_delay_0subtract1_h','time_delay_0subtract2_h']]#[['impulsivity_h','impulsivity_v']]#[['impulsivity_h','impulsivity_v'],['phi_best_h','phi_best_v'], ['elevation_best_h','elevation_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]
            
            # if True:
            #     for key_x, key_y in plot_param_pairs:
            #         fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=False)
            # else:
            #     for slicer in ds.data_slicers:
            #         for key_x, key_y in plot_param_pairs:
            #             fig, ax = slicer.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=False)



            runs_1 = numpy.arange(1720,1729)
            runs_2 = numpy.arange(1732,1743)
            runs_3 = numpy.arange(1743,1760)

        if False:

            ds_1 = dataSlicer(runs_1, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-140,max_time_delays_val=-90,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

            ds_2 = dataSlicer(runs_2, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-140,max_time_delays_val=-90,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

            ds_3 = dataSlicer(runs_3, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-140,max_time_delays_val=-90,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

            ds_2.data_slicers[0].checkDataAvailability(verbose=True)
            ds_2.data_slicers[-1].checkDataAvailability(verbose=True)
            ds_1.plotROI2dHist('time_delay_0subtract1_h', 'time_delay_0subtract2_h', cmap='coolwarm', include_roi=False)
            ds_2.plotROI2dHist('time_delay_0subtract1_h', 'time_delay_0subtract2_h', cmap='coolwarm', include_roi=False)
            ds_3.plotROI2dHist('time_delay_0subtract1_h', 'time_delay_0subtract2_h', cmap='coolwarm', include_roi=False)

        if False:
            ds_1 = dataSlicer(runs_1, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-90,max_time_delays_val=0,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

            ds_2 = dataSlicer(runs_2, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-90,max_time_delays_val=0,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

            ds_3 = dataSlicer(runs_3, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-90,max_time_delays_val=0,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

            ds_1.plotROI2dHist('time_delay_1subtract3_h', 'time_delay_2subtract3_h', cmap='coolwarm', include_roi=False)
            ds_2.plotROI2dHist('time_delay_1subtract3_h', 'time_delay_2subtract3_h', cmap='coolwarm', include_roi=False)
            ds_3.plotROI2dHist('time_delay_1subtract3_h', 'time_delay_2subtract3_h', cmap='coolwarm', include_roi=False)

        if True:
            runs = numpy.arange(1650,1700)#numpy.arange(1650,1651)#numpy.arange(1650,1700)#numpy.arange(1650,1700)
            runs = runs[runs != 1663]
            ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, remove_incomplete_runs=False,\
                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=2000,time_delays_n_bins_v=2000,min_time_delays_val=-200,max_time_delays_val=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

            ds.addROI('Simple Template H > 0.7',{'cr_template_search_h':[0.7,1.0]})
            ds.addROI('Simple Template V > 0.7',{'cr_template_search_v':[0.7,1.0]})# Adding 2 ROI in different rows and appending as below allows for "OR" instead of "AND"

            #Done for OR condition
            _eventids_dict_h = ds.getCutsFromROI('Simple Template H > 0.7',load=False,save=False)
            _eventids_dict_v = ds.getCutsFromROI('Simple Template V > 0.7',load=False,save=False)
            eventids_dict = {}
            for key in list(_eventids_dict_h.keys()):
                eventids_dict[key] = numpy.sort(numpy.unique(numpy.append(_eventids_dict_h[key],_eventids_dict_v[key])))
            ds.resetAllROI()

            ds.addROI('triple cluster A',{'time_delay_2subtract3_h':[-61,-59.4], 'time_delay_1subtract3_h':[-46,-44], 'snr_h':[15,24], 'snr_v':[14,20]})
            ds.addROI('triple cluster B1',{'time_delay_2subtract3_h':[-60.5,-59.1], 'time_delay_1subtract3_h':[-46.8,-46], 'impulsivity_h':[0.38,0.64], 'impulsivity_v':[0.55,0.75]})
            ds.addROI('triple cluster B2',{'time_delay_2subtract3_h':[-60.5,-59.1], 'time_delay_1subtract3_h':[-46.8,-46], 'snr_h':[20,27], 'snr_v':[11,15],'impulsivity_h':[0.4,0.6], 'impulsivity_v':[0.4,0.52]})
            ds.addROI('triple cluster C',{'time_delay_2subtract3_h':[-59.5,-58.2], 'time_delay_1subtract3_h':[-48,-47], 'snr_h':[12,24], 'snr_v':[9,15]})
            ds.addROI('between',{'time_delay_1subtract3_h':[0,60], 'time_delay_0subtract2_h':[-200,0]})
            ds.plotROI2dHist('cr_template_search_h', 'cr_template_search_v', cmap='coolwarm', include_roi=True)

            ds.data_slicers[0].plotROI2dHist('cr_template_search_h', 'cr_template_search_v', cmap='coolwarm', include_roi=True)
            ds.plotROI2dHist('cr_template_search_h', 'cr_template_search_v', cmap='coolwarm', include_roi=True, eventids_dict=eventids_dict)
            ds.plotROI2dHist('time_delay_0subtract2_h','time_delay_0subtract1_h', cmap='coolwarm', include_roi=True, eventids_dict=eventids_dict)
            ds.plotROI2dHist('time_delay_1subtract3_h','time_delay_2subtract3_h', cmap='coolwarm', include_roi=True, eventids_dict=eventids_dict)


            plot_param_pairs = [['impulsivity_h','impulsivity_v'],['phi_best_h','elevation_best_h']]#],['phi_best_h','elevation_best_h']]#, ['snr_h','snr_v']]#[['impulsivity_h','impulsivity_v'], ['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v']]#[['phi_best_h','phi_best_v'], ['elevation_best_h','elevation_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]#[['phi_best_h','theta_best_h'], ['phi_best_v','theta_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]#,  ['std_h', 'std_v'], ['cw_freq_Mhz','cw_dbish'],['impulsivity_h','impulsivity_v']]#[['impulsivity_h','impulsivity_v'], ['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v']]
            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)

    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


