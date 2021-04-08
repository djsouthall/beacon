'''
This is a script load waveforms using the sine subtraction method, and save any identified CW present in events.
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
    #TODO: Add these parameters to the 2d data slicer.
    plt.close('all')
    runs = numpy.arange(1650,1800)#numpy.array([1650,1728,1773,1774,1783,1784])#numpy.arange(1650,1675)#numpy.array([1774])#numpy.array([1650,1728,1773,1774,1783,1784])#numpy.array([1650])#
    #run = 1774 #want run with airplane.  This has 1774-178 in it. 

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
    map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'

    crit_freq_low_pass_MHz = 100 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = 8

    crit_freq_high_pass_MHz = None
    high_pass_filter_order = None

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.03
    sine_subtract_max_freq_GHz = 0.09
    sine_subtract_percent = 0.03

    hilbert=False
    final_corr_length = 2**10

    trigger_types = [2]#[2]
    db_subset_plot_ranges = [[0,30],[30,40],[40,50]] #Used as bin edges.  
    plot_maps = True

    sum_events = True #If true will add all plots together, if False will loop over runs in runs.
    lognorm = True
    cmap = 'YlOrRd'#'binary'#'coolwarm'
    subset_cm = plt.cm.get_cmap('autumn', 10)
    subset_colors = subset_cm(numpy.linspace(0, 1, len(db_subset_plot_ranges)))[0:len(db_subset_plot_ranges)]

    sources_ENU, data_slicer_cut_dict = info.loadValleySourcesENU() #Plot all potential sources
    try:
        ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)


        # ds.addROI('Simple Template V > 0.7',{'cr_template_search_v':[0.7,1.0]})# Adding 2 ROI in different rows and appending as below allows for "OR" instead of "AND"
        # ds.addROI('Simple Template H > 0.7',{'cr_template_search_h':[0.7,1.0]})
        # #Done for OR condition
        # _eventids = numpy.sort(numpy.unique(numpy.append(ds.getCutsFromROI('Simple Template H > 0.7',load=False,save=False),ds.getCutsFromROI('Simple Template V > 0.7',load=False,save=False))))

        for source_key, cut_dict in data_slicer_cut_dict.items():
            if source_key in include_sources:
                ds.addROI(source_key,cut_dict)

        ds.trackROICounts()


    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


