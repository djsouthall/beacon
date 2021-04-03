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
from tools.data_slicer import dataSlicerSingleRun
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
    run = 1774 #want run with airplane.  This has 1774-178 in it. 

    datapath = os.environ['BEACON_DATA']

    impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    # map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'
    map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1'
    #THE CURRENT MAP DIRECT DSET SUCKS

    #map_direction_dset_key = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1'

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

    subset_cm = plt.cm.get_cmap('autumn', 10)
    subset_colors = subset_cm(numpy.linspace(0, 1, len(db_subset_plot_ranges)))[0:len(db_subset_plot_ranges)]

    try:
        run = int(run)

        reader = Reader(datapath,run)
        
        try:
            print(reader.status())
        except Exception as e:
            print('Status Tree not present.  Returning Error.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(1)
        
        filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
        
        known_planes, calibrated_trigtime, output_tracks = pt.getKnownPlaneTracks(ignore_planes=[])
        if run == 1774:
            eventids_airplane = known_planes['1774-178']['eventids'][:,1]
        else:
            eventids_airplane = []

        

        if filename is not None:
            with h5py.File(filename, 'r') as file:
                eventids = file['eventids'][...]

                trigger_type_cut = numpy.isin(file['trigger_type'][...], trigger_types)

                dsets = list(file.keys()) #Existing datasets

                print('Time delay keys')
                print(list(file['time_delays'].keys()))

                print('Impulsivity keys')
                print(list(file['impulsivity'].keys()))

                print('Map keys')
                print(list(file['map_direction'].keys()))

                file.close()
            
            
            ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

            if False:
                plot_param_pairs = [['phi_best_h','elevation_best_h'],['impulsivity_h','impulsivity_v']]
                for key_x, key_y in plot_param_pairs:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)

            if trigger_types == [2]:
                #This set of plots aims to compare high impulsivity events across a few plots.  
                ds.addROI('Impulsivity V > 0.5',{'impulsivity_v':[0.5,1.0]})#,'impulsivity_v':[0.5,1.0]
                ds.addROI('Impulsivity H > 0.5',{'impulsivity_h':[0.5,1.0]})#,'impulsivity_v':[0.5,1.0]
                _eventids = numpy.unique(numpy.append(ds.getCutsFromROI('Impulsivity H > 0.5',load=False,save=False),ds.getCutsFromROI('Impulsivity V > 0.5',load=False,save=False)))

                plot_param_pairs = [['phi_best_h','elevation_best_h'],['impulsivity_h','impulsivity_v']]

                for key_x, key_y in plot_param_pairs:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    if 'impulsivity' in key_x:
                        fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=False)
                        fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=_eventids, cmap='coolwarm', include_roi=False)
                    else:
                        fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=_eventids, cmap='coolwarm', include_roi=False)
                    
                    ds.addContour(ax, key_x, key_y, eventids_airplane, 'r', load=False, n_contour=5, alpha=0.85, log_contour=True)
                    
                    fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=eventids_airplane, cmap='coolwarm', include_roi=False)
            else:
                plot_param_pairs = [['impulsivity_h','impulsivity_v'],['phi_best_h','phi_best_v'], ['elevation_best_h','elevation_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]#[['phi_best_h','phi_best_v'], ['elevation_best_h','elevation_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]#[['phi_best_h','theta_best_h'], ['phi_best_v','theta_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]#,  ['std_h', 'std_v'], ['cw_freq_Mhz','cw_dbish'],['impulsivity_h','impulsivity_v']]#[['impulsivity_h','impulsivity_v'], ['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v']]
                for key_x, key_y in plot_param_pairs:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=False)

        else:
            print('filename is None, indicating empty tree.  Skipping run %i'%run)
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


