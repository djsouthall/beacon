'''
Given 2 runs, and cuts on a specific cluster of events, this will compare the runs in those cuts for any apparent
shift in time delays.  This has been set up to compare eventids across runs, which inevitably results in pulling
events out of order, which can be much slower than sequentially.  In general this script was designed to be fast to
write, rather than fast to run.
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
    #plt.close('all')

    '''
    Both cut dicts should have the same high level ROI keys.  Event ids will be compared within groups of the same ROI key.
    Loops occur only over keys from cut_dict_A (but then that key is used in cut_dict_B.)
    '''

    choose_N = 10e3 #Max number of events to pull from each ROI
    runs_A = numpy.array([1726])#numpy.array([1726,1727])
    cut_dict_A = {'1'       :{'time_delay_2subtract3_h':[-56,-52],'time_delay_1subtract3_h':[-56,-53]}}

    runs_B = numpy.array([1749])#numpy.array([1749,1750])
    cut_dict_B = {'1'       :{'time_delay_2subtract3_h':[-57,-53],'time_delay_1subtract3_h':[-56,-53]}}

    datapath = os.environ['BEACON_DATA']

    impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_belowhorizon'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'

    trigger_types = [2]#[2]
    plot_maps = True

    channels = numpy.array([0,2,4,6],dtype=int) #The channels you want these calculations performed for.  

    hist_bin_edges = numpy.arange(-50,50+0.1,2)  #Used for looking and storing the data.  Represents time difference when cross correlating signals before and after shift.  Assumes centered around 0. 
    hist_bin_centers = (hist_bin_edges[:-1] + hist_bin_edges[1:]) / 2
    hist_counts = numpy.zeros((len(channels),len(hist_bin_centers)),dtype=int)

    try:
        ds_A = dataSlicer(runs_A, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-90,max_time_delays_val=0,\
                std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

        ds_B = dataSlicer(runs_B, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-90,max_time_delays_val=0,\
                std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

        t = ds_A.data_slicers[0].reader.t()
        dt = numpy.diff(t)[0]

        for key in list(cut_dict_A.keys()):
            ds_A.addROI(key,cut_dict_A[key])
            ds_B.addROI(key,cut_dict_B[key])

            eventids_dict_A = ds_A.getCutsFromROI(key,verbose=False)
            draw_ratio_A = numpy.array([len(eventids) for k, eventids in eventids_dict_A.items()]) #Used to determine what portion of events to draw from each Run based on number of events in each runs ROI/ 
            total_A = numpy.sum(draw_ratio_A)
            draw_per_run_A = numpy.floor(min(total_A, choose_N)*draw_ratio_A/numpy.sum(draw_ratio_A)).astype(int)

            eventids_dict_B = ds_B.getCutsFromROI(key,verbose=False)
            draw_ratio_B = numpy.array([len(eventids) for k, eventids in eventids_dict_B.items()]) #Used to determine what portion of events to draw from each Run based on number of events in each runs ROI/ 
            run_indices_B = numpy.arange(len(list(eventids_dict_B.keys()))) #To be randomly sampled from for each run


            for run_key_index, run_key in enumerate(list(eventids_dict_A.keys())):
                # Go through runs A sequentially, and pull from B randomly. 
                run_events = numpy.sort(numpy.random.choice(eventids_dict_A[run_key], size=draw_per_run_A[run_key_index], replace=False)) #Draw the appropriate number of events from run A
                run_indices_B = numpy.random.choice(numpy.arange(len(list(eventids_dict_B.keys()))),size=draw_per_run_A[run_key_index],replace=True,p=draw_ratio_B/numpy.sum(draw_ratio_B)) #The index of the run each eventid will be pulled from for set B
                for event_A_index, event_A in enumerate(run_events):
                    event_B = numpy.random.choice(eventids_dict_B[list(eventids_dict_B.keys())[run_indices_B[event_A_index]]],size=1)[0] #Randomly choosing event B from the appropriate run. 

                    ds_A.data_slicers[run_key_index].reader.setEntry(event_A)
                    ds_B.data_slicers[run_key_index].reader.setEntry(event_B)
                    
                    for channel_index, channel in enumerate(channels):
                        wf_A = ds_A.data_slicers[run_key_index].reader.wf(int(channel))
                        wf_B = ds_B.data_slicers[run_key_index].reader.wf(int(channel))
                        cc = scipy.signal.correlate(wf_A, wf_B)
                        td = dt*(numpy.argmax(cc) - len(t) + 1) #May be off or inverted in direction, but magnitude should be good for simple check. 
                        hist_counts[channel_index] += numpy.histogram(td,bins=hist_bin_edges)[0]

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t,wf_A)
        plt.plot(t,wf_B)
        plt.subplot(2,1,2)
        plt.plot(dt*(numpy.arange(len(cc)) - len(t) + 1),cc)


        for channel_index, channel in enumerate(channels):
            fig = plt.figure()
            plt.title('Channel %i'%channel)
            plt.hist(hist_bin_centers, bins = hist_bin_edges, weights=hist_counts[channel_index])
            plt.xlabel('Time Offset (ns)')
            plt.ylabel('Counts')

    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


