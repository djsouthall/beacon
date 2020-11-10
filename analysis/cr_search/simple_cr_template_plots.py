#!/usr/bin/env python3
'''
The purpose of this script is to generate plots based on the simple_cr_template_search.py results.

The documention for each function needs to be updated to represent the new functionality.  Additionally this should
all be moved to the tools folder and just be called here.  

'''
import sys
import os
import inspect
import h5py
import ast

import numpy
import scipy
import scipy.signal
import scipy.signal

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_handler import createFile
from tools.data_slicer import dataSlicerSingleRun
from tools.correlator import Correlator
from tools.fftmath import TemplateCompareTool


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.patches import Rectangle
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

datapath = os.environ['BEACON_DATA']


if __name__ == '__main__':
    
    plot_param_pairs = [['time_delay_0subtract1_h','time_delay_0subtract2_h']]#[['impulsivity_h','impulsivity_v'], ['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v'],['time_delay_0subtract1_h','time_delay_0subtract2_h']]
    
    #plt.close('all')
    #runs = [1642,1643,1644,1645,1646,1647]
    if len(sys.argv) == 2:
        runs = [int(sys.argv[1])]
    else:
        runs = [1650]

    crit_freq_low_pass_MHz = None#60 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = None#5

    crit_freq_high_pass_MHz = None#60
    high_pass_filter_order = None#6

    waveform_index_range = (None,None)#(150,400)
    plot_filter=False

    n_phi = 720
    n_theta = 720

    upsample = 2**13
    apply_phase_response = False
    hilbert = False
    
    max_method = 0
    

    for run in runs:
        reader = Reader(datapath,run)
        impulsivity_dset_key = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_262144-align_0'#'LPf_None-LPo_None-HPf_None-HPo_None-Phase_0-Hilb_0-corlen_262144-align_0'
        time_delays_dset_key = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_262144-align_0'

        print("Preparing dataSlicerSingleRun")
        # ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
        #                 cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
        #                 std_n_bins_h=300,std_n_bins_v=300,max_std_val=9,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
        #                 snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False)

        ds = dataSlicerSingleRun(   reader, impulsivity_dset_key, time_delays_dset_key, \
                                    curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                                    time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                                    p2p_n_bins_h=256,p2p_n_bins_v=256,max_p2p_val=128,\
                                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

        #cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, n_theta=n_theta, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter,apply_phase_response=apply_phase_response)
        tct = TemplateCompareTool(reader, final_corr_length=upsample, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False,apply_phase_response=apply_phase_response)


        print('Adding ROI to dataSlicerSingleRun')

        #ds.addROI('corr A',{'cr_template_search_h':[0.575,0.665],'cr_template_search_v':[0.385,0.46]})
        #ds.addROI('high v imp',{'impulsivity_h':[0.479,0.585],'impulsivity_h':[0.633,0.7]})
        #ds.addROI('small h.4 v.4 imp',{'impulsivity_h':[0.34,0.45],'impulsivity_h':[0.42,0.5]})

        #ds.addROI('imp cluster',{'impulsivity_h':[0.35,0.46],'impulsivity_v':[0.45,0.50]})

        # roi_key = 'imp cluster'
        # with h5py.File(ds.analysis_filename, 'r') as file:
        #     cut_dict = ast.literal_eval(file['ROI'][roi_key].attrs['dict'])
        #     print('cut_dict = ', cut_dict)
        #     included_antennas = file['ROI'][roi_key].attrs['included_antennas']
        #     print('included_antennas = ', included_antennas)
        #     cr_template_curve_choice = file['ROI'][roi_key].attrs['cr_template_curve_choice']
        #     print('cr_template_curve_choice = ', cr_template_curve_choice)
        #     trigger_types = file['ROI'][roi_key].attrs['trigger_types']
        #     print('trigger_types = ', trigger_types)
        #     file.close()

        print('Generating plots:')
        fig_dict = {}
        ax_dict = {}

        if False:
            #Getting clusters for Run 1650.  Should maybe do this for more runs stacked to have better statistics and better grouping.
            #But would need to develop the tool further to achieve this, and I want to do it this way for now, to understand what tools I need to build.
            #Percieved SNR Clusters
            test_ROI_index = 0
            ds.addROI('SNR ROI %i'%test_ROI_index,{'snr_h':[14,20.25],'snr_v':[21,27.5]}); test_ROI_index += 1
            ds.addROI('SNR ROI %i'%test_ROI_index,{'snr_h':[24.5,28.5],'snr_v':[15,18]}); test_ROI_index += 1
            ds.addROI('SNR ROI %i'%test_ROI_index,{'snr_h':[15,21],'snr_v':[10.5,15]}); test_ROI_index += 1
            ds.addROI('SNR ROI %i'%test_ROI_index,{'snr_h':[21,24],'snr_v':[12,17]}); test_ROI_index += 1

            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                # fig_dict[key_x + ' vs ' + key_y] = fig
                # ax_dict[key_x + ' vs ' + key_y] = ax
            ds.resetAllROI()

        if False:
            #Percieved Impulsivity Clusters
            test_ROI_index = 0
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.51,0.61],'impulsivity_v':[0.34,0.44]}); test_ROI_index += 1
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.45,0.54],'impulsivity_v':[0.20,0.25]}); test_ROI_index += 1
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.38,0.50],'impulsivity_v':[0.28,0.40]}); test_ROI_index += 1
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.36,0.42],'impulsivity_v':[0.47,0.52]}); test_ROI_index += 1
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.44,0.58],'impulsivity_v':[0.50,0.59]}); test_ROI_index += 1
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.50,0.56],'impulsivity_v':[0.64,0.68]}); test_ROI_index += 1 #partial zone with overlap
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.57,0.68],'impulsivity_v':[0.63,0.71]}); test_ROI_index += 1
            
            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                # fig_dict[key_x + ' vs ' + key_y] = fig
                # ax_dict[key_x + ' vs ' + key_y] = ax
            ds.resetAllROI()

        if False:
            #Percieved CR Template Clusters
            test_ROI_index = 0
            ds.addROI('CR ROI %i'%test_ROI_index,{'cr_template_search_h':[0.56,0.71],'cr_template_search_v':[0.31,0.39]}); test_ROI_index += 1
            ds.addROI('CR ROI %i'%test_ROI_index,{'cr_template_search_h':[0.59,0.68],'cr_template_search_v':[0.39,0.50]}); test_ROI_index += 1
            ds.addROI('CR ROI %i'%test_ROI_index,{'cr_template_search_h':[0.61,0.71],'cr_template_search_v':[0.895,0.93]}); test_ROI_index += 1 #partial zone with overlap
            ds.addROI('CR ROI %i'%test_ROI_index,{'cr_template_search_h':[0.50,0.625],'cr_template_search_v':[0.85,0.89]}); test_ROI_index += 1 #partial zone with overlap

            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                # fig_dict[key_x + ' vs ' + key_y] = fig
                # ax_dict[key_x + ' vs ' + key_y] = ax
            ds.resetAllROI()

        if False:
            #Percieved STD Clusters #There are too many STD groupings to easily group, and many of them are understanably
            #linear relationships between the 2.  Which is hard to use a box for.  
            # test_ROI_index = 0
            # ds.addROI('STD ROI %i'%test_ROI_index,{'std_h':[2.5,5.0],'std_v':[6.0,8.0]}); test_ROI_index += 1
            # ds.addROI('STD ROI %i'%test_ROI_index,{'std_h':[1.5,2.0],'std_v':[1.2,1.5]}); test_ROI_index += 1
            # ds.addROI('STD ROI %i'%test_ROI_index,{'std_h':[2.2,2.9],'std_v':[0.73,0.83]}); test_ROI_index += 1
            # ds.addROI('STD ROI %i'%test_ROI_index,{'std_h':[2.5,5.0],'std_v':[6.0,8.0]}); test_ROI_index += 1

            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                # fig_dict[key_x + ' vs ' + key_y] = fig
                # ax_dict[key_x + ' vs ' + key_y] = ax
            ds.resetAllROI()
            
        if False:
            #Percieved CR Template Clusters
            test_ROI_index = 0
            ds.addROI('p2p ROI %i'%test_ROI_index,{'p2p_h':[52,74],'p2p_v':[66,90]}); test_ROI_index += 1
            ds.addROI('p2p ROI %i'%test_ROI_index,{'p2p_h':[35,55],'p2p_v':[45,62]}); test_ROI_index += 1
            ds.addROI('p2p ROI %i'%test_ROI_index,{'p2p_h':[52,70],'p2p_v':[6,11]}); test_ROI_index += 1
            #There are many more but too hard to seperate.  
            
            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                # fig_dict[key_x + ' vs ' + key_y] = fig
                # ax_dict[key_x + ' vs ' + key_y] = ax
            ds.resetAllROI()

        if False:
            plt.close('all')
            #Percieved CR Template Clusters
            test_ROI_index = 0
            ds.addROI('cr_template_search ROI %i'%test_ROI_index,{'cr_template_search_h':[0.53,0.75],'cr_template_search_v':[0.29,0.39],'impulsivity_h':[0.425,0.63],'impulsivity_v':[0.49,0.6],'snr_h':[14,22],'snr_v':[10,15]}); test_ROI_index += 1
            #There are many more but too hard to seperate.  
            
            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                # fig_dict[key_x + ' vs ' + key_y] = fig
                # ax_dict[key_x + ' vs ' + key_y] = ax
            ds.resetAllROI()

        if True:
            #plt.close('all')

            #Clusters based on multiple parameters
            # ds.addROI('ROI 1650-1',{'cr_template_search_h':[0.61,0.71],'cr_template_search_v':[0.89,0.93],'impulsivity_h':[0.56,0.68],'impulsivity_v':[0.62,0.7],'std_h':[2.9,3.7],'std_v':[2.55,3.3],'p2p_h':[50,75],'p2p_v':[63.5,91],'snr_h':[16,21],'snr_v':[23.5,30]})
            # eventids_1 = ds.getCutsFromROI('ROI 1650-1',load=False,save=False)
            
            ds.addROI('ROI 1650-2',{'cr_template_search_h':[0.50,0.63],'cr_template_search_v':[0.845,0.895],'impulsivity_h':[0.5,0.64],'impulsivity_v':[0.575,0.65],'std_h':[2.4,3.2],'std_v':[1.95,2.55],'p2p_h':[35,55],'p2p_v':[45,65],'snr_h':[13,17.5],'snr_v':[21,26.3]})
            eventids_2 = ds.getCutsFromROI('ROI 1650-2',load=False,save=False)
            
            # ds.addROI('ROI 1650-3',{'cr_template_search_h':[0.53,0.75],'cr_template_search_v':[0.29,0.39],'impulsivity_h':[0.425,0.63],'impulsivity_v':[0.49,0.6],'snr_h':[14,22],'snr_v':[10,15]})
            # eventids_3 = ds.getCutsFromROI('ROI 1650-3',load=False,save=False)


            '''
            These 2 ROI are what I will develop further analysis procedure on.  What I want to know:
            How many events meet BOTH sets of ROI cuts.
            How many events in each cut.
            I want to save the cuts to the analysis file.
            I want to load in those cuts to get eventids and generate sample waveform of each time, and maps.
            '''
            plot_2d = False
            plot_template = False
            plot_time_delays = True

            #ds.printDatasets()
            if plot_time_delays:
                ds.plotTimeDelayHist(include_roi=True, load=False)
            
            if plot_2d == True:
                for key_x, key_y in plot_param_pairs:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)

            if plot_template == True:
                for roi_key in list(ds.roi.keys()):
                    eventids = ds.getCutsFromROI(roi_key,load=False,save=False)

                    times, averaged_waveforms = tct.averageAlignedSignalsPerChannel(eventids, template_eventid=eventids[-1], align_method=0, plot=False)
                    times_ns = times/1e9
                    freqs_MHz = numpy.fft.rfftfreq(len(times_ns),d=numpy.diff(times_ns)[0])/1e6
                    #Plot averaged FFT per channel
                    fft_fig = plt.figure()
                    fft_fig.canvas.set_window_title('FFT %s'%(roi_key))
                    for mode_index, mode in enumerate(['hpol','vpol']):
                        plt.subplot(2,1,1+mode_index)
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                        plt.ylabel('%s dBish'%mode)
                        plt.xlabel('MHz')
                        fft_ax = plt.gca()
                        for channel, averaged_waveform in enumerate(averaged_waveforms):
                            if mode == 'hpol' and channel%2 == 1:
                                continue
                            elif mode == 'vpol' and channel%2 == 0:
                                continue

                            freqs, spec_dbish, spec = tct.rfftWrapper(times, averaged_waveform)
                            fft_ax.plot(freqs/1e6,spec_dbish/2.0,label='Ch %i'%channel)#Dividing by 2 to match monutau.  Idk why I have to do this though normally this function has worked well...
                        plt.xlim(10,110)
                        plt.ylim(-20,30)

