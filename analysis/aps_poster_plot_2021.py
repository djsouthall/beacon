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
from tools.fftmath import TimeDelayCalculator
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
    runs = numpy.array([1650])#numpy.arange(1650,1750)#numpy.array([1650,1728,1773,1774,1783,1784])#numpy.arange(1650,1675)#numpy.array([1774])#numpy.array([1650,1728,1773,1774,1783,1784])#numpy.array([1650])#
    #run = 1774 #want run with airplane.  This has 1774-178 in it. 

    datapath = os.environ['BEACON_DATA']

    impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    #map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'
    map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_belowhorizon'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_belowhorizon' # 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_allsky'
    
    '''
    'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_abovehorizon'
    'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_allsky'
    'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_belowhorizon'
    'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1'
    'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_abovehorizon'
    'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_allsky'
    'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_belowhorizon'
    'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0'
    '''

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

    sum_events = False #If true will add all plots together, if False will loop over runs in runs.
    lognorm = True
    cmap = 'YlOrRd'#'binary'#'coolwarm'
    subset_cm = plt.cm.get_cmap('autumn', 10)
    subset_colors = subset_cm(numpy.linspace(0, 1, len(db_subset_plot_ranges)))[0:len(db_subset_plot_ranges)]

    try:
        if False:
            if len(runs) == 1 or sum_events == False:
                for run in runs:
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
                    
                    known_planes, calibrated_trigtime, output_tracks = pt.getKnownPlaneTracks(ignore_planes=[])
                    if run == 1774:
                        eventids_airplane = []#known_planes['1774-178']['eventids'][:,1]
                    else:
                        eventids_airplane = []


                    #This set of plots aims to compare high impulsivity events across a few plots.  
                    if False:
                        ds.addROI('Impulsivity V > 0.5',{'impulsivity_v':[0.5,1.0]})#,'impulsivity_v':[0.5,1.0]
                        ds.addROI('Impulsivity H > 0.5',{'impulsivity_h':[0.5,1.0]})#,'impulsivity_v':[0.5,1.0]
                        _eventids = numpy.unique(numpy.append(ds.getCutsFromROI('Impulsivity H > 0.5',load=False,save=False),ds.getCutsFromROI('Impulsivity V > 0.5',load=False,save=False)))
                        ds.resetAllROI() #already have the eventids with sufficient impulsivity
                    else:
                        _eventids = None
                    #ds.addROI('Large Above Horizon',{'phi_best_h':[24,25.5],'elevation_best_h':[6.2,6.8]})
                    #ds.getCutsFromROI('Impulsivity H > 0.5',load=False,save=False)
                    

                    if run == 1650:
                        if False:
                            ds.addROI('-19',{'phi_best_h':[-21,-18],'elevation_best_h':[-0.5,0.5]})
                            ds.addROI('-10h',{'phi_best_h':[-11,-9],'elevation_best_h':[-1,0.5]})
                            ds.addROI('-10l',{'phi_best_h':[-11,-9],'elevation_best_h':[-6,-4]})
                            ds.addROI('-2',{'phi_best_h':[-3,-1],'elevation_best_h':[-3.5,-1.5]})
                            ds.addROI('8',{'phi_best_h':[7,9],'elevation_best_h':[-2,0]})
                            ds.addROI('19',{'phi_best_h':[18,20],'elevation_best_h':[-3,-1.5]})
                            ds.addROI('28h',{'phi_best_h':[27,29],'elevation_best_h':[-3,-1.5]})
                            ds.addROI('28l',{'phi_best_h':[27,29],'elevation_best_h':[-4,-3]})
                            ds.addROI('32',{'phi_best_h':[31,33],'elevation_best_h':[-2,0]})
                            plot_param_pairs = [['impulsivity_h','impulsivity_v'],['phi_best_h','phi_best_v'], ['elevation_best_h','elevation_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]
                        else:
                            plot_param_pairs = [['phi_best_h','elevation_best_h']]
                    else:
                        plot_param_pairs = [['phi_best_h','elevation_best_h']]



                    for key_x, key_y in plot_param_pairs:
                        print('Generating %s plot'%(key_x + ' vs ' + key_y))
                        # if 'impulsivity' in key_x:
                        #     fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, include_roi=True, lognorm=lognorm)
                        #     fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=_eventids, cmap=cmap, include_roi=False, lognorm=lognorm)
                        # else:
                        #     fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=_eventids, cmap=cmap, include_roi=True, lognorm=lognorm)
                        fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=_eventids, cmap=cmap, include_roi=True, lognorm=lognorm)
                        
                        if len(eventids_airplane) > 0:
                            ds.addContour(ax, key_x, key_y, eventids_airplane, 'r', load=False, n_contour=5, alpha=0.85, log_contour=True)
                            fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=eventids_airplane, cmap=cmap, include_roi=False, lognorm=lognorm)
                        #fig.savefig(key_x + key_y + '.svg', bbox_inches=0, transparent=False)
                        fig.savefig('run%i_'%run + key_x + key_y + '.svg', bbox_inches=0, transparent=False)
                        import pdb; pdb.set_trace()
            else:
                ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                        curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                        impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                        p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

                plot_param_pairs = [['phi_best_h','elevation_best_h'],['impulsivity_h','impulsivity_v']]
                
                for key_x, key_y in plot_param_pairs:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, include_roi=False, lognorm=lognorm)
        if True:
            run = 1507
            reader = Reader(datapath,run)
            apply_phase_response = True
            tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,plot_filters=False,apply_phase_response=apply_phase_response)#,waveform_index_range=
            known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
            eventid = known_pulser_ids['run1507']['hpol'][0]
            #tdc.plotEvent(eventid, channels=[0,2,4,6], apply_filter=False, hilbert=False, sine_subtract=False, apply_tukey=None, additional_title_text=None, time_delays=None, verbose=False)
            channels=[0,2,4,6]
            apply_filter=False
            hilbert=False
            sine_subtract=False
            apply_tukey=None
            additional_title_text=None
            time_delays=None
            verbose=False
            fig = plt.figure()
            ax = plt.gca()
            plt.ylabel('Normalized Pulser Signals',fontsize=22)
            plt.xlabel('Time (ns)',fontsize=22)
            # plt.minorticks_on()
            # plt.grid(b=True, which='major', color='k', linestyle='-')
            # plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            tdc.setEntry(eventid)
            t_ns = tdc.t()
            if verbose:
                print(eventid)
            if apply_tukey is None:
                apply_tukey = tdc.tukey_default
            
            max_wf_mag = 0

            for channel_index, channel in enumerate(channels):
                channel=int(channel)
                if sine_subtract == True:
                    wf, ss_freqs, n_fits = tdc.wf(channel,apply_filter=apply_filter,hilbert=hilbert,tukey=apply_tukey,sine_subtract=sine_subtract, return_sine_subtract_info=sine_subtract)
                    if verbose:
                        print(list(zip(n_fits, ss_freqs)))
                else:
                    wf = tdc.wf(channel,apply_filter=apply_filter,hilbert=hilbert,tukey=apply_tukey,sine_subtract=sine_subtract, return_sine_subtract_info=sine_subtract)

                if max(numpy.abs(wf)) > max_wf_mag:
                    max_wf_mag = max(numpy.abs(wf))

            for channel_index, channel in enumerate(channels):
                channel=int(channel)
                if sine_subtract == True:
                    wf, ss_freqs, n_fits = tdc.wf(channel,apply_filter=apply_filter,hilbert=hilbert,tukey=apply_tukey,sine_subtract=sine_subtract, return_sine_subtract_info=sine_subtract)
                    if verbose:
                        print(list(zip(n_fits, ss_freqs)))
                else:
                    wf = tdc.wf(channel,apply_filter=apply_filter,hilbert=hilbert,tukey=apply_tukey,sine_subtract=sine_subtract, return_sine_subtract_info=sine_subtract)

                plt.plot(t_ns - 3100,wf/(2*max_wf_mag) + channel_index, label='HPol Channel %i'%channel_index)
            plt.legend(fontsize=16,loc='center right')
            plt.xlim(0,600)
            plt.yticks([0,1,2,3])
            ax.set_ylim(ax.get_ylim()[::-1])
            fig.savefig('pulser_waveforms.svg', bbox_inches=0, transparent=True)

        if False:
            run = 1773
            #[1774,178],[1774,381],[1774,1348],[1774,1485][178,381,1348,1485]
            center_dir = 'W'
            apply_phase_response = True
            reader = Reader(datapath,run)
            cor = Correlator(reader,  upsample=2**16, n_phi=720, n_theta=720, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True)
            for eventid in [14413]:        
                mean_corr_values, fig, ax = cor.map(eventid, 'hpol', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, plot_corr=False, hilbert=False, interactive=False, max_method=None, waveforms=None, verbose=True, mollweide=True, center_dir=center_dir, circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False)
            #fig.savefig('1773-14413-map.svg', bbox_inches=0, transparent=True)



    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


