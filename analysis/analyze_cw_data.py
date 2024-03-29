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
from analysis.background_identify_60hz import get60HzEvents

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
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 1650

    datapath = os.environ['BEACON_DATA']

    impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    #map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'
    map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_belowhorizon'
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
    
    runs = [1728,1773,1774,1783,1784] #Airplane runs 

    for run in runs:

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
            

            

            if filename is not None:
                with h5py.File(filename, 'r') as file:
                    eventids = file['eventids'][...]
                    eventids_60Hz = get60HzEvents(file)

                    trigger_type_cut = numpy.isin(file['trigger_type'][...], trigger_types)

                    dsets = list(file.keys()) #Existing datasets

                    print('Time delay keys')
                    print(list(file['time_delays'].keys()))

                    print('Impulsivity keys')
                    print(list(file['impulsivity'].keys()))

                    print('Map keys')
                    print(list(file['map_direction'].keys()))

                    if not numpy.isin('cw',dsets):
                        print('cw dataset does not exist for this run.')
                    else:
                        cw_dsets = list(file['cw'].keys())
                        print(list(file['cw'].attrs))
                        prep = FFTPrepper(reader, final_corr_length=int(file['cw'].attrs['final_corr_length']), crit_freq_low_pass_MHz=float(file['cw'].attrs['crit_freq_low_pass_MHz']), crit_freq_high_pass_MHz=float(file['cw'].attrs['crit_freq_high_pass_MHz']), low_pass_filter_order=float(file['cw'].attrs['low_pass_filter_order']), high_pass_filter_order=float(file['cw'].attrs['high_pass_filter_order']), waveform_index_range=(None,None), plot_filters=False)
                        prep.addSineSubtract(file['cw'].attrs['sine_subtract_min_freq_GHz'], file['cw'].attrs['sine_subtract_max_freq_GHz'], file['cw'].attrs['sine_subtract_percent'], max_failed_iterations=3, verbose=False, plot=False)
            

                        #Add attributes for future replicability. 
                        
                        raw_freqs = prep.rfftWrapper(prep.t(), numpy.ones_like(prep.t()))[0]
                        df = raw_freqs[1] - raw_freqs[0]
                        freq_bins = (numpy.append(raw_freqs,raw_freqs[-1]+df) - df/2)/1e6 #MHz

                        freq_hz = file['cw']['freq_hz'][...][trigger_type_cut]
                        linear_magnitude = file['cw']['linear_magnitude'][...][trigger_type_cut]
                        binary_cw_cut = file['cw']['has_cw'][...][trigger_type_cut]
                        if not numpy.isin('dbish',cw_dsets):
                            dbish = 10.0*numpy.log10( linear_magnitude[binary_cw_cut]**2 / len(prep.t()))
                        else:
                            dbish = file['cw']['dbish'][...][trigger_type_cut][binary_cw_cut]
                        if plot_maps:
                            cor = Correlator(reader,  upsample=2**16, n_phi=720, n_theta=720, waveform_index_range=(None,None),crit_freq_low_pass_MHz=float(file['cw'].attrs['crit_freq_low_pass_MHz']), crit_freq_high_pass_MHz=float(file['cw'].attrs['crit_freq_high_pass_MHz']), low_pass_filter_order=float(file['cw'].attrs['low_pass_filter_order']), high_pass_filter_order=float(file['cw'].attrs['high_pass_filter_order']), plot_filter=False,apply_phase_response=True, tukey=False, sine_subtract=True)
                            cor.prep.addSineSubtract(file['cw'].attrs['sine_subtract_min_freq_GHz'], file['cw'].attrs['sine_subtract_max_freq_GHz'], file['cw'].attrs['sine_subtract_percent'], max_failed_iterations=3, verbose=False, plot=False)

                        sine_subtract_min_freq_MHz = 1000*float(file['cw'].attrs['sine_subtract_min_freq_GHz'])
                        sine_subtract_max_freq_MHz = 1000*float(file['cw'].attrs['sine_subtract_max_freq_GHz'])

                    file.close()
                
                
                ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                        curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                        impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                        p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

                # ds.addROI('48 MHz',{'cw_freq_Mhz':[47,49]})
                # ds.addROI('42 MHz',{'cw_freq_Mhz':[41,43]})
                # ds.addROI('52 MHz',{'cw_freq_Mhz':[51,53]})
                # ds.addROI('88 MHz',{'cw_freq_Mhz':[88,90]})
                # ds.addROI('Background Cluster',{'phi_best_h':[35,50],'elevation_best_h':[-45,-28]})
                if False:
                    plot_param_pairs = [['impulsivity_h','impulsivity_v'],['phi_best_h','phi_best_v'], ['elevation_best_h','elevation_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]
                    for key_x, key_y in plot_param_pairs:
                        print('Generating %s plot'%(key_x + ' vs ' + key_y))
                        fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)

                if True:
                    if trigger_types == [2]:
                        #This set of plots aims to compare high impulsivity events across a few plots.  
                        # ds.addROI('Impulsivity V > 0.5',{'impulsivity_v':[0.5,1.0]})#,'impulsivity_v':[0.5,1.0]
                        # ds.addROI('Impulsivity H > 0.5',{'impulsivity_h':[0.5,1.0]})#,'impulsivity_v':[0.5,1.0]
                        # _eventids = numpy.unique(numpy.append(ds.getCutsFromROI('Impulsivity H > 0.5',load=False,save=False),ds.getCutsFromROI('Impulsivity V > 0.5',load=False,save=False)))
                        #_eventids = _eventids[~numpy.isin(_eventids,eventids_60Hz)] #only events not 60Hz, this is a simplification, because the algorithm does not catch ALL 60Hz events. 
                        _eventids = eventids
                        #eventids_60Hz = _eventids[numpy.isin(_eventids,eventids_60Hz)]
                        plot_param_pairs = [['impulsivity_h','impulsivity_v']]#[['time_delay_0subtract1_h','time_delay_0subtract2_h'],['impulsivity_h','impulsivity_v'],['phi_best_h','phi_best_v'], ['elevation_best_h','elevation_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]#[['phi_best_h','phi_best_v'], ['elevation_best_h','elevation_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]#[['phi_best_h','theta_best_h'], ['phi_best_v','theta_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]#,  ['std_h', 'std_v'], ['cw_freq_Mhz','cw_dbish'],['impulsivity_h','impulsivity_v']]#[['impulsivity_h','impulsivity_v'], ['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v']]
                        for key_x, key_y in plot_param_pairs:
                            print('Generating %s plot'%(key_x + ' vs ' + key_y))
                            # if 'impulsivity' in key_x:
                            #     fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=False)
                            #     fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=_eventids, cmap='coolwarm', include_roi=False)
                            # else:
                            #     fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=_eventids, cmap='coolwarm', include_roi=False)
                            #ds.addContour(ax, key_x, key_y, eventids_60Hz, 'r', load=False, n_contour=5, alpha=0.85, log_contour=True)
                            # fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=eventids_60Hz, cmap='coolwarm', include_roi=False)
                            fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=_eventids, cmap='coolwarm', include_roi=False)
                    else:
                        plot_param_pairs = [['impulsivity_h','impulsivity_v'],['phi_best_h','phi_best_v'], ['elevation_best_h','elevation_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]#[['phi_best_h','phi_best_v'], ['elevation_best_h','elevation_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]#[['phi_best_h','theta_best_h'], ['phi_best_v','theta_best_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v']]#,  ['std_h', 'std_v'], ['cw_freq_Mhz','cw_dbish'],['impulsivity_h','impulsivity_v']]#[['impulsivity_h','impulsivity_v'], ['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v']]
                        for key_x, key_y in plot_param_pairs:
                            print('Generating %s plot'%(key_x + ' vs ' + key_y))
                            fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=False)


                if False:
                    fig = plt.figure()
                    plt.suptitle('Run %i - Considering trigger types: %s'%(run, str(trigger_types)))

                    ax1 = plt.subplot(3,1,1)
                    plt.hist(freq_hz/1e6,bins=freq_bins)
                    plt.xlim(sine_subtract_min_freq_MHz,sine_subtract_max_freq_MHz)
                    plt.yscale('log', nonposy='clip')
                    plt.grid(which='both', axis='both')
                    ax1.minorticks_on()
                    ax1.grid(b=True, which='major', color='k', linestyle='-')
                    ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.xlabel('Freq (MHz)')

                    ax2 = plt.subplot(3,1,2)
                    plt.hist(dbish,bins=50) #I think this factor of 2 makes it match monutau?
                    plt.yscale('log', nonposy='clip')
                    plt.grid(which='both', axis='both')
                    ax2.minorticks_on()
                    ax2.grid(b=True, which='major', color='k', linestyle='-')
                    ax2.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.xlabel('Power (dBish)')
                    plt.ylabel('Counts')

                    ax3 = plt.subplot(3,1,3)
                    plt.hist(binary_cw_cut.astype(int),bins=3,weights=numpy.ones(len(binary_cw_cut))/len(binary_cw_cut)) 
                    plt.grid(which='both', axis='both')
                    ax3.minorticks_on()
                    ax3.grid(b=True, which='major', color='k', linestyle='-')
                    ax3.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.xlabel('Without (0) --- With CW (1)')
                    plt.ylabel('Percent of Counts')

                    print('Max CW Eventid: %i'%numpy.where(trigger_type_cut)[0][numpy.where(binary_cw_cut)[0][numpy.argmax(dbish)]])
                    print('Approximate power is %0.3f dBish, at %0.2f MHz'%(numpy.max(dbish), freq_hz[numpy.where(binary_cw_cut)[0][numpy.argmax(dbish)]]/1e6))

                    if len(db_subset_plot_ranges) > 0:
                        for subset_index, subrange in enumerate(db_subset_plot_ranges):
                            cut = numpy.logical_and(dbish > subrange[0], dbish <= subrange[1])
                            if run  == 1650:
                                eventid = [89436,21619,113479][subset_index]
                            else:
                                eventid = numpy.random.choice(numpy.where(trigger_type_cut)[0][numpy.where(binary_cw_cut)[0][cut]])

                            prep.plotEvent(eventid, channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=False, sine_subtract=False, apply_tukey=None)
                            prep.plotEvent(eventid, channels=[0,1,2,3,4,5,6,7], apply_filter=True, hilbert=False, sine_subtract=True, apply_tukey=None)

                            ax1.axvline(freq_hz[numpy.where(eventids[trigger_type_cut] == eventid)[0][0] ]/1e6,color=subset_colors[subset_index],label='r%ie%i'%(run, eventid))
                            ax1.legend(loc='upper left')
                            ax2.axvline(dbish[numpy.where(eventids[trigger_type_cut][binary_cw_cut] == eventid)[0][0] ],color=subset_colors[subset_index],label='r%ie%i'%(run, eventid))
                            ax2.legend(loc='upper left')
                            if plot_maps:
                                for ss in [False,True]:
                                    cor.apply_sine_subtract = ss
                                    result = cor.map(eventid, 'hpol', center_dir='E', plot_map=True, plot_corr=False, hilbert=hilbert, interactive=True, max_method=0,mollweide=True,circle_zenith=None,circle_az=None)
            else:
                print('filename is None, indicating empty tree.  Skipping run %i'%run)
        except Exception as e:
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


