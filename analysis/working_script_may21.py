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
    runs = numpy.arange(1600,1729)#numpy.array([1657])#numpy.array([1650])#numpy.arange(1650,1750)#numpy.array([1650,1728,1773,1774,1783,1784])#numpy.arange(1650,1675)#numpy.array([1774])#numpy.array([1650,1728,1773,1774,1783,1784])#numpy.array([1650])#
    #run = 1774 #want run with airplane.  This has 1774-178 in it. 

    datapath = os.environ['BEACON_DATA']

    #Deploy 30

    time_delays_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    impulsivity_dset_key = time_delays_dset_key
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_65536-maxmethod_0-sinesubtract_1-deploy_calibration_30-scope_allsky'

    crit_freq_low_pass_MHz = 85 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = 6

    crit_freq_high_pass_MHz = 25
    high_pass_filter_order = 8

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.03
    sine_subtract_max_freq_GHz = 0.13
    sine_subtract_percent = 0.03

    hilbert=False
    final_corr_length = 2**12

    trigger_types = [2]#[2]
    plot_maps = True
    n_phi = 720 #Used in dataSlicer
    range_phi_deg = (-180,180) #Used in dataSlicer
    n_theta = 720 #Used in dataSlicer
    range_theta_deg = (0,180) #Used in dataSlicer

    sum_events = True #If true will add all plots together, if False will loop over runs in runs.
    lognorm = True
    cmap = 'binary'#'YlOrRd'#'binary'#'coolwarm'
    subset_cm = plt.cm.get_cmap('autumn', 10)

    try:
        if len(runs) == 1 or sum_events == False:
            for run in runs:
                run = int(run)

                reader = Reader(datapath,run)
                if reader.failed_setup == True:
                    print('Error for run %i, skipping.'%run)
                    continue
                cor = Correlator(reader,  upsample=2**16, n_phi=720, n_theta=720, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=True, tukey=False, sine_subtract=True)
                if sine_subtract:
                    cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
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
                
                ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                        curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                        impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                        p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,\
                        n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg)

                if filename is not None:
                    with h5py.File(filename, 'r') as file:
                        eventids = file['eventids'][...]

                        trigger_type_cut = numpy.isin(file['trigger_type'][...], trigger_types)

                        dsets = list(file.keys()) #Existing datasets

                        print('Time delay keys')
                        for d in list(file['time_delays'].keys()):
                            print('\t' + d)

                        print('Impulsivity keys')
                        for d in list(file['impulsivity'].keys()):
                            print('\t' + d)

                        print('Map keys')
                        for d in list(file['map_direction'].keys()):
                            print('\t' + d)

                        file.close()
                #This set of plots aims to compare high impulsivity events across a few plots.  
                if False:
                    min_imp = 0.3
                    ds.addROI('Impulsivity V > %s'%str(min_imp),{'impulsivity_v':[min_imp,1.0]})
                    ds.addROI('Impulsivity H > %s'%str(min_imp),{'impulsivity_h':[min_imp,1.0]})
                    _eventids = numpy.unique(numpy.append(ds.getCutsFromROI('Impulsivity H > %s'%str(min_imp),load=False,save=False),ds.getCutsFromROI('Impulsivity V > %s'%str(min_imp),load=False,save=False)))
                    ds.resetAllROI() #already have the eventids with sufficient impulsivity
                elif True:
                    min_imp = 0.3
                    ds.addROI('Impulsivity > %s'%str(min_imp),{'impulsivity_h':[min_imp,1.0],'impulsivity_v':[min_imp,1.0]})
                    _eventids = ds.getCutsFromROI('Impulsivity > %s'%str(min_imp),load=False,save=False)
                    ds.resetAllROI() #already have the eventids with sufficient impulsivity
                else:
                    _eventids = None
                

                ds.addROI('above horizon',{'elevation_best_h':[0,90]})
                sky_cut = ds.getCutsFromROI('above horizon',load=False,save=False)

                sky_cut = sky_cut[numpy.isin(sky_cut,_eventids)]


                plot_param_pairs = [['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v'],['impulsivity_h','impulsivity_v']]

                for key_x, key_y in plot_param_pairs:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=_eventids, cmap=cmap, include_roi=True, lognorm=lognorm)
                for key_x, key_y in plot_param_pairs:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=sky_cut, cmap='viridis', include_roi=False, lognorm=False)



                _plots = []
                for eventid in sky_cut:
                    mean_corr_values, fig, ax = cor.map(eventid, 'hpol', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=None, verbose=True, mollweide=False, zenith_cut_ENU=None, zenith_cut_array_plane=None, center_dir='E', circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={},window_title='Imp H %0.2f'%(ds.getDataFromParam(eventid, 'impulsivity_h')),add_airplanes=True, return_max_possible_map_value=False)
                    _plots.append(fig)
                    mean_corr_values, fig, ax = cor.map(eventid, 'vpol', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=None, verbose=True, mollweide=False, zenith_cut_ENU=None, zenith_cut_array_plane=None, center_dir='E', circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={},window_title='Imp V %0.2f'%(ds.getDataFromParam(eventid, 'impulsivity_v')),add_airplanes=True, return_max_possible_map_value=False)
                    _plots.append(fig)

                    import pdb; pdb.set_trace()
                    [plt.close(p) for p in _plots]


        else:
            ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, remove_incomplete_runs=True,\
                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,\
                    n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg)
            

            ds.addROI('A',{'time_delay_0subtract1_h':[-127,-123],'time_delay_0subtract2_h':[-127,-123.5]})
            ds.addROI('B',{'time_delay_0subtract1_h':[-135,-131],'time_delay_0subtract2_h':[-111,-105]})
            ds.addROI('C',{'time_delay_0subtract1_h':[-140.5,-137],'time_delay_0subtract2_h':[-90,-83.5],'time_delay_0subtract3_h':[-167,-161],'time_delay_1subtract2_h':[46,55]})
            ds.addROI('D',{'time_delay_0subtract1_h':[-143,-140],'time_delay_0subtract2_h':[-60.1,-57.4]})
            ds.addROI('E',{'time_delay_0subtract1_h':[-124.5,-121],'time_delay_0subtract2_h':[22.5,28.5]})
            ds.addROI('F',{'time_delay_0subtract1_h':[-138,-131.7],'time_delay_0subtract2_h':[-7,-1]})



            for cut in [False, True]:
                if cut == True:
                    min_imp = 0.68
                    ds.addROI('Both Impulsivity > %s'%str(min_imp),{'impulsivity_h':[min_imp,1.0],'impulsivity_v':[min_imp,1.0]})
                    eventids_dict = ds.getCutsFromROI('Both Impulsivity > %s'%str(min_imp),load=False,save=False,verbose=False)
                    # cmap = 'gray'
                else:
                    # cmap = 'binary'
                    eventids_dict = None


                plot_param_pairs = [['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v'],['impulsivity_h','impulsivity_v']]
                
                for key_x, key_y in plot_param_pairs:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=eventids_dict,include_roi=True, lognorm=lognorm)



    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


