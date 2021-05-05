'''
This is a script is intended to load in some signals that are dominated by TV signals.  Hopefully we can point to them
and test Cosmin's predicted time delays. 
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
import pymap3d as pm

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
    run = 1650
    deploy_index = info.returnDefaultDeploy()

    datapath = os.environ['BEACON_DATA']

    impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    #map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'
    map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_belowhorizon'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_belowhorizon' # 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_22-scope_allsky'
    
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

    trigger_types = [1,3]#[2]
    plot_maps = True

    sum_events = False #If true will add all plots together, if False will loop over runs in runs.
    lognorm = True
    cmap = 'YlOrRd'#'binary'#'coolwarm'
    subset_cm = plt.cm.get_cmap('autumn', 10)

    origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
    latlonel_khsv = numpy.array([36.008611, -115.005556, 3316*0.3048 + 10]) #Height on google maps is ~ 3316 feet, which I convert to meters and add additional height for the tower it is on.
    enu_khsv = numpy.array(pm.geodetic2enu(latlonel_khsv[0],latlonel_khsv[1],latlonel_khsv[2],origin[0],origin[1],origin[2]))
    distance_m = numpy.linalg.norm(enu_khsv)
    zenith_deg = numpy.rad2deg(numpy.arccos(enu_khsv[2]/distance_m))
    elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(enu_khsv[2]/distance_m))
    azimuth_deg = numpy.rad2deg(numpy.arctan2(enu_khsv[1],enu_khsv[0]))

    try:
        if True:
            center_dir = 'E'
            apply_phase_response = True
            reader = Reader(datapath,run)
            cor = Correlator(reader,  upsample=2**16, n_phi=720, n_theta=720, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True)
            # mean_corr_values, fig, ax = cor.map(eventid, 'hpol', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, plot_corr=False, hilbert=False, interactive=False, max_method=None, waveforms=None, verbose=True, mollweide=True, center_dir=center_dir, circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False)
            ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                                    time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)
            
            #ds.addROI('A',{'std_h':[2,3],'std_v':[5.5,7]}) #Trig types 1 and 3 # 48 Mhz strong
            #ds.addROI('B',{'std_h':[1,2],'std_v':[2.6,3.2]}) #Trig types 1 and 3 # 42 Mhz with strong vpol component
            #ds.addROI('C',{'std_h':[1.8,2.4],'std_v':[2,2.5]}) #Trig types 1 and 3 # 48 Mhz with TV in hpol band
            #ds.addROI('D',{'std_h':[1.8,2.4],'std_v':[1.7,1.9]}) #Trig types 1 and 3 # 42 MHz crosspol with TV in hpol
            #ds.addROI('E',{'std_h':[0.9,1.5],'std_v':[1.05,1.3],'snr_v':[6,6.8]}) #Trig types 1 and 3 #Only Hpol TV signals with strong 53 MHz voik
            ds.addROI('F',{'std_h':[0.5,1.5],'std_v':[0.5,0.9],'snr_v':[5.3,6.25]}) #Trig types 1 and 3 #Only Hpol TV signals
            ds.addROI('G',{'std_h':[0.5,1.5],'std_v':[0.5,0.9],'snr_v':[6.3,7.8]}) #Trig types 1 and 3 #Only Hpol TV signals
            ds.addROI('H',{'std_h':[0.5,1.5],'std_v':[0.5,0.9],'snr_v':[8.0,9.0]}) #Trig types 1 and 3 #Only Hpol TV signals
            
            if False:

                plot_param_pairs = [['phi_best_h','elevation_best_h'],['impulsivity_h','impulsivity_v'],['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v']]
                ds.plotROIWaveforms(roi_key='H', final_corr_length=2**15, crit_freq_low_pass_MHz=70, low_pass_filter_order=8, crit_freq_high_pass_MHz=None, high_pass_filter_order=None, waveform_index_range=(None,None), plot_filter=False, apply_phase_response=True, save=False, plot_saved_templates=False, sine_subtract=False)

                for key_x, key_y in plot_param_pairs:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, include_roi=True, lognorm=lognorm)

                for key in list(ds.roi.keys()):
                    #ds.plotROIWaveforms(roi_key=None, final_corr_length=2**13, crit_freq_low_pass_MHz=None, low_pass_filter_order=None, crit_freq_high_pass_MHz=None, high_pass_filter_order=None, waveform_index_range=(None,None), plot_filter=False, apply_phase_response=True, save=False, plot_saved_templates=False, sine_subtract=False)
                    print('ROI %s has %i events'%(key,len(ds.getCutsFromROI(key))))
            else:
                window_ns = 10
                time_delay_dict = {'hpol':{'[0, 1]' : [-77.6], '[0, 2]': [95.76], '[0, 3]': [-15.82], '[1, 2]': [173.4], '[1, 3]': [61.77], '[2, 3]': [-111.6]}}
                cor.generateTimeDelayOverlapMap('hpol', time_delay_dict, window_ns, value_mode='distance', plot_map=True, mollweide=False,center_dir='E',window_title=None, include_baselines=[0,1,2,3,4,5])
                
                for roi_key in list(ds.roi.keys()):
                    #ds.plotROIWaveforms(roi_key='H', final_corr_length=2**15, crit_freq_low_pass_MHz=70, low_pass_filter_order=8, crit_freq_high_pass_MHz=None, high_pass_filter_order=None, waveform_index_range=(None,None), plot_filter=False, apply_phase_response=True, save=False, plot_saved_templates=False, sine_subtract=False)
                    eventids = ds.getCutsFromROI('H')#[0:10] #Limited for testing
                    for hilbert in [True,False]:
                        cor.histMapPeak(eventids, 'hpol', initial_hist=None, initial_thetas=None, initial_phis=None, plot_map=True, hilbert=hilbert, max_method=None, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=[85,180], zenith_cut_array_plane=[0,91], circle_zenith=zenith_deg, circle_az=azimuth_deg, window_title=None,radius=1.0,iterate_sub_baselines=None)
                        cor.averagedMap(eventids, 'hpol', plot_map=True, hilbert=hilbert, max_method=None, mollweide=False, zenith_cut_ENU=[85,180],zenith_cut_array_plane=[0,91], center_dir='E', circle_zenith=zenith_deg, circle_az=azimuth_deg, radius=1.0, time_delay_dict={})


    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


