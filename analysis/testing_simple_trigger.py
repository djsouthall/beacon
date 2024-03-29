'''
Eric had an idea for a simple trigger that involves ensuring the top 2 antennas trigger first.
The idea behind this is that in a simple model it would mean that we are triggering on signals
coming from above.  The goal would be to run this for a few days to collect some data that
is designed (by trigger) to mostly/only contain signals from above (like CRs).  Because the
array geometry is not so simple, and we predict most CRs to come in at around 60-70 zentih,
I am not sure this is a complete picture.   This script is intended to work through that problem.
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv
import glob

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_slicer import dataSlicerSingleRun, dataSlicer
import tools.clock_correct as cc
import tools.info as info
from tools.data_handler import createFile, getTimes
from tools.correlator import Correlator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import pprint
import itertools
import warnings
import h5py
import pandas as pd
import tools.get_plane_tracks as pt
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

datapath = os.environ['BEACON_DATA']

cable_delays = info.loadCableDelays()



if __name__ == '__main__':
    plt.close('all')
    runs = numpy.arange(1650,1700)
    run = runs[0] #Doesn't really matter, using correlator for antenna positions more than anything
    final_corr_length = 2**16
    #FILTER STRING USED IF ABOVE IS FALSE
    default_align_method=0 #WILL BE CHANGED IF GIVEN ABOVE
    filter_string = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_%i-align_%i'%(final_corr_length,default_align_method)
    impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'


    crit_freq_low_pass_MHz = None#60 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = None#5

    crit_freq_high_pass_MHz = None#60
    high_pass_filter_order = None#6

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.03
    sine_subtract_max_freq_GHz = 0.09
    sine_subtract_percent = 0.03

    waveform_index_range = (None,None)#(150,400)
    plot_filter=False

    n_phi = 720
    n_theta = 720
    upsample = final_corr_length

    max_method = 0

    apply_phase_response = False
    hilbert = False
    use_interpolated_tracks = True

    reader = Reader(datapath,run)
    cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, n_theta=n_theta, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter,apply_phase_response=apply_phase_response)

    cor.calculateArrayNormalVector(plot_map=False,mollweide=True, pol='both')
    roi_dicts = {}

    ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                        curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                        impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                        p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

    center_dir = 'W'    

    pol = 'hpol'

    if True:
        #This tests Eric's original idea of just setting a trigger where the top antennas trigger first. 
        if pol == 'hpol':
            top_antenna_cut = numpy.logical_and(cor.t_hpol_0subtract1 >= 10.0, cor.t_hpol_2subtract3 >= 10.0)
            mean_corr_values = top_antenna_cut
            time_delay_dict = {'hpol':{'[0,1]':[10.0],'[2,3]':[10.0]}}
            roi_dicts['Simple Pass'] = {'time_delay_0subtract1_h':[10.0,300],'time_delay_2subtract3_h':[10.0,300]}
            roi_dicts['Simple Pass, High CR Corr'] = {'time_delay_0subtract1_h':[10.0,300],'time_delay_2subtract3_h':[10.0,300],'cr_template_search_h':[0.4,1.0], 'cr_template_search_v':[0.4,1.0]}
        else:
            top_antenna_cut = numpy.logical_and(cor.t_vpol_0subtract1 >= 10.0, cor.t_vpol_2subtract3 >= 10.0)
            mean_corr_values = top_antenna_cut
            time_delay_dict = {'hpol':{'[0,1]':[10.0],'[2,3]':[10.0]}}
            roi_dicts['Simple Pass'] = {'time_delay_0subtract1_v':[10.0,300],'time_delay_2subtract3_v':[10.0,300]}


    elif False:
        #This is where I want to play with some other baselines to see if I can isolate the sky.
        if pol == 'hpol':
            top_antenna_cut = numpy.abs(cor.t_hpol_0subtract1) + numpy.abs(cor.t_hpol_0subtract2) + numpy.abs(cor.t_hpol_0subtract3) + numpy.abs(cor.t_hpol_1subtract2) + numpy.abs(cor.t_hpol_1subtract3) + numpy.abs(cor.t_hpol_2subtract3)
            mean_corr_values = top_antenna_cut < 300
            time_delay_dict = {'hpol':{'[0,1]':[0.0],'[0,2]':[0.0],'[0,3]':[0.0],'[1,2]':[0.0],'[1,3]':[0.0],'[2,3]':[0.0]}}
        else:
            top_antenna_cut = numpy.abs(cor.t_vpol_0subtract1) + numpy.abs(cor.t_vpol_0subtract2) + numpy.abs(cor.t_vpol_0subtract3) + numpy.abs(cor.t_vpol_1subtract2) + numpy.abs(cor.t_vpol_1subtract3) + numpy.abs(cor.t_vpol_2subtract3)
            mean_corr_values = top_antenna_cut < 300
            time_delay_dict = {'vpol':{'[0,1]':[0.0],'[0,2]':[0.0],'[0,3]':[0.0],'[1,2]':[0.0],'[1,3]':[0.0],'[2,3]':[0.0]}}
    elif False:
        #This cuts out the sky below 50
        #This is where I want to play with some other baselines to see if I can isolate the sky.
        #top_antenna_cut = numpy.abs(cor.t_hpol_0subtract1) + numpy.abs(cor.t_hpol_0subtract2) + numpy.abs(cor.t_hpol_0subtract3) + numpy.abs(cor.t_hpol_1subtract2) + numpy.abs(cor.t_hpol_1subtract3) + numpy.abs(cor.t_hpol_2subtract3)
        top_antenna_cut = (cor.t_hpol_0subtract1 > -25).astype(int) + (cor.t_hpol_0subtract2 < 130).astype(int) + (cor.t_hpol_0subtract3 > -65).astype(int) + (cor.t_hpol_1subtract3 > -75).astype(int) + (cor.t_hpol_1subtract2 < 60).astype(int) + (cor.t_hpol_2subtract3 > -20).astype(int) #(cor.t_hpol_0subtract1 < -30.0 )*(cor.t_hpol_0subtract3 < -30.0)*(cor.t_hpol_1subtract2 > 30.0)*(cor.t_hpol_2subtract3 < -30.0)# + numpy.abs(cor.t_hpol_0subtract2) + numpy.abs(cor.t_hpol_0subtract3) + numpy.abs(cor.t_hpol_1subtract2) + numpy.abs(cor.t_hpol_1subtract3) + numpy.abs(cor.t_hpol_2subtract3)
        #top_antenna_cut = top_antenna_cut == 4
        top_antenna_cut = numpy.multiply(top_antenna_cut,cor.hpol_dot_angle_from_plane_deg > 0)
        #top_antenna_cut = top_antenna_cut == 4
        mean_corr_values = top_antenna_cut
        time_delay_dict = {'hpol':{'[0,1]':[-25.0],'[0,2]':[130.0],'[0,3]':[-65.0],'[1,2]':[60.0],'[1,3]':[-75.0],'[2,3]':[-20.0]}}#{'hpol':{'[0,1]':[0.0],'[0,2]':[0.0],'[0,3]':[0.0],'[1,2]':[0.0],'[1,3]':[0.0],'[2,3]':[0.0]}}
    elif False:
        #This tries to identify the existence of a point where expected time delays exist.  
        #basically just using this tool to explore a different problem.
        top_antenna_cut = (numpy.abs(cor.t_hpol_0subtract1) < 10).astype(int) + (numpy.abs(cor.t_hpol_0subtract2) < 10).astype(int) + (numpy.abs(cor.t_hpol_0subtract3) < 10).astype(int) + (numpy.abs(cor.t_hpol_1subtract3) < 10).astype(int) + (numpy.abs(cor.t_hpol_1subtract2) < 10).astype(int) + (numpy.abs(cor.t_hpol_2subtract3) < 10).astype(int) 
        #top_antenna_cut = top_antenna_cut == 4
        #top_antenna_cut = top_antenna_cut == 4
        mean_corr_values = top_antenna_cut
        time_delay_dict = {'hpol':{'[0,1]':[0],'[0,2]':[0],'[0,3]':[0],'[1,2]':[0],'[1,3]':[0],'[2,3]':[0]}}
        cor.generateTimeDelayOverlapMap('hpol', time_delay_dict, 10.0, plot_map=True, mollweide=False,center_dir='E', window_title='TESTING', include_baselines=[0,1,2,3,4,5])

    else:
        top_antenna_cut = cor.t_hpol_1subtract2 # + numpy.abs(cor.t_hpol_0subtract2) + numpy.abs(cor.t_hpol_0subtract3) + numpy.abs(cor.t_hpol_1subtract2) + numpy.abs(cor.t_hpol_1subtract3) + numpy.abs(cor.t_hpol_2subtract3)
        mean_corr_values = top_antenna_cut
        time_delay_dict = {'hpol':{'[1,2]':[20.0]}}#{'hpol':{'[0,1]':[0.0],'[0,2]':[0.0],'[0,3]':[0.0],'[1,2]':[0.0],'[1,3]':[0.0],'[2,3]':[0.0]}}

    
    include_baselines = numpy.array([])
    plot_map = False
    hilbert = False
    interactive = False
    max_method = None
    waveforms = None
    verbose = True
    mollweide = False
    zenith_cut_ENU = None
    zenith_cut_array_plane = None
    center_dir = 'E'
    circle_zenith = None
    circle_az = None
    #time_delay_dict = {'hpol':{'[0,1]':[0.0],'[0,2]':[0.0],'[0,3]':[0.0],'[1,2]':[0.0],'[1,3]':[0.0],'[2,3]':[0.0]},'vpol':{'[0,1]':[0.0],'[0,2]':[0.0],'[0,3]':[0.0],'[1,2]':[0.0],'[1,3]':[0.0],'[2,3]':[0.0]}}
    
    #This assumes None-geometric time delay, as in the observed time delay.  And plots it onto the map which is intended to show actual time delays as they hit the antennas?
    # The first level of the dict should specify 'hpol' and/or 'vpol'
    # The following key within should have each of the baseline pairs that you wish to plot.  Each of these
    # will correspond to a list of floats with all of the time delays for that baseline you want plotted. 

    #The code below is pulled from cor.map, but tuned to make this plot.
    if ~numpy.all(numpy.isin(include_baselines, numpy.array([0,1,2,3,4,5]))):
        add_xtext = '\nIncluded baselines = ' + str(numpy.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])[include_baselines])
    else:
                add_xtext = ''
    if center_dir.upper() == 'E':
        center_dir_full = 'East'
        azimuth_offset_rad = 0 #This is subtracted from the xaxis to roll it effectively.
        azimuth_offset_deg = 0 #This is subtracted from the xaxis to roll it effectively.
        xlabel = 'Azimuth (From East = 0 deg, North = 90 deg)' + add_xtext
        roll = 0
    elif center_dir.upper() == 'N':
        center_dir_full = 'North'
        azimuth_offset_rad = numpy.pi/2 #This is subtracted from the xaxis to roll it effectively. 
        azimuth_offset_deg = 90 #This is subtracted from the xaxis to roll it effectively. 
        xlabel = 'Azimuth (From North = 0 deg, West = 90 deg)' + add_xtext
        roll = numpy.argmin(abs(cor.phis_rad - azimuth_offset_rad))
    elif center_dir.upper() == 'W':
        center_dir_full = 'West'
        azimuth_offset_rad = numpy.pi #This is subtracted from the xaxis to roll it effectively.
        azimuth_offset_deg = 180 #This is subtracted from the xaxis to roll it effectively.
        xlabel = 'Azimuth (From West = 0 deg, South = 90 deg)' + add_xtext
        roll = len(cor.phis_rad)//2
    elif center_dir.upper() == 'S':
        center_dir_full = 'South'
        azimuth_offset_rad = -numpy.pi/2 #This is subtracted from the xaxis to roll it effectively.
        azimuth_offset_deg = -90 #This is subtracted from the xaxis to roll it effectively.
        xlabel = 'Azimuth (From South = 0 deg, East = 90 deg)' + add_xtext
        roll = numpy.argmin(abs(cor.phis_rad - azimuth_offset_rad))

    rolled_values = numpy.roll(mean_corr_values,roll,axis=1)

    #elevation_best_deg = 90.0 - theta_best

    fig = plt.figure()
    #fig.canvas.set_window_title('r%i-e%i-%s Correlation Map'%(cor.reader.run,eventid,pol.title()))
    #fig.canvas.set_window_title('Trigger Map %i-%i'%(i,j))
    if mollweide == True:
        ax = fig.add_subplot(1,1,1, projection='mollweide')
    else:
        ax = fig.add_subplot(1,1,1)

    if mollweide == True:
        #Automatically converts from rads to degs
        im = ax.pcolormesh(cor.mesh_azimuth_rad, cor.mesh_elevation_rad, rolled_values, vmin=numpy.min(rolled_values), vmax=numpy.max(rolled_values),cmap=plt.cm.coolwarm)
    else:
        im = ax.pcolormesh(cor.mesh_azimuth_deg, cor.mesh_elevation_deg, rolled_values, vmin=numpy.min(rolled_values), vmax=numpy.max(rolled_values),cmap=plt.cm.coolwarm)

    cbar = fig.colorbar(im)

    cbar.set_label('Cut Value')

    plt.xlabel(xlabel,fontsize=18)
    plt.ylabel('Elevation Angle (Degrees)',fontsize=18)
    plt.grid(True)

    #Prepare array cut curves
    if pol is not None:
        if zenith_cut_array_plane is not None:
            if len(zenith_cut_array_plane) != 2:
                print('zenith_cut_array_plane must be a 2 valued list.')
                pass
            else:
                if zenith_cut_array_plane[0] is None:
                    zenith_cut_array_plane[0] = 0
                if zenith_cut_array_plane[1] is None:
                    zenith_cut_array_plane[1] = 180

    #Prepare center line and plot the map.  Prep cut lines as well.
    if pol == 'hpol':
        selection_index = 1
    elif pol == 'vpol':
        selection_index = 2 
    
    plane_xy = cor.getArrayPlaneZenithCurves(90.0, azimuth_offset_deg=azimuth_offset_deg)[selection_index]

    # #Plot array plane 0 elevation curve.
    im = cor.addCurveToMap(im, plane_xy,  mollweide=mollweide, linewidth = cor.min_elevation_linewidth, color='k')

    if zenith_cut_array_plane is not None:
        upper_plane_xy = cor.getArrayPlaneZenithCurves(zenith_cut_array_plane[0], azimuth_offset_deg=azimuth_offset_deg)[selection_index]
        lower_plane_xy = cor.getArrayPlaneZenithCurves(zenith_cut_array_plane[1], azimuth_offset_deg=azimuth_offset_deg)[selection_index]
        #Plot upper zenith array cut
        im = cor.addCurveToMap(im, upper_plane_xy,  mollweide=mollweide, linewidth = cor.min_elevation_linewidth, color='k',linestyle = '--')
        #Plot lower zenith array cut
        im = cor.addCurveToMap(im, lower_plane_xy,  mollweide=mollweide, linewidth = cor.min_elevation_linewidth, color='k',linestyle = '--')

    #Add curves for time delays if present.
    im = cor.addTimeDelayCurves(im, time_delay_dict, pol,  ax, mollweide=mollweide, azimuth_offset_deg=azimuth_offset_deg, include_baselines=include_baselines)

    #Added circles as specified.
    #ax, peak_circle = cor.addCircleToMap(ax, phi_best, elevation_best_deg, azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius=5.0, crosshair=True, return_circle=True, color='lime', linewidth=0.5,fill=False)

    if circle_az is not None:
        if circle_zenith is not None:
            if circle_az is list:
                circle_az = numpy.array(circle_az)
            elif circle_az != numpy.ndarray:
                circle_az = numpy.array([circle_az])

            _circle_az = circle_az.copy() #These were changing when 'both' was called for pol.  So copying them to ensure original input maintains values.

            if circle_zenith is list:
                circle_zenith = numpy.array(circle_zenith)
            elif circle_zenith != numpy.ndarray:
                circle_zenith = numpy.array([circle_zenith])

            _circle_zenith = circle_zenith.copy() #These were changing when 'both' was called for pol.  So copying them to ensure original input maintains values.

            if len(_circle_zenith) == len(_circle_az):
                additional_circles = []
                for i in range(len(_circle_az)):
                    ax, _circ = cor.addCircleToMap(ax, _circle_az[i], 90.0-_circle_zenith[i], azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius=5.0, crosshair=False, return_circle=True, color='fuchsia', linewidth=0.5,fill=False)
                    additional_circles.append(_circ)

    if zenith_cut_ENU is not None:
        if mollweide == True:
            #Block out simple ENU zenith cut region. 
            plt.axhspan(numpy.deg2rad(90 - min(zenith_cut_ENU)),numpy.deg2rad(90.0),alpha=0.5)
            plt.axhspan(numpy.deg2rad(-90) , numpy.deg2rad(90 - max(zenith_cut_ENU)),alpha=0.5)
        else:
            #Block out simple ENU zenith cut region. 
            plt.axhspan(90 - min(zenith_cut_ENU),90.0,alpha=0.5)
            plt.axhspan(-90 , 90 - max(zenith_cut_ENU),alpha=0.5)

    #Enable Interactive Portion
    if interactive == True:
        fig.canvas.mpl_connect('button_press_event',lambda event : cor.interactivePlotter(event,  mollweide=mollweide, center_dir=center_dir))

    #ax.legend(loc='lower left')
    cor.figs.append(fig)
    cor.axs.append(ax)


    #Here I loop over runs to match the above plotted cut.  

    runs = numpy.arange(1650,1675)

    ds = dataSlicer(    runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                        curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                        impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                        p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

    [ds.addROI(key, item) for key, item in roi_dicts.items()]



    eventids_dict = ds.getCutsFromROI('Simple Pass')
    if pol == 'hpol':
        plot_param_pairs = [['phi_best_h','elevation_best_h'],['time_delay_0subtract1_h', 'time_delay_2subtract3_h'],['cr_template_search_h', 'cr_template_search_v']]
    else:
        plot_param_pairs = [['phi_best_v','elevation_best_v'],['time_delay_0subtract1_v', 'time_delay_2subtract3_v'],['cr_template_search_h', 'cr_template_search_v']]
    for key_x, key_y in plot_param_pairs:
        fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
    for key_x, key_y in plot_param_pairs:
        fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True, eventids_dict=eventids_dict)
    if True:
        print(ds.getCutsFromROI('Simple Pass, High CR Corr',verbose=True))
        azs_dict = ds.getDataFromParam(eventids_dict, 'phi_best_h')
        run_index = -1
        hist = None
        for run, eventids in eventids_dict.items():
            run_index += 1
            azs = azs_dict[run]
            reader = Reader(datapath,run)
            cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, n_theta=n_theta, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter,apply_phase_response=apply_phase_response)

            cor.calculateArrayNormalVector(plot_map=False,mollweide=True, pol='both')

            # for event_index, eventid in enumerate(eventids):
            #     if abs(azs[event_index]) < 45:
            #         center_dir='E'
            #     elif numpy.logical_and(azs[event_index] > 45,azs[event_index] < 135):
            #         center_dir='N'
            #     elif numpy.logical_and(azs[event_index] < -45,azs[event_index] > -135):
            #         center_dir='S'
            #     else:
            #         center_dir='W'
            #     result = cor.map(eventid, pol, center_dir=center_dir, plot_map=True, plot_corr=False, hilbert=False, interactive=True, max_method=0,mollweide=False,circle_zenith=None,circle_az=None)
            if run_index == 0:
                #First call, no input histogram.
                hist, all_phi_best, all_theta_best = cor.histMapPeak(eventids, pol, plot_map=False, initial_hist=None, initial_phis=None, initial_thetas=None, hilbert=False, max_method=0, plot_max=False, use_weight=False, mollweide=False, center_dir='E', zenith_cut_ENU=[0,90],zenith_cut_array_plane=[0,90],window_title='Simple Pass Hist')
            elif run_index < len(list(eventids_dict.keys())) - 1:
                #Every other time use the previous hist as input, but don't plot.
                hist, all_phi_best, all_theta_best = cor.histMapPeak(eventids, pol, plot_map=False, initial_hist=hist, initial_phis=all_phi_best, initial_thetas=all_theta_best, hilbert=False, max_method=0, plot_max=False, use_weight=False, mollweide=False, center_dir='E', zenith_cut_ENU=[0,90],zenith_cut_array_plane=[0,90],window_title='Simple Pass Hist')
            else:
                #Final time, plot the hist. 
                hist, all_phi_best, all_theta_best = cor.histMapPeak(eventids, pol, plot_map=True, initial_hist=hist, initial_phis=all_phi_best, initial_thetas=all_theta_best, hilbert=False, max_method=0, plot_max=False, use_weight=False, mollweide=False, center_dir='E', zenith_cut_ENU=[0,90],zenith_cut_array_plane=[0,90],window_title='Simple Pass Hist')

