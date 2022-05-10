#!/usr/bin/env python3
'''
This script is intended to look at the events that construct best above horizon in allsky maps.  I want to view the
peak to sidelobe values and other parameters for the belowhorizon and abovehorizon maps and determine if there is
an obvious cut for which sidelobed above horizon events can be discriminated. 
'''

import sys
import os
import inspect
import h5py
import copy
from pprint import pprint
import textwrap
import pandas

import numpy
import scipy
import scipy.signal

from beacon.tools.data_slicer import dataSlicer
import  beacon.tools.get_plane_tracks as pt
from beacon.tools.correlator import Correlator
import beacon.tools.info as info
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import time
from datetime import datetime
import pytz
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)

if True:
    crit_freq_low_pass_MHz = 85
    low_pass_filter_order = 6

    crit_freq_high_pass_MHz = 25
    high_pass_filter_order = 8

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.00
    sine_subtract_max_freq_GHz = 0.25
    sine_subtract_percent = 0.03
else:
    crit_freq_low_pass_MHz = 80
    low_pass_filter_order = 14

    crit_freq_high_pass_MHz = 20
    high_pass_filter_order = 4

    sine_subtract = False
    sine_subtract_min_freq_GHz = 0.02
    sine_subtract_max_freq_GHz = 0.15
    sine_subtract_percent = 0.01

plot_filter=False

apply_phase_response=True

mollweide = False

if mollweide == True:
    map_resolution_theta = 0.5 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.25 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

    range_phi_deg = (min_phi, max_phi)
    range_theta_deg = (min_theta, max_theta)
else:        
    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.25 #degrees
    min_phi     = -90
    max_phi     = 90
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

    range_phi_deg = (min_phi, max_phi)
    range_theta_deg = (min_theta, max_theta)

upsample = 2**16
max_method = 0

apply_filter = True

deploy_index = info.returnDefaultDeploy()
datapath = os.environ['BEACON_DATA']
map_source_distance_m = info.returnDefaultSourceDistance()
waveform_index_range = info.returnDefaultWaveformIndexRange()

if __name__ == '__main__':
    start_time = time.time()
    plt.close('all')

    fontsize = 22

    font = {'family' : 'normal',
        'size'   : fontsize}

    matplotlib.rc('font', **font)

    args = copy.copy(sys.argv)
    infile = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated.xlsx')#
    with pd.ExcelFile(infile) as xls:
        for sheet_index, sheet_name in enumerate(xls.sheet_names):
            print(sheet_name)
            if sheet_name == 'good-with-airplane':

                passing_df = xls.parse(sheet_name)
                passing_df = passing_df.sort_values(by = ['run', 'eventid'], ascending = [True, True], na_position = 'last')

                if True:
                    airplanes = ['a15c54']#['a4d5f0', 'a466fc', 'ad7828']
                else:
                    airplanes = numpy.unique(passing_df['suspected_airplane_icao24'].to_numpy(dtype=str))

                for airplane in airplanes:
                    _passing_df = passing_df.query('suspected_airplane_icao24 == "%s"'%airplane)
                    run = int(_passing_df['run'].to_numpy(dtype=int)[0])

                    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
                    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
                    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'

                    ds = dataSlicer([run], impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath)
                    ds.conference_mode = True


                    reader = Reader(datapath,run)

                    cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, deploy_index=deploy_index, map_source_distance_m=map_source_distance_m)
                    cor.conference_mode = True
                    if sine_subtract:
                        cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    cor.apply_filter = apply_filter
                    cor.apply_sine_subtract = apply_filter

                    eventids = _passing_df['eventid'].to_numpy(dtype=int)

                    eventids_dict = {run:eventids}

                    # ds.eventInspector(eventids_dict)


                    cor.conference_mode = True
                    for pol in ['all']:
                        if pandas.__version__ == '1.4.0' and True:
                            from tools.airplane_traffic_loader import getFileNamesFromTimestamps, enu2Spherical, addDirectionInformationToDataFrame, readPickle, getDataFrames
                            from tools.get_plane_tracks import plotAirplaneTrackerStatus
                            # Needs to match the version of pandas which was used to store airplane data.
                        
                            time_window_s = 1*60
                            plot_distance_cut_limit = 500
                            min_approach_cut_km = 1e6
                            if pol == 'hpol':
                                origin = cor.A0_latlonel_hpol
                            elif pol == 'vpol':
                                origin = cor.A0_latlonel_vpol
                            elif pol == 'all':
                                origin = cor.A0_latlonel_hpol

                            force_fit_order = 3#None

                            event_times = ds.getDataArrayFromParam('calibrated_trigtime',eventids_dict=eventids_dict)#cor.getEventTimes()[eventids]
                            start = min(event_times) - time_window_s/2
                            stop = max(event_times) +  time_window_s/2

                            files = getFileNamesFromTimestamps(start, stop, verbose=True)
                            df = getDataFrames(start, stop, query='up > 0 and azimuth >= -90 and azimuth <= 90 and zenith < 85')
                            minimum_approach = 1e10
                            minimum_approach_airplane = ''
                            minimum_approach_rpt = None
                            rpt_at_event_time = None
                            minimum_rpt_at_event_time = None
                            all_min_angular_distances = {}

                            try:
                                traj = df.query('icao24 == "%s" and distance < %f'%(airplane, plot_distance_cut_limit*1000))
                            except Exception as e:
                                print(e)
                                continue

                            if len(traj) == 0:
                                print('Skipping ', airplane)
                                continue
                            if force_fit_order is not None:
                                order = force_fit_order
                            else:
                                if len(traj) == 0:
                                    continue
                                elif len(traj) < 40:
                                    order = 3
                                elif len(traj) < 80:
                                    order = 5
                                else:
                                    order = 7
                            poly = pt.PlanePoly(traj['utc_timestamp'].to_numpy(),(traj['east'].to_numpy(),traj['north'].to_numpy(),traj['up'].to_numpy()),order=order,plot=False)
                            if poly.valid == True:
                                #Might fail for single data point, or similar
                                t = numpy.arange(float(min(traj['utc_timestamp'])), float(max(traj['utc_timestamp'])))

                                interpolated_airplane_locations = poly.poly(t)
                                rpt = enu2Spherical(interpolated_airplane_locations)
                            
                                rpe = numpy.copy(rpt)
                                rpe[:,2] = 90.0 - rpe[:,2]

                                expected_airplane_locations = poly.poly(ds.getDataArrayFromParam('calibrated_trigtime',eventids_dict=eventids_dict))
                                expected_rpt = enu2Spherical(expected_airplane_locations)
                            
                                expected_rpe = numpy.copy(expected_rpt)
                                expected_rpe[:,2] = 90.0 - expected_rpe[:,2]

                        if True:
                            cor.overwriteSourceDistance(numpy.mean(rpt[:,0]), verbose=False, suppress_time_delay_calculations=False, debug=False)
                            for hilbert in [True, False]:
                                


                                fig, (_gs_ax1, gs_ax2) = plt.subplots(1, 2, figsize=(20,12), gridspec_kw={'width_ratios': [3, 1]})

                                if mollweide == True:
                                    _gs_ax1.remove()
                                    _fig, (gs_ax1, _gs_ax2) = plt.subplots(1, 2, figsize=(20,12), gridspec_kw={'width_ratios': [3, 1]}, projection='mollweide')
                                    plt.close(_fig)
                                    gs_ax1 = fig.add_subplot(gs_ax1)
                                else:
                                    gs_ax1 = _gs_ax1


                                plt.sca(gs_ax1)
                                total_max_corr_values, airplane_phi_deg, airplane_theta_deg, fig, gs_ax1 = cor.multiEventMaxMap(eventids, pol, plot_map=True, hilbert=hilbert, max_method=None, mollweide=mollweide, zenith_cut_ENU=None,zenith_cut_array_plane=None, center_dir='E', fig=fig, ax=gs_ax1)
                                if mollweide:
                                    airplane_phi_deg = numpy.rad2deg(airplane_phi_deg)
                                    airplane_theta_deg = numpy.rad2deg(airplane_theta_deg)
                                airplane_el_deg = 90.0 - airplane_theta_deg

                                if False:
                                    #IF want to us preloaded data not presently calculated data.
                                    if True:
                                        airplane_phi_deg = ds.getDataArrayFromParam(eventids_dict, 'phi_best_choice')
                                        airplane_el_deg = ds.getDataArrayFromParam(eventids_dict, 'elevation_best_choice')
                                    else:
                                        if pol == 'all':
                                            airplane_phi_deg = ds.getDataArrayFromParam(eventids_dict, 'phi_best_choice')
                                            airplane_el_deg = ds.getDataArrayFromParam(eventids_dict, 'elevation_best_choice')
                                        else:
                                            airplane_phi_deg = ds.getDataArrayFromParam(eventids_dict, 'phi_best_%s'%pol[0])
                                            airplane_el_deg = ds.getDataArrayFromParam(eventids_dict, 'elevation_best_%s'%pol[0])

                                # gs_ax1.scatter(airplane_phi_deg,airplane_el_deg, s=4, c='k',label=('Event Map Peaks'))
                                
                                if pandas.__version__ == '1.4.0' and True:
                                    if poly.valid == True:
                                        if mollweide == True:
                                            rpe[:,1] = numpy.deg2rad(rpe[:,1])
                                            rpe[:,2] = numpy.deg2rad(rpe[:,2])
                                        gs_ax1.plot(rpe[:,1], rpe[:,2], linestyle = '-', c='k', alpha=1.0, label='Airplane Trajectory')

                                        el_f = scipy.interpolate.interp1d(rpe[:,1], rpe[:,2], kind='cubic')

                                    #plt.title('Airplane ICAO24 %s\n%i RF Triggered Events'%(airplane,len(eventids)))
                                    plt.legend(loc='upper right', fontsize=fontsize-2)
                                
                                # if mollweide == True:
                                #     gs_ax1.scatter(numpy.deg2rad(airplane_phi_deg),numpy.deg2ra(airplane_el_deg), s=4, c='k',label=('Event Map Peaks'))
                                # else:
                                #     gs_ax1.scatter(airplane_phi_deg,airplane_el_deg, s=4, c='k',label=('Event Map Peaks'))

                                if mollweide == True:
                                    for i in range(len(expected_rpe)):
                                        gs_ax1.plot(numpy.deg2rad([expected_rpe[:,1][i], airplane_phi_deg[i]]) , numpy.deg2rad([expected_rpe[:,2][i], airplane_el_deg[i]]), linestyle='-', lw=1, c='k',alpha=0.5)
                                    gs_ax1.scatter(numpy.deg2rad(expected_rpe[:,1]), numpy.deg2rad(expected_rpe[:,2]), c='k', label='Expected Location at Trigger Time', s=10)
                                    gs_ax1.scatter(numpy.deg2rad(airplane_phi_deg), numpy.deg2rad(airplane_el_deg), c='b', label='Measured Location at Trigger Time', s=10)
                                    gs_ax1.legend(loc='upper center')
                                    
                                    # gs_ax2.hist2d(airplane_phi_deg - expected_rpe[:,1] , airplane_el_deg - expected_rpe[:,2], bins=10, cmap='coolwarm')
                                    # gs_ax2.scatter(numpy.deg2rad(airplane_phi_deg - expected_rpe[:,1]) , numpy.deg2rad(airplane_el_deg - expected_rpe[:,2]), c='k')
                                    # gs_ax2.set_xlabel('Azimuth Residual (rad)')
                                    # gs_ax2.set_ylabel('Elevation Residual (rad)')
                                    gs_ax2.hist(numpy.deg2rad(airplane_el_deg - el_f(airplane_phi_deg)), bins=10, orientation='horizontal')
                                    gs_ax2.set_xlabel('Counts')
                                    gs_ax2.set_ylabel('Elevation Residual to Trajectory\nat Measured Azimuth (deg)')

                                else:
                                    for i in range(len(expected_rpe)):
                                        gs_ax1.plot([expected_rpe[:,1][i], airplane_phi_deg[i]] , [expected_rpe[:,2][i], airplane_el_deg[i]], linestyle='-', lw=1, c='k',alpha=0.5)
                                    gs_ax1.scatter(expected_rpe[:,1], expected_rpe[:,2], c='k', label='Expected Location at Trigger Time', s=10)
                                    gs_ax1.scatter(airplane_phi_deg, airplane_el_deg, c='b', label='Measured Location at Trigger Time', s=10)
                                    gs_ax1.legend(loc='upper center')
                                    
                                    
                                    gs_ax2.hist(airplane_el_deg - el_f(airplane_phi_deg), bins=10, orientation='horizontal')
                                    gs_ax2.set_xlabel('Counts')
                                    gs_ax2.set_ylabel('Elevation Residual to Curve\nat Measured Azimuth (deg)')
                                    

                                    # # gs_ax2.scatter(airplane_phi_deg,airplane_el_deg, s=4, c='k',label=('Filler Data'))

                                    # # gs_ax2.hist2d(airplane_phi_deg - expected_rpe[:,1] , airplane_el_deg - expected_rpe[:,2], bins=10, cmap='coolwarm')
                                    # gs_ax2.scatter(airplane_phi_deg - expected_rpe[:,1] , airplane_el_deg - expected_rpe[:,2], c='k')





                                plt.tight_layout()
                                if hilbert == True:
                                    if mollweide == True:
                                        fig.savefig('./figures/airplanes/%i_events_%s_%s_hilbert_mollweide_%i.pdf'%(len(eventids), pol, airplane, int(start_time)),dpi=300)
                                    else:
                                        fig.savefig('./figures/airplanes/%i_events_%s_%s_hilbert_%i.pdf'%(len(eventids), pol, airplane, int(start_time)),dpi=300)
                                else:
                                    if mollweide == True:
                                        fig.savefig('./figures/airplanes/%i_events_%s_%s_raw_mollweide_%i.pdf'%(len(eventids), pol, airplane, int(start_time)),dpi=300)
                                    else:
                                        fig.savefig('./figures/airplanes/%i_events_%s_%s_raw_%i.pdf'%(len(eventids), pol, airplane, int(start_time)),dpi=300)
                                plt.close(fig)

                        if True:
                            pass
                            #Write code to make histograms between best reconstruction directions (pre-saved in passed_df)
                            #and the trajectories


