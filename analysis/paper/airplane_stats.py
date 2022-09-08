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
from beacon.analysis.paper.rfi_resolution_plot import *

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

min_event_cut = 0


deploy_index = info.returnDefaultDeploy()
datapath = os.environ['BEACON_DATA']
map_source_distance_m = info.returnDefaultSourceDistance()
waveform_index_range = info.returnDefaultWaveformIndexRange()


def addStatsPlot(ax=None, major_fontsize=24, minor_fontsize=18, altitude_str='geoaltitude'):
    start_time = time.time()

    font = {'family' : 'normal',
        'size'   : minor_fontsize}

    matplotlib.rc('font', **font)

    args = copy.copy(sys.argv)
    infile = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated.xlsx')#

    map_reconstruction_extensions = ['h', 'v']

    plane_data = {}
    for map_pol in map_reconstruction_extensions:
        plane_data[map_pol] = {}
        with pd.ExcelFile(infile) as xls:
            for sheet_index, sheet_name in enumerate(xls.sheet_names):
                print(sheet_name)
                if sheet_name == 'good-with-airplane':

                    passing_df = xls.parse(sheet_name)
                    passing_df = passing_df.sort_values(by = ['run', 'eventid'], ascending = [True, True], na_position = 'last')

                    if False:
                        airplanes = ['a15c54']#['a4d5f0', 'a466fc', 'ad7828']
                    else:
                        airplanes = numpy.unique(passing_df['suspected_airplane_icao24'].to_numpy(dtype=str))

                    for airplane_index, airplane in enumerate(airplanes):
                        if len(airplane) != 6:
                            continue

                        # elif airplane != 'a219ea':
                        #     continue
                        # elif airplane != 'a15c54':
                        #     continue

                        _passing_df = passing_df.query('suspected_airplane_icao24 == "%s"'%airplane)
                        print('\n\nOn airplane %i/%i\nlen(_passing_df) = %i\n\n'%(airplane_index+1,len(airplanes), len(_passing_df)))

                        if len(_passing_df) < min_event_cut:
                            continue
                        else:
                            plane_data[map_pol][airplane] = {}
                            plane_data[map_pol][airplane]['eventids_dict'] = None
                            plane_data[map_pol][airplane]['poly'] = None
                            plane_data[map_pol][airplane]['expected_rpe'] = None
                            plane_data[map_pol][airplane]['measured_times'] = None
                            plane_data[map_pol][airplane]['measured_pe'] = None
                            plane_data[map_pol][airplane]['df'] = None

                        val,count = numpy.unique(_passing_df['run'],return_counts=True)
                        run = val[numpy.argmax(count)]

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



                        eventids = _passing_df.query('run == %i'%run)['eventid'].to_numpy(dtype=int)

                        eventids_dict = {run:eventids}

                        plane_data[map_pol][airplane]['eventids_dict'] = {run:eventids}

                        # ds.eventInspector(eventids_dict)


                        cor.conference_mode = True

                        if False:
                            meta = pd.DataFrame()
                            meta['runs'] = run*numpy.ones_like(plane_data[map_pol][airplane]['eventids_dict'][run])
                            meta['eventids'] = plane_data[map_pol][airplane]['eventids_dict'][run]
                            meta['measured_phi'] = plane_data[map_pol][airplane]['measured_pe'][:,0]
                            meta['measured_elevation'] = plane_data[map_pol][airplane]['measured_pe'][:,1]
                            meta.to_csv('./%s_eventids.csv'%airplane)

                        if map_pol == 'h':
                            pol = 'hpol'
                        elif map_pol == 'v':
                            pol = 'vpol'
                        elif map_pol == 'choice':
                            pol = 'all'

                        if pandas.__version__ == '1.4.0' and True:
                            from tools.airplane_traffic_loader import getFileNamesFromTimestamps, enu2Spherical, addDirectionInformationToDataFrame, readPickle, getDataFrames
                            from tools.get_plane_tracks import plotAirplaneTrackerStatus
                            # Needs to match the version of pandas which was used to store airplane data.
                        
                            time_window_s = 5*60
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

                            minimum_approach = 1e10
                            minimum_approach_airplane = ''
                            minimum_approach_rpt = None
                            rpt_at_event_time = None
                            minimum_rpt_at_event_time = None
                            all_min_angular_distances = {}


                            try:
                                if True:
                                    pickle_filename = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'paper', 'dataframes', altitude_str, airplane + '.pkl')
                                    
                                    if os.path.exists(pickle_filename):
                                        traj = pd.read_pickle(pickle_filename)
                                    else:
                                        df = getDataFrames(start, stop, query='up > 0 and azimuth >= -90 and azimuth <= 90 and zenith < 85', verbose=True, altitude_str = altitude_str) # Calls addDirectionInformationToDataFrame and adds ENU information.  
                                        traj = df.query('icao24 == "%s" and distance < %f'%(airplane, plot_distance_cut_limit*1000))
                                        traj.to_pickle(pickle_filename)

                                else:
                                    df = getDataFrames(start, stop, query='up > 0 and azimuth >= -90 and azimuth <= 90 and zenith < 85', verbose=True, altitude_str = altitude_str)
                                    traj = df.query('icao24 == "%s" and distance < %f'%(airplane, plot_distance_cut_limit*1000))
                                plane_data[map_pol][airplane]['df'] = traj
                            except Exception as e:
                                print(e)
                                import pdb; pdb.set_trace()
                                continue

                            # import pdb; pdb.set_trace()

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

                                event_times = ds.getDataArrayFromParam('calibrated_trigtime',eventids_dict=eventids_dict)
                                expected_airplane_locations = poly.poly(event_times)
                                expected_rpt = enu2Spherical(expected_airplane_locations)
                            
                                expected_rpe = numpy.copy(expected_rpt)
                                expected_rpe[:,2] = 90.0 - expected_rpe[:,2] 

                                plane_data[map_pol][airplane]['poly'] = poly
                                plane_data[map_pol][airplane]['expected_rpe'] = expected_rpe
                                plane_data[map_pol][airplane]['measured_times'] = event_times


                        if True:
                            expected_zenith = expected_rpt[:,2]
                            expected_azimuth = expected_rpt[:,1]
                            if map_pol == 'choice':
                                airplane_phi_deg = ds.getDataArrayFromParam('phi_best_%s'%map_pol,eventids_dict=eventids_dict)
                                airplane_el_deg = ds.getDataArrayFromParam('elevation_best_%s'%map_pol,eventids_dict=eventids_dict)
                                plane_data[map_pol][airplane]['measured_pe'] = numpy.array(list(zip(airplane_phi_deg,airplane_el_deg))) #Phi, elevation measured
                            else:
                                cor.overwriteSourceDistance(numpy.mean(rpt[:,0]), verbose=False, suppress_time_delay_calculations=False, debug=False)
                                # total_max_corr_values, airplane_phi_deg, airplane_theta_deg, fig, gs_ax1 = cor.multiEventMaxMap(eventids, pol, plot_map=True, hilbert=False, max_method=None, mollweide=False, zenith_cut_ENU=None,zenith_cut_array_plane=None, center_dir='E', fig=fig, ax=gs_ax1, expected_zenith=expected_zenith, expected_azimuth=expected_azimuth)
                                total_max_corr_values, airplane_phi_deg, airplane_theta_deg = cor.multiEventMaxMap(eventids, pol, plot_map=False, hilbert=False, max_method=None, mollweide=False, zenith_cut_ENU=None,zenith_cut_array_plane=None, center_dir='E', fig=None, ax=None, expected_zenith=expected_zenith, expected_azimuth=expected_azimuth)

                                airplane_el_deg = 90.0 - airplane_theta_deg
                                plane_data[map_pol][airplane]['measured_pe'] = numpy.array(list(zip(airplane_phi_deg,airplane_el_deg))) #Phi, elevation measured

    if ax is None:
        fig1 = plt.figure()
        ax = plt.subplot(1,1,1, aspect='equal')
    plt.sca(ax)
    ax.set_xticks(ticks=[-3,0,3])
    ax.set_yticks(ticks=[-3,0,3])
    ax.tick_params(axis="y",direction="in", pad=-40)
    ax.tick_params(axis="x",direction="in", pad=-25)
    ax.xaxis.set_tick_params(labelsize=minor_fontsize-4)
    ax.yaxis.set_tick_params(labelsize=minor_fontsize-4)
    ax.grid(True)
    # ax.axhline(0.0, c='k', linestyle='--', alpha=0.4)
    # ax.axvline(0.0, c='k', linestyle='--', alpha=0.4)
    plt.xlabel('Azimuth Residual (deg)', fontsize=major_fontsize)
    plt.ylabel('Elevation Residual (deg)', fontsize=major_fontsize)

    colors = ['dodgerblue', 'red', 'k']

    for map_pol_index, map_pol in enumerate(map_reconstruction_extensions):

        if map_pol == 'h':
            pol = 'HPol'
        elif map_pol == 'v':
            pol = 'VPol'
        elif map_pol == 'choice':
            pol = 'Choice'

        residuals_elevation = []
        expected_elevation = []
        elevation = []
        residuals_phi = []
        expected_phi = []
        phi = []
        runs = []
        eventids = []
        event_times = []
        for airplane_index, airplane in enumerate(list(plane_data[map_pol].keys())):
            residuals_elevation.append(plane_data[map_pol][airplane]['measured_pe'][:,1] - plane_data[map_pol][airplane]['expected_rpe'][:,2])
            expected_elevation.append(plane_data[map_pol][airplane]['expected_rpe'][:,2])
            elevation.append(plane_data[map_pol][airplane]['measured_pe'][:,1])
            residuals_phi.append(plane_data[map_pol][airplane]['measured_pe'][:,0] - plane_data[map_pol][airplane]['expected_rpe'][:,1])
            expected_phi.append(plane_data[map_pol][airplane]['expected_rpe'][:,1])
            phi.append(plane_data[map_pol][airplane]['measured_pe'][:,0])
            run = list(plane_data[map_pol][airplane]['eventids_dict'].keys())[0]
            event_times.append(plane_data[map_pol][airplane]['measured_times'])

            runs.append(numpy.ones(len(plane_data[map_pol][airplane]['measured_pe'][:,0]))*run)
            eventids.append(plane_data[map_pol][airplane]['eventids_dict'][run])

        residuals_phi = numpy.concatenate(residuals_phi)
        expected_phi = numpy.concatenate(expected_phi)
        phi = numpy.concatenate(phi)
        residuals_elevation = numpy.concatenate(residuals_elevation)
        expected_elevation = numpy.concatenate(expected_elevation)
        elevation = numpy.concatenate(elevation)
        runs = numpy.concatenate(runs)
        eventids = numpy.concatenate(eventids)
        event_times = numpy.concatenate(event_times)
        event_times_days = (event_times - numpy.min(event_times))/(24*3600)

        guess_phi = 0
        guess_elevation = -1.5#-1.89
        outlier_atol = 5

        outliers_cut = numpy.logical_or(numpy.abs(residuals_phi - guess_phi) > outlier_atol, numpy.abs(residuals_elevation - guess_elevation) > outlier_atol)
        outliers_runs = runs[outliers_cut]
        outliers_eventids = eventids[outliers_cut]

        outliers = numpy.asarray(list(zip(outliers_runs,outliers_eventids)), dtype=int)

        if False:
            residuals_phi = residuals_phi[~outliers_cut]
            expected_phi = expected_phi[~outliers_cut]
            phi = phi[~outliers_cut]
            residuals_elevation = residuals_elevation[~outliers_cut]
            expected_elevation = expected_elevation[~outliers_cut]
            elevation = elevation[~outliers_cut]


        phi_bins = numpy.linspace(-20,20,400)
        phi_bin_centers = (phi_bins[1:] + phi_bins[:-1]) / 2
        el_bins = numpy.linspace(-20,20,400)
        el_bin_centers = (el_bins[1:] + el_bins[:-1]) / 2

        _fig = plt.figure()
        _ax = plt.gca()
        counts, xedges, yedges, im = plt.hist2d(residuals_phi, residuals_elevation, bins=[phi_bins, el_bins])
        plt.close(_fig)
        plt.sca(ax)
        counts = counts.T
        x, y = numpy.meshgrid(phi_bin_centers, el_bin_centers)

        ax.scatter(residuals_phi, residuals_elevation, alpha=0.5, c=colors[map_pol_index], s=1)

        #                       bivariateGaus(xy ,                                                                                                                          x0, sigma_x,    y0,     sigma_y,    rho, scale_factor=1.0, return_2d=False)
        popt, pcov = curve_fit(bivariateGaus,(x.ravel(), y.ravel()), counts.ravel() / (numpy.sum(counts.ravel())*numpy.diff(x,axis=1)[0][0]*numpy.diff(y,axis=0)[0][0]),p0=[0,  2,          -2,     2,          0])

        x0          = popt[0]
        sigma_x     = popt[1]
        y0          = popt[2]
        sigma_y     = popt[3]
        rho         = popt[4]
        mean = numpy.array([x0,y0])
        sigma = numpy.array([[sigma_x**2, rho*sigma_x*sigma_y],[rho*sigma_x*sigma_y,sigma_y**2]])

        confidence_integral_value = 0.9
        
        ellipse_vertices = parametricCovarianceEllipse(sigma, mean, confidence_integral_value, n=1000)
        ellipse_path = matplotlib.path.Path(ellipse_vertices)
        ellipse_area = contourArea(ellipse_path.vertices) #square degrees

        scale_factor = (numpy.sum(counts)*numpy.diff(x,axis=1)[0][0]*numpy.diff(y,axis=0)[0][0])
        fit_z = bivariateGaus( (x, y) ,popt[0], popt[1], popt[2], popt[3], popt[4], scale_factor = scale_factor, return_2d=True)

        confidence_integral_value = 0.9
        ellipse_vertices = parametricCovarianceEllipse(sigma, mean, confidence_integral_value, n=1000)
        ellipse_path = matplotlib.path.Path(ellipse_vertices)
        ellipse_area = contourArea(ellipse_path.vertices) #square degrees

        ax.plot(ellipse_vertices[:,0],ellipse_vertices[:,1], color=colors[map_pol_index],label=pol + ' %i'%(confidence_integral_value*100) + r'%' + ' Fit Area\n= %0.2f deg^2'%(ellipse_area))

        print(pol)
        print('x0 :         fit  %0.4f'%(x0))
        print('sigma_x :    fit  %0.4f'%(sigma_x))
        print('y0 :         fit  %0.4f'%(y0))
        print('sigma_y :    fit  %0.4f'%(sigma_y))
        print('rho :        fit  %0.4f'%(rho))
        print(r'90% Confidence = ' + '%0.4f deg^2'%(ellipse_area))
        print('pcov = ')
        print(pcov)

        ax.set_ylim(-4.5,4.5)
        ax.set_xlim(-4.5,4.5)

        plt.sca(ax)
        plt.legend(loc='upper right', title='All Airplane Residuals', fontsize = minor_fontsize-4 , prop={'weight':'normal'})
        plt.tight_layout()

    return ax



if __name__ == '__main__':

    start_time = time.time()
    # plt.close('all')

    major_fontsize = 24
    minor_fontsize = 16

    font = {'family' : 'normal',
        'size'   : major_fontsize}

    matplotlib.rc('font', **font)

    args = copy.copy(sys.argv)
    infile = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated.xlsx')#

    if True:
        altitude_str = 'geoaltitude'
    else:
        altitude_str = 'altitude'

    map_reconstruction_extensions = ['h', 'v']#'choice', 

    plane_data = {}
    for map_pol in map_reconstruction_extensions:
        plane_data[map_pol] = {}
        with pd.ExcelFile(infile) as xls:
            for sheet_index, sheet_name in enumerate(xls.sheet_names):
                print(sheet_name)
                if sheet_name == 'good-with-airplane':

                    passing_df = xls.parse(sheet_name)
                    passing_df = passing_df.sort_values(by = ['run', 'eventid'], ascending = [True, True], na_position = 'last')

                    if False:
                        airplanes = ['a15c54']#['a4d5f0', 'a466fc', 'ad7828']
                    else:
                        airplanes = numpy.unique(passing_df['suspected_airplane_icao24'].to_numpy(dtype=str))

                    for airplane_index, airplane in enumerate(airplanes):
                        if len(airplane) != 6:
                            continue



                        # elif airplane != 'a219ea':
                        #     continue
                        # elif airplane != 'a15c54':
                        #     continue




                        _passing_df = passing_df.query('suspected_airplane_icao24 == "%s"'%airplane)
                        print('\n\nOn airplane %i/%i\nlen(_passing_df) = %i\n\n'%(airplane_index+1,len(airplanes), len(_passing_df)))

                        if len(_passing_df) < min_event_cut:
                            continue
                        else:
                            plane_data[map_pol][airplane] = {}
                            plane_data[map_pol][airplane]['eventids_dict'] = None
                            plane_data[map_pol][airplane]['poly'] = None
                            plane_data[map_pol][airplane]['expected_rpe'] = None
                            plane_data[map_pol][airplane]['measured_times'] = None
                            plane_data[map_pol][airplane]['measured_pe'] = None
                            plane_data[map_pol][airplane]['df'] = None

                        val,count = numpy.unique(_passing_df['run'],return_counts=True)
                        run = val[numpy.argmax(count)]

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



                        eventids = _passing_df.query('run == %i'%run)['eventid'].to_numpy(dtype=int)

                        eventids_dict = {run:eventids}

                        plane_data[map_pol][airplane]['eventids_dict'] = {run:eventids}

                        # ds.eventInspector(eventids_dict)


                        cor.conference_mode = True

                        if False:
                            meta = pd.DataFrame()
                            meta['runs'] = run*numpy.ones_like(plane_data[map_pol][airplane]['eventids_dict'][run])
                            meta['eventids'] = plane_data[map_pol][airplane]['eventids_dict'][run]
                            meta['measured_phi'] = plane_data[map_pol][airplane]['measured_pe'][:,0]
                            meta['measured_elevation'] = plane_data[map_pol][airplane]['measured_pe'][:,1]
                            meta.to_csv('./%s_eventids.csv'%airplane)

                        if map_pol == 'h':
                            pol = 'hpol'
                        elif map_pol == 'v':
                            pol = 'vpol'
                        elif map_pol == 'choice':
                            pol = 'all'

                        if pandas.__version__ == '1.4.0' and True:
                            from tools.airplane_traffic_loader import getFileNamesFromTimestamps, enu2Spherical, addDirectionInformationToDataFrame, readPickle, getDataFrames
                            from tools.get_plane_tracks import plotAirplaneTrackerStatus
                            # Needs to match the version of pandas which was used to store airplane data.
                        
                            time_window_s = 5*60
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

                            minimum_approach = 1e10
                            minimum_approach_airplane = ''
                            minimum_approach_rpt = None
                            rpt_at_event_time = None
                            minimum_rpt_at_event_time = None
                            all_min_angular_distances = {}

                            try:
                                if True:
                                    pickle_filename = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'paper', 'dataframes', altitude_str, airplane + '.pkl')
                                    
                                    if os.path.exists(pickle_filename):
                                        traj = pd.read_pickle(pickle_filename)
                                    else:
                                        df = getDataFrames(start, stop, query='up > 0 and azimuth >= -90 and azimuth <= 90 and zenith < 85', verbose=True, altitude_str = altitude_str) # Calls addDirectionInformationToDataFrame and adds ENU information.  
                                        traj = df.query('icao24 == "%s" and distance < %f'%(airplane, plot_distance_cut_limit*1000))
                                        traj.to_pickle(pickle_filename)

                                else:
                                    df = getDataFrames(start, stop, query='up > 0 and azimuth >= -90 and azimuth <= 90 and zenith < 85', verbose=True, altitude_str = altitude_str)
                                    traj = df.query('icao24 == "%s" and distance < %f'%(airplane, plot_distance_cut_limit*1000))
                                plane_data[map_pol][airplane]['df'] = traj
                            except Exception as e:
                                print(e)
                                import pdb; pdb.set_trace()
                                continue

                            # import pdb; pdb.set_trace()

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

                                event_times = ds.getDataArrayFromParam('calibrated_trigtime',eventids_dict=eventids_dict)
                                expected_airplane_locations = poly.poly(event_times)
                                expected_rpt = enu2Spherical(expected_airplane_locations)
                            
                                expected_rpe = numpy.copy(expected_rpt)
                                expected_rpe[:,2] = 90.0 - expected_rpe[:,2] 

                                plane_data[map_pol][airplane]['poly'] = poly
                                plane_data[map_pol][airplane]['expected_rpe'] = expected_rpe
                                plane_data[map_pol][airplane]['measured_times'] = event_times


                        if True:
                            expected_zenith = expected_rpt[:,2]
                            expected_azimuth = expected_rpt[:,1]
                            if map_pol == 'choice':
                                airplane_phi_deg = ds.getDataArrayFromParam('phi_best_%s'%map_pol,eventids_dict=eventids_dict)
                                airplane_el_deg = ds.getDataArrayFromParam('elevation_best_%s'%map_pol,eventids_dict=eventids_dict)
                                plane_data[map_pol][airplane]['measured_pe'] = numpy.array(list(zip(airplane_phi_deg,airplane_el_deg))) #Phi, elevation measured
                            else:
                                cor.overwriteSourceDistance(numpy.mean(rpt[:,0]), verbose=False, suppress_time_delay_calculations=False, debug=False)
                                # total_max_corr_values, airplane_phi_deg, airplane_theta_deg, fig, gs_ax1 = cor.multiEventMaxMap(eventids, pol, plot_map=True, hilbert=False, max_method=None, mollweide=False, zenith_cut_ENU=None,zenith_cut_array_plane=None, center_dir='E', fig=fig, ax=gs_ax1, expected_zenith=expected_zenith, expected_azimuth=expected_azimuth)
                                total_max_corr_values, airplane_phi_deg, airplane_theta_deg = cor.multiEventMaxMap(eventids, pol, plot_map=False, hilbert=False, max_method=None, mollweide=False, zenith_cut_ENU=None,zenith_cut_array_plane=None, center_dir='E', fig=None, ax=None, expected_zenith=expected_zenith, expected_azimuth=expected_azimuth)

                                airplane_el_deg = 90.0 - airplane_theta_deg
                                plane_data[map_pol][airplane]['measured_pe'] = numpy.array(list(zip(airplane_phi_deg,airplane_el_deg))) #Phi, elevation measured


    plt.close('all')
    individual = False

    if individual == False:
        fig1 = plt.figure()
        ax_b = plt.subplot(1,1,1)
        ax_b.axhline(0.0, c='k', linestyle='--', alpha=0.5)
        ax_b.axvline(0.0, c='k', linestyle='--', alpha=0.5)
        plt.xlabel('Azimuth Residual\n(deg)')
        plt.ylabel('Elevation Residual\n(deg)')

        fig2 = plt.figure()
        ax_1 = plt.subplot(3,2,1)
        plt.xlabel('Expected Azimuth\n(deg)')
        plt.ylabel('Azimuth Residual\n(deg)')
        ax_2 = plt.subplot(3,2,2,sharex=ax_1)
        plt.xlabel('Expected Azimuth\n(deg)')
        plt.ylabel('Elevation Residual\n(deg)')
        ax_3 = plt.subplot(3,2,3,sharey=ax_1)
        plt.xlabel('Expected Elevation\n(deg)')
        plt.ylabel('Azimuth Residual\n(deg)')
        ax_4 = plt.subplot(3,2,4,sharex=ax_3, sharey=ax_2)
        plt.xlabel('Expected Elevation\n(deg)')
        plt.ylabel('Elevation Residual\n(deg)')
        ax_5 = plt.subplot(3,2,5,sharey=ax_1)
        plt.xlabel('Days from Start')
        plt.ylabel('Azimuth Residual\n(deg)')
        ax_6 = plt.subplot(3,2,6,sharex=ax_5, sharey=ax_2)
        plt.xlabel('Days from Start')
        plt.ylabel('Elevation Residual\n(deg)')

        ax_1.axhline(0.0, c='k', linestyle='--', alpha=0.5)
        ax_2.axhline(0.0, c='k', linestyle='--', alpha=0.5)
        ax_3.axhline(0.0, c='k', linestyle='--', alpha=0.5)
        ax_4.axhline(0.0, c='k', linestyle='--', alpha=0.5)
        ax_5.axhline(0.0, c='k', linestyle='--', alpha=0.5)
        ax_6.axhline(0.0, c='k', linestyle='--', alpha=0.5)


    colors = ['dodgerblue', 'red', 'k']

    for map_pol_index, map_pol in enumerate(map_reconstruction_extensions):

        if map_pol == 'h':
            pol = 'HPol'
        elif map_pol == 'v':
            pol = 'VPol'
        elif map_pol == 'choice':
            pol = 'Choice'

        if individual == True:
            fig1 = plt.figure()
            ax_a = plt.subplot(1,2,1)
            plt.xlabel('Azimuth Residual\n(deg)\n%s'%pol)
            plt.ylabel('Elevation Residual\n(deg)\n%s'%pol)
            ax_b = plt.subplot(1,2,2,sharex=ax_a, sharey=ax_a)
            plt.xlabel('Azimuth Residual\n(deg)\n%s'%pol)
            plt.ylabel('Elevation Residual\n(deg)\n%s'%pol)

            fig2 = plt.figure()
            ax_1 = plt.subplot(3,2,1)
            plt.xlabel('Expected Azimuth\n(deg)\n%s'%pol)
            plt.ylabel('Azimuth Residual\n(deg)\n%s'%pol)
            ax_2 = plt.subplot(3,2,2,sharex=ax_1)
            plt.xlabel('Expected Azimuth\n(deg)\n%s'%pol)
            plt.ylabel('Elevation Residual\n(deg)\n%s'%pol)
            ax_3 = plt.subplot(3,2,3,sharey=ax_1)
            plt.xlabel('Expected Elevation\n(deg)\n%s'%pol)
            plt.ylabel('Azimuth Residual\n(deg)\n%s'%pol)
            ax_4 = plt.subplot(3,2,4,sharex=ax_3, sharey=ax_2)
            plt.xlabel('Expected Elevation\n(deg)\n%s'%pol)
            plt.ylabel('Elevation Residual\n(deg)\n%s'%pol)
            ax_5 = plt.subplot(3,2,5,sharey=ax_1)
            plt.xlabel('Days from Start')
            plt.ylabel('Azimuth Residual\n(deg)\n%s'%pol)
            ax_6 = plt.subplot(3,2,6,sharex=ax_5, sharey=ax_2)
            plt.xlabel('Days from Start')
            plt.ylabel('Elevation Residual\n(deg)\n%s'%map_pol)

            ax_1.axhline(0.0, c='k', linestyle='--', alpha=0.5)
            ax_2.axhline(0.0, c='k', linestyle='--', alpha=0.5)
            ax_3.axhline(0.0, c='k', linestyle='--', alpha=0.5)
            ax_4.axhline(0.0, c='k', linestyle='--', alpha=0.5)
            ax_5.axhline(0.0, c='k', linestyle='--', alpha=0.5)
            ax_6.axhline(0.0, c='k', linestyle='--', alpha=0.5)
        else:
            fig = plt.figure()
            ax_a = plt.subplot(1,1,1,sharex=ax_b, sharey=ax_b)
            plt.xlabel('Azimuth Residual\n(deg)')
            plt.ylabel('Elevation Residual\n(deg)')

        residuals_elevation = []
        expected_elevation = []
        elevation = []
        residuals_phi = []
        expected_phi = []
        phi = []
        runs = []
        eventids = []
        event_times = []
        for airplane_index, airplane in enumerate(list(plane_data[map_pol].keys())):
            residuals_elevation.append(plane_data[map_pol][airplane]['measured_pe'][:,1] - plane_data[map_pol][airplane]['expected_rpe'][:,2])
            expected_elevation.append(plane_data[map_pol][airplane]['expected_rpe'][:,2])
            elevation.append(plane_data[map_pol][airplane]['measured_pe'][:,1])
            residuals_phi.append(plane_data[map_pol][airplane]['measured_pe'][:,0] - plane_data[map_pol][airplane]['expected_rpe'][:,1])
            expected_phi.append(plane_data[map_pol][airplane]['expected_rpe'][:,1])
            phi.append(plane_data[map_pol][airplane]['measured_pe'][:,0])
            run = list(plane_data[map_pol][airplane]['eventids_dict'].keys())[0]
            event_times.append(plane_data[map_pol][airplane]['measured_times'])

            runs.append(numpy.ones(len(plane_data[map_pol][airplane]['measured_pe'][:,0]))*run)
            eventids.append(plane_data[map_pol][airplane]['eventids_dict'][run])

        residuals_phi = numpy.concatenate(residuals_phi)
        expected_phi = numpy.concatenate(expected_phi)
        phi = numpy.concatenate(phi)
        residuals_elevation = numpy.concatenate(residuals_elevation)
        expected_elevation = numpy.concatenate(expected_elevation)
        elevation = numpy.concatenate(elevation)
        runs = numpy.concatenate(runs)
        eventids = numpy.concatenate(eventids)
        event_times = numpy.concatenate(event_times)
        event_times_days = (event_times - numpy.min(event_times))/(24*3600)

        guess_phi = 0
        guess_elevation = -1.5#-1.89
        outlier_atol = 5

        outliers_cut = numpy.logical_or(numpy.abs(residuals_phi - guess_phi) > outlier_atol, numpy.abs(residuals_elevation - guess_elevation) > outlier_atol)
        outliers_runs = runs[outliers_cut]
        outliers_eventids = eventids[outliers_cut]

        outliers = numpy.asarray(list(zip(outliers_runs,outliers_eventids)), dtype=int)

        if False:
            residuals_phi = residuals_phi[~outliers_cut]
            expected_phi = expected_phi[~outliers_cut]
            phi = phi[~outliers_cut]
            residuals_elevation = residuals_elevation[~outliers_cut]
            expected_elevation = expected_elevation[~outliers_cut]
            elevation = elevation[~outliers_cut]


        phi_bins = numpy.linspace(-20,20,400)
        phi_bin_centers = (phi_bins[1:] + phi_bins[:-1]) / 2
        el_bins = numpy.linspace(-20,20,400)
        el_bin_centers = (el_bins[1:] + el_bins[:-1]) / 2


        counts, xedges, yedges, im = ax_a.hist2d(residuals_phi, residuals_elevation, bins=[phi_bins, el_bins])
        counts = counts.T
        x, y = numpy.meshgrid(phi_bin_centers, el_bin_centers)

        ax_b.scatter(residuals_phi, residuals_elevation, alpha=0.5, c=colors[map_pol_index], s=1)
        # ax_b.contour(phi_bin_centers, el_bin_centers, counts.T, c=colors[map_pol_index], levels=3)

        ax_b.set_ylim(-6,2)
        ax_b.set_xlim(-4,4)

        popt, pcov = curve_fit(bivariateGaus,(x.ravel(), y.ravel()), counts.ravel() / (numpy.sum(counts.ravel())*numpy.diff(x,axis=1)[0][0]*numpy.diff(y,axis=0)[0][0]),p0=[0,2,-2,2,0])

        x0          = popt[0]
        sigma_x     = popt[1]
        y0          = popt[2]
        sigma_y     = popt[3]
        rho         = popt[4]
        mean = numpy.array([x0,y0])
        sigma = numpy.array([[sigma_x**2, rho*sigma_x*sigma_y],[rho*sigma_x*sigma_y,sigma_y**2]])

        confidence_integral_value = 0.9
        
        ellipse_vertices = parametricCovarianceEllipse(sigma, mean, confidence_integral_value, n=1000)
        ellipse_path = matplotlib.path.Path(ellipse_vertices)
        ellipse_area = contourArea(ellipse_path.vertices) #square degrees

        scale_factor = (numpy.sum(counts)*numpy.diff(x,axis=1)[0][0]*numpy.diff(y,axis=0)[0][0])
        fit_z = bivariateGaus( (x, y) ,popt[0], popt[1], popt[2], popt[3], popt[4], scale_factor = scale_factor, return_2d=True)

        confidence_integral_value = 0.9
        ellipse_vertices = parametricCovarianceEllipse(sigma, mean, confidence_integral_value, n=1000)
        ellipse_path = matplotlib.path.Path(ellipse_vertices)
        ellipse_area = contourArea(ellipse_path.vertices) #square degrees

        ax_b.plot(ellipse_vertices[:,0],ellipse_vertices[:,1], color=colors[map_pol_index],label=pol + ' %i'%(confidence_integral_value*100) + r'%' + ' Fit Area\n= %0.2f deg^2'%(ellipse_area))

        print(pol)
        print('x0 :         fit  %0.4f'%(x0))
        print('sigma_x :    fit  %0.4f'%(sigma_x))
        print('y0 :         fit  %0.4f'%(y0))
        print('sigma_y :    fit  %0.4f'%(sigma_y))
        print('rho :        fit  %0.4f'%(rho))
        print(r'90% Confidence = ' + '%0.4f deg^2'%(ellipse_area))
        print('pcov = ')
        print(pcov)

        ax_1.scatter(phi, residuals_phi, alpha=0.5, c=colors[map_pol_index], s=1, label=pol)
        ax_2.scatter(phi, residuals_elevation, alpha=0.5, c=colors[map_pol_index], s=1)
        ax_3.scatter(elevation, residuals_phi, alpha=0.5, c=colors[map_pol_index], s=1)
        ax_4.scatter(elevation, residuals_elevation, alpha=0.5, c=colors[map_pol_index], s=1)
        ax_5.scatter(event_times_days, residuals_phi, alpha=0.5, c=colors[map_pol_index], s=1)
        ax_6.scatter(event_times_days, residuals_elevation, alpha=0.5, c=colors[map_pol_index], s=1)


        ax_1.set_ylim(-4.5,4.5)
        ax_2.set_ylim(-4.5,4.5)
        ax_3.set_ylim(-4.5,4.5)
        ax_4.set_ylim(-4.5,4.5)
        ax_5.set_ylim(-4.5,4.5)
        ax_6.set_ylim(-4.5,4.5)

        ax_b.set_ylim(-4.5,4.5)
        ax_b.set_xlim(-4.5,4.5)

        plt.sca(ax_1)
        plt.legend(loc='upper right', fontsize=18)
        plt.tight_layout()
        plt.sca(ax_b)
        plt.legend(loc='upper right', fontsize=18)
        plt.tight_layout()

