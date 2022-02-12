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

import numpy
import scipy
import scipy.signal
import time

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.flipbook_reader import flipbookToDict
import  beacon.tools.get_plane_tracks as pt
from tools.airplane_traffic_loader import getDataFrames, getFileNamesFromTimestamps

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)


def enu2Spherical(enu):
    '''
    2d array like ((e_0, n_0, u_0), (e_1, n_1, u_1), ... , (e_i, n_i, u_i))

    Return in degrees
    '''
    r = numpy.linalg.norm(enu, axis=1)
    theta = numpy.degrees(numpy.arccos(enu[:,2]/r))
    phi = numpy.degrees(numpy.arctan2(enu[:,1],enu[:,0]))
    # import pdb; pdb.set_trace()
    return numpy.vstack((r,phi,theta)).T



    


if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length

    # _runs = numpy.arange(5733,5974)#[0:100]
    # bad_runs = numpy.array([5775])
    #_runs = numpy.arange(5733,5800)

    flipbook_path = '/home/dsouthall/scratch-midway2/event_flipbook_1643154940'#'/home/dsouthall/scratch-midway2/event_flipbook_1642725413'
    sorted_dict = flipbookToDict(flipbook_path)
    
    if True:
        #good_dict = sorted_dict['very-good']['eventids_dict']
        good_dict = sorted_dict['maybe']['eventids_dict']
        # maybe_dict = sorted_dict['maybe']['eventids_dict']
        # bad_dict = sorted_dict['bad']['eventids_dict']
        runs = list(good_dict.keys())
    elif False:
        good_dict = {5844:[62626]}
        runs = numpy.array([list(good_dict.keys())[0]])
    elif False:
        good_dict = {5805:[11079]}
        runs = numpy.array([list(good_dict.keys())[0]])
    else:
        good_dict = {5903:[86227]}
        runs = numpy.array([5903])

    force_fit_order = 3 #None to use varying order
    save_figures = True
    if save_figures == True:
        outpath = './airplane_event_flipbook_%i'%time.time() 
        os.mkdir(outpath)
    use_old_code = False
    use_new_code = True

    print("Preparing dataSlicer")

    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)


    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                    n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)

    ds.prepareCorrelator()

    time_window_s = 5*60
    plot_distance_cut_limit = 500
    min_approach_cut_km = 1e6
    origin = ds.cor.A0_latlonel_hpol

    elevation_best_choice = ds.getDataFromParam(good_dict, 'elevation_best_choice')
    phi_best_choice = ds.getDataFromParam(good_dict, 'phi_best_choice')


    for run in runs:
        run_index = numpy.where(numpy.isin(ds.runs, run))[0][0]
        if ds.cor.reader.run != run:
            ds.cor.setReader(ds.data_slicers[run_index].reader, verbose=False)

        eventids = good_dict[run]
        event_times = ds.cor.getEventTimes()[eventids]

        for event_index, eventid in enumerate(eventids):
            start = event_times[event_index] - time_window_s/2
            stop = event_times[event_index] +  time_window_s/2
            flight_tracks_ENU, all_vals = pt.getENUTrackDict(start,stop,min_approach_cut_km,hour_window=2,flights_of_interest=[],origin=origin)



            best_elevation = float(elevation_best_choice[run][numpy.array(good_dict[run]) == eventid][0])
            best_phi = float(phi_best_choice[run][numpy.array(good_dict[run]) == eventid][0])

            ds.cor.reader.setEntry(eventid)

            m, fig, ax = ds.cor.map(eventid, 'hpol', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=None, plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=None, verbose=False, mollweide=False, zenith_cut_ENU=None, zenith_cut_array_plane=(0,90), center_dir='E', circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=True, circle_map_max=False, override_to_time_window=(None,None))
            ax.set_xlim(-90,90)
            ax.set_ylim(-40,90)
            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = ds.cor.mapMax(m, max_method=0, verbose=False, zenith_cut_ENU=[0,80], zenith_cut_array_plane=[0,90], pol='hpol', return_peak_to_sidelobe=False, theta_cut=None)

            if use_old_code == True:
                ax, airplane_direction_dict = ds.cor.addAirplanesToMap([eventid], 'hpol', ax, azimuth_offset_deg=0.0, mollweide=False, radius = 1.0, crosshair=False, color='r', min_approach_cut_km=min_approach_cut_km,plot_distance_cut_limit=plot_distance_cut_limit, time_window_s=time_window_s)
                airplanes = []
                apd = numpy.dtype([('airplane',numpy.asarray(list(airplane_direction_dict.keys())).dtype),('ang_dist','f')])
                airplanes = numpy.array([],dtype=apd)
                for airplane in list(airplane_direction_dict.keys()):
                    if 'zenith' in list(airplane_direction_dict[airplane].keys()):
                        dist_angle = numpy.sqrt((theta_best - airplane_direction_dict[airplane]['zenith'])**2 + (phi_best - airplane_direction_dict[airplane]['azimuth'])**2)
                        airplanes = numpy.append(airplanes, numpy.asarray([(airplane, min(dist_angle))],dtype=apd))

                minimum_approach = 1e10
                minimum_approach_airplane = ''
                for airplane_key in list(flight_tracks_ENU.keys()):
                    original_norms = numpy.sqrt(flight_tracks_ENU[airplane_key][:,0]**2 + flight_tracks_ENU[airplane_key][:,1]**2 + flight_tracks_ENU[airplane_key][:,2]**2 )
                    cut = numpy.logical_and(original_norms/500.0 < plot_distance_cut_limit,numpy.logical_and(flight_tracks_ENU[airplane_key][:,3] >= start ,flight_tracks_ENU[airplane_key][:,3] <= stop))
                    
                    if force_fit_order is not None:
                        order = force_fit_order
                    else:
                        if sum(cut) == 0:
                            continue
                        elif sum(cut) < 40:
                            order = 3
                        elif sum(cut) < 80:
                            order = 5
                        else:
                            order = 7

                    poly = pt.PlanePoly(flight_tracks_ENU[airplane_key][cut,3],(flight_tracks_ENU[airplane_key][cut,0],flight_tracks_ENU[airplane_key][cut,1],flight_tracks_ENU[airplane_key][cut,2]),order=order,plot=False)

                    t = numpy.arange(min(flight_tracks_ENU[airplane_key][:,3][cut]), max(flight_tracks_ENU[airplane_key][:,3][cut]))

                    interpolated_airplane_locations = poly.poly(t)
                    rpt = enu2Spherical(interpolated_airplane_locations)

                    approach_angle = numpy.sqrt((rpt[:,1] - best_phi)**2 + (rpt[:,2] - (90.0 - best_elevation ))**2)
                    minimum_approach_index = numpy.argmin(approach_angle)
                    if approach_angle[minimum_approach_index] < minimum_approach:
                        minimum_approach = approach_angle[minimum_approach_index]
                        minimum_approach_t = t[minimum_approach_index]
                        minimum_approach_rpt = rpt[minimum_approach_index,:]
                        minimum_approach_airplane = airplane_key

                print('run = ', run)
                print('eventid = ', eventid)
                print('minimum_approach = ', minimum_approach, ' deg')
                print('minimum_approach_airplane = ', minimum_approach_airplane)


                if len(airplanes) != 0:
                    min_airplane = airplanes[numpy.argmin(airplanes['ang_dist'])]
                    for airplane_key in numpy.unique(airplanes['airplane']):
                        if ~numpy.isin(airplane_key, list(flight_tracks_ENU.keys())):
                            print('airplane_direction_dict.keys()', airplane_direction_dict.keys())
                            print('flight_tracks_ENU.keys()',flight_tracks_ENU.keys())
                            print(start)
                            print(stop)
                            print(min_approach_cut_km)
                            print(origin)
                            ax, airplane_direction_dict = ds.cor.addAirplanesToMap([eventid], 'hpol', ax, azimuth_offset_deg=0.0, mollweide=False, radius = 1.0, crosshair=False, color='r', min_approach_cut_km=min_approach_cut_km,plot_distance_cut_limit=plot_distance_cut_limit, time_window_s=time_window_s, debug=True)
                        if len(flight_tracks_ENU[airplane_key]) == 0:
                            continue
                        original_norms = numpy.sqrt(flight_tracks_ENU[airplane_key][:,0]**2 + flight_tracks_ENU[airplane_key][:,1]**2 + flight_tracks_ENU[airplane_key][:,2]**2 )
                        cut = numpy.logical_and(original_norms/500.0 < plot_distance_cut_limit,numpy.logical_and(flight_tracks_ENU[airplane_key][:,3] >= start ,flight_tracks_ENU[airplane_key][:,3] <= stop))
                        if force_fit_order is not None:
                            order = force_fit_order
                        else:
                            if sum(cut) == 0:
                                continue
                            elif sum(cut) < 40:
                                order = 3
                            elif sum(cut) < 80:
                                order = 5
                            else:
                                order = 7
                        poly = pt.PlanePoly(flight_tracks_ENU[airplane_key][cut,3],(flight_tracks_ENU[airplane_key][cut,0],flight_tracks_ENU[airplane_key][cut,1],flight_tracks_ENU[airplane_key][cut,2]),order=order,plot=False)

                        t = numpy.arange(min(flight_tracks_ENU[airplane_key][:,3][cut]), max(flight_tracks_ENU[airplane_key][:,3][cut]))

                        interpolated_airplane_locations = poly.poly(t)
                        rpt = enu2Spherical(interpolated_airplane_locations)
                        rpe = numpy.copy(rpt)
                        rpe[:,2] = 90.0 - rpe[:,2]

                        ax.plot(rpe[:,1], rpe[:,2])

            if use_new_code == True:
                files = getFileNamesFromTimestamps(start, stop, verbose=True)
                df = getDataFrames(start, stop, query='up > 0 and azimuth >= -90 and azimuth <= 90 and zenith < 85')
                minimum_approach = 1e10
                minimum_approach_airplane = ''
                minimum_approach_rpt = None
                rpt_at_event_time = None
                minimum_rpt_at_event_time = None
                all_min_angular_distances = {}
                for index, icao24 in enumerate(numpy.unique(df['icao24'])):
                    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][index%len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
                    traj = df.query('icao24 == "%s" and distance < %f'%(icao24, plot_distance_cut_limit*1000))
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

                        approach_angle = numpy.sqrt((rpt[:,1] - best_phi)**2 + (rpt[:,2] - (90.0 - best_elevation ))**2)
                        # numpy.sqrt((minimum_approach_rpt[1] - best_phi)**2 + (minimum_approach_rpt[2] - (90.0 - best_elevation ))**2)


                        minimum_approach_index = numpy.argmin(approach_angle)

                        rpt_at_event_time = enu2Spherical(poly.poly(event_times[event_index]))[0]

                        all_min_angular_distances[icao24] = {}
                        all_min_angular_distances[icao24]['angular distance'] = approach_angle[minimum_approach_index]
                        all_min_angular_distances[icao24]['trigtime - t'] = event_times[event_index] - t[minimum_approach_index]

                        if approach_angle[minimum_approach_index] < minimum_approach:
                            minimum_approach = approach_angle[minimum_approach_index]
                            minimum_approach_t = t[minimum_approach_index]
                            minimum_approach_rpt = rpt[minimum_approach_index,:]
                            minimum_approach_airplane = icao24

                            minimum_rpt_at_event_time = enu2Spherical(poly.poly(event_times[event_index]))[0]


                            print(minimum_approach , str(minimum_approach_rpt))


                        rpe = numpy.copy(rpt)
                        rpe[:,2] = 90.0 - rpe[:,2]
                        ax.plot(rpe[:,1], rpe[:,2], linestyle = '--', c=color, alpha=0.2)
                        ax.scatter(rpt_at_event_time[1], 90.0 - rpt_at_event_time[2],marker='|',c=color)

                    ax.scatter(traj['azimuth'], 90.0 - traj['zenith'], c=color)#linestyle = '-.' , c='r', 

                print('run = ', run)
                print('eventid = ', eventid)
                print('minimum_approach = ', minimum_approach, ' deg')
                print('minimum_approach_airplane = ', minimum_approach_airplane)

                print('All minimum approaches:')
                pprint(all_min_angular_distances)

                if minimum_approach_rpt is not None:
                    ax.scatter(minimum_approach_rpt[1], 90.0 - minimum_approach_rpt[2],
                        marker='*',c='k',
                        label='Minimum Angular Approach\nr,phi,el = %0.2f km, %0.2f deg, %0.2f deg\nat triggertime - t = %0.2f\nicao24 = %s'%(minimum_approach_rpt[0]/1000.0, minimum_approach_rpt[1], 90.0 - minimum_approach_rpt[2], event_times[event_index] - minimum_approach_t, minimum_approach_airplane))
                if minimum_rpt_at_event_time is not None and rpt_at_event_time is not None:
                    ax.scatter(minimum_rpt_at_event_time[1], 90.0 - minimum_rpt_at_event_time[2],
                        marker='o',c='k',
                        label='At Trig Time = %0.2f\nr,phi,el = %0.2f km, %0.2f deg, %0.2f deg'%(event_times[event_index], rpt_at_event_time[0]/1000.0, rpt_at_event_time[1], 90.0 - rpt_at_event_time[2]))

                plt.legend(loc='lower center', fontsize = 16)


            #Make 3d plot of full plane trajectory for min_airplane
            #Plot vector of pointing direction from map, see if they make sense
            if save_figures == True:
                fig.set_size_inches(25, 12.5)
                fig.savefig(os.path.join(outpath,'./r%ie%i.png'%(run, eventid)), dpi=300)
            else:
                import pdb; pdb.set_trace()
            print('Closing figure')
            plt.close(fig)
