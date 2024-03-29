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

def maximizeAllFigures():
    '''
    Maximizes all matplotlib plots.
    '''
    for i in plt.get_fignums():
        plt.figure(i)
        fm = plt.get_current_fig_manager()
        fm.resize(*fm.window.maxsize())

'''
'impulsivity_h','impulsivity_v', 'cr_template_search_h', 'cr_template_search_v', 'std_h', 'std_v', 'p2p_h', 'p2p_v', 'snr_h', 'snr_v',\
'time_delay_0subtract1_h','time_delay_0subtract2_h','time_delay_0subtract3_h','time_delay_1subtract2_h','time_delay_1subtract3_h','time_delay_2subtract3_h',\
'time_delay_0subtract1_v','time_delay_0subtract2_v','time_delay_0subtract3_v','time_delay_1subtract2_v','time_delay_1subtract3_v','time_delay_2subtract3_v',
'cw_present','cw_freq_Mhz','cw_linear_magnitude','cw_dbish','theta_best_h','theta_best_v','elevation_best_h','elevation_best_v','phi_best_h','phi_best_v',\
'calibrated_trigtime','triggered_beams','beam_power','hpol_peak_to_sidelobe','vpol_peak_to_sidelobe','hpol_max_possible_map_value','vpol_max_possible_map_value',\
'map_max_time_delay_0subtract1_h','map_max_time_delay_0subtract2_h','map_max_time_delay_0subtract3_h',\
'map_max_time_delay_1subtract2_h','map_max_time_delay_1subtract3_h','map_max_time_delay_2subtract3_h',\
'map_max_time_delay_0subtract1_v','map_max_time_delay_0subtract2_v','map_max_time_delay_0subtract3_v',\
'map_max_time_delay_1subtract2_v','map_max_time_delay_1subtract3_v','map_max_time_delay_2subtract3_v'
'''

#Include special conditions for certain events
special_conditions = {}
# Default Values
include_baselines = numpy.array([0,1,2,3,4,5])
append_notches = None

special_conditions['r5755e112418'] = {
                                    'include_baselines':numpy.array([3,4,5]),
                                    'append_notches':[[62,63.5],[67,69]]
                                    }

special_conditions['r5966e45159'] = {
                                    'append_notches':[[33,38]]
                                    }


special_conditions['r5896e46823'] = {
                                    'append_notches':[[33,38]]
                                    }

special_conditions['r5889e70102'] = {
                                    'append_notches':[[33,38]]
                                    }
special_conditions['r5853e114664'] = {
                                    'append_notches':[[62.5,70]]
                                    }
special_conditions['r5978e120178'] = {
                                    'append_notches':[[33,38]]
                                    }
special_conditions['r5984e85580'] = {
                                    'append_notches':[[33,38]]
                                    }
special_conditions['r6019e31292'] = {
                                    'append_notches':[[60,66]]
                                    }
special_conditions['r6235e54175'] = {
                                    'append_notches':[[43,47]]
                                    }

special_conditions['r6243e68912'] = {
                                    'append_notches':[[43,47]]
                                    }

special_conditions['r6262e93320'] = {
                                    'append_notches':[[48,54]]
                                    }

special_conditions['r6263e33446'] = {
                                    'append_notches':[[48,54]]
                                    }
special_conditions['r6580e36332'] = {
                                    'append_notches':[[48,54]]
                                    }

ignore_airplanes = []#['a73278']

if __name__ == '__main__':
    args = copy.copy(sys.argv)
    if len(sys.argv) == 2:
        if 'r' in sys.argv[1] and 'e' in sys.argv[1]:
            args = [sys.argv[0], int(sys.argv[1].replace('r','').split('e')[0]), int(sys.argv[1].replace('r','').split('e')[1])]

    if len(args) >= 3:
        run = int(args[1])
        eventid = int(args[2])
        if len(args) == 4:
            apply_additional_notches = bool(args[3])
        else:
            # plt.close('all')
            apply_additional_notches = True

        cmap = 'cool'#'coolwarm'
        impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'

        ds = dataSlicer([run], impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath)
        
        # Custom testing values
        if apply_additional_notches:
            event_key = 'r%ie%i'%(run,eventid)
            if event_key in list(special_conditions.keys()):
                if 'append_notches' in list(special_conditions[event_key].keys()):
                    append_notches = special_conditions[event_key]['append_notches']
                if 'include_baselines' in list(special_conditions[event_key].keys()):
                    include_baselines = special_conditions[event_key]['include_baselines']


        # ds.eventInspector({run:[eventid]}, show_all=True, include_time_delays=True,append_notches=append_notches)

        for conference_mode in [False]:
            ds.conference_mode = conference_mode
            ds.eventInspector({run:[eventid]}, show_all=False, include_time_delays=not ds.conference_mode,append_notches=append_notches,include_baselines=include_baselines, div128=ds.conference_mode)
            

        # ds.eventInspector({run:[eventid]}, show_all=False, include_time_delays=True,append_notches=append_notches)
        # ds.eventInspector({run:[eventid]}, show_all=False, include_time_delays=False,append_notches=append_notches)
        print('https://users.rcc.uchicago.edu/~cozzyd/monutau/#event&run=%i&entry=%i'%(run,eventid))


        suspected_airplanes = {}#{6027:[21206,21515,21718,21812,21822,21905,21989,21993,22000,22008,22098,22176,22191,22208,22220,22323,22504,22507,22509,22565,22566,22652,22655,22731,22742,22749,22845,22847,22903,22917,22932,23364,23419,23429,23440,23462,23471,23477,23514,23519,23527,23557,23558,23593,23621,23627,23636,23643,23653,23659,23677,23678,23687,23920,28991,29275,29365,29459,29475,29595,29935,29939,29945,29948,30344,44049,51866,51871,51970,51986,51989,52086,52090,52099,52106,52199,52861,52957,52973,52974,53074,53175]}
        if run in list(suspected_airplanes.keys()):
            if eventid in suspected_airplanes[run]:
                total_mean_corr_values, fig_averaged, ax_averaged = ds.cor.averagedMap(suspected_airplanes[run], 'hpol', plot_map=True, hilbert=False, max_method=None, mollweide=False, zenith_cut_ENU=None,zenith_cut_array_plane=None, center_dir='E', circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={})
                try_animated = False
                if try_animated:
                    animatedMap(suspected_airplanes[run], 'hpol', '', include_baselines=[0,1,2,3,4,5], plane_zenith=None, plane_az=None, map_source_distance_m=None, radius=1.0, hilbert=False, max_method=None,center_dir='E',save=False,dpi=300,fps=3)
                    ax_animated = ds.cor.axs[-1]
            else:
                ax_averaged = None
        else:
            ax_averaged = None
        pprint(ds.inspector_mpl['current_table'])

        if pandas.__version__ == '1.4.0' and True:
            from tools.airplane_traffic_loader import getFileNamesFromTimestamps, enu2Spherical, addDirectionInformationToDataFrame, readPickle, getDataFrames
            from tools.get_plane_tracks import plotAirplaneTrackerStatus
            # Needs to match the version of pandas which was used to store airplane data.
            ax_keys = ['fig1_map_h','fig1_map_v']
            if ds.show_all:
                ax_keys.append('fig1_map_all')
            try:
                print('Looking for airplanes...')
                time_window_s = 5*60
                plot_distance_cut_limit = 500
                min_approach_cut_km = 1e6
                origin = ds.cor.A0_latlonel_hpol
                force_fit_order = 3#None

                elevation_best_choice = ds.getDataFromParam({run:[eventid]}, 'elevation_best_choice')
                phi_best_choice = ds.getDataFromParam({run:[eventid]}, 'phi_best_choice')
                best_elevation = float(elevation_best_choice[run][numpy.array([eventid]) == eventid][0])
                best_phi = float(phi_best_choice[run][numpy.array([eventid]) == eventid][0])

                event_time = ds.cor.getEventTimes()[numpy.array([eventid])][0]
                start = event_time - time_window_s/2
                stop = event_time +  time_window_s/2

                files = getFileNamesFromTimestamps(start, stop, verbose=True)
                df = getDataFrames(start, stop, query='up > 0 and azimuth >= -90 and azimuth <= 90 and zenith < 85')
                minimum_approach = 1e10
                minimum_approach_airplane = ''
                minimum_approach_rpt = None
                rpt_at_event_time = None
                minimum_rpt_at_event_time = None
                all_min_angular_distances = {}
                print(numpy.unique(df['icao24']))
                for index, icao24 in enumerate(numpy.unique(df['icao24'])):
                    if icao24 in ignore_airplanes:
                        continue
                    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][index%len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
                    try:
                        traj = df.query('icao24 == "%s" and distance < %f'%(icao24, plot_distance_cut_limit*1000))
                    except Exception as e:
                        print(e)
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

                        approach_angle = numpy.sqrt((rpt[:,1] - best_phi)**2 + (rpt[:,2] - (90.0 - best_elevation ))**2)
                        # numpy.sqrt((minimum_approach_rpt[1] - best_phi)**2 + (minimum_approach_rpt[2] - (90.0 - best_elevation ))**2)


                        minimum_approach_index = numpy.argmin(approach_angle)

                        rpt_at_event_time = enu2Spherical(poly.poly(event_time))[0]

                        all_min_angular_distances[icao24] = {}
                        all_min_angular_distances[icao24]['angular distance'] = approach_angle[minimum_approach_index]
                        all_min_angular_distances[icao24]['trigtime - t'] = event_time - t[minimum_approach_index]

                        if approach_angle[minimum_approach_index] < minimum_approach:
                            minimum_approach = approach_angle[minimum_approach_index]
                            minimum_approach_t = t[minimum_approach_index]
                            minimum_approach_rpt = rpt[minimum_approach_index,:]
                            minimum_approach_airplane = icao24

                            minimum_rpt_at_event_time = enu2Spherical(poly.poly(event_time))[0]


                            print(minimum_approach , str(minimum_approach_rpt))


                        rpe = numpy.copy(rpt)
                        rpe[:,2] = 90.0 - rpe[:,2]

                        for ax_key in ax_keys:
                            ax = ds.inspector_mpl[ax_key]
                            ax.plot(rpe[:,1], rpe[:,2], linestyle = '--', c=color, alpha=0.2)
                            ax.scatter(rpt_at_event_time[1], 90.0 - rpt_at_event_time[2],marker='|',c=color)

                            if ax_averaged is not None:
                                ax_averaged.plot(rpe[:,1], rpe[:,2], linestyle = '--', c=color, alpha=0.2)
                                ax_averaged.scatter(rpt_at_event_time[1], 90.0 - rpt_at_event_time[2],marker='|',c=color)
                                if try_animated:
                                    ax_animated.plot(rpe[:,1], rpe[:,2], linestyle = '--', c=color, alpha=0.2)
                                    ax_animated.scatter(rpt_at_event_time[1], 90.0 - rpt_at_event_time[2],marker='|',c=color)



                    for ax_key in ax_keys:
                        ax = ds.inspector_mpl[ax_key]
                        ax.scatter(traj['azimuth'], 90.0 - traj['zenith'], c=color)

                        if ax_averaged is not None:
                            ax_averaged.scatter(traj['azimuth'], 90.0 - traj['zenith'], c=color)
                            if try_animated:
                                ax_animated.scatter(traj['azimuth'], 90.0 - traj['zenith'], c=color)

                [print(icao24, ' ang = ', all_min_angular_distances[icao24]['angular distance'], ' dt = ', all_min_angular_distances[icao24]['trigtime - t']) for icao24 in list(all_min_angular_distances.keys())]

                print('Relative to\nphi = %0.2f, eleveation = %0.2f'%(best_phi, best_elevation))
                print('minimum_approach = ',minimum_approach)
                print('minimum_approach_t = ',minimum_approach_t)
                print('minimum_approach_rpt = ',minimum_approach_rpt)
                print('minimum_approach_airplane = ',minimum_approach_airplane)
                print('minimum_rpt_at_event_time = ',minimum_rpt_at_event_time)

                for ax_key in ax_keys:
                    if minimum_approach_rpt is not None:
                        ax.scatter(minimum_approach_rpt[1], 90.0 - minimum_approach_rpt[2],
                            marker='*',c='k',
                            label='Minimum Angular Approach\nr,phi,el = %0.2f km, %0.2f deg, %0.2f deg\nat triggertime - t = %0.2f\nicao24 = %s'%(minimum_approach_rpt[0]/1000.0, minimum_approach_rpt[1], 90.0 - minimum_approach_rpt[2], event_time - minimum_approach_t, minimum_approach_airplane))
                        if ax_averaged is not None:
                            ax_averaged.scatter(minimum_approach_rpt[1], 90.0 - minimum_approach_rpt[2],
                                                marker='*',c='k',
                                                label='Minimum Angular Approach\nr,phi,el = %0.2f km, %0.2f deg, %0.2f deg\nat triggertime - t = %0.2f\nicao24 = %s'%(minimum_approach_rpt[0]/1000.0, minimum_approach_rpt[1], 90.0 - minimum_approach_rpt[2], event_time - minimum_approach_t, minimum_approach_airplane))
                            if try_animated:
                                ax_animated.scatter(minimum_approach_rpt[1], 90.0 - minimum_approach_rpt[2],
                                                    marker='*',c='k',
                                                    label='Minimum Angular Approach\nr,phi,el = %0.2f km, %0.2f deg, %0.2f deg\nat triggertime - t = %0.2f\nicao24 = %s'%(minimum_approach_rpt[0]/1000.0, minimum_approach_rpt[1], 90.0 - minimum_approach_rpt[2], event_time - minimum_approach_t, minimum_approach_airplane))

                    if minimum_rpt_at_event_time is not None and rpt_at_event_time is not None:
                        ax.scatter(minimum_rpt_at_event_time[1], 90.0 - minimum_rpt_at_event_time[2],
                            marker='o',c='k',
                            label='At Trig Time = %0.2f\nr,phi,el = %0.2f km, %0.2f deg, %0.2f deg'%(event_time, rpt_at_event_time[0]/1000.0, rpt_at_event_time[1], 90.0 - rpt_at_event_time[2]))
                        if ax_averaged is not None:
                            ax_averaged.scatter(minimum_rpt_at_event_time[1], 90.0 - minimum_rpt_at_event_time[2],
                                                marker='o',c='k',
                                                label='At Trig Time = %0.2f\nr,phi,el = %0.2f km, %0.2f deg, %0.2f deg'%(event_time, rpt_at_event_time[0]/1000.0, rpt_at_event_time[1], 90.0 - rpt_at_event_time[2]))
                            if try_animated:
                                ax_animated.scatter(minimum_rpt_at_event_time[1], 90.0 - minimum_rpt_at_event_time[2],
                                                    marker='o',c='k',
                                                    label='At Trig Time = %0.2f\nr,phi,el = %0.2f km, %0.2f deg, %0.2f deg'%(event_time, rpt_at_event_time[0]/1000.0, rpt_at_event_time[1], 90.0 - rpt_at_event_time[2]))

                    # ax.legend(loc='lower center', fontsize = 16)
                    # for t in ax.get_legend().get_texts():
                    #     print(t)

                maximizeAllFigures()
            except Exception as e:
                print(e)

        # import pdb; pdb.set_trace()
