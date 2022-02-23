#!/usr/bin/env python3
'''
This will add to the functionality of flipbook_reader.py and take any eventids_dict and try to collate the information
for the contained events into an excel file for easy note taking and viewing.  It attempts to add airplane information
and thus should be run with code that has pandas version >= 1.4.0.  This version is also necessary for the used
"overlay" function when writing to excel. 
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
import pandas as pd

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.flipbook_reader import flipbookToDict, concatenateFlipbookToDict, concatenateFlipbookToArray, concatenateEventDictToArray
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


def writeEventDictionaryToDataFrame(initial_eventids_dict, ds=None, include_airplanes=True):
    try:
        data_keys = [
                    'phi_best_choice',
                    'elevation_best_choice',
                    'cr_template_search_h',
                    'cr_template_search_v',
                    'cr_template_search_hSLICERMAXcr_template_search_v',
                    'hpol_peak_to_sidelobe',
                    'vpol_peak_to_sidelobe',
                    'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe',
                    'hpol_normalized_map_value',
                    'vpol_normalized_map_value',
                    'above_normalized_map_max_line',
                    'above_snr_line',
                    'impulsivity_h',
                    'impulsivity_v',
                    'impulsivity_hSLICERADDimpulsivity_v',
                    'similarity_count_h',
                    'similarity_count_v',
                    'csnr_h',
                    'csnr_v',
                    'snr_h',
                    'snr_v',
                    'p2p_h',
                    'p2p_v',
                    'std_h',
                    'std_v'
                    ]

        if numpy.all([numpy.issubdtype(k, numpy.integer) for k in initial_eventids_dict.keys()]) or numpy.all(numpy.array(list(initial_eventids_dict.keys()))%1 == 0):
            print('Assuming passed "initial_eventids_dict" as eventids_dict format')
            eventids_dict = copy.deepcopy(initial_eventids_dict)
            eventids_array = concatenateEventDictToArray(initial_eventids_dict)
        else:
            print('Assuming passed "initial_eventids_dict" as flipbook format')
            eventids_dict = concatenateFlipbookToDict(initial_eventids_dict)
            eventids_array = concatenateFlipbookToArray(initial_eventids_dict)

        runs = list(eventids_dict.keys())

        force_fit_order = 3 #None to use varying order

        # outpath = './airplane_event_flipbook_%i'%time.time() 
        # os.mkdir(outpath)


        if ds is None:
            print("Preparing dataSlicer")
            impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
            time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
            map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'

            ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, verbose_setup=False)
            ds.prepareCorrelator()
        else:
            print('Using passed dataSlicer')

        status = numpy.zeros(len(eventids_array), dtype=str)

        data = {    
                    'run'       : eventids_array['run'],
                    'eventid'   : eventids_array['eventid'],
                    'key'       : eventids_array['key'],
                }

        monutau_links = []
        for eid in eventids_array:
            eventid = eid['eventid']
            run = eid['run']
            url = "https://users.rcc.uchicago.edu/~cozzyd/monutau/#event&run=%i&entry=%i"%(run,eventid)
            monutau_links.append('=HYPERLINK("%s", "link")'%url)

        data['monutau'] = numpy.asarray(monutau_links)
        data['notes'] = [numpy.nan]*len(eventids_array)

        for key in data_keys:
            d = ds.getDataArrayFromParam(key, trigger_types=None, eventids_dict=copy.deepcopy(eventids_dict))
            data[key] = d

        if include_airplanes == True:
            print('Calculating airplane information')
            ds.prepareCorrelator()
            time_window_s = 5*60
            plot_distance_cut_limit = 500
            min_approach_cut_km = 1e6
            origin = ds.cor.A0_latlonel_hpol

            elevation_best_choice = ds.getDataFromParam(eventids_dict, 'elevation_best_choice')
            phi_best_choice = ds.getDataFromParam(eventids_dict, 'phi_best_choice')

            all_minimum_approach = numpy.zeros(len(eventids_array), dtype=float)
            all_event_times = numpy.zeros(len(eventids_array), dtype=float)
            all_at_event_time_r = numpy.zeros(len(eventids_array), dtype=float)
            all_at_event_time_phi = numpy.zeros(len(eventids_array), dtype=float)
            all_at_event_time_theta = numpy.zeros(len(eventids_array), dtype=float)
            all_minimum_approach_t = numpy.zeros(len(eventids_array), dtype=float)
            all_minimum_rpt_at_event_time = numpy.zeros(len(eventids_array), dtype=float)
            all_minimum_approach_r = numpy.zeros(len(eventids_array), dtype=float)
            all_minimum_approach_phi = numpy.zeros(len(eventids_array), dtype=float)
            all_minimum_approach_theta = numpy.zeros(len(eventids_array), dtype=float)
            all_minimum_approach_airplane = numpy.zeros(len(eventids_array), dtype='<U6')

            for loop_index, run in enumerate(runs):
                sys.stdout.write('Run %i/%i\r'%(loop_index+1,len(runs)))
                sys.stdout.flush()
                run_index = numpy.where(numpy.isin(ds.runs, run))[0][0]
                if ds.cor.reader.run != run:
                    ds.cor.setReader(ds.data_slicers[run_index].reader, verbose=False)

                eventids = eventids_dict[run]
                event_times = ds.cor.getEventTimes()[eventids]

                for event_index, eventid in enumerate(eventids):
                    entry_index = numpy.where(numpy.logical_and(eventids_array['run'] == run , eventids_array['eventid'] == eventid))[0]

                    start = event_times[event_index] - time_window_s/2
                    stop = event_times[event_index] +  time_window_s/2
                    flight_tracks_ENU, all_vals = pt.getENUTrackDict(start,stop,min_approach_cut_km,hour_window=2,flights_of_interest=[],origin=origin)

                    best_elevation = float(elevation_best_choice[run][numpy.array(eventids_dict[run]) == eventid][0])
                    best_phi = float(phi_best_choice[run][numpy.array(eventids_dict[run]) == eventid][0])

                    ds.cor.reader.setEntry(eventid)

                    m = ds.cor.map(eventid, 'hpol', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=False, map_ax=None, plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=None, verbose=False, mollweide=False, zenith_cut_ENU=None, zenith_cut_array_plane=(0,90), center_dir='E', circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=True, circle_map_max=False, override_to_time_window=(None,None))
                    linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = ds.cor.mapMax(m, max_method=0, verbose=False, zenith_cut_ENU=[0,80], zenith_cut_array_plane=[0,90], pol='hpol', return_peak_to_sidelobe=False, theta_cut=None)

                    files = getFileNamesFromTimestamps(start, stop, verbose=True)
                    df = getDataFrames(start, stop, query='up > 0 and azimuth >= -90 and azimuth <= 90 and zenith < 85')
                    minimum_approach = 1e10
                    minimum_approach_airplane = ''
                    minimum_approach_rpt = None
                    rpt_at_event_time = None
                    minimum_rpt_at_event_time = None
                    all_min_angular_distances = {}
                    for index, icao24 in enumerate(numpy.unique(df['icao24'])):
                        if 'icao24' not in df.columns:
                            print('icao24 not in df')
                            print('Skipping airplane data %s for %i %i with start=%f, stop=%f'%(icao24, run, eventid, start, stop))
                            continue
                        if 'distance' not in df.columns:
                            print('distance not in df')
                            print('Skipping airplane data %s for %i %i with start=%f, stop=%f'%(icao24, run, eventid, start, stop))
                            continue
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


                                # print(minimum_approach , str(minimum_approach_rpt))


                            rpe = numpy.copy(rpt)
                            rpe[:,2] = 90.0 - rpe[:,2]

                    # print('run = ', run)
                    # print('eventid = ', eventid)
                    # print('minimum_approach = ', minimum_approach, ' deg')
                    # print('minimum_approach_airplane = ', minimum_approach_airplane)

                    if minimum_approach_rpt is not None:
                        all_minimum_approach[entry_index] = minimum_approach #degrees
                        all_minimum_approach_t[entry_index] = minimum_approach_t #time of minimum aproach
                        all_minimum_approach_r[entry_index] = minimum_approach_rpt[0] #vector of minimal approach
                        all_minimum_approach_phi[entry_index] = minimum_approach_rpt[1] #vector of minimal approach
                        all_minimum_approach_theta[entry_index] = minimum_approach_rpt[2] #vector of minimal approach

                        all_minimum_approach_airplane[entry_index] = minimum_approach_airplane #icao24 of minimum approach
                        all_event_times[entry_index] = event_times[event_index] #actual event time
                        all_at_event_time_r[entry_index] = minimum_rpt_at_event_time[0] #vector of minimal approach
                        all_at_event_time_phi[entry_index] = minimum_rpt_at_event_time[1] #vector of minimal approach
                        all_at_event_time_theta[entry_index] = minimum_rpt_at_event_time[2] #vector of minimal approach
                    else:
                        all_minimum_approach[entry_index] = numpy.nan
                        all_minimum_approach_t[entry_index] = numpy.nan
                        all_minimum_approach_r[entry_index] = numpy.nan
                        all_minimum_approach_phi[entry_index] = numpy.nan
                        all_minimum_approach_theta[entry_index] = numpy.nan

                        all_minimum_approach_airplane[entry_index] = numpy.nan
                        all_event_times[entry_index] = numpy.nan
                        all_at_event_time_r[entry_index] = numpy.nan
                        all_at_event_time_phi[entry_index] = numpy.nan
                        all_at_event_time_theta[entry_index] = numpy.nan

                    # print('All minimum approaches:')
                    # pprint(all_min_angular_distances)

            data['airplane_minimum_approach_icao24']   = all_minimum_approach_airplane
            data['airplane_minimum_approach']          = all_minimum_approach

            data['airplane_minimum_approach_t']        = all_minimum_approach_t
            data['airplane_minimum_approach_r']        = all_minimum_approach_r
            data['airplane_minimum_approach_phi']      = all_minimum_approach_phi
            data['airplane_minimum_approach_theta']    = all_minimum_approach_theta

            data['event_time_t']        = all_event_times
            data['airplane_at_event_time_r']        = all_at_event_time_r
            data['airplane_at_event_time_phi']      = all_at_event_time_phi
            data['airplane_at_event_time_theta']    = all_at_event_time_theta

        

        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        import pdb; pdb.set_trace()

def writeDataFrameToExcel(df, filename, sheetname):
    if os.path.exists(filename):
        # File exists, attempt to append to it.
        print('File exists, attempt to append to it.')
        with pd.ExcelWriter(filename, engine="openpyxl", mode='a', if_sheet_exists='new') as writer:
            df.to_excel(writer,sheet_name=sheetname,float_format="%0.2f", index=False)
    else:
        # File does not exist, attempt to create to it.
        print('File does not exist, attempt to create to it.')
        with pd.ExcelWriter(filename, engine="openpyxl", mode='w') as writer:
            df.to_excel(writer,sheet_name=sheetname,float_format="%0.2f", index=False)


def writeEventDictionaryToExcel(initial_eventids_dict, filename, sheetname, ds=None, include_airplanes=True):
    df = writeEventDictionaryToDataFrame(initial_eventids_dict, ds=ds, include_airplanes=include_airplanes)
    writeDataFrameToExcel(df, filename, sheetname)


if __name__ == '__main__':
    if int(pd.__version__.split('.')[0]) >= 1 and int(pd.__version__.split('.')[1]) >= 4:
        # flipbook_path = '/home/dsouthall/scratch-midway2/event_flipbook_1643154940'#'/home/dsouthall/scratch-midway2/event_flipbook_1642725413'
        # flipbook_path = './airplane_event_flipbook_1643947072'
        flawed_runs = numpy.array([6537,6538,6539]) #numpy.array([5775,5981,5993,6033,6090,6520,6537,6538,6539]) 
        filename = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'analysis','sept2021-week1-analysis','hand-scanned-event-info.xlsx')
        # include_airplanes = False
        for include_airplanes in [False, True]:
            #['/home/dsouthall/scratch-midway2/event_flipbook_1643154940', './airplane_event_flipbook_1643947072']
            for flipbook_path in ['./september-flipbook']:
                sorted_dict = flipbookToDict(flipbook_path, ignore_runs=flawed_runs)
                if True:
                    sheetname = os.path.split(flipbook_path)[-1] + '_airplanes-included-%s'%str(include_airplanes)
                else:
                    sheetname = 'raw_airplanes-included-%s'%str(include_airplanes)

                df = writeEventDictionaryToDataFrame(sorted_dict, include_airplanes=include_airplanes)
                writeDataFrameToExcel(df, filename, sheetname)
                # writeEventDictionaryToExcel(sorted_dict, filename, ds=None)
    else:
        print('This script requires pandas version >= 1.4.0')
