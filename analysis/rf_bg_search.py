#!/usr/bin/env python3
'''
This script is meant to determine where each map points for each event and then save these values to file. 
'''
import os
import sys
import h5py
from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
import beacon.tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import beacon.tools.info as info
from beacon.tools.data_handler import createFile

import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import numpy
import sys
import math
import matplotlib
from scipy.fftpack import fft
from datetime import datetime
import inspect
from ast import literal_eval
import astropy.units as apu
from astropy.coordinates import SkyCoord
plt.ion()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tools.correlator import Correlator

# import multiprocessing
# import concurrent.futures
# from multiprocessing import cpu_count
# import threading
# n_cores = cpu_count()


font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 16})

def getEventIds(reader, trigger_type=None):
    '''
    Will get a list of eventids for the given reader, but only those matching the trigger
    types supplied.  If trigger_type  == None then all eventids will be returned. 
    trigger_type:
    1 Software
    2 RF
    3 GPS
    '''
    try:
        if trigger_type == None:
            trigger_type = numpy.array([1,2,3])
        elif type(trigger_type) == int:
            trigger_type = numpy.array([trigger_type])

        eventids = numpy.array([])
        for trig_index, trig in enumerate(trigger_type):
            N = reader.head_tree.Draw("Entry$","trigger_type==%i"%trig,"goff") 
            eventids = numpy.append(eventids,numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N).astype(int))
        eventids.sort()
        return eventids
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return e


datapath = os.environ['BEACON_DATA']



# def batchedMapMax(filter_string, map_max_mode, event_index, cor, eventid, lock, debug=False):
#     '''
#     This is to allow multiprocessing specifically on the portion of the code where the maximum value and peak to
#     sidelobe is calculated for each of the different filter strings.
#     '''
#     if debug == True:
#         if event_index == 0:
#             m, max_possible_map_value = cor.map(eventid, mode, plot_map=False, plot_corr=False, verbose=False, hilbert=hilbert,return_max_possible_map_value=True)
#     else:
#         m, max_possible_map_value = cor.map(eventid, mode, plot_map=False, plot_corr=False, verbose=False, hilbert=hilbert,return_max_possible_map_value=True)

#     for filter_string_index, filter_string in enumerate(filter_strings):
#         file['map_properties'][filter_string]['%s_max_possible_map_value'%mode][eventid] = max_possible_map_value #doesn't really depend on filter_string but would depend on calibration, so keeping it sorted.
#         #Determine cut values
#         if mapmax_cut_modes[filter_string_index] == 'abovehorizon':
#             # print('abovehorizon')
#             zenith_cut_ENU=[0,90] #leaving some tolerance
#             zenith_cut_array_plane=None
#         elif mapmax_cut_modes[filter_string_index] == 'belowhorizon':
#             # print('belowhorizon')
#             zenith_cut_ENU=[90,180]
#             zenith_cut_array_plane=[0,91] #Up to 1 degree below projected array plane.
#         elif mapmax_cut_modes[filter_string_index] == 'allsky':
#             # print('allsky')
#             zenith_cut_ENU=None
#             zenith_cut_array_plane=[0,91] #Up to 1 degree below projected array plane.
#         else:
#             zenith_cut_ENU=None
#             zenith_cut_array_plane=None

#         #Calculate best reconstruction direction
#         if debug == True:
#             if event_index == 0:
#                 if max_method is not None:
#                     linear_max_index, theta_best, phi_best, t_0subtract1, t_0subtract2, t_0subtract3, t_1subtract2, t_1subtract3, t_2subtract3, peak_to_sidelobe = cor.mapMax(m,max_method=max_method,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol=mode, return_peak_to_sidelobe=True)
#                 else:
#                     linear_max_index, theta_best, phi_best, t_0subtract1, t_0subtract2, t_0subtract3, t_1subtract2, t_1subtract3, t_2subtract3, peak_to_sidelobe = cor.mapMax(m,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol=mode, return_peak_to_sidelobe=True)
#         else:
#             if max_method is not None:
#                 linear_max_index, theta_best, phi_best, t_0subtract1, t_0subtract2, t_0subtract3, t_1subtract2, t_1subtract3, t_2subtract3, peak_to_sidelobe = cor.mapMax(m,max_method=max_method,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol=mode, return_peak_to_sidelobe=True)
#             else:
#                 linear_max_index, theta_best, phi_best, t_0subtract1, t_0subtract2, t_0subtract3, t_1subtract2, t_1subtract3, t_2subtract3, peak_to_sidelobe = cor.mapMax(m,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol=mode, return_peak_to_sidelobe=True)

#         if debug == True:
#             if event_index == 0:
#                 file['map_direction'][filter_string]['%s_ENU_zenith'%mode][eventid] = theta_best 
#                 file['map_direction'][filter_string]['%s_ENU_azimuth'%mode][eventid] = phi_best 
#                 file['map_times'][filter_string]['%s_0subtract1'%mode][eventid] = t_0subtract1 
#                 file['map_times'][filter_string]['%s_0subtract2'%mode][eventid] = t_0subtract2 
#                 file['map_times'][filter_string]['%s_0subtract3'%mode][eventid] = t_0subtract3 
#                 file['map_times'][filter_string]['%s_1subtract2'%mode][eventid] = t_1subtract2 
#                 file['map_times'][filter_string]['%s_1subtract3'%mode][eventid] = t_1subtract3 
#                 file['map_times'][filter_string]['%s_2subtract3'%mode][eventid] = t_2subtract3
#                 file['map_properties'][filter_string]['%s_peak_to_sidelobe'%mode][eventid] = peak_to_sidelobe
#         else:
#             file['map_direction'][filter_string]['%s_ENU_zenith'%mode][eventid] = theta_best 
#             file['map_direction'][filter_string]['%s_ENU_azimuth'%mode][eventid] = phi_best 
#             file['map_times'][filter_string]['%s_0subtract1'%mode][eventid] = t_0subtract1 
#             file['map_times'][filter_string]['%s_0subtract2'%mode][eventid] = t_0subtract2 
#             file['map_times'][filter_string]['%s_0subtract3'%mode][eventid] = t_0subtract3 
#             file['map_times'][filter_string]['%s_1subtract2'%mode][eventid] = t_1subtract2 
#             file['map_times'][filter_string]['%s_1subtract3'%mode][eventid] = t_1subtract3 
#             file['map_times'][filter_string]['%s_2subtract3'%mode][eventid] = t_2subtract3
#             file['map_properties'][filter_string]['%s_peak_to_sidelobe'%mode][eventid] = peak_to_sidelobe



known_pulser_ids = info.load2021PulserEventids()
index_window_dict = {'hpol': {'d2sa': (1250, 2274),
                              'd3sa': (2273, 3297),
                              'd3sb': (2426, 3450),
                              'd3sc': (1889, 2913),
                              'd4sa': (2060, 3084),
                              'd4sb': (1097, 2121)},
                     'vpol': {'d2sa': (1233, 2257),
                              'd3sa': (2246, 3270),
                              'd3sb': (2445, 3469),
                              'd3sc': (1941, 2965),
                              'd4sa': (2074, 3098),
                              'd4sb': (1063, 2087)}}

direction_dict = {  'd2sa': {'azimuth_deg': -45.613736741990834,
                          'zenith_deg': 99.09445476373067,
                          'elevation_deg': -9.094454763730667,
                          'distance_m': 521.616462754899},
                    'd3sa': {'azimuth_deg': -21.04237515116415,
                          'zenith_deg': 97.8769623011641,
                          'elevation_deg': -7.876962301164099,
                          'distance_m': 1118.0401456335342},
                    'd3sb': {'azimuth_deg': -8.413762469954158,
                          'zenith_deg': 101.19711809638179,
                          'elevation_deg': -11.197118096381786,
                          'distance_m': 1192.8539386915534},
                    'd3sc': {'azimuth_deg': -3.5486704218142275,
                          'zenith_deg': 105.81645053640865,
                          'elevation_deg': -15.816450536408652,
                          'distance_m': 897.7425893600894},
                    'd4sa': {'azimuth_deg': 22.644625829641022,
                          'zenith_deg': 96.5140587981508,
                          'elevation_deg': -6.514058798150799,
                          'distance_m': 959.9056107333164},
                    'd4sb': {'azimuth_deg': 58.80941058645383,
                          'zenith_deg': 97.30396628369822,
                          'elevation_deg': -7.303966283698216,
                          'distance_m': 358.97094565670193}}

                            

known_pulser_runs = numpy.unique(numpy.concatenate([numpy.append(numpy.unique(known_pulser_ids[site]['hpol']['run']),numpy.unique(known_pulser_ids[site]['vpol']['run'])) for site in list(known_pulser_ids.keys())]))

pulser_run_sites = {}
for site in list(known_pulser_ids.keys()):
    for run in numpy.unique(numpy.append(numpy.unique(known_pulser_ids[site]['hpol']['run']),numpy.unique(known_pulser_ids[site]['vpol']['run']))):
        pulser_run_sites[run] = site

# This one was double checking that there are no runs that have multiple pulsing sites in them.
# pulser_run_sites = {}
# for site in list(known_pulser_ids.keys()):
#     for run in numpy.unique(numpy.append(numpy.unique(known_pulser_ids[site]['hpol']['run']),numpy.unique(known_pulser_ids[site]['vpol']['run']))):
#         if run not in list(pulser_run_sites.keys()):
#             pulser_run_sites[run] = [site]
#         else:
#             pulser_run_sites[run].append(site)


if __name__=="__main__":


    '''
    NEED TO ADD IN SOMETHING TO HANDLED TIME WINDOWED SIGNALS FOR SPECIFIED RUNS SUCH AS PULSING EVENTS.  
    SPECIFICALLY USING THE WINDOW INDEX RANGE RATHER THAN THE SHORTEN SIGNALS METRIC. 
    '''





    starting_timestamp = datetime.timestamp(datetime.now())
    #multithread = True
    debug = False#THIS IS A TEST, WILL POPULATE ALL VALUES WITH 0, JUST TO MAKE DEBUGGING QUICKER.
    if len(sys.argv) > 1:
        run = int(sys.argv[1])
        if len(sys.argv) >= 3:
            deploy_index = str(sys.argv[2])
        elif run in numpy.arange(1643,1729):
            deploy_index = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'config', 'chi2_optimized_deploy_from_rtk-gps-day1-june20-2021.json')
        else:
            # elif run > 5050:
            #     deploy_index = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'config/rtk-gps-day3-june22-2021.json')
            deploy_index = None

        if len(sys.argv) == 4:
            polarizations = [str(sys.argv[3])]
        else:
            polarizations = ['hpol', 'vpol']
    else:
        run = 1701
        deploy_index = None
        polarizations = ['hpol', 'vpol']

    print('Running rf_bg_search.py')
    print('Performing calculations for %s'%str(polarizations))
    if len(polarizations) > 1:
        print('WARNING, THIS SCRIPT TAKES A LOT OF TIME TO RUN.  IF BOTH POLARIZATIONS ARE ENABLE IT MAY REACH JOB TIME LIMIT.  IT IS RECOMMENDED TO SUBMIT THE DIFFERENT POLARIZATIONS SEPERATELY AND SEQUENTIALLY.')

    impose_time_limit = 35.0 #hours.  If the script is running longer then this time limit then it will attempt to close the file without corrupting it (by having the job cut off.)  To desiable set to None

    plot_filter=False

    crit_freq_low_pass_MHz = 85
    low_pass_filter_order = 6

    crit_freq_high_pass_MHz = 25
    high_pass_filter_order = 8

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.03
    sine_subtract_max_freq_GHz = 0.13
    sine_subtract_percent = 0.03

    apply_phase_response = True

    upsample = 2**15 #Just upsample in this case
    max_method = 0


    #This code will loop over all options included here, and they will be stored as seperate dsets.  Each of these applies different cuts to mapmax when it is attempting to select the best reconstruction direction.
    mapmax_cut_modes = ['abovehorizon','belowhorizon','allsky']
    #['hpol','vpol'] #Will loop over both if hpol and vpol present
    hilbert_modes = [False]#[True,False] #Will loop over both if True and False present

    #Note that the below values set the angular resolution of the plot, while the presets from mapmax_cut_modes limit where in the generated plots will be considered for max values.

    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)


    if deploy_index is None:
        deploy_index = info.returnDefaultDeploy()

    print('DEPLOY INDEX:')
    print(deploy_index)

    for hilbert in hilbert_modes:

        filter_strings = []
        for mapmax_cut_mode in mapmax_cut_modes:

            filter_string = ''

            if crit_freq_low_pass_MHz is None:
                filter_string += 'LPf_%s-'%('None')
            else:
                filter_string += 'LPf_%0.1f-'%(crit_freq_low_pass_MHz)

            if low_pass_filter_order is None:
                filter_string += 'LPo_%s-'%('None')
            else:
                filter_string += 'LPo_%i-'%(low_pass_filter_order)

            if crit_freq_high_pass_MHz is None:
                filter_string += 'HPf_%s-'%('None')
            else:
                filter_string += 'HPf_%0.1f-'%(crit_freq_high_pass_MHz)

            if high_pass_filter_order is None:
                filter_string += 'HPo_%s-'%('None')
            else:
                filter_string += 'HPo_%i-'%(high_pass_filter_order)

            if apply_phase_response is None:
                filter_string += 'Phase_%s-'%('None')
            else:
                filter_string += 'Phase_%i-'%(apply_phase_response)

            if hilbert is None:
                filter_string += 'Hilb_%s-'%('None')
            else:
                filter_string += 'Hilb_%i-'%(hilbert)

            if upsample is None:
                filter_string += 'upsample_%s-'%('None')
            else:
                filter_string += 'upsample_%i-'%(upsample)

            if max_method is None:
                filter_string += 'maxmethod_%s-'%('None')
            else:
                filter_string += 'maxmethod_%i-'%(max_method)

            filter_string += 'sinesubtract_%i-'%(int(sine_subtract))

            filter_string += 'deploy_calibration_%s-'%(os.path.split(str(deploy_index))[1])

            phi_str = 'n_phi_%i-min_phi_%s-max_phi_%s-'%(n_phi, str(min_phi).replace('-','neg'),str(max_phi).replace('-','neg'))
            filter_string += phi_str

            theta_str = 'n_theta_%i-min_theta_%s-max_theta_%s-'%(n_theta, str(min_theta).replace('-','neg'),str(max_theta).replace('-','neg'))
            filter_string += theta_str

            filter_string += 'scope_%s'%(mapmax_cut_mode)

            print(filter_string)

            filter_strings.append(filter_string)

        try:
            run = int(run)

            reader = Reader(datapath,run)
            try:
                print(reader.status())
            except Exception as e:
                print('Status Tree not present.  Returning Error.')
                print('\nError in %s'%inspect.stack()[0][3])
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                sys.exit(1)
            filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
            eventids = getEventIds(reader,trigger_type=[1,2,3]) #eventids with rf trigger
            if numpy.size(eventids) != 0:
                print('run = ',run)

            if filename is not None:
                with h5py.File(filename, 'a') as file:
                    try:
                        dsets = list(file.keys()) #Existing datasets

                        if not numpy.isin('map_direction',dsets):
                            file.create_group('map_direction')
                        else:
                            print('map_direction group already exists in file %s'%filename)

                        map_direction_dsets = list(file['map_direction'].keys())

                        if not numpy.isin('map_times',dsets):
                            file.create_group('map_times')
                        else:
                            print('map_times group already exists in file %s'%filename)

                        map_times_dsets = list(file['map_times'].keys())

                        if not numpy.isin('peak_to_sidelobe',dsets):
                            file.create_group('peak_to_sidelobe')
                        else:
                            print('peak_to_sidelobe group already exists in file %s'%filename)

                        peak_to_sidelobe_dsets = list(file['peak_to_sidelobe'].keys())

                        if not numpy.isin('max_possible_map_value',dsets):
                            file.create_group('max_possible_map_value')
                        else:
                            print('max_possible_map_value group already exists in file %s'%filename)

                        max_possible_map_value_dsets = list(file['max_possible_map_value'].keys())

                        if not numpy.isin('map_properties',dsets):
                            file.create_group('map_properties')
                        else:
                            print('map_properties group already exists in file %s'%filename)

                        map_properties_dsets = list(file['map_properties'].keys())

                        file['map_direction'].attrs['sine_subtract_min_freq_GHz']   = sine_subtract_min_freq_GHz 
                        file['map_direction'].attrs['sine_subtract_max_freq_GHz']   = sine_subtract_max_freq_GHz 
                        file['map_direction'].attrs['sine_subtract_percent']        = sine_subtract_percent
                        file['map_times'].attrs['sine_subtract_min_freq_GHz']       = sine_subtract_min_freq_GHz 
                        file['map_times'].attrs['sine_subtract_max_freq_GHz']       = sine_subtract_max_freq_GHz 
                        file['map_times'].attrs['sine_subtract_percent']            = sine_subtract_percent

                        file['map_direction'].attrs['n_phi']        = n_phi
                        file['map_direction'].attrs['min_phi']      = min_phi
                        file['map_direction'].attrs['max_phi']      = max_phi

                        file['map_direction'].attrs['n_theta']      = n_theta
                        file['map_direction'].attrs['min_theta']    = min_theta
                        file['map_direction'].attrs['max_theta']    = max_theta

                        
                        for filter_string_index, filter_string in enumerate(filter_strings):
                            '''
                            Prepares output file for data.
                            '''

                            if not numpy.isin(filter_string,map_direction_dsets):
                                file['map_direction'].create_group(filter_string)
                            else:
                                print('%s group already exists in file %s'%(filter_string,filename))


                            map_direction_subsets = list(file['map_direction'][filter_string].keys())


                            if not numpy.isin(filter_string,map_times_dsets):
                                file['map_times'].create_group(filter_string)
                            else:
                                print('%s group already exists in file %s'%(filter_string,filename))

                            map_times_subsets = list(file['map_times'][filter_string].keys())

                            if not numpy.isin(filter_string,map_properties_dsets):
                                file['map_properties'].create_group(filter_string)
                            else:
                                print('%s group already exists in file %s'%(filter_string,filename))

                            map_properties_subsets = list(file['map_properties'][filter_string].keys())


                            #Directions
                            if not numpy.isin('hpol_ENU_azimuth',map_direction_subsets):
                                file['map_direction'][filter_string].create_dataset('hpol_ENU_azimuth', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in hpol_ENU_azimuth of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('hpol_ENU_zenith',map_direction_subsets):
                                file['map_direction'][filter_string].create_dataset('hpol_ENU_zenith', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in hpol_ENU_zenith of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('vpol_ENU_azimuth',map_direction_subsets):
                                file['map_direction'][filter_string].create_dataset('vpol_ENU_azimuth', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in vpol_ENU_azimuth of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('vpol_ENU_zenith',map_direction_subsets):
                                file['map_direction'][filter_string].create_dataset('vpol_ENU_zenith', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in vpol_ENU_zenith of %s will be overwritten by this analysis script.'%filename)

                            #Time Delays for directions
                            #01
                            if not numpy.isin('hpol_0subtract1',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('hpol_0subtract1', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in hpol_0subtract1 of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('vpol_0subtract1',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('vpol_0subtract1', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in vpol_0subtract1 of %s will be overwritten by this analysis script.'%filename)

                            #02
                            if not numpy.isin('hpol_0subtract2',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('hpol_0subtract2', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in hpol_0subtract2 of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('vpol_0subtract2',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('vpol_0subtract2', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in vpol_0subtract2 of %s will be overwritten by this analysis script.'%filename)

                            #03
                            if not numpy.isin('hpol_0subtract3',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('hpol_0subtract3', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in hpol_0subtract3 of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('vpol_0subtract3',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('vpol_0subtract3', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in vpol_0subtract3 of %s will be overwritten by this analysis script.'%filename)

                            #12
                            if not numpy.isin('hpol_1subtract2',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('hpol_1subtract2', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in hpol_1subtract2 of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('vpol_1subtract2',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('vpol_1subtract2', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in vpol_1subtract2 of %s will be overwritten by this analysis script.'%filename)

                            #13
                            if not numpy.isin('hpol_1subtract3',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('hpol_1subtract3', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in hpol_1subtract3 of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('vpol_1subtract3',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('vpol_1subtract3', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in vpol_1subtract3 of %s will be overwritten by this analysis script.'%filename)

                            #23
                            if not numpy.isin('hpol_2subtract3',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('hpol_2subtract3', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in hpol_2subtract3 of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('vpol_2subtract3',map_times_subsets):
                                file['map_times'][filter_string].create_dataset('vpol_2subtract3', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in vpol_2subtract3 of %s will be overwritten by this analysis script.'%filename)


                            #import pdb; pdb.set_trace()
                            #Map Properties
                            if not numpy.isin('hpol_peak_to_sidelobe',map_properties_subsets):
                                file['map_properties'][filter_string].create_dataset('hpol_peak_to_sidelobe', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in hpol_peak_to_sidelobe of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('vpol_peak_to_sidelobe',map_properties_subsets):
                                file['map_properties'][filter_string].create_dataset('vpol_peak_to_sidelobe', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in vpol_peak_to_sidelobe of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('hpol_max_possible_map_value',map_properties_subsets):
                                file['map_properties'][filter_string].create_dataset('hpol_max_possible_map_value', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in hpol_max_possible_map_value of %s will be overwritten by this analysis script.'%filename)

                            if not numpy.isin('vpol_max_possible_map_value',map_properties_subsets):
                                file['map_properties'][filter_string].create_dataset('vpol_max_possible_map_value', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in vpol_max_possible_map_value of %s will be overwritten by this analysis script.'%filename)

                            #Fill in attributes for the different map cuts
                            #Determine cut values
                            if mapmax_cut_modes[filter_string_index] == 'abovehorizon':
                                # print('abovehorizon')
                                zenith_cut_ENU=[0,90] #leaving some tolerance
                                zenith_cut_array_plane=None
                            elif mapmax_cut_modes[filter_string_index] == 'belowhorizon':
                                # print('belowhorizon')
                                zenith_cut_ENU=[90,180]
                                zenith_cut_array_plane=[0,91] #Up to 1 degree below projected array plane.
                            elif mapmax_cut_modes[filter_string_index] == 'allsky':
                                # print('allsky')
                                zenith_cut_ENU=None
                                zenith_cut_array_plane=[0,91] #Up to 1 degree below projected array plane.
                            else:
                                zenith_cut_ENU=None
                                zenith_cut_array_plane=None

                            #Record cut values, only needs to be done once, not per event                        
                            if zenith_cut_ENU is None:
                                file['map_direction'][filter_string].attrs['zenith_cut_ENU'] = 'None' 
                                file['map_times'][filter_string].attrs['zenith_cut_ENU'] = 'None' 
                            else:
                                _zenith_cut_ENU = []
                                if zenith_cut_ENU[0] is None:
                                    _zenith_cut_ENU.append('None')
                                else:
                                    _zenith_cut_ENU.append(zenith_cut_ENU[0])
                                if zenith_cut_ENU[1] is None:
                                    _zenith_cut_ENU.append('None')
                                else:
                                    _zenith_cut_ENU.append(zenith_cut_ENU[1])

                                file['map_direction'][filter_string].attrs['zenith_cut_ENU'] = _zenith_cut_ENU 
                                file['map_times'][filter_string].attrs['zenith_cut_ENU'] = _zenith_cut_ENU 

                            if zenith_cut_array_plane is None:
                                file['map_direction'][filter_string].attrs['zenith_cut_array_plane'] = 'None'
                                file['map_times'][filter_string].attrs['zenith_cut_array_plane'] = 'None' 
                            else:
                                _zenith_cut_array_plane = []
                                if zenith_cut_array_plane[0] is None:
                                    _zenith_cut_array_plane.append('None')
                                else:
                                    _zenith_cut_array_plane.append(zenith_cut_array_plane[0])
                                if zenith_cut_array_plane[1] is None:
                                    _zenith_cut_array_plane.append('None')
                                else:
                                    _zenith_cut_array_plane.append(zenith_cut_array_plane[1])


                                file['map_direction'][filter_string].attrs['zenith_cut_array_plane'] = _zenith_cut_array_plane
                                file['map_times'][filter_string].attrs['zenith_cut_array_plane'] = _zenith_cut_array_plane



                        for mode in polarizations:
                            if impose_time_limit is not None:
                                if datetime.timestamp(datetime.now()) - starting_timestamp >= 3600*impose_time_limit:
                                    print('\n\n')
                                    print('SELF IMPOSED TIME LIMIT REACHED, ENDING SCRIPT EARLY.  ENDING BEFORE CALCULATING MAPS FOR %s'%(mode))
                                    break

                            print('Performing calculations for %s'%mode)

                            if numpy.isin(run, known_pulser_runs):
                                waveform_index_range = index_window_dict[mode][pulser_run_sites[run]]
                                print('USING WAVEFORM_INDEX_RANGE OF ', str(waveform_index_range))

                                map_source_distance_m = direction_dict[pulser_run_sites[run]]['distance_m']
                                print('USING MAP SOURCE DISTANCE OF %0.2f m'%(map_source_distance_m))

                            else:
                                waveform_index_range = (100, 611)#This is handled inclusivelym meaning the 611th time is also included.  So 100-611 is a 512 long waveform starting at 200ns into the waveform.  This is shortened around the triggered waveform to improve map snr.  #(None, None) 
                                map_source_distance_m = 1e6


                            file['map_times'].attrs['waveform_index_range_%s'%mode]     = str(waveform_index_range)
                            file['map_direction'].attrs['waveform_index_range_%s'%mode] = str(waveform_index_range)

                            cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter, sine_subtract=sine_subtract, deploy_index=deploy_index,map_source_distance_m=map_source_distance_m)

                            print('Cor setup to use deploy index %s'%str(cor.deploy_index))

                            if sine_subtract:
                                cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                            print('Performing access sanity check:')
                            try:
                                file['map_properties'][filter_string]['%s_max_possible_map_value'%mode][0] = file['map_properties'][filter_string]['%s_max_possible_map_value'%mode][0]
                                file['map_direction'][filter_string]['%s_ENU_zenith'%mode][0] = file['map_direction'][filter_string]['%s_ENU_zenith'%mode][0]
                                file['map_direction'][filter_string]['%s_ENU_azimuth'%mode][0] = file['map_direction'][filter_string]['%s_ENU_azimuth'%mode][0]
                                file['map_times'][filter_string]['%s_0subtract1'%mode][0] = file['map_times'][filter_string]['%s_0subtract1'%mode][0]
                                file['map_times'][filter_string]['%s_0subtract2'%mode][0] = file['map_times'][filter_string]['%s_0subtract2'%mode][0]
                                file['map_times'][filter_string]['%s_0subtract3'%mode][0] = file['map_times'][filter_string]['%s_0subtract3'%mode][0]
                                file['map_times'][filter_string]['%s_1subtract2'%mode][0] = file['map_times'][filter_string]['%s_1subtract2'%mode][0]
                                file['map_times'][filter_string]['%s_1subtract3'%mode][0] = file['map_times'][filter_string]['%s_1subtract3'%mode][0]
                                file['map_times'][filter_string]['%s_2subtract3'%mode][0] = file['map_times'][filter_string]['%s_2subtract3'%mode][0]
                                file['map_properties'][filter_string]['%s_peak_to_sidelobe'%mode][0] = file['map_properties'][filter_string]['%s_peak_to_sidelobe'%mode][0]
                                print('Test successful for %s'%mode)
                            except Exception as e:
                                print('Print Failed')
                                print(e)

                            for event_index, eventid in enumerate(eventids):
                                if (event_index + 1) % 1000 == 0:
                                    sys.stdout.write('(%i/%i)\t\t\t\n'%(event_index+1,len(eventids)))
                                    sys.stdout.flush()


                                    # CONSIDER ADDING HDF5 FLUSH HERE

                                    if impose_time_limit is not None:
                                        if datetime.timestamp(datetime.now()) - starting_timestamp >= 3600*impose_time_limit:
                                            print('\n\n')
                                            print('SELF IMPOSED TIME LIMIT REACHED, ENDING SCRIPT EARLY.  MOST RECENT EVENTID COMPLETED IS %i'%(eventids[event_index-1]))
                                            break 

                                if debug == True:
                                    if event_index == 0:
                                        m, max_possible_map_value = cor.map(eventid, mode, plot_map=False, plot_corr=False, verbose=False, hilbert=hilbert,return_max_possible_map_value=True)
                                else:
                                    m, max_possible_map_value = cor.map(eventid, mode, plot_map=False, plot_corr=False, verbose=False, hilbert=hilbert,return_max_possible_map_value=True)

                                for filter_string_index, filter_string in enumerate(filter_strings):
                                    file['map_properties'][filter_string]['%s_max_possible_map_value'%mode][eventid] = max_possible_map_value #doesn't really depend on filter_string but would depend on calibration, so keeping it sorted.
                                    #Determine cut values
                                    if mapmax_cut_modes[filter_string_index] == 'abovehorizon':
                                        # print('abovehorizon')
                                        zenith_cut_ENU=[0,90] #leaving some tolerance
                                        zenith_cut_array_plane=None
                                    elif mapmax_cut_modes[filter_string_index] == 'belowhorizon':
                                        # print('belowhorizon')
                                        zenith_cut_ENU=[90,180]
                                        zenith_cut_array_plane=[0,91] #Up to 1 degree below projected array plane.
                                    elif mapmax_cut_modes[filter_string_index] == 'allsky':
                                        # print('allsky')
                                        zenith_cut_ENU=None
                                        zenith_cut_array_plane=[0,91] #Up to 1 degree below projected array plane.
                                    else:
                                        zenith_cut_ENU=None
                                        zenith_cut_array_plane=None

                                    #Calculate best reconstruction direction
                                    if debug == True:
                                        if event_index == 0:
                                            if max_method is not None:
                                                linear_max_index, theta_best, phi_best, t_0subtract1, t_0subtract2, t_0subtract3, t_1subtract2, t_1subtract3, t_2subtract3, peak_to_sidelobe = cor.mapMax(m,max_method=max_method,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol=mode, return_peak_to_sidelobe=True)
                                            else:
                                                linear_max_index, theta_best, phi_best, t_0subtract1, t_0subtract2, t_0subtract3, t_1subtract2, t_1subtract3, t_2subtract3, peak_to_sidelobe = cor.mapMax(m,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol=mode, return_peak_to_sidelobe=True)
                                    else:
                                        if max_method is not None:
                                            linear_max_index, theta_best, phi_best, t_0subtract1, t_0subtract2, t_0subtract3, t_1subtract2, t_1subtract3, t_2subtract3, peak_to_sidelobe = cor.mapMax(m,max_method=max_method,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol=mode, return_peak_to_sidelobe=True)
                                        else:
                                            linear_max_index, theta_best, phi_best, t_0subtract1, t_0subtract2, t_0subtract3, t_1subtract2, t_1subtract3, t_2subtract3, peak_to_sidelobe = cor.mapMax(m,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol=mode, return_peak_to_sidelobe=True)

                                    if debug == True:
                                        if event_index == 0:
                                            file['map_direction'][filter_string]['%s_ENU_zenith'%mode][eventid] = theta_best 
                                            file['map_direction'][filter_string]['%s_ENU_azimuth'%mode][eventid] = phi_best 
                                            file['map_times'][filter_string]['%s_0subtract1'%mode][eventid] = t_0subtract1 
                                            file['map_times'][filter_string]['%s_0subtract2'%mode][eventid] = t_0subtract2 
                                            file['map_times'][filter_string]['%s_0subtract3'%mode][eventid] = t_0subtract3 
                                            file['map_times'][filter_string]['%s_1subtract2'%mode][eventid] = t_1subtract2 
                                            file['map_times'][filter_string]['%s_1subtract3'%mode][eventid] = t_1subtract3 
                                            file['map_times'][filter_string]['%s_2subtract3'%mode][eventid] = t_2subtract3
                                            file['map_properties'][filter_string]['%s_peak_to_sidelobe'%mode][eventid] = peak_to_sidelobe
                                    else:
                                        file['map_direction'][filter_string]['%s_ENU_zenith'%mode][eventid] = theta_best 
                                        file['map_direction'][filter_string]['%s_ENU_azimuth'%mode][eventid] = phi_best 
                                        file['map_times'][filter_string]['%s_0subtract1'%mode][eventid] = t_0subtract1 
                                        file['map_times'][filter_string]['%s_0subtract2'%mode][eventid] = t_0subtract2 
                                        file['map_times'][filter_string]['%s_0subtract3'%mode][eventid] = t_0subtract3 
                                        file['map_times'][filter_string]['%s_1subtract2'%mode][eventid] = t_1subtract2 
                                        file['map_times'][filter_string]['%s_1subtract3'%mode][eventid] = t_1subtract3 
                                        file['map_times'][filter_string]['%s_2subtract3'%mode][eventid] = t_2subtract3
                                        file['map_properties'][filter_string]['%s_peak_to_sidelobe'%mode][eventid] = peak_to_sidelobe


                        file.close()
                    except Exception as e:
                        file.close()
                        print('\nError in %s'%inspect.stack()[0][3])
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        sys.exit(1)
            else:
                print('filename is None, indicating empty tree.  Skipping run %i'%run)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(1)

