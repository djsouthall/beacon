#!/usr/bin/env python3
'''
This is a followup script for rf_bg_script, which takes the values of that and attempts to select the preferred
map value.  This is meant to avoid recalculating the maps and be minimal in calculation.  This should be run after
all polarizations maps are completed.

The results will be stored in a new datasat per filterstring (ignoring scope, which is consolodated).  The dataset
will be named with the new scope "best"
'''
import os
import sys
import h5py
#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
import beacon.tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import beacon.tools.info as info
from beacon.tools.data_handler import createFile
from beacon.analysis.rf_bg_search import getEventIds

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
from pprint import pprint


font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 16})

datapath = os.environ['BEACON_DATA']

if __name__=="__main__":
    debug = False
    if debug == True:
        print("WARNING DEBUG MODE ENABLED, NO DATA WILL BE SAVED")
        readmode = 'r'
    else:
        readmode = 'a'

    print('Running rf_bg_search_calculate_best.py')
    if len(sys.argv) > 1:
        run = int(sys.argv[1])

    possible_polarizations = ['hpol', 'vpol', 'all'] #used to exclude best case scenario created here
    possible_mapmax_cut_modes = ['abovehorizon','belowhorizon','allsky']
    # mapmax_cut_modes = ['abovehorizon','belowhorizon','allsky']
    hilbert_modes = [False]
    #numpy.arange(5733,5974,dtype=int)
    _runs = run
    for run in [_runs]:
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
                with h5py.File(filename, readmode) as file:
                    try:
                        dsets = list(file.keys()) #Existing datasets
                        map_direction_dsets = list(file['map_direction'].keys())

                        #Get filter strings but ignoring the scope. Each filter string but with scope subbed put will have
                        #the same result.
                        filter_string_roots = numpy.unique([m.replace('allsky','SCOPEROOT').replace('belowhorizon','SCOPEROOT').replace('abovehorizon','SCOPEROOT').replace('best','SCOPEROOT') for m in map_direction_dsets])

                        #Check which roots have all necessary datasets
                        paired_options = {}
                        for filter_string_root in filter_string_roots:
                            output_filter_string = filter_string_root.replace('SCOPEROOT','best')
                            #Check if the output dataset already exist
                            if not numpy.isin(output_filter_string,map_direction_dsets):
                                print('Creating %s group'%output_filter_string)
                                file['map_direction'].create_group(output_filter_string)

                            subsets = list(file['map_direction'][output_filter_string])

                            if debug == False:
                                if not numpy.isin('best_ENU_azimuth',subsets):
                                    print('Creating %s'%'best_ENU_azimuth')
                                    file['map_direction'][output_filter_string].create_dataset('best_ENU_azimuth', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                                else:
                                    print('Values in best_ENU_azimuth of %s will be overwritten by this analysis script.'%filename)

                                if not numpy.isin('best_ENU_zenith',subsets):
                                    print('Creating %s'%'best_ENU_zenith')
                                    file['map_direction'][output_filter_string].create_dataset('best_ENU_zenith', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                                else:
                                    print('Values in best_ENU_zenith of %s will be overwritten by this analysis script.'%filename)


                            mapmax_cut_modes = []
                            for filter_string in map_direction_dsets:
                                if filter_string_root.split('SCOPEROOT')[0] in filter_string and filter_string_root.split('SCOPEROOT')[1] in filter_string and filter_string != output_filter_string:
                                    mapmax_cut_modes.append(filter_string.replace(filter_string_root.split('SCOPEROOT')[0],'').replace(filter_string_root.split('SCOPEROOT')[1],''))
                                    keys = list(file['map_direction'][filter_string].keys()) #Must be inthe truth of this scoperoot
                            mapmax_cut_modes = numpy.array(mapmax_cut_modes)[numpy.isin(mapmax_cut_modes,possible_mapmax_cut_modes)] #Cut out the scope called 'best' created by this. 

                            #Now each scope is found, below is each polarization in the final scope's list of datasets.
                            polarizations = []
                            for key in keys:
                                if 'azimuth' in key:
                                    polarizations.append(key.replace('_ENU_azimuth',''))

                            polarizations = numpy.array(polarizations)[numpy.isin(polarizations,possible_polarizations)]

                            # The maximum value from each combination in paired_options will be stored as the optimal value. This will be done per filter_string_root.
                            paired_options[filter_string_root] = {}
                            paired_options[filter_string_root]['polarizations'] = polarizations
                            paired_options[filter_string_root]['mapmax_cut_modes'] = mapmax_cut_modes
                            paired_options[filter_string_root]['pairs'] = []

                            for polarization_mode in paired_options[filter_string_root]['polarizations']:
                                for mapmax_mode in paired_options[filter_string_root]['mapmax_cut_modes']:
                                    paired_options[filter_string_root]['pairs'].append([mapmax_mode, polarization_mode])


                        for filter_string_root in filter_string_roots:
                            output_filter_string = filter_string_root.replace('SCOPEROOT','best')
                            print('Making calculations for %s'%output_filter_string)

                            print('Best pulled from following combinations for %s'%output_filter_string)
                            pprint(paired_options[filter_string_root]['pairs'])

                            peak_to_sidelobes = numpy.array([None]*len(paired_options[filter_string_root]['pairs']))
                            mapmax_values = numpy.array([None]*len(paired_options[filter_string_root]['pairs']))
                            if debug == True:
                                zeniths = numpy.array([None]*len(paired_options[filter_string_root]['pairs']))
                                azimuths = numpy.array([None]*len(paired_options[filter_string_root]['pairs']))
                            for event_index, eventid in enumerate(eventids):
                                if debug == False:
                                    if (event_index + 1) % 100 == 0:
                                        sys.stdout.write('(%i/%i)\t\t\t\n'%(event_index+1,len(eventids)))
                                        sys.stdout.flush()
                                    for option_index, (mapmax_mode, polarization_mode) in enumerate(paired_options[filter_string_root]['pairs']):
                                        try:
                                            peak_to_sidelobes[option_index] = file['map_properties'][filter_string_root.replace('SCOPEROOT',mapmax_mode)]['%s_peak_to_sidelobe'%polarization_mode][eventid]
                                        except:
                                            peak_to_sidelobes[option_index] = 1.0
                                        try:
                                            mapmax_values[option_index] = file['map_properties'][filter_string_root.replace('SCOPEROOT',mapmax_mode)]['%s_max_map_value'%polarization_mode][eventid]
                                        except:
                                            mapmax_values[option_index] = 0.01 #Not 0 so multiplying doesn't erase p2sidelobe info, and not 1 just incase other mapmax value exist for some reason and this one probably shouldn't be interpreted as optimal
                                                    
                                    optimal_index = numpy.argmax(peak_to_sidelobes*mapmax_values)
                                    
                                    zenith = file['map_direction'][filter_string_root.replace('SCOPEROOT',paired_options[filter_string_root]['pairs'][optimal_index][0])]['%s_ENU_zenith'%paired_options[filter_string_root]['pairs'][optimal_index][1]][eventid]
                                    azimuth = file['map_direction'][filter_string_root.replace('SCOPEROOT',paired_options[filter_string_root]['pairs'][optimal_index][0])]['%s_ENU_azimuth'%paired_options[filter_string_root]['pairs'][optimal_index][1]][eventid]

                                    file['map_direction'][output_filter_string]['best_ENU_zenith'][eventid] = zenith
                                    file['map_direction'][output_filter_string]['best_ENU_azimuth'][eventid] = azimuth

                                elif debug == True:
                                    if eventid not in [ 2865, 4053,  6438, 16956, 45475]:
                                        continue
                                    elif 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_SCOPEROOT':
                                        if True:
                                            option_index = 0
                                            for polarization_mode in paired_options[filter_string_root]['polarizations']:
                                                for mapmax_mode in paired_options[filter_string_root]['mapmax_cut_modes']:
                                                    try:
                                                        peak_to_sidelobes[option_index] = file['map_properties'][filter_string_root.replace('SCOPEROOT',mapmax_mode)]['%s_peak_to_sidelobe'%polarization_mode][eventid]
                                                    except:
                                                        peak_to_sidelobes[option_index] = 1.0
                                                    try:
                                                        mapmax_values[option_index] = file['map_properties'][filter_string_root.replace('SCOPEROOT',mapmax_mode)]['%s_max_map_value'%polarization_mode][eventid]
                                                    except:
                                                        mapmax_values[option_index] = 0.01 #Not 0 so multiplying doesn't erase p2sidelobe info, and not 1 just incase other mapmax value exist for some reason and this one probably shouldn't be interpreted as optimal
                                                    
                                                    zeniths[option_index] = file['map_direction'][filter_string_root.replace('SCOPEROOT',mapmax_mode)]['%s_ENU_zenith'%polarization_mode][eventid]
                                                    azimuths[option_index] = file['map_direction'][filter_string_root.replace('SCOPEROOT',mapmax_mode)]['%s_ENU_azimuth'%polarization_mode][eventid]
                                                    option_index += 1

                                            metric = peak_to_sidelobes*mapmax_values
                                        else:
                                            option_index = 0
                                            for polarization_mode in paired_options[filter_string_root]['polarizations']:
                                                max_value_per_pol = -1e6
                                                start_index = option_index
                                                for mapmax_mode in paired_options[filter_string_root]['mapmax_cut_modes']:
                                                    peak_to_sidelobes[option_index] = file['map_properties'][filter_string_root.replace('SCOPEROOT',mapmax_mode)]['%s_peak_to_sidelobe'%polarization_mode][eventid]
                                                    mapmax_values[option_index] = file['map_properties'][filter_string_root.replace('SCOPEROOT',mapmax_mode)]['%s_max_map_value'%polarization_mode][eventid]
                                                    if mapmax_values[option_index] > max_value_per_pol:
                                                        max_value_per_pol = mapmax_values[option_index]

                                                    zeniths[option_index] = file['map_direction'][filter_string_root.replace('SCOPEROOT',mapmax_mode)]['%s_ENU_zenith'%polarization_mode][eventid]
                                                    azimuths[option_index] = file['map_direction'][filter_string_root.replace('SCOPEROOT',mapmax_mode)]['%s_ENU_azimuth'%polarization_mode][eventid]

                                                    option_index += 1

                                                option_index = start_index 

                                                for mapmax_mode in paired_options[filter_string_root]['mapmax_cut_modes']:
                                                    mapmax_values[option_index] = mapmax_values[option_index]/max_value_per_pol #Each max map represented as a portion of the max for that polarization regardless of map scope
                                                    option_index += 1

                                            metric = (peak_to_sidelobes/peak_to_sidelobes.max())*(mapmax_values)

                                        elevations = 90 - zeniths
                                        azimuths = azimuths
                                        optimal_index = numpy.argmax(metric)
                                        zenith = file['map_direction'][filter_string_root.replace('SCOPEROOT',paired_options[filter_string_root]['pairs'][optimal_index][0])]['%s_ENU_zenith'%paired_options[filter_string_root]['pairs'][optimal_index][1]][eventid]
                                        azimuth = file['map_direction'][filter_string_root.replace('SCOPEROOT',paired_options[filter_string_root]['pairs'][optimal_index][0])]['%s_ENU_azimuth'%paired_options[filter_string_root]['pairs'][optimal_index][1]][eventid]
                                        print(90.0-zenith,azimuth)
                                        if eventid == 45475:
                                            for i in list(zip(paired_options[filter_string_root]['pairs'],  elevations, azimuths, metric, peak_to_sidelobes, mapmax_values)):
                                                print(i)
                                            import pdb; pdb.set_trace()
                                        elif eventid == 2865:
                                            for i in list(zip(paired_options[filter_string_root]['pairs'],  elevations, azimuths, metric, peak_to_sidelobes, mapmax_values)):
                                                print(i)
                                            import pdb; pdb.set_trace()

                        file.close()
                    except Exception as e:
                        file.close()
                        print('\nError in %s'%inspect.stack()[0][3])
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        #sys.exit(1)
            else:
                print('filename is None, indicating empty tree.  Skipping run %i'%run)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            #sys.exit(1)

