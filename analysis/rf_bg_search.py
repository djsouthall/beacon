#!/usr/bin/env python3
'''
This script is meant to determine where each map points for each event and then save these values to file. 
'''
import os
import sys
import h5py
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
from tools.data_handler import createFile

import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import numpy
import sys
import math
import matplotlib
from scipy.fftpack import fft
import datetime as dt
import inspect
from ast import literal_eval
import astropy.units as apu
from astropy.coordinates import SkyCoord
plt.ion()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tools.correlator import Correlator


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


if __name__=="__main__":
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 1701

    datapath = os.environ['BEACON_DATA']
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

    upsample = 2**14 #Just upsample in this case
    max_method = 0


    #This code will loop over all options included here, and they will be stored as seperate dsets.  Each of these applies different cuts to mapmax when it is attempting to select the best reconstruction direction.
    mapmax_cut_modes = ['abovehorizon','belowhorizon','allsky'] 
    polarizations = ['hpol','vpol'] #Will loop over both if hpol and vpol present
    hilbert_modes = [False]#[True,False] #Will loop over both if True and False present
    deploy_index = None #None results in default_deploy
    
    #Note that the below values set the angular resolution of the plot, while the presets from mapmax_cut_modes limit where in the generated plots will be considered for max values.
    n_phi       = 1440
    min_phi     = -180
    max_phi     = 180

    n_theta     = 720
    min_theta   = 0
    max_theta   = 180



    if deploy_index is None:
        deploy_index = info.returnDefaultDeploy()

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

            filter_string += 'deploy_calibration_%s-'%(str(deploy_index))

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

                        for filter_string in filter_strings:
                            '''
                            Prepares output file for data.
                            '''

                            if not numpy.isin(filter_string,map_direction_dsets):
                                file['map_direction'].create_group(filter_string)
                            else:
                                print('%s group already exists in file %s'%(filter_string,filename))

                            map_direction_subsets = list(file['map_direction'][filter_string].keys())



                            if not numpy.isin('map_times',dsets):
                                file.create_group('map_times')
                            else:
                                print('map_times group already exists in file %s'%filename)

                            map_times_dsets = list(file['map_times'].keys())

                            if not numpy.isin(filter_string,map_times_dsets):
                                file['map_times'].create_group(filter_string)
                            else:
                                print('%s group already exists in file %s'%(filter_string,filename))

                            map_times_subsets = list(file['map_times'][filter_string].keys())


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


                        cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter, sine_subtract=sine_subtract, deploy_index=deploy_index)

                        print('Cor setup to use deploy index %i'%cor.deploy_index)

                        if sine_subtract:
                            cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                        for mode in polarizations:
                            print('Performing calculations for %s'%mode)
                            for event_index, eventid in enumerate(eventids):
                                if (event_index + 1) % 1000 == 0:
                                    sys.stdout.write('(%i/%i)\t\t\t\n'%(event_index+1,len(eventids)))
                                    sys.stdout.flush()
                                m = cor.map(eventid, mode, plot_map=False, plot_corr=False, verbose=False, hilbert=hilbert)

                                for filter_string_index, filter_string in enumerate(filter_strings):

                                    #Determine cut values
                                    if mapmax_cut_modes[filter_string_index] == 'abovehorizon':
                                        # print('abovehorizon')
                                        zenith_cut_ENU=[0,90] #leaving some tolerance
                                        zenith_cut_array_plane=None
                                    elif mapmax_cut_modes[filter_string_index] == 'belowhorizon':
                                        # print('belowhorizon')
                                        zenith_cut_ENU=[90,180]
                                        zenith_cut_array_plane=[0,92] #Up to 2 degrees below projectd array plane.
                                    elif mapmax_cut_modes[filter_string_index] == 'allsky':
                                        # print('allsky')
                                        zenith_cut_ENU=None
                                        zenith_cut_array_plane=[0,100]
                                    else:
                                        zenith_cut_ENU=None
                                        zenith_cut_array_plane=None

                                    #Record cut values                                    
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

                                    #Calculate best reconstruction direction
                                    if max_method is not None:
                                        linear_max_index, theta_best, phi_best, t_0subtract1, t_0subtract2, t_0subtract3, t_1subtract2, t_1subtract3, t_2subtract3 = cor.mapMax(m,max_method=max_method,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol=mode)
                                    else:
                                        linear_max_index, theta_best, phi_best, t_0subtract1, t_0subtract2, t_0subtract3, t_1subtract2, t_1subtract3, t_2subtract3 = cor.mapMax(m,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol=mode)        
                                    file['map_direction'][filter_string]['%s_ENU_zenith'%mode][eventid] = theta_best 
                                    file['map_direction'][filter_string]['%s_ENU_azimuth'%mode][eventid] = phi_best 
                                    file['map_times'][filter_string]['%s_0subtract1'%mode][eventid] = t_0subtract1 
                                    file['map_times'][filter_string]['%s_0subtract2'%mode][eventid] = t_0subtract2 
                                    file['map_times'][filter_string]['%s_0subtract3'%mode][eventid] = t_0subtract3 
                                    file['map_times'][filter_string]['%s_1subtract2'%mode][eventid] = t_1subtract2 
                                    file['map_times'][filter_string]['%s_1subtract3'%mode][eventid] = t_1subtract3 
                                    file['map_times'][filter_string]['%s_2subtract3'%mode][eventid] = t_2subtract3 

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

