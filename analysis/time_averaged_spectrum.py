#!/usr/bin/env python3
'''
This is meant to be used for tracking galactic noise, by averaging spectrum and saving the outcome.
'''
import os
import sys
import h5py
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
from tools.fftmath import TimeDelayCalculator
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
import itertools
plt.ion()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 16})


if __name__=="__main__":
    if len(sys.argv) >= 2:
        run = int(sys.argv[1])
    else:
        print('NO RUN SELECTED')

    datapath = os.environ['BEACON_DATA']

    try:



        crit_freq_low_pass_MHz = None#
        low_pass_filter_order = None#

        crit_freq_high_pass_MHz = None#60
        high_pass_filter_order = None#6

        apply_phase_response = True
        hilbert = False

        sine_subtract = True
        sine_subtract_min_freq_GHz = 0.03
        sine_subtract_max_freq_GHz = 0.09
        sine_subtract_percent = 0.03

        final_corr_length = 2**14

        run = int(run)
        time_window = 5*60 #seconds
        frequency_bin_edges_MHz = numpy.arange(0,150,5)
        antennas_of_interest = numpy.array([4,5])
        trigger_types_of_interest = numpy.array([1,3])
        save = True
        plot = False
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

        if filename is not None:
            with h5py.File(filename, 'a') as file:
                tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=(None,None), plot_filters=False, apply_phase_response=apply_phase_response)
                if sine_subtract:
                    tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
                #eventids with rf trigger
                eventids = file['eventids'][...]
                trigger_type_cut = numpy.isin(file['trigger_type'][...], trigger_types_of_interest)
                # plt.figure()
                # plt.hist(file['trigger_type'][...])
                # plt.title('%i in trigger_types_of_interest'%sum(trigger_type_cut))
                event_times = file['calibrated_trigtime'][...]

                if numpy.size(eventids) != 0:
                    print('run = ',run)
                dsets = list(file.keys()) #Existing datasets
                
                if not numpy.isin('spectral_bin_average_series',dsets):
                    file.create_group('spectral_bin_average_series')
                else:
                    print('spectral_bin_average_series group already exists in file %s'%filename)

                spectral_bin_average_series_dsets = list(file['spectral_bin_average_series'].keys())
                
                #Check for frequency bins and store them
                if not numpy.isin('frequency_bin_edges_MHz',spectral_bin_average_series_dsets):
                    file['spectral_bin_average_series'].create_dataset('frequency_bin_edges_MHz', (len(frequency_bin_edges_MHz),), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('spectral_bin_average_series["%s"] group already exists in file %s, replacing'%('frequency_bin_edges_MHz',filename))
                    del file['spectral_bin_average_series']['frequency_bin_edges_MHz']
                    
                    file['spectral_bin_average_series'].create_dataset('frequency_bin_edges_MHz', (len(frequency_bin_edges_MHz),), dtype='f', compression='gzip', compression_opts=4, shuffle=True)

                file['spectral_bin_average_series']['frequency_bin_edges_MHz'][...] = frequency_bin_edges_MHz

                frequency_bin_centers_MHz = (frequency_bin_edges_MHz[:-1] + frequency_bin_edges_MHz[1:]) / 2

                #Get time windows (event cuts for each average)
                time_window_edges = numpy.arange(file['calibrated_trigtime'][0], file['calibrated_trigtime'][-1], time_window)
                time_window_centers = (time_window_edges[:-1] + time_window_edges[1:]) / 2

                #Check for time bins and store them
                if not numpy.isin('time_window_centers',spectral_bin_average_series_dsets):
                    file['spectral_bin_average_series'].create_dataset('time_window_centers', (len(time_window_centers),), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('spectral_bin_average_series["%s"] group already exists in file %s, replacing'%('time_window_centers',filename))
                    del file['spectral_bin_average_series']['time_window_centers']
                    file['spectral_bin_average_series'].create_dataset('time_window_centers', (len(time_window_centers),), dtype='f', compression='gzip', compression_opts=4, shuffle=True)

                file['spectral_bin_average_series']['time_window_centers'][...] = time_window_centers

                #prepare the antenna dependant groups if they don't already exist.
                for antenna_index, antenna in enumerate(antennas_of_interest):
                    data_set_name = 'ant%i'%antenna

                    if not numpy.isin(data_set_name,spectral_bin_average_series_dsets):
                        file['spectral_bin_average_series'].create_dataset(data_set_name, (len(time_window_edges)-1, len(frequency_bin_edges_MHz)-1), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                    else:
                        print('spectral_bin_average_series["%s"] group already exists in file %s, deleting'%(data_set_name,filename))
                        del file['spectral_bin_average_series'][data_set_name]
                        file['spectral_bin_average_series'].create_dataset(data_set_name, (len(time_window_edges)-1, len(frequency_bin_edges_MHz)-1), dtype='f', compression='gzip', compression_opts=4, shuffle=True)

                #Determine frequency indices for which to average.
                #Each row should contain all frequency indices that should be averaged for that bin.
                freq_bin_indices = []
                for freq_bin_index in range(len(frequency_bin_edges_MHz) - 1):
                    freq_bin_indices.append(numpy.where(numpy.logical_and(tdc.freqs_corr >= frequency_bin_edges_MHz[freq_bin_index]*1e6, tdc.freqs_corr < frequency_bin_edges_MHz[freq_bin_index+1]*1e6))[0])

                print(freq_bin_indices)
                # Each matrix is an antenna.  Then each row is a time window, each column is a freq window.  Add to these just the value, then divide each
                #row by the number of events in the cut to get the average.  Be sure to include just the events that
                #match the trigger type cut as well. 
                averaged_spectral_values = numpy.zeros((len(antennas_of_interest), len(time_window_edges)-1, len(frequency_bin_edges_MHz)-1))

                print('Looping over time windows')
                for time_window_index in range(len(time_window_edges)-1):
                    print('%i/%i'%(time_window_index+1, len(time_window_edges)-1))
                    time_cut = numpy.logical_and(event_times >= time_window_edges[time_window_index], event_times <= time_window_edges[time_window_index+1])
                    cut = numpy.logical_and(time_cut,trigger_type_cut)
                    average_factor = sum(cut)

                    for eventid in eventids[cut]:
                        ffts = tdc.loadFilteredFFTs(eventid, hilbert=hilbert, sine_subtract=sine_subtract)
                        for antenna_index, antenna in enumerate(antennas_of_interest):
                            for freq_bin_index in range(len(frequency_bin_edges_MHz) - 1):
                                averaged_spectral_values[antenna_index][time_window_index][freq_bin_index] += numpy.mean(ffts[antenna_index][freq_bin_indices[freq_bin_index]])/average_factor #mean in that window then divided by number of events to be averaged in time window
                
                for antenna_index, antenna in enumerate(antennas_of_interest):
                    data_set_name = 'ant%i'%antenna
                    file['spectral_bin_average_series'][data_set_name][...] = averaged_spectral_values[antenna_index]

                file.close()
        else:
            print('filename is None, indicating empty tree.  Skipping run %i'%run)

        if plot == True:
            for antenna_index, antenna in enumerate(antennas_of_interest):
                plt.figure()
                plt.title('Antenna %i'%antenna)
                for freq_index, freq in enumerate(frequency_bin_centers_MHz):
                    plt.plot((time_window_centers - time_window_edges[0])/3600.0,averaged_spectral_values[antenna_index][:,freq_index] - numpy.mean(averaged_spectral_values[antenna_index][:,freq_index]),label='%0.2f MHz'%freq)
                plt.xlabel('Time Since First Event (hours)')
                plt.ylabel('Average FFT Value in Freq Bin\n(mean subtracted) (arb)')
                plt.grid(which='both', axis='both')
                ax1.minorticks_on()
                ax1.grid(b=True, which='major', color='k', linestyle='-')
                ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.legend()

    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)
    sys.exit(0)

