#!/usr/bin/env python3
'''
This script is intended to be a batchable version of background_indentify_60hz. 
'''
import os
import sys
import h5py

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from tools.data_handler import createFile, getTimes, loadTriggerTypes, getEventTimes

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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tools.correlator import Correlator
from analysis.background_identify_60hz import *
def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))


if __name__=="__main__":
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        print('No run number given, returning without calculations.')

    print('Running analyze_event_rate_frequency.py for run %i'%run)

    expected_rates_hz = [60.0] #The plotter will perform calculations assuming a frequency of 1/thing numbers.  Currently the output dataset is named using these values to 3 decimal places, so be weary of overlapping rates that are extremely precise.
    time_windows = [5,10,20] #The test statistic will be calculated for each event window given here.  Select integers.
    datapath = os.environ['BEACON_DATA']
    normalize_by_window_index = True


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

        if filename is not None:
            with h5py.File(filename, 'a') as file:
                try:
                    dsets = list(file.keys()) #Existing datasets

                    if not numpy.isin('event_rate_testing',dsets):
                        file.create_group('event_rate_testing')
                    else:
                        print('event_rate_testing group already exists in file %s'%filename)

                    event_rate_testing_dsets = list(file['event_rate_testing'].keys())

                    #file['event_rate_testing'].attrs['sample']   = something

                    #Just make the output datasets in advance
                    for rate_hz in expected_rates_hz:
                        '''
                        Prepares output file for data.
                        '''
                        rate_string = '%0.3fHz'%rate_hz

                        if not numpy.isin(rate_string,event_rate_testing_dsets):
                            file['event_rate_testing'].create_group(rate_string)
                        else:
                            print('%s group already exists in file %s'%(rate_string,filename))

                        event_rate_testing_subsets = list(file['event_rate_testing'][rate_string].keys())
                        
                        for time_window in time_windows:
                            time_window_string = '%is'%time_window

                            if not numpy.isin(time_window_string,event_rate_testing_subsets):
                                file['event_rate_testing'][rate_string].create_dataset(time_window_string, (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                            else:
                                print('Values in %s of %s will be overwritten by this analysis script.'%(time_window_string, filename))


                    #Loop back through settings for actual calculations
                    load_cut = file['trigger_type'][...] == 2
                    loaded_eventids = numpy.where(load_cut)[0]
                    calibrated_trig_time = file['calibrated_trigtime'][load_cut]
                    randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))

                    for rate_hz in expected_rates_hz:
                        rate_string = '%0.3fHz'%rate_hz
                        for time_window in time_windows:
                            time_window_string = '%is'%time_window

                            sys.stdout.write('Running for rate %s and window %s\n'%(rate_string,time_window_string))
                            sys.stdout.flush()
                            #file['event_rate_testing'][rate_string][time_window_string][eventid] = 0.0
                            metric_true = diffFromPeriodic(calibrated_trig_time,window_s=time_window, atol=0.001, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=False, plot_test_plot_A=False)
                            metric_rand = diffFromPeriodic(randomized_times,window_s=time_window, atol=0.001, normalize_by_window_index=normalize_by_window_index) 

                            bins = numpy.linspace(min(min(metric_true),min(metric_rand)),max(max(metric_true),max(metric_rand)),100)
                            #counts_true, bins_true = numpy.histogram(metric_true,bins=bins)
                            counts_rand, bins_rand = numpy.histogram(metric_rand,bins=bins)

                            #Fit Gaussian
                            x = (bins_rand[1:len(bins_rand)] + bins_rand[0:len(bins_rand)-1] )/2.0
                            plot_x = numpy.linspace(min(x),max(x),1000)

                            try:
                                popt, pcov = curve_fit(gaus,x,counts_rand,p0=[numpy.max(counts_rand),0.0,0.5*numpy.mean(metric_rand[metric_rand > 0])])
                                popt[2] = abs(popt[2]) #I want positive sigma.

                            except Exception as e:
                                try:
                                    popt, pcov = curve_fit(gaus,x,counts_rand,p0=[numpy.max(counts_rand),0.0,0.025])
                                    popt[2] = abs(popt[2]) #I want positive sigma.
                                except Exception as e:
                                    popt = (numpy.max(counts_rand),0.0,0.025)
                                    print('Failed at deepest level, using dummy gaus vallues too.')
                                    print(e)

                            #Store fit of randomized values for this run to use for reference in further cutting or calculations.
                            file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_scale'] = popt[0]
                            file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_mean'] = popt[1]
                            file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_sigma'] = popt[2]
                            file['event_rate_testing'][rate_string][time_window_string].attrs['window_s'] = time_window
                            file['event_rate_testing'][rate_string][time_window_string].attrs['rate_hz'] = rate_hz
                            file['event_rate_testing'][rate_string][time_window_string].attrs['normalize_by_window_index'] = normalize_by_window_index

                            file['event_rate_testing'][rate_string][time_window_string][loaded_eventids] = metric_true


                except Exception as e:
                    print('Error while file open, closing file.')
                    file.close()
                    print('\nError in %s'%inspect.stack()[0][3])
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    sys.exit(1)
                    #------------------

                file.close()
        else:
            print('filename is None, indicating empty tree.  Skipping run %i'%run)
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)

