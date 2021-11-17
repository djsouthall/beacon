#!/usr/bin/env python3
'''
This script is test the saved values from it's counterpart analyze script.
'''
import os
import sys
import h5py

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])

from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from tools.data_handler import createFile, getTimes, loadTriggerTypes, getEventTimes

import matplotlib.pyplot as plt
plt.ion()
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
        sys.exit(1)

    print('Running analyze_event_rate_frequency.py')

    expected_rates_hz = [60.0] #The plotter will perform calculations assuming a frequency of 1/thing numbers.  Currently the output dataset is named using these values to 3 decimal places, so be weary of overlapping rates that are extremely precise.
    time_windows = [20] #The test statistic will be calculated for each event window given here.  Select integers.
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
            with h5py.File(filename, 'r') as file:
                try:
                    #Loop back through settings for actual calculations
                    load_cut = file['trigger_type'][...] == 2
                    loaded_eventids = numpy.where(load_cut)[0]
                    calibrated_trig_time = file['calibrated_trigtime'][load_cut]
                    randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))

                    for rate_hz in expected_rates_hz:
                        rate_string = '%0.3fHz'%rate_hz
                        for time_window in time_windows:
                            time_window_string = '%is'%time_window
                            #Store fit of randomized values for this run to use for reference in further cutting or calculations.
                            popt = [0,0,0]
                            popt[0] = file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_scale']
                            popt[1] = file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_mean']
                            popt[2] = file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_sigma']
                            normalize_by_window_index = file['event_rate_testing'][rate_string][time_window_string].attrs['normalize_by_window_index']

                            metric_true = file['event_rate_testing'][rate_string][time_window_string][...][loaded_eventids]

                            sigma = (metric_true - popt[1])/popt[2]

                            plt.figure()
                            plt.subplot(2,1,1)
                            counts, bins, patches = plt.hist(metric_true,bins=100)
                            x = (bins[1:len(bins)] + bins[0:len(bins)-1] )/2.0
                            plot_x = numpy.linspace(min(x),max(x),1000)
                            plt.plot(plot_x,gaus(plot_x,*popt),'-',c='k',label='Fit Sigma = %f ns'%popt[2])
                            plt.axvline(popt[1],linestyle='-',c='k',label='Randomized Baseline Fit Center = %f'%(popt[1]))
                            plt.axvline(popt[1] + popt[2],linestyle='--',c='k',label='Randomized Baseline Fit Sigma = %f'%(popt[2]))
                            plt.axvline(popt[1] - popt[2],linestyle='--',c='k')
                            plt.legend()
                            plt.subplot(2,1,2)
                            plt.hist(sigma,bins=100)


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

