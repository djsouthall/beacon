#!/usr/bin/env python3
'''
This is a script that aims to identify noise that is of the 60 Hz category.
It will first attempt to do this by plotting subsecond time v.s. time, and
looking for regular near horizontal clusters.  Things that arrive at regular
intervales near 60 Hz are likely coming from the same source.  With this, a
template search could potentially be performed, or time delays, to further
clean up the events as either 60 Hz or not. 
'''
'''
When calibrating the antenna positions I am seeing two peaks on correlation histograms for 
antennas 0 and 4.  I am using this to explore an characteristic differences between signals
in each peak. 
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info
from tools.data_handler import createFile, getTimes
from objects.fftmath import FFTPrepper

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
import h5py
import inspect
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()


def plotSubVSec(_calibrated_trig_time, *args, **kwargs):
    '''
    Given the calibrated trigger time, this will plot the subsecond
    on the y axis and the second on the x.  Preferrably give this
    pre cut data that only contains the correct trigger type.
    '''
    try:
        _fig = plt.figure()
        _ax = plt.gca()
        _scatter = plt.plot(_calibrated_trig_time-min(_calibrated_trig_time),1e9*(_calibrated_trig_time - numpy.floor(_calibrated_trig_time)),marker=',',linestyle='None')
        plt.ylabel('Trigger Subsecond (ns)')
        plt.xlabel('Trigger Time From Start of Run (s)')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        return _fig, _ax, _scatter
    except Exception as e:
        print('Error in plotSubVSec.')
        file.close()
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)

def plotSubVSecHist(_calibrated_trig_time,s_per_x_bin=5*60.0,n_y_bin=100):
    '''
    Given the calibrated trigger time, this will plot the subsecond
    on the y axis and the second on the x.  Preferrably give this
    pre cut data that only contains the correct trigger type.
    '''
    try:
        _fig = plt.figure()
        _ax = plt.gca()
        x = _calibrated_trig_time-min(_calibrated_trig_time)
        print([numpy.ceil(max(x)/60.0),1000])
        _hist = plt.hist2d(x,1e9*(_calibrated_trig_time - numpy.floor(_calibrated_trig_time)),bins=[int(numpy.ceil(max(x)/s_per_x_bin)),n_y_bin])
        plt.ylabel('Trigger Subsecond (ns)')
        plt.xlabel('Trigger Time From Start of Run (s)')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        return _fig, _ax, _hist
    except Exception as e:
        print('Error in plotSubVSec.')
        file.close()
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)

def alg1(_calibrated_trig_time, atol=0.001):
    '''
    This is intended to cluster any 60 Hz noise by simply looping over events.
    It will determine if any events are within 1/(60 Hz) +/- atol of that event.
    This will then be histogrammed to see outlier events where this number is higher. 
    Atol should be given in seconds. 
    This is bad an only considers the first 1/60 interval.
    '''

    counts = numpy.zeros(len(_calibrated_trig_time),dtype=int)

    half_window = 1/60.0 + atol

    max_times = _calibrated_trig_time + atol
    min_times = _calibrated_trig_time - atol
    cut = numpy.ones(len(_calibrated_trig_time),dtype=bool)
    for event_index, t in enumerate(_calibrated_trig_time):
        cut[event_index] = False
        counts[event_index] = numpy.sum(numpy.logical_and(_calibrated_trig_time[cut] >= min_times[event_index], _calibrated_trig_time[cut] <= max_times[event_index]))

    plt.figure()
    plt.hist(counts,bins=100,density=True)
    plt.ylabel('PDF')
    plt.xlabel('Counts')

    plot_cut = counts > 0.0
    plt.figure()
    plt.gca()
    plt.plot(_calibrated_trig_time[plot_cut]-min(_calibrated_trig_time),1e9*(_calibrated_trig_time[plot_cut] - numpy.floor(_calibrated_trig_time[plot_cut])),marker=',',linestyle='None')
    plt.ylabel('Trigger Subsecond (ns)')
    plt.xlabel('Trigger Time From Start of Run (s)')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

def alg2(_calibrated_trig_time, atol=0.025, window_s=10, randomized_data=False):
    '''
    This is intended to cluster any 60 Hz noise by simply looping over events.
    It will determine if any events are within 1/(60 Hz) +/- atol of that event.
    This will then be histogrammed to see outlier events where this number is higher. 
    Atol should be given in seconds. 
    This is bad an only considers the first 1/60 interval.
    '''
    try:
        counts = numpy.zeros(len(_calibrated_trig_time),dtype=int)
        output_metric = numpy.zeros(len(_calibrated_trig_time))

        expected_period = 1/60.0
        index_window = int(window_s * (len(_calibrated_trig_time)/(_calibrated_trig_time[-1] - _calibrated_trig_time[0]) )) #window_s * approximate events/s = event indices in approximate window second chunks.  This is rough.
        print(index_window)
        plotted=False
        max_diff = expected_period/2
        bin_edges = numpy.linspace(0,max_diff,21) #21 posts = 20 bins = 5% of possible outcome space per bin.
        for event_index, t in enumerate(_calibrated_trig_time):
            diff_from_period = numpy.abs((_calibrated_trig_time[event_index+1:event_index+index_window] - t + expected_period/2)%expected_period - expected_period/2)
            counts[event_index] = numpy.sum((_calibrated_trig_time[event_index+1:event_index+index_window] - t)%expected_period < atol) #How many events are spaced an even integer number of cycles away from this event.
            #hist_counts,hist_edges = numpy.histogram(numpy.abs((_calibrated_trig_time[event_index+1:event_index+index_window] - t + expected_period/2)%expected_period - + expected_period/2),bins = 10)
            #output_metric[event_index] = hist_counts[0] - numpy.mean(hist_counts[2:10])
            hist_counts,hist_edges = numpy.histogram(diff_from_period,bins=bin_edges)
            output_metric[event_index] = hist_counts[0] - numpy.mean(hist_counts[10:20])



            #hist_counts[0] > numpy.mean(hist_counts[2:10])
            if False:
                if plotted == False and (counts[event_index] > 150 or randomized_data==True):
                    plt.figure()
                    plt.hist(numpy.abs((_calibrated_trig_time[event_index+1:event_index+index_window] - t + expected_period/2)%expected_period - + expected_period/2),bins = 10)
                    plt.axvline(expected_period/2,c='r')
                    plt.figure()
                    plt.hist((_calibrated_trig_time[event_index+1:event_index+index_window] - t)%expected_period,bins = 100)
                    plt.axvline(atol,c='r')
                    
                    plt.figure()

                    values = numpy.abs((_calibrated_trig_time[event_index+1:event_index+index_window] - t + expected_period/2)%expected_period - + expected_period/2) #Excess of small values in this indicates periodicity.

                    hist_values, hist_edges = numpy.histogram(values,bins=100)
                    hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
                    plt.subplot(2,1,1)
                    plt.hist(values,bins=100)
                    plt.subplot(2,1,2)
                    plt.plot(hist_centers,numpy.cumsum(hist_values),label='cumsum')
                    plt.plot(hist_centers,numpy.linspace(0,sum(hist_values),len(hist_centers)),c='k',label='linear')
                    plt.legend(loc='lower right')
                    plotted=True


        if False:
            plt.figure()
            plt.hist(counts,bins=100,density=True)
            plt.ylabel('PDF')
            plt.xlabel('Counts')

            plot_cut = counts < 100.0
            plt.figure()
            plt.gca()
            plt.plot(_calibrated_trig_time[plot_cut]-min(_calibrated_trig_time),1e9*(_calibrated_trig_time[plot_cut] - numpy.floor(_calibrated_trig_time[plot_cut])),marker=',',linestyle='None')
            plt.plot(_calibrated_trig_time[~plot_cut]-min(_calibrated_trig_time),1e9*(_calibrated_trig_time[~plot_cut] - numpy.floor(_calibrated_trig_time[~plot_cut])),marker=',',linestyle='None',c='r',label='Cut')
            plt.ylabel('Trigger Subsecond (ns)')
            plt.xlabel('Trigger Time From Start of Run (s)')
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            plt.legend()
        return counts, output_metric
    except Exception as e:
        print('Error in plotSubVSec.')
        file.close()
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)
   



if __name__ == '__main__':
    plt.close('all')
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 1700
    datapath = os.environ['BEACON_DATA']

    for run in [1700,1721]:
        reader = Reader(datapath,run)
        filename = createFile(reader)

        with h5py.File(filename, 'r') as file:
            try:
                load_cut = file['trigger_type'][...] == 2
                calibrated_trig_time = file['calibrated_trigtime'][load_cut]
                # fig, ax, scatter = plotSubVSec(calibrated_trig_time)
                # fig_hist, ax_hist, hist = plotSubVSecHist(calibrated_trig_time)
                
                # if True:
                #     fig2 = plt.figure()
                #     ax2 = plt.gca()
                #     x = file['raw_approx_trigger_time'][load_cut]

                #     scatter2 = plt.plot(x - min(x), file['raw_approx_trigger_time_nsecs'][load_cut],marker=',',linestyle='None')
                #     plt.ylabel('Raw Approx Trigger Subsecond (ns)')
                #     plt.xlabel('Raw Approx Trigger Time From Start of Run (s)')
                #     plt.minorticks_on()
                #     plt.grid(b=True, which='major', color='k', linestyle='-')
                #     plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))

                if False:
                    for window_s in [1,5,10,20,60,120,360]:
                        counts_true, metric_true = alg2(calibrated_trig_time,window_s=window_s, atol=0.001)
                        counts_rand,metric_rand = alg2(randomized_times,window_s=window_s, atol=0.001, randomized_data=True)
                        bins = numpy.linspace(min(min(metric_true),min(metric_rand)),max(max(metric_true),max(metric_rand)),100)
                        plt.figure()
                        plt.hist(metric_true,alpha=0.8,label='True',bins=bins)
                        plt.hist(metric_rand,alpha=0.8,label='Rand',bins=bins)
                        plt.legend()
                        plt.title('Window_s = %f'%window_s)

                if True:
                    window_s = 20.0
                    counts, metric = alg2(calibrated_trig_time,window_s=window_s)
                    counts_rand,metric_rand = alg2(randomized_times,window_s=window_s, atol=0.001, randomized_data=True)
                    bins = numpy.linspace(min(min(metric),min(metric_rand)),max(max(metric),max(metric_rand)),100)
                    plt.figure()
                    plt.hist(metric,alpha=0.8,label='Real Data',bins=bins)
                    plt.hist(metric_rand,alpha=0.8,label='Rand',bins=bins)
                    plt.title('Window_s = %f'%window_s)
                    plt.ylabel('Counts')
                    plt.xlabel('TS')

                    TS_cut_level = 10
                    cut = metric <= TS_cut_level
                    plt.axvline(TS_cut_level,c='r',linestyle='--',label='Cut')
                    plt.legend()

                    fig = plt.figure()
                    ax = plt.gca()
                    plt.subplot(1,2,1)
                    scatter_a = plt.plot(calibrated_trig_time[cut]-min(calibrated_trig_time),1e9*(calibrated_trig_time[cut] - numpy.floor(calibrated_trig_time[cut])),marker=',',linestyle='None',c='b',label='Within Normal Distribution')
                    plt.legend()
                    plt.ylabel('Trigger Subsecond (ns)')
                    plt.xlabel('Trigger Time From Start of Run (s)')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                    plt.subplot(1,2,2)
                    scatter_b = plt.plot(calibrated_trig_time[~cut]-min(calibrated_trig_time),1e9*(calibrated_trig_time[~cut] - numpy.floor(calibrated_trig_time[~cut])),marker=',',linestyle='None',c='r',label='Potential 60 Hz')

                    plt.legend()
                    plt.ylabel('Trigger Subsecond (ns)')
                    plt.xlabel('Trigger Time From Start of Run (s)')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            except Exception as e:
                print('Error while file open, closing file.')
                file.close()
                print('\nError in %s'%inspect.stack()[0][3])
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                sys.exit(1)
   