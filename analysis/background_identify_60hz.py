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
from tools.data_handler import createFile, getTimes, loadTriggerTypes, getEventTimes
from tools.fftmath import FFTPrepper

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
        _scatter = plt.plot(_calibrated_trig_time-min(_calibrated_trig_time),(_calibrated_trig_time - numpy.floor(_calibrated_trig_time)),marker=',',linestyle='None')
        plt.ylabel('Trigger Subsecond (s)')
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
        _hist = plt.hist2d(x,(_calibrated_trig_time - numpy.floor(_calibrated_trig_time)),bins=[int(numpy.ceil(max(x)/s_per_x_bin)),n_y_bin])
        plt.ylabel('Trigger Subsecond (s)')
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
    plt.plot(_calibrated_trig_time[plot_cut]-min(_calibrated_trig_time),(_calibrated_trig_time[plot_cut] - numpy.floor(_calibrated_trig_time[plot_cut])),marker=',',linestyle='None')
    plt.ylabel('Trigger Subsecond (s)')
    plt.xlabel('Trigger Time From Start of Run (s)')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

def diffFromPeriodic(_calibrated_trig_time, atol=0.025, window_s=10, expected_period=1/60.0, normalize_by_window_index=False, plot_sample_hist=False):
    '''
    This is intended to cluster any 60 Hz noise by simply looping over events.

    60 Hz events are expected to arrive with temporal seperations of N*(1/60Hz).
    This algorithm looks at events in time windows roughly defined by window_s around event i, and
    determines how many events (j) within that window have a remainder (t_i - t_j)%(1/60) within atol
    of the expected periodic value.  Values near 0 and 1/60 both indicate near periodicity, so the
    remainder calculation is performed offset by half a period such that those extreme values are
    expected to be centered about (1/60)/2, then (1/60)/2 is subtracted from the remainder so it is
    centered on 0, and the absolute value is taken.  The shifted abs(remainders) are binned into 20 bins,
    each representing 5% of the possible outcomes.  If 60 Hz noise is not present then this should
    produce a uniform distrobution (events evenly distributed in time roughly).  The bin closest to 0 
    shifted remainder will present a significant excess in the presence of 60 Hz noise.  The test 
    statistic (TS) is the difference between the counts in this bin and the average content of the 
    furthest 10 bins (50% of possible remainders).  The TS will result in a Gaussian for randomly
    distributed times, but with 60 Hz noise will broaden to significantly larger values.

    The number is given in differences in counts, which may change between runs for the same time
    window due to noisier runs.  It is recommended to run this with randomized times to get a guage
    of the expected distrobution for absence of 60 Hz with the same event rate, and use that to
    determine where to cut on the TS for confident 60 Hz selection.

    If normalize_by_window_index == True then the TS is divided by the window length (2*half_index_window) + 1.

    Note that edge effects will be present for events within a half_index_window of the beginning and
    end of the run.
    '''
    try:
        l = len(_calibrated_trig_time)
        output_metric = numpy.zeros(l)
        bin_edges = numpy.linspace(0,expected_period/2,21) #21 posts = 20 bins = 5% of possible outcome space per bin.

        half_index_window = int((window_s * (len(_calibrated_trig_time)/(_calibrated_trig_time[-1] - _calibrated_trig_time[0]) ))//2) #window_s * approximate events/s = event indices in approximate window second chunks.  Halved for half window.  This is rough.
        half_expected_period = expected_period/2.0

        for event_index, t in enumerate(_calibrated_trig_time):
            diff_from_period = numpy.abs((_calibrated_trig_time[max(event_index - half_index_window,0):int(min(event_index + half_index_window + 1,l))] - t + half_expected_period)%expected_period - half_expected_period)
            hist_counts,hist_edges = numpy.histogram(diff_from_period,bins=bin_edges)
            output_metric[event_index] = hist_counts[0] - numpy.mean(hist_counts[10:20])
            
        if plot_sample_hist == True:
            max_event_index = numpy.argmax(output_metric)
            median_event_index = numpy.argsort(output_metric)[l//2]

            max_diff_from_period = numpy.abs((_calibrated_trig_time[max(max_event_index - half_index_window,0):int(min(max_event_index + half_index_window + 1,l))] - _calibrated_trig_time[max_event_index] + half_expected_period)%expected_period - half_expected_period)
            median_diff_from_period = numpy.abs((_calibrated_trig_time[max(median_event_index - half_index_window,0):int(min(median_event_index + half_index_window + 1,l))] - _calibrated_trig_time[median_event_index] + half_expected_period)%expected_period - half_expected_period)
            plt.figure()
            plt.subplot(1,2,1)
            plt.hist(max_diff_from_period,bins=bin_edges,label='High TS')
            plt.legend(loc='upper right')
            plt.ylabel('Counts')
            plt.xlabel('Remainder (Shifted)')
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            plt.subplot(1,2,2)
            plt.hist(median_diff_from_period,bins=bin_edges,label='Median TS')
            plt.legend(loc='upper right')
            plt.ylabel('Counts')
            plt.xlabel('Remainder (Shifted)')
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        if normalize_by_window_index == True:
            return output_metric/(2.0*half_index_window + 1.0)
        else:
            return output_metric
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
    normalize_by_window_index = True

    for run in [1721]:
        reader = Reader(datapath,run)

        try:
            filename = createFile(reader)
            print(filename)
            loaded = True
        except:
            loaded = False

        #IF YOU WANT TO FORCE THE SECOND MODE THEN TYPE loaded = False BELOW THIS

        if loaded == True:
            with h5py.File(filename, 'r') as file:
                try:
                    load_cut = file['trigger_type'][...] == 2
                    calibrated_trig_time = file['calibrated_trigtime'][load_cut]
                    randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))

                    if False:
                        for window_s in [1,5,10,20,60,120,360]:
                            metric_true = diffFromPeriodic(calibrated_trig_time,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index)
                            metric_rand = diffFromPeriodic(randomized_times,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index)
                            bins = numpy.linspace(min(min(metric_true),min(metric_rand)),max(max(metric_true),max(metric_rand)),100)
                            plt.figure()
                            plt.hist(metric_true,alpha=0.8,label='True',bins=bins)
                            plt.hist(metric_rand,alpha=0.8,label='Rand',bins=bins)
                            plt.legend()
                            plt.title('Window_s = %f'%window_s)

                    if True:
                        #This was used in initial testing of diffFromPeriodic to demonstrate it can work.
                        window_s = 20.0
                        metric = diffFromPeriodic(calibrated_trig_time,window_s=window_s, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=True)
                        metric_rand = diffFromPeriodic(randomized_times,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=True)
                        bins = numpy.linspace(min(min(metric),min(metric_rand)),max(max(metric),max(metric_rand)),100)
                        bin_centers = (bins[:-1] + bins[1:]) / 2.0
                        TS_cut_level = 0.1

                        plt.figure()
                        plt.hist(metric,alpha=0.8,label='Real Data',bins=bins)
                        plt.hist(metric_rand,alpha=0.8,label='Rand',bins=bins)
                        plt.title('Window_s = %f'%window_s)
                        plt.ylabel('Counts')
                        plt.xlabel('TS')

                        cut = metric <= TS_cut_level
                        plt.axvline(TS_cut_level,c='r',linestyle='--',label='Cut')
                        plt.legend()

                        fig = plt.figure()
                        plt.subplot(1,2,1)
                        ax = plt.gca()
                        scatter_1 = plt.plot(calibrated_trig_time-min(calibrated_trig_time),(calibrated_trig_time - numpy.floor(calibrated_trig_time)),marker=',',linestyle='None',c='b',label='All RF Events')
                        plt.legend()
                        plt.ylabel('Trigger Subsecond (s)')
                        plt.xlabel('Trigger Time From Start of Run (s)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.subplot(1,2,2)
                        scatter_2 = plt.plot(calibrated_trig_time-min(calibrated_trig_time),(calibrated_trig_time - numpy.floor(calibrated_trig_time))%(1/60.0),marker=',',linestyle='None',c='b',label='All RF Events')
                        plt.legend()
                        plt.ylabel('(Trigger Subsecond) % (1/60Hz)')
                        plt.xlabel('Trigger Time From Start of Run (s)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.subplot(1,2,2)

                        fig = plt.figure()
                        ax = plt.gca()
                        plt.subplot(1,2,1)
                        scatter_a = plt.plot(calibrated_trig_time[cut]-min(calibrated_trig_time),(calibrated_trig_time[cut] - numpy.floor(calibrated_trig_time[cut])),marker=',',linestyle='None',c='b',label='Within Normal Distribution')
                        plt.legend()
                        plt.ylabel('Trigger Subsecond (s)')
                        plt.xlabel('Trigger Time From Start of Run (s)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                        plt.subplot(1,2,2)
                        scatter_b = plt.plot(calibrated_trig_time[~cut]-min(calibrated_trig_time),(calibrated_trig_time[~cut] - numpy.floor(calibrated_trig_time[~cut])),marker=',',linestyle='None',c='r',label='Potential 60 Hz')

                        plt.legend()
                        plt.ylabel('Trigger Subsecond (s)')
                        plt.xlabel('Trigger Time From Start of Run (s)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                    if False:
                        #This is intended to automate the process slightly without have to plot.
                        window_s = 20.0
                        metric = diffFromPeriodic(calibrated_trig_time,window_s=window_s, normalize_by_window_index=normalize_by_window_index)
                        metric_rand = diffFromPeriodic(randomized_times,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index)
                        bins = numpy.linspace(min(min(metric),min(metric_rand)),max(max(metric),max(metric_rand)),100)
                        bin_centers = (bins[:-1] + bins[1:]) / 2.0
                        TS_cut_level = max(metric_rand)

                        plt.figure()
                        plt.hist(metric,alpha=0.8,label='Real Data',bins=bins)
                        plt.hist(metric_rand,alpha=0.8,label='Rand',bins=bins)
                        plt.axvline(TS_cut_level,c='r',linestyle='--',label='Cut')
                        plt.title('Window_s = %f'%window_s)
                        plt.ylabel('Counts')
                        plt.xlabel('TS')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                        cut = metric <= TS_cut_level

                        eventids = file['eventids'][load_cut][cut]
                        filter_string = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_32768-align_0-shorten_signals-1-shorten_thresh-0.70-shorten_delay-10.00-shorten_length-90.00'
                        plt.figure()
                        bins = numpy.linspace(-250,250,1000)
                        for pair_index, pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
                            print(pair)
                            plt.subplot(6,1,pair_index + 1)
                            plt.hist(file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(pair[0],pair[1])][...][eventids],bins = 100, range=[-250,250])
                            plt.ylabel('Counts')
                            plt.xlabel('Hpol Time Delay (ns)')
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
        else:
            #Sloppy work around to run code if file isn't loading (i.e. if scratch files aren't available)
            print('Running script in mode not requiring access to file.')
            trigger_type = loadTriggerTypes(reader)
            eventids = numpy.arange(len(trigger_type))
            load_cut = trigger_type == 2
            calibrated_trig_time = getEventTimes(reader)[load_cut]
            randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))

            if False:
                for window_s in [1,5,10,20,60,120,360]:
                    metric_true = diffFromPeriodic(calibrated_trig_time,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index)
                    metric_rand = diffFromPeriodic(randomized_times,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index)
                    bins = numpy.linspace(min(min(metric_true),min(metric_rand)),max(max(metric_true),max(metric_rand)),100)
                    plt.figure()
                    plt.hist(metric_true,alpha=0.8,label='True',bins=bins)
                    plt.hist(metric_rand,alpha=0.8,label='Rand',bins=bins)
                    plt.legend()
                    plt.title('Window_s = %f'%window_s)

            if True:
                #This was used in initial testing of diffFromPeriodic to demonstrate it can work.
                window_s = 20.0
                metric = diffFromPeriodic(calibrated_trig_time,window_s=window_s, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=True)
                metric_rand = diffFromPeriodic(randomized_times,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=True)
                bins = numpy.linspace(min(min(metric),min(metric_rand)),max(max(metric),max(metric_rand)),100)
                bin_centers = (bins[:-1] + bins[1:]) / 2.0

                plt.figure()
                plt.hist(metric,alpha=0.8,label='Real Data',bins=bins)
                plt.hist(metric_rand,alpha=0.8,label='Rand',bins=bins)
                plt.title('Window_s = %f'%window_s)
                plt.ylabel('Counts')
                plt.xlabel('TS')

                TS_cut_level = 0.1

                cut = metric <= TS_cut_level

                plt.axvline(TS_cut_level,c='r',linestyle='--',label='Cut')
                plt.legend()

                fig = plt.figure()
                plt.subplot(1,2,1)
                ax = plt.gca()
                scatter_1 = plt.plot(calibrated_trig_time-min(calibrated_trig_time),(calibrated_trig_time - numpy.floor(calibrated_trig_time)),marker=',',linestyle='None',c='b',label='All RF Events')
                plt.legend()
                plt.ylabel('Trigger Subsecond (s)')
                plt.xlabel('Trigger Time From Start of Run (s)')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.subplot(1,2,2)
                scatter_2 = plt.plot(calibrated_trig_time-min(calibrated_trig_time),(calibrated_trig_time - numpy.floor(calibrated_trig_time))%(1/60.0),marker=',',linestyle='None',c='b',label='All RF Events')
                plt.legend()
                plt.ylabel('(Trigger Subsecond) % (1/60Hz)')
                plt.xlabel('Trigger Time From Start of Run (s)')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.subplot(1,2,2)

                fig = plt.figure()
                ax = plt.gca()
                plt.subplot(1,2,1)
                scatter_a = plt.plot(calibrated_trig_time[cut]-min(calibrated_trig_time),(calibrated_trig_time[cut] - numpy.floor(calibrated_trig_time[cut])),marker=',',linestyle='None',c='b',label='Within Normal Distribution')
                plt.legend()
                plt.ylabel('Trigger Subsecond (s)')
                plt.xlabel('Trigger Time From Start of Run (s)')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.subplot(1,2,2)
                scatter_b = plt.plot(calibrated_trig_time[~cut]-min(calibrated_trig_time),(calibrated_trig_time[~cut] - numpy.floor(calibrated_trig_time[~cut])),marker=',',linestyle='None',c='r',label='Potential 60 Hz')

                plt.legend()
                plt.ylabel('Trigger Subsecond (s)')
                plt.xlabel('Trigger Time From Start of Run (s)')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                print('Events from run %i that are expected to be 60 Hz events using the test statistic cut of %0.2f can be found by printing eventids[load_cut][~cut], which is executed below:'%(run,TS_cut_level))
                print(eventids[load_cut][~cut])
                print('If you would like to save the above eventids you can do so by either hacking in code below this print statement or doing so after the fact.')
                #numpy.savetxt('./run%i_60Hz_eventids.csv',eventids[load_cut][~cut])  #COULD USE THIS TO SAVE

            if False:
                #This is intended to automate the process slightly without have to plot.
                window_s = 20.0
                metric = diffFromPeriodic(calibrated_trig_time,window_s=window_s, normalize_by_window_index=normalize_by_window_index)
                metric_rand = diffFromPeriodic(randomized_times,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index)
                bins = numpy.linspace(min(min(metric),min(metric_rand)),max(max(metric),max(metric_rand)),100)
                bin_centers = (bins[:-1] + bins[1:]) / 2.0
                TS_cut_level = max(metric_rand)

                plt.figure()
                plt.hist(metric,alpha=0.8,label='Real Data',bins=bins)
                plt.hist(metric_rand,alpha=0.8,label='Rand',bins=bins)
                plt.axvline(TS_cut_level,c='r',linestyle='--',label='Cut')
                plt.title('Window_s = %f'%window_s)
                plt.ylabel('Counts')
                plt.xlabel('TS')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)