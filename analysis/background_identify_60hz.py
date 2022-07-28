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

def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))


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

def diffFromPeriodic(_calibrated_trig_time, atol=0.025, window_s=10, expected_period=1/60.0, normalize_by_window_index=False, normalize_by_density=True, plot_sample_hist=False, plot_test_plot_A=False, axs=None, fontsize=18):
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
        if normalize_by_density and normalize_by_window_index:
            print('Density normalization overriding window, normalize_by_window_index -> False')
            normalize_by_window_index = False

        l = len(_calibrated_trig_time)
        output_metric = numpy.zeros(l)
        bin_edges = numpy.linspace(0,expected_period/2,21) #21 posts = 20 bins = 5% of possible outcome space per bin.
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        half_index_window = int((window_s * (len(_calibrated_trig_time)/(_calibrated_trig_time[-1] - _calibrated_trig_time[0]) ))//2) #window_s * approximate events/s = event indices in approximate window second chunks.  Halved for half window.  This is rough.
        #print('half_index_window = ',half_index_window)
        half_expected_period = expected_period/2.0

        if plot_sample_hist == True or axs is not None:
            total_hist_counts = numpy.zeros(len(bin_edges)-1)
            
        for event_index, t in enumerate(_calibrated_trig_time):
            diff_from_period = numpy.abs((_calibrated_trig_time[max(event_index - half_index_window,0):int(min(event_index + half_index_window + 1,l))] - t + half_expected_period)%expected_period - half_expected_period)
            hist_counts,hist_edges = numpy.histogram(diff_from_period,bins=bin_edges, weights=numpy.ones(len(diff_from_period))/(len(diff_from_period) if normalize_by_density else 1))
            output_metric[event_index] = hist_counts[0] - numpy.mean(hist_counts[10:20])
            if plot_sample_hist == True or axs is not None:
                total_hist_counts += hist_counts
            
        if plot_test_plot_A:
            test_plot_A_bin_edges = numpy.linspace(-0.002,0.002,100)#numpy.linspace(-0.01,0.01,100)
            test_plot_A_bin_centers = 0.5*(test_plot_A_bin_edges[1:]+test_plot_A_bin_edges[:-1])
            test_plot_A_total_hist_counts = numpy.zeros(len(test_plot_A_bin_edges)-1)
            cycle_count = 1 #How many future events to consider
            expected_diffs = -(numpy.arange(cycle_count)+1)*expected_period
            for event_index, t in enumerate(_calibrated_trig_time):
                if event_index + cycle_count < len(_calibrated_trig_time):
                    diffs = t - _calibrated_trig_time[event_index+1:event_index+1+cycle_count]
                    hist_counts,hist_edges = numpy.histogram(diffs - expected_diffs,bins=test_plot_A_bin_edges)
                    # if event_index == numpy.argmax(output_metric):
                    #     import pdb; pdb.set_trace()
                    test_plot_A_total_hist_counts += hist_counts

            plt.figure()
            plt.hist(test_plot_A_bin_centers,bins=test_plot_A_bin_edges,weights=test_plot_A_total_hist_counts, label='All Events')
            plt.legend(loc='upper right')
            plt.ylabel('Counts')
            plt.xlabel('Remainder of Follow %i Events from %0.4f s Periodicity'%(cycle_count,expected_period))
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        if plot_sample_hist == True or axs is not None:
            max_event_index = numpy.argmax(output_metric)

            mode = 'median'#'low'#'median'

            if mode == 'median':
                secondary_event_index = numpy.argsort(output_metric)[l//2]
            elif mode == 'low':
                secondary_event_index = numpy.argmin(output_metric)

            max_diff_from_period = numpy.abs((_calibrated_trig_time[max(max_event_index - half_index_window,0):int(min(max_event_index + half_index_window + 1,l))] - _calibrated_trig_time[max_event_index] + half_expected_period)%expected_period - half_expected_period)
            secondary_diff_from_period = numpy.abs((_calibrated_trig_time[max(secondary_event_index - half_index_window,0):int(min(secondary_event_index + half_index_window + 1,l))] - _calibrated_trig_time[secondary_event_index] + half_expected_period)%expected_period - half_expected_period)

            if False:
                fig = plt.figure(figsize = (12,16))
                plt.subplot(2,2,1)
                plt.hist(max_diff_from_period,bins=bin_edges,label='High TS')
                plt.legend(loc='upper right', fontsize=fontsize-2)
                plt.ylabel('Counts', fontsize=fontsize)
                plt.xlabel(r'$(t_i - t_j)\%r^{-1}$ (s)', fontsize=fontsize)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.subplot(2,2,2)
                plt.hist(secondary_diff_from_period,bins=bin_edges,label='%s TS'%mode.title())
                plt.legend(loc='upper right', fontsize=fontsize-2)
                plt.ylabel('Counts', fontsize=fontsize)
                plt.xlabel('Remainder (Shifted)', fontsize=fontsize)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.subplot(2,1,2)
                plt.hist(bin_centers,bins=bin_edges,weights=total_hist_counts, label='All Events')
                plt.legend(loc='upper right', fontsize=fontsize-2)
                plt.ylabel('Counts', fontsize=fontsize)
                plt.xlabel('Remainder (Shifted)', fontsize=fontsize)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.tight_layout()
                plt.subplots_adjust(left=0.8, bottom=0.11,right=0.96, top=0.96, wspace=0.2, hspace=0.2)
            else:

                if axs is None:
                    fig = plt.figure(figsize = (12,16))
                    ax = plt.subplot(2,1,1)
                else:
                    ax = axs[0]
                    
                
                ax.axvspan(bin_edges[0],bin_edges[1],color='r',alpha=0.3,label='5% Bin Most\nConsistent With $T$')
                ax.axvspan(numpy.mean(bin_edges),bin_edges[-1],color='g',alpha=0.3,label='50% of Bins Least\nConsistent With $T$')
                
                n, bins, patches = ax.hist(max_diff_from_period,bins=bin_edges, facecolor="dodgerblue", label='Distribution for Max TS Event', alpha=1.0 , weights=numpy.ones(len(max_diff_from_period))/(len(max_diff_from_period) if normalize_by_density else 1))
                y = numpy.zeros(len(bin_edges)+1)
                y[1:-1] = n
                x = numpy.append(bin_edges,max(bin_edges) + numpy.diff(bin_edges)[0])
                ax.plot(x, y, drawstyle='steps', lw=2, c='k')

                ax.set_xlim(min(bin_edges), max(bin_edges))
                ax.set_ylim(0,max(n)*1.2)
                if normalize_by_density:
                    ax.set_ylabel('Normalized Counts\nin Window $w$', fontsize=fontsize)
                else:
                    ax.set_ylabel('Counts in Window $w$', fontsize=fontsize)
                ax.minorticks_on()
                ax.grid(b=True, which='major', color='k', linestyle='-')
                ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                ax.legend(loc='upper right', fontsize=fontsize-4,framealpha=1)

                if axs is None:
                    ax = plt.subplot(2,1,2)
                else:
                    ax = axs[1]

                ax.hist(bin_centers,bins=bin_edges, facecolor="dodgerblue", weights=total_hist_counts, label='Combined Distribution\nfor All Events', alpha=0.7 )
                ax.plot(bin_centers + numpy.diff(bin_centers)[0]/2, total_hist_counts, drawstyle='steps', lw=2, c='k')
                ax.legend(loc='upper right', fontsize=fontsize-4)
                ax.set_ylabel('Combined Counts', fontsize=fontsize)
                ax.grid(b=True, which='major', color='k', linestyle='-')
                ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                ax.set_xlabel('Absolute Time Difference Remainder\nfrom Expected Periodicity (s)', fontsize=fontsize)
                #ax.set_xlabel(r'$(t_i - t_j)\%r^{-1}$ (s)', fontsize=fontsize)


                if axs is None:
                    ax.tight_layout()
                    ax.subplots_adjust(left=0.08, bottom=0.11,right=0.96, top=0.96, wspace=0.2, hspace=0.2)
            
            if axs is None:
                if normalize_by_window_index == True:
                    return output_metric/(2.0*half_index_window + 1.0), fig
                else:
                    return output_metric, fig
            else:
                if normalize_by_window_index == True:
                    return output_metric/(2.0*half_index_window + 1.0), axs
                else:
                    return output_metric, axs

        if normalize_by_window_index == True:
            return output_metric/(2.0*half_index_window + 1.0)
        else:
            return output_metric
    except Exception as e:
        print('Error in diffFromPeriodic.')
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        import pdb; pdb.set_trace()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)
   
def get60HzEvents2(file, n_cycles=10, expected_period=1/60.0, TS_cut_level=0.0005, plot=True):
    '''
    File should be a safely open readable hdf5 file.
    '''
    try:
        load_cut = file['trigger_type'][...] == 2
        loaded_eventids = numpy.where(load_cut)[0]
        calibrated_trig_time = file['calibrated_trigtime'][load_cut]
        bin_edges = numpy.linspace(-0.2,0.2,1000)#numpy.linspace(-0.002,0.002,100)#numpy.linspace(-0.01,0.01,100)
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        total_hist_counts = numpy.zeros(len(bin_edges)-1)
        expected_diffs = -(numpy.arange(n_cycles)+1)*expected_period
        l = len(calibrated_trig_time)
        flagged_eventids_cut = numpy.zeros_like(loaded_eventids,dtype=bool)


        if plot:
            randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))
            # plt.figure()
            # plt.hist(numpy.diff(randomized_times),bins=100)
            random_hist_counts = numpy.zeros(len(bin_edges)-1)
            flagged_counts = numpy.zeros(len(bin_edges)-1)
            old_counts = numpy.zeros(len(bin_edges)-1)
            old_flagged_events = get60HzEvents(file,window_s=20.0, TS_cut_level=0.1, normalize_by_window_index=True)
            old_flagged_events_cut = numpy.isin(loaded_eventids,old_flagged_events)

            for event_index, t in enumerate(randomized_times):
                if event_index + 1 < l:
                    #Single diff compared to many potential times. 
                    diffs = t - randomized_times[event_index+1]
                    vals = diffs - expected_diffs#(diffs - expected_diffs)[numpy.argmin(numpy.abs(diffs - expected_diffs))]
                    hist_counts, hist_edges = numpy.histogram(vals,bins=bin_edges)
                    random_hist_counts += hist_counts


        for event_index, t in enumerate(calibrated_trig_time):
            if event_index + 1 < l:
                #Single diff compared to many potential times. 
                diffs = t - calibrated_trig_time[event_index+1]#calibrated_trig_time[event_index+1:event_index+1+n_cycles]
                vals = diffs - expected_diffs#(diffs - expected_diffs)[numpy.argmin(numpy.abs(diffs - expected_diffs))]
                hist_counts, hist_edges = numpy.histogram(vals,bins=bin_edges)

                flagged_eventids_cut[event_index] = numpy.any(numpy.abs(vals) <= TS_cut_level)

                total_hist_counts += hist_counts
                if plot and flagged_eventids_cut[event_index]:
                    flagged_counts += hist_counts
                if plot and old_flagged_events_cut[event_index]:
                    old_counts += hist_counts


        if plot:
            plt.figure()
            plt.hist(bin_centers,bins=bin_edges,weights=total_hist_counts, color='tab:blue' , label='All Events',alpha=0.8)
            # plt.hist(bin_centers,bins=bin_edges,weights=flagged_counts, color='tab:orange', label='Flagged Events',alpha=0.8)
            # plt.hist(bin_centers,bins=bin_edges,weights=old_counts, color='tab:red', label='Old Events',alpha=0.8)
            plt.hist(bin_centers,bins=bin_edges,weights=random_hist_counts, color='tab:purple' , label='Randomized Time',alpha=0.8)
            #plt.hist(bin_centers,bins=bin_edges,weights=random_hist_counts, label='Randomized Times',alpha=0.8)
            # plt.axvline(TS_cut_level,linestyle='--',c='r',label='TS Cut')
            # plt.axvline(-TS_cut_level,linestyle='--',c='r',label='TS Cut')
            plt.legend(loc='upper right')
            plt.ylabel('Counts')
            plt.xlabel('Residual Difference from Each of %i Cycles with %0.4f s Periodicity'%(n_cycles,expected_period))
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        flagged_eventids = loaded_eventids[flagged_eventids_cut]
        return flagged_eventids
    except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


def get60HzEvents3(file, expected_period=1/60.0, TS_cut_level=0.0005, plot=True):
    '''
    File should be a safely open readable hdf5 file.
    '''
    try:
        load_cut = file['trigger_type'][...] == 2
        loaded_eventids = numpy.where(load_cut)[0]
        calibrated_trig_time = file['calibrated_trigtime'][load_cut]
        bin_edges = numpy.linspace(-expected_period/2,expected_period/2,1000)#numpy.linspace(-0.002,0.002,100)#numpy.linspace(-0.01,0.01,100)
        bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
        total_hist_counts = numpy.zeros(len(bin_edges)-1)
        expected_diffs = -expected_period
        l = len(calibrated_trig_time)
        flagged_eventids_cut = numpy.zeros_like(loaded_eventids,dtype=bool)


        dt = (numpy.diff(calibrated_trig_time) + expected_period/2)%expected_period - expected_period/2


        if plot:
            randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))
            random_dt = (numpy.diff(randomized_times)  + expected_period/2 )%expected_period - expected_period/2

            plt.figure()
            plt.hist(dt, bins=bin_edges, color='tab:blue' , label='All Events',alpha=0.8)
            plt.hist(random_dt, bins=bin_edges, color='tab:purple' , label='Randomized Time',alpha=0.8)
            plt.legend(loc='upper right')
            plt.ylabel('Counts')
            plt.xlabel('mod(dt,%0.5f + %0.5f) - %0.5f'%(expected_period,expected_period/2,expected_period/2))
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        # flagged_eventids = loaded_eventids[flagged_eventids_cut]
        # return flagged_eventids
    except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

def get60HzEvents(file,window_s=20.0, TS_cut_level=0.1, normalize_by_window_index=True):
    '''
    File should be a safely open readable hdf5 file.
    '''
    try:

        load_cut = file['trigger_type'][...] == 2
        loaded_eventids = numpy.where(load_cut)[0]
        calibrated_trig_time = file['calibrated_trigtime'][load_cut]
        randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))

        #This was used in initial testing of diffFromPeriodic to demonstrate it can work.
        metric = diffFromPeriodic(calibrated_trig_time,window_s=window_s, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=False)
        return loaded_eventids[metric >= TS_cut_level]
    except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    plt.close('all')
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 1700
    datapath = os.environ['BEACON_DATA']
    normalize_by_window_index = True
    #1721
    for run in [run]:
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
                    loaded_eventids = numpy.where(load_cut)[0]
                    calibrated_trig_time = file['calibrated_trigtime'][load_cut]
                    randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))

                    if True:
                        #for window_s in [1,5,10,20,60,120,360]:
                        for window_s in [20]:
                            metric_true = diffFromPeriodic(calibrated_trig_time,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=True, plot_test_plot_A=True)
                            metric_rand = diffFromPeriodic(randomized_times,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index)
                            bins = numpy.linspace(min(min(metric_true),min(metric_rand)),max(max(metric_true),max(metric_rand)),100)
                            plt.figure()
                            counts_true, bins_true, patches_true = plt.hist(metric_true,alpha=0.8,label='True',bins=bins)
                            counts_rand, bins_rand, patches_rand = plt.hist(metric_rand,alpha=0.8,label='Rand',bins=bins)

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

                            plt.plot(plot_x,gaus(plot_x,*popt),'-',c='k',label='Fit Sigma = %f ns'%popt[2])
                            plt.axvline(popt[1],linestyle='-',c='k',label='Fit Center = %f'%(popt[1]))
                            
                            plt.legend()
                            plt.title('Window_s = %f'%window_s)
                            plt.ylabel('Counts')
                            plt.xlabel('TS')

                        flagged_eventids = get60HzEvents2(file, n_cycles=1, expected_period=1/60.0, TS_cut_level=0.0005, plot=True)
                        # flagged_eventids = get60HzEvents2(file, n_cycles=2, expected_period=1/60.0, TS_cut_level=0.0005, plot=True)
                        # flagged_eventids = get60HzEvents2(file, n_cycles=10, expected_period=1/60.0, TS_cut_level=0.0005, plot=True)

                        get60HzEvents3(file, expected_period=1/60.0, TS_cut_level=0.0005, plot=True)

                    if False:
                        #This was used in initial testing of diffFromPeriodic to demonstrate it can work.
                        window_s = 20.0
                        metric = diffFromPeriodic(calibrated_trig_time,window_s=window_s, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=True)
                        metric_rand = diffFromPeriodic(randomized_times,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=True)
                        bins = numpy.linspace(min(min(metric),min(metric_rand)),max(max(metric),max(metric_rand)),100)
                        bin_centers = (bins[:-1] + bins[1:]) / 2.0
                        TS_cut_level = 0.075

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
                        # scatter_2 = plt.plot(calibrated_trig_time-min(calibrated_trig_time),(calibrated_trig_time - numpy.floor(calibrated_trig_time))%(1/60.0),marker=',',linestyle='None',c='b',label='All RF Events')
                        # plt.legend()
                        # plt.ylabel('(Trigger Subsecond) % (1/60Hz)')
                        scatter_2 = plt.plot(calibrated_trig_time-min(calibrated_trig_time),(calibrated_trig_time)%(1/60.0),marker=',',linestyle='None',c='b',label='All RF Events')
                        plt.legend()
                        plt.ylabel('(Trigger Time) % (1/60Hz)')
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
                            #flagged_eventids = numpy.vstack(numpy.ones(sum(~cut),dtype=int)*int(run),loaded_eventids[~cut]) #Stacked with run number in case
                            numpy.savetxt("./flagged_60Hz_eventids_run%i.csv"%run, loaded_eventids[~cut], delimiter=",")


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
                # scatter_2 = plt.plot(calibrated_trig_time-min(calibrated_trig_time),(calibrated_trig_time - numpy.floor(calibrated_trig_time))%(1/60.0),marker=',',linestyle='None',c='b',label='All RF Events')
                # plt.legend()
                # plt.ylabel('(Trigger Subsecond) % (1/60Hz)')
                scatter_2 = plt.plot(calibrated_trig_time-min(calibrated_trig_time),(calibrated_trig_time)%(1/60.0),marker=',',linestyle='None',c='b',label='All RF Events')
                plt.legend()
                plt.ylabel('(Trigger Time) % (1/60Hz)')
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