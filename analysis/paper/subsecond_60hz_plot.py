#!/usr/bin/env python3
'''
This is a script that aims to identify noise that is of the 60 Hz category.
It will first attempt to do this by plotting sub-Second time v.s. time, and
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
from analysis.background_identify_60hz import *

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
import h5py
import inspect
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

label_fontsize = 16
plt.rc('xtick',labelsize=label_fontsize)
plt.rc('ytick',labelsize=label_fontsize)

if __name__ == '__main__':
    plt.close('all')
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 5911
        #5760
        #5840
    datapath = os.environ['BEACON_DATA']
    normalize_by_window_index = True
    if False:
        runs = run + numpy.arange(10)
    else:
        runs = numpy.array([run])


    for run in runs:
        run = int(run)
        reader = Reader(datapath,run)

        try:
            filename = createFile(reader)
            print(filename)
            loaded = True
        except:
            loaded = False

        #Sloppy work around to run code if file isn't loading (i.e. if scratch files aren't available)
        print('Running script in mode not requiring access to file.')
        trigger_type = loadTriggerTypes(reader)
        eventids = numpy.arange(len(trigger_type))
        load_cut = trigger_type == 2
        calibrated_trig_time = getEventTimes(reader)[load_cut]
        randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))


        #This was used in initial testing of diffFromPeriodic to demonstrate it can work.
        window_s = 20.0
        metric = diffFromPeriodic(calibrated_trig_time,window_s=window_s, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=False)
        metric_rand = diffFromPeriodic(randomized_times,window_s=window_s, atol=0.001, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=False)
        bins = numpy.linspace(min(min(metric),min(metric_rand)),max(max(metric),max(metric_rand)),100)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        
        TS_cut_level = 0.08#numpy.max(metric_rand) + 0.005#0.08#0.1
        cut = metric <= TS_cut_level

        marker='o'

        numpy.random.seed(2)
        fig = plt.figure(figsize=(20,7))
        ax = plt.subplot(1,3,1)
        if False:
            if marker == ',':
                scatter_1 = plt.plot((calibrated_trig_time-min(calibrated_trig_time))/60.0,(calibrated_trig_time - numpy.floor(calibrated_trig_time)),marker=marker,linestyle='None',c='b',label='All RF Events')#,s=1
            else:
                scatter_1 = plt.scatter((calibrated_trig_time-min(calibrated_trig_time))/60.0,(calibrated_trig_time - numpy.floor(calibrated_trig_time)),marker=marker,linestyle='None',c='b',label='All RF Events',s=0.5)
        else:
            scatter_1 = plt.plot((calibrated_trig_time-min(calibrated_trig_time))/60.0,(calibrated_trig_time - numpy.floor(calibrated_trig_time)),marker=',',linestyle='None',c='b',label='All RF Events')
        plt.legend(fontsize=label_fontsize, loc='upper right',framealpha=1)
        plt.ylabel('Trigger Sub-Second (s)',fontsize=label_fontsize)
        plt.xlabel('Time From Start of Run (min)',fontsize=label_fontsize)
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        plt.subplot(1,3,2, sharex=ax, sharey=ax)
        if marker == ',':
            scatter_b = plt.plot((calibrated_trig_time[~cut]-min(calibrated_trig_time))/60.0,(calibrated_trig_time[~cut] - numpy.floor(calibrated_trig_time[~cut])),marker=marker,linestyle='None',c='r',label='Flagged 60 Hz')#,s=1
        else:
            scatter_b = plt.scatter((calibrated_trig_time[~cut]-min(calibrated_trig_time))/60.0,(calibrated_trig_time[~cut] - numpy.floor(calibrated_trig_time[~cut])),marker=marker,linestyle='None',c='r',label='Flagged 60 Hz',s=0.5)

        if run  == 5840:
            plt.xlim(0,100)
            plt.ylim(0,1)
        plt.legend(fontsize=label_fontsize, loc='upper right',framealpha=1)
        plt.ylabel('Trigger Sub-Second (s)',fontsize=label_fontsize)
        plt.xlabel('Time From Start of Run (min)',fontsize=label_fontsize)
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


        plt.subplot(1,3,3)
        normalize = False

        if normalize:
            plt.hist(metric,alpha=0.7,density=True,label='Actual\nSub-Seconds',bins=bins)
            plt.hist(metric_rand,alpha=0.7,density=True,label='Uniformly\nRandomized\nSub-Seconds',bins=bins)
            plt.ylabel('Normalized Counts',fontsize=label_fontsize)
            arrow_xy = (0.115,1.15)
            relative_xy = (0.05, 0.1)
        else:
            plt.hist(metric,alpha=0.7,density=False,label='Actual\nSub-Seconds',bins=bins)
            plt.hist(metric_rand,alpha=0.7,density=False,label='Uniformly\nRandomized\nSub-Seconds',bins=bins)
            plt.ylabel('Counts',fontsize=label_fontsize)
            arrow_xy = (0.115,4700)
            relative_xy = (0.05, 1000)
        plt.xlabel('Test Statistic',fontsize=label_fontsize)
        plt.axvline(TS_cut_level,c='r',linestyle='--',linewidth=3, label='Cut')
        plt.legend(fontsize=label_fontsize, loc='upper right',framealpha=1)
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        ann = ax.annotate("Flagged 60Hz",
          xy=arrow_xy, xycoords='data',
          xytext=(arrow_xy[0]+relative_xy[0],arrow_xy[1]+relative_xy[1]), textcoords='data',
          size=label_fontsize, va="center", ha="center",
          bbox=dict(boxstyle="round", fc="w"),
          arrowprops=dict(  arrowstyle="-[,widthB=0.55, lengthB=0.2, angleB=0",
                            fc="w"),
          )


        plt.tight_layout()
        plt.subplots_adjust(left=0.06, bottom=0.12,right=0.98, top=0.95, wspace=0.3, hspace=0.2)

        fig.savefig('./run%i_subsecond_plot.pdf'%run, dpi=300)

        print('Events from run %i that are expected to be 60 Hz events using the test statistic cut of %0.2f can be found by printing eventids[load_cut][~cut], which is executed below:'%(run,TS_cut_level))
        print(eventids[load_cut][~cut])
        print('If you would like to save the above eventids you can do so by either hacking in code below this print statement or doing so after the fact.')
        # numpy.savetxt('./run%i_60Hz_eventids.csv'%run,eventids[load_cut][~cut])  #COULD USE THIS TO SAVE