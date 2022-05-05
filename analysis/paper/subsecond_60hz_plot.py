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
from beacon.tools.data_slicer import dataSlicer

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from pprint import pprint
import itertools
import warnings
import h5py
import inspect
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

label_fontsize = 20
plt.rc('xtick',labelsize=label_fontsize)
plt.rc('ytick',labelsize=label_fontsize)

if __name__ == '__main__':
    plt.close('all')
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 5840
        #run = 5911
        #5760
        #5840
    datapath = os.environ['BEACON_DATA']
    normalize_by_window_index = False
    normalize_by_density = True
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

        numpy.random.seed(2)
        randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))


        #This was used in initial testing of diffFromPeriodic to demonstrate it can work, fontsize=label_fontsize.
        window_s = 20.0
        rate_hz = 60.0
        expected_period = 1/rate_hz
        fold_subsecond_plot = False

        metric = diffFromPeriodic(calibrated_trig_time,window_s=window_s, expected_period=expected_period, normalize_by_window_index=normalize_by_window_index, normalize_by_density=normalize_by_density, plot_sample_hist=False, plot_test_plot_A=False, fontsize=label_fontsize)
        metric_rand = diffFromPeriodic(randomized_times,window_s=window_s, expected_period=expected_period, atol=0.001, normalize_by_window_index=normalize_by_window_index, normalize_by_density=normalize_by_density, plot_sample_hist=False, fontsize=label_fontsize)
        bins = numpy.linspace(min(min(metric),min(metric_rand)),max(max(metric),max(metric_rand)),100)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        

        TS_cut_level = 0.05#numpy.max(metric_rand) + 0.005#0.08#0.1
        zoom = 4
        double_sided = False

        # sub region of the original image
        x1, x2, y1, y2 = 30, 40, 0.2, 0.3#20, 40, 0.2, 0.4

        if double_sided:
            cut = numpy.abs(metric) <= TS_cut_level
        else:
            cut = metric <= TS_cut_level


        for mode in [2]:
            if mode == 0:
                fig = plt.figure(figsize=(30,15))
                ax = plt.subplot(1,3,1)
                inset_axes = zoomed_inset_axes(ax,
                                       zoom, # zoom factor
                                       loc=1)

                for _ax, marker in [[ax, ','], [inset_axes, 'o']]:

                    if marker == ',':
                        scatter_1 = _ax.plot((calibrated_trig_time-min(calibrated_trig_time))/60.0,calibrated_trig_time % (expected_period if fold_subsecond_plot else 1.0),marker=marker,linestyle='None',c='b',label='All RF Events')#,s=1
                    else:
                        scatter_1 = _ax.scatter((calibrated_trig_time-min(calibrated_trig_time))/60.0,calibrated_trig_time % (expected_period if fold_subsecond_plot else 1.0),marker=marker,linestyle='None',c='b',label='All RF Events',s=4)
                plt.sca(ax)
                plt.legend(fontsize=label_fontsize, loc='lower right',framealpha=1)
                
                if fold_subsecond_plot == True:
                    plt.ylabel('Trigger Time Remainder\nFrom Expected Period (s)',fontsize=label_fontsize)
                    plt.ylim(0,expected_period)
                else:
                    plt.ylabel('Trigger Sub-Second (s)',fontsize=label_fontsize)
                    plt.ylim(0,1)
                plt.xlabel('Time From Start of Run (min)',fontsize=label_fontsize)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                #Zoomed inset axis
                plt.sca(inset_axes)

                inset_axes.set_xlim(x1, x2)
                inset_axes.set_ylim(y1, y2)

                plt.xticks(visible=False)
                plt.yticks(visible=False)

                # draw a bbox of the region of the inset axes in the parent axes and
                # connecting lines between the bbox and the inset axes area
                mark_inset(ax, inset_axes, loc1=2, loc2=4, fc="none", ec="k", lw=2)


                ax2 = plt.subplot(1,3,2, sharex=ax, sharey=ax)
                # for _ax, marker in [[ax2, 'o'], [inset_axes, 'o']]:
                for _ax, marker in [[ax2, 'o']]:
                    if marker == ',':
                        scatter_2 = _ax.plot((calibrated_trig_time[~cut]-min(calibrated_trig_time))/60.0,calibrated_trig_time[~cut] % (expected_period if fold_subsecond_plot else 1.0),marker=marker,linestyle='None',c='r',label='Flagged %i Hz'%rate_hz)#,s=1
                    else:
                        scatter_2 = _ax.scatter((calibrated_trig_time[~cut]-min(calibrated_trig_time))/60.0,calibrated_trig_time[~cut] % (expected_period if fold_subsecond_plot else 1.0),marker=marker,linestyle='None',c='r',label='Flagged %i Hz'%rate_hz,s=0.5)
                plt.sca(ax2)
                plt.xlim(0,100)
                if fold_subsecond_plot == True:
                    plt.ylim(0,expected_period)
                else:
                    plt.ylim(0,1)
                plt.legend(fontsize=label_fontsize, loc='upper right',framealpha=1)
                
                if False:
                    plt.ylabel('Trigger Time Remainder\nFrom Expected Period (s)',fontsize=label_fontsize)
                else:
                    plt.ylabel('Trigger Sub-Second (s)',fontsize=label_fontsize)
                plt.xlabel('Time From Start of Run (min)',fontsize=label_fontsize)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                plt.subplot(1,3,3)
                normalize = False
                if double_sided:
                    plt.axvspan(-1,-TS_cut_level,color='r',alpha=0.3,label='Cut Region')
                    plt.axvspan(TS_cut_level,1,color='r',alpha=0.3)
                if normalize:
                    n, bin_edges, patches = plt.hist(metric,alpha=0.7,density=True,label='Actual\nTrigger Times',bins=bins)
                    plt.hist(metric_rand,alpha=0.7,density=True,label='Uniformly\nRandomized\nTrigger Times',bins=bin_edges)
                    plt.ylabel('Normalized Counts',fontsize=label_fontsize)
                    arrow_xy = (0.115,1.15)
                    relative_xy = (0.05, 0.1)
                else:
                    n, bin_edges, patches = plt.hist(metric,alpha=0.7,density=False,label='Actual\nTrigger Times',bins=bins)
                    plt.hist(metric_rand,alpha=0.7,density=False,label='Uniformly\nRandomized\nTrigger Times',bins=bin_edges)
                    plt.ylabel('Counts',fontsize=label_fontsize)
                    arrow_xy = (0.115,4700)
                    relative_xy = (0.05, 1000)
                plt.xlabel('Test Statistic',fontsize=label_fontsize)

                plt.xlim(min(bin_edges),max(bin_edges))
                if not double_sided:
                    plt.axvline(TS_cut_level,c='r',linestyle='--',linewidth=3, label='Cut')
                

                plt.legend(fontsize=label_fontsize, loc='upper right',framealpha=1)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                ann = ax.annotate("Flagged %i Hz"%rate_hz,
                  xy=arrow_xy, xycoords='data',
                  xytext=(arrow_xy[0]+relative_xy[0],arrow_xy[1]+relative_xy[1]), textcoords='data',
                  size=label_fontsize, va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(  arrowstyle="-[,widthB=0.55, lengthB=0.2, angleB=0",
                                    fc="w"),
                  )


                plt.tight_layout()
                plt.subplots_adjust(left=0.06, bottom=0.12,right=0.98, top=0.95, wspace=0.3, hspace=0.2)

                fig.savefig('./run%i_subsecond_plot_mode%i.pdf'%(run,mode), dpi=300)

                print('Events from run %i that are expected to be 60 Hz events using the test statistic cut of %0.2f can be found by printing eventids[load_cut][~cut], which is executed below:'%(run,TS_cut_level))
                print(eventids[load_cut][~cut])
                print('If you would like to save the above eventids you can do so by either hacking in code below this print statement or doing so after the fact.')
                # numpy.savetxt('./run%i_60Hz_eventids.csv'%run,eventids[load_cut][~cut])  #COULD USE THIS TO SAVE
            elif mode == 1:
                fig = plt.figure(figsize=(16,8))
                ax = plt.subplot(1,2,1)

                scatter_1 = ax.plot((calibrated_trig_time-min(calibrated_trig_time))/60.0,calibrated_trig_time % (expected_period if fold_subsecond_plot else 1.0),marker=',',linestyle='None',c='b',label='All RF Events')#,s=1
                scatter_2 = ax.scatter((calibrated_trig_time[~cut]-min(calibrated_trig_time))/60.0,calibrated_trig_time[~cut] % (expected_period if fold_subsecond_plot else 1.0),marker=marker,linestyle='None',c='r',label='Flagged %i Hz'%rate_hz,s=4)
                
                plt.legend(fontsize=label_fontsize, loc='lower right',framealpha=1)
                
                if fold_subsecond_plot == True:
                    plt.ylabel('Trigger Time Remainder\nFrom Expected Period (s)',fontsize=label_fontsize)
                    plt.ylim(0,expected_period)
                else:
                    plt.ylabel('Trigger Sub-Second (s)',fontsize=label_fontsize)
                    plt.ylim(0,1)
                plt.xlabel('Time From Start of Run (min)',fontsize=label_fontsize)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.xlim(0,100)


                plt.subplot(1,2,2)
                normalize = False

                if double_sided:
                    plt.axvspan(-1,-TS_cut_level,color='r',alpha=0.3,label='Cut Region')
                    plt.axvspan(TS_cut_level,1,color='r',alpha=0.3)
                if normalize:
                    n, bin_edges, patches = plt.hist(metric,alpha=0.7,density=True,label='Actual\nTrigger Times',bins=bins)
                    plt.hist(metric_rand,alpha=0.7,density=True,label='Uniformly\nRandomized\nTrigger Times',bins=bin_edges)
                    plt.ylabel('Normalized Counts',fontsize=label_fontsize)
                    arrow_xy = (0.115,1.15)
                    relative_xy = (0.05, 0.1)
                else:
                    n, bin_edges, patches = plt.hist(metric,alpha=0.7,density=False,label='Actual\nTrigger Times',bins=bins)
                    plt.hist(metric_rand,alpha=0.7,density=False,label='Uniformly\nRandomized\nTrigger Times',bins=bin_edges)
                    plt.ylabel('Counts',fontsize=label_fontsize)
                    arrow_xy = (0.115,4700)
                    relative_xy = (0.05, 1000)

                if not double_sided:
                    plt.axvline(TS_cut_level,c='r',linestyle='--',linewidth=3, label='Cut')
                plt.xlim(min(bin_edges),max(bin_edges))

                plt.xlabel('Test Statistic',fontsize=label_fontsize)
                plt.legend(fontsize=label_fontsize, loc='upper right',framealpha=1)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                ann = ax.annotate("Flagged %i Hz"%rate_hz,
                  xy=arrow_xy, xycoords='data',
                  xytext=(arrow_xy[0]+relative_xy[0],arrow_xy[1]+relative_xy[1]), textcoords='data',
                  size=label_fontsize, va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(  arrowstyle="-[,widthB=0.55, lengthB=0.2, angleB=0",
                                    fc="w"),
                  )


                plt.tight_layout()
                plt.subplots_adjust(left=0.06, bottom=0.12,right=0.98, top=0.95, wspace=0.3, hspace=0.2)

                fig.savefig('./run%i_subsecond_plot_mode%i.pdf'%(run,mode), dpi=300)

                print('Events from run %i that are expected to be 60 Hz events using the test statistic cut of %0.2f can be found by printing eventids[load_cut][~cut], which is executed below:'%(run,TS_cut_level))
                print(eventids[load_cut][~cut])
                print('If you would like to save the above eventids you can do so by either hacking in code below this print statement or doing so after the fact.')
                # numpy.savetxt('./run%i_60Hz_eventids.csv'%run,eventids[load_cut][~cut])  #COULD USE THIS TO SAVE
            elif mode == 2:
                fig = plt.figure(figsize=(20,10))

                inset = False

                axs = [plt.subplot(2,3,2)]
                axs.append(plt.subplot(2,3,5, sharex=axs[0]))

                metric, axs = diffFromPeriodic(calibrated_trig_time,window_s=window_s, expected_period=expected_period, normalize_by_window_index=normalize_by_window_index, normalize_by_density=normalize_by_density, plot_sample_hist=False, plot_test_plot_A=False, axs=axs, fontsize=label_fontsize)
                
                axs[1].set_ylim(0,axs[1].get_ylim()[1]*1.2)

                if inset:
                    ax = plt.subplot(1,3,1)
                    inset_axes = zoomed_inset_axes(ax,
                                       zoom, # zoom factor
                                       loc=1)
                    scatter_1 = ax.plot((calibrated_trig_time[cut]-min(calibrated_trig_time))/60.0,calibrated_trig_time[cut] % (expected_period if fold_subsecond_plot else 1.0),marker=',',linestyle='None',c='b',label='Non-Flagged RF Triggers')#,s=1
                    scatter_2 = ax.scatter((calibrated_trig_time[~cut]-min(calibrated_trig_time))/60.0,calibrated_trig_time[~cut] % (expected_period if fold_subsecond_plot else 1.0),marker='o',linestyle='None',c='r',label='Flagged %i Hz'%rate_hz,s=0.5)

                    inset_axes.scatter((calibrated_trig_time[~cut]-min(calibrated_trig_time))/60.0,calibrated_trig_time[~cut] % (expected_period if fold_subsecond_plot else 1.0),marker='o',linestyle='None',c='r',label='Flagged %i Hz'%rate_hz,s=4)
                    plt.sca(ax)
                    plt.legend(fontsize=label_fontsize, loc='lower right',framealpha=1)
                    if False:
                        plt.ylabel('Trigger Time Remainder\nFrom Expected Period (s)',fontsize=label_fontsize)
                    else:
                        plt.ylabel('Trigger Sub-Second (s)',fontsize=label_fontsize)
                    plt.xlabel('Time From Start of Run (min)',fontsize=label_fontsize)
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.xlim(0,100)
                    if fold_subsecond_plot == True:
                        plt.ylim(0,expected_period)
                    else:
                        plt.ylim(0,1)

                    #Zoomed inset axis
                    plt.sca(inset_axes)

                    inset_axes.set_xlim(x1, x2)
                    inset_axes.set_ylim(y1, y2)

                    plt.xticks(visible=False)
                    plt.yticks(visible=False)

                    # draw a bbox of the region of the inset axes in the parent axes and
                    # connecting lines between the bbox and the inset axes area
                    mark_inset(ax, inset_axes, loc1=2, loc2=4, fc="none", ec="k", lw=2)
                else:
                    ax1 = plt.subplot(2,3,1)
                    scatter_1 = ax1.plot((calibrated_trig_time[cut]-min(calibrated_trig_time))/60.0,calibrated_trig_time[cut] % (expected_period if fold_subsecond_plot else 1.0),marker=',',linestyle='None',c='b',label='Non-Flagged RF Triggers')#,s=1
                    ax2 = plt.subplot(2,3,4, sharex=ax1, sharey=ax1)
                    scatter_2 = ax2.plot((calibrated_trig_time[~cut]-min(calibrated_trig_time))/60.0,calibrated_trig_time[~cut] % (expected_period if fold_subsecond_plot else 1.0),marker=',',linestyle='None',c='r',label='Flagged %i Hz'%rate_hz)
                    #scatter_2 = ax2.scatter((calibrated_trig_time[~cut]-min(calibrated_trig_time))/60.0,calibrated_trig_time[~cut] % (expected_period if fold_subsecond_plot else 1.0),marker='o',linestyle='None',c='r',label='Flagged %i Hz'%rate_hz,s=0.5)
                    
                    ax1.legend(fontsize=label_fontsize, loc='lower right',framealpha=1)
                    ax2.legend(fontsize=label_fontsize, loc='lower right',framealpha=1)

                    #Zoomed inset axis
                    inset_axes1 = zoomed_inset_axes(ax1,
                                       zoom, # zoom factor
                                       loc=1)
                    [inset_axes1.spines[k].set_linewidth(2) for k in inset_axes1.spines.keys()]
                    plt.sca(inset_axes1)
                    inset_axes1.scatter((calibrated_trig_time[cut]-min(calibrated_trig_time))/60.0,calibrated_trig_time[cut] % (expected_period if fold_subsecond_plot else 1.0),marker='o',linestyle='None',c='b',label='Flagged %i Hz'%rate_hz,s=1)

                    inset_axes1.set_xlim(x1, x2)
                    inset_axes1.set_ylim(y1, y2)

                    minor_ticks = numpy.arange(inset_axes1.get_ylim()[0] - expected_period, inset_axes1.get_ylim()[1] + expected_period, expected_period) + expected_period/2
            

                    inset_axes1.set_yticks(minor_ticks)
                    plt.grid(b=True, which='major', color='tab:gray', linestyle='--',alpha=0.5)

                    plt.xticks(visible=False)
                    plt.yticks(visible=False)
                    # draw a bbox of the region of the inset axes in the parent axes and
                    # connecting lines between the bbox and the inset axes area
                    mark_inset(ax1, inset_axes1, loc1=3, loc2=4, fc="none", ec="k", lw=2)

                    #Zoomed inset axis
                    inset_axes2 = zoomed_inset_axes(ax2,
                                       zoom, # zoom factor
                                       loc=1)
                    [inset_axes2.spines[k].set_linewidth(2) for k in inset_axes2.spines.keys()]
                    plt.sca(inset_axes2)
                    inset_axes2.scatter((calibrated_trig_time[~cut]-min(calibrated_trig_time))/60.0,calibrated_trig_time[~cut] % (expected_period if fold_subsecond_plot else 1.0),marker='o',linestyle='None',c='r',label='Flagged %i Hz'%rate_hz,s=4)

                    inset_axes2.set_xlim(x1, x2)
                    inset_axes2.set_ylim(y1, y2)

                    inset_axes2.set_yticks(minor_ticks)
                    plt.grid(b=True, which='major', color='tab:gray', linestyle='--',alpha=0.5)

                    plt.xticks(visible=False)
                    plt.yticks(visible=False)

                    start_tick = 2
                    arrow_xy = ( x1 + (x2-x1)/2  - 0.10*(x2-x1)/2 , (minor_ticks[start_tick] + minor_ticks[start_tick+1])/2)
                    text_xy =  ( x1 + (x2-x1)/2  - 0.40*(x2-x1)/2 , (minor_ticks[start_tick] + minor_ticks[start_tick+1])/2)
                    ann = inset_axes2.annotate("$1/r$",
                          weight='bold',
                          xy=arrow_xy, xycoords='data',
                          xytext=text_xy, textcoords='data',
                          size=label_fontsize-2, va="center", ha="center",
                          bbox=dict(boxstyle="round", fc="w", lw=2),
                          arrowprops=dict(  arrowstyle="-[,widthB=0.55, lengthB=0.2, angleB=0",
                                            fc="w", lw=2),
                          )

                    # draw a bbox of the region of the inset axes in the parent axes and
                    # connecting lines between the bbox and the inset axes area
                    mark_inset(ax2, inset_axes2, loc1=3, loc2=4, fc="none", ec="k", lw=2)

                    if False:
                        ax1.set_ylabel('Trigger Time Remainder\nFrom Expected Period (s)',fontsize=label_fontsize)
                        ax2.set_ylabel('Trigger Time Remainder\nFrom Expected Period (s)',fontsize=label_fontsize)
                    else:
                        ax1.set_ylabel('Trigger Sub-Second (s)',fontsize=label_fontsize)
                        ax2.set_ylabel('Trigger Sub-Second (s)',fontsize=label_fontsize)
                    ax2.set_xlabel('Time From Start of Run (min)',fontsize=label_fontsize)
                    
                    # plt.minorticks_on()
                    # plt.grid(b=True, which='major', color='k', linestyle='-')
                    # plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    ax1.set_xlim(0,60)
                    if fold_subsecond_plot == True:
                        ax1.set_ylim(0,expected_period)
                    else:
                        ax1.set_ylim(0,1)





                plt.subplot(1,3,3)
                normalize = False

                if double_sided:
                    plt.axvspan(-1,-TS_cut_level,color='r',alpha=0.3,label='Cut Region')
                    plt.axvspan(TS_cut_level,1,color='r',alpha=0.3)
                if normalize:
                    n, bin_edges, patches = plt.hist(metric,alpha=0.7,density=True,label='Actual\nTrigger Times',bins=bins)
                    plt.hist(metric_rand,alpha=0.7,density=True,label='Uniformly\nRandomized\nTrigger Times',bins=bin_edges)
                    plt.ylabel('Normalized Counts',fontsize=label_fontsize, labelpad=0)
                    arrow_xy = (0.115,1.15)
                    relative_xy = (0.05, 0.1)
                else:
                    n, bin_edges, patches = plt.hist(metric,alpha=0.7,density=False,label='Actual\nTrigger Times',bins=bins)
                    plt.hist(metric_rand,alpha=0.7,density=False,label='Uniformly\nRandomized\nTrigger Times',bins=bin_edges)
                    plt.ylabel('Counts',fontsize=label_fontsize, labelpad=-5)
                    arrow_xy = (0.115,4700)
                    relative_xy = (0.05, 1000)
                plt.xlabel('Test Statistic',fontsize=label_fontsize)

                if not double_sided:
                    plt.axvline(TS_cut_level,c='r',linestyle='--',linewidth=3, label='Cut')
                plt.xlim(min(bin_edges),max(bin_edges))

                plt.legend(fontsize=label_fontsize, loc='upper right',framealpha=1)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                # ann = ax.annotate("Flagged %i Hz"%rate_hz,
                #   xy=arrow_xy, xycoords='data',
                #   xytext=(arrow_xy[0]+relative_xy[0],arrow_xy[1]+relative_xy[1]), textcoords='data',
                #   size=label_fontsize, va="center", ha="center",
                #   bbox=dict(boxstyle="round", fc="w"),
                #   arrowprops=dict(  arrowstyle="-[,widthB=0.55, lengthB=0.2, angleB=0",
                #                     fc="w"),
                #   )


                plt.tight_layout()
                plt.subplots_adjust(left=0.06, bottom=0.12,right=0.98, top=0.95, wspace=0.3, hspace=0.2)

                fig.savefig('./run%i_subsecond_plot_mode%i.pdf'%(run,mode), dpi=300)

                print('Events from run %i that are expected to be 60 Hz events using the test statistic cut of %0.2f can be found by printing eventids[load_cut][~cut], which is executed below:'%(run,TS_cut_level))
                print(eventids[load_cut][~cut])
                print('If you would like to save the above eventids you can do so by either hacking in code below this print statement or doing so after the fact.')

                if False:
                    cmap = 'cool'
                    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
                    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
                    map_length = 16384
                    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length

                    print("Preparing dataSlicer")
                    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=os.environ['BEACON_PROCESSED_DATA'])
                    ds.conference_mode = True

                    eventids_dict = {run:numpy.where(~cut)[0]}
                    map_fig, map_ax = ds.plot2dHist('phi_best_choice', 'elevation_best_choice', eventids_dict, title=None,cmap='coolwarm', lognorm=True, return_counts=False, mask_top_N_bins=0, fill_value=0)
