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
from tools.fftmath import FFTPrepper

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()


if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = os.environ['BEACON_DATA']
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        print('No run number given.  Defaulting to 1507')
        run = 1507

    if run == 1507:
        waveform_index_range = (1500,None) #Looking at the later bit of the waveform only, 10000 will cap off.  
    elif run == 1509:
            waveform_index_range = (2000,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
    elif run == 1511:
            waveform_index_range = (1250,2000) #Looking at the later bit of the waveform only, 10000 will cap off. 
    eventids = {}
    known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
    eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])
    eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])
    all_eventids = numpy.sort(numpy.append(eventids['hpol'],eventids['vpol']))
    cut_hpol = numpy.isin(all_eventids,eventids['hpol'])
    cut_vpol = numpy.isin(all_eventids,eventids['vpol'])

    print('Run %i'%run)
    final_corr_length = 2**18 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = 80 #This new pulser seems to peak in the region of 85 MHz or so
    crit_freq_high_pass_MHz = 65
    low_pass_filter_order = 3
    high_pass_filter_order = 6
    plot_filters = True

    reader = Reader(datapath,run)
    prep = FFTPrepper(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=plot_filters)

    wfs = {}
    rms = {}
    argmax = {}
    p2p = {}
    channels = numpy.arange(8)
    for channel in channels:
        wfs[channel] = numpy.zeros((len(all_eventids),prep.buffer_length))
        rms[channel] = numpy.zeros((len(all_eventids)))
        p2p[channel] = numpy.zeros((len(all_eventids)))
        argmax[channel] = numpy.zeros((len(all_eventids)))

    t = prep.t()
    for event_index, eventid in enumerate(all_eventids):
        sys.stdout.write('(%i/%i)\r'%(event_index,len(all_eventids)))
        sys.stdout.flush()
        prep.setEntry(eventid)
        event_times = prep.t()
        for channel in channels:
            channel=int(channel)
            wfs[channel][event_index] = prep.wf(channel)
            rms[channel][event_index] = numpy.std(prep.wf(channel))
            p2p[channel][event_index] = numpy.max(prep.wf(channel)) - numpy.min(prep.wf(channel))
            argmax[channel][event_index] = numpy.argmax(prep.wf(channel))
            if False:
                fig = plt.figure()
                plt.plot(t,prep.wf(channel))
                import pdb;pdb.set_trace()
                plt.close(fig)

    alpha = 0.2
    split_plots = False
    for channel in channels:
        plt.figure()
        plt.suptitle(str(channel))
        plt.subplot(4,1,1)
        plt.scatter(eventids['hpol'],argmax[channel][cut_hpol],color='r',alpha=0.5,label='hpol')
        plt.scatter(eventids['vpol'],argmax[channel][cut_vpol],color='b',alpha=0.5,label='vpol')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.xlabel('eventid')
        plt.ylabel('argmax')
        plt.legend()

        plt.subplot(4,1,2)
        plt.scatter(eventids['hpol'],rms[channel][cut_hpol],color='r',alpha=0.5,label='hpol')
        plt.scatter(eventids['vpol'],rms[channel][cut_vpol],color='b',alpha=0.5,label='vpol')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.xlabel('eventid')
        plt.ylabel('rms')
        plt.legend()

        plt.subplot(4,1,3)
        plt.scatter(eventids['hpol'],p2p[channel][cut_hpol],color='r',alpha=0.5,label='hpol')
        plt.scatter(eventids['vpol'],p2p[channel][cut_vpol],color='b',alpha=0.5,label='vpol')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.xlabel('eventid')
        plt.ylabel('p2p')
        plt.legend()

        plt.subplot(4,1,4)



        for index, wf in enumerate(wfs[channel]):
            if cut_hpol[index]:
                plt.plot(t,wf,c='r',alpha=alpha)
            elif cut_vpol[index]:
                plt.plot(t,wf,c='b',alpha=alpha)
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    if run == 1507:

        argmax_cuts_x = {0:numpy.logical_or(numpy.logical_and(all_eventids > 15795, all_eventids < 16700) , numpy.logical_and(all_eventids > 17500, all_eventids < 19000)) }

        rms_cuts_y = {  0:numpy.logical_or( numpy.logical_and(rms[0] < 8.0, all_eventids < 17000), numpy.logical_and(rms[0] < 2.6, all_eventids > 17000)),\
                        1:numpy.logical_or( numpy.logical_and(rms[1] < 1.5, all_eventids < 17000), numpy.logical_and(rms[1] < 6.2, all_eventids > 17000)),\
                        2:numpy.logical_or( numpy.logical_and(rms[2] < 7.0, all_eventids < 17000), numpy.logical_and(rms[2] < 2.5, all_eventids > 17000)),\
                        3:numpy.logical_or( numpy.logical_and(rms[3] < 2.0, all_eventids < 17000), numpy.logical_and(rms[3] < 7.0, all_eventids > 17000)),\
                        4:numpy.logical_or( numpy.logical_and(rms[4] < 12.0, all_eventids < 17000), numpy.logical_and(rms[4] < 2.7, all_eventids > 17000)),\
                        5:numpy.logical_or( numpy.logical_and(rms[5] < 1.5, all_eventids < 17000), numpy.logical_and(rms[5] < 6.3, all_eventids > 17000)),\
                        6:numpy.logical_or( numpy.logical_and(rms[6] < 11.0, all_eventids < 17000), numpy.logical_and(rms[6] < 4.5, all_eventids > 17000)),\
                        7:numpy.logical_or( numpy.logical_and(rms[7] < 1.8, all_eventids < 17000), numpy.logical_and(rms[7] < 8.5, all_eventids > 17000))}

        x_cuts = numpy.ones_like(all_eventids,dtype=bool)
        y_cuts = numpy.ones_like(all_eventids,dtype=bool)
        z_cuts = numpy.ones_like(all_eventids,dtype=bool)

        for key, val in argmax_cuts_x.items():
            x_cuts = numpy.logical_and(x_cuts,val)

        plt.figure()
        for key, val in rms_cuts_y.items():
            y_cuts = numpy.logical_and(y_cuts,val)
            plt.plot(val,alpha=0.5)

        for key, val in p2p.items():
            z_cuts = numpy.logical_and(z_cuts,val < 127)

        total_cut = numpy.logical_and(numpy.logical_and(x_cuts,y_cuts),z_cuts)
        good_events = all_eventids[total_cut]
        print(good_events)
        ignore_events = all_eventids[~total_cut]
        numpy.savetxt('./run%i_pulser_ignoreids.csv'%(run), numpy.sort(ignore_events), delimiter=",")

    if run == 1509:

        argmax_cuts_x = {0:numpy.logical_or(numpy.logical_and(all_eventids > 2860, all_eventids < 3500) , numpy.logical_and(all_eventids > 4691, all_eventids < 10000)) }

        rms_cuts_y = {  0:numpy.logical_or( numpy.logical_and(rms[0] < 3.5, all_eventids < 4000), numpy.logical_and(rms[0] < 1.3, all_eventids > 4000)),\
                        1:numpy.logical_or( numpy.logical_and(rms[1] < 0.7, all_eventids < 4000), numpy.logical_and(rms[1] < 2.6, all_eventids > 4000)),\
                        2:numpy.logical_or( numpy.logical_and(rms[2] < 3.0, all_eventids < 4000), numpy.logical_and(rms[2] < 1.4, all_eventids > 4000)),\
                        3:numpy.logical_or( numpy.logical_and(rms[3] < 0.9, all_eventids < 4000), numpy.logical_and(rms[3] < 3.5, all_eventids > 4000)),\
                        4:numpy.logical_or( numpy.logical_and(rms[4] < 4.6, all_eventids < 4000), numpy.logical_and(rms[4] < 1.3, all_eventids > 4000)),\
                        5:numpy.logical_or( numpy.logical_and(rms[5] < 0.92, all_eventids < 4000), numpy.logical_and(rms[5] < 3.8, all_eventids > 4000)),\
                        6:numpy.logical_or( numpy.logical_and(rms[6] < 5.0, all_eventids < 4000), numpy.logical_and(rms[6] < 1.9, all_eventids > 4000)),\
                        7:numpy.logical_or( numpy.logical_and(rms[7] < 1.0, all_eventids < 4000), numpy.logical_and(rms[7] < 3.75, all_eventids > 4000))}

        x_cuts = numpy.ones_like(all_eventids,dtype=bool)
        y_cuts = numpy.ones_like(all_eventids,dtype=bool)
        z_cuts = numpy.ones_like(all_eventids,dtype=bool)

        for key, val in argmax_cuts_x.items():
            x_cuts = numpy.logical_and(x_cuts,val)

        plt.figure()
        for key, val in rms_cuts_y.items():
            y_cuts = numpy.logical_and(y_cuts,val)
            plt.plot(val,alpha=0.5)

        for key, val in p2p.items():
            z_cuts = numpy.logical_and(z_cuts,val < 127)

        total_cut = numpy.logical_and(numpy.logical_and(x_cuts,y_cuts),z_cuts)
        good_events = all_eventids[total_cut]
        print(good_events)
        ignore_events = all_eventids[~total_cut]
        numpy.savetxt('./run%i_pulser_ignoreids.csv'%(run), numpy.sort(ignore_events), delimiter=",")

    if run == 1511:

        argmax_cuts_x = {0:numpy.logical_or(numpy.logical_and(all_eventids > 1800, all_eventids < 2255) , numpy.logical_and(all_eventids > 4200, all_eventids < 6000)) }

        rms_cuts_y = {  0:numpy.logical_or( numpy.logical_and(rms[0] < 3.6, all_eventids < 3000), numpy.logical_and(rms[0] < 2.2, all_eventids > 3000)),\
                        1:numpy.logical_or( numpy.logical_and(rms[1] < 1.0, all_eventids < 3000), numpy.logical_and(rms[1] < 5.0, all_eventids > 3000)),\
                        2:numpy.logical_or( numpy.logical_and(rms[2] < 5.5, all_eventids < 3000), numpy.logical_and(rms[2] < 2.5, all_eventids > 3000)),\
                        3:numpy.logical_or( numpy.logical_and(rms[3] < 1.4, all_eventids < 3000), numpy.logical_and(rms[3] < 6.0, all_eventids > 3000)),\
                        4:numpy.logical_or( numpy.logical_and(rms[4] < 3.0, all_eventids < 3000), numpy.logical_and(rms[4] < 2.1, all_eventids > 3000)),\
                        5:numpy.logical_or( numpy.logical_and(rms[5] < 1.5, all_eventids < 3000), numpy.logical_and(rms[5] < 5.0, all_eventids > 3000)),\
                        6:numpy.logical_or( numpy.logical_and(rms[6] < 4.3, all_eventids < 3000), numpy.logical_and(rms[6] < 1.9, all_eventids > 3000)),\
                        7:numpy.logical_or( numpy.logical_and(rms[7] < 1.2, all_eventids < 3000), numpy.logical_and(rms[7] < 5.0, all_eventids > 3000))}

        x_cuts = numpy.ones_like(all_eventids,dtype=bool)
        y_cuts = numpy.ones_like(all_eventids,dtype=bool)
        z_cuts = numpy.ones_like(all_eventids,dtype=bool)

        for key, val in argmax_cuts_x.items():
            x_cuts = numpy.logical_and(x_cuts,val)

        for key, val in rms_cuts_y.items():
            y_cuts = numpy.logical_and(y_cuts,val)

        for key, val in p2p.items():
            z_cuts = numpy.logical_and(z_cuts,val < 127)

        total_cut = numpy.logical_and(numpy.logical_and(x_cuts,y_cuts),z_cuts)
        good_events = all_eventids[total_cut]
        print(good_events)
        ignore_events = all_eventids[~total_cut]
        numpy.savetxt('./run%i_pulser_ignoreids.csv'%(run), numpy.sort(ignore_events), delimiter=",")

