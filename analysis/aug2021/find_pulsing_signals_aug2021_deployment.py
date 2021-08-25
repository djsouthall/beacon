#!/usr/bin/env python3
'''
This script is intended to be used to determine if pulsing signals are visible in the current (directed to) run of data.
It will do so mostly by creating a subsecond v.s. second plot.  Events from this plot can be circled, and their event
ids will be printed.  These can be copy-pasted into another document or somewhere for saving.
'''
import os
import sys
import numpy

from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
import beacon.tools.interpret #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_slicer import dataSlicerSingleRun,dataSlicer
from beacon.tools.correlator import Correlator

import csv
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
from pprint import pprint
import itertools
import warnings
import h5py
import inspect
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import pandas as pd
numpy.set_printoptions(threshold=sys.maxsize)
def txtToClipboard(txt):
    '''
    Seems to bug out on midway.
    '''
    df=pd.DataFrame([txt])
    df.to_clipboard(index=False,header=False)
plt.ion()

# import beacon.tools.info as info
from beacon.tools.data_handler import createFile, getTimes, loadTriggerTypes, getEventTimes
# from beacon.tools.fftmath import FFTPrepper
# from beacon.analysis.background_identify_60hz import plotSubVSec, plotSubVSecHist, alg1, diffFromPeriodic, get60HzEvents2, get60HzEvents3, get60HzEvents
from beacon.tools.fftmath import TemplateCompareTool

from beacon.analysis.find_pulsing_signals_june2021_deployment import Selector

known_pulser_cuts_for_template = {}

if __name__ == '__main__':
    plt.close('all')
    #### Initial Search ####
    '''
    This run should be the easiest to find a signal, though it is not the first run.  I may use this to get a template
    and then use that template to get everything else.
    '''
    limit_events = 40000
    mask_events = False #Will mask events not meeting some minimum z value

    if len(sys.argv) == 2:
        runs = numpy.array([int(sys.argv[1])])
    else:
        runs = numpy.array([5608])#numpy.arange(5181,5186)

    lassos = []

    for run in runs:
        try:
            run_key = 'run%i'%run
            print(run_key)
            reader = Reader('/home/dsouthall/Projects/Beacon/beacon/analysis/aug2021/data/',run)#Reader(os.environ['BEACON_DATA'],run)
            tct = TemplateCompareTool(reader,apply_phase_response=True)#waveform_index_range=[0,2100]

            #Sloppy work around to run code if file isn't loading (i.e. if scratch files aren't available)
            print('Running script in mode not requiring access to file.')
            trigger_type = loadTriggerTypes(reader)
            eventids = numpy.arange(len(trigger_type))
            all_beams = reader.returnTriggerInfo()[0][eventids]

            c = numpy.zeros_like(eventids).astype(float)

            if run_key in list(known_pulser_cuts_for_template.keys()):
                passing_eventids = numpy.array([],dtype=int)
                for cut_entry in known_pulser_cuts_for_template[run_key]:
                    #Loop over the list of cuts, and append passing events to the list of passing_eventids.
                    load_cut = trigger_type == cut_entry['trigger_type']
                    calibrated_trig_time = getEventTimes(reader)[load_cut]
                    second_time = numpy.floor(calibrated_trig_time)
                    subsecond_time = calibrated_trig_time - second_time
                    secondary_cut = numpy.ones_like(eventids[load_cut]).astype(bool)
                    if 'time_window' in list(cut_entry.keys()):
                        secondary_cut = numpy.logical_and(secondary_cut, numpy.logical_and(calibrated_trig_time >= min(cut_entry['time_window']), calibrated_trig_time <= max(cut_entry['time_window'])))
                    if 'time_window_ns' in list(cut_entry.keys()):
                        secondary_cut = numpy.logical_and(secondary_cut, numpy.logical_and(subsecond_time*1e9 >= min(cut_entry['time_window_ns']), subsecond_time*1e9 <= max(cut_entry['time_window_ns'])))
                    _passing_eventids = eventids[load_cut][secondary_cut]
                    passing_eventids = numpy.append(passing_eventids,_passing_eventids)

                #Sort and sift through the passing eventids. 
                passing_eventids = numpy.unique(passing_eventids)

                #Randomly choose the events upto the limit given.
                if len(passing_eventids) > 0:
                    passing_eventids = numpy.sort(numpy.random.choice(passing_eventids,size=min(len(passing_eventids),limit_events),replace=False))
                    template_eventid = numpy.random.choice(passing_eventids,size=1,replace=False)[0]
                else:
                    template_eventid = None

                times, averaged_waveforms = tct.averageAlignedSignalsPerChannel(passing_eventids, align_method=0, template_eventid=template_eventid, plot=True,event_type=None,sine_subtract=True)


            else:
                passing_eventids = None
                template_eventid = None


            passing_beams = reader.returnTriggerInfo()[0][passing_eventids]

            if True:
                plt.figure()
                ax = plt.subplot(2,1,1)
                plt.hist(all_beams,bins=numpy.arange(21)-0.5)
                plt.ylabel('Counts')
                plt.xlabel('Triggered Beam')
                plt.subplot(2,1,2,sharex=ax)
                for trigger in [1,2,3]:
                    plt.hist(passing_beams[trigger_type[passing_eventids] == trigger],bins=numpy.arange(21)-0.5,label='Trigtype = %i'%trigger,alpha=0.6)
                plt.legend()
                plt.ylabel('Counts')
                plt.xlabel('Triggered Beam')

            default_c = 'ptp' #Either 'beam' or 'ptp' to select colorscheme when no template is available.

            if template_eventid is None and default_c == 'beam':
                ylabel = 'Beam Number'
                zlim = (0,21)
                c = numpy.copy(all_beams)
            elif template_eventid is None and default_c == 'ptp':
                ylabel = 'Peak To Peak'
                zlim = (0,128)
                for event_index, eventid in enumerate(eventids):
                    reader.setEntry(eventid)
                    c[event_index] = numpy.ptp(reader.wf(0))
            else:
                ylabel = 'Max Corr Value'
                zlim = (0,1)
                for event_index, eventid in enumerate(eventids):
                    corrs, corrs_fft = tct.crossCorrelateWithTemplate(eventid, load_upsampled_waveforms=False,sine_subtract=True)
                    c[event_index] = numpy.max(corrs)

            corr_lim = 0.7
            if template_eventid is not None:
                all_events_above_corr = numpy.where(c > corr_lim)[0]

            for cut_trigger in [3]:#[1,2,3]:
                load_cut = trigger_type == cut_trigger
                calibrated_trig_time = getEventTimes(reader)[load_cut]
                #passing_load_cut = numpy.sort(numpy.where(load_cut)[0] in passing_eventids)
                print('Trigger type %i, %i events'%(cut_trigger, numpy.sum(load_cut)))

                if template_eventid is not None:
                    print('Events of trigger type %i with correlation above %0.2f correlation value:'%(cut_trigger,corr_lim))
                    print(pprint(eventids[load_cut][c[load_cut] > corr_lim]))

                #randomized_times = numpy.sort(numpy.random.uniform(low=calibrated_trig_time[0],high=calibrated_trig_time[-1],size=(len(calibrated_trig_time),)))
                plt.figure()
                plt.title('Run %i'%(run))
                plt.hist(c[load_cut])
                plt.title('Trigtype = %i'%cut_trigger)
                plt.xlabel(ylabel)
                plt.ylabel('Counts')

                second_time = numpy.floor(calibrated_trig_time)
                subsecond_time = calibrated_trig_time - second_time

                #period_factor = 1.0#1/20.0#1.0#1/60.0 #1.0 #In seconds, what the remainder will be taken of.
                if cut_trigger == 2:
                    pfs = [1.0]#[1.0,1/20.0,1/60.0,1/100.0,1/120.0]
                else:
                    pfs = [1.0]

                for index, period_factor in enumerate(pfs):

                    fig = plt.figure()
                    if index == 0:
                        ax1 = plt.gca()
                    else:
                        plt.subplot(1,1,1, sharex=ax1)
                    ax = plt.gca()
                    plt.title('Run %i Trigger Type = %i\n%0.2f Hz Expected'%(run, cut_trigger, 1/period_factor))

                    if mask_events == True:
                        if template_eventid is not None:
                            calibrated_trig_time = numpy.ma.masked_array(calibrated_trig_time, mask = c[load_cut] < corr_lim)
                            subsecond_time = numpy.ma.masked_array(subsecond_time, mask = c[load_cut] < corr_lim)
                        else:
                            if default_c == 'ptp':




                                colour_lim = 0#124






                                zlim = (colour_lim, zlim[1])
                                calibrated_trig_time = numpy.ma.masked_array(calibrated_trig_time, mask = c[load_cut] < colour_lim)
                                subsecond_time = numpy.ma.masked_array(subsecond_time, mask = c[load_cut] < colour_lim)

                        _eventids = numpy.ma.masked_array(eventids[load_cut], mask = calibrated_trig_time.mask)
                    else:
                        _eventids = eventids[load_cut]
                    scatter = plt.scatter(calibrated_trig_time, (subsecond_time%period_factor)*1e9 ,c=c[load_cut])#plt.scatter(calibrated_trig_time, subsecond_time*1e9,marker=',',lw=0, s=1,c=c[load_cut])
                    if True:
                        # plt.axvline(1629248191,c='r',label='0dB Sat Cut On')
                        # plt.axvline(1629248550,c='b',label='0dB Sat Cut Off')
                        #plt.axvline(1629325054,c='b',label='Set GPS Delay to 2.29 us')
                        # plt.axvline(1629326556,c='r',label='Andrew returned pulser delay to 0 us (from 5 us)')
                        # plt.axvline(1629326662,c='b',label='Andrew returned pulser delay to 5 us (from 0 us)')
                        # plt.axvline(1629327560 , c='b',label='20 db Attenuation Started')
                        # plt.axvline(1629328334 , c='g',label='10 db Attenuation Started')
                        #plt.axvline(1629330230 , c='g',label='30 db Attenuation Started (From 13 dB before)')
                        plt.axvline(1629332414 , c='g',label='10 db Attenuation Started (From 13 dB before)')
                        plt.axvline(1629333157 , c='r',label='20 db Attenuation Started (From 10 dB before)')
                        
                        

                        plt.legend()

                    cbar = fig.colorbar(scatter)
                    plt.clim(zlim[0],zlim[1])
                    plt.ylabel('Subsecond Time (ns)')
                    plt.xlabel('Trigger Time (s)')
                    _s = Selector(ax,scatter,_eventids, run)
                    lassos.append(_s)

                    if True:
                        if passing_eventids is not None:
                            c_cut = numpy.isin(eventids[load_cut], passing_eventids)
                            scatter2 = plt.scatter(calibrated_trig_time[c_cut], (subsecond_time[c_cut]%period_factor)*1e9,marker=',',lw=0, s=1,c='r')

                if False:
                    if cut_trigger == 3:
                        plt.figure()
                        plt.title('Run %i'%(run))
                        t = reader.t()
                        for eventid in eventids[load_cut]:
                            reader.setEntry(eventid)
                            wf = reader.wf(0)
                            roll = numpy.where(wf >= 0.8*numpy.max(wf))[0][0]
                            wf = numpy.roll(wf, roll)
                            plt.plot(t, wf, alpha = 0.3, c='k')
                            #plt.scatter(eventid, roll, alpha = 0.3, c='k')
                        plt.xlabel('t (ns)')
                        plt.ylabel('V (adu)')

                if False:
                    if cut_trigger == 3:
                        times, averaged_waveforms = tct.averageAlignedSignalsPerChannel(eventids[load_cut], template_eventid=eventids[load_cut][-1], align_method=0, plot=True, sine_subtract=False)
                        times_ns = times/1e9
                        freqs_MHz = numpy.fft.rfftfreq(len(times_ns),d=numpy.diff(times_ns)[0])/1e6
                        #Plot averaged FFT per channel
                        fft_fig = plt.figure()
                        plt.title('Run %i'%(run))
                        for mode_index, mode in enumerate(['hpol','vpol']):
                            plt.subplot(2,1,1+mode_index)
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                            plt.ylabel('%s dBish'%mode)
                            plt.xlabel('MHz')
                            fft_ax = plt.gca()
                            for channel, averaged_waveform in enumerate(averaged_waveforms):
                                if mode == 'hpol' and channel%2 == 1:
                                    continue
                                elif mode == 'vpol' and channel%2 == 0:
                                    continue

                                freqs, spec_dbish, spec = tct.rfftWrapper(times, averaged_waveform)
                                fft_ax.plot(freqs/1e6,spec_dbish/2.0,label='Ch %i'%channel)#Dividing by 2 to match monutau.  Idk why I have to do this though normally this function has worked well...
                            plt.xlim(10,110)
                            plt.ylim(-20,30)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

