#!/usr/bin/env python3
'''
Pulsing was conducted over 2 days during the June 2021 deployment.  On the day it was unclear if signals were actually
being seen.  This script is a playground for the work needed to find those signals in the data such that they can be
used in calibration.
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


known_pulser_cuts_for_template = {  'run5176':[{'trigger_type':2, 'time_window':[1624489853.9598,1624489854.0140]}],\
                                    'run5181':[{'trigger_type':2, 'time_window':[1624496478.4,1624497197.2],'time_window_ns':[9.9997512e8,1.00000547e9]}], \
                                    'run5182':[{'trigger_type':2, 'time_window':[1624497215.3,1624497569.2],'time_window_ns':[9.9997873e8,9.9998749e9]}], \
                                    'run5183':[{'trigger_type':2, 'time_window':[1624497577.4,1624497887.8],'time_window_ns':[999980056,999984146]}], \
                                    'run5185':[{'trigger_type':2, 'time_window':[1624498820.7,1624499232.7],'time_window_ns':[9.9999e8,1e9]}],\
                                    'run5191':[{'trigger_type':2, 'time_window':[1624546559.9,1624546835.2],'time_window_ns':[1.62070e8,1.62105e8]}],\
                                    'run5195':[{'trigger_type':2, 'time_window':[1624549749.0,1624549879.2],'time_window_ns':[1.618e8,1.620e8]}, {'trigger_type':3, 'time_window':[1624549894.44,1624550506.50]}],\
                                    'run5198':[{'trigger_type':2, 'time_window':[1624552409.76406,1624552409.77920]}],\
                                    'run5196':[{'trigger_type':2, 'time_window':[1624551426.80823,1624551426.85754]}],\
                                    }
known_pulser_cuts_for_template = {}
#'run5179':[{'trigger_type':2, 'time_window':[1624492539.0,1624492541.0]}],\
#'run5198':[{'trigger_type':2, 'time_window':[1624552128.18,1624552156.68],'time_window_ns':[2.47e7,2.52e7]}],\                            

# 'run5198':[{'trigger_type':2, 'time_window':[1624552231,1624552232.3969],'time_window_ns':[0.6e8,2.20e8]}] # 60 Hz
# 'run5196':[{'trigger_type':2, 'time_window':[1624551082.933,1624551089.886],'time_window_ns':[3.9e8,4.20e8]}],\ #60 Hz events                                    
# 'run5190':[{'trigger_type':2, 'time_window':[1624546184.6,1624546293.4],'time_window_ns':[7.3e8,7.45e8]}],\ # All just 60 hz
# 'run5185':{'trigger_type':2, 'time_window':[1624498820.7,1624499232.7],'time_window_ns':[9.9999e8,1e9]},\
# 'run5185':{'trigger_type':2, 'time_window':[1624502721.86,1624502758.76],'time_window_ns':[5.2847e8,5.4067e8]},\


class Selector(object):
    def __init__(self,ax,scatter_data, eventids, run):
        self.ax = ax
        self.xys = scatter_data.get_offsets()
        self.eventids = eventids
        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.run = run
        self.reader = Reader(os.environ['BEACON_DATA'],run)
        self.cor = Correlator(self.reader,upsample=len(self.reader.t())*8, n_phi=180*4 + 1, n_theta=360*4 + 1)
    def onselect(self,verts):
        #print(verts)
        path = Path(verts)
        ind = numpy.nonzero(path.contains_points(self.xys))[0]

        if type(self.eventids) == numpy.ma.core.MaskedArray:
            print('Only printing non-masked events:')
            eventids = self.eventids[ind][~self.eventids.mask[[ind]]]
            pprint(numpy.asarray(eventids))
        else:
            eventids = self.eventids[ind]
            pprint(eventids)
        if len(eventids) == 1:
            eventid = eventids[0]
            fig = plt.figure()
            plt.title('Run %i, Eventid %i'%(self.run, eventid))
            self.reader.setEntry(eventid)
            ax = plt.subplot(2,1,1)
            for channel in [0,2,4,6]:
                plt.plot(self.reader.t(),self.reader.wf(channel))
            plt.ylabel('HPol Adu')
            plt.xlabel('t (ns)')
            plt.subplot(2,1,2,sharex=ax)
            plt.xlabel('VPol Adu')
            plt.ylabel('t (ns)')
            for channel in [1,3,5,7]:
                plt.plot(self.reader.t(),self.reader.wf(channel))
            self.cor.map(eventid,'both',interactive=True)

if __name__ == '__main__':
    plt.close('all')
    time_delays_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    impulsivity_dset_key = time_delays_dset_key
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_65536-maxmethod_0-sinesubtract_1-deploy_calibration_30-scope_allsky'


    #### Initial Search ####
    '''
    This run should be the easiest to find a signal, though it is not the first run.  I may use this to get a template
    and then use that template to get everything else.
    '''
    site_2_runs = numpy.arange(5181,5186)
    limit_events = 40000
    mask_events = True #Will mask events not meeting some minimum z value

    if len(sys.argv) == 2:
        runs = [int(sys.argv[1])]
    else:
        runs = [5159]
    

    #ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, trigger_types=[1,2,3],included_antennas=[0,1,2,3,4,5,6,7])

    lassos = []

    for run in runs:
        try:
            run_key = 'run%i'%run
            print(run_key)
            reader = Reader(os.environ['BEACON_DATA'],run)
            tct = TemplateCompareTool(reader,apply_phase_response=True)#waveform_index_range=[0,2100]

            #Sloppy work around to run code if file isn't loading (i.e. if scratch files aren't available)
            print('Running script in mode not requiring access to file.')
            trigger_type = loadTriggerTypes(reader)
            eventids = numpy.arange(len(trigger_type))
            #event_limit_cut = eventids < limit_events
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

            for cut_trigger in [1,2,3]:
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
                plt.ylabel(ylabel)

                second_time = numpy.floor(calibrated_trig_time)
                subsecond_time = calibrated_trig_time - second_time

                #period_factor = 1.0#1/20.0#1.0#1/60.0 #1.0 #In seconds, what the remainder will be taken of.
                if cut_trigger == 2:
                    pfs = [1.0,1/20.0,1/60.0,1/100.0,1/120.0]
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

