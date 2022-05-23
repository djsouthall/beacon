#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import numpy
import matplotlib.pyplot as plt
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader
import inspect
import matplotlib.ticker as ticker
plt.ion()


# plt.style.use('/Users/decoene/Documents/MyLibs_python/l3.mplstyle')
##############################################
##Color mapping
beams = {
                0 : {"theta": 90, "phi": -50},
                1: {"theta": 90, "phi": -25},
                2: {"theta": 90, "phi": 0},
                3: {"theta": 90, "phi": 25},
                4: {"theta": 90, "phi": 50},
                5: {"theta": 30, "phi": -20},
                6: {"theta": 43, "phi": -20},
                7: {"theta": 57, "phi": -20},
                8: {"theta": 70, "phi": -20},
                9: {"theta": 30, "phi": 20},
                10: {"theta": 43, "phi": 20},
                11: {"theta": 57, "phi": 20},
                12: {"theta": 70, "phi": 20},
                13: {"theta": 80, "phi": 0},
                14: {"theta": 70, "phi": 0},
                15: {"theta": 60, "phi": 0},
                16: {"theta": 50, "phi": 0},
                17: {"theta": 40, "phi": 0},
                18: {"theta": 30, "phi": 0},
                19: {"theta": 20, "phi": 0}
                }


thetas = numpy.array([])
phis = numpy.array([])
for i in range(20):
    if i < 5:
        continue
    thetas = numpy.append(thetas, beams[i]['theta'])
    phis = numpy.append(phis, beams[i]['phi'])
sort_indices = numpy.lexsort((phis,90-thetas), axis=0)

from matplotlib import cm
colors = numpy.zeros((20,4))
colors[0:5]               = numpy.asarray([cm.YlOrRd(x) for x in numpy.linspace(0.3, 0.9, 5)])
colors[5+sort_indices]    = numpy.asarray([cm.GnBu(x) for x in numpy.linspace(0.2, 0.9, 15)])[::-1]
##############################################
##############################################
##Function from Dan
def getTimes(reader):
    '''
    This pulls timing information for each event from the reader object..

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        This is the reader for the selected run.

    Returns
    -------
    raw_approx_trigger_time : numpy.ndarray of floats
        The raw_approx_trigger_time values for each event from the Tree.  This does not include subsecond timing, which comes from raw_approx_trigger_time_nsecs
    raw_approx_trigger_time_nsecs : numpy.ndarray of floats
        The raw_approx_trigger_time_nsecs values for each event from the Tree. These are the subsecond timings for events.
    trig_time : numpy.ndarray of floats
        The trig_time values for each event from the Tree.
    '''
    N = reader.head_tree.Draw("raw_approx_trigger_time_nsecs:raw_approx_trigger_time:trig_time:Entry$","","goff")
    #ROOT.gSystem.ProcessEvents()
    raw_approx_trigger_time_nsecs = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N)
    raw_approx_trigger_time = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N)
    trig_time = numpy.frombuffer(reader.head_tree.GetV3(), numpy.dtype('float64'), N)
    eventids = numpy.frombuffer(reader.head_tree.GetV4(), numpy.dtype('float64'), N).astype(int)

    return raw_approx_trigger_time, raw_approx_trigger_time_nsecs, trig_time, eventids
##############################################

##load data
data_path = os.environ['BEACON_DATA']
run_num = 5950
plot_cut = 100 #skip few seconds at the beginning of the run for plot cosmetic

##read run
reader = Reader(data_path,run_num)

##get "header" info for the run
thresholds = reader.returnTriggerThresholds()
beam_scalers, trigger_thresholds, readout_time = reader.returnBeamScalers()
triggered_beams, beams_power = reader.returnTriggerInfo()
time_run = (readout_time - min(readout_time))/3600. #in hours

##get forced trigger events
raw_approx_trigger_time, raw_approx_trigger_time_nsecs, trig_time, eventids = getTimes(reader) #function provided by Dan
trigger_time_s = raw_approx_trigger_time + raw_approx_trigger_time_nsecs/1e9
trig_time_hour = (trigger_time_s-min(trigger_time_s))/3600. #in hours

print('Length of run is %0.2f hours'%max(trig_time_hour))

##loop over events to retrieve the waveforms
trace_rms, trace_noise = [], []
for id in eventids:
    reader.setEntry(id)
    waveforms = numpy.array([reader.wf(j) for j in range(8)])
    times = reader.t()

    ##compute RMS for each channels
    channels_rms = numpy.array([numpy.sqrt(numpy.mean(waveforms[0]**2)), numpy.sqrt(numpy.mean(waveforms[4]**2)), numpy.sqrt(numpy.mean(waveforms[6]**2))]) #Hpol only
    ##compute RMS of all channels
    trace_rms.append(numpy.sqrt(numpy.mean(channels_rms**2)))
    ##get the corresponding noise for this event
    trace_noise.append(64.*trace_rms[-1])

    ##some check plots
    # fx, xx = plt.subplots()
    # xx.set_xlabel("time (ns)")
    # xx.set_ylabel("Waveform (Hpols)")
    # xx.plot(times, waveforms[0], label='NE Hpol', color='C0', linewidth=3)
    # xx.plot(times, waveforms[2], label='NW Hpol ', color='C1', linewidth=3)
    # xx.plot(times, waveforms[4], label='SW Hpol ', color='C2', linewidth=3)
    # xx.axhline(y=trace_rms[-1], label='RMS', color='r')
    # xx.axhline(y=trace_noise[-1], label='64xE(N**2)', color='k')
    # xx.legend()
    # plt.show()

##array handling stuff
trace_rms = numpy.array(trace_rms)
trace_noise = numpy.array(trace_noise)

## rebin the noise to each readout time for the beams
trace_rms_binned, trace_noise_binned = [], []
for i in range(len(time_run)-1):
    sel = numpy.where((trig_time_hour<=time_run[i])&(trig_time_hour<time_run[i+1]))[0]
    trace_rms_binned.append(numpy.mean(trace_rms[sel]))
    trace_noise_binned.append(numpy.mean(trace_noise[sel]))


def makeBeamPlot(fig, ax, major_fontsize=24, minor_fontsize=20, mode='a', figsize=(16,9), suppress_legend=True, _colors=None):
    try:
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        if _colors is None:
            _colors = colors

        times = time_run[plot_cut:-1] - min(time_run[plot_cut:-1])

        if mode == 'a':
            ax.set_xlabel("Noise (all events)", fontsize=major_fontsize)
            # ax.set_title(r"$\rm run: $"+" "+str(run_num))
            ax.hist(trace_rms, label='RMS')
            ax.hist(trace_noise, label='64xE(N**2)', alpha=0.8)
            if suppress_legend == False:
                ax.legend(fontsize=minor_fontsize)
        elif mode == 'b':
            ax.set_xlabel("run time (h)", fontsize=major_fontsize)
            ax.set_ylabel("Noise (binned events)", fontsize=major_fontsize)
            # ax.set_title(r"$\rm run: $"+" "+str(run_num))
            ax.plot(times, trace_rms_binned[plot_cut:], label='RMS')
            ax.plot(times, trace_noise_binned[plot_cut:], label='64xE(N**2)')
            if suppress_legend == False:
                ax.legend(fontsize=minor_fontsize)
        elif mode == 'c':
            ax.set_xlabel("Time (h)", fontsize=major_fontsize)
            ax.set_ylabel("Beam Power Threshold", fontsize=major_fontsize)#"Beam Power Threshold (SNR)"
            # ax.set_title(r"$\rm run: $"+" "+str(run_num))
            ##below horizon beams
            for i in range(0,5):
                ax.plot(times, thresholds[i,plot_cut:-1]/trace_noise_binned[plot_cut:], label='beam %i'%(i), linewidth=4, color=_colors[i])
            ##above horizon beams
            for i in range(5,20):
                ax.plot(times, thresholds[i,plot_cut:-1]/trace_noise_binned[plot_cut:], label='beam %i'%(i), linewidth=2, linestyle='--', color=_colors[i])
            if suppress_legend == False:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_xlim(min(times), max(times))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(r'%i $\sigma$'))

            ##save figures to pdf
            # fb.savefig("noise_evol.pdf", format='pdf')
            # fc.savefig("SNR_thesholds.pdf", format='pdf')
        return fig, ax
    except Exception as e:  
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)




if __name__ == '__main__':
    for mode in ['a', 'b', 'c']:
        fig, ax = makeBeamPlot(None, None, major_fontsize=24, minor_fontsize=20, mode=mode, figsize=(16,9))

