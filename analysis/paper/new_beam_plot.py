#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import numpy
import matplotlib.pyplot as plt
#from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from beaconroot.examples.beacon_data_reader import Reader
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

##Conversion from SNR Power to SNR Voltage from Andrew's work
def get_voltage_snr(power_snr):
    return 1.806*numpy.sqrt(power_snr)-0.383
def get_power_snr(voltage_snr):
    return 0.333*voltage_snr**2 -0.217*voltage_snr+1.33
##############################################

##load data
data_path = os.environ['BEACON_DATA']
run_num = 5796#5792#5764#5752#5950#5874
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

    noise_trace = 200 #only select the begginning of the traces to avoid signal
    ##compute RMS for each channels
    channels_rms = numpy.sqrt(numpy.mean(waveforms[:,:noise_trace]**2, axis=1))
    ##compute RMS of all channels
    trace_rms.append(numpy.sqrt(numpy.mean(channels_rms**2)))
    ##get the corresponding noise for this event
    trace_noise.append(64.*trace_rms[-1]**2)

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

## convert power to voltages
Psnr = thresholds[:,plot_cut:-1]/trace_noise_binned[plot_cut:]
Vsnr = get_voltage_snr(Psnr)
##tocheck correct ratio between Psnr and Vsnr
# fx, xx = plt.subplots()
# xx.scatter(Psnr, Vsnr)

def makeBeamPlot(fig, ax, major_fontsize=24, minor_fontsize=20, mode='a', figsize=(16,9), suppress_legend=True, _colors=None, primary='power'):
    try:
        
        if _colors is None:
            _colors = colors

        # times = time_run[plot_cut:-1] - min(time_run[plot_cut:-1])

        if mode == 'a':
            if fig is None or ax is None:
                fa, (a1x, a2x) = plt.subplots(1,2, figsize=figsize, constrained_layout=True)
            else:
                fa = fig
                (a1x, a2x) = ax
            a1x.set_ylabel("RMS noise (all events)", fontsize=major_fontsize)
            fa.suptitle(r"$\rm Run $"+" "+str(run_num)+", September 09 2021 (18h-22h UTC)")
            a1x.hist(trace_rms[trace_rms<15], bins=30, label='RMS')
            a1x.set_xlim(0,15)
            a1x.axvline(x=3.5, label='3.5', color='r')
            a2x.hist(trace_noise[trace_noise<10000], bins=30, label='64xE(N**2)')
            a2x.set_xlim(0,10000)
            a2x.axvline(x=800, label='800', color='r')
            a1x.legend(fontsize=minor_fontsize)
            a2x.legend(fontsize=minor_fontsize)

        elif mode == 'b':
            if fig is None or ax is None:
                fb, bx = plt.subplots(figsize=figsize)
            else:
                fb = fig
                bx = ax
            bx.set_xlabel("run time (h)", fontsize=major_fontsize)
            bx.set_ylabel("RMS noise (binned events)", fontsize=major_fontsize)
            bx.set_title(r"$\rm Run $"+" "+str(run_num)+", September 09 2021 (18h-22h UTC)")
            bx.plot(time_run[plot_cut:-1], trace_rms_binned[plot_cut:], label='RMS')
            bx.plot(time_run[plot_cut:-1], trace_noise_binned[plot_cut:], label='64xE(N**2)')
            bx.legend(fontsize=minor_fontsize)

        elif mode == 'c':
            if fig is None or ax is None:
                fc, cx = plt.subplots(figsize=figsize)
            else:
                fc = fig
                cx = ax
            cx.set_xlabel("Time (h)\nSeptember 09 2021 (11.5h-14.5h PDT)", fontsize=major_fontsize)
            
            if primary == 'power':
                cx.set_ylabel("Beam Power Threshold", fontsize=major_fontsize)#"Beam Power Threshold (SNR)"
                
                secax_y = cx.secondary_yaxis('right', functions=(get_voltage_snr, get_power_snr))
                secax_y.set_ylabel("Beam Voltage Threshold", fontsize=major_fontsize)
                #secax_y.set_ylabel(r'$\rm beam\ voltage\ thresholds\ (\sigma)$', fontsize=major_fontsize)
                
                ##below horizon beams
                for i in range(0,5):
                    cx.plot(time_run[plot_cut:-1], Psnr[i], label='beam %i'%(i), linewidth=4, color=colors[i])
                ##above horizon beams
                for i in range(5,20):
                    cx.plot(time_run[plot_cut:-1], Psnr[i], label='beam %i'%(i), linewidth=2, linestyle='--', color=colors[i])
            elif primary == 'voltage':
                cx.set_ylabel("Beam Voltage Threshold", fontsize=major_fontsize)#"Beam Power Threshold (SNR)"
                
                secax_y = cx.secondary_yaxis('right', functions=(get_power_snr, get_voltage_snr))
                secax_y.set_ylabel("Beam Power Threshold", fontsize=major_fontsize)
                #secax_y.set_ylabel(r'$\rm beam\ voltage\ thresholds\ (\sigma)$', fontsize=major_fontsize)
                
                for i in range(20):
                    print('Beam %i'%i)
                    print(min(Vsnr[i]))
                    print(numpy.mean(Vsnr[i]))
                    print(max(Vsnr[i]))

                ##below horizon beams
                for i in range(0,5):
                    cx.plot(time_run[plot_cut:-1], Vsnr[i], label='beam %i'%(i), linewidth=4, color=colors[i])
                ##above horizon beams
                for i in range(5,20):
                    cx.plot(time_run[plot_cut:-1], Vsnr[i], label='beam %i'%(i), linewidth=2, linestyle='--', color=colors[i])
            if suppress_legend == False:
                cx.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

            cx.set_xlim(0,3)
            cx.yaxis.set_major_formatter(ticker.FormatStrFormatter(r'%0.1f $\sigma$'))#_\mathrm{P}
            secax_y.yaxis.set_major_formatter(ticker.FormatStrFormatter(r'%0.1f $\sigma$'))#_\mathrm{V}

            cx.xaxis.set_tick_params(labelsize=minor_fontsize)
            cx.yaxis.set_tick_params(labelsize=minor_fontsize)
            secax_y.yaxis.set_tick_params(labelsize=minor_fontsize)

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

