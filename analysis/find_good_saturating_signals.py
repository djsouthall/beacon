'''
This script is to be used to find events within some pre-found pulser events that pass some
specified filters.  These events can then be printed out and used as input for other scripts.
'''

import numpy
import scipy.spatial
import scipy.signal
import os
import sys
import csv

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
plt.ion()

#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.
known_pulser_ids = info.loadPulserEventids()
ignorable_pulser_ids = info.loadPulserIgnorableEventids()

def rfftWrapper(waveform_times, *args, **kwargs):
    spec = numpy.fft.rfft(*args, **kwargs)
    real_power_multiplier = 2.0*numpy.ones_like(spec) #The factor of 2 because rfft lost half of the power except for dc and Nyquist bins (handled below).
    if len(numpy.shape(spec)) != 1:
        real_power_multiplier[:,[0,-1]] = 1.0
    else:
        real_power_multiplier[[0,-1]] = 1.0
    spec_dbish = 10.0*numpy.log10( real_power_multiplier*spec * numpy.conj(spec) / len(waveform_times)) #10 because doing power in log.  Dividing by N to match monutau. 
    freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
    return freqs, spec_dbish

def makeFilter(waveform_times,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order, plot_filter=False):
    dt = waveform_times[1] - waveform_times[0]
    freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
    b, a = scipy.signal.butter(filter_order, crit_freq_low_pass_MHz*1e6, 'low', analog=True)
    d, c = scipy.signal.butter(filter_order, crit_freq_high_pass_MHz*1e6, 'high', analog=True)

    filter_x_low_pass, filter_y_low_pass = scipy.signal.freqs(b, a,worN=freqs)
    filter_x_high_pass, filter_y_high_pass = scipy.signal.freqs(d, c,worN=freqs)
    filter_x = freqs
    filter_y = numpy.multiply(filter_y_low_pass,filter_y_high_pass)
    return filter_y, freqs

def loadSignals(reader,eventid,filter_y):
    try:
        reader.setEntry(eventid)
        raw = numpy.zeros((8,reader.header().buffer_length))
        filter_y = numpy.tile(filter_y,(8,1))
        for channel in range(8):
            #Load waveform
            raw[channel] = reader.wf(channel)
        #Upsample
        upsampled = scipy.signal.resample(raw,2*(numpy.shape(filter_y)[1]-1),axis=1)
        #Apply filter
        upsampled = numpy.fft.irfft(numpy.multiply(filter_y,numpy.fft.rfft(upsampled,axis=1)),axis=1)
    except Exception as e:
        print('Error in loadSignals')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return raw, upsampled


if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine
    use_known_ids = True
    
    resample_factor = 1

    #Filter settings
    crit_freq_low_pass_MHz = 75
    crit_freq_high_pass_MHz = 35
    filter_order = 6
    plot_filter = True
    power_sum_cut_location = 50 #index
    power_sum_cut_value = 13000 #Events with larger power sum then this are ignored. 
    peak_cut = 60 #At least one channel has to have a signal cross this thresh. 

    for run_index, run in enumerate(runs):
        eventids = known_pulser_ids['run%i'%run]
        reader = Reader(datapath,run)
        waveform_times = reader.t()
        waveforms_upsampled = {}
        waveforms_raw = {}

        #Prepare filter
        reader.setEntry(98958)
        wf = reader.wf(0)
        wf , waveform_times = scipy.signal.resample(wf,len(wf)*resample_factor,t=reader.t())
        dt = waveform_times[1] - waveform_times[0]
        filter_y,freqs = makeFilter(waveform_times,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order, plot_filter=plot_filter)

        try:
            for channel in range(8):
                waveforms_upsampled['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length*resample_factor))
                waveforms_raw['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length))

            print('Loading Waveforms_upsampled')
            plt.figure()
            ax = plt.subplot(4,2,1)
            good_events = []
            for waveform_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\r'%(waveform_index,len(eventids)))
                sys.stdout.flush()
                reader.setEntry(eventid)

                raw_wfs, upsampled_wfs = loadSignals(reader,eventid,filter_y)

                good_events.append(~numpy.any(numpy.cumsum(raw_wfs**2,axis=1)[:,power_sum_cut_location] > power_sum_cut_value) and numpy.any(raw_wfs > peak_cut))

                for channel in range(8):
                    waveforms_upsampled['ch%i'%channel][waveform_index] = upsampled_wfs[channel]
                    waveforms_raw['ch%i'%channel][waveform_index] = raw_wfs[channel]
                    if good_events[-1]:
                        plt.subplot(4,2,channel+1)
                        plt.plot(waveform_times,upsampled_wfs[channel])
            eventids = eventids[good_events]
            for channel in range(8):
                waveforms_upsampled['ch%i'%channel] = waveforms_upsampled['ch%i'%channel][good_events]
                waveforms_raw['ch%i'%channel] = waveforms_raw['ch%i'%channel][good_events]

            print('The chosen events are:')
            print(eventids)

        except Exception as e:
            print('Error in main loop.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)






