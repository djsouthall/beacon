import numpy
import scipy.spatial
import os
import sys

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
plt.ion()


if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([792])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine
    nearest_neighbor = 10 #Adjust until works.
    scale_subtimes = 10.0 #The larger this is the the less the nearest neighbor favors vertical lines.
    scale_times = 1.0  #The larger this is the the less the nearest neighbor favors horizontal lines.
    slope_bound = 1.0e-9
    percent_cut = 0.001
    nominal_clock_rate = 31.25e6
    lower_rate_bound = 31.2e6 #Don't make the bounds too large or the bisection method will overshoot and roll over.
    upper_rate_bound = 31.3e6 #Don't make the bounds too large or the bisection method will overshoot and roll over.
    plot = True
    verbose = False

    for run_index, run in enumerate(runs):
        reader = Reader(datapath,run)
        waveform_times = reader.t()
        waveforms = {}
        
        try:
            clock_rate, times, subtimes, trig_times, eventids = cc.getClockCorrection(reader, nearest_neighbor=nearest_neighbor, scale_subtimes=scale_subtimes, scale_times=scale_times, slope_bound=slope_bound, percent_cut=percent_cut, nominal_clock_rate=nominal_clock_rate, lower_rate_bound=lower_rate_bound, upper_rate_bound=upper_rate_bound, plot=plot, verbose=verbose)
            adjusted_trig_times = trig_times%clock_rate

            for channel in range(8):
                waveforms['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length),dtype=int)

            for waveform_index, eventid in enumerate(eventids):
                reader.setEntry(eventid)
                event_times = reader.t()
                for channel in range(8): 
                    waveforms['ch%i'%channel][waveform_index] = reader.wf(channel)

            for channel in range(8):
                plt.figure()
                all_wv = []
                all_times = []
                for waveform in waveforms['ch%i'%channel]: 
                    all_wv.append(waveform)
                    all_times.append(waveform_times)
                all_wv = numpy.array(all_wv).flatten()
                all_times = numpy.array(all_times).flatten()
                plt.hist2d(all_times,all_wv,bins=[reader.header().buffer_length,128],norm=LogNorm()) 
                plt.title('ch%i'%channel)
        except Exception as e:
            print('Error in main loop.')
            print(e)






