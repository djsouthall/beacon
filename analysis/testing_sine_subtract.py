'''
This is a script that I am using to test the development and implementation of the sine_subtraction method.
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
warnings.filterwarnings("ignore")
plt.ion()

datapath = os.environ['BEACON_DATA']

if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    run = 1650
    eventids = [499,45059,58875]
    print('Run %i'%run)
    final_corr_length = 2**12 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = None#80 #This new pulser seems to peak in the region of 85 MHz or so
    crit_freq_high_pass_MHz = None#65
    low_pass_filter_order = None#3
    high_pass_filter_order = None#6
    plot_filters = False

    enable_plots = True
    
    for val in [False,True]:

        reader = Reader(datapath,run)
        prep = FFTPrepper(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=(None,None), plot_filters=plot_filters)
        if val:
            prep.addSineSubtract(0.03, 0.090, 0.05, max_failed_iterations=3, verbose=False, plot=False)#Test purposes
        for event_index, eventid in enumerate(eventids):
            if enable_plots:
                plt.figure()
                plt.title('Run %i, eventid %i'%(run,eventid))
                plt.subplot(2,1,1)
                plt.ylabel('adu')
                plt.xlabel('ns')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.subplot(2,1,2)
                plt.ylabel('dBish')
                plt.xlabel('freq')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                

            prep.setEntry(eventid)
            t_ns = prep.t()
            print(eventid)
            for channel in [0,1,2,3,4,5,6,7]:
                channel=int(channel)
                wf, ss_freqs, n_fits = prep.wf(channel,apply_filter=False,hilbert=False,tukey=None,sine_subtract=True, return_sine_subtract_info=True)
                freqs, spec_dbish, spec = prep.rfftWrapper(t_ns, wf)
                
                #Without sine_subtract to plot what the old signal looked like.
                raw_wf = prep.wf(channel,apply_filter=False,hilbert=False,tukey=None,sine_subtract=False, return_sine_subtract_info=True)
                raw_freqs, raw_spec_dbish, raw_spec = prep.rfftWrapper(t_ns, raw_wf)

                peak_freqs = numpy.array([])
                peak_db = numpy.array([])

                for ss_n in range(len(n_fits)):
                    unique_peak_indices = numpy.unique(numpy.argmin(numpy.abs(numpy.tile(raw_freqs,(n_fits[ss_n],1)).T - 1e9*ss_freqs[0]),axis=0)) #Gets indices of freq of peaks in non-upsampled spectrum.
                    unique_peak_freqs = raw_freqs[unique_peak_indices]
                    unique_peak_linear = raw_spec[unique_peak_indices]
                    unique_peak_db = raw_spec_dbish[unique_peak_indices].astype(numpy.double)


                    peak_freqs = numpy.append(peak_freqs,unique_peak_freqs)
                    peak_db = numpy.append(peak_db,unique_peak_db) #divided by 2 when plotting

                print(n_fits)
                print(ss_freqs)

                if enable_plots:
                    plt.subplot(2,1,1)
                    plt.plot(t_ns,wf)

                    plt.subplot(2,1,2)
                    plt.plot(freqs/1e6,spec_dbish/2.0,label='Ch %i'%channel)
                    # if len(peak_freqs) > 0:
                    #     plt.plot(raw_freqs/1e6,raw_spec_dbish/2.0,alpha=0.5,linestyle='--',label='Ch %i'%channel)
                    if len(peak_freqs) > 0:
                        plt.scatter(peak_freqs/1e6,peak_db/2.0,label='Ch %i Removed Peak Max'%channel)
                    plt.legend(loc = 'upper right')
                    plt.xlim(10,110)
                    plt.ylim(-10,30)
                
