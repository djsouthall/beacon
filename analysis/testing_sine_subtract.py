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
            for channel in [0]:#[0,1,2,3,4,5,6,7]:
                channel=int(channel)
                wf, ss_freqs, n_fits = prep.wf(channel,apply_filter=False,hilbert=False,tukey=None,sine_subtract=True, return_sine_subtract_info=True)
                freqs, spec_dbish, spec = prep.rfftWrapper(t_ns, wf)
                print(n_fits)
                print(ss_freqs)

                if enable_plots:
                    plt.subplot(2,1,1)
                    plt.plot(t_ns,wf)

                    plt.subplot(2,1,2)
                    plt.plot(freqs/1e6,spec_dbish/2.0,label='Ch %i'%channel)
                    plt.legend(loc = 'upper right')
                    plt.xlim(10,110)
                    plt.ylim(-10,30)
                
