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
from tools.correlator import Correlator

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
    eventids = [45059]#[499,45059,58875]
    print('Run %i'%run)
    final_corr_length = 2**15 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = 100 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = 8
    crit_freq_high_pass_MHz = None#65
    high_pass_filter_order = None#6
    plot_filters = False

    enable_plots = True

    apply_phase_response=True
    n_phi = 720
    n_theta = 720

    max_method = 0
    hilbert = False
    
    for val in [False,True]:

        reader = Reader(datapath,run)
        cor = Correlator(reader,  upsample=final_corr_length, n_phi=n_phi, n_theta=n_theta, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filters,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True)
        prep = FFTPrepper(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=(None,None), plot_filters=plot_filters)
        if val:
            cor.prep.addSineSubtract(0.03, 0.090, 0.05, max_failed_iterations=3, verbose=False, plot=False)#Test purposes
            prep.addSineSubtract(0.03, 0.090, 0.05, max_failed_iterations=3, verbose=False, plot=False)#Test purposes
        for event_index, eventid in enumerate(eventids):

            prep.setEntry(eventid)
            t_ns = prep.t()
            print(eventid)
            

            map_values, _fig, ax = cor.map(eventid, 'hpol',center_dir='E', plot_map=True, plot_corr=False, hilbert=hilbert, interactive=True, max_method=max_method,mollweide=True,circle_zenith=None,circle_az=None)
            
            if enable_plots:
                fig = plt.figure()
                fig.canvas.set_window_title('wfs r%i-e%i'%(run,eventid))
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
                
            for channel in [0,1,2,3,4,5,6,7]:
                channel=int(channel)
                wf, ss_freqs, n_fits = prep.wf(channel,apply_filter=True,hilbert=hilbert,tukey=None,sine_subtract=True, return_sine_subtract_info=True)
                freqs, spec_dbish, spec = prep.rfftWrapper(t_ns, wf)
                
                #Without sine_subtract to plot what the old signal looked like.
                raw_wf = prep.wf(channel,apply_filter=True,hilbert=hilbert,tukey=None,sine_subtract=False, return_sine_subtract_info=False)
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
                    plt.legend(loc = 'upper right', fontsize=10)
                    plt.xlim(10,150)
                    plt.ylim(-10,30)
                
