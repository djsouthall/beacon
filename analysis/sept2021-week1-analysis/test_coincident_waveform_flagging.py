#!/usr/bin/env python3
'''
The purpose of this script is to look at the results from various maps and find a way to determine how to choose the
best map for reconstruction.
'''

import sys
import os
import inspect
import h5py
import copy
from pprint import pprint

import numpy
import scipy
import scipy.signal

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TimeDelayCalculator

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)



if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'

    final_corr_length = 2**17
    waveform_index_range = (None,None)
    plot_filters = False

    datapath = os.environ['BEACON_DATA']
    align_method_13_n = 2

    crit_freq_low_pass_MHz = 80
    low_pass_filter_order = 14

    crit_freq_high_pass_MHz = 20
    high_pass_filter_order = 4

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.00
    sine_subtract_max_freq_GHz = 0.25
    sine_subtract_percent = 0.03

    apply_phase_response = True

    shorten_signals = False
    shorten_thresh = 0.7
    shorten_delay = 10.0
    shorten_length = 90.0

    run = 5740
    reader = Reader(datapath,run)
    tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=plot_filters,apply_phase_response=apply_phase_response)
    f = lambda x : (x-x.min())/(x.max()-x.min())

    if True:
        channels = [1,3,5,7]
        mode = 'same'
        normalization = numpy.correlate(numpy.ones_like(tdc.t()),numpy.ones_like(tdc.t()),mode=mode)*len(channels)

        for eventid in range(40000,50000):
            tdc.setEntry(eventid)
            for channel_index, channel in enumerate(channels):
                wf_hilb = tdc.wf(channel, apply_filter=True, hilbert=True , tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)
                _hcorr = numpy.correlate(wf_hilb,wf_hilb,mode=mode)/normalization
                _hcorr_env = numpy.abs(scipy.signal.hilbert(_hcorr))

                if channel_index == 0:
                    hcorr = _hcorr
                    hcorr_env = _hcorr_env
                else:
                    hcorr += _hcorr
                    hcorr_env += _hcorr_env

            peaks, props = scipy.signal.find_peaks(f(hcorr_env), rel_height=0.5, height=0.3, prominence=0.15, distance = int(75.0/tdc.t()[1]) , width = int(75.0/tdc.t()[1]))

            if len(peaks[peaks > len(tdc.t())/2.0 - 1]) > 1:
                print(props)
                fig = plt.figure()
                plt.suptitle('r%ie%i'%(run,eventid))
                plt.subplot(2,1,1)

                for channel_index, channel in enumerate(channels):
                    wf      = tdc.wf(channel, apply_filter=True, hilbert=False, tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)
                    wf_hilb = tdc.wf(channel, apply_filter=True, hilbert=True , tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)

                    plt.plot(tdc.t(),wf)

                    _corr = numpy.correlate(wf,wf,mode=mode)/normalization
                    _corr_env = numpy.abs(scipy.signal.hilbert(_corr))

                    _hcorr = numpy.correlate(wf_hilb,wf_hilb,mode=mode)/normalization
                    _hcorr_env = numpy.abs(scipy.signal.hilbert(_hcorr))

                    if channel_index == 0:
                        corr = _corr
                        hcorr = _hcorr
                        corr_env = _corr_env
                        hcorr_env = _hcorr_env
                    else:
                        corr += _corr
                        hcorr += _hcorr
                        corr_env += _corr_env
                        hcorr_env += _hcorr_env


                plt.subplot(2,1,2)
                plt.plot(tdc.t(),f(corr_env)    ,label='mean(Corr(wf,wf)_i)')
                plt.plot(tdc.t(),f(hcorr)       ,label='mean(Corr(hilb(wf),hilb(wf))_i)')
                plt.plot(tdc.t(),f(hcorr_env)   ,label='hilb(mean(Corr(hilb(wf),hilb(wf))_i))')
                plt.legend()

                # peaks, props = scipy.signal.find_peaks(f(hcorr_env), height=0.3, prominence=0.15, distance = int(75.0/tdc.t()[1]) , width = int(20.0/tdc.t()[1]))

                for peak_index, peak in enumerate(peaks):
                    plt.scatter(peak*tdc.t()[1], props['peak_heights'][peak_index], c='r')

                import pdb; pdb.set_trace()
                plt.close(fig)


    else:
        for eventid in [ 4053,  6438, 16956, 45475]:
            tdc.setEntry(eventid)

            for channels in [[0,2,4,6],[1,3,5,7]]:
                plt.figure()
                plt.subplot(2,1,1)

                for channel_index, channel in enumerate(channels):
                    wf      = tdc.wf(channel, apply_filter=True, hilbert=False, tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)
                    wf_hilb = tdc.wf(channel, apply_filter=True, hilbert=True , tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)

                    plt.plot(tdc.t(),wf)

                    mode = 'same'
                    normalization = numpy.correlate(numpy.ones_like(wf),numpy.ones_like(wf),mode=mode)*len(channels)

                    _corr = numpy.correlate(wf,wf,mode=mode)/normalization
                    _corr_env = numpy.abs(scipy.signal.hilbert(_corr))

                    _hcorr = numpy.correlate(wf_hilb,wf_hilb,mode=mode)/normalization
                    _hcorr_env = numpy.abs(scipy.signal.hilbert(_hcorr))

                    if channel_index == 0:
                        corr = _corr
                        hcorr = _hcorr
                        corr_env = _corr_env
                        hcorr_env = _hcorr_env
                    else:
                        corr += _corr
                        hcorr += _hcorr
                        corr_env += _corr_env
                        hcorr_env += _hcorr_env


                plt.subplot(2,1,2)
                plt.plot(tdc.t(),f(corr_env)    ,label='mean(Corr(wf,wf)_i)')
                plt.plot(tdc.t(),f(hcorr)       ,label='mean(Corr(hilb(wf),hilb(wf))_i)')
                plt.plot(tdc.t(),f(hcorr_env)   ,label='hilb(mean(Corr(hilb(wf),hilb(wf))_i))')
                plt.legend()

                peaks, props = scipy.signal.find_peaks(f(hcorr_env), height=0.3, prominence=0.15, distance = int(75.0/tdc.t()[1]) , width = int(20.0/tdc.t()[1]))

                for peak_index, peak in enumerate(peaks):
                    plt.scatter(peak*tdc.t()[1], props['peak_heights'][peak_index], c='r')

