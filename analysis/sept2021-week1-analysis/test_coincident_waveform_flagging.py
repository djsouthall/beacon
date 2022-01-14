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


class methodTester():
    '''
    This just collects the common info needed to be accessed by all methods tested
    of identifying multiple traces in a given waveform.
    '''
    final_corr_length = 2**17
    waveform_index_range = (None,None)
    plot_filters = False

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

    def __init__(self,  reader):
        self.reader = reader
        self.tdc = TimeDelayCalculator(self.reader, final_corr_length=methodTester.final_corr_length, crit_freq_low_pass_MHz=methodTester.crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=methodTester.crit_freq_high_pass_MHz, low_pass_filter_order=methodTester.low_pass_filter_order, high_pass_filter_order=methodTester.high_pass_filter_order,waveform_index_range=methodTester.waveform_index_range,plot_filters=methodTester.plot_filters,apply_phase_response=methodTester.apply_phase_response)
        self.method1_normalization = numpy.correlate(numpy.ones_like(self.tdc.t()),numpy.ones_like(self.tdc.t()),mode='same')

    def normalizeCorr(self, x):
        return (x-x.min())/(x.max()-x.min())

    def plotWf(self, eventid, mode='all',title=''):
        '''
        Quickly plots the event.
        '''
        if mode == 'all':
            channels = [0,1,2,3,4,5,6,7]
            dims = (2,4)
        elif mode == 'hpol':
            dims = (4,1)
            channels = [0,2,4,6]
        elif mode == 'vpol':
            dims = (4,1)
            channels = [1,3,5,7]
        fig = plt.figure()
        plt.suptitle(title)
        for channel_index, channel in enumerate(channels):
            plt.subplot(dims[0],dims[1],channel_index+1)
            plt.plot(self.tdc.t(),self.tdc.wf(channel, apply_filter=True, hilbert=False , tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False), c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channel])
            plt.ylabel('adu')
            plt.xlabel('ns')
        return fig


    def method1(self, eventid, mode=None, flag_plot=False, force_plot=False, return_fig=False, plot_mode='full'):
        '''
        This method calculates the autocorrelation of hilbert enveloped waveforms, takes the hilbert envelope
        of the autocorrelation, then searches for peaks within this resulting trace.

        Mode selects the channels considered.  None, hpol, or vpol

        flag_plot will plot whenever an event has sufficient peaks flagged if True.

        force_plot will plot regardless of flagged peaks.
        '''

        if mode is None:
            channels = [0,1,2,3,4,5,6,7]
        elif mode == 'hpol':
            channels = [0,2,4,6]
        elif mode == 'vpol':
            channels = [1,3,5,7]

        self.tdc.setEntry(eventid)

        for channel_index, channel in enumerate(channels):
            wf_hilb = self.tdc.wf(channel, apply_filter=True, hilbert=True , tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)
            _hcorr = numpy.correlate(wf_hilb,wf_hilb,mode='same')/(self.method1_normalization*len(channels))
            _hcorr_env = numpy.abs(scipy.signal.hilbert(_hcorr))

            if channel_index == 0:
                hcorr = _hcorr
                hcorr_env = _hcorr_env
            else:
                hcorr += _hcorr
                hcorr_env += _hcorr_env

        #Tuning parameters
        rel_height  = 0.5
        height      = 0.3
        prominence  = 0.2
        distance    = int(50.0/self.tdc.t()[1])
        width       = (int(50.0/mt.tdc.t()[1]) , int(250.0/mt.tdc.t()[1]))

        peaks, props = scipy.signal.find_peaks(self.normalizeCorr(hcorr_env), rel_height=rel_height, height=height, prominence=prominence, distance=distance, width=width)

        if (flag_plot and len(peaks[peaks > len(self.tdc.t())/2.0 - 1]) > 1) or force_plot:
            print('\neventid %i Method 1'%eventid)
            for peak_index, peak in enumerate(peaks):
                print('Peak at %i'%peak)
                for key, item in props.items():
                    print('\t{0:<15} - {1:>10.3f}'.format(key, item[peak_index]))
                    #print(key, item[peak_index])
            fig = plt.figure()
            plt.suptitle('Method 1\nr%ie%i'%(run,eventid),fontsize=18)
            if plot_mode == 'full':
                plt.subplot(2,1,1)

            for channel_index, channel in enumerate(channels):
                wf      = self.tdc.wf(channel, apply_filter=True, hilbert=False, tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)
                wf_hilb = self.tdc.wf(channel, apply_filter=True, hilbert=True , tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)

                if plot_mode == 'full':
                    plt.plot(self.tdc.t(),wf)

                _corr = numpy.correlate(wf,wf,mode='same')/(self.method1_normalization*len(channels))
                _corr_env = numpy.abs(scipy.signal.hilbert(_corr))

                _hcorr = numpy.correlate(wf_hilb,wf_hilb,mode='same')/(self.method1_normalization*len(channels))
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

            if plot_mode == 'full':
                plt.subplot(2,1,2)
                plt.plot(self.tdc.t(),self.normalizeCorr(corr_env)    ,label='mean(Corr(wf,wf)_i)')
                plt.plot(self.tdc.t(),self.normalizeCorr(hcorr)       ,label='mean(Corr(hilb(wf),hilb(wf))_i)')
            plt.plot(self.tdc.t(),self.normalizeCorr(hcorr_env)   ,label='hilb(mean(Corr(hilb(wf),hilb(wf))_i))')
            plt.legend()

            for peak_index, peak in enumerate(peaks):
                plt.scatter(peak*self.tdc.t()[1], props['peak_heights'][peak_index], c='r')

            if return_fig == False:
                import pdb; pdb.set_trace()
                plt.close(fig)
        if return_fig == True and ((flag_plot and len(peaks[peaks > len(self.tdc.t())/2.0 - 1]) > 1) or force_plot):
            return peaks, props, fig
        else:
            return peaks, props
    def method2(self, eventid, mode=None, debug=True, flag_plot=False, force_plot=False, return_fig=False, plot_mode='full'):
        '''
        This method calculates autocorrelation of the waveform in small windowed chunks.

        Mode selects the channels considered.  None, hpol, or vpol

        flag_plot will plot whenever an event has sufficient peaks flagged if True.

        force_plot will plot regardless of flagged peaks.
        '''
        if mode is None:
            channels = [0,1,2,3,4,5,6,7]
        elif mode == 'hpol':
            channels = [0,2,4,6]
        elif mode == 'vpol':
            channels = [1,3,5,7]

        self.tdc.setEntry(eventid)
        t = self.tdc.t()

        chunk_width = int(len(self.tdc.t())/32)
        shift_factor = 8
        chunk_shift = int(chunk_width/shift_factor)
        starts = numpy.arange(0,len(self.tdc.t())-chunk_width,chunk_shift).astype(int)
        stops = (starts + chunk_width).astype(int)
        norm = numpy.correlate(numpy.ones(chunk_width),numpy.ones(chunk_width),mode='same')*len(channels)

        absmean = lambda x : numpy.mean(numpy.abs(x)); absmean.__name__ = 'absmean'

        for hilbert in [False]:
            fs = (numpy.max,)#(absmean, numpy.max, numpy.std)
            for f in fs:

                if debug:
                    fig = plt.figure()
                    plt.suptitle('Method 2\nr%ie%i'%(run,eventid),fontsize=18)

                traces = numpy.zeros((len(channels),len(starts)))
                for channel_index, channel in enumerate(channels):
                    wf = self.tdc.wf(channel, apply_filter=True, hilbert=hilbert , tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)

                    if debug and plot_mode == 'full':
                        plt.subplot(len(channels)+1,1,channel_index + 1)
                    for i, (start, stop) in enumerate(zip(starts,stops)):
                        traces[channel_index][i] = f(numpy.correlate(wf[start:stop],wf[start:stop],mode='same')/norm)
                        if debug and plot_mode == 'full':
                            plt.plot(t[start:stop],wf[start:stop]/max(numpy.abs(wf)) + 2*(i%shift_factor))
                
                    traces[channel_index] = traces[channel_index]/traces[channel_index].max()
                # traces = traces/traces.max()

                #Tuning parameters
                rel_height  = 0.5
                height      = 0.2
                prominence  = 0.15
                distance    = 4
                width       = (3,15)

                n_peaks = numpy.zeros(len(channels))

                if debug:
                    print('\neventid %i Method 2'%eventid)

                for channel_index, channel in enumerate(channels):
                    peaks, props = scipy.signal.find_peaks(traces[channel_index], rel_height=rel_height, height=height, prominence=prominence, distance=distance, width=width)
                    n_peaks[channel_index] = len(peaks)
                    if debug:
                        print('\nchannel %i'%channel)
                        for peak_index, peak in enumerate(peaks):
                            print('  Peak at %i'%peak)
                            for key, item in props.items():
                                print('\t{0:<15} - {1:>10.3f}'.format(key, item[peak_index]))

                        if plot_mode == 'full':
                            plt.subplot(len(channels)+1,1,len(channels)+1)
                        plt.plot(traces[channel_index],label='Ch%i'%channel)
                        for peak_index, peak in enumerate(peaks):
                            plt.scatter(peak, props['peak_heights'][peak_index], c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channel_index])


                if debug:
                    # plt.suptitle('f = ' + f.__name__ + ', hilb = ' + str(hilbert))
                    if plot_mode == 'full':
                        for channel_index, channel in enumerate(channels):
                            plt.subplot(len(channels)+1,1,channel_index + 1)
                            plt.xlabel('Time')
                            plt.ylabel('Ch %i\nStaggered Chunks'%channel)
                        plt.subplot(len(channels)+1,1,len(channels)+1)
                    plt.legend()
                    plt.xlabel('Chunk')
                    plt.ylabel('Max Sub AutoCorr')

        if debug:
            if return_fig == False:
                import pdb; pdb.set_trace()
                plt.close(fig)
                return n_peaks
            else:
                return n_peaks, fig
        else:
            return n_peaks

    def method3(self, eventid, mode=None, debug=True, flag_plot=False, force_plot=False, return_fig=False):
        '''
        This method calculates autocorrelation of the waveform in small windowed chunks.

        Mode selects the channels considered.  None, hpol, or vpol

        flag_plot will plot whenever an event has sufficient peaks flagged if True.

        force_plot will plot regardless of flagged peaks.
        '''
        if mode is None:
            channels = [0,1,2,3,4,5,6,7]
        elif mode == 'hpol':
            channels = [0,2,4,6]
        elif mode == 'vpol':
            channels = [1,3,5,7]

        self.tdc.setEntry(eventid)
        t = self.tdc.t()

        if debug:
            fig = plt.figure()
            plt.suptitle('Method 3\nr%ie%i'%(run,eventid),fontsize=18)

        traces = numpy.zeros((len(channels),len(t)))
        for channel_index, channel in enumerate(channels):
            traces[channel_index] = self.tdc.wf(channel, apply_filter=True, hilbert=True , tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)
            traces[channel_index] = self.tdc.smoothArray(traces[channel_index], index_window_width=40)
            traces[channel_index] = traces[channel_index]/traces[channel_index].max()
        
        # traces = traces/traces.max()

        #Tuning parameters
        rel_height  = 0.4
        height      = 0.2
        prominence  = 0.3
        distance    = 15
        width       = (5,60)

        n_peaks = numpy.zeros(len(channels))

        if debug:
            print('\neventid %i Method 3'%eventid)

        for channel_index, channel in enumerate(channels):
            peaks, props = scipy.signal.find_peaks(traces[channel_index], rel_height=rel_height, height=height, prominence=prominence, distance=distance, width=width)
            n_peaks[channel_index] = len(peaks)
            if debug:
                print('\nchannel %i'%channel)
                for peak_index, peak in enumerate(peaks):
                    print('  Peak at %i'%peak)
                    for key, item in props.items():
                        print('\t{0:<15} - {1:>10.3f}'.format(key, item[peak_index]))
                plt.plot(traces[channel_index],label='Ch%i'%channel, c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channel_index])
                for peak_index, peak in enumerate(peaks):
                    plt.scatter(peak, props['peak_heights'][peak_index], c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channel_index])


        if debug:
            plt.xlabel('ns')
            plt.ylabel('Hilbert Evelope (adu)')
            plt.legend()


        if debug:
            if return_fig == False:
                import pdb; pdb.set_trace()
                plt.close(fig)
                return n_peaks
            else:
                return n_peaks, fig
        else:
            return n_peaks



if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'

    run = 5740
    datapath = os.environ['BEACON_DATA']
    reader = Reader(datapath,run)
    mt = methodTester(reader)


    if False:
        for eventid in range(40000,50000):
            peaks, props = mt.method1(eventid, mode='hpol', flag_plot=True, force_plot=False)

    elif False:
        for eventid in [ 4053,  6438, 16956, 45475]:
            peaks, props = mt.method1(eventid, mode='hpol', flag_plot=False, force_plot=True)
    elif False:
        for eventid in [ 4053,  6438, 16956, 45475, 40055][::-1]:
            npeaks = mt.method2(eventid, mode='hpol', debug=True)
    elif False:
        for eventid in [ 4053,  6438, 16956, 45475, 40055][::-1]:
            npeaks = mt.method3(eventid, mode='hpol', debug=True)
    elif False:
        # for eventid in range(40000,50000):
        for eventid in [ 4053,  6438, 16956, 45475, 40055][::-1]:
            peaks, props = mt.method1(eventid, mode='hpol', flag_plot=False, force_plot=False)
            npeaks = mt.method2(eventid, mode='hpol', debug=False)

            if sum(npeaks > 2) > 2 and len(peaks) == 1:
                fig = mt.plotWf(eventid,title='Method 2 Flagged')
                peaks, props, fig1 = mt.method1(eventid, mode='hpol', flag_plot=False, force_plot=True,return_fig=True)
                npeaks, fig2 = mt.method2(eventid, mode='hpol', debug=True,return_fig=True)
                import pdb; pdb.set_trace()
                plt.close(fig1)
                plt.close(fig2)
                plt.close(fig)
            elif sum(npeaks > 2) <= 2 and len(peaks) > 1:
                fig = mt.plotWf(eventid,title='Method 1 Flagged', mode='hpol')
                peaks, props, fig1 = mt.method1(eventid, mode='hpol', flag_plot=False, force_plot=True,return_fig=True)
                npeaks, fig2 = mt.method2(eventid, mode='hpol', debug=True,return_fig=True)
                import pdb; pdb.set_trace()
                plt.close(fig1)
                plt.close(fig2)
                plt.close(fig)
            elif sum(npeaks > 2) > 2 and len(peaks) > 1:
                continue
                fig = mt.plotWf(eventid,title='Method 1 and 2 Flagged', mode='hpol')
                import pdb; pdb.set_trace()
                plt.close(fig)
            else:
                continue
    elif True:
        # for eventid in [40063]:
        # for eventid in range(40000,50000):
        for eventid in [ 4053,  6438, 16956, 45475, 40055][::-1]:
            #Test method 1
            peaks, props = mt.method1(eventid, mode='hpol', flag_plot=False, force_plot=False)
            m1_pass = len(peaks) > 1

            npeaks = mt.method2(eventid, mode='hpol', debug=False)
            m2_pass = sum(npeaks >= 2) > 2

            npeaks = mt.method3(eventid, mode='hpol', debug=False)
            m3_pass = sum(npeaks >= 2) > 2

            passes = numpy.array([m1_pass, m2_pass, m3_pass])

            if sum(passes == False) == 2 or True:
                fig = mt.plotWf(eventid, mode='hpol' ,title='Method Flagged by Method(s) %s'%str(numpy.arange(3)[passes] + 1))

                peaks, props, fig1 = mt.method1(eventid, mode='hpol', flag_plot=False, force_plot=True,return_fig=True, plot_mode='simple')
                npeaks, fig2 = mt.method2(eventid, mode='hpol', debug=True,return_fig=True, plot_mode='simple')
                npeaks, fig3 = mt.method3(eventid, mode='hpol', debug=True,return_fig=True)
                import pdb; pdb.set_trace()
                plt.close(fig)
                plt.close(fig1)
                plt.close(fig2)
                plt.close(fig3)


