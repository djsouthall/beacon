'''
This is a script load waveforms using the sine subtraction method, and save any identified CW present in events.
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv
import h5py
import inspect

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
from tools.fftmath import TimeDelayCalculator
from tools.data_handler import createFile

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
plt.ion()

datapath = os.environ['BEACON_DATA']


class CoincidenceCalculator():
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

    notch_tv = True
    misc_notches = True
    # , notch_tv=notch_tv, misc_notches=misc_notches

    def __init__(self,  reader):
        try:
            final_corr_length=CoincidenceCalculator.final_corr_length
            crit_freq_low_pass_MHz=CoincidenceCalculator.crit_freq_low_pass_MHz
            crit_freq_high_pass_MHz=CoincidenceCalculator.crit_freq_high_pass_MHz
            low_pass_filter_order=CoincidenceCalculator.low_pass_filter_order
            high_pass_filter_order=CoincidenceCalculator.high_pass_filter_order
            waveform_index_range=CoincidenceCalculator.waveform_index_range
            plot_filters=CoincidenceCalculator.plot_filters
            apply_phase_response=CoincidenceCalculator.apply_phase_response
            notch_tv=CoincidenceCalculator.notch_tv
            misc_notches=CoincidenceCalculator.misc_notches
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        import pdb; pdb.set_trace()
        

        self.reader = reader
        self.tdc = TimeDelayCalculator(self.reader, final_corr_length=CoincidenceCalculator.final_corr_length, crit_freq_low_pass_MHz=CoincidenceCalculator.crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=CoincidenceCalculator.crit_freq_high_pass_MHz, low_pass_filter_order=CoincidenceCalculator.low_pass_filter_order, high_pass_filter_order=CoincidenceCalculator.high_pass_filter_order,waveform_index_range=CoincidenceCalculator.waveform_index_range,plot_filters=CoincidenceCalculator.plot_filters,apply_phase_response=CoincidenceCalculator.apply_phase_response, notch_tv=CoincidenceCalculator.notch_tv, misc_notches=CoincidenceCalculator.misc_notches)
        self.tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=5, verbose=False, plot=False)

        self.method1_normalization = numpy.correlate(numpy.ones_like(self.tdc.t()),numpy.ones_like(self.tdc.t()),mode='same')

        self.chunk_width = int(len(self.tdc.t())/32)
        self.shift_factor = 8
        self.chunk_shift = int(self.chunk_width/self.shift_factor)
        self.starts = numpy.arange(0,len(self.tdc.t())-self.chunk_width,self.chunk_shift).astype(int)
        self.stops = (self.starts + self.chunk_width).astype(int)
        self.norm = numpy.correlate(numpy.ones(self.chunk_width),numpy.ones(self.chunk_width),mode='same')

        # Method 1 Peak Tuning parameters
        self.method_1_rel_height  = 0.5
        self.method_1_height      = 0.3
        self.method_1_prominence  = 0.2
        self.method_1_distance    = int(50.0/self.tdc.t()[1])
        self.method_1_width       = (int(50.0/self.tdc.t()[1]) , int(250.0/self.tdc.t()[1]))

        # Method 2 Peak Tuning parameters
        self.method_2_rel_height  = 0.5
        self.method_2_height      = 0.2
        self.method_2_prominence  = 0.15
        self.method_2_distance    = 4
        self.method_2_width       = (3,15)

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

    def countPeaksMethod1(self, eventid, mode=None, flag_plot=False, force_plot=False, return_fig=False, plot_mode='full'):
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
                hcorr_env = _hcorr_env
            else:
                hcorr_env += _hcorr_env

        peaks, props = scipy.signal.find_peaks(self.normalizeCorr(hcorr_env), rel_height=self.method_1_rel_height, height=self.method_1_height, prominence=self.method_1_prominence, distance=self.method_1_distance, width=self.method_1_width)

        npeaks = len(peaks)

        if (flag_plot and len(peaks[peaks > len(self.tdc.t())/2.0 - 1]) > 1) or force_plot:
            print('\neventid %i Method 1'%eventid)
            for peak_index, peak in enumerate(peaks):
                print('Peak at %i'%peak)
                for key, item in props.items():
                    print('\t{0:<15} - {1:>10.3f}'.format(key, item[peak_index]))
            fig = plt.figure()
            plt.suptitle('Method 1\nr%ie%i'%(run,eventid),fontsize=18)
            if plot_mode == 'full':
                plt.subplot(2,1,1)

            for channel_index, channel in enumerate(channels):
                wf      = self.tdc.wf(channel, apply_filter=True, hilbert=False, tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)
                wf_hilb = self.tdc.wf(channel, apply_filter=True, hilbert=True , tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)

                if plot_mode == 'full':
                    plt.plot(self.tdc.t(),wf, c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channel])
                    plt.plot(self.tdc.t(),wf_hilb,linestyle='--', c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channel])

                _hcorr = numpy.correlate(wf_hilb,wf_hilb,mode='same')/(self.method1_normalization*len(channels))
                _hcorr_env = numpy.abs(scipy.signal.hilbert(_hcorr))

                if channel_index == 0:
                    hcorr_env = _hcorr_env
                else:
                    hcorr_env += _hcorr_env

            if plot_mode == 'full':
                plt.subplot(2,1,2)
            plt.plot(self.tdc.t(),self.normalizeCorr(hcorr_env)   ,label='hilb(mean(Corr(hilb(wf),hilb(wf))_i))')
            plt.legend()

            for peak_index, peak in enumerate(peaks):
                plt.scatter(peak*self.tdc.t()[1], props['peak_heights'][peak_index], c='r')

            if return_fig == False:
                import pdb; pdb.set_trace()
                plt.close(fig)
        if return_fig == True and ((flag_plot and len(peaks[peaks > len(self.tdc.t())/2.0 - 1]) > 1) or force_plot):
            return npeaks, fig
        else:
            return npeaks

    def countPeaksMethod2(self, eventid, mode=None, debug=False, flag_plot=False, force_plot=False, return_fig=False, plot_mode='full'):
        '''
        This calculates autocorrelation of the waveform in small windowed chunks.

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

        if debug:
            fig = plt.figure()
            plt.suptitle('Method 2\nr%ie%i'%(run,eventid),fontsize=18)

        traces = numpy.zeros((len(channels),len(self.starts)))
        for channel_index, channel in enumerate(channels):
            wf = self.tdc.wf(channel, apply_filter=True, hilbert=False , tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)

            if debug and plot_mode == 'full':
                plt.subplot(len(channels)+1,1,channel_index + 1)
            for i, (start, stop) in enumerate(zip(self.starts,self.stops)):
                traces[channel_index][i] = numpy.max(numpy.correlate(wf[start:stop],wf[start:stop],mode='same')/(self.norm*len(channels)))
                if debug and plot_mode == 'full':
                    plt.plot(self.tdc.t()[start:stop],wf[start:stop]/max(numpy.abs(wf)) + 2*(i%self.shift_factor))
        
            traces[channel_index] = traces[channel_index]/traces[channel_index].max()

        n_peaks = numpy.zeros(len(channels))

        if debug:
            print('\neventid %i Method 2'%eventid)

        for channel_index, channel in enumerate(channels):
            peaks, props = scipy.signal.find_peaks(traces[channel_index], rel_height=self.method_2_rel_height, height=self.method_2_height, prominence=self.method_2_prominence, distance=self.method_2_distance, width=self.method_2_width)
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




if __name__=="__main__":
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        print('No Run Given')
        sys.exit(1)

    debug = True

    if debug == True:
        read_mode = 'r'
    else:
        read_mode = 'a'


    try:
        run = int(run)

        reader = Reader(datapath,run)
        filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
        
        try:
            print(reader.status())
        except Exception as e:
            print('Status Tree not present.  Returning Error.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(1)
        
        cc = CoincidenceCalculator(reader)

        
        if filename is not None:
            with h5py.File(filename, read_mode) as file:
                eventids = file['eventids'][...]
                dsets = list(file.keys()) #Existing datasets

                if debug == False:

                    if not numpy.isin('coincidence_count',dsets):
                        file.create_group('coincidence_count')
                    else:
                        print('coincidence_count dataset already exists in file %s'%filename)

                    cc_dsets = list(file['coincidence_count'].keys())

                    if not numpy.isin('method_1',cc_dsets):
                        file['coincidence_count'].create_group('method_1')
                    else:
                        print('coincidence_count method_1 group already exists in file %s'%filename)

                    if not numpy.isin('hpol',list(file['coincidence_count']['method_1'].keys())):
                        file['coincidence_count']['method_1'].create_dataset('hpol', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                    else:
                        print('coincidence_count method_1 hpol dset already exists in file %s'%filename)

                    if not numpy.isin('vpol',list(file['coincidence_count']['method_1'].keys())):
                        file['coincidence_count']['method_1'].create_dataset('vpol', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                    else:
                        print('coincidence_count method_1 vpol dset already exists in file %s'%filename)

                                   
                    if not numpy.isin('method_2',cc_dsets):
                        file['coincidence_count'].create_dataset('method_2', (file.attrs['N'],8), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                    else:
                        print('coincidence_count method_2 dataset already exists in file %s'%filename)


                    #Add attributes for future replicability. 
                    file['coincidence_count'].attrs['chunk_width']  = str(cc.chunk_width)
                    file['coincidence_count'].attrs['shift_factor'] = str(cc.shift_factor)
                    file['coincidence_count'].attrs['chunk_shift']  = str(cc.chunk_shift)

                    file['coincidence_count'].attrs['final_corr_length']            = str(CoincidenceCalculator.final_corr_length)
                    file['coincidence_count'].attrs['waveform_index_range']         = str(CoincidenceCalculator.waveform_index_range)
                    file['coincidence_count'].attrs['plot_filters']                 = str(CoincidenceCalculator.plot_filters)
                    file['coincidence_count'].attrs['align_method_13_n']            = str(CoincidenceCalculator.align_method_13_n)
                    file['coincidence_count'].attrs['crit_freq_low_pass_MHz']       = str(CoincidenceCalculator.crit_freq_low_pass_MHz)
                    file['coincidence_count'].attrs['low_pass_filter_order']        = str(CoincidenceCalculator.low_pass_filter_order)
                    file['coincidence_count'].attrs['crit_freq_high_pass_MHz']      = str(CoincidenceCalculator.crit_freq_high_pass_MHz)
                    file['coincidence_count'].attrs['high_pass_filter_order']       = str(CoincidenceCalculator.high_pass_filter_order)
                    file['coincidence_count'].attrs['sine_subtract']                = str(CoincidenceCalculator.sine_subtract)
                    file['coincidence_count'].attrs['sine_subtract_min_freq_GHz']   = str(CoincidenceCalculator.sine_subtract_min_freq_GHz)
                    file['coincidence_count'].attrs['sine_subtract_max_freq_GHz']   = str(CoincidenceCalculator.sine_subtract_max_freq_GHz)
                    file['coincidence_count'].attrs['sine_subtract_percent']        = str(CoincidenceCalculator.sine_subtract_percent)
                    file['coincidence_count'].attrs['apply_phase_response']         = str(CoincidenceCalculator.apply_phase_response)
                    file['coincidence_count'].attrs['shorten_signals']              = str(CoincidenceCalculator.shorten_signals)
                    file['coincidence_count'].attrs['shorten_thresh']               = str(CoincidenceCalculator.shorten_thresh)
                    file['coincidence_count'].attrs['shorten_delay']                = str(CoincidenceCalculator.shorten_delay)
                    file['coincidence_count'].attrs['shorten_length']               = str(CoincidenceCalculator.shorten_length)

                    # Method 1 Peak Tuning parameters
                    file['coincidence_count'].attrs['method_1_rel_height']  = str(cc.method_1_rel_height)
                    file['coincidence_count'].attrs['method_1_height']      = str(cc.method_1_height)
                    file['coincidence_count'].attrs['method_1_prominence']  = str(cc.method_1_prominence)
                    file['coincidence_count'].attrs['method_1_distance']    = str(cc.method_1_distance)
                    file['coincidence_count'].attrs['method_1_width']       = str(cc.method_1_width)

                    # Method 2 Peak Tuning parameters
                    file['coincidence_count'].attrs['method_2_rel_height']  = str(cc.method_2_rel_height)
                    file['coincidence_count'].attrs['method_2_height']      = str(cc.method_2_height)
                    file['coincidence_count'].attrs['method_2_prominence']  = str(cc.method_2_prominence)
                    file['coincidence_count'].attrs['method_2_distance']    = str(cc.method_2_distance)
                    file['coincidence_count'].attrs['method_2_width']       = str(cc.method_2_width)

                
                else:
                    print('Attempting to access and print existing coincidence_count values.')
                    try:
                        if numpy.isin('coincidence_count',dsets):
                            print('coincidence_count group in dsets.')
                        else:
                            print('coincidence_count group NOT in dsets.')

                        cc_dsets = list(file['coincidence_count'].keys())

                        #Add attributes for future replicability. 
                        print("file['coincidence_count'].attrs['chunk_width'] = ", file['coincidence_count'].attrs['chunk_width'])
                        print("file['coincidence_count'].attrs['shift_factor'] = ", file['coincidence_count'].attrs['shift_factor'])
                        print("file['coincidence_count'].attrs['chunk_shift'] = ", file['coincidence_count'].attrs['chunk_shift'])

                        print("file['coincidence_count'].attrs['final_corr_length'] = ", file['coincidence_count'].attrs['final_corr_length'])
                        print("file['coincidence_count'].attrs['waveform_index_range'] = ", file['coincidence_count'].attrs['waveform_index_range'])
                        print("file['coincidence_count'].attrs['plot_filters'] = ", file['coincidence_count'].attrs['plot_filters'])
                        print("file['coincidence_count'].attrs['align_method_13_n'] = ", file['coincidence_count'].attrs['align_method_13_n'])
                        print("file['coincidence_count'].attrs['crit_freq_low_pass_MHz'] = ", file['coincidence_count'].attrs['crit_freq_low_pass_MHz'])
                        print("file['coincidence_count'].attrs['low_pass_filter_order'] = ", file['coincidence_count'].attrs['low_pass_filter_order'])
                        print("file['coincidence_count'].attrs['crit_freq_high_pass_MHz'] = ", file['coincidence_count'].attrs['crit_freq_high_pass_MHz'])
                        print("file['coincidence_count'].attrs['high_pass_filter_order'] = ", file['coincidence_count'].attrs['high_pass_filter_order'])
                        print("file['coincidence_count'].attrs['sine_subtract'] = ", file['coincidence_count'].attrs['sine_subtract'])
                        print("file['coincidence_count'].attrs['sine_subtract_min_freq_GHz'] = ", file['coincidence_count'].attrs['sine_subtract_min_freq_GHz'])
                        print("file['coincidence_count'].attrs['sine_subtract_max_freq_GHz'] = ", file['coincidence_count'].attrs['sine_subtract_max_freq_GHz'])
                        print("file['coincidence_count'].attrs['sine_subtract_percent'] = ", file['coincidence_count'].attrs['sine_subtract_percent'])
                        print("file['coincidence_count'].attrs['apply_phase_response'] = ", file['coincidence_count'].attrs['apply_phase_response'])
                        print("file['coincidence_count'].attrs['shorten_signals'] = ", file['coincidence_count'].attrs['shorten_signals'])
                        print("file['coincidence_count'].attrs['shorten_thresh'] = ", file['coincidence_count'].attrs['shorten_thresh'])
                        print("file['coincidence_count'].attrs['shorten_delay'] = ", file['coincidence_count'].attrs['shorten_delay'])
                        print("file['coincidence_count'].attrs['shorten_length'] = ", file['coincidence_count'].attrs['shorten_length'])

                        # Method 1 Peak Tuning parameters
                        print("file['coincidence_count'].attrs['method_1_rel_height'] = ", file['coincidence_count'].attrs['method_1_rel_height'])
                        print("file['coincidence_count'].attrs['method_1_height'] = ", file['coincidence_count'].attrs['method_1_height'])
                        print("file['coincidence_count'].attrs['method_1_prominence'] = ", file['coincidence_count'].attrs['method_1_prominence'])
                        print("file['coincidence_count'].attrs['method_1_distance'] = ", file['coincidence_count'].attrs['method_1_distance'])
                        print("file['coincidence_count'].attrs['method_1_width'] = ", file['coincidence_count'].attrs['method_1_width'])

                        # Method 2 Peak Tuning parameters
                        print("file['coincidence_count'].attrs['method_2_rel_height'] = ", file['coincidence_count'].attrs['method_2_rel_height'])
                        print("file['coincidence_count'].attrs['method_2_height'] = ", file['coincidence_count'].attrs['method_2_height'])
                        print("file['coincidence_count'].attrs['method_2_prominence'] = ", file['coincidence_count'].attrs['method_2_prominence'])
                        print("file['coincidence_count'].attrs['method_2_distance'] = ", file['coincidence_count'].attrs['method_2_distance'])
                        print("file['coincidence_count'].attrs['method_2_width'] = ", file['coincidence_count'].attrs['method_2_width'])

                    except Exception as e:
                        print(e)


                for eventid in eventids: 
                    if eventid%500 == 0:
                        sys.stdout.write('(%i/%i)\r'%(eventid,len(eventids)))
                        sys.stdout.flush()
                    try:
                        npeaks1_h = cc.countPeaksMethod1(eventid, mode='hpol', flag_plot=False, force_plot=False, return_fig=False, plot_mode='full')
                        npeaks1_v = cc.countPeaksMethod1(eventid, mode='vpol', flag_plot=False, force_plot=False, return_fig=False, plot_mode='full')

                        npeaks2 = cc.countPeaksMethod2(eventid, mode=None, debug=False, flag_plot=False, force_plot=False, return_fig=False, plot_mode='full')
                        if debug == False:
                            file['coincidence_count']['method_1']['hpol'][eventid] = npeaks1_h
                            file['coincidence_count']['method_1']['vpol'][eventid] = npeaks1_v
                            file['coincidence_count']['method_2'][eventid,:] = npeaks2
                        else:
                            print(npeaks1_h, npeaks1_v, npeaks2)
                    except Exception as e:
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                file.close()
        else:
            print('filename is None, indicating empty tree.  Skipping run %i'%run)
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
