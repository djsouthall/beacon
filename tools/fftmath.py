'''
This file contains the classes used for time delays and cross correlations.
The goal is to centralize essential calculations such that I am at least consistent
between various scripts. 
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv
import inspect
import ROOT
ROOT.gSystem.Load(os.environ['LIB_ROOT_FFTW_WRAPPER_DIR'] + 'build/libRootFftwWrapper.so.3')
ROOT.gInterpreter.ProcessLine('#include "%s"'%(os.environ['LIB_ROOT_FFTW_WRAPPER_DIR'] + 'include/FFTtools.h'))

from ROOT import FFTtools

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
#from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_handler import loadTriggerTypes
import analysis.phase_response as pr
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button, RadioButtons
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
plt.ion()


class FFTPrepper:
    '''
    Takes a run reader and does the math required to prepare for calculations such as
    cross correlations for time delays or template making/searching.

    Parameters
    ----------
    reader : beaconroot.examples.beacon_data_reader.reader or beacon.tools.sine_subtract_cache.sineSubtractedReader
        The run reader you wish to examine time delays for.
    final_corr_length : int
        Should be given as a power of 2.  This is the goal length of the cross correlations, and can set the time resolution
        of the time delays.
    crit_freq_low_pass_MHz : None, float, or numpy.array of length 2 or 8
        Sets the critical frequency of the low pass filter to be applied to the data.  If given as a single float value
        then the same filter will be applied to all channels.  If given as an array of 2 values, the first will be 
        applied to all Hpol channels, and the second to all Vpol channels.  If given as an array of 8 values then each
        channel will use a different filter. 
    crit_freq_high_pass_MHz : None, float, or numpy.array of length 2 or 8
        Sets the critical frequency of the high pass filter to be applied to the data.  If given as a single float value
        then the same filter will be applied to all channels.  If given as an array of 2 values, the first will be 
        applied to all Hpol channels, and the second to all Vpol channels.  If given as an array of 8 values then each
        channel will use a different filter.
    low_pass_filter_order : None, int, or numpy.array of length 2 or 8
        Sets the order of the low pass filter to be applied to the data.  If given as a single float value
        then the same filter will be applied to all channels.  If given as an array of 2 values, the first will be 
        applied to all Hpol channels, and the second to all Vpol channels.  If given as an array of 8 values then each
        channel will use a different filter.
    high_pass_filter_order : None, int, or numpy.array of length 2 or 8
        Sets the order of the high pass filter to be applied to the data.  If given as a single float value
        then the same filter will be applied to all channels.  If given as an array of 2 values, the first will be 
        applied to all Hpol channels, and the second to all Vpol channels.  If given as an array of 8 values then each
        channel will use a different filter.
    waveform_index_range : tuple
        Tuple of values.  If the first is None then the signals will be loaded from the beginning of their default buffer,
        otherwise it will start at this index.  If the second is None then the window will go to the end of the default
        buffer, otherwise it will end in this.  

        Essentially waveforms will be loaded using wf = self.reader.wf(channel)[waveform_index_range[0]:waveform_index_range[1]]

        Bounds will be adjusted based on buffer length (in case of overflow. )
    plot_filters : bool
        Enables plotting of the generated filters to be used.
    tukey_alpha : int
        Sets the degree of tukey filter to be applied.  Default is 0.1.
    tukey_default : bool
        If True to loaded wf will have tapered edges of the waveform on the 1% level to help
        with edge effects.  This will be applied before hilbert if hilbert is true, and before
        the filter.
    apply_phase_response : bool
        If True, then the phase response will be included with the filter for each channel.  This hopefully 
        deconvolves the effects of the phase response in the signal. 

    See Also
    --------
    examples.beacon_data_reader.reader
    '''
    def __init__(self, reader, final_corr_length=2**15, crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, waveform_index_range=(None,None), plot_filters=False,tukey_alpha=0.1,tukey_default=False,apply_phase_response=False, notch_tv=True, misc_notches=True):
        try:
            self.reader = None #Value before setReader has been called.  If setReader is called multiple times this will be checked each to to throw warnings that some things might not change.
            
            # Requested params stored, as they may change internally, but original requests are preserved.
            self.requested_final_corr_length = final_corr_length
            self.requested_crit_freq_low_pass_MHz = crit_freq_low_pass_MHz
            self.requested_crit_freq_high_pass_MHz = crit_freq_high_pass_MHz
            self.requested_low_pass_filter_order = low_pass_filter_order
            self.requested_high_pass_filter_order = high_pass_filter_order
            self.requested_waveform_index_range = waveform_index_range
            self.requested_plot_filters = plot_filters
            self.requested_tukey_alpha = tukey_alpha
            self.requested_tukey_default = tukey_default
            self.requested_apply_phase_response = apply_phase_response
            self.raw_buffer_length = reader.header().buffer_length #Value before prepared.  Will be overwriten once prepareWaveformIndexing called.
            self.interpretFiltersPerChannel(crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, low_pass_filter_order, high_pass_filter_order)

            # Prepare the reader and waveform info.
            self.setReader(reader) #First call this won't call self.prepareWaveformIndexing, but any subsequent calls will call self.prepareWaveformIndexing.  self.prepareWaveformIndexing is called below on first call and is forced to set everything.  All other times these will only be updated if a difference is detected.
            self.prepareWaveformIndexing(self.requested_waveform_index_range, force=True, skip_additional_prep=True,notch_tv=notch_tv, misc_notches=misc_notches)
            self.prepForFFTs(plot=plot_filters,apply_phase_response=self.requested_apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches) #must be called because skip_additional_prep above is True. 

            # self.use_sinc_interpolation = use_sinc_interpolation #Testing this right now.

            # self.crit_freq_low_pass_MHz = crit_freq_low_pass_MHz
            # self.crit_freq_high_pass_MHz = crit_freq_high_pass_MHz
            # self.low_pass_filter_order = low_pass_filter_order
            # self.high_pass_filter_order = high_pass_filter_order

            self.hpol_pairs = numpy.array(list(itertools.combinations((0,2,4,6), 2)))
            self.vpol_pairs = numpy.array(list(itertools.combinations((1,3,5,7), 2)))
            self.pairs = numpy.vstack((self.hpol_pairs,self.vpol_pairs)) 
            self.crosspol_pairs = numpy.reshape(numpy.arange(8),(4,2))

            self.tukey_default = tukey_default #This sets the default in self.wf whether to use tukey or not. 

            self.persistent_object = [] #For any objects that need to be kept alive/referenced (i.e. interactive plots)


            self.sine_subtracts = []
            self.plot_ss = []

            self.averaged_bg_squared_spectrum = None #call calculateAverageNoiseSpectrum to populate this.

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def setReader(self,reader,verbose=True):
        '''
        This will set the reader to the given reader and do other related preparations.

        Will return True if something signficant about the timing has changed.
        '''
        try:
            if verbose:
                print('Setting reader to reader of run %i'%reader.run)

            first_time = self.reader is None #True if first time the reader has been set.
            self.reader = reader
            self.reader.setEntry(0)
            self.ss_reader_mode = False
            if hasattr(self.reader, "ss_event_file"):
                if self.reader.ss_event_file is not None:
                    if verbose == True:
                        print('Sine Subtracted Reader detected and ss_event_file appears to be present.  Any sine subtraction added to this FFTPrepper object will be ignored, assuming that it will be automatically handled via precomputed sine subtraction values.')
                    self.ss_reader_mode = True

            if first_time == False:
                '''
                First time setup will call prepareWaveformIndexing itself.
                '''
                major_changes_made = self.prepareWaveformIndexing(self.requested_waveform_index_range, verbose=verbose, force=False, skip_additional_prep=False)
            else:
                major_changes_made = True

            return major_changes_made
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def prepareWaveformIndexing(self, waveform_index_range, verbose=True, force=False, skip_additional_prep=False, notch_tv=True, misc_notches=True):
        '''
        This will inspect the current reader and determine the appropriate waveform index range based upon the requested
        range and the available range. 

        Will return True if something signficant about the timing has changed.
        '''
        try:
            major_changes_made = False
            if force == True or self.raw_buffer_length != self.reader.header().buffer_length:
                '''
                Reset the buffer length only if there is a mismatch in buffer length from previous reader to current reader,
                unless forced to.
                '''
                major_changes_made = True
                if verbose:
                    print('Resetting indexing to match current reader.')
                self.raw_buffer_length = self.reader.header().buffer_length
                self.buffer_length = self.reader.header().buffer_length
                #Allowing for a subset of the waveform to be isolated.  Helpful to speed this up if you know where signals are in long traces.
                waveform_index_range = list(waveform_index_range)
                if waveform_index_range[0] is None:
                    waveform_index_range[0] = 0
                if waveform_index_range[1] is None:
                    waveform_index_range[1] = self.buffer_length - 1

                if not(waveform_index_range[0] < waveform_index_range[1]):
                    if verbose:
                        print('Given window range invalid, minimum index greater than or equal to max')
                        print('Setting full range.')
                    self.start_waveform_index = 0
                    self.end_waveform_index = self.buffer_length - 1
                else:
                    if waveform_index_range[0] < 0:
                        if verbose:
                            print('Negative start index given, setting to 0.')
                        self.start_waveform_index = 0
                    else:
                        self.start_waveform_index = waveform_index_range[0]
                    if waveform_index_range[1] >= self.buffer_length:
                        if verbose:
                            print('Greater than or equal to buffer length given for end index, setting to buffer_length - 1.')
                        self.end_waveform_index = self.buffer_length - 1
                    else:
                        self.end_waveform_index = waveform_index_range[1]

                #Resetting buffer length to account for new load in length. 
                self.buffer_length = self.end_waveform_index - self.start_waveform_index + 1 

                if self.requested_final_corr_length is None:
                    self.final_corr_length = self.buffer_length 
                else:
                    self.final_corr_length = self.requested_final_corr_length

                self.tukey = scipy.signal.tukey(self.buffer_length, alpha=self.requested_tukey_alpha, sym=True)

                if skip_additional_prep == False:
                    self.prepForFFTs(plot=False,apply_phase_response=self.requested_apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches)
            return major_changes_made
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def interpretFiltersPerChannel(self, crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, low_pass_filter_order, high_pass_filter_order):
        '''
        This is a chunk of code that will interpet the values from input filter parameters, and organize them in a
        consistent way for use elsewhere.  The result will be 4 numbers per channel indicating the filter orders and
        frequencies of the corresponding filter to be applied to each filter.

        Parameters
        ----------
        crit_freq_low_pass_MHz : None, float, or numpy.array of length 2 or 8
            Sets the critical frequency of the low pass filter to be applied to the data.  If given as a single float 
            value then the same filter will be applied to all channels.  If given as an array of 2 values, the first 
            will be applied to all Hpol channels, and the second to all Vpol channels.  If given as an array of 8 values
            then each channel will use a different filter. 
        crit_freq_high_pass_MHz : None, float, or numpy.array of length 2 or 8
            Sets the critical frequency of the high pass filter to be applied to the data.  If given as a single float 
            value then the same filter will be applied to all channels.  If given as an array of 2 values, the first 
            will be applied to all Hpol channels, and the second to all Vpol channels.  If given as an array of 8 values
            then each channel will use a different filter.
        low_pass_filter_order : None, int, or numpy.array of length 2 or 8
            Sets the order of the low pass filter to be applied to the data.  If given as a single float value then the 
            same filter will be applied to all channels.  If given as an array of 2 values, the first will be applied to
            all Hpol channels, and the second to all Vpol channels.  If given as an array of 8 values then each channel 
            will use a different filter.
        high_pass_filter_order : None, int, or numpy.array of length 2 or 8
            Sets the order of the high pass filter to be applied to the data.  If given as a single float value then the 
            same filter will be applied to all channels.  If given as an array of 2 values, the first will be applied to 
            all Hpol channels, and the second to all Vpol channels.  If given as an array of 8 values then each channel 
            will use a different filter.
        '''
        try:
            if crit_freq_low_pass_MHz is None:
                self.crit_freq_low_pass_MHz = None
            elif type(crit_freq_low_pass_MHz) is float:
                self.crit_freq_low_pass_MHz = numpy.ones(8)*crit_freq_low_pass_MHz
            elif type(crit_freq_low_pass_MHz) is int:
                self.crit_freq_low_pass_MHz = numpy.ones(8)*float(crit_freq_low_pass_MHz)
            elif type(crit_freq_low_pass_MHz) is numpy.ndarray or type(crit_freq_low_pass_MHz) is list:
                if len(crit_freq_low_pass_MHz) == 2:
                    self.crit_freq_low_pass_MHz = numpy.ones(8)
                    self.crit_freq_low_pass_MHz[0::2] = crit_freq_low_pass_MHz[0] #Hpol filters
                    self.crit_freq_low_pass_MHz[1::2] = crit_freq_low_pass_MHz[1] #Vpol filters
                elif len(crit_freq_low_pass_MHz) == 8:
                    self.crit_freq_low_pass_MHz = crit_freq_low_pass_MHz
                else:
                    print('WARNING!!!\ncrit_freq_low_pass_MHz SHOULD BE FLOAT, OR ARRAY OF LEN 1, 2, OR 8.  SETTING VALUE TO None.')
                    self.crit_freq_low_pass_MHz = None
            else:
                print('WARNING!!!\ncrit_freq_low_pass_MHz SHOULD BE FLOAT, OR ARRAY OF LEN 1, 2, OR 8.  CODE WILL BREAK')
                self.crit_freq_low_pass_MHz = None

            if crit_freq_high_pass_MHz is None:
                self.crit_freq_high_pass_MHz = None
            elif type(crit_freq_high_pass_MHz) is float:
                self.crit_freq_high_pass_MHz = numpy.ones(8)*crit_freq_high_pass_MHz
            elif type(crit_freq_high_pass_MHz) is int:
                self.crit_freq_high_pass_MHz = numpy.ones(8)*float(crit_freq_high_pass_MHz)
            elif type(crit_freq_high_pass_MHz) is numpy.ndarray or type(crit_freq_high_pass_MHz) is list:
                if len(crit_freq_high_pass_MHz) == 2:
                    self.crit_freq_high_pass_MHz = numpy.ones(8)
                    self.crit_freq_high_pass_MHz[0::2] = crit_freq_high_pass_MHz[0] #Hpol filters
                    self.crit_freq_high_pass_MHz[1::2] = crit_freq_high_pass_MHz[1] #Vpol filters
                elif len(crit_freq_high_pass_MHz) == 8:
                    self.crit_freq_high_pass_MHz = crit_freq_high_pass_MHz
                else:
                    print('WARNING!!!\ncrit_freq_high_pass_MHz SHOULD BE FLOAT, OR ARRAY OF LEN 1, 2, OR 8.  SETTING VALUE TO None.')
                    self.crit_freq_high_pass_MHz = None
            else:
                print('WARNING!!!\ncrit_freq_high_pass_MHz SHOULD BE FLOAT, OR ARRAY OF LEN 1, 2, OR 8.  CODE WILL BREAK')
                self.crit_freq_high_pass_MHz = None

            if low_pass_filter_order is None:
                self.low_pass_filter_order = None
            elif type(low_pass_filter_order) is float or type(low_pass_filter_order) is int:
                self.low_pass_filter_order = numpy.ones(8,dtype=int)*int(low_pass_filter_order)
            elif type(low_pass_filter_order) is numpy.ndarray or type(low_pass_filter_order) is list:
                if len(low_pass_filter_order) == 2:
                    self.low_pass_filter_order = numpy.ones(8)
                    self.low_pass_filter_order[0::2] = low_pass_filter_order[0] #Hpol filters
                    self.low_pass_filter_order[1::2] = low_pass_filter_order[1] #Vpol filters
                elif len(low_pass_filter_order) == 8:
                    self.low_pass_filter_order = low_pass_filter_order
                else:
                    print('WARNING!!!\ncrit_freq_low_pass_MHz SHOULD BE int, OR ARRAY OF LEN 1, 2, OR 8.  SETTING VALUE TO None.')
                    self.crit_freq_low_pass_MHz = None
            else:
                print('WARNING!!!\ncrit_freq_low_pass_MHz SHOULD BE int, OR ARRAY OF LEN 1, 2, OR 8.  CODE WILL BREAK')
                self.crit_freq_low_pass_MHz = None

            if high_pass_filter_order is None:
                self.high_pass_filter_order = None
            elif type(high_pass_filter_order) is float or type(high_pass_filter_order) is int:
                self.high_pass_filter_order = numpy.ones(8,dtype=int)*int(high_pass_filter_order)
            elif type(high_pass_filter_order) is numpy.ndarray or type(high_pass_filter_order) is list:
                if len(high_pass_filter_order) == 2:
                    self.high_pass_filter_order = numpy.ones(8)
                    self.high_pass_filter_order[0::2] = high_pass_filter_order[0] #Hpol filters
                    self.high_pass_filter_order[1::2] = high_pass_filter_order[1] #Vpol filters
                elif len(high_pass_filter_order) == 8:
                    self.high_pass_filter_order = high_pass_filter_order
                else:
                    print('WARNING!!!\nhigh_pass_filter_order SHOULD BE int, OR ARRAY OF LEN 1, 2, OR 8.  SETTING VALUE TO None.')
                    self.high_pass_filter_order = None
            else:
                print('WARNING!!!\nhigh_pass_filter_order SHOULD BE int, OR ARRAY OF LEN 1, 2, OR 8.  CODE WILL BREAK')
                self.high_pass_filter_order = None
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

        try:
            self.low_pass_filter_order
            self.crit_freq_low_pass_MHz
            self.high_pass_filter_order
            self.crit_freq_high_pass_MHz
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def rfftWrapper(self, waveform_times, *args, **kwargs):
        '''
        This basically just does an rfft but also converts linear to dB like units.  waveform_times should be given
        in ns.
        '''
        try:
            if 'return_dbish' in kwargs:
                return_dbish = kwargs['return_dbish']
                del kwargs['return_dbish']
            else:
                return_dbish =True
            numpy.seterr(divide = 'ignore') 
            spec = numpy.fft.rfft(*args, **kwargs)
            real_power_multiplier = 2.0*numpy.ones_like(spec) #The factor of 2 because rfft lost half of the power except for dc and Nyquist bins (handled below).
            if len(numpy.shape(spec)) != 1:
                real_power_multiplier[:,[0,-1]] = 1.0
            else:
                real_power_multiplier[[0,-1]] = 1.0
            freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
            if return_dbish == True:
                #If either of the above failes then will return spec_dbish
                spec_dbish = 10.0*numpy.log10( real_power_multiplier*spec * numpy.conj(spec) / len(waveform_times)) #10 because doing power in log.  Dividing by N to match monutau. 
                numpy.seterr(divide = 'warn')
                return freqs, spec_dbish, spec
            else:
                return freqs, spec

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def addSineSubtract(self, min_freq, max_freq, min_power_ratio, max_failed_iterations=3, verbose=False, plot=False):
        '''
        Each sine subtract object currently as a different range.  These are stored in a list.  Give the arguments,
        this will append to the list of sine_subtracts a new filter object matching the desired frequencies.

        Parameters
        ----------
        min_freq : float
            The minium frequency to be considered part of the same known CW source.  This should be given in GHz.  
        max_freq : float
            The maximum frequency to be considered part of the same known CW source.  This should be given in GHz.  
        min_power_ratio : float
            This is the threshold for a prominence to be considered CW.  If the power in the band defined by min_freq and
            max_freq contains more than min_power_ratio percent (where min_power_ratio <= 1.0) of the total signal power,
            then it is considered a CW source, and will be removed. 
        max_failed_iterations : int
            This sets a limiter on the number of attempts to make when removing signals, before exiting.
        '''
        if self.ss_reader_mode == True:
            print('Attempt to addSineSubtract ignored due to sineSubtractedReader being detected and usable.')
        else:
            sine_subtract = FFTtools.SineSubtract(max_failed_iterations, min_power_ratio,plot)
            if plot == True:
                print('Showing plots from SineSubtract enabled')
                self.plot_ss.append(True)
                print('WARNING!  Enabling plot for sine subtraction will result in plotting for EVERY waveform that is processed with this object.')
            else: 
                self.plot_ss.append(False)
            sine_subtract.setVerbose(verbose) #Don't print a bunch to the screen
            if numpy.logical_and(min_freq is not None,min_freq is not None):
                sine_subtract.setFreqLimits(min_freq, max_freq)
            self.sine_subtracts.append(sine_subtract)


    def wf(self, channel, apply_filter=False, hilbert=False, tukey=None, sine_subtract=False, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False):
        '''
        This loads a wf but only the section that is selected by the start and end indices specified.

        If tukey is True to loaded wf will have tapered edges of the waveform on the 1% level to help
        with edge effects.  This will be applied before hilbert if hilbert is true, and before
        the filter.

        If ss_first is True then the waveform index range will only be applied AFTER sine subtraction is applied to
        help the filtering. 

        If attempt_raw_reader == True AND self.ss_reader_mode is true (and thus the raw_wf callable is presnt), this 
        will use the raw_wf method instead.  This avoids using stored sine subtraction and can be helpful for
        comparison.  Only works if sine_subtract and 
        '''
        try:
            if self.ss_reader_mode == True and numpy.logical_and(sine_subtract, return_sine_subtract_info == False):
                temp_wf = self.reader.wf(int(channel))[self.start_waveform_index:self.end_waveform_index+1]
            else:
                if self.ss_reader_mode == True:
                    if ss_first == True:
                        temp_wf = self.reader.raw_wf(int(channel))
                    else:
                        temp_wf = self.reader.raw_wf(int(channel))[self.start_waveform_index:self.end_waveform_index+1]
                else:
                    if ss_first == True:
                        temp_wf = self.reader.wf(int(channel))
                    else:
                        temp_wf = self.reader.wf(int(channel))[self.start_waveform_index:self.end_waveform_index+1]

                temp_wf -= numpy.mean(temp_wf)
                temp_wf = temp_wf.astype(numpy.double)
                ss_freqs = []
                n_fits = []
                if numpy.logical_and(sine_subtract, len(self.sine_subtracts) > 0):
                    for ss_index, ss in enumerate(self.sine_subtracts):
                        #_temp_wf is the output array for the subtractCW function, and must be predefined.  
                        _temp_wf = numpy.zeros(len(temp_wf),dtype=numpy.double)#numpy.zeros_like(temp_wf)
                        #Do the sine subtraction
                        ss.subtractCW(len(temp_wf),temp_wf.data,self.dt_ns_original,_temp_wf)#*1e-9,_temp_wf)#self.dt_ns_original

                        #Check how many solutions were found
                        n_fit = ss.getNSines()
                        n_fits.append(n_fit)
                        #Save all frequencies in array
                        ss_freqs.append(numpy.frombuffer(ss.getFreqs(),dtype=numpy.float64,count=n_fit))
                        if self.plot_ss[ss_index] == True:
                            plt.figure()
                            plt.semilogy(numpy.array(ss.storedSpectra(0).GetX()), ss.storedSpectra(0).GetY())
                        if n_fit > 0:
                            temp_wf = _temp_wf

                if ss_first == True:
                    temp_wf = temp_wf[self.start_waveform_index:self.end_waveform_index+1] #indexed AFTER sine subtract because sine subtract benefits from seeing the whole wf. 

            if tukey is None:
                tukey = self.tukey_default
            if tukey == True:
                temp_wf = numpy.multiply( temp_wf , self.tukey  )
            if apply_filter == True:
                wf = numpy.fft.irfft(numpy.multiply(self.filter_original[channel],numpy.fft.rfft(temp_wf)),n=self.buffer_length) #Might need additional normalization
            else:
                wf = temp_wf

            if hilbert == True:
                wf = numpy.abs(scipy.signal.hilbert(wf))

            if numpy.logical_and(sine_subtract, return_sine_subtract_info):
                return wf, ss_freqs, n_fits
            else:
                return wf
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def t(self):
        '''
        This loads a times but only the section that is selected by the start and end indices specified.

        This will also roll the starting point to zero. 
        '''
        try:
            return self.reader.t()[self.start_waveform_index:self.end_waveform_index+1] - self.reader.t()[self.start_waveform_index]
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def setEntry(self, entry):
        try:
            self.reader.setEntry(entry)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def prepForFFTs(self,plot=False,apply_phase_response=False , notch_tv=True, misc_notches=True):
        '''
        This will get timing information from the reader and use it to determine values such as timestep
        that will be used when performing ffts later.  
        '''
        try:
            self.eventid = 0
            self.reader.setEntry(self.eventid)

            #Below are for the original times of the waveforms and correspond frequencies.
            self.waveform_times_original = self.t()
            self.dt_ns_original = self.waveform_times_original[1]-self.waveform_times_original[0] #ns
            self.freqs_original = numpy.fft.rfftfreq(len(self.waveform_times_original), d=self.dt_ns_original/1.0e9)
            self.df_original = self.freqs_original[1] - self.freqs_original[0]

            #Below are for waveforms padded to a power of 2 to make ffts faster.  This has the same timestep as original.  
            self.waveform_times_padded_to_power2 = numpy.arange(2**(numpy.ceil(numpy.log2(len(self.waveform_times_original)))))*self.dt_ns_original #Rounding up to a factor of 2 of the len of the waveforms
            self.freqs_padded_to_power2 = numpy.fft.rfftfreq(len(self.waveform_times_padded_to_power2), d=self.dt_ns_original/1.0e9)
            self.df_padded_to_power2 = self.freqs_padded_to_power2[1] - self.freqs_padded_to_power2[0]

            #Below are for waveforms that were padded to a power of 2, then multiplied by two to add zero padding for cross correlation.
            self.waveform_times_corr = numpy.arange(2*len(self.waveform_times_padded_to_power2))*self.dt_ns_original #multiplying by 2 for cross correlation later.
            self.freqs_corr = numpy.fft.rfftfreq(len(self.waveform_times_corr), d=self.dt_ns_original/1.0e9)
            self.df_corr = self.freqs_corr[1] - self.freqs_corr[0]

            #The above is used in the frequency domain for cross correlations.  But these are typically upsampled in the frequency domain to given better time
            #resolution information for time shifts.  The below correspond to the upsampled times.
            self.dt_ns_upsampled = 1.0e9/(2*(self.final_corr_length//2 + 1)*self.df_corr)
            self.corr_time_shifts = self.calculateTimeShifts(self.final_corr_length,self.dt_ns_upsampled)#This results in the maxiumum of an autocorrelation being located at a time shift of 0.0
            self.corr_index_to_delay_index = -numpy.arange(-(self.final_corr_length-1)//2,(self.final_corr_length-1)//2 + 1) #Negative because with how it is programmed you would want to roll the template the normal amount, but I will be rolling the waveforms.

            #Prepare Filters
            self.filter_original = self.makeFilter(self.freqs_original, plot_filter=plot, apply_phase_response=apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches)
            #self.filter_padded_to_power2 = self.makeFilter(self.freqs_padded_to_power2,plot_filter=False)
            #self.filter_corr = self.makeFilter(self.freqs_corr,plot_filter=False)
            


        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def prepPhaseFilter(self, goal_freqs, plot=False):
        '''
        This will load the phase responses for the second stage board and preamps (preamps meaned)
        and create the additional portion of the filter that will be applied to signals.  This
        ideally will make the signals more impulsive.
        '''
        phase_response_2nd_stage = pr.loadInterpolatedPhaseResponse2ndStage(goal_freqs, plot=plot)[1]
        phase_response_preamp = pr.loadInterpolatedPhaseResponseMeanPreamp(goal_freqs, plot=plot)[1]

        return numpy.exp(-1j*(phase_response_2nd_stage + phase_response_preamp)) #One row per channel.

    def calculateTimeShifts(self, final_corr_length, dt):
        return numpy.arange(-(self.final_corr_length-1)//2,(self.final_corr_length-1)//2 + 1)*self.dt_ns_upsampled
    

    def makeFilter(self,freqs, plot_filter=False,apply_phase_response=False, notch_tv=True, misc_notches=True):
        '''
        This will make a frequency domain filter based on the given specifications. 

        Parameters
        ----------
        freqs : numpy.array of floats
            These are the frequencies for which you would like the values of the filters to be generated.  These should 
            be given in Hz.  These are usually the default frequencies of the BEACON signal.
        plot_filter : bool
            Enables plotting of the generated filters to be used.
        apply_phase_response : bool
            If True, then the phase response will be included with the filter for each channel.  This hopefully 
            deconvolves the effects of the phase response in the signal. 
        '''
        try:
            filter_x = freqs
            filter_y = numpy.ones((8,len(filter_x)),dtype='complex128')
            
            if apply_phase_response == True:
                phase_response_filter = self.prepPhaseFilter(filter_x)
                if False:
                    plt.figure()
                    for channel in range(8):
                        plt.subplot(3,1,1)
                        plt.plot(filter_x,phase_response_filter[channel])
                        plt.subplot(3,1,2)
                        plt.plot(filter_x,numpy.log(phase_response_filter[channel])/(-1j))
                    import pdb; pdb.set_trace()

            if plot_filter == True:
                #NEEDS TO BE REMADE TO PLOT ON CHANNEL BY CHANNEL BASIS
                fig = plt.figure()
                fig.canvas.set_window_title('Filter')
                numpy.seterr(divide = 'ignore') 
                plt.suptitle('Butterworth filter frequency response')

            fs = freqs[1] - freqs[0]
            
            #Calculate non-channel specific filters

            #TV Notch filter
            if notch_tv == True:
                notch_tv_start_MHz = 52.5
                notch_tv_stop_MHz = 60.25
                tv_b, tv_a = scipy.signal.butter(3, (notch_tv_start_MHz*1e6, notch_tv_stop_MHz*1e6), 'bandstop', analog=True)
                filter_x_tv_notch, filter_y_tv_notch = scipy.signal.freqs(tv_b, tv_a,worN=freqs)
            else:
                filter_y_tv_notch = numpy.ones_like(filter_x)
                filter_x_tv_notch = filter_x

            #Apply Miscellanous notches for known sources
            if misc_notches:
                filter_y_misc_notches = numpy.ones(len(filter_x),dtype='complex128')
                for notch_start_MHz, notch_stop_MHz in ((26,28),(88,89),(106,108),(117,119),(125,127)):
                    notch_b, notch_a = scipy.signal.butter(4, (notch_start_MHz*1e6, notch_stop_MHz*1e6), 'bandstop', analog=True)
                    filter_x_misc_notches, filter_y_notch = scipy.signal.freqs(notch_b, notch_a, worN=freqs)
                    filter_y_misc_notches = numpy.multiply(filter_y_misc_notches , filter_y_notch)
            else:
                filter_y_misc_notches = numpy.ones_like(filter_x)
                filter_x_misc_notches = filter_x

            for channel in range(8):
                if numpy.logical_and(self.low_pass_filter_order is not None, self.crit_freq_low_pass_MHz is not None):
                    b, a = scipy.signal.butter(self.low_pass_filter_order[channel], self.crit_freq_low_pass_MHz[channel]*1e6, 'low', analog=True)
                    filter_x_low_pass, filter_y_low_pass = scipy.signal.freqs(b, a,worN=freqs)
                else:
                    filter_x_low_pass = filter_x
                    filter_y_low_pass = numpy.ones_like(filter_x)

                if numpy.logical_and(self.high_pass_filter_order is not None, self.crit_freq_high_pass_MHz is not None):
                    d, c = scipy.signal.butter(self.high_pass_filter_order[channel], self.crit_freq_high_pass_MHz[channel]*1e6, 'high', analog=True)
                    filter_x_high_pass, filter_y_high_pass = scipy.signal.freqs(d, c,worN=freqs)
                else:
                    filter_x_high_pass = filter_x
                    filter_y_high_pass = numpy.ones_like(filter_x)

                if apply_phase_response == True:
                    # print(filter_y.shape)
                    # print(phase_response_filter[channel].shape)
                    # print(numpy.multiply(filter_y_low_pass,filter_y_high_pass).shape)
                    # version = ".".join(map(str, sys.version_info[:3]))
                    # print(version)
                    # import pdb; pdb.set_trace()
                    filter_y[channel] = numpy.multiply(filter_y[channel], phase_response_filter[channel])

                filter_y[channel] = numpy.multiply(filter_y[channel],numpy.multiply(filter_y_low_pass,filter_y_high_pass))

                if notch_tv == True and channel%2 == 0:
                    filter_y[channel] = numpy.multiply(filter_y[channel] , filter_y_tv_notch)

                if misc_notches:
                    filter_y[channel] = numpy.multiply(filter_y[channel] , filter_y_misc_notches)
                
                if plot_filter == True:
                    plt.subplot(4,2,channel+1)
                    plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y[channel])),color='k',label='final filter ch %i'%channel)
                    if self.low_pass_filter_order is not None:
                        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_low_pass)),color='r',linestyle='--',label='low pass = order %i'%self.low_pass_filter_order[channel])
                    else:
                        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_low_pass)),color='r',linestyle='--',label='low pass = order [None]')
                    if self.high_pass_filter_order is not None:
                        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_high_pass)),color='orange',linestyle='--',label='high pass = order %i'%self.high_pass_filter_order[channel])
                    else:
                        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_high_pass)),color='orange',linestyle='--',label='high pass = order [None]')

                    if misc_notches:
                        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_misc_notches)),color='lime',linestyle='--',label='Notched Misc Frequency')

                    if notch_tv == True and channel%2 == 0:
                        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_tv_notch)),color='b',linestyle='--',label='Notched TV Frequency')

                    numpy.seterr(divide = 'warn') 
                    if channel %2 == 0:
                        plt.ylabel('Amplitude [dB]')
                    if channel >= 6:
                        plt.xlabel('Frequency [MHz]')
                    plt.margins(0, 0.1)
                    plt.grid(which='both', axis='both')
                    
                    if self.crit_freq_low_pass_MHz is not None:
                        plt.axvline(self.crit_freq_low_pass_MHz[channel], color='magenta',label='LP Crit at %0.2f MHz'%self.crit_freq_low_pass_MHz[channel]) # cutoff frequency
                    if self.crit_freq_high_pass_MHz is not None:
                        plt.axvline(self.crit_freq_high_pass_MHz[channel], color='cyan',label='HP Crit at %0.2f MHz'%self.crit_freq_high_pass_MHz[channel]) # cutoff frequency
                    plt.xlim(0,200)
                    plt.ylim(-50,10)
                    #plt.legend()

            return filter_y
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    # def sincInterp(self, x, s, u):
    #     """
    #     Interpolates x, sampled at "s" instants
    #     Output y is sampled at "u" instants ("u" for "upsampled")
        
    #     from Matlab:
    #     http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html        
    #     """
        
    #     # if len(x) != len(s):
    #     #     raise Exception, 'x and s must be the same length'
        
    #     # Find the period    
    #     T = s[1] - s[0]
        
    #     sincM = numpy.tile(u, (len(s), 1)) - numpy.tile(s[:, numpy.newaxis], (1, len(u)))
    #     y = numpy.dot(x, numpy.sinc(sincM/T))
    #     return y

    def shortenSignals(self, waveforms, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0,shorten_keep_leading=100):
        '''
        Given waveforms this will reduce time window to suround just the main signal pulses.

        waveforms should be a 2d array where each row is a waveform, with timestamp determined by self.dt_ns_original.
        '''
        try:
            for wf_index, wf in enumerate(waveforms):
                trigger_index = numpy.where(wf/max(wf) > shorten_thresh)[0][0]
                weights = numpy.ones_like(wf)
                cut = numpy.arange(len(wf)) < trigger_index + int(shorten_delay/self.dt_ns_original)
                slope = -1.0/(shorten_length/self.dt_ns_original) #Go to zero by 100ns after initial dampening.
                weights[~cut] = numpy.max(numpy.vstack((slope * numpy.arange(sum(~cut)) + 1,numpy.zeros(sum(~cut)))),axis=0) #Max so no negative weights
                cut2 = numpy.arange(len(wf)) < trigger_index - int(shorten_keep_leading/self.dt_ns_original)
                weights[cut2] = numpy.max(numpy.vstack((numpy.zeros(sum(cut2)) , slope * numpy.arange(sum(cut2))[::-1] + 1)),axis=0)
                wf = numpy.multiply(wf,weights)
                waveforms[wf_index] = wf
            return waveforms
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def shortenSignal(self, waveform, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0,shorten_keep_leading=100):
        '''
        Given waveforms this will reduce time window to suround just the main signal pulses.

        waveforms should be a 2d array where each row is a waveform, with timestamp determined by self.dt_ns_original.
        '''
        try:
            trigger_index = numpy.where(waveform/max(waveform) > shorten_thresh)[0][0]
            weights = numpy.ones_like(waveform)
            cut = numpy.arange(len(waveform)) < trigger_index + int(shorten_delay/self.dt_ns_original)
            slope = -1.0/(shorten_length/self.dt_ns_original) #Go to zero by 100ns after initial dampening.
            weights[~cut] = numpy.max(numpy.vstack((slope * numpy.arange(sum(~cut)) + 1,numpy.zeros(sum(~cut)))),axis=0) #Max so no negative weights
            cut2 = numpy.arange(len(waveform)) < trigger_index - int(shorten_keep_leading/self.dt_ns_original)
            weights[cut2] = numpy.max(numpy.vstack((numpy.zeros(sum(cut2)) , slope * numpy.arange(sum(cut2))[::-1] + 1)),axis=0)
            waveform = numpy.multiply(waveform,weights)
            return waveform
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def loadFilteredFFTs(self, eventid, channels=[0,1,2,3,4,5,6,7], hilbert=False, load_upsampled_waveforms=False, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, sine_subtract=False):
        '''
        Loads the waveforms (with pre applied filters) and upsamples them for
        for the cross correlation. 
        '''
        try:
            self.reader.setEntry(eventid)
            self.eventid = eventid
            raw_wfs_corr = numpy.zeros((len(channels),len(self.waveform_times_corr))) #upsampled to nearest power of 2 then by 2 for correlation.
            if load_upsampled_waveforms:
                upsampled_waveforms = numpy.zeros((len(channels),self.final_corr_length//2)) #These are the waveforms with the same dt as the cross correlation.
            for channel_index, channel in enumerate(channels):
                # if load_upsampled_waveforms:
                #     temp_raw_wf = self.wf(channel,hilbert=False,apply_filter=True,sine_subtract=sine_subtract) #apply hilbert after upsample
                # else:
                #     temp_raw_wf = self.wf(channel,hilbert=hilbert,apply_filter=True,sine_subtract=sine_subtract)
                temp_raw_wf = self.wf(channel,hilbert=False,apply_filter=True,sine_subtract=sine_subtract) #Used to be the above, but I think this double hilbert envelopes, as it is done below. 

                if shorten_signals == True:
                    temp_raw_wf = self.shortenSignal(temp_raw_wf,shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)

                if hilbert == True:
                    raw_wfs_corr[channel_index][0:self.buffer_length] = numpy.abs(scipy.signal.hilbert(temp_raw_wf))
                else:
                    raw_wfs_corr[channel_index][0:self.buffer_length] = temp_raw_wf
                

                if load_upsampled_waveforms:
                    temp_upsampled = numpy.fft.irfft(numpy.fft.rfft(raw_wfs_corr[channel_index][0:len(self.waveform_times_padded_to_power2)]),n=self.final_corr_length//2) * ((self.final_corr_length//2)/len(self.waveform_times_padded_to_power2))
                    if hilbert == True:
                        upsampled_waveforms[channel_index] = numpy.abs(scipy.signal.hilbert(temp_upsampled))
                    else:
                        upsampled_waveforms[channel_index] = temp_upsampled
            
            if False:
                plt.figure()
                plt.title(str(eventid))
                raw_wfs_corr_t = numpy.arange(numpy.shape(raw_wfs_corr)[1])*(self.t()[1] - self.t()[0])
                for channel_index, channel in enumerate(channels):
                    plt.subplot(4,2,channel+1)
                    plt.plot(self.t(),self.wf(channel,sine_subtract=sine_subtract),alpha=0.7)
                    plt.plot(raw_wfs_corr_t,raw_wfs_corr[channel_index])
                plt.figure()
                plt.title(str(eventid))
                for channel_index, channel in enumerate(channels):
                    plt.plot(raw_wfs_corr[channel_index][0:len(self.waveform_times_padded_to_power2)])
                import pdb; pdb.set_trace()
                    
            # if self.use_sinc_interpolation == True:
            #     #This was done to test the sinc interpolation that Steven suggested.  It turns out that the way
            #     #I was doing things is effectively the same, despite sinc likely being more accurate.
            #     for channel in range(1):
            #         plt.figure()
            #         plt.subplot(3,1,1)
            #         plt.plot(numpy.arange(len(raw_wfs_corr[channel]))*self.dt_ns_original,raw_wfs_corr[channel])
            #         plt.plot(self.t(),self.wf(channel,hilbert=False,apply_filter=True,sine_subtract=sine_subtract))
            #         plt.xlim(0,1000)
            #         plt.subplot(3,1,2)
            #         test_up = numpy.fft.irfft(numpy.fft.rfft(raw_wfs_corr[channel][0:len(self.waveform_times_padded_to_power2)]),n=self.final_corr_length//2) * ((self.final_corr_length//2)/len(self.waveform_times_padded_to_power2))
            #         plt.plot(numpy.arange(len(test_up))*self.dt_ns_upsampled, test_up)
            #         plt.xlim(0,1000)

            #         plt.subplot(3,1,3)
            #         test_sinc = self.sincInterp(self.wf(channel,hilbert=False,apply_filter=True,sine_subtract=sine_subtract), self.t(), numpy.arange(len(test_up))*self.dt_ns_upsampled)
            #         plt.plot(numpy.arange(len(test_sinc))*self.dt_ns_upsampled, test_sinc)
            #         plt.xlim(0,1000)

            #         plt.figure()
            #         plt.subplot(2,1,1)
            #         plt.plot(numpy.arange(len(test_sinc))*self.dt_ns_upsampled,test_sinc - test_up)
            #         plt.subplot(2,1,2)
            #         plt.plot(numpy.arange(len(test_sinc))*self.dt_ns_upsampled,test_sinc)
            #         plt.plot(numpy.arange(len(test_sinc))*self.dt_ns_upsampled,test_up)
                    

            #     print(numpy.shape(raw_wfs_corr))
            #     import pdb; pdb.set_trace()

            #     waveform_ffts_filtered_corr = numpy.fft.rfft(raw_wfs_corr,axis=1) #Now upsampled
            # else:
            #     waveform_ffts_filtered_corr = numpy.fft.rfft(raw_wfs_corr,axis=1) #Now upsampled

            waveform_ffts_filtered_corr = numpy.fft.rfft(raw_wfs_corr,axis=1) #Now upsampled

            if load_upsampled_waveforms:
                return waveform_ffts_filtered_corr, upsampled_waveforms
            else:
                return waveform_ffts_filtered_corr
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def loadFilteredWaveformsMultiple(self,eventids,hilbert=False,sine_subtract=False):
        '''
        Hacky way to get multiple upsampled waveforms.  Helpful for initial passes at getting a template.
        '''
        try:
            upsampled_waveforms = {}
            for channel in range(8):
                upsampled_waveforms['ch%i'%channel] = numpy.zeros((len(eventids),self.final_corr_length//2))

            for index, eventid in enumerate(eventids):
                wfs = self.loadFilteredFFTs(eventid, hilbert=hilbert, load_upsampled_waveforms=True,sine_subtract=sine_subtract)[1]
                for channel in range(8):
                    upsampled_waveforms['ch%i'%channel][index] = wfs[channel]

            return upsampled_waveforms
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateGroupDelays(self, t, wfs, plot=False, event_type=None, xlim=(20e6,130e6), group_delay_band=(55e6,70e6)):
        '''
        This will calculate the group delay curve for a given event. 
        event_type specifies which channels to plot.  If 'hpol' the even channels are plotted,
        if 'vpol' then odd, and if None then all are (if plot=True).  All channels are computed
        regardless of this setting, this is solely for display.

        group_delay_band is the region used to determine average group delay

        xlim is the frequency range you want plotted.

        '''
        try:
            ffts_dBish = []#numpy.zeros((8,len(self.freqs_original)))
            ffts = []#numpy.zeros((8,len(self.freqs_original)),dtype=numpy.complex64)
            group_delays = []#numpy.zeros((8,len(self.freqs_original)-1))
            phases = []#numpy.zeros((8,len(self.freqs_original)))

            for channel in range(8):
                freqs, _ffts_dBish, _ffts = self.rfftWrapper(t,wfs[channel])
                ffts_dBish.append(_ffts_dBish)
                ffts.append(_ffts)

                if channel == 0:
                    #Doesn't need to be calculated several times.
                    group_delay_freqs = numpy.diff(freqs) + freqs[0:len(freqs)-1]
                    omega = 2.0*numpy.pi*freqs

                phases.append(numpy.unwrap(numpy.angle(ffts[channel])))
                group_delays.append((-numpy.diff(phases[channel])/numpy.diff(omega)) * 1e9)

            ffts_dBish = numpy.array(ffts_dBish)
            ffts = numpy.array(ffts)
            group_delays = numpy.array(group_delays)
            phases = numpy.array(phases)

            weighted_group_delays = numpy.zeros(8)
            for channel in range(8):
                weights = scipy.interpolate.interp1d(freqs, abs(ffts[channel]), kind='cubic', assume_sorted=True)(group_delay_freqs)
                weighted_group_delays[channel] = numpy.average(group_delays[channel], weights = weights)

            if plot == True:
                fig = plt.figure()
                fig.canvas.set_window_title('Group Delays')


                if event_type == 'hpol':
                    channels = numpy.arange(4)*2
                elif event_type == 'vpol':
                    channels = numpy.arange(4)*2 + 1
                else:
                    channels = numpy.arange(8)

                for channel in channels:
                    channel = int(channel)

                    ax = plt.subplot(3,1,1)
                    cut = numpy.logical_and(freqs>= xlim[0],freqs<= xlim[1])
                    plt.plot(freqs[cut]/1e6,ffts_dBish[channel][cut],label=str(channel))
                    plt.ylabel('Magnitude (dB ish)')
                    plt.legend()
                    plt.xlabel('MHz')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    

                    plt.subplot(3,1,2,sharex=ax)
                    cut = numpy.logical_and(freqs>= xlim[0],freqs<= xlim[1])
                    plt.plot(freqs[cut]/1e6,phases[channel][cut],label=str(channel))
                    plt.ylabel('Unwrapped Phase (rad)')
                    plt.legend()
                    plt.xlabel('MHz')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                    plt.subplot(3,1,3,sharex=ax)
                    cut = numpy.logical_and(group_delay_freqs>= xlim[0],group_delay_freqs<= xlim[1])
                    plt.ylabel('Group Delay (ns)')
                    plt.legend()
                    plt.xlabel('MHz')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.plot(group_delay_freqs[cut]/1e6,group_delays[channel][cut],label='Ch %i Weighted Mean in Band = %0.3f ns'%(channel, weighted_group_delays[channel]))
                    plt.legend()

                fig = plt.figure()
                fig.canvas.set_window_title('Group Delay Means')
                cut = numpy.logical_and(group_delay_freqs>= xlim[0],group_delay_freqs<= xlim[1])
                plt.ylabel('Shifted Group Delay (ns)\n%0.1f-%0.1fMHz'%(group_delay_band[0]/1e6,group_delay_band[1]/1e6))
                plt.legend()
                plt.xlabel('MHz')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                for channel in channels:
                    plt.plot(group_delay_freqs[cut]/1e6,group_delays[channel][cut] - weighted_group_delays[channel],label='Ch %i - Shift = %0.3f ns'%(channel, weighted_group_delays[channel]))
                plt.legend()

            return group_delay_freqs, group_delays, weighted_group_delays
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def calculateGroupDelaysFromEvent(self, eventid, apply_filter=False, plot=False,event_type=None,sine_subtract=False):
        '''
        This will calculate the group delay curve for a given event. 
        event_type specifies which channels to plot.  If 'hpol' the even channels are plotted,
        if 'vpol' then odd, and if None then all are (if plot=True).  All channels are computed
        regardless of this setting, this is solely for display.
        '''
        try:
            self.reader.setEntry(eventid)
            self.eventid = eventid

            wfs = numpy.zeros((8,self.buffer_length))
            
            for channel in range(8):
                temp_raw_wf = self.wf(channel,hilbert=False,apply_filter=apply_filter,sine_subtract=sine_subtract) 
                wfs[channel] = temp_raw_wf - numpy.mean(temp_raw_wf) #Subtracting dc offset

            return self.calculateGroupDelays(self.t(), wfs, plot=plot,event_type=event_type)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def smoothArray(self, values, index_window_width = 20):
        '''
        This will take the input array and smooth it using a hamming window of the specified width.
        '''
        #Smoothing out rate
        hamming_filter = scipy.signal.hamming(index_window_width)
        hamming_filter = hamming_filter/sum(hamming_filter)
        if index_window_width%2 == 0:
            padded = numpy.append(numpy.append(numpy.ones(index_window_width//2)*values[0],values),numpy.ones(index_window_width//2 - 1)*values[-1])
        else:
            padded = numpy.append(numpy.append(numpy.ones(index_window_width//2)*values[0],values),numpy.ones(index_window_width//2)*values[-1])
        #import pdb; pdb.set_trace()
        smoothed = numpy.convolve(padded,hamming_filter, mode='valid')
        return smoothed

    def spec2Todbish(self, spec2, len_t_ns):
        real_power_multiplier = 2.0*numpy.ones_like(spec2) #The factor of 2 because rfft lost half of the power except for dc and Nyquist bins (handled below).
        if len(numpy.shape(spec2)) != 1:
            real_power_multiplier[:,[0,-1]] = 1.0
        else:
            real_power_multiplier[[0,-1]] = 1.0
        spec_dbish = 10.0*numpy.log10( real_power_multiplier*spec2 / len_t_ns) #10 because doing power in log.  Dividing by N to match monutau. 
        return spec_dbish

    def calculateAverageNoiseSpectrum(self,trigger_type=[1,3],td_std_cut=20, apply_filter=False, sine_subtract=False, apply_tukey=None, verbose=False, plot=False, max_counts=1000):
        '''
        This will take the average of all events in the specific trigger type and store it to the class as self.averaged_bg_squared_spectrum. 
        To avoid oddities with phases this squares signals before adding them.  It will stop averaging if max_counts is reached.  
        '''
        channels = numpy.arange(8)
        trigger_types = loadTriggerTypes(self.reader)
        cut_eventids = numpy.where(numpy.isin(trigger_types,trigger_type))[0]
        t_ns = self.t()

        added_counts = numpy.zeros(8)
        averaged_bg_squared_spectrum = None
        if plot == True:
            stds = []

        for event_index, eventid in enumerate(cut_eventids):
            if verbose:
                sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,min(max_counts,len(cut_eventids))))
            self.setEntry(eventid)
            if numpy.any(added_counts > max_counts):
                break
            for channel in channels:
                channel=int(channel)
                if sine_subtract == True:
                    wf, ss_freqs, n_fits = self.wf(channel,apply_filter=apply_filter,hilbert=False,tukey=apply_tukey,sine_subtract=sine_subtract, return_sine_subtract_info=sine_subtract)
                    if verbose:
                        print(list(zip(n_fits, ss_freqs)))
                else:
                    wf = self.wf(channel,apply_filter=apply_filter,hilbert=False,tukey=apply_tukey,sine_subtract=sine_subtract, return_sine_subtract_info=sine_subtract)
                std = numpy.std(wf)
                if plot:
                    stds.append(std)

                if std < td_std_cut:
                    #import pdb; pdb.set_trace()
                    freqs, spec = self.rfftWrapper(t_ns, wf, return_dbish=False)
                    spec2 = spec * numpy.conj(spec)
    
                    if averaged_bg_squared_spectrum is None:
                        averaged_bg_squared_spectrum = numpy.zeros((len(channels),len(spec2)))

                    added_counts[channel] += 1
                    averaged_bg_squared_spectrum[channel] += spec2.astype(float)


        self.averaged_bg_squared_spectrum = numpy.divide(averaged_bg_squared_spectrum.T, added_counts).T

        if plot:
            plt.figure()
            plt.hist(stds,bins=100)

        if verbose:
            print('An average spectrum was generated for each channel for trigger types %s, excluding %0.2f percent of events in this category due to exceeding the time domain rms of %0.2f'%(str(trigger_type), 100.0*(1- numpy.mean(added_counts)/len(cut_eventids)), td_std_cut))


        return self.averaged_bg_squared_spectrum 


    def calculateBandwidth(self, eventid, channels=[0,1,2,3,4,5,6,7], apply_filter=False, sine_subtract=False, apply_tukey=None, additional_title_text=None, time_delays=None, verbose=False, plot=False, min_freq_MHz=20, max_freq_MHz=90, step_freq_MHz=10, remove_averaged_spectrum=False):
        '''
        For each event this attempts to determine the bandwidth.

        If remove_averaged_spectrum is called then it will subtract the output of calculateAverageNoiseSpectrum from each spec2. 

        This function was never satisfactorily refined to provide adequate characterization imo.  
        '''
        try:
            bin_edges = numpy.arange(min_freq_MHz, max_freq_MHz+step_freq_MHz, step_freq_MHz)

            self.setEntry(eventid)
            t_ns = self.t()
            len_t_ns = len(t_ns)
            if verbose:
                print(eventid)
            if apply_tukey is None:
                apply_tukey = self.tukey_default
            
            if plot:
                fig = plt.figure()
            
            processed_specs = numpy.zeros((len(channels),len_t_ns//2 + 1))              

            for channel_index, channel in enumerate(channels):
                channel=int(channel)
                if sine_subtract == True:
                    wf, ss_freqs, n_fits = self.wf(channel,apply_filter=apply_filter,hilbert=False,tukey=apply_tukey,sine_subtract=sine_subtract, return_sine_subtract_info=sine_subtract)
                    if verbose:
                        print(list(zip(n_fits, ss_freqs)))
                else:
                    wf = self.wf(channel,apply_filter=apply_filter,hilbert=False,tukey=apply_tukey,sine_subtract=sine_subtract, return_sine_subtract_info=sine_subtract)

                freqs, spec = self.rfftWrapper(t_ns, wf, return_dbish=False)
                spec2 = spec * numpy.conj(spec)
                spec2 = spec2.astype(float)

                if remove_averaged_spectrum:
                    if self.averaged_bg_squared_spectrum is None:
                        self.calculateAverageNoiseSpectrum(trigger_type=[1,3], apply_filter=apply_filter, sine_subtract=sine_subtract, apply_tukey=apply_tukey, verbose=verboses)
                    spec2[spec2 > self.averaged_bg_squared_spectrum[channel]] -= self.averaged_bg_squared_spectrum[channel][spec2 > self.averaged_bg_squared_spectrum[channel]]

                smoothed_values = self.smoothArray(spec2, index_window_width = int(step_freq_MHz*1e6 / freqs[1]))#self.smoothArray(spec_dbish, index_window_width = int(step_freq_MHz*1e6 / freqs[1]))
                processed_specs[channel_index] = smoothed_values


                if plot:
                    spec_dbish = self.spec2Todbish(spec2, len_t_ns)
                    spec_dbish_smoothed = self.spec2Todbish(smoothed_values, len_t_ns)
                    plt.subplot(5,1,1)
                    plt.plot(freqs/1e6, spec_dbish)
                    plt.ylim(-20,60)
                    plt.xlim(0,150)

                    plt.subplot(5,1,2)
                    plt.plot(freqs/1e6, spec_dbish_smoothed)
                    plt.ylim(-20,60)
                    plt.xlim(0,150)

            hpol = numpy.mean(processed_specs[numpy.array(channels)%2 == 0],axis=0)
            vpol = numpy.mean(processed_specs[numpy.array(channels)%2 == 1],axis=0)


            hpol = hpol - numpy.mean(hpol[freqs > 100e6])
            vpol = vpol - numpy.mean(vpol[freqs > 100e6])


            freq_cut = numpy.logical_and(freqs/1e6 >= min_freq_MHz , freqs/1e6 <= max_freq_MHz)

            hpol = self.spec2Todbish(hpol, len_t_ns)
            hpol = numpy.ma.masked_array(hpol,mask=numpy.isnan(hpol))

            vpol = self.spec2Todbish(vpol, len_t_ns)
            vpol = numpy.ma.masked_array(vpol,mask=numpy.isnan(vpol))

            if False:
                metric_hpol = numpy.abs(numpy.std(hpol[freq_cut]))/ numpy.mean(hpol[freq_cut])
                metric_vpol = numpy.abs(numpy.std(vpol[freq_cut]))/ numpy.mean(vpol[freq_cut])

            elif False:
                metric_hpol = numpy.mean(numpy.abs(numpy.diff(hpol[freq_cut],n=2)))*numpy.mean(numpy.abs(numpy.diff(hpol[freq_cut],n=1))) / numpy.mean(hpol[freq_cut])
                metric_vpol = numpy.mean(numpy.abs(numpy.diff(vpol[freq_cut],n=2)))*numpy.mean(numpy.abs(numpy.diff(vpol[freq_cut],n=1))) / numpy.mean(vpol[freq_cut])
            else:
                metric_hpol = numpy.std(hpol[freq_cut])*numpy.mean(numpy.abs(numpy.diff(hpol[freq_cut],n=1))) / numpy.mean(hpol[freq_cut])**2
                metric_vpol = numpy.std(vpol[freq_cut])*numpy.mean(numpy.abs(numpy.diff(vpol[freq_cut],n=1))) / numpy.mean(vpol[freq_cut])**2
            
            #import pdb; pdb.set_trace()
 
            # import pdb; pdb.set_trace()
            if plot:
                plt.subplot(5,1,3)
                plt.plot(freqs/1e6, hpol, label='Metric = %0.3f'%metric_hpol)
                plt.plot(freqs/1e6, vpol, label='Metric = %0.3f'%metric_vpol)
                plt.legend(loc='upper right')
                plt.ylim(-20,80)
                plt.xlim(0,150)

                plt.subplot(5,1,4)
                plt.plot(numpy.diff(hpol, n=1))
                plt.plot(numpy.diff(vpol, n=1))
                plt.xlim(0,150)

                plt.subplot(5,1,5)
                plt.plot(numpy.abs(numpy.diff(hpol, n=2)))
                plt.plot(numpy.abs(numpy.diff(vpol, n=2)))
                plt.xlim(0,150)

                return metric_hpol, metric_vpol, fig, freqs
            else:
                return metric_hpol, metric_vpol

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def plotEvent(self, eventid, channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=False, sine_subtract=False, apply_tukey=None, additional_title_text=None, time_delays=None, verbose=False):
        '''
        This will plot all given channels in both time domain and frequency domain.
        If time delays are given then this will attempt to roll signals as specified.  Expects time delays to be
        given in the format: [0-2, 0-4, 0-6, 2-4, 2-6, 4-6, 1-3, 1-5, 1-7, 3-5, 3-7, 5-7].

        It will plot twice, once using the first three time delays to align to antenna 0/1, then using the later
        three time delays of each polarity to align signals with antenna 6/7.

        It does not use any upsampling.
        '''
        try:
            fig = plt.figure()
            if additional_title_text is not None:
                fig.canvas.set_window_title('%s: r%i-e%i Waveform and Spectrum'%(additional_title_text,self.reader.run,eventid))
                plt.suptitle('%s\nRun %i, Eventid %i'%(additional_title_text,self.reader.run,eventid))
            else:
                fig.canvas.set_window_title('r%i-e%i Waveform and Spectrum'%(self.reader.run,eventid))
                plt.suptitle('Run %i, eventid %i'%(self.reader.run,eventid))

            if time_delays is None:
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
            else:
                plt.subplot(3,1,1)
                plt.ylabel('adu')
                plt.xlabel('ns\n(Unshifted)')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                
                plt.subplot(3,1,2)
                plt.ylabel('adu')
                plt.xlabel('ns\n(Aligned to Antenna 3)')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.subplot(3,1,3)
                plt.ylabel('adu')
                plt.xlabel('ns\n(Aligned to Antenna 3)')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)



                    
            self.setEntry(eventid)
            t_ns = self.t()
            if verbose:
                print(eventid)
            if apply_tukey is None:
                apply_tukey = self.tukey_default
            
            for channel in channels:
                channel=int(channel)
                if sine_subtract == True:
                    wf, ss_freqs, n_fits = self.wf(channel,apply_filter=apply_filter,hilbert=hilbert,tukey=apply_tukey,sine_subtract=sine_subtract, return_sine_subtract_info=sine_subtract)
                    if verbose:
                        print(list(zip(n_fits, ss_freqs)))
                else:
                    wf = self.wf(channel,apply_filter=apply_filter,hilbert=hilbert,tukey=apply_tukey,sine_subtract=sine_subtract, return_sine_subtract_info=sine_subtract)

                freqs, spec_dbish, spec = self.rfftWrapper(t_ns, wf)

                if time_delays is None:

                    plt.subplot(2,1,1)
                    plt.plot(t_ns,wf)

                    plt.subplot(2,1,2)
                    plt.plot(freqs/1e6,spec_dbish/2.0,label='Ch %i'%channel)
                    plt.legend(loc = 'upper right')
                    plt.xlim(10,110)
                    plt.ylim(-20,30)
                else:
                    plt.subplot(3,1,1)
                    plt.plot(t_ns,wf)


                    plt.subplot(3,1,2)
                    #Aligning to channel 0/1
                    if channel//2 == 0:
                        plt.plot(t_ns,wf,label='Ch%i'%channel)
                    else:
                        plt.plot(t_ns + time_delays[int([0,1,2][channel//2 - 1]) + 6*(channel%2)],wf,label='Ch%i shifted %0.3f'%(channel, time_delays[int([0,1,2][channel//2 - 1]) + 6*(channel%2)]))

                    plt.legend(loc='upper right')

                    #Aligning to channel 6/7
                    plt.subplot(3,1,3)
                    if channel//2 == 3:
                        plt.plot(t_ns,wf,label='Ch%i'%channel)
                    else:
                        plt.plot(t_ns - time_delays[int([2,4,5][channel//2]) + 6*(channel%2)],wf,label='Ch%i shifted %0.3f'%(channel, - time_delays[int([2,4,5][channel//2]) + 6*(channel%2)]))
                    plt.legend(loc='upper right')


            ax = plt.gca()
            return fig, ax
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


class TimeDelayCalculator(FFTPrepper):
    '''
    Takes a run reader and does the math required for determining time delays between the antennas.

    Parameters
    ----------
    reader : examples.beacon_data_reader.reader
        The run reader you wish to examine time delays for.
    final_corr_length : int
        Should be given as a power of 2.  This is the goal length of the cross correlations, and can set the time resolution
        of the time delays.
    crit_freq_low_pass_MHz : float
        Sets the critical frequency of the low pass filter to be applied to the data.
    crit_freq_high_pass_MHz
        Sets the critical frequency of the high pass filter to be applied to the data.
    low_pass_filter_order
        Sets the order of the low pass filter to be applied to the data.
    high_pass_filter_order
        Sets the order of the high pass filter to be applied to the data.
    
    See Also
    --------
    examples.beacon_data_reader.reader
    '''
    def calculateTimeDelays(self, ffts, upsampled_waveforms, return_full_corrs=False, align_method=0, print_warning=True, align_method_10_estimate=None, align_method_10_window_ns=8, align_method_13_n=2, crosspol_delays=False):
        '''
        This will calculate the time delays based on the supplied waveforms and ffts.  If crosspol_delays == False then 
        it will perform this calculation for each baseline pair, and return the result.  If crosspol_delays == True then
        this will calculate the cross correlation between hpol and vpol for the given set of values.

        Align method can be one of a few:
        0:  argmax of corrs (default)
        1:  argmax of hilbert of corrs
        2:  Average of argmin and argmax
        3:  argmin of corrs
        4:  Pick the largest max peak preceding the max of the hilbert of corrs
        5:  Pick the average indices of values > 95% peak height in corrs
        6:  Pick the average indices of values > 98% peak height in hilbert of corrs
        7:  Gets argmax of abs(corrs) and then finds highest positive peak before this value
        8:  Apply cfd to waveforms to get first pass guess at delays, then pick the best correlation near that. 
        9:  For this I want to use the 'slider' method of visual alignment. I.e. plot the two curves, and have a slide controlling the roll of one of the waveforms. Once satisfied with the roll, lock it down.
        10: This requires expected time delays, and will just snap to the highest value within a small window around that.  Good for fine adjustments after using method 9.  
        11: For hpol baselines this will use 0, and for vpol it will use 9.  If crosspol_delays == True then this will behave identical to 9.
        12: For vpol baselines this will use 0, and for hpol it will use 9.  If crosspol_delays == True then this will behave identical to 9.
        13: This will return the top align_method_13_n peaks of the correlations for each baseline of each event.
        'max_corrs' corresponds to the value of the selected methods peak. 
        '''
        try:
            if print_warning:
                print('Note that calculateTimeDelays expects the ffts to be the same format as those loaded with loadFilteredFFTs().  If this is not the case the returned time shifts may be incorrect.')


            if crosspol_delays == False:
                pairs = self.pairs
            else:
                pairs = self.crosspol_pairs
                if align_method in [11,12]:
                    align_method = 9
                    if print_warning:
                        print('WARNING!!! Changing align method to method 9 while in crosspol mode.')

            #Handle scenario where waveform has all zeros (which results in 0 SNR which messes up FFT/correlation)
            bad_channels = numpy.where(numpy.std(ffts,axis=1) == 0)[0] 
            bad_pairs = numpy.any(numpy.isin(pairs,bad_channels),axis=1) #Calculation will proceed as normal (nan's will exist), but the output of these channel will be overwritten.

            corrs_fft = numpy.multiply((ffts[pairs[:,0]].T/numpy.std(ffts[pairs[:,0]],axis=1)).T,(numpy.conj(ffts[pairs[:,1]]).T/numpy.std(numpy.conj(ffts[pairs[:,1]]),axis=1)).T) / (len(self.waveform_times_corr)//2 + 1)

            # if ~numpy.all(numpy.isfinite(corrs_fft)):
            #     import pdb; pdb.set_trace()

            corrs = numpy.fft.fftshift(numpy.fft.irfft(corrs_fft,axis=1,n=self.final_corr_length),axes=1) * (self.final_corr_length//2) #Upsampling and keeping scale

            corrs[bad_pairs] = 0.0

            # if ~numpy.all(numpy.isfinite(corrs)):
            #     import pdb; pdb.set_trace()


            if align_method == 0:
                indices = numpy.argmax(corrs,axis=1)
                max_corrs = numpy.max(corrs,axis=1)
            elif align_method == 1:
                corr_hilbert = numpy.abs(scipy.signal.hilbert(corrs,axis=1))
                indices = numpy.argmax(corr_hilbert,axis=1)
                max_corrs = numpy.max(corr_hilbert,axis=1)
            elif align_method == 2:
                indices = numpy.mean(numpy.vstack((numpy.argmax(corrs,axis=1), numpy.argmin(corrs,axis=1))),axis=0).astype(int)
                max_corrs = numpy.max(corrs,axis=1)
            elif align_method == 3:
                indices = numpy.argmin(corrs,axis=1)
                max_corrs = numpy.min(corrs,axis=1)
            elif align_method == 4:
                corr_hilbert = numpy.abs(scipy.signal.hilbert(corrs,axis=1))
                max_indices = numpy.argmax(corr_hilbert,axis=1)
                indices = numpy.zeros_like(max_indices)
                max_corrs = numpy.zeros(len(max_indices))
                for i, c in enumerate(corr_hilbert):
                    indices[i] = numpy.argmax(c[0:max_indices[i]])
                    max_corrs[i] = numpy.max(c[0:max_indices[i]])
            elif align_method == 5:
                threshold = 0.95*numpy.max(corrs,axis=1)#numpy.tile(0.9*numpy.max(corrs,axis=1),(numpy.shape(corrs)[1],1)).T
                indices = numpy.zeros(numpy.shape(corrs)[0],dtype=int)
                max_corrs = numpy.zeros(numpy.shape(corrs)[0])
                for i, corr in enumerate(corrs):
                    indices[i] = numpy.average(numpy.where(corr > threshold[i])[0],weights=corr[corr > threshold[i]]).astype(int)
                    max_corrs[i] = numpy.max(corr[indices[i]])
            elif align_method == 6:
                corr_hilbert = numpy.abs(scipy.signal.hilbert(corrs,axis=1))
                threshold = 0.98*numpy.max(corr_hilbert,axis=1)#numpy.tile(0.9*numpy.max(corrs,axis=1),(numpy.shape(corrs)[1],1)).T
                indices = numpy.zeros(numpy.shape(corr_hilbert)[0],dtype=int)
                max_corrs = numpy.zeros(numpy.shape(corr_hilbert)[0])
                for i, corr in enumerate(corr_hilbert):
                    indices[i] = numpy.average(numpy.where(corr > threshold[i])[0],weights=corr[corr > threshold[i]]).astype(int)
                    max_corrs[i] = numpy.max(corr[indices[i]])
            elif align_method == 7:
                abs_corrs = numpy.fabs(corrs)
                initial_indices = numpy.argmax(abs_corrs,axis=1)
                #import pdb; pdb.set_trace()

                indices = numpy.zeros(numpy.shape(corrs)[0],dtype=int)
                max_corrs = numpy.zeros(numpy.shape(corrs)[0])
                for i,index in enumerate(initial_indices):
                    if corrs[i][index] > 0:
                        indices[i] = index
                    else:
                        indices[i] = numpy.argmax(corrs[i][0:index]) #Choose highest peak before max.  
                    max_corrs[i] = corrs[i][indices[i]]
            elif align_method == 8:
                '''
                Apply cfd to waveforms to get first pass guess at delays,
                then pick the best correlation near that. 
                '''
                run = int(self.reader.run)
                if run == 1507:
                    cfd_thresh = 0.8
                elif run == 1509:
                    cfd_thresh = 0.8
                elif run == 1511:
                    cfd_thresh = 0.3
                elif run == 1774:
                    cfd_thresh = 0.5
                else:
                    cfd_thresh = 0.75

                cfd_trig_times = []
                #times = self.dt_ns_upsampled*numpy.arange(numpy.shape(upsampled_waveforms)[1])
                for index, wf in enumerate(upsampled_waveforms):
                    cfd_trig_times.append(min(numpy.where(wf > max(wf)*cfd_thresh)[0])*self.dt_ns_upsampled)

                time_delays_cfd = numpy.zeros(len(pairs))
                time_windows_oneside = 5 #ns3

                indices = numpy.zeros(numpy.shape(corrs)[0],dtype=int)
                max_corrs = numpy.zeros(numpy.shape(corrs)[0])
                for pair_index, pair in enumerate(pairs):
                    time_delays_cfd = cfd_trig_times[int(min(pair))] - cfd_trig_times[int(max(pair))]
                    cut = numpy.logical_and(self.corr_time_shifts < time_delays_cfd + time_windows_oneside, self.corr_time_shifts >time_delays_cfd - time_windows_oneside)
                    indices[pair_index] = numpy.argmax( numpy.multiply( cut , corrs[pair_index] ) )
                    max_corrs[pair_index] = corrs[pair_index][indices[pair_index]]

                if False:
                    times = numpy.arange(len(wf))*self.dt_ns_upsampled
                    for pair_index, pair in enumerate(pairs):
                        plt.figure()
                        plt.subplot(2,1,1)
                        a = pair[0]
                        b = pair[1]
                        plt.plot(times, upsampled_waveforms[a],color='b',label='Ch %i'%a,alpha=0.9)
                        plt.plot(times, upsampled_waveforms[b],color='r',label='Ch %i'%b,alpha=0.9)
                        plt.axvline(cfd_trig_times[a],color='b',linestyle='--',label='Ch %i, CFD Trig %0.2f'%(a,cfd_thresh),alpha=0.7)
                        plt.axvline(cfd_trig_times[b],color='r',linestyle='--',label='Ch %i, CFD Trig %0.2f'%(b,cfd_thresh),alpha=0.7)
                        plt.ylabel('Upsampled Waveforms (Adu)')
                        plt.xlabel('Time (ns)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.legend()

                        time_delays_cfd = cfd_trig_times[a] - cfd_trig_times[b]
                        cut = numpy.logical_and(self.corr_time_shifts < time_delays_cfd + time_windows_oneside, self.corr_time_shifts >time_delays_cfd - time_windows_oneside)

                        min(self.corr_time_shifts[self.corr_time_shifts > time_delays_cfd - time_windows_oneside])

                        plt.subplot(2,1,2)
                        plt.plot(self.corr_time_shifts, corrs[pair_index],color='k',label='Cross Correlation')
                        plt.axvspan(min(self.corr_time_shifts), min(self.corr_time_shifts[self.corr_time_shifts > time_delays_cfd - time_windows_oneside]), label = 'Excluding Outside CFD Time %0.2f +- %0.1f ns'%(time_delays_cfd,time_windows_oneside),color='k',alpha=0.2)
                        plt.axvspan(max(self.corr_time_shifts[self.corr_time_shifts < time_delays_cfd + time_windows_oneside]),max(self.corr_time_shifts),color='k',alpha=0.2)
                        plt.axvline(self.corr_time_shifts[indices[pair_index]],label='Selected Time Delay = %0.3f'%self.corr_time_shifts[indices[pair_index]])
                        plt.ylabel('Cross Correlation')
                        plt.xlabel('Time (ns)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.legend()
                    import pdb; pdb.set_trace()
            elif numpy.any([align_method == 9,align_method == 11,align_method == 12]):
                '''
                For this I want to use the 'slider' method of visual alignment.
                I.e. plot the two curves, and have a slide controlling the roll
                of one of the waveforms. Once satisfied with the roll, lock it down.
                '''

                if align_method == 9:
                    loop_pairs = pairs
                    indices = numpy.zeros(numpy.shape(corrs)[0],dtype=int)
                    max_corrs = numpy.zeros(numpy.shape(corrs)[0])
                elif align_method == 11:
                    loop_pairs = self.vpol_pairs
                    indices = numpy.argmax(corrs,axis=1)
                    max_corrs = numpy.max(corrs,axis=1)
                elif align_method == 12:
                    loop_pairs = self.hpol_pairs
                    indices = numpy.argmax(corrs,axis=1)
                    max_corrs = numpy.max(corrs,axis=1)


                time_delays = numpy.zeros(len(pairs))
                display_half_time_window = 1000
                display_half_time_window_index = int(display_half_time_window/self.dt_ns_upsampled) #In indices
                slider_half_time_window = 500
                slider_half_time_window_index = int(slider_half_time_window/self.dt_ns_upsampled) #In indices

                t = numpy.arange(upsampled_waveforms.shape[1])*self.dt_ns_upsampled
                for pair_index, pair in enumerate(pairs):
                    if pair not in loop_pairs:
                        continue
                    #self.dt_ns_upsampled
                    fig_index = len(self.persistent_object)
                    ax_index = fig_index + 1
                    fig = plt.figure(figsize=(15,5))
                    ax = plt.gca()
                    #fig, ax = plt.subplots(figsize=(15,5))

                    self.persistent_object.append(fig)
                    self.persistent_object.append(ax)
                    self.persistent_object[ax_index].margins(x=0)

                    plt.subplots_adjust(bottom=0.25)
                    plt.plot(t, upsampled_waveforms[pair[0]]/max(upsampled_waveforms[pair[0]]), c='k', alpha=0.8, lw=2,label = 'Ant %i'%pair[0])

                    start_roll = numpy.argmax(corrs[pair_index])-len(t)

                    plot, = plt.plot(t + start_roll*self.dt_ns_upsampled, upsampled_waveforms[pair[1]]/max(upsampled_waveforms[pair[1]]), c='r', alpha=0.5, lw=2,label = 'Ant %i'%pair[1])
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.xlabel('t (ns)')
                    plt.ylabel('Normalized Amplitude')
                    plt.xlim(t[numpy.argmax(upsampled_waveforms[pair[0]])] - 1.5*display_half_time_window , t[numpy.argmax(upsampled_waveforms[pair[0]])] + 2.0*display_half_time_window)
                    plt.tight_layout()
                    plt.legend()
                    ax_roll = plt.axes([0.25, 0.05, 0.65, 0.03])

                    slider_roll = Slider(ax_roll, 'Roll Ant %i'%pair[1], max(-upsampled_waveforms.shape[1],start_roll-slider_half_time_window_index), min(start_roll+slider_half_time_window_index,upsampled_waveforms.shape[1]), valinit=start_roll, valstep=1.0)


                    def update(val):
                        roll = slider_roll.val
                        plot.set_xdata(t + roll*self.dt_ns_upsampled)

                        selected_index = int(len(t) + slider_roll.val)
                        search_window_cut = numpy.logical_and(self.corr_time_shifts > (self.corr_time_shifts[selected_index] - align_method_10_window_ns),  self.corr_time_shifts < (self.corr_time_shifts[selected_index] + align_method_10_window_ns) )
                        search_window_indices = numpy.where(search_window_cut)[0]
                        index = search_window_indices[numpy.argmax(corrs[pair_index][search_window_cut])]
                        plt.xlabel('t (ns)  Snapped dt=%0.3f ns'%(self.corr_time_shifts[index]))

                        self.persistent_object[fig_index].canvas.draw_idle()


                    slider_roll.on_changed(update)

                    input('If satisfied with current slider location, press Enter to lock it down.')
                    plt.close(self.persistent_object[fig_index])
                    selected_index = int(len(t) + slider_roll.val)


                    search_window_cut = numpy.logical_and(self.corr_time_shifts > (self.corr_time_shifts[selected_index] - align_method_10_window_ns),  self.corr_time_shifts < (self.corr_time_shifts[selected_index] + align_method_10_window_ns) )
                    search_window_indices = numpy.where(search_window_cut)[0]

                    indices[pair_index] = search_window_indices[numpy.argmax(corrs[pair_index][search_window_cut])]
                    max_corrs[pair_index] = numpy.max(corrs[pair_index][search_window_cut])              
                    print('int of %i chosen, snapping to time delay of %0.3f ns\nCorresponding correlation value of %0.3f (max = %0.3f)'%(slider_roll.val,self.corr_time_shifts[indices[pair_index]], max_corrs[pair_index], max(corrs[pair_index])))

            elif align_method == 10:
                '''
                Requires an expected time delay that is used to snap to the closest peak to that.  
                '''
                # print(align_method_10_estimate)
                # print(align_method_10_window_ns)
                if align_method_10_estimate is None:
                    print('NEED TO GIVE VALUE FOR align_method_10_estimate IF USING aligned_method = 10.  Failing.')
                else:
                    #align_method_10_estimate must be have the same number of elements as max_corrs has rows.

                    indices = numpy.zeros(numpy.shape(corrs)[0],dtype=int)
                    max_corrs = numpy.zeros(numpy.shape(corrs)[0])
                    for pair_index, pair in enumerate(pairs):
                        search_window_cut = numpy.logical_and(self.corr_time_shifts > (align_method_10_estimate[pair_index] - align_method_10_window_ns),  self.corr_time_shifts < (align_method_10_estimate[pair_index] + align_method_10_window_ns) )
                        search_window_indices = numpy.where(search_window_cut)[0]

                        indices[pair_index] = search_window_indices[numpy.argmax(corrs[pair_index][search_window_cut])]
                        max_corrs[pair_index] = numpy.max(corrs[pair_index][search_window_cut])


            elif align_method == 13:
                indices = numpy.argmax(corrs,axis=1)
                max_corrs = numpy.max(corrs,axis=1)
                
                indices = numpy.zeros((numpy.shape(corrs)[0],align_method_13_n))
                time_shifts = numpy.zeros((numpy.shape(corrs)[0],align_method_13_n))
                max_corrs = numpy.zeros((numpy.shape(corrs)[0],align_method_13_n))
                for corr_index, corr in enumerate(corrs):
                    prom = 0.5
                    peaks, properties = scipy.signal.find_peaks(corr,height=0.25*max(corr),distance=10.0/self.dt_ns_upsampled,prominence=prom)
                    while len(peaks) < align_method_13_n:
                        prom *= 0.8
                        peaks, properties = scipy.signal.find_peaks(corr,height=0.25*max(corr),distance=10.0/self.dt_ns_upsampled,prominence=prom)

                    peak_heights = properties['peak_heights']
                    max_indices = peaks[numpy.argsort(peak_heights)[::-1][0:align_method_13_n]]
                    indices[corr_index][:] = max_indices
                    time_shifts[corr_index][:] = self.corr_time_shifts[max_indices]
                    max_corrs[corr_index][:] = peak_heights[0:align_method_13_n]

            if align_method != 13:
                if False:
                    if True:#numpy.any(self.corr_time_shifts[indices] > 2000):
                        plt.figure()
                        for channel, wf in enumerate(upsampled_waveforms):
                            plt.plot(self.dt_ns_upsampled*numpy.arange(len(wf)),wf,label=str(channel))
                        plt.legend()
                        plt.figure()
                        for channel, wf in enumerate(corrs):
                            plt.plot(self.corr_time_shifts,wf,label=str(channel))
                        plt.legend()
                        plt.figure()
                        try:
                            for channel, wf in enumerate(corr_hilbert):
                                plt.plot(self.corr_time_shifts,wf,label=str(channel))
                            plt.legend()
                        except:
                            corr_hilbert = numpy.abs(scipy.signal.hilbert(corrs,axis=1))
                            for channel, wf in enumerate(corr_hilbert):
                                plt.plot(self.corr_time_shifts,wf,label=str(channel))
                            plt.legend()
                        import pdb; pdb.set_trace()
                if return_full_corrs == True:
                    return indices, self.corr_time_shifts[indices], max_corrs, pairs, corrs
                else:
                    return indices, self.corr_time_shifts[indices], max_corrs, pairs
            else:
                if return_full_corrs == True:
                    return indices, time_shifts, max_corrs, pairs, corrs
                else:
                    return indices, time_shifts, max_corrs, pairs
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateTimeDelaysFromEvent(self, eventid, return_full_corrs=False, align_method=0, hilbert=False, align_method_10_estimate=None, align_method_10_window_ns=8, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0,sine_subtract=False, crosspol_delays=False):
        '''
        If crosspol_delays == False then it will perform this calculation for each baseline pair, and return the result.
        If crosspol_delays == True then this will calculate the cross correlation between hpol and vpol for the given 
        set of values.

        Align method can be one of a few:
        0: argmax of corrs (default)
        1: argmax of hilbert of corrs
        2: Average of argmin and argmax
        3: argmin of corrs
        4: Pick the largest max peak preceding the max of the hilbert of corrs
        5: Pick the average indices of values > 95% peak height in corrs
        6: Pick the average indices of values > 98% peak height in hilbert of corrs
        7: Gets argmax of abs(corrs) and then finds highest positive peak before this value

        'max_corrs' corresponds to the value of the selected methods peak.
        '''
        try:
            ffts, upsampled_waveforms = self.loadFilteredFFTs(eventid,load_upsampled_waveforms=True,hilbert=hilbert, shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading,sine_subtract=sine_subtract)
            

            return self.calculateTimeDelays(ffts, upsampled_waveforms, return_full_corrs=return_full_corrs, align_method=align_method, print_warning=False, align_method_10_estimate=align_method_10_estimate, align_method_10_window_ns=align_method_10_window_ns, crosspol_delays=crosspol_delays)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateMultipleTimeDelays(self, eventids, align_method=None, hilbert=False, plot=False, hpol_cut=None, vpol_cut=None, colors=None, align_method_10_estimates=None, align_method_10_window_ns=8, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0,sine_subtract=False, crosspol_delays=False):
        '''
        If crosspol_delays == False then it will perform this calculation for each baseline pair, and return the result.
        If crosspol_delays == True then this will calculate the cross correlation between hpol and vpol for the given 
        set of values.

        If colors is some set of values that matches len(eventids) then they will be used to color the event curves.
        '''
        try:
            if ~numpy.all(eventids == numpy.sort(eventids)):
                print('eventids NOT SORTED, WOULD BE FASTER IF SORTED.')
            timeshifts = []
            max_corrs = []
            print('Calculating time delays:')
            
            if crosspol_delays == False:
                pairs = self.pairs
            else:
                pairs = self.crosspol_pairs

            if plot == True:
                print('Warning!  This likely will run out of ram and fail. ')
                figs = []
                axs = []
                if colors is not None:
                    if len(colors) == len(eventids):
                        norm = plt.Normalize()
                        norm_colors = plt.cm.coolwarm(norm(colors))#plt.cm.viridis(norm(colors))
                    else:
                        norm = None
                else:
                    norm = None
                for pair in pairs:
                    fig = plt.figure()
                    plt.suptitle(str(self.reader.run))
                    fig.canvas.set_window_title('%s%i-%s%i x-corr'%(['H','V'][pair[0]%2],pair[0]//2,['H','V'][pair[1]%2],pair[1]//2))
                    plt.ylabel('%s x-corr'%str(pair))
                    plt.xlabel('Time Delay (ns)')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                    figs.append(fig)
                    axs.append(plt.gca())
            one_percent_event = max(1,int(len(eventids)/100))

            if align_method_10_estimates is None:
                align_method_10_estimates = [None]*len(eventids)
            elif len(numpy.shape(align_method_10_estimates)) == 1:
                if len(align_method_10_estimates) == 6:
                    align_method_10_estimates = numpy.tile(align_method_10_estimates,(len(eventids),2)) #Giving same estimates for both hpol and vpol for all events
                elif len(align_method_10_estimates) == 12:
                    align_method_10_estimates = numpy.tile(align_method_10_estimates,(len(eventids),1)) #Giving same estimates for for all events

            for event_index, eventid in enumerate(eventids):
                if align_method == 9:
                    sys.stdout.write('(%i/%i)\t\t\t\n'%(event_index+1,len(eventids)))
                elif event_index%one_percent_event == 0:
                    sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
                sys.stdout.flush()
                if align_method is None:
                    if plot == True:
                        indices, time_shift, corr_value, _pairs, corrs = self.calculateTimeDelaysFromEvent(eventid,hilbert=hilbert,return_full_corrs=True,align_method_10_estimate=align_method_10_estimates[event_index],align_method_10_window_ns=align_method_10_window_ns,shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading,sine_subtract=sine_subtract, crosspol_delays=crosspol_delays) #Using default of the other function
                    else:
                        indices, time_shift, corr_value, _pairs = self.calculateTimeDelaysFromEvent(eventid,hilbert=hilbert,return_full_corrs=False,align_method_10_estimate=align_method_10_estimates[event_index],align_method_10_window_ns=align_method_10_window_ns,shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading,sine_subtract=sine_subtract, crosspol_delays=crosspol_delays) #Using default of the other function
                else:
                    if plot == True:
                        indices, time_shift, corr_value, _pairs, corrs = self.calculateTimeDelaysFromEvent(eventid,align_method=align_method,hilbert=hilbert,return_full_corrs=True,align_method_10_estimate=align_method_10_estimates[event_index],align_method_10_window_ns=align_method_10_window_ns,shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading,sine_subtract=sine_subtract, crosspol_delays=crosspol_delays)
                    else:
                        indices, time_shift, corr_value, _pairs = self.calculateTimeDelaysFromEvent(eventid,align_method=align_method,hilbert=hilbert,return_full_corrs=False,align_method_10_estimate=align_method_10_estimates[event_index],align_method_10_window_ns=align_method_10_window_ns,shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading,sine_subtract=sine_subtract, crosspol_delays=crosspol_delays)
                

                timeshifts.append(time_shift)
                max_corrs.append(corr_value)
                if plot == True:

                    for index, pair in enumerate(pairs):
                        #plotting less points by plotting subset of data
                        time_mid = self.corr_time_shifts[numpy.argmax(corrs[index])]
                        
                        half_window = 200 #ns
                        index_cut_min = numpy.max(numpy.append(numpy.where(self.corr_time_shifts < time_mid - half_window)[0],0))
                        index_cut_max = numpy.min(numpy.append(numpy.where(self.corr_time_shifts > time_mid + half_window)[0],len(self.corr_time_shifts) - 1))
                        '''
                        index_loc = numpy.where(corrs[index] > 0.2*max(corrs[index]))[0] #Indices of values greater than some threshold
                        index_cut_min = numpy.min(index_loc)
                        index_cut_max = numpy.max(index_loc)
                        '''
                        if (hpol_cut is not None) and numpy.all([len(hpol_cut) == len(eventids),hpol_cut[event_index], pair[0]%2 == 0]):
                            axs[index].plot(self.corr_time_shifts[index_cut_min:index_cut_max], corrs[index][index_cut_min:index_cut_max],alpha=0.8)
                            if abs(time_mid > 300):
                                fig = plt.figure()
                                plt.suptitle(str(eventid))
                                plt.subplot(2,1,1)
                                plt.plot(self.corr_time_shifts,corrs[index])
                                plt.axvline(time_mid,c='r',linestyle='--')
                                plt.subplot(2,1,2)
                                plt.plot(self.t(),self.wf(int(pair[0]),apply_filter=True,hilbert=False),label='%s, sine_subtract = %i'%(str(pair[0]),sine_subtract))
                                plt.plot(self.t(),self.wf(int(pair[1]),apply_filter=True,hilbert=False),label='%s, sine_subtract = %i'%(str(pair[1]),sine_subtract))
                                plt.legend()
                                import pdb; pdb.set_trace()
                                plt.close(fig)
                        elif (vpol_cut is not None) and numpy.all([len(vpol_cut) == len(eventids),vpol_cut[event_index], pair[0]%2 != 0]):
                            axs[index].plot(self.corr_time_shifts[index_cut_min:index_cut_max], corrs[index][index_cut_min:index_cut_max],alpha=0.8)
                            if abs(time_mid > 300):
                                fig = plt.figure()
                                plt.suptitle(str(eventid))
                                plt.subplot(2,1,1)
                                plt.plot(self.corr_time_shifts,corrs[index])
                                plt.axvline(time_mid,c='r',linestyle='--')
                                plt.subplot(2,1,2)
                                plt.plot(self.t(),self.wf(int(pair[0]),apply_filter=True,hilbert=False),label='%s, sine_subtract = %i'%(str(pair[0]),sine_subtract))
                                plt.plot(self.t(),self.wf(int(pair[1]),apply_filter=True,hilbert=False),label='%s, sine_subtract = %i'%(str(pair[1]),sine_subtract))
                                plt.legend()
                                import pdb; pdb.set_trace()
                                plt.close(fig)
                        else:
                            if norm is not None:
                                axs[index].plot(self.corr_time_shifts[index_cut_min:index_cut_max], corrs[index][index_cut_min:index_cut_max],alpha=0.8,c=norm_colors[event_index],label=colors[event_index])
                            else:
                                axs[index].plot(self.corr_time_shifts[index_cut_min:index_cut_max], corrs[index][index_cut_min:index_cut_max],alpha=0.8)
                            # indices, self.corr_time_shifts[indices], max_corrs, pairs, corrs
                            # if norm is not None:
                            #     axs[index].plot(self.corr_time_shifts, corrs[index],alpha=0.8,c=norm_colors[event_index],label=colors[event_index])
                            # else:
                            #     axs[index].plot(self.corr_time_shifts, corrs[index],alpha=0.8)
                            

                            axs[index].axvline(time_shift[index],linewidth=0.5,c='k',label='Max Value at t = %0.2f'%(time_shift[index]))


            # if plot == True:
            #     for ax in axs:
            #         ax.legend()
            sys.stdout.write('\n')
            sys.stdout.flush()
            timeshifts = numpy.array(timeshifts)
            max_corrs = numpy.array(max_corrs)


            if len(numpy.shape(timeshifts)) == 3:
                timeshifts = numpy.transpose(timeshifts,axes=(1,0,2))
                max_corrs = numpy.transpose(max_corrs,axes=(1,0,2))
            else:
                timeshifts = numpy.transpose(timeshifts)
                max_corrs = numpy.transpose(max_corrs)
            return timeshifts, max_corrs, pairs
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateImpulsivityFromTimeDelays(self, eventid, time_delays, upsampled_waveforms=None,return_full_corrs=False, align_method=0, hilbert=False,plot=False,impulsivity_window=400,sine_subtract=False):
        '''
        This will call calculateTimeDelaysFromEvent with the given settings, and use these to calculate impulsivity. 

        If calculate_impulsivity == True then this will also use the upsampled waveforms to determine impulsivity.
        If hilbert==True then the waveforms used for impulsivity will be recalculated to NOT be enveloped.  

        Impulsivity_window is given in ns and says how wide the window around the peak of the hilbert envelope (not necessarily symmetric)
        to sum the envelope for the metric.

        Time delays should mean how much to roll each signal, with sign indicating forward or backward in time.
        '''
        try:
            if upsampled_waveforms is None:
                ffts, upsampled_waveforms = self.loadFilteredFFTs(eventid,load_upsampled_waveforms=True,hilbert=hilbert,sine_subtract=sine_subtract)
                #Else use the given waveforms, assumed to have sampling of self.dt_ns_upsampled

            rolls = (time_delays/self.dt_ns_upsampled).astype(int)
            #[0-1H,0-2H,0-3H,1-2H,1-3H,2-3H,0-1V,0-2V,0-3V,1-2V,1-3V,2-3V]
            #rolls[0:3] = [0-1H,0-2H,0-3H]
            #rolls[6:9] = [0-1V,0-2V,0-3V]
            hpols_rolls = numpy.append(numpy.array([0]),rolls[0:3])#How much to roll hpol
            vpols_rolls = numpy.append(numpy.array([0]),rolls[6:9])#How much to roll vpol
            times = self.dt_ns_upsampled*numpy.arange(numpy.shape(upsampled_waveforms)[1])

            if plot == True:
                plt.figure()
                for channel_index, waveform in enumerate(upsampled_waveforms):
                    plt.subplot(2,2,channel_index%2+1)
                    plt.plot(times,waveform,label='ch%i'%channel_index)
                    if channel_index%2 == 0:
                        pol='hpol'
                    else:
                        pol='vpol'
                    plt.ylabel('%s Upsampled Waveforms (adu)'%pol)
                    plt.xlabel('t (ns)')
                    plt.legend()

            #import pdb; pdb.set_trace()
            for channel_index in range(8):
                if channel_index%2 == 1:
                    roll = vpols_rolls[channel_index//2]
                else:
                    roll = hpols_rolls[channel_index//2]

                upsampled_waveforms[channel_index] = numpy.roll(upsampled_waveforms[channel_index],-roll)

            if plot == True:
                for channel_index, waveform in enumerate(upsampled_waveforms):
                    if channel_index%2 == 0:
                        pol='hpol'
                    else:
                        pol='vpol'
                    plt.subplot(2,2,3+channel_index%2)
                    plt.plot(times,waveform,label='ch%i'%channel_index)
                    plt.ylabel('%s Rolled Upsampled Waveforms (adu)'%pol)
                    plt.xlabel('t (ns)')
                    plt.legend()

            summed_hpol_waveforms = numpy.sum(upsampled_waveforms[::2],axis=0)
            hilbert_summed_hpol_waveforms = numpy.abs(scipy.signal.hilbert(summed_hpol_waveforms))
            sorted_hilbert_summed_hpol_waveforms = hilbert_summed_hpol_waveforms[numpy.argsort(abs(numpy.arange(len(hilbert_summed_hpol_waveforms)) - numpy.argmax(hilbert_summed_hpol_waveforms)))]
            
            impulsivity_window_cut = times < impulsivity_window
            unscaled_hpol_impulsivity = numpy.cumsum(sorted_hilbert_summed_hpol_waveforms[impulsivity_window_cut])/sum(sorted_hilbert_summed_hpol_waveforms[impulsivity_window_cut])
            impulsivity_hpol = 2*numpy.mean(unscaled_hpol_impulsivity) - 1

            summed_vpol_waveforms = numpy.sum(upsampled_waveforms[1::2],axis=0)
            hilbert_summed_vpol_waveforms = numpy.abs(scipy.signal.hilbert(summed_vpol_waveforms))
            sorted_hilbert_summed_vpol_waveforms = hilbert_summed_vpol_waveforms[numpy.argsort(abs(numpy.arange(len(hilbert_summed_vpol_waveforms)) - numpy.argmax(hilbert_summed_vpol_waveforms)))]
            
            impulsivity_window_cut = times < impulsivity_window
            unscaled_vpol_impulsivity = numpy.cumsum(sorted_hilbert_summed_vpol_waveforms[impulsivity_window_cut])/sum(sorted_hilbert_summed_vpol_waveforms[impulsivity_window_cut])
            impulsivity_vpol = 2*numpy.mean(unscaled_vpol_impulsivity) - 1


            if plot == True:
                plt.figure()
                plt.subplot(3,1,1)
                plt.plot(times,summed_hpol_waveforms,label='Summed Aligned Hpol Waveforms',alpha=0.8)
                plt.plot(times,hilbert_summed_hpol_waveforms,linestyle='--',label='Hpol Hilbert',alpha=0.8)

                plt.plot(times,summed_vpol_waveforms,label='Summed Aligned Vpol Waveforms',alpha=0.8)
                plt.plot(times,hilbert_summed_vpol_waveforms,linestyle='--',label='Vpol Hilbert',alpha=0.8)

                plt.legend(loc='upper right')
                plt.ylabel('Aligned and\nSummed Signal')
                plt.xlabel('t (ns)')
                plt.subplot(3,1,2)
                plt.plot(times,sorted_hilbert_summed_hpol_waveforms,label='Hpol Aligned and Summed Hilbert Sorted in Time by Proximity to Peak',alpha=0.8)
                plt.plot(times,sorted_hilbert_summed_vpol_waveforms,label='Vpol Aligned and Summed Hilbert Sorted in Time by Proximity to Peak',alpha=0.8)
                plt.ylabel('Aligned and\nSummed Signal')
                plt.xlabel('Window Width (Sorted Out from Max Value) (ns)')
                plt.legend(loc = 'upper right')
                plt.subplot(3,1,3)
                
                plt.plot(times[impulsivity_window_cut], unscaled_hpol_impulsivity,label='Normalized Cumulative Sum of Hilbert Envelope',alpha=0.8)
                plt.axhline(numpy.mean(unscaled_hpol_impulsivity),color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],linestyle='--',label='Mean (A) = %f, Impulsivity (2A+1) = %f'%(numpy.mean(unscaled_hpol_impulsivity),impulsivity_hpol))

                plt.plot(times[impulsivity_window_cut], unscaled_vpol_impulsivity,label='Normalized Cumulative Sum of Hilbert Envelope',alpha=0.8)
                plt.axhline(numpy.mean(unscaled_vpol_impulsivity),color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1],linestyle='--',label='Mean (A) = %f, Impulsivity (2A+1) = %f'%(numpy.mean(unscaled_vpol_impulsivity),impulsivity_vpol))

                plt.xlabel('Window Width (Sorted Out from Max Value) (ns)')
                plt.ylabel('Cumulative Sum')
                plt.legend(loc = 'lower right')

            return impulsivity_hpol, impulsivity_vpol

            #return self.calculateTimeDelays(ffts, upsampled_waveforms, return_full_corrs=return_full_corrs, align_method=align_method, print_warning=False)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)



    def calculateImpulsivityFromEvent(self, eventid, return_full_corrs=False, align_method=0, hilbert=False,plot=False,impulsivity_window=750,sine_subtract=False):
        '''
        This will call calculateTimeDelaysFromEvent with the given settings, and use these to calculate impulsivity. 

        If calculate_impulsivity == True then this will also use the upsampled waveforms to determine impulsivity.
        If hilbert==True then the waveforms used for impulsivity will be recalculated to NOT be enveloped.  

        Impulsivity_window is given in ns and says how wide the window around the peak of the hilbert envelope (not necessarily symmetric)
        to sum the envelope for the metric.
        '''
        try:
            ffts, upsampled_waveforms = self.loadFilteredFFTs(eventid,load_upsampled_waveforms=True,hilbert=hilbert,sine_subtract=sine_subtract)
            #self.dt_ns_upsampled

            if return_full_corrs == True:
                indices, corr_time_shifts, max_corrs, pairs, corrs = self.calculateTimeDelays(ffts, upsampled_waveforms, return_full_corrs=return_full_corrs, align_method=align_method, print_warning=False)
            else:
                indices, corr_time_shifts, max_corrs, pairs = self.calculateTimeDelays(ffts, upsampled_waveforms, return_full_corrs=return_full_corrs, align_method=align_method, print_warning=False)
            
            time_delays = -corr_time_shifts
            print('From event')
            print(time_delays)
            return self.calculateImpulsivityFromTimeDelays(eventid, time_delays, upsampled_waveforms=upsampled_waveforms,return_full_corrs=return_full_corrs, align_method=align_method, hilbert=hilbert,plot=plot,impulsivity_window=impulsivity_window,sine_subtract=sine_subtract)

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

class TemplateCompareTool(FFTPrepper):
    '''
    Takes a run reader and does the math required for cross correlations.  The user can then
    set a template (templates, one for each antenna) for which to later correlate other events
    against.  The template can be set to a particular event by passing the eventid, or can be
    set manually if a different template is to be used.

    Parameters
    ----------
    reader : examples.beacon_data_reader.reader
        The run reader you wish to examine time delays for.
    final_corr_length : int
        Should be given as a power of 2.  This is the goal length of the cross correlations, and can set the time resolution
        of the time delays.
    crit_freq_low_pass_MHz : float
        Sets the critical frequency of the low pass filter to be applied to the data.
    crit_freq_high_pass_MHz
        Sets the critical frequency of the high pass filter to be applied to the data.
    low_pass_filter_order
        Sets the order of the low pass filter to be applied to the data.
    high_pass_filter_order
        Sets the order of the high pass filter to be applied to the data.
    
    See Also
    --------
    examples.beacon_data_reader.reader
    '''
    def __init__(self, reader, final_corr_length=2**15, crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, waveform_index_range=(None,None), plot_filters=False, initial_template_id=None,apply_phase_response=False,sine_subtract=False):
        try:
            super().__init__(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range, plot_filters=plot_filters,apply_phase_response=apply_phase_response)
            self.corr_index_to_delay_index = -numpy.arange(-(self.final_corr_length-1)//2,(self.final_corr_length-1)//2 + 1) #Negative because with how it is programmed you would want to roll the template the normal amount, but I will be rolling the waveforms.
            self.setTemplateToEvent(initial_template_id,sine_subtract=sine_subtract)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def setTemplateToEvent(self,eventid,sine_subtract=False, hilbert=False, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0,shorten_keep_leading=100):
        '''
        Sets the template to the waveforms from eventid.
        '''
        try:
            if eventid is not None:
                print('Event %i set as template'%eventid)
                pass
            else:
                print('None given as eventid, setting event 0 as template.')
                eventid = 0
            self.template_eventid = eventid
            self.template_ffts_filtered,self.template_waveform_upsampled_filtered = self.loadFilteredFFTs(eventid,load_upsampled_waveforms=True,sine_subtract=sine_subtract, hilbert=hilbert, shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length,shorten_keep_leading=shorten_keep_leading)

            #Scaled and conjugate template for one time prep of cross correlations.
            self.scaled_conj_template_ffts_filtered = (numpy.conj(self.template_ffts_filtered).T/numpy.std(numpy.conj(self.template_ffts_filtered),axis=1) ).T
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def setTemplateToCustom(self, template_ffts_filtered):
        '''
        Sets the template to the given waveforms.
        '''
        try:
            self.template_eventid = 'custom'

            self.template_ffts_filtered = template_ffts_filtered

            #Scaled and conjugate template for one time prep of cross correlations.
            self.scaled_conj_template_ffts_filtered = (numpy.conj(self.template_ffts_filtered).T/numpy.std(numpy.conj(self.template_ffts_filtered),axis=1)).T
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def crossCorrelateWithTemplate(self, eventid, channels=[0,1,2,3,4,5,6,7],load_upsampled_waveforms=False,sine_subtract=False, hilbert=False, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0,shorten_keep_leading=100):
        '''
        Performs a cross correlation between the waveforms for eventid and the
        internally defined template.
        '''
        try:
            #Load events waveforms
            if load_upsampled_waveforms:
                ffts, upsampled_waveforms = self.loadFilteredFFTs(eventid, channels=channels, load_upsampled_waveforms=load_upsampled_waveforms,sine_subtract=sine_subtract, hilbert=hilbert, shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length,shorten_keep_leading=shorten_keep_leading)
            else:
                ffts = self.loadFilteredFFTs(eventid, channels=channels, load_upsampled_waveforms=load_upsampled_waveforms,sine_subtract=sine_subtract, hilbert=hilbert, shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length,shorten_keep_leading=shorten_keep_leading)

            #Perform cross correlations with the template.
            # import pdb; pdb.set_trace()
            corrs_fft = numpy.multiply((ffts.T/numpy.std(ffts,axis=1)).T,(self.scaled_conj_template_ffts_filtered[channels])) / (len(self.waveform_times_corr)//2 + 1)
            corrs = numpy.fft.fftshift(numpy.fft.irfft(corrs_fft,axis=1,n=self.final_corr_length),axes=1) * (self.final_corr_length//2 ) #Upsampling and keeping scale might be an off by 1 in scaling.  Should be small effect.

            if load_upsampled_waveforms:
                return corrs, corrs_fft, upsampled_waveforms
            else:
                return corrs, corrs_fft
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def alignToTemplate(self, eventid, channels=[0,1,2,3,4,5,6,7],align_method=0,sine_subtract=False, hilbert=False, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0,shorten_keep_leading=100, return_delays=False):
        '''
        Attempts to align the waveforms from eventid with the internally
        defined template.  This may use one of several methods of aligning
        the waveforms, as specified by align_method.

        Note the waveforms that are returned are upsampled to a factor of 2
        but are not the double length ones that are used for the correlation.
        '''
        try:
            corrs, corrs_fft, upsampled_waveforms = self.crossCorrelateWithTemplate(eventid,channels=channels,load_upsampled_waveforms=True,sine_subtract=sine_subtract, hilbert=hilbert, shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length,shorten_keep_leading=shorten_keep_leading)

            if align_method == 0:
                #Align to maximum correlation value.
                index_delays = numpy.argmax(corrs,axis=1) #These are the indices within corr that are max.
                max_corrs = []
                for index, corr in enumerate(corrs):
                    max_corrs.append(corr[index_delays[index]])
                max_corrs = numpy.array(max_corrs)
            elif align_method == 1:
                #Looks for best alignment within window after cfd trigger, cfd applied on hilber envelope.
                corr_hilbert = numpy.abs(scipy.signal.hilbert(corrs,axis=1))
                index_delays = []
                max_corrs = []
                for index, hwf in enumerate(corr_hilbert):
                    cfd_indices = numpy.where(hwf/numpy.max(hwf) > 0.5)[0]
                    #cfd_indices = cfd_indices[0:int(0.50*len(cfd_indices))] #Top 50% close past 50% max rise 
                    # The above line is not workinggggggg
                    index_delays.append(cfd_indices[numpy.argmax(corrs[index][cfd_indices])])
                    max_corrs.append(corrs[index][index_delays[-1]])

                index_delays = numpy.array(index_delays) #The indices that have been selected as best alignment.  I.e. how much to roll each wf.  
                max_corrs = numpy.array(max_corrs)

            elif align_method == 2:
                #Of the peaks in the correlation plot within range of the max peak, choose the peak with the minimum index.
                index_delays = []
                max_corrs = []

                for corr in corrs:
                    index_delays.append(min(scipy.signal.find_peaks(corr,height = 0.8*max(corr),distance=int(0.05*len(corr)))[0]))
                    max_corrs.append(corr[index_delays[-1]])

                index_delays = numpy.array(index_delays)
                max_corrs = numpy.array(max_corrs)

            else:
                print('Given align_method invalid.  This will probably break.')

            #With the indices determine relevant information and roll wf.
            
            rolled_wfs = numpy.zeros_like(upsampled_waveforms)
            for index, wf in enumerate(upsampled_waveforms):
                rolled_wfs[index] = numpy.roll(wf,self.corr_index_to_delay_index[index_delays[index]])
            
            if return_delays == False:
                return max_corrs, upsampled_waveforms, rolled_wfs
            else:
                return max_corrs, upsampled_waveforms, rolled_wfs, index_delays, self.corr_time_shifts[index_delays]
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def averageAlignedSignalsPerChannel(self, eventids, align_method=0, template_eventid=None, plot=False,event_type=None,sine_subtract=None):
        '''
        This will use alignToTemplate to align an events within a channel to the current set template.
        This means that you should set the template with either setTemplateToEvent or setTemplateToCustom
        in advance.  template_eventid will call setTemplateToEvent if given.  align_method is passed to
        alignToTemplate to align signals within the channel.

        Note that the template is stored as a conjugated fft cross-correlation ready version of the template.
        '''
        try:
            if template_eventid is not None:
                self.setTemplateToEvent(template_eventid,sine_subtract=sine_subtract)

            averaged_waveforms = numpy.zeros((8,self.final_corr_length//2))
            times = numpy.arange(self.final_corr_length//2)*self.dt_ns_upsampled
            if plot == True:
                if event_type == 'hpol':
                    channels = numpy.arange(4)*2
                elif event_type == 'vpol':
                    channels = numpy.arange(4)*2 + 1
                else:
                    channels = numpy.arange(8)
                figs = []
                axs = []
                for channel in range(8):
                    if numpy.isin(channel,channels):
                        fig = plt.figure()
                        fig.canvas.set_window_title('Ch %i Aligned Waveforms'%channel)
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.xlabel('t (ns)')
                        plt.ylabel('Ch %i Amp (Adu)'%channel)
                        figs.append(fig)
                        axs.append(plt.gca())
                    else:
                        figs.append(None)
                        axs.append(None)

            for event_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
                max_corrs, upsampled_waveforms, rolled_wfs = self.alignToTemplate(eventid, align_method=align_method)
                averaged_waveforms += rolled_wfs/len(eventids)

                if plot == True:
                    for channel in channels:
                        ax = axs[channel]
                        ax.plot(times, rolled_wfs[channel],alpha=0.2)

            if plot == True:
                #Repeat for template and plot on top. 
                max_corrs, upsampled_waveforms, rolled_wfs = self.alignToTemplate(template_eventid, align_method=align_method)

                for channel in channels:
                    ax = axs[channel]
                    ax.plot(times, rolled_wfs[channel],linestyle='--',c='b',label=str(channel)+' template')
                    ax.legend()

                for channel in channels:
                    ax = axs[channel]
                    ax.plot(times, averaged_waveforms[channel],linestyle='--',c='r',label=str(channel)+' avg')
                    ax.legend()

                fig = plt.figure()
                fig.canvas.set_window_title('Average Waveforms')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.xlabel('t (ns)')
                plt.ylabel('Adu')
                for channel in channels:
                    plt.plot(averaged_waveforms[channel],alpha=0.7,label=str(channel))
                plt.legend()

            return times, averaged_waveforms
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def returnTemplateTimeDelays(self):
        '''
        Using the template FFTs as they exist, this will return the time delays using the simplest method.
        '''
        try:
            #Handle scenario where waveform has all zeros (which results in 0 SNR which messes up FFT/correlation)
            bad_channels = numpy.where(numpy.std(self.template_ffts_filtered,axis=1) == 0)[0] 
            bad_pairs = numpy.any(numpy.isin(self.pairs,bad_channels),axis=1) #Calculation will proceed as normal (nan's will exist), but the output of these channel will be overwritten.

            corrs_fft = numpy.multiply((self.template_ffts_filtered[self.pairs[:,0]].T/numpy.std(self.template_ffts_filtered[self.pairs[:,0]],axis=1)).T,(numpy.conj(self.template_ffts_filtered[self.pairs[:,1]]).T/numpy.std(numpy.conj(self.template_ffts_filtered[self.pairs[:,1]]),axis=1)).T) / (len(self.waveform_times_corr)//2 + 1)
            corrs = numpy.fft.fftshift(numpy.fft.irfft(corrs_fft,axis=1,n=self.final_corr_length),axes=1) * (self.final_corr_length//2) #Upsampling and keeping scale
            corrs[bad_pairs] = 0.0

            indices = numpy.argmax(corrs,axis=1)
            max_corrs = numpy.max(corrs,axis=1)
            return indices, self.corr_time_shifts[indices], max_corrs, self.pairs, corrs
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    # def perChannelTimeDelays(self, eventids, channels=[0,1,2,3,4,5,6,7] align_method=0, template_eventid=None, plot=False,event_type=None,sine_subtract=None):
    #     '''
    #     This will align each eventids waveforms to the waveform set as the template.  The output will then be the
    #     measured time delays between each waveform and the template within the given channels.  This does not do
    #     any comparison between antennas and is thus not baseline dependant. 

    #     Note that the template is stored as a conjugated fft cross-correlation ready version of the template.
    #     '''
    #     try:
    #         if template_eventid is not None:
    #             self.setTemplateToEvent(template_eventid,sine_subtract=sine_subtract)

    #         averaged_waveforms = numpy.zeros((len(channels),self.final_corr_length//2))
    #         times = numpy.arange(self.final_corr_length//2)*self.dt_ns_upsampled

    #         for event_index, eventid in enumerate(eventids):
    #             sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
    #             max_corrs, upsampled_waveforms, rolled_wfs, index_delays, time_shifts = self.alignToTemplate(eventid, channels=[0,1,2,3,4,5,6,7],align_method=align_method,sine_subtract=sine_subtract, return_delays=True)





    #         return times, averaged_waveforms
    #     except Exception as e:
    #         print('\nError in %s'%inspect.stack()[0][3])
    #         print(e)
    #         exc_type, exc_obj, exc_tb = sys.exc_info()
    #         fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #         print(exc_type, fname, exc_tb.tb_lineno)
        

if __name__ == '__main__':
    try:
        plt.close('all')
        datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
        shorten_signals = False
        shorten_thresh = 0.7 #Percent of max height to trigger leading edge of signal.  Should only apply if a certain snr is met?
        #shorten_delay = 20.0 #The delay in ns after the trigger thresh to wait before an exponential decrease is applied. 
        #shorten_length = 100.0
        plot_shorten_2d = False

        delays = numpy.array([10.0])#numpy.linspace(5,70,15)#numpy.array([12.0])
        lengths = numpy.array([90.0])#numpy.linspace(20,200,20)#numpy.array([110.0])
        delays_mesh, lengths_mesh = numpy.meshgrid(delays, lengths)
        

        make_plot = False
        verbose = False
        mode = 'hpol'
        try_removing_expected_simple_reflection = False
        allowed_plots = 15
        current_n_plots = 0

        waveform_index_range = (None,None)
        apply_phase_response = True

        #Filter settings
        final_corr_length = 2**16 #Should be a factor of 2 for fastest performance (I think)
        
        crit_freq_low_pass_MHz = None#70 #This new pulser seems to peak in the region of 85 MHz or so
        low_pass_filter_order = None#12
        
        high_pass_filter_order = None#8
        crit_freq_high_pass_MHz = None#20

        sine_subtract = False

        import tools.get_plane_tracks as pt
        known_planes, calibrated_trigtime, output_tracks = pt.getKnownPlaneTracks()
        use_filter = True
        plot_filters = False
        pairs = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
        
        antennas_cm = plt.cm.get_cmap('tab10', 10)
        antenna_colors = antennas_cm(numpy.linspace(0, 1, 10))[0:4]

        for index, key in enumerate(list(known_planes.keys())):
            # if key == '1774-88800':
            #     continue
            # else:
            #     pass
            # # if index > 0:
            # #     continue
            pair_cut = numpy.array([pair in known_planes[key]['baselines'][mode] for pair in [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]] ])
            run = int(key.split('-')[0])
            reader = Reader(datapath,run)
            eventids = known_planes[key]['eventids'][:,1]
            mean_residuals = numpy.zeros_like(delays_mesh)
            n_residuals = numpy.zeros_like(delays_mesh)
            tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False,apply_phase_response=apply_phase_response)
            for shorten_delay_index, shorten_delay in enumerate(delays):
                for shorten_length_index, shorten_length in enumerate(lengths):
                    time_delay_residuals_all_events = []
                    
                    multi_out_timeshifts, multi_out_max_corrs, multi_out_pairs = tdc.calculateMultipleTimeDelays(eventids, align_method=13,hilbert=False, shorten_signals=True, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0)
                    
                    for event_index, eventid in enumerate(eventids):
                        # if event_index != 2:
                        #     continue
                        ffts, upsampled_waveforms = tdc.loadFilteredFFTs(eventid,load_upsampled_waveforms=True,hilbert=False,shorten_signals=shorten_signals,shorten_thresh=shorten_thresh,shorten_delay=shorten_delay,shorten_length=shorten_length,sine_subtract=sine_subtract)


                        indices, corr_time_shifts, max_corrs, output_pairs, corrs = tdc.calculateTimeDelays(ffts, upsampled_waveforms, return_full_corrs=True, align_method=0, print_warning=False)
                        
                        if mode == 'hpol':
                            corr_time_shifts = corr_time_shifts[0:6]
                            waveforms = upsampled_waveforms[0::2,:]
                        else:
                            corr_time_shifts = corr_time_shifts[6:12]
                            waveforms = upsampled_waveforms[1::2,:]
                        

                        times = numpy.arange(len(waveforms[0]))*tdc.dt_ns_upsampled
                        diff_times = numpy.arange(len(times)-1)+numpy.diff(times)/2.0 
                        

                        for pair_index, pair in enumerate(pairs):
                            if ~pair_cut[pair_index]:
                                continue
                            i = pair[0]
                            j = pair[1]
                            expected_time_delay_manual = known_planes[key]['time_delays'][mode][event_index][pair_index]
                            measured_corr_delay = corr_time_shifts[pair_index]

                            time_delay_residuals_all_events.append(expected_time_delay_manual - measured_corr_delay)

                            if (abs(expected_time_delay_manual - measured_corr_delay) > 5) and (current_n_plots < allowed_plots):
                                hilbert_corr = numpy.abs(scipy.signal.hilbert(corrs[pair_index]))
                                peaks = scipy.signal.find_peaks(corrs[pair_index],height=0.9*max(hilbert_corr),distance=10.0/tdc.dt_ns_upsampled,prominence=1)[0]
                                current_n_plots +=1
                                plt.figure()
                                plt.suptitle(key + ' ' + str(eventid) + ' ' + str(pair))
                                plt.subplot(3,1,1)
                                plt.plot(tdc.corr_time_shifts,corrs[pair_index])
                                plt.plot(tdc.corr_time_shifts,hilbert_corr)
                                plt.xlim(-200,200)
                                plt.axvline(expected_time_delay_manual,linestyle='--',c='k',label='Manual Time Delay = %0.3f'%expected_time_delay_manual)
                                plt.axvline(measured_corr_delay,linestyle='--',c='g',label='Max Corr Time Delay = %0.3f'%measured_corr_delay)
                                for peak in peaks:
                                    prom = scipy.signal.peak_prominences(corrs[pair_index],[peak])[0]
                                    plt.scatter(tdc.corr_time_shifts[peaks],corrs[pair_index][peaks],label='Peak Prominence = ' + str(prom[0]), c='r')
                                plt.legend()
                                plt.minorticks_on()
                                plt.grid(b=True, which='major', color='k', linestyle='-')
                                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                                plt.ylabel('Cross Corr')
                                plt.xlabel('t (ns)')


                                plt.subplot(3,1,2)
                                max_i = numpy.max(waveforms[i])
                                max_j = numpy.max(waveforms[j])
                                plt.plot(times,waveforms[i]/max_i, c = antenna_colors[i] ,label='Raw Signal Ant %i Not Rolled'%i,alpha=0.7)
                                plt.plot(times,waveforms[j]/max_j, c = antenna_colors[j] ,label='Raw Signal Ant %i Not Rolled'%(j),linewidth=0.5,linestyle='--')
                                plt.plot(times,numpy.roll(waveforms[j]/max_j,int(expected_time_delay_manual/tdc.dt_ns_upsampled)), c = antenna_colors[j] ,label='Raw Signal Ant %i Rolled %0.3f ns'%(j,expected_time_delay_manual),alpha=0.7)
                                plt.legend()
                                plt.minorticks_on()
                                plt.grid(b=True, which='major', color='k', linestyle='-')
                                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                                plt.ylabel('normalized adu')
                                plt.xlabel('t (ns)')

                                plt.subplot(3,1,3)
                                plt.plot(times,waveforms[i]/max_i, c = antenna_colors[i] ,label='Raw Signal Ant %i Not Rolled'%i,alpha=0.7)
                                plt.plot(times,waveforms[j]/max_j, c = antenna_colors[j] ,label='Raw Signal Ant %i Not Rolled'%(j),linewidth=0.5,linestyle='--')
                                plt.plot(times,numpy.roll(waveforms[j]/max_j,int(measured_corr_delay/tdc.dt_ns_upsampled)), c = antenna_colors[j] ,label='Raw Signal Ant %i Rolled %0.3f ns'%(j,measured_corr_delay),alpha=0.7)
                                plt.legend()
                                plt.minorticks_on()
                                plt.grid(b=True, which='major', color='k', linestyle='-')
                                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                                plt.ylabel('normalized adu')
                                plt.xlabel('t (ns)')



                            if try_removing_expected_simple_reflection:
                                n = 1.0003 #Index of refraction of air  #Should use https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf 
                                c = 299792458/n #m/s

                                expected_reflection_time_delay = 1e9*(1.0 + numpy.cos(numpy.deg2rad(73.0))) * 3.5814/c #~17 deg elevation for 1784-7166, 3.5814m is rough height of antenna.
                                expected_reflection_roll = int(expected_reflection_time_delay/tdc.dt_ns_upsampled)
                                mod_i = waveforms[i] + numpy.roll(waveforms[i],expected_reflection_roll)
                                mod_j = waveforms[j] + numpy.roll(waveforms[j],expected_reflection_roll)
                                cc_raw = scipy.signal.correlate(waveforms[i],waveforms[j])/(len(mod_i)*numpy.std(waveforms[i])*numpy.std(waveforms[j]))
                                cc_mod = scipy.signal.correlate(mod_i,mod_j)/(len(mod_i)*numpy.std(mod_i)*numpy.std(mod_j))

                                plt.figure()

                                plt.subplot(3,1,1)
                                plt.plot(times,waveforms[i], c = antenna_colors[i] ,label='Raw Signal Ant %i'%i)
                                plt.plot(times,waveforms[j], c = antenna_colors[j] ,label='Raw Signal Ant %i'%j)
                                plt.legend()
                                plt.ylabel('Signal')
                                plt.subplot(3,1,2)
                                plt.plot(times,mod_i, c = antenna_colors[i] ,label='Modified Signal %i'%i) #Subtracting the signal offset by time for reflection (signal would be flipped, so subtracting is adding)
                                plt.plot(times,mod_j, c = antenna_colors[j] ,label='Modified Signal %i'%j) #Subtracting the signal offset by time for reflection (signal would be flipped, so subtracting is adding)
                                plt.legend()
                                plt.ylabel('Signal')
                                plt.subplot(3,1,3)
                                plt.plot(scipy.signal.hilbert(cc_raw), c = antenna_colors[i] ,label = 'Raw Signals')
                                plt.plot(scipy.signal.hilbert(cc_mod), c = antenna_colors[j] ,label = 'Modified Signals')
                                plt.legend()
                                plt.ylabel('Cross Corr')

                    mean_residuals[shorten_length_index,shorten_delay_index] = numpy.mean(numpy.abs(time_delay_residuals_all_events))
                    n_residuals[shorten_length_index,shorten_delay_index] = numpy.sum(numpy.abs(time_delay_residuals_all_events) > 5.0)
            if plot_shorten_2d:
                fig = plt.figure()
                plt.suptitle(key)
                plt.subplot(1,2,1)
                ax = plt.gca()
                im = ax.pcolormesh(delays_mesh, lengths_mesh, mean_residuals, vmin=numpy.min(mean_residuals), vmax=numpy.max(mean_residuals),cmap=plt.cm.coolwarm)
                cbar = fig.colorbar(im)
                plt.xlabel('Delays')
                plt.ylabel('Lengths')
                cbar.set_label('Mean Residual')

                plt.subplot(1,2,2)
                ax = plt.gca()
                im = ax.pcolormesh(delays_mesh, lengths_mesh, n_residuals, vmin=numpy.min(n_residuals), vmax=numpy.max(n_residuals),cmap=plt.cm.coolwarm)
                cbar = fig.colorbar(im)
                plt.xlabel('Delays')
                plt.ylabel('Lengths')
                cbar.set_label('# Residual > 5.0 ns')

    except Exception as e:
        print('Error in FFTPrepper.__main__()')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
