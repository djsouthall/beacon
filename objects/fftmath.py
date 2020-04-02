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

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
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
plt.ion()


class FFTPrepper:
    '''
    Takes a run reader and does the math required to prepare for calculations such as
    cross correlations for time delays or template making/searching.

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
    waveform_index_range
        Tuple of values.  If the first is None then the signals will be loaded from the beginning of their default buffer,
        otherwise it will start at this index.  If the second is None then the window will go to the end of the default
        buffer, otherwise it will end in this.  

        Essentially waveforms will be loaded using wf = self.reader.wf(channel)[waveform_index_range[0]:waveform_index_range[1]]

        Bounds will be adjusted based on buffer length (in case of overflow. )
    
    See Also
    --------
    examples.beacon_data_reader.reader
    '''
    def __init__(self, reader, final_corr_length=2**17, crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, waveform_index_range=(None,None), plot_filters=False,tukey_alpha=0.1,tukey_default=True,apply_phase_response=False):
        try:
            self.reader = reader
            self.buffer_length = reader.header().buffer_length
            self.final_corr_length = final_corr_length #Should be a factor of 2 for fastest performance
            self.crit_freq_low_pass_MHz = crit_freq_low_pass_MHz
            self.crit_freq_high_pass_MHz = crit_freq_high_pass_MHz
            self.low_pass_filter_order = low_pass_filter_order
            self.high_pass_filter_order = high_pass_filter_order

            self.hpol_pairs = numpy.array(list(itertools.combinations((0,2,4,6), 2)))
            self.vpol_pairs = numpy.array(list(itertools.combinations((1,3,5,7), 2)))
            self.pairs = numpy.vstack((self.hpol_pairs,self.vpol_pairs)) 

            self.tukey_default = tukey_default #This sets the default in self.wf whether to use tukey or not. 

            self.persistent_object = [] #For any objects that need to be kept alive/referenced (i.e. interactive plots)


            #Allowing for a subset of the waveform to be isolated.  Helpful to speed this up if you know where signals are in long traces.
            waveform_index_range = list(waveform_index_range)
            if waveform_index_range[0] is None:
                waveform_index_range[0] = 0
            if waveform_index_range[1] is None:
                waveform_index_range[1] = self.buffer_length - 1

            if not(waveform_index_range[0] < waveform_index_range[1]):
                print('Given window range invalid, minimum index greater than or equal to max')
                print('Setting full range.')
                self.start_waveform_index = 0
                self.end_waveform_index = self.buffer_length - 1
            else:
                if waveform_index_range[0] < 0:
                    print('Negative start index given, setting to 0.')
                    self.start_waveform_index = 0
                else:
                    self.start_waveform_index = waveform_index_range[0]
                if waveform_index_range[1] >= self.buffer_length:
                    print('Greater than or equal to buffer length given for end index, setting to buffer_length - 1.')
                    self.end_waveform_index = self.buffer_length - 1
                else:
                    self.end_waveform_index = waveform_index_range[1]

            #Resetting buffer length to account for new load in length. 
            self.buffer_length = self.end_waveform_index - self.start_waveform_index + 1 

            self.tukey = scipy.signal.tukey(self.buffer_length, alpha=tukey_alpha, sym=True)

            self.prepForFFTs(plot=plot_filters,apply_phase_response=apply_phase_response)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def rfftWrapper(self, waveform_times, *args, **kwargs):
        '''
        This basically just does an rfft but also converts linear to dB like units. 
        '''
        try:
            numpy.seterr(divide = 'ignore') 
            spec = numpy.fft.rfft(*args, **kwargs)
            real_power_multiplier = 2.0*numpy.ones_like(spec) #The factor of 2 because rfft lost half of the power except for dc and Nyquist bins (handled below).
            if len(numpy.shape(spec)) != 1:
                real_power_multiplier[:,[0,-1]] = 1.0
            else:
                real_power_multiplier[[0,-1]] = 1.0
            spec_dbish = 10.0*numpy.log10( real_power_multiplier*spec * numpy.conj(spec) / len(waveform_times)) #10 because doing power in log.  Dividing by N to match monutau. 
            freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
            numpy.seterr(divide = 'warn')
            return freqs, spec_dbish, spec
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)




    def wf(self,channel,apply_filter=False,hilbert=False,tukey=None):
        '''
        This loads a wf but only the section that is selected by the start and end indices specified.

        If tukey is True to loaded wf will have tapered edges of the waveform on the 1% level to help
        with edge effects.  This will be applied before hilbert if hilbert is true, and before
        the filter.
        '''
        try:
            temp_wf = self.reader.wf(channel)[self.start_waveform_index:self.end_waveform_index+1]
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

    def prepForFFTs(self,plot=False,apply_phase_response=False):
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
            self.filter_original = self.makeFilter(self.freqs_original,plot_filter=plot,apply_phase_response=apply_phase_response)
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
    

    def makeFilter(self,freqs, plot_filter=False,apply_phase_response=False):
        '''
        This will make a frequency domain filter based on the given specifications. 
        '''
        try:
            filter_x = freqs
            
            if numpy.logical_and(self.low_pass_filter_order is not None, self.crit_freq_low_pass_MHz is not None):
                b, a = scipy.signal.butter(self.low_pass_filter_order, self.crit_freq_low_pass_MHz*1e6, 'low', analog=True)
                filter_x_low_pass, filter_y_low_pass = scipy.signal.freqs(b, a,worN=freqs)
            else:
                filter_x_low_pass = filter_x
                filter_y_low_pass = numpy.ones_like(filter_x)

            if numpy.logical_and(self.high_pass_filter_order is not None, self.crit_freq_high_pass_MHz is not None):
                d, c = scipy.signal.butter(self.high_pass_filter_order, self.crit_freq_high_pass_MHz*1e6, 'high', analog=True)
                filter_x_high_pass, filter_y_high_pass = scipy.signal.freqs(d, c,worN=freqs)
            else:
                filter_x_high_pass = filter_x
                filter_y_high_pass = numpy.ones_like(filter_x)

            filter_y = numpy.multiply(filter_y_low_pass,filter_y_high_pass)

            if plot_filter == True:
                fig = plt.figure()
                fig.canvas.set_window_title('Filter')
                numpy.seterr(divide = 'ignore') 
                plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y)),color='k',label='final filter')
                plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_low_pass)),color='r',linestyle='--',label='low pass')
                plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_high_pass)),color='orange',linestyle='--',label='high pass')
                numpy.seterr(divide = 'warn') 
                plt.title('Butterworth filter frequency response')
                plt.xlabel('Frequency [MHz]')
                plt.ylabel('Amplitude [dB]')
                plt.margins(0, 0.1)
                plt.grid(which='both', axis='both')
                if self.crit_freq_low_pass_MHz is not None:
                    plt.axvline(self.crit_freq_low_pass_MHz, color='magenta',label='LP Crit') # cutoff frequency
                if self.crit_freq_high_pass_MHz is not None:
                    plt.axvline(self.crit_freq_high_pass_MHz, color='cyan',label='HP Crit') # cutoff frequency
                plt.xlim(0,200)
                plt.ylim(-50,10)
                plt.legend()

            filter_y = numpy.tile(filter_y,(8,1))
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
                filter_y = numpy.multiply(phase_response_filter,filter_y)
            return filter_y
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def loadFilteredFFTs(self, eventid, hilbert=False, load_upsampled_waveforms=False, shorten_signals=False, shorten_thresh=0.5, shorten_delay=15.0, shorten_length=110.0):
        '''
        Loads the waveforms (with pre applied filters) and upsamples them for
        for the cross correlation. 
        '''
        try:
            self.reader.setEntry(eventid)
            self.eventid = eventid
            raw_wfs_corr = numpy.zeros((8,len(self.waveform_times_corr))) #upsampled to nearest power of 2 then by 2 for correlation.
            if load_upsampled_waveforms:
                upsampled_waveforms = numpy.zeros((8,self.final_corr_length//2)) #These are the waveforms with the same dt as the cross correlation.
            for channel in range(8):
                if load_upsampled_waveforms:
                    temp_raw_wf = self.wf(channel,hilbert=False,apply_filter=True) #apply hilbert after upsample
                else:
                    temp_raw_wf = self.wf(channel,hilbert=hilbert,apply_filter=True)

                temp_raw_wf = temp_raw_wf - numpy.mean(temp_raw_wf) #Subtracting dc offset so cross corrs behave well

                if shorten_signals == True:
                    trigger_index = numpy.where(temp_raw_wf/max(temp_raw_wf) > shorten_thresh)[0][0]
                    weights = numpy.ones_like(temp_raw_wf)
                    cut = numpy.arange(len(temp_raw_wf)) < trigger_index + int(shorten_delay/self.dt_ns_original)
                    slope = -1.0/(shorten_length/self.dt_ns_original) #Go to zero by 100ns after initial dampening.
                    weights[~cut] = numpy.max(numpy.vstack((slope * numpy.arange(sum(~cut)) + 1,numpy.zeros(sum(~cut)))),axis=0) #Max so no negative weights
                    temp_raw_wf = numpy.multiply(temp_raw_wf,weights)

                if hilbert == True:
                    raw_wfs_corr[channel][0:self.buffer_length] = numpy.abs(scipy.signal.hilbert(temp_raw_wf))
                else:
                    raw_wfs_corr[channel][0:self.buffer_length] = temp_raw_wf
                

                if load_upsampled_waveforms:
                    temp_upsampled = numpy.fft.irfft(numpy.fft.rfft(raw_wfs_corr[channel][0:len(self.waveform_times_padded_to_power2)]),n=self.final_corr_length//2) * ((self.final_corr_length//2)/len(self.waveform_times_padded_to_power2))
                    if hilbert == True:
                        upsampled_waveforms[channel] = numpy.abs(scipy.signal.hilbert(temp_upsampled))
                    else:
                        upsampled_waveforms[channel] = temp_upsampled
            


            if False:
                plt.figure()
                plt.title(str(eventid))
                raw_wfs_corr_t = numpy.arange(numpy.shape(raw_wfs_corr)[1])*(self.t()[1] - self.t()[0])
                for channel in range(8):
                    plt.subplot(4,2,channel+1)
                    plt.plot(self.t(),self.wf(channel),alpha=0.7)
                    plt.plot(raw_wfs_corr_t,raw_wfs_corr[channel])
                plt.figure()
                plt.title(str(eventid))
                for channel in range(8):
                    plt.plot(raw_wfs_corr[channel][0:len(self.waveform_times_padded_to_power2)])
                import pdb; pdb.set_trace()
                    

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

    def loadFilteredWaveformsMultiple(self,eventids,hilbert=False):
        '''
        Hacky way to get multiple upsampled waveforms.  Helpful for initial passes at getting a template.
        '''
        try:
            upsampled_waveforms = {}
            for channel in range(8):
                upsampled_waveforms['ch%i'%channel] = numpy.zeros((len(eventids),self.final_corr_length//2))

            for index, eventid in enumerate(eventids):
                wfs = self.loadFilteredFFTs(eventid, hilbert=hilbert, load_upsampled_waveforms=True)[1]
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


    def calculateGroupDelaysFromEvent(self, eventid, apply_filter=False, plot=False,event_type=None):
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
                temp_raw_wf = self.wf(channel,hilbert=False,apply_filter=apply_filter) 
                wfs[channel] = temp_raw_wf - numpy.mean(temp_raw_wf) #Subtracting dc offset

            return self.calculateGroupDelays(self.t(), wfs, plot=plot,event_type=event_type)
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
    def calculateTimeDelays(self, ffts, upsampled_waveforms, return_full_corrs=False, align_method=0, print_warning=True, align_method_10_estimate=None, align_method_10_window_ns=8):
        '''
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
        11: For hpol baselines this will use 0, and for vpol it will use 9.
        12: For vpol baselines this will use 0, and for hpol it will use 9.
        'max_corrs' corresponds to the value of the selected methods peak. 
        '''
        try:
            if print_warning:
                print('Note that calculateTimeDelays expects the ffts to be the same format as those loaded with loadFilteredFFTs().  If this is not the case the returned time shifts may be incorrect.')

            corrs_fft = numpy.multiply((ffts[self.pairs[:,0]].T/numpy.std(ffts[self.pairs[:,0]],axis=1)).T,(numpy.conj(ffts[self.pairs[:,1]]).T/numpy.std(numpy.conj(ffts[self.pairs[:,1]]),axis=1)).T) / (len(self.waveform_times_corr)//2 + 1)

            if ~numpy.all(numpy.isfinite(corrs_fft)):
                import pdb; pdb.set_trace()

            corrs = numpy.fft.fftshift(numpy.fft.irfft(corrs_fft,axis=1,n=self.final_corr_length),axes=1) * (self.final_corr_length//2) #Upsampling and keeping scale

            if ~numpy.all(numpy.isfinite(corrs)):
                import pdb; pdb.set_trace()


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

                time_delays_cfd = numpy.zeros(len(self.pairs))
                time_windows_oneside = 5 #ns3

                indices = numpy.zeros(numpy.shape(corrs)[0],dtype=int)
                max_corrs = numpy.zeros(numpy.shape(corrs)[0])
                for pair_index, pair in enumerate(self.pairs):
                    time_delays_cfd = cfd_trig_times[int(min(pair))] - cfd_trig_times[int(max(pair))]
                    cut = numpy.logical_and(self.corr_time_shifts < time_delays_cfd + time_windows_oneside, self.corr_time_shifts >time_delays_cfd - time_windows_oneside)
                    indices[pair_index] = numpy.argmax( numpy.multiply( cut , corrs[pair_index] ) )
                    max_corrs[pair_index] = corrs[pair_index][indices[pair_index]]

                if False:
                    times = numpy.arange(len(wf))*self.dt_ns_upsampled
                    for pair_index, pair in enumerate(self.pairs):
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
                    loop_pairs = self.pairs
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


                time_delays = numpy.zeros(len(self.pairs))
                display_half_time_window = 200
                display_half_time_window_index = int(display_half_time_window/self.dt_ns_upsampled) #In indices
                slider_half_time_window = 300
                slider_half_time_window_index = int(slider_half_time_window/self.dt_ns_upsampled) #In indices

                t = numpy.arange(upsampled_waveforms.shape[1])*self.dt_ns_upsampled
                for pair_index, pair in enumerate(self.pairs):
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
                    ax_roll = plt.axes([0.25, 0.15, 0.65, 0.03])

                    slider_roll = Slider(ax_roll, 'Roll Ant %i'%pair[1], max(-upsampled_waveforms.shape[1],start_roll-slider_half_time_window_index), min(start_roll+slider_half_time_window_index,upsampled_waveforms.shape[1]), valinit=start_roll, valstep=1.0)


                    def update(val):
                        roll = slider_roll.val
                        plot.set_xdata(t + roll*self.dt_ns_upsampled)
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
                print(align_method_10_estimate)
                print(align_method_10_window_ns)
                if align_method_10_estimate is None:
                    print('NEED TO GIVE VALUE FOR align_method_10_estimate IF USING aligned_method = 10.  Failing.')
                else:
                    #align_method_10_estimate must be have the same number of elements as max_corrs has rows.

                    indices = numpy.zeros(numpy.shape(corrs)[0],dtype=int)
                    max_corrs = numpy.zeros(numpy.shape(corrs)[0])
                    for pair_index, pair in enumerate(self.pairs):
                        search_window_cut = numpy.logical_and(self.corr_time_shifts > (align_method_10_estimate[pair_index] - align_method_10_window_ns),  self.corr_time_shifts < (align_method_10_estimate[pair_index] + align_method_10_window_ns) )
                        search_window_indices = numpy.where(search_window_cut)[0]

                        indices[pair_index] = search_window_indices[numpy.argmax(corrs[pair_index][search_window_cut])]
                        max_corrs[pair_index] = numpy.max(corrs[pair_index][search_window_cut])




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
                return indices, self.corr_time_shifts[indices], max_corrs, self.pairs, corrs
            else:
                return indices, self.corr_time_shifts[indices], max_corrs, self.pairs
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateTimeDelaysFromEvent(self, eventid, return_full_corrs=False, align_method=0, hilbert=False, align_method_10_estimate=None, align_method_10_window_ns=8):
        '''
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
            ffts, upsampled_waveforms = self.loadFilteredFFTs(eventid,load_upsampled_waveforms=True,hilbert=hilbert)
            

            return self.calculateTimeDelays(ffts, upsampled_waveforms, return_full_corrs=return_full_corrs, align_method=align_method, print_warning=False, align_method_10_estimate=align_method_10_estimate, align_method_10_window_ns=align_method_10_window_ns)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateMultipleTimeDelays(self,eventids, align_method=None,hilbert=False,plot=False,hpol_cut=None,vpol_cut=None,colors=None, align_method_10_estimates=None, align_method_10_window_ns=8):
        '''
        If colors is some set of values that matches len(eventids) then they will be used to color the event curves.
        '''
        try:
            if ~numpy.all(eventids == numpy.sort(eventids)):
                print('eventids NOT SORTED, WOULD BE FASTER IF SORTED.')
            timeshifts = []
            max_corrs = []
            print('Calculating time delays:')
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
                for pair in self.pairs:
                    fig = plt.figure()
                    plt.suptitle(str(self.reader.run))
                    fig.canvas.set_window_title('%s%i-%i x-corr'%(['H','V'][pair[0]%2],pair[0]//2,pair[1]//2))
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
            for event_index, eventid in enumerate(eventids):
                if align_method == 9:
                    sys.stdout.write('(%i/%i)\t\t\t\n'%(event_index+1,len(eventids)))
                elif event_index%one_percent_event == 0:
                    sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
                sys.stdout.flush()
                if align_method is None:
                    if plot == True:
                        indices, time_shift, corr_value, pairs, corrs = self.calculateTimeDelaysFromEvent(eventid,hilbert=hilbert,return_full_corrs=True,align_method_10_estimate=align_method_10_estimates[event_index],align_method_10_window_ns=align_method_10_window_ns) #Using default of the other function
                    else:
                        indices, time_shift, corr_value, pairs = self.calculateTimeDelaysFromEvent(eventid,hilbert=hilbert,return_full_corrs=False,align_method_10_estimate=align_method_10_estimates[event_index],align_method_10_window_ns=align_method_10_window_ns) #Using default of the other function
                else:
                    if plot == True:
                        indices, time_shift, corr_value, pairs, corrs = self.calculateTimeDelaysFromEvent(eventid,align_method=align_method,hilbert=hilbert,return_full_corrs=True,align_method_10_estimate=align_method_10_estimates[event_index],align_method_10_window_ns=align_method_10_window_ns)
                    else:
                        indices, time_shift, corr_value, pairs = self.calculateTimeDelaysFromEvent(eventid,align_method=align_method,hilbert=hilbert,return_full_corrs=False,align_method_10_estimate=align_method_10_estimates[event_index],align_method_10_window_ns=align_method_10_window_ns)
                timeshifts.append(time_shift)
                max_corrs.append(corr_value)
                if plot == True:

                    for index, pair in enumerate(self.pairs):
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
                                plt.plot(self.t(),self.wf(int(pair[0]),apply_filter=True,hilbert=False),label=str(int(pair[0])))
                                plt.plot(self.t(),self.wf(int(pair[1]),apply_filter=True,hilbert=False),label=str(int(pair[1])))
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
                                plt.plot(self.t(),self.wf(int(pair[0]),apply_filter=True,hilbert=False),label=str(int(pair[0])))
                                plt.plot(self.t(),self.wf(int(pair[1]),apply_filter=True,hilbert=False),label=str(int(pair[1])))
                                plt.legend()
                                import pdb; pdb.set_trace()
                                plt.close(fig)
                        else:
                            if norm is not None:
                                axs[index].plot(self.corr_time_shifts[index_cut_min:index_cut_max], corrs[index][index_cut_min:index_cut_max],alpha=0.8,c=norm_colors[event_index],label=colors[event_index])
                            else:
                                axs[index].plot(self.corr_time_shifts[index_cut_min:index_cut_max], corrs[index][index_cut_min:index_cut_max],alpha=0.8)
            if plot == True:
                for ax in axs:
                    ax.legend()
            sys.stdout.write('\n')
            sys.stdout.flush()
            return numpy.array(timeshifts).T, numpy.array(max_corrs).T, self.pairs
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateImpulsivityFromTimeDelays(self, eventid, time_delays, upsampled_waveforms=None,return_full_corrs=False, align_method=0, hilbert=False,plot=False,impulsivity_window=400):
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
                ffts, upsampled_waveforms = self.loadFilteredFFTs(eventid,load_upsampled_waveforms=True,hilbert=hilbert)
                #Else use the given waveforms, assumed to have sampling of self.dt_ns_upsampled

            rolls = (time_delays/self.dt_ns_upsampled).astype(int)

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



    def calculateImpulsivityFromEvent(self, eventid, return_full_corrs=False, align_method=0, hilbert=False,plot=False,impulsivity_window=750):
        '''
        This will call calculateTimeDelaysFromEvent with the given settings, and use these to calculate impulsivity. 

        If calculate_impulsivity == True then this will also use the upsampled waveforms to determine impulsivity.
        If hilbert==True then the waveforms used for impulsivity will be recalculated to NOT be enveloped.  

        Impulsivity_window is given in ns and says how wide the window around the peak of the hilbert envelope (not necessarily symmetric)
        to sum the envelope for the metric.
        '''
        try:
            ffts, upsampled_waveforms = self.loadFilteredFFTs(eventid,load_upsampled_waveforms=True,hilbert=hilbert)
            #self.dt_ns_upsampled

            if return_full_corrs == True:
                indices, corr_time_shifts, max_corrs, pairs, corrs = self.calculateTimeDelays(ffts, upsampled_waveforms, return_full_corrs=return_full_corrs, align_method=align_method, print_warning=False)
            else:
                indices, corr_time_shifts, max_corrs, pairs = self.calculateTimeDelays(ffts, upsampled_waveforms, return_full_corrs=return_full_corrs, align_method=align_method, print_warning=False)
            
            time_delays = -corr_time_shifts
            print('From event')
            print(time_delays)
            return self.calculateImpulsivityFromTimeDelays(eventid, time_delays, upsampled_waveforms=upsampled_waveforms,return_full_corrs=return_full_corrs, align_method=align_method, hilbert=hilbert,plot=plot,impulsivity_window=impulsivity_window)

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
    def __init__(self, reader, final_corr_length=2**17, crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, waveform_index_range=(None,None), plot_filters=False, initial_template_id=None,apply_phase_response=False):
        try:
            super().__init__(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range, plot_filters=plot_filters,apply_phase_response=apply_phase_response)
            self.corr_index_to_delay_index = -numpy.arange(-(self.final_corr_length-1)//2,(self.final_corr_length-1)//2 + 1) #Negative because with how it is programmed you would want to roll the template the normal amount, but I will be rolling the waveforms.
            self.setTemplateToEvent(initial_template_id)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def setTemplateToEvent(self,eventid):
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
            self.template_ffts_filtered,self.template_waveform_upsampled_filtered = self.loadFilteredFFTs(eventid,load_upsampled_waveforms=True)

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

    def crossCorrelateWithTemplate(self, eventid, load_upsampled_waveforms=False):
        '''
        Performs a cross correlation between the waveforms for eventid and the
        internally defined template.
        '''
        try:
            #Load events waveforms
            ffts, upsampled_waveforms = self.loadFilteredFFTs(eventid, load_upsampled_waveforms=load_upsampled_waveforms)

            #Perform cross correlations with the template.
            corrs_fft = numpy.multiply((ffts.T/numpy.std(ffts,axis=1)).T,(self.scaled_conj_template_ffts_filtered)) / (len(self.waveform_times_corr)//2 + 1)
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


    def alignToTemplate(self, eventid, align_method=0):
        '''
        Attempts to align the waveforms from eventid with the internally
        defined template.  This may use one of several methods of aligning
        the waveforms, as specified by align_method.

        Note the waveforms that are returned are upsampled to a factor of 2
        but are not the double length ones that are used for the correlation.
        '''
        try:
            corrs, corrs_fft, upsampled_waveforms = self.crossCorrelateWithTemplate(eventid,load_upsampled_waveforms=True)

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
            
            return max_corrs, upsampled_waveforms, rolled_wfs
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def averageAlignedSignalsPerChannel(self, eventids, align_method=0, template_eventid=None, plot=False,event_type=None):
        '''
        This will use alignToTemplate to align an events within a channel to the current set template.
        This means that you should set the template with either setTemplateToEvent or setTemplateToCustom
        in advance.  template_eventid will call setTemplateToEvent if given.  align_method is passed to
        alignToTemplate to align signals within the channel.

        Note that the template is stored as a conjugated fft cross-correlation ready version of the template.
        '''
        try:
            if template_eventid is not None:
                self.setTemplateToEvent(template_eventid)

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
                for channel in channels:
                    ax = axs[channel]
                    ax.plot(times, averaged_waveforms[channel],linestyle='--',c='r',label=str(channel)+' avg')

            if plot == True:
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
        

if __name__ == '__main__':
    try:
        plt.close('all')
        datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
        shorten_signals = True
        shorten_thresh = 0.5 #Percent of max height to trigger leading edge of signal.  Should only apply if a certain snr is met?
        #shorten_delay = 20.0 #The delay in ns after the trigger thresh to wait before an exponential decrease is applied. 
        #shorten_length = 100.0
        plot_shorten_2d = False

        delays = numpy.array([15.0])#numpy.linspace(5,70,15)
        lengths = numpy.array([110.0])#numpy.linspace(20,200,20)
        delays_mesh, lengths_mesh = numpy.meshgrid(delays, lengths)
        

        make_plot = False
        verbose = False
        mode = 'hpol'
        try_removing_expected_simple_reflection = False
        allowed_plots = 10
        current_n_plots = 0

        waveform_index_range = (None,None)
        apply_phase_response = True

        #Filter settings
        final_corr_length = 2**16 #Should be a factor of 2 for fastest performance (I think)
        
        crit_freq_low_pass_MHz = None#70 #This new pulser seems to peak in the region of 85 MHz or so
        low_pass_filter_order = None#12
        
        high_pass_filter_order = None#8
        crit_freq_high_pass_MHz = None#20

        import tools.get_plane_tracks as pt
        known_planes, calibrated_trigtime, output_tracks = pt.getKnownPlaneTracks()
        use_filter = True
        plot_filters = False
        pairs = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
        
        antennas_cm = plt.cm.get_cmap('tab10', 10)
        antenna_colors = antennas_cm(numpy.linspace(0, 1, 10))[0:4]

        for index, key in enumerate(list(known_planes.keys())):
            # if key == '1774-88800':
            #     pass
            # else:
            #     pass
            # if index > 0:
            #     continue
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
                    for event_index, eventid in enumerate(eventids):
                        # if event_index != 2:
                        #     continue
                        ffts, upsampled_waveforms = tdc.loadFilteredFFTs(eventid,load_upsampled_waveforms=True,hilbert=False,shorten_signals=shorten_signals,shorten_thresh=shorten_thresh,shorten_delay=shorten_delay,shorten_length=shorten_length)


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
                                peaks = scipy.signal.find_peaks(hilbert_corr,height=0.7*max(hilbert_corr),distance=20.0/tdc.dt_ns_upsampled,prominence=0.05)[0]
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
                                    prom = scipy.signal.peak_prominences(hilbert_corr,[peak])[0]
                                    plt.scatter(tdc.corr_time_shifts[peaks],hilbert_corr[peaks],label='Peak Prominence = ' + str(prom[0]), c='r')
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
