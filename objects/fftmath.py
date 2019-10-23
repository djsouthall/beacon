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

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
    def __init__(self, reader, final_corr_length=2**17, crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, waveform_index_range=(None,None), plot_filters=False):
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

            self.start_waveform_index = waveform_index_range[0]

            self.prepForFFTs(plot=plot_filters)

        except Exception as e:
            print('Error in FFTPrepper.__init__()')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def rfftWrapper(self, waveform_times, *args, **kwargs):
        '''
        This basically just does an rfft but also converts linear to dB like units. 
        '''
        spec = numpy.fft.rfft(*args, **kwargs)
        real_power_multiplier = 2.0*numpy.ones_like(spec) #The factor of 2 because rfft lost half of the power except for dc and Nyquist bins (handled below).
        if len(numpy.shape(spec)) != 1:
            real_power_multiplier[:,[0,-1]] = 1.0
        else:
            real_power_multiplier[[0,-1]] = 1.0
        spec_dbish = 10.0*numpy.log10( real_power_multiplier*spec * numpy.conj(spec) / len(waveform_times)) #10 because doing power in log.  Dividing by N to match monutau. 
        freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
        return freqs, spec_dbish, spec

    def wf(self,channel,apply_filter=False,hilbert=False):
        '''
        This loads a wf but only the section that is selected by the start and end indices specified.

        '''
        temp_wf = self.reader.wf(channel)[self.start_waveform_index:self.end_waveform_index+1]
        if apply_filter == True:
            wf = numpy.fft.irfft(numpy.multiply(self.filter_original,numpy.fft.rfft(temp_wf)),n=self.buffer_length) #Might need additional normalization
        else:
            wf = temp_wf

        if hilbert == True:
            wf = numpy.abs(scipy.signal.hilbert(wf))

        return wf

    def t(self):
        '''
        This loads a times but only the section that is selected by the start and end indices specified.

        This will also roll the starting point to zero. 
        '''
        return self.reader.t()[self.start_waveform_index:self.end_waveform_index+1] - self.reader.t()[self.start_waveform_index]

    def setEntry(self, entry):
        self.reader.setEntry(entry)

    def prepForFFTs(self,plot=False):
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
            self.corr_time_shifts = numpy.arange(-(self.final_corr_length-1)//2,(self.final_corr_length-1)//2 + 1)*self.dt_ns_upsampled #This results in the maxiumum of an autocorrelation being located at a time shift of 0.0

            #Prepare Filters
            self.filter_original = self.makeFilter(self.freqs_original,plot_filter=plot)
            #self.filter_padded_to_power2 = self.makeFilter(self.freqs_padded_to_power2,plot_filter=False)
            #self.filter_corr = self.makeFilter(self.freqs_corr,plot_filter=False)


        except Exception as e:
            print('Error in FFTPrepper.prepForFFTs()')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def makeFilter(self,freqs, plot_filter=False):
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
                plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y)),color='k',label='final filter')
                plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_low_pass)),color='r',linestyle='--',label='low pass')
                plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_high_pass)),color='orange',linestyle='--',label='high pass')
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

            return filter_y
        except Exception as e:
            print('Error in FFTPrepper.makeFilter')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def loadFilteredFFTs(self, eventid, hilbert=False, load_upsampled_waveforms=False):
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

            waveform_ffts_filtered_corr = numpy.fft.rfft(raw_wfs_corr,axis=1) #Now upsampled

            if load_upsampled_waveforms:
                return waveform_ffts_filtered_corr, upsampled_waveforms
            else:
                return waveform_ffts_filtered_corr
        except Exception as e:
            print('Error in TimeDelayCalculator.loadFilteredFFTs')
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
            print('Error in TimeDelayCalculator.loadFilteredWaveformsMultiple')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateGroupDelays(self, eventid, apply_filter=False, plot=False,event_type=None):
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
            ffts_dBish = numpy.zeros((8,len(self.freqs_original)))
            ffts = numpy.zeros((8,len(self.freqs_original)),dtype=numpy.complex64)
            group_delays = numpy.zeros((8,len(self.freqs_original)-1))
            phases = numpy.zeros((8,len(self.freqs_original)))

            for channel in range(8):
                temp_raw_wf = self.wf(channel,hilbert=False,apply_filter=apply_filter) 
                wfs[channel] = temp_raw_wf - numpy.mean(temp_raw_wf) #Subtracting dc offset
                freqs, ffts_dBish[channel], ffts[channel] = self.rfftWrapper(self.t(),wfs[channel])
    
                if channel == 0:
                    #Doesn't need to be calculated several times.
                    group_delay_freqs = numpy.diff(freqs) + freqs[0:len(freqs)-1]
                    omega = 2.0*numpy.pi*freqs

                phases[channel] = numpy.unwrap(numpy.angle(ffts[channel]))
                group_delays[channel] = (-numpy.diff(phases[channel])/numpy.diff(omega)) * 1e9


            if plot == True:
                plt.figure()

                xlim = (60e6,85e6)

                if event_type == 'hpol':
                    channels = numpy.arange(4)*2
                elif event_type == 'vpol':
                    channels = numpy.arange(4)*2 + 1
                else:
                    channels = numpy.arange(8)

                for channel in channels:
                    channel = int(channel)

                    plt.subplot(3,1,1)
                    cut = numpy.logical_and(freqs>= xlim[0],freqs<= xlim[1])
                    plt.plot(freqs[cut]/1e6,ffts_dBish[channel][cut],label=str(channel))
                    plt.ylabel('Magnitude (dB ish)')
                    plt.legend()
                    plt.xlabel('MHz')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    

                    plt.subplot(3,1,2)
                    cut = numpy.logical_and(freqs>= xlim[0],freqs<= xlim[1])
                    plt.plot(freqs[cut]/1e6,phases[channel][cut],label=str(channel))
                    plt.ylabel('Unwrapped Phase (rad)')
                    plt.legend()
                    plt.xlabel('MHz')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                    plt.subplot(3,1,3)
                    cut = numpy.logical_and(group_delay_freqs>= xlim[0],group_delay_freqs<= xlim[1])
                    plt.plot(group_delay_freqs[cut]/1e6,group_delays[channel][cut],label=str(channel))
                    plt.ylabel('Group Delay (ns)')
                    plt.legend()
                    plt.xlabel('MHz')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            return group_delay_freqs, group_delays




        except Exception as e:
            print('Error in TimeDelayCalculator.loadFilteredFFTs')
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
    def calculateTimeDelays(self, eventid, return_full_corrs=False, align_method=0, hilbert=False):
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

            corrs_fft = numpy.multiply((ffts[self.pairs[:,0]].T/numpy.std(ffts[self.pairs[:,0]],axis=1)).T,(numpy.conj(ffts[self.pairs[:,1]]).T/numpy.std(numpy.conj(ffts[self.pairs[:,1]]),axis=1)).T) / (len(self.waveform_times_corr)//2 + 1)
            corrs = numpy.fft.fftshift(numpy.fft.irfft(corrs_fft,axis=1,n=self.final_corr_length),axes=1) * (self.final_corr_length//2) #Upsampling and keeping scale


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
                return self.corr_time_shifts[indices], max_corrs, self.pairs, corrs
            else:
                return self.corr_time_shifts[indices], max_corrs, self.pairs
        except Exception as e:
            print('Error in TimeDelayCalculator.calculateTimeDelays')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateMultipleTimeDelays(self,eventids, align_method=None,hilbert=False,plot=False,hpol_cut=None,vpol_cut=None):
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
                for pair in self.pairs:
                    fig = plt.figure()
                    fig.canvas.set_window_title('%s%i-%i x-corr'%(['H','V'][pair[0]%2],pair[0]//2,pair[1]//2))
                    plt.ylabel('%s x-corr'%str(pair))
                    figs.append(fig)
                    axs.append(plt.gca())
            for event_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
                sys.stdout.flush()
                if align_method is None:
                    if plot == True:
                        time_shift, corr_value, pairs, corrs = self.calculateTimeDelays(eventid,hilbert=hilbert,return_full_corrs=True) #Using default of the other function
                    else:
                        time_shift, corr_value, pairs = self.calculateTimeDelays(eventid,hilbert=hilbert,return_full_corrs=False) #Using default of the other function
                else:
                    if plot == True:
                        time_shift, corr_value, pairs, corrs = self.calculateTimeDelays(eventid,align_method=align_method,hilbert=hilbert,return_full_corrs=True)
                    else:
                        time_shift, corr_value, pairs = self.calculateTimeDelays(eventid,align_method=align_method,hilbert=hilbert,return_full_corrs=False)
                timeshifts.append(time_shift)
                max_corrs.append(corr_value)
                if plot == True:
                    for index, pair in enumerate(self.pairs):
                        #plotting less points by plotting subset of data
                        time_mid = self.corr_time_shifts[numpy.argmax(corrs[index])]
                        
                        half_window = 150 #ns
                        index_cut_min = numpy.max(numpy.append(numpy.where(self.corr_time_shifts < time_mid - half_window)[0],0))
                        index_cut_max = numpy.min(numpy.append(numpy.where(self.corr_time_shifts > time_mid + half_window)[0],len(self.corr_time_shifts) - 1))
                        '''
                        index_loc = numpy.where(corrs[index] > 0.2*max(corrs[index]))[0] #Indices of values greater than some threshold
                        index_cut_min = numpy.min(index_loc)
                        index_cut_max = numpy.max(index_loc)
                        '''
                        if hpol_cut is not None:
                            if len(hpol_cut) == len(eventids):
                                if numpy.logical_and(hpol_cut[event_index], pair[0]%2 == 0):
                                    axs[index].plot(self.corr_time_shifts[index_cut_min:index_cut_max], corrs[index][index_cut_min:index_cut_max],alpha=0.2)
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
                            axs[index].plot(self.corr_time_shifts[index_cut_min:index_cut_max], corrs[index][index_cut_min:index_cut_max],alpha=0.2)
                        if vpol_cut is not None:
                            if len(vpol_cut) == len(eventids):
                                if numpy.logical_and(vpol_cut[event_index], pair[0]%2 != 0):
                                    axs[index].plot(self.corr_time_shifts[index_cut_min:index_cut_max], corrs[index][index_cut_min:index_cut_max],alpha=0.2)
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
                            axs[index].plot(self.corr_time_shifts[index_cut_min:index_cut_max], corrs[index][index_cut_min:index_cut_max],alpha=0.2)

            sys.stdout.write('\n')
            sys.stdout.flush()
            return numpy.array(timeshifts).T, numpy.array(max_corrs).T, self.pairs

        except Exception as e:
            print('Error in TimeDelayCalculator.calculateMultipleTimeDelays')
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
    def __init__(self, reader, final_corr_length=2**17, crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, initial_template_id=None):
        super().__init__(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order)
        self.corr_index_to_delay_index = -numpy.arange(-(self.final_corr_length-1)//2,(self.final_corr_length-1)//2 + 1) #Negative because with how it is programmed you would want to roll the template the normal amount, but I will be rolling the waveforms.

        try:
            self.setTemplateToEvent(initial_template_id)
            print('Event %i set as template'%self.template_eventid)
        except Exception as e:
            print('Error in TemplateCompareTool.__init__')
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
                pass
            else:
                print('None given as eventid, setting event 0 as template.')
                eventid = 0
            self.template_eventid = eventid
            self.template_ffts_filtered,self.template_waveform_upsampled_filtered = self.loadFilteredFFTs(eventid,load_upsampled_waveforms=True)

            #Scaled and conjugate template for one time prep of cross correlations.
            self.scaled_conj_template_ffts_filtered = (numpy.conj(self.template_ffts_filtered).T/numpy.std(numpy.conj(self.template_ffts_filtered),axis=1) ).T

        except Exception as e:
            print('Error in TemplateCompareTool.setTemplateToEvent')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def setTemplateToCustom(self, template_ffts_filtered):
        '''
        Sets the template to the given waveforms.
        '''
        self.template_eventid = 'custom'

        try:
            self.template_ffts_filtered = template_ffts_filtered

            #Scaled and conjugate template for one time prep of cross correlations.
            self.scaled_conj_template_ffts_filtered = (numpy.conj(self.template_ffts_filtered).T/numpy.std(numpy.conj(self.template_ffts_filtered),axis=1)).T

        except Exception as e:
            print('Error in TemplateCompareTool.setTemplateToCustom')
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
            print('Error in TemplateCompareTool.setTemplateToCustom')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def alignToTemplate(self, eventid, align_method):
        '''
        Attempts to align the waveforms from eventid with the internally
        defined template.  This may use one of several methods of aligning
        the waveforms, as specified by align_method.

        Note the waveforms that are returned are upsampled to a factor of 2
        but are not the double length ones that are used for the correlation.
        '''
        try:
            corrs, corrs_fft, upsampled_waveforms = self.crossCorrelateWithTemplate(eventid,load_upsampled_waveforms=True)

            if align_method == 1:
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

            elif align_method == 3:
                #Align to maximum correlation value.
                index_delays = numpy.argmax(corrs,axis=1) #These are the indices within corr that are max.
                max_corrs = []
                for index, corr in enumerate(corrs):
                    max_corrs.append(corr[index_delays[index]])
                max_corrs = numpy.array(max_corrs)
            else:
                print('Given align_method invalid.  This will probably break.')

            #With the indices determine relevant information and roll wf.
            
            rolled_wfs = numpy.zeros_like(upsampled_waveforms)
            for index, wf in enumerate(upsampled_waveforms):
                rolled_wfs[index] = numpy.roll(wf,self.corr_index_to_delay_index[index_delays[index]])
            
            return max_corrs, upsampled_waveforms, rolled_wfs
        except Exception as e:
            print('Error in TemplateCompareTool.alignToTemplate')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)



def alignToTemplate(eventids,_upsampled_waveforms,_waveforms_corr, _template, _final_corr_length, _waveform_times_corr, _filter_y_corr=None, align_method=1, plot_wf=False, template_pre_filtered=False):
    '''
    _waveforms_corr should be a 2d array where each row is a waveform of a different event, already zero padded for cross correlation
    (i.e. must be over half zeros on the right side).

    _upsampled_waveforms are the waveforms that have been upsampled (BUT WERE NOT ORIGINALLY PADDED BY A FACTOR OF 2).  These are
    the waveforms that will be aligned based on the time delays from correlations performed with _waveforms_corr.
        
    If a filter is given then it will be applied to all signals.  It is assumed the upsampled signals are already filtered.

    This given _template must be in the same form upsampled nature as _waveforms_corr. 

    #1. Looks for best alignment within window after cfd trigger, cfd applied on hilbert envelope.
    #2. Of the peaks in the correlation plot within range of the max peak, choose the peak with the minimum index.
    #3. Align to maximum correlation value.
    '''
    try:
        #Prepare template correlation storage
        max_corrs = numpy.zeros(len(eventids))
        index_delays = numpy.zeros(len(eventids),dtype=int) #Within the correlations for max corr
        corr_index_to_delay_index = -numpy.arange(-(_final_corr_length-1)//2,(_final_corr_length-1)//2 + 1) #Negative because with how it is programmed you would want to roll the template the normal amount, but I will be rolling the waveforms.
        rolled_wf = numpy.zeros_like(_upsampled_waveforms)

        if numpy.logical_and(_filter_y_corr is not None,template_pre_filtered == False):
            template_fft = numpy.multiply(numpy.fft.rfft(_template),_filter_y_corr)
            #template_fft = numpy.multiply(numpy.fft.rfft(_waveforms_corr[_template_event_index]),_filter_y_corr)
        else:
            template_fft = numpy.fft.rfft(_template)
            #template_fft = numpy.fft.rfft(_waveforms_corr[_template_event_index])
            
        scaled_conj_template_fft = numpy.conj(template_fft)/numpy.std(numpy.conj(template_fft)) 

        for event_index, eventid in enumerate(eventids):
            sys.stdout.write('(%i/%i)\r'%(event_index,len(eventids)))
            sys.stdout.flush()
            reader.setEntry(eventid)

            if _filter_y_corr is not None:
                fft = numpy.multiply(numpy.fft.rfft(_waveforms_corr[event_index]),_filter_y_corr)
            else:
                fft = numpy.fft.rfft(_waveforms_corr[event_index])

            corr_fft = numpy.multiply((fft/numpy.std(fft)),(scaled_conj_template_fft)) / (len(_waveform_times_corr)//2 + 1)
            corr = numpy.fft.fftshift(numpy.fft.irfft(corr_fft,n=_final_corr_length)) * (_final_corr_length//2 + 1) #Upsampling and keeping scale
            

            if align_method == 1:
                #Looks for best alignment within window after cfd trigger, cfd applied on hilber envelope.
                corr_hilbert = numpy.abs(scipy.signal.hilbert(corr))
                cfd_indices = numpy.where(corr_hilbert/numpy.max(corr_hilbert) > 0.5)[0]
                cfd_indices = cfd_indices[0:int(0.50*len(cfd_indices))] #Top 50% close past 50% max rise
                index_delays[event_index] = cfd_indices[numpy.argmax(corr[cfd_indices])]

                max_corrs[event_index] = corr[index_delays[event_index]]
                rolled_wf[event_index] = numpy.roll(_upsampled_waveforms[event_index],corr_index_to_delay_index[index_delays[event_index]])

            elif align_method == 2:
                #Of the peaks in the correlation plot within range of the max peak, choose the peak with the minimum index.
                index_delays[event_index] = min(scipy.signal.find_peaks(corr,height = 0.8*max(corr),distance=int(0.05*len(corr)))[0])

                max_corrs[event_index] = corr[index_delays[event_index]]
                rolled_wf[event_index] = numpy.roll(_upsampled_waveforms[event_index],corr_index_to_delay_index[index_delays[event_index]])

            elif align_method == 3:
                #Align to maximum correlation value.
                index_delays[event_index] = numpy.argmax(corr) #These are the indices within corr that are max.
                max_corrs[event_index] = corr[index_delays[event_index]]
                rolled_wf[event_index] = numpy.roll(_upsampled_waveforms[event_index],corr_index_to_delay_index[index_delays[event_index]])
        

        if False:
            #Use weighted averages for the correlations in the template
            upsampled_template_out = numpy.average(rolled_wf,axis=0,weights = max_corrs)
            upsampled_template_better = numpy.average(rolled_wf[max_corrs > 0.7],axis=0,weights = max_corrs[max_corrs > 0.7])
            upsampled_template_worse = numpy.average(rolled_wf[max_corrs <= 0.7],axis=0,weights = max_corrs[max_corrs <= 0.7])
        else:
            #DON'T Use weighted averages for the correlations in the template.
            upsampled_template_out = numpy.average(rolled_wf,axis=0)
            upsampled_template_better = numpy.average(rolled_wf[max_corrs > 0.7],axis=0)
            upsampled_template_worse = numpy.average(rolled_wf[max_corrs <= 0.7],axis=0)

        #import pdb;pdb.set_trace()
        #The template at this stage is in the upsampled waveforms which did not have the factor of 2 zeros added.  To make it an exceptable form
        #To be ran back into this function, it must be downsampled to the length before the factor of 2 was added.  Then add a factor of 2 of zeros.
        downsampled_template_out = numpy.zeros(numpy.shape(_waveforms_corr)[1])
        downsampled_template_out[0:numpy.shape(_waveforms_corr)[1]//2] = numpy.fft.irfft(numpy.fft.rfft(upsampled_template_out), n = numpy.shape(_waveforms_corr)[1]//2) * ((numpy.shape(_waveforms_corr)[1]//2)/numpy.shape(_upsampled_waveforms)[1])  #This can then be fed back into this function

        downsampled_template_worse = numpy.zeros(numpy.shape(_waveforms_corr)[1])
        downsampled_template_worse[0:numpy.shape(_waveforms_corr)[1]//2] = numpy.fft.irfft(numpy.fft.rfft(upsampled_template_worse), n = numpy.shape(_waveforms_corr)[1]//2) * ((numpy.shape(_waveforms_corr)[1]//2)/numpy.shape(_upsampled_waveforms)[1])  #This can then be fed back into this function

        downsampled_template_better = numpy.zeros(numpy.shape(_waveforms_corr)[1])
        downsampled_template_better[0:numpy.shape(_waveforms_corr)[1]//2] = numpy.fft.irfft(numpy.fft.rfft(upsampled_template_better), n = numpy.shape(_waveforms_corr)[1]//2) * ((numpy.shape(_waveforms_corr)[1]//2)/numpy.shape(_upsampled_waveforms)[1])  #This can then be fed back into this function


        if plot_wf:
            plt.figure()
            plt.title('Aligned Waveforms')
            for event_index, eventid in enumerate(eventids):
                plt.plot(numpy.linspace(0,1,len(rolled_wf[event_index])),rolled_wf[event_index],alpha=0.5)
            plt.plot(numpy.linspace(0,1,len(downsampled_template_out[0:len(downsampled_template_out)//2])),downsampled_template_out[0:len(downsampled_template_out)//2],color='r',linestyle='--')
            plt.ylabel('Adu',fontsize=16)
            plt.xlabel('Time (arb)',fontsize=16)

            plt.figure()
            for event_index, eventid in enumerate(eventids):
                if max_corrs[event_index] < 0.70:
                    plt.subplot(2,1,1)
                    plt.plot(numpy.linspace(0,1,len(rolled_wf[event_index])),rolled_wf[event_index],alpha=0.5)
                else:
                    plt.subplot(2,1,2)
                    plt.plot(numpy.linspace(0,1,len(rolled_wf[event_index])),rolled_wf[event_index],alpha=0.5)
            plt.subplot(2,1,1)
            #plt.plot(numpy.linspace(0,1,len(downsampled_template_out[0:len(downsampled_template_out)//2])),downsampled_template_out[0:len(downsampled_template_out)//2],color='r',linestyle='--')
            plt.plot(numpy.linspace(0,1,len(downsampled_template_worse[0:len(downsampled_template_worse)//2])),downsampled_template_worse[0:len(downsampled_template_worse)//2],color='r',linestyle='--')
            plt.ylabel('Adu',fontsize=16)
            plt.xlabel('Time (arb)',fontsize=16)

            plt.subplot(2,1,2)
            #plt.plot(numpy.linspace(0,1,len(downsampled_template_out[0:len(downsampled_template_out)//2])),downsampled_template_out[0:len(downsampled_template_out)//2],color='r',linestyle='--')
            plt.plot(numpy.linspace(0,1,len(downsampled_template_better[0:len(downsampled_template_better)//2])),downsampled_template_better[0:len(downsampled_template_better)//2],color='r',linestyle='--')
            plt.ylabel('Adu',fontsize=16)
            plt.xlabel('Time (arb)',fontsize=16)



            #plt.legend(fontsize=16)
        return index_delays, max_corrs, downsampled_template_out
    except Exception as e:
        print('Error in alignToTemplate')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    run = 1507
    waveform_index_range = (1500,None)

    #Filter settings
    final_corr_length = 2**16 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = 70 #This new pulser seems to peak in the region of 85 MHz or so
    crit_freq_high_pass_MHz = 65
    low_pass_filter_order = 8
    high_pass_filter_order = 4
    use_filter = True
    plot_filters= True

    known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
    eventids = {}
    eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])
    eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])
    all_eventids = numpy.sort(numpy.append(eventids['hpol'],eventids['vpol']))

    hpol_eventids_cut = numpy.isin(all_eventids,eventids['hpol'])
    vpol_eventids_cut = numpy.isin(all_eventids,eventids['vpol'])

    reader = Reader(datapath,run)
    prep = FFTPrepper(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=plot_filters)
    


    eventid = all_eventids[0]
    if numpy.isin(eventid,eventids['hpol']):
        event_type = 'hpol'
    elif numpy.isin(eventid,eventids['vpol']):
        event_type = 'vpol'
    else:
        event_type = None

    group_delay_freqs, group_delays = prep.calculateGroupDelays(eventid, apply_filter=False, plot=True,event_type=event_type)

