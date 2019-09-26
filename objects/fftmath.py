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
    
    See Also
    --------
    examples.beacon_data_reader.reader
    '''
    def __init__(self, reader, final_corr_length=2**17, crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None):
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

            self.prepForFFTs()

        except Exception as e:
            print('Error in FFTPrepper.__init__()')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

           
    def prepForFFTs(self):
        '''
        This will get timing information from the reader and use it to determine values such as timestep
        that will be used when performing ffts later.  
        '''
        try:
            self.eventid = 0
            self.reader.setEntry(self.eventid)

            #Below are for the original times of the waveforms and correspond frequencies.
            self.waveform_times_original = self.reader.t()
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
            self.filter_original = self.makeFilter(self.freqs_original,plot_filter=False)
            self.filter_padded_to_power2 = self.makeFilter(self.freqs_padded_to_power2,plot_filter=False)
            self.filter_corr = self.makeFilter(self.freqs_corr,plot_filter=False)


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

    def loadFilteredFFTs(self, eventid, load_upsampled_waveforms=False):
        '''
        Loads the waveforms and applies filters.  These are filtered FFTs with lengths
        for the cross correlation. 

        If load_upsampled_waveforms is True than the waveforms with the same timestep as
        the cross correlations are also returned.  These have timestep self.dt_ns_upsampled.
        '''
        self.reader.setEntry(eventid)
        raw_wfs_corr = numpy.zeros((8,len(self.waveform_times_corr))) #upsampled to nearest power of 2 then by 2 for correlation.
        if load_upsampled_waveforms:
            upsampled_waveforms = numpy.zeros((8,self.final_corr_length//2)) #These are the waveforms with the same dt as the cross correlation.
        for channel in range(8):
            raw_wfs_corr[channel][0:self.buffer_length] = self.reader.wf(channel)
            if load_upsampled_waveforms:
                upsampled_waveforms[channel] = numpy.fft.irfft(numpy.multiply(self.filter_padded_to_power2,numpy.fft.rfft(raw_wfs_corr[channel][0:len(self.waveform_times_padded_to_power2)])),n=self.final_corr_length//2) * ((self.final_corr_length//2)/len(self.waveform_times_padded_to_power2))


        ffts_corr = numpy.fft.rfft(raw_wfs_corr,axis=1) #Now upsampled
        waveform_ffts_filtered_corr = numpy.multiply(ffts_corr,self.filter_corr)

        if load_upsampled_waveforms:
            return waveform_ffts_filtered_corr, upsampled_waveforms
        else:
            return waveform_ffts_filtered_corr

    def loadFilteredFFTsWithWaveforms(self,eventid):
        '''
        Loads the waveforms and applies filters.  These are filtered FFTs with lengths
        for the cross correlation.  These should share dt with self.dt_ns_upsampled (same as
        cross correlations).
        '''
        self.reader.setEntry(eventid)
        raw_wfs_corr = numpy.zeros((8,len(self.waveform_times_corr))) #upsampled to nearest power of 2 then by 2 for correlation.
        upsampled_waveforms = numpy.zeros((8,self.final_corr_length//2)) #These are the waveforms with the same dt as the cross correlation.
        for channel in range(8):
            raw_wfs_corr[channel][0:self.buffer_length] = self.reader.wf(channel)
            upsampled_waveforms[channel] = numpy.fft.irfft(numpy.multiply(self.filter_padded_to_power2,numpy.fft.rfft(raw_wfs_corr[channel][0:len(self.waveform_times_padded_to_power2)])),n=self.final_corr_length//2) * ((self.final_corr_length//2)/len(self.waveform_times_padded_to_power2))

        ffts_corr = numpy.fft.rfft(raw_wfs_corr,axis=1) #Now upsampled
        waveform_ffts_filtered_corr = numpy.multiply(ffts_corr,self.filter_corr)

        return waveform_ffts_filtered_corr, upsampled_waveforms

    def loadFilteredWaveformsMultiple(self,eventids):
        '''
        Hacky way to get multiple upsampled waveforms.  Helpful for initial passes at getting a template.
        '''
        try:
            upsampled_waveforms = {}
            for channel in range(8):
                upsampled_waveforms['ch%i'%channel] = numpy.zeros((len(eventids),self.final_corr_length//2))

            for index, eventid in enumerate(eventids):
                wfs = self.loadFilteredFFTsWithWaveforms(eventid)[1]
                for channel in range(8):
                    upsampled_waveforms['ch%i'%channel][index] = wfs[channel]

            return upsampled_waveforms
        except Exception as e:
            print('Error in TimeDelayCalculator.loadFilteredWaveformsMultiple')
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
    def calculateTimeDelays(self,eventid):
        try:
            ffts = self.loadFilteredFFTs(eventid)

            corrs_fft = numpy.multiply((ffts[self.pairs[:,0]].T/numpy.std(ffts[self.pairs[:,0]],axis=1)).T,(numpy.conj(ffts[self.pairs[:,1]]).T/numpy.std(numpy.conj(ffts[self.pairs[:,1]]),axis=1)).T) / (len(self.waveform_times_corr)//2 + 1)
            corrs = numpy.fft.fftshift(numpy.fft.irfft(corrs_fft,axis=1,n=self.final_corr_length),axes=1) * (self.final_corr_length//2) #Upsampling and keeping scale

            indices = numpy.argmax(corrs,axis=1)
            max_corrs = numpy.max(corrs,axis=1)

            return self.corr_time_shifts[indices], max_corrs, self.pairs
        except Exception as e:
            print('Error in TimeDelayCalculator.calculateTimeDelays')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateMultipleTimeDelays(self,eventids):
        try:
            if ~numpy.all(eventids == numpy.sort(eventids)):
                print('eventids NOT SORTED, WOULD BE FASTER IF SORTED.')
            timeshifts = []
            corrs = []
            print('Calculating time delays:')
            for event_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
                sys.stdout.flush()
                time_shift, corr_value, pairs = self.calculateTimeDelays(eventid)
                timeshifts.append(time_shift)
                corrs.append(corr_value)
            sys.stdout.write('\n')
            sys.stdout.flush()
            return numpy.array(timeshifts).T, numpy.array(corrs).T, self.pairs

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
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine
    #The below are not ALL pulser points, but a set that has been precalculated and can be used
    #if you wish to skip the calculation finding them.

    known_pulser_ids = info.loadPulserEventids()
    ignorable_pulser_ids = info.loadIgnorableEventids()

    #Filter settings
    final_corr_length = 2**17 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = 80
    crit_freq_high_pass_MHz = 20
    low_pass_filter_order = 6
    high_pass_filter_order = 6
    use_filter = False
    align_method = 3
    #1. Looks for best alignment within window after cfd trigger, cfd applied on hilbert envelope.
    #2. Of the peaks in the correlation plot within range of the max peak, choose the peak with the minimum index.
    #3. Align to maximum correlation value.

    #Main loop
    for run_index, run in enumerate(runs):
        if 'run%i'%run in list(known_pulser_ids.keys()):
            try:
                if 'run%i'%run in list(ignorable_pulser_ids.keys()):
                    eventids = numpy.sort(known_pulser_ids['run%i'%run][~numpy.isin(known_pulser_ids['run%i'%run],ignorable_pulser_ids['run%i'%run])])
                else:
                    eventids = numpy.sort(known_pulser_ids['run%i'%run])

                reader = Reader(datapath,run)

                tct = TemplateCompareTool(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, initial_template_id=0)
                out = tct.alignToTemplate(0, align_method)

                #The above should result in wf and rolled_wf to be the same because correlating with self.
            except Exception as e:
                print('Error in main loop.')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)