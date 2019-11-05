'''
This class is adopted from a script originally written by Kaeli Hughes and as been significantly
restructured/altered for my BEACON analysis framework.  

The purpose of this script is toprovide tools for plotting cross correlation maps for 
interpretting signal directions of BEACON signals.

'''
import os
import sys
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info

import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import numpy
import sys
import math
import matplotlib
from scipy.fftpack import fft
import datetime as dt
import inspect
from ast import literal_eval
plt.ion()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 16})


class Correlator:
    '''
    This will take in a reader and do the prep work for correlation plots.
    It provides functions for plotting correlation maps.

    This assumes the current antenna positions (ENU) as defined by info.loadAntennaPositionsENU
    are correct.  To overwrite these use overwriteAntennaLocations, which will input the new
    coordinates and update the time delays.

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        This is the reader for the selected run.
    upsample : int
        The length that waveforms are upsampled to.
    n_phi : int
        The number of azimuthal angles to probe in the specified range.
    range_phi_deg : tuple of floats with len = 2 
        The specified range of azimuthal angles to probe.
    n_theta : int 
        The number of zenith angles to probe in the specified range.
    range_theta_deg : tuple of floats with len = 2  
        The specified range of zenith angles to probe.
    '''
    def __init__(self, reader,  upsample=2**14, n_phi=181, range_phi_deg=(-180,180), n_theta=361, range_theta_deg=(0,180), crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=False, waveform_index_range=(None,None)):
        try:
            n = 1.0003 #Index of refraction of air  #Should use https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf 
            self.c = 299792458/n #m/s
            self.reader = reader
            self.reader.setEntry(0)

            self.figs = []
            self.axs = []

            self.n_phi = n_phi
            self.range_phi_deg = range_phi_deg
            self.n_theta = n_theta
            self.range_theta_deg = range_theta_deg

            cable_delays = info.loadCableDelays()
            self.cable_delays = numpy.array([cable_delays['hpol'][0],cable_delays['vpol'][0],cable_delays['hpol'][1],cable_delays['vpol'][1],cable_delays['hpol'][2],cable_delays['vpol'][2],cable_delays['hpol'][3],cable_delays['vpol'][3]])

            #Prepare waveform length handling
            #Allowing for a subset of the waveform to be isolated.  Helpful to speed this up if you know where signals are in long traces.
            self.buffer_length = reader.header().buffer_length
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

            self.upsample = upsample
            self.prepareTimes()

            self.crit_freq_low_pass_MHz = crit_freq_low_pass_MHz
            self.crit_freq_high_pass_MHz = crit_freq_high_pass_MHz
            self.low_pass_filter_order = low_pass_filter_order
            self.high_pass_filter_order = high_pass_filter_order

            self.filter = self.makeFilter(plot_filter=plot_filter)
            if numpy.all(self.filter == 1.0):
                self.apply_filter = False
            else:
                self.apply_filter = True

            self.thetas_deg = numpy.linspace(min(range_theta_deg),max(range_theta_deg),n_theta)
            self.phis_deg = numpy.linspace(min(range_phi_deg),max(range_phi_deg),n_phi)
            self.thetas_rad = numpy.radians(self.thetas_deg)
            self.phis_rad = numpy.radians(self.phis_deg)

            antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU()

            self.A0_physical = numpy.asarray(antennas_physical[0])
            self.A1_physical = numpy.asarray(antennas_physical[1])
            self.A2_physical = numpy.asarray(antennas_physical[2])
            self.A3_physical = numpy.asarray(antennas_physical[3])

            self.A0_hpol = numpy.asarray(antennas_phase_hpol[0])
            self.A1_hpol = numpy.asarray(antennas_phase_hpol[1])
            self.A2_hpol = numpy.asarray(antennas_phase_hpol[2])
            self.A3_hpol = numpy.asarray(antennas_phase_hpol[3])

            self.A0_vpol = numpy.asarray(antennas_phase_vpol[0])
            self.A1_vpol = numpy.asarray(antennas_phase_vpol[1])
            self.A2_vpol = numpy.asarray(antennas_phase_vpol[2])
            self.A3_vpol = numpy.asarray(antennas_phase_vpol[3])

            self.generateTimeIndices()
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

    def prepareTimes(self):
        '''
        Uses the current upsample factor to prepare the upsampled times.  Should be called after
        chaning self.upsample if that is changed.
        '''
        try:
            self.times = self.reader.t()
            self.dt = numpy.diff(self.times)[0]

            self.times_resampled = scipy.signal.resample(numpy.zeros(self.buffer_length),self.upsample,t=self.times)[1]
            self.dt_resampled = numpy.diff(self.times_resampled)[0]
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def overwriteAntennaLocations(self, A0_physical,A1_physical,A2_physical,A3_physical,A0_hpol,A1_hpol,A2_hpol,A3_hpol,A0_vpol,A1_vpol,A2_vpol,A3_vpol):
        '''
        Allows you to set the antenna physical and phase positions, if the default values saved in info.py are not
        what you want. Will then recalculate time delays corresponding to arrival directions.

        A0_physical : float
            The ENU coordinates of the physical location for antenna 0.
        A1_physical : float
            The ENU coordinates of the physical location for antenna 1.
        A2_physical : float
            The ENU coordinates of the physical location for antenna 2.
        A3_physical : float
            The ENU coordinates of the physical location for antenna 3.
        A0_hpol : float
            The ENU coordinates of the hpol phase center location for antenna 0.
        A1_hpol : float
            The ENU coordinates of the hpol phase center location for antenna 1.
        A2_hpol : float
            The ENU coordinates of the hpol phase center location for antenna 2.
        A3_hpol : float
            The ENU coordinates of the hpol phase center location for antenna 3.
        A0_vpol : float
            The ENU coordinates of the vpol phase center location for antenna 0.
        A1_vpol : float
            The ENU coordinates of the vpol phase center location for antenna 1.
        A2_vpol : float
            The ENU coordinates of the vpol phase center location for antenna 2.
        A3_vpol : float
            The ENU coordinates of the vpol phase center location for antenna 3.
        '''
        try:
            self.A0_physical = numpy.asarray(A0_physical)
            self.A1_physical = numpy.asarray(A1_physical)
            self.A2_physical = numpy.asarray(A2_physical)
            self.A3_physical = numpy.asarray(A3_physical)

            self.A0_hpol = numpy.asarray(A0_hpol)
            self.A1_hpol = numpy.asarray(A1_hpol)
            self.A2_hpol = numpy.asarray(A2_hpol)
            self.A3_hpol = numpy.asarray(A3_hpol)

            self.A0_vpol = numpy.asarray(A0_vpol)
            self.A1_vpol = numpy.asarray(A1_vpol)
            self.A2_vpol = numpy.asarray(A2_vpol)
            self.A3_vpol = numpy.asarray(A3_vpol)

            print('Rerunning time delay prep with antenna positions.')
            self.generateTimeIndices()
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def overwriteCableDelays(self, ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7):
        '''
        Allows user to reset the cable delays if the default values are insufficient.
        
        Parameters
        ----------
        ch0 : float
            The new cable delay for channel 0.  Should be given in ns.  Positive values imply
            the signal takes that much longer to propogate through the cable corresponding to
            that channel. 
        ch1 : float
            The new cable delay for channel 1.  Should be given in ns.  Positive values imply
            the signal takes that much longer to propogate through the cable corresponding to
            that channel. 
        ch2 : float
            The new cable delay for channel 2.  Should be given in ns.  Positive values imply
            the signal takes that much longer to propogate through the cable corresponding to
            that channel. 
        ch3 : float
            The new cable delay for channel 3.  Should be given in ns.  Positive values imply
            the signal takes that much longer to propogate through the cable corresponding to
            that channel. 
        ch4 : float
            The new cable delay for channel 4.  Should be given in ns.  Positive values imply
            the signal takes that much longer to propogate through the cable corresponding to
            that channel. 
        ch5 : float
            The new cable delay for channel 5.  Should be given in ns.  Positive values imply
            the signal takes that much longer to propogate through the cable corresponding to
            that channel. 
        ch6 : float
            The new cable delay for channel 6.  Should be given in ns.  Positive values imply
            the signal takes that much longer to propogate through the cable corresponding to
            that channel. 
        ch7 : float
            The new cable delay for channel 7.  Should be given in ns.  Positive values imply
            the signal takes that much longer to propogate through the cable corresponding to
            that channel. 
        '''
        try:
            self.cable_delays = numpy.array([ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7])
            print('Rerunning time delay prep with new cable delays.')
            self.generateTimeIndices()
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def makeFilter(self,plot_filter=False):
        '''
        This will make a frequency domain filter based on the given specifications. 
        '''
        try:
            freqs = numpy.fft.rfftfreq(self.buffer_length,self.dt*1e-9)
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
                ax = fig.gca()
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

                self.figs.append(fig)
                self.axs.append(ax)

            return filter_y
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def wf(self, eventid, channels, div_std=False, hilbert=False, apply_filter=False):
        '''
        Calls reader.wf.  If multiple channels are given it will load them in sorted order into a 2d array with each
        row corresponding to the upsampled waveform (with dc offset removed).  The lowest channels corresponds
        to the lowest row numbers.

        If div_std == True then the output waveforms will each be divided by their respect standard deviations.

        Parameters
        ----------
        eventid : int
            The entry number you wish to plot the correlation map for.
        channels : numpy.ndarray
            List of channels to load.  Typically either even or odd channels for different polarizations.
        div_std : bool
            If True then the returned waveforms have been divided by their respective standard deviations.
            Useful for cross correlation normalization.
        hilbert : bool
            Enables performing calculations with Hilbert envelopes of waveforms. 
        '''
        try:
            self.reader.setEntry(eventid)
            channels = numpy.sort(numpy.asarray(channels))
            temp_waveforms = numpy.zeros((len(channels),self.buffer_length))
            for channel_index, channel in enumerate(channels):
                temp_wf = numpy.asarray(self.reader.wf(int(channel)))[self.start_waveform_index:self.end_waveform_index+1]
                temp_wf = temp_wf - numpy.mean(temp_wf)
                if apply_filter == True:
                    temp_wf = numpy.fft.irfft(numpy.multiply(self.filter,numpy.fft.rfft(temp_wf)),n=self.buffer_length) #Might need additional normalization
                if hilbert == True:
                    temp_wf = abs(scipy.signal.hilbert(temp_wf))
                if div_std:
                    temp_wf = temp_wf/numpy.std(temp_wf)
                temp_waveforms[channel_index] = temp_wf

            waveforms = scipy.signal.resample(temp_waveforms,self.upsample,axis=1)

            return waveforms
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

    def generateTimeIndices(self):
        '''
        This is meant to calculate all of the time delays corresponding to each source direction in advance.
        '''
        try:
            #Prepare grids of thetas and phis
            thetas  = numpy.tile(self.thetas_rad,(self.n_phi,1)).T  #Each row is a different theta (zenith)
            phis    = numpy.tile(self.phis_rad,(self.n_theta,1))    #Each column is a different phi (azimuth)

            #Source direction is the direction FROM BEACON you look to see the source.
            signal_source_direction        = numpy.zeros((self.n_theta, self.n_phi, 3))
            signal_source_direction[:,:,0] = numpy.multiply( numpy.cos(phis) , numpy.sin(thetas) )
            signal_source_direction[:,:,1] = numpy.multiply( numpy.sin(phis) , numpy.sin(thetas) )
            signal_source_direction[:,:,2] = numpy.cos(thetas)
            #Propogate direction is the direction FROM THE SOURCE that the ray travels towards BEACON.  
            signal_propogate_direction = - signal_source_direction

            hpol_baseline_vectors = numpy.array([   self.A0_hpol - self.A1_hpol,\
                                                    self.A0_hpol - self.A2_hpol,\
                                                    self.A0_hpol - self.A3_hpol,\
                                                    self.A1_hpol - self.A2_hpol,\
                                                    self.A1_hpol - self.A3_hpol,\
                                                    self.A2_hpol - self.A3_hpol])

            vpol_baseline_vectors = numpy.array([   self.A0_vpol - self.A1_vpol,\
                                                    self.A0_vpol - self.A2_vpol,\
                                                    self.A0_vpol - self.A3_vpol,\
                                                    self.A1_vpol - self.A2_vpol,\
                                                    self.A1_vpol - self.A3_vpol,\
                                                    self.A2_vpol - self.A3_vpol])

            self.t_hpol_0subtract1 = ((hpol_baseline_vectors[0][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[0][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[0][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[0] - self.cable_delays[2]) #ns
            self.t_hpol_0subtract2 = ((hpol_baseline_vectors[1][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[1][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[1][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[0] - self.cable_delays[4]) #ns
            self.t_hpol_0subtract3 = ((hpol_baseline_vectors[2][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[2][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[2][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[0] - self.cable_delays[6]) #ns
            self.t_hpol_1subtract2 = ((hpol_baseline_vectors[3][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[3][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[3][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[2] - self.cable_delays[4]) #ns
            self.t_hpol_1subtract3 = ((hpol_baseline_vectors[4][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[4][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[4][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[2] - self.cable_delays[6]) #ns
            self.t_hpol_2subtract3 = ((hpol_baseline_vectors[5][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[5][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[5][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[4] - self.cable_delays[6]) #ns

            self.t_vpol_0subtract1 = ((vpol_baseline_vectors[0][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[0][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[0][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[1] - self.cable_delays[3]) #ns
            self.t_vpol_0subtract2 = ((vpol_baseline_vectors[1][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[1][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[1][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[1] - self.cable_delays[5]) #ns
            self.t_vpol_0subtract3 = ((vpol_baseline_vectors[2][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[2][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[2][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[1] - self.cable_delays[7]) #ns
            self.t_vpol_1subtract2 = ((vpol_baseline_vectors[3][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[3][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[3][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[3] - self.cable_delays[5]) #ns
            self.t_vpol_1subtract3 = ((vpol_baseline_vectors[4][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[4][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[4][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[3] - self.cable_delays[7]) #ns
            self.t_vpol_2subtract3 = ((vpol_baseline_vectors[5][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[5][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[5][2]*signal_propogate_direction[:,:,2])/self.c)*1e9 + (self.cable_delays[5] - self.cable_delays[7]) #ns

            #Should double check when using these via rolling signals.
            #Calculate indices in corr for each direction.
            center = len(self.times_resampled)

            self.delay_indices_hpol_0subtract1 = numpy.rint((self.t_hpol_0subtract1/self.dt_resampled + center)).astype(int)
            self.delay_indices_hpol_0subtract2 = numpy.rint((self.t_hpol_0subtract2/self.dt_resampled + center)).astype(int)
            self.delay_indices_hpol_0subtract3 = numpy.rint((self.t_hpol_0subtract3/self.dt_resampled + center)).astype(int)
            self.delay_indices_hpol_1subtract2 = numpy.rint((self.t_hpol_1subtract2/self.dt_resampled + center)).astype(int)
            self.delay_indices_hpol_1subtract3 = numpy.rint((self.t_hpol_1subtract3/self.dt_resampled + center)).astype(int)
            self.delay_indices_hpol_2subtract3 = numpy.rint((self.t_hpol_2subtract3/self.dt_resampled + center)).astype(int)

            self.delay_indices_vpol_0subtract1 = numpy.rint((self.t_vpol_0subtract1/self.dt_resampled + center)).astype(int)
            self.delay_indices_vpol_0subtract2 = numpy.rint((self.t_vpol_0subtract2/self.dt_resampled + center)).astype(int)
            self.delay_indices_vpol_0subtract3 = numpy.rint((self.t_vpol_0subtract3/self.dt_resampled + center)).astype(int)
            self.delay_indices_vpol_1subtract2 = numpy.rint((self.t_vpol_1subtract2/self.dt_resampled + center)).astype(int)
            self.delay_indices_vpol_1subtract3 = numpy.rint((self.t_vpol_1subtract3/self.dt_resampled + center)).astype(int)
            self.delay_indices_vpol_2subtract3 = numpy.rint((self.t_vpol_2subtract3/self.dt_resampled + center)).astype(int)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def interactivePlotter(self, event):
        '''
        This hopefully will make a plot when called by a double clock in the map.
        '''
        if event.dblclick == True:
            try:
                event_ax = event.inaxes
                pol = event_ax.get_title().split('-')[2]
                eventid = int(event_ax.get_title().split('-')[1])
                hilbert = literal_eval(event_ax.get_title().split('-')[3].split('=')[1])

                theta_index = numpy.argmin( abs(self.thetas_deg - event.ydata) )
                phi_index = numpy.argmin( abs(self.phis_deg - event.xdata) )
                if pol == 'hpol':
                    channels = numpy.array([0,2,4,6])
                    waveforms = self.wf(eventid, channels,div_std=False,hilbert=hilbert,apply_filter=self.apply_filter)
                    t_best_0subtract1 = self.t_hpol_0subtract1[theta_index,phi_index]
                    t_best_0subtract2 = self.t_hpol_0subtract2[theta_index,phi_index]
                    t_best_0subtract3 = self.t_hpol_0subtract3[theta_index,phi_index]
                elif pol == 'vpol':
                    channels = numpy.array([1,3,5,7])
                    waveforms = self.wf(eventid, channels,div_std=False,hilbert=hilbert,apply_filter=self.apply_filter)
                    t_best_0subtract1 = self.t_vpol_0subtract1[theta_index,phi_index]
                    t_best_0subtract2 = self.t_vpol_0subtract2[theta_index,phi_index]
                    t_best_0subtract3 = self.t_vpol_0subtract3[theta_index,phi_index]

                #Determine how many indices to roll each waveform.
                roll0 = 0
                roll1 = int(numpy.rint(t_best_0subtract1/self.dt_resampled))
                roll2 = int(numpy.rint(t_best_0subtract2/self.dt_resampled))
                roll3 = int(numpy.rint(t_best_0subtract3/self.dt_resampled))

                try:
                    plt.close(self.popout_fig)
                except:
                    pass #Just maintaining only one popout at a time.
                self.popout_fig = plt.figure()
                self.popout_ax = self.popout_fig.gca()
                plt.suptitle('%s\nAzimuth = %0.3f, Zenith = %0.3f'%(event_ax.get_title().replace('-',' ').title(), event.xdata,event.ydata))
                plt.subplot(2,1,1)
                
                plt.plot(self.times_resampled, waveforms[0],label='Ch%i'%channels[0])
                plt.plot(self.times_resampled, waveforms[1],label='Ch%i'%(channels[1]))
                plt.plot(self.times_resampled, waveforms[2],label='Ch%i'%(channels[2]))
                plt.plot(self.times_resampled, waveforms[3],label='Ch%i'%(channels[3]))

                plt.ylabel('adu')
                plt.xlabel('Time (ns)')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.legend()

                plt.subplot(2,1,2)
                
                plt.plot(self.times_resampled, waveforms[0],label='Ch%i'%channels[0])
                plt.plot(self.times_resampled + t_best_0subtract1, waveforms[1],label='Ch%i, shifted %0.2f ns'%(channels[1], t_best_0subtract1))
                plt.plot(self.times_resampled + t_best_0subtract2, waveforms[2],label='Ch%i, shifted %0.2f ns'%(channels[2], t_best_0subtract2))
                plt.plot(self.times_resampled + t_best_0subtract3, waveforms[3],label='Ch%i, shifted %0.2f ns'%(channels[3], t_best_0subtract3))

                plt.ylabel('adu')
                plt.xlabel('Time (ns)')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.legend()

            except Exception as e:
                print('\nError in %s'%inspect.stack()[0][3])
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

    def mapMax(self, map_values, max_method=0):
        '''
        Determines the indices in the given map of the optimal source direction.

        Parameters
        ----------
        map_values : numpy.ndarray of floats
            The correlation map.
        max_method : int
            The index of the method you would like to use.
            0 : argmax of map. (Default).
            1 : Averages each point in the map with the four surrounding points (by rolling map by 1 index in each direction and summing).

        Returns
        -------
        row_index : int
            The selected row.
        col_index : int
            The selected column.
        '''
        if max_method == 0:
            row_index, column_index = numpy.unravel_index(map_values.argmax(),numpy.shape(map_values))
        elif max_method == 1:
            #Calculates sum of each point plus surrounding four points to get max.
            rounded_corr_values = (map_values + numpy.roll(map_values,1,axis=0) + numpy.roll(map_values,-1,axis=0) + numpy.roll(map_values,1,axis=1) + numpy.roll(map_values,-1,axis=1))/5.0
            row_index, column_index = numpy.unravel_index(rounded_corr_values.argmax(),numpy.shape(rounded_corr_values))
        return row_index, column_index


    def map(self, eventid, pol, plot_map=True, plot_corr=False, hilbert=False, interactive=False, max_method=None):
        '''
        Makes the cross correlation make for the given event.

        Parameters
        ----------
        eventid : int
            The entry number you wish to plot the correlation map for.
        pol : str
            The polarization you wish to plot.  Options: 'hpol', 'vpol', 'both'
        plot_map : bool
            Whether to actually plot the results.  
        plot_corr : bool
            Plot the cross-correlations for each baseline.
        hilbert : bool
            Enables performing calculations with Hilbert envelopes of waveforms. 
        interactive : bool
            Enables an interactive correlation map, where double clicking will result in a plot
            of waveforms aligned using the corresponding time delays of that location.
        max_method : bool
            Determines how the most probable source direction is from the map.
        '''
        try:
            if hilbert == True:
                print('WARNING! Enabling Hilbert envelopes throws off correlation normalization.')
            if pol == 'both':
                hpol_result = self.map(eventid,'hpol', plot_map=plot_map, plot_corr=plot_corr, hilbert=hilbert)
                vpol_result = self.map(eventid,'vpol', plot_map=plot_map, plot_corr=plot_corr, hilbert=hilbert)
                return hpol_result, vpol_result

            elif pol == 'hpol':
                waveforms = self.wf(eventid, numpy.array([0,2,4,6]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter) #Div by std and resampled waveforms normalizes the correlations

                corr01 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[1])))/(len(self.times_resampled))
                corr02 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[2])))/(len(self.times_resampled))
                corr03 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[3])))/(len(self.times_resampled))
                corr12 = (numpy.asarray(scipy.signal.correlate(waveforms[1],waveforms[2])))/(len(self.times_resampled))
                corr13 = (numpy.asarray(scipy.signal.correlate(waveforms[1],waveforms[3])))/(len(self.times_resampled))
                corr23 = (numpy.asarray(scipy.signal.correlate(waveforms[2],waveforms[3])))/(len(self.times_resampled))
                
                corr_value_0subtract1 = corr01[self.delay_indices_hpol_0subtract1]
                corr_value_0subtract2 = corr02[self.delay_indices_hpol_0subtract2]
                corr_value_0subtract3 = corr03[self.delay_indices_hpol_0subtract3]
                corr_value_1subtract2 = corr12[self.delay_indices_hpol_1subtract2]
                corr_value_1subtract3 = corr13[self.delay_indices_hpol_1subtract3]
                corr_value_2subtract3 = corr23[self.delay_indices_hpol_2subtract3]

                mean_corr_values = numpy.mean(numpy.array([corr_value_0subtract1, corr_value_0subtract2, corr_value_0subtract3, corr_value_1subtract2, corr_value_1subtract3, corr_value_2subtract3] ),axis=0)

                if max_method is not None:
                    a1, a2 = self.mapMax(mean_corr_values,max_method=max_method)
                else:
                    a1, a2 = self.mapMax(mean_corr_values)

                theta_best  = self.thetas_deg[a1]
                phi_best    = self.phis_deg[a2]

                t_best_0subtract1 = self.t_hpol_0subtract1[a1,a2]
                t_best_0subtract2 = self.t_hpol_0subtract2[a1,a2]
                t_best_0subtract3 = self.t_hpol_0subtract3[a1,a2]
                t_best_1subtract2 = self.t_hpol_1subtract2[a1,a2]
                t_best_1subtract3 = self.t_hpol_1subtract3[a1,a2]
                t_best_2subtract3 = self.t_hpol_2subtract3[a1,a2]

            elif pol == 'vpol':
                waveforms = self.wf(eventid, numpy.array([1,3,5,7]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter) #Div by std and resampled waveforms normalizes the correlations

                corr01 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[1])))/(len(self.times_resampled))
                corr02 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[2])))/(len(self.times_resampled))
                corr03 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[3])))/(len(self.times_resampled))
                corr12 = (numpy.asarray(scipy.signal.correlate(waveforms[1],waveforms[2])))/(len(self.times_resampled))
                corr13 = (numpy.asarray(scipy.signal.correlate(waveforms[1],waveforms[3])))/(len(self.times_resampled))
                corr23 = (numpy.asarray(scipy.signal.correlate(waveforms[2],waveforms[3])))/(len(self.times_resampled))

                corr_value_0subtract1 = corr01[self.delay_indices_vpol_0subtract1]
                corr_value_0subtract2 = corr02[self.delay_indices_vpol_0subtract2]
                corr_value_0subtract3 = corr03[self.delay_indices_vpol_0subtract3]
                corr_value_1subtract2 = corr12[self.delay_indices_vpol_1subtract2]
                corr_value_1subtract3 = corr13[self.delay_indices_vpol_1subtract3]
                corr_value_2subtract3 = corr23[self.delay_indices_vpol_2subtract3]

                mean_corr_values = numpy.mean(numpy.array([corr_value_0subtract1, corr_value_0subtract2, corr_value_0subtract3, corr_value_1subtract2, corr_value_1subtract3, corr_value_2subtract3] ),axis=0)
                
                if max_method is not None:
                    a1, a2 = self.mapMax(mean_corr_values,max_method=max_method)
                else:
                    a1, a2 = self.mapMax(mean_corr_values)

                theta_best  = self.thetas_deg[a1]
                phi_best    = self.phis_deg[a2]

                t_best_0subtract1 = self.t_vpol_0subtract1[a1,a2]
                t_best_0subtract2 = self.t_vpol_0subtract2[a1,a2]
                t_best_0subtract3 = self.t_vpol_0subtract3[a1,a2]
                t_best_1subtract2 = self.t_vpol_1subtract2[a1,a2]
                t_best_1subtract3 = self.t_vpol_1subtract3[a1,a2]
                t_best_2subtract3 = self.t_vpol_2subtract3[a1,a2]

            else:
                print('Invalid polarization option.  Returning nothing.')
                return



            if plot_corr:
                
                fig = plt.figure()
                fig.canvas.set_window_title('Correlations')
                center = len(self.times_resampled)
                shifts = (numpy.arange(len(corr01)) - center + 1)*self.dt_resampled
                if True:
                    #To check normalization.  Should be 1 at 0 time. 
                    corr00 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[0])))/len(self.times_resampled) 
                    plt.plot(shifts,corr00,alpha=0.7,label='autocorr00')
                plt.plot(shifts,corr01,alpha=0.7,label='corr01')
                plt.plot(shifts,corr02,alpha=0.7,label='corr02')
                plt.plot(shifts,corr03,alpha=0.7,label='corr03')
                plt.plot(shifts,corr12,alpha=0.7,label='corr12')
                plt.plot(shifts,corr13,alpha=0.7,label='corr13')
                plt.plot(shifts,corr23,alpha=0.7,label='corr23')
                plt.legend()
                self.figs.append(fig)
                self.axs.append(fig.gca())
            if plot_map:
                fig = plt.figure()
                fig.canvas.set_window_title('r%i-e%i-%s Correlation Map'%(reader.run,eventid,pol.title()))
                ax = fig.add_subplot(1,1,1)
                ax.set_title('%i-%i-%s-Hilbert=%s'%(reader.run,eventid,pol,str(hilbert))) #FORMATTING SPECIFIC AND PARSED ELSEWHERE, DO NOT CHANGE. 
                im = ax.imshow(mean_corr_values, interpolation='none', extent=[min(self.phis_deg),max(self.phis_deg),max(self.thetas_deg),min(self.thetas_deg)],cmap=plt.cm.coolwarm) #cmap=plt.cm.jet)
                cbar = fig.colorbar(im)
                cbar.set_label('Mean Correlation Value')
                plt.xlabel('Azimuth Angle (Degrees)')
                plt.ylabel('Zenith Angle (Degrees)')

                print("From the correlation plot:")
                print("Best zenith angle:",theta_best)
                print("Best azimuth angle:",phi_best)
                print('Predicted time delays %run between A0 and A1:', t_best_0subtract1)
                print('Predicted time delays between A0 and A2:', t_best_0subtract2)
                print('Predicted time delays between A0 and A3:', t_best_0subtract3)
                print('Predicted time delays between A1 and A2:', t_best_1subtract2)
                print('Predicted time delays between A1 and A3:', t_best_1subtract3)
                print('Predicted time delays between A2 and A3:', t_best_2subtract3)

                radius = 5.0 #Degrees I think?  Should eventually represent error. 
                circle = plt.Circle((phi_best, theta_best), radius, edgecolor='lime',linewidth=2,fill=False)
                ax.axvline(phi_best,c='lime',linewidth=1)
                ax.axhline(theta_best,c='lime',linewidth=1)

                ax.add_artist(circle)
                if interactive == True:
                    fig.canvas.mpl_connect('button_press_event',self.interactivePlotter)

                self.figs.append(fig)
                self.axs.append(ax)

                
                return mean_corr_values, fig, ax
            else:
                return mean_corr_values
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def averagedMap(self, eventids, pol, plot_map=True, hilbert=False, max_method=None):
        '''
        Does the same thing as map, but averages over all eventids given.  Mostly helpful for 
        repeated sources such as background sources or pulsers.

        Parameters
        ----------
        eventids : numpy.ndarray of ints
            The entry numbers to include in the calculation.
        pol : str
            The polarization you wish to plot.  Options: 'hpol', 'vpol', 'both'
        plot_map : bool
            Whether to actually plot the results.  
        plot_corr : bool
            Plot the cross-correlations for each baseline.
        hilbert : bool
            Enables performing calculations with Hilbert envelopes of waveforms. 
        max_method : bool
            Determines how the most probable source direction is from the map.
        '''
        if pol == 'both':
            hpol_result = self.averagedMap(eventids, 'hpol', plot_map=plot_map, hilbert=hilbert)
            vpol_result = self.averagedMap(eventids, 'vpol', plot_map=plot_map, hilbert=hilbert)
            return hpol_result, vpol_result

        total_mean_corr_values = numpy.zeros((self.n_theta, self.n_phi))
        for event_index, eventid in enumerate(eventids):
            sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
            sys.stdout.flush()
            total_mean_corr_values += self.map(eventid, pol, plot_map=False, plot_corr=False, hilbert=hilbert)/len(eventids)
        print('')

        if max_method is not None:
            a1, a2 = self.mapMax(total_mean_corr_values,max_method=max_method)
        else:
            a1, a2 = self.mapMax(total_mean_corr_values)
        
        theta_best  = self.thetas_deg[a1]
        phi_best    = self.phis_deg[a2]

        if pol == 'hpol':
            t_best_0subtract1  = self.t_hpol_0subtract1[a1,a2]
            t_best_0subtract2  = self.t_hpol_0subtract2[a1,a2]
            t_best_0subtract3  = self.t_hpol_0subtract3[a1,a2]
            t_best_1subtract2  = self.t_hpol_1subtract2[a1,a2]
            t_best_1subtract3  = self.t_hpol_1subtract3[a1,a2]
            t_best_2subtract3  = self.t_hpol_2subtract3[a1,a2]
        elif pol == 'vpol':
            t_best_0subtract1  = self.t_vpol_0subtract1[a1,a2]
            t_best_0subtract2  = self.t_vpol_0subtract2[a1,a2]
            t_best_0subtract3  = self.t_vpol_0subtract3[a1,a2]
            t_best_1subtract2  = self.t_vpol_1subtract2[a1,a2]
            t_best_1subtract3  = self.t_vpol_1subtract3[a1,a2]
            t_best_2subtract3  = self.t_vpol_2subtract3[a1,a2]
        else:
            print('Invalid polarization option.  Returning nothing.')
            return


        if plot_map:
            fig = plt.figure()
            fig.canvas.set_window_title('r%i %s Averaged Correlation Map'%(reader.run,pol.title()))
            ax = fig.add_subplot(1,1,1)
            im = ax.imshow(total_mean_corr_values, interpolation='none', extent=[min(self.phis_deg),max(self.phis_deg),max(self.thetas_deg),min(self.thetas_deg)],cmap=plt.cm.coolwarm) #cmap=plt.cm.jet)
            cbar = fig.colorbar(im)
            cbar.set_label('Mean Correlation Value')
            plt.xlabel('Azimuth Angle (Degrees)')
            plt.ylabel('Zenith Angle (Degrees)')

            print('From the correlation plot:')
            print('Best zenith angle:',theta_best)
            print('Best azimuth angle:',phi_best)
            print('Predicted time delays %run between A0 and A1:', t_best_0subtract1)
            print('Predicted time delays between A0 and A2:', t_best_0subtract2)
            print('Predicted time delays between A0 and A3:', t_best_0subtract3)
            print('Predicted time delays between A1 and A2:', t_best_1subtract2)
            print('Predicted time delays between A1 and A3:', t_best_1subtract3)
            print('Predicted time delays between A2 and A3:', t_best_2subtract3)

            radius = 5.0 #Degrees I think?  Should eventually represent error. 
            circle = plt.Circle((phi_best, theta_best), radius, edgecolor='lime',linewidth=2,fill=False)
            ax.axvline(phi_best,c='lime',linewidth=1)
            ax.axhline(theta_best,c='lime',linewidth=1)
            ax.add_artist(circle)
            self.figs.append(fig)
            self.axs.append(ax)
            
            return total_mean_corr_values, fig, ax
        else:
            return total_mean_corr_values

def testMain():
    '''
    This was used for testing.
    '''
    if len(sys.argv) == 2:
        if str(sys.argv[1]) in ['vpol', 'hpol']:
            mode = str(sys.argv[1])
        else:
            print('Given mode not in options.  Defaulting to vpol')
            mode = 'vpol'
    else:
        print('No mode given.  Defaulting to vpol')
        mode = 'vpol'

    datapath = os.environ['BEACON_DATA']

    all_figs = []
    all_axs = []
    all_cors = []

    for run in [1507,1509,1511]:

        if run == 1507:
            waveform_index_range = (1500,None) #Looking at the later bit of the waveform only, 10000 will cap off.  
        elif run == 1509:
            waveform_index_range = (2000,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
        elif run == 1511:
            waveform_index_range = (1250,2000) #Looking at the later bit of the waveform only, 10000 will cap off.  
        else:
            waveform_index_range = (None,None)

        reader = Reader(datapath,run)

        crit_freq_low_pass_MHz = None#70 #This new pulser seems to peak in the region of 85 MHz or so
        low_pass_filter_order = None#12

        crit_freq_high_pass_MHz = None#55
        high_pass_filter_order = None#4
        plot_filter=True

        known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
        eventids = {}
        eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])
        eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])
        all_eventids = numpy.sort(numpy.append(eventids['hpol'],eventids['vpol']))

        hpol_eventids_cut = numpy.isin(all_eventids,eventids['hpol'])
        vpol_eventids_cut = numpy.isin(all_eventids,eventids['vpol'])

        cor = Correlator(reader,  upsample=2**15, n_phi=361, n_theta=361, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False)
        if True:
            for mode in ['hpol','vpol']:
                eventid = eventids[mode][0]
                mean_corr_values, fig, ax = cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=False, interactive=True)

                pulser_locations_ENU = info.loadPulserPhaseLocationsENU()[mode]['run%i'%run]
                pulser_locations_ENU = pulser_locations_ENU/numpy.linalg.norm(pulser_locations_ENU)
                pulser_theta = numpy.degrees(numpy.arccos(pulser_locations_ENU[2]))
                pulser_phi = numpy.degrees(numpy.arctan(pulser_locations_ENU[1]/pulser_locations_ENU[0]))
                print('%s Expected pulser location: Zenith = %0.2f, Az = %0.2f'%(mode.title(), pulser_theta,pulser_phi))

                ax.axvline(pulser_phi,c='r')
                ax.axhline(pulser_theta,c='r')

                all_figs.append(fig)
                all_axs.append(ax)
        if False:
            mean_corr_values, fig, ax = cor.averagedMap(eventids[mode], mode, plot_map=True, hilbert=False)

            pulser_locations_ENU = info.loadPulserPhaseLocationsENU()[mode]['run%i'%run]
            pulser_locations_ENU = pulser_locations_ENU/numpy.linalg.norm(pulser_locations_ENU)
            pulser_theta = numpy.degrees(numpy.arccos(pulser_locations_ENU[2]))
            pulser_phi = numpy.degrees(numpy.arctan(pulser_locations_ENU[1]/pulser_locations_ENU[0]))
            print('%s Expected pulser location: Zenith = %0.2f, Az = %0.2f'%(mode.title(), pulser_theta,pulser_phi))

            ax.axvline(pulser_phi,c='r')
            ax.axhline(pulser_theta,c='r')

            all_figs.append(fig)
            all_axs.append(ax)
        all_cors.append(cor)

if __name__=="__main__":
    crit_freq_low_pass_MHz = 60 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = 4

    crit_freq_high_pass_MHz = 30
    high_pass_filter_order = 4
    plot_filter=True

    max_method = 0
    
    if len(sys.argv) == 3:
        run = int(sys.argv[1])
        eventid = int(sys.argv[2])
    
        datapath = os.environ['BEACON_DATA']

        all_figs = []
        all_axs = []
        all_cors = []

        if run == 1507:
            waveform_index_range = (1500,None) #Looking at the later bit of the waveform only, 10000 will cap off.  
        elif run == 1509:
            waveform_index_range = (2000,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
        elif run == 1511:
            waveform_index_range = (1250,2000) #Looking at the later bit of the waveform only, 10000 will cap off.  
        else:
            waveform_index_range = (None,None)

        reader = Reader(datapath,run)

        cor = Correlator(reader,  upsample=2**15, n_phi=420, n_theta=420, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter)

        for mode in ['hpol','vpol']:
            mean_corr_values, fig, ax = cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=False, interactive=True)
            all_figs.append(fig)
            all_axs.append(ax)

        all_cors.append(cor)
    else:
        datapath = os.environ['BEACON_DATA']

        all_figs = []
        all_axs = []
        all_cors = []

        for run in [1507,1509,1511]:

            if run == 1507:
                waveform_index_range = (1500,None) #Looking at the later bit of the waveform only, 10000 will cap off.  
            elif run == 1509:
                waveform_index_range = (2000,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
            elif run == 1511:
                waveform_index_range = (1250,2000) #Looking at the later bit of the waveform only, 10000 will cap off.  
            else:
                waveform_index_range = (None,None)

            reader = Reader(datapath,run)
            plot_filter=True

            known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
            eventids = {}
            eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])
            eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])
            all_eventids = numpy.sort(numpy.append(eventids['hpol'],eventids['vpol']))

            hpol_eventids_cut = numpy.isin(all_eventids,eventids['hpol'])
            vpol_eventids_cut = numpy.isin(all_eventids,eventids['vpol'])

            cor = Correlator(reader,  upsample=2**15, n_phi=420, n_theta=420, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter)
            if True:
                for mode in ['hpol','vpol']:
                    eventid = eventids[mode][0]
                    mean_corr_values, fig, ax = cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=False, interactive=True, max_method=max_method)

                    pulser_locations_ENU = info.loadPulserPhaseLocationsENU()[mode]['run%i'%run]
                    pulser_locations_ENU = pulser_locations_ENU/numpy.linalg.norm(pulser_locations_ENU)
                    pulser_theta = numpy.degrees(numpy.arccos(pulser_locations_ENU[2]))
                    pulser_phi = numpy.degrees(numpy.arctan(pulser_locations_ENU[1]/pulser_locations_ENU[0]))
                    print('%s Expected pulser location: Zenith = %0.2f, Az = %0.2f'%(mode.title(), pulser_theta,pulser_phi))

                    ax.axvline(pulser_phi,c='r')
                    ax.axhline(pulser_theta,c='r')

                    all_figs.append(fig)
                    all_axs.append(ax)
            if False:
                for mode in ['hpol','vpol']:
                    mean_corr_values, fig, ax = cor.averagedMap(eventids[mode], mode, plot_map=True, hilbert=False, max_method=max_method)

                    pulser_locations_ENU = info.loadPulserPhaseLocationsENU()[mode]['run%i'%run]
                    pulser_locations_ENU = pulser_locations_ENU/numpy.linalg.norm(pulser_locations_ENU)
                    pulser_theta = numpy.degrees(numpy.arccos(pulser_locations_ENU[2]))
                    pulser_phi = numpy.degrees(numpy.arctan(pulser_locations_ENU[1]/pulser_locations_ENU[0]))
                    print('%s Expected pulser location: Zenith = %0.2f, Az = %0.2f'%(mode.title(), pulser_theta,pulser_phi))

                    ax.axvline(pulser_phi,c='r')
                    ax.axhline(pulser_theta,c='r')

                all_figs.append(fig)
                all_axs.append(ax)
            all_cors.append(cor)
