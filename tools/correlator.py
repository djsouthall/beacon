'''
This class is adopted from a script originally written by Kaeli Hughes and as been significantly
restructured/altered for my BEACON analysis framework.  

The purpose of this script is toprovide tools for plotting cross correlation maps for 
interpretting signal directions of BEACON signals.

'''
import os
import sys
import gc
import pymap3d as pm
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
import analysis.phase_response as pr
import tools.get_plane_tracks as pt
from tools.fftmath import FFTPrepper, TimeDelayCalculator
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import scipy.signal
import scipy.stats
import numpy
import sys
import math
from scipy.fftpack import fft
import datetime as dt
from ast import literal_eval
import astropy.units as apu
from astropy.coordinates import SkyCoord
plt.ion()
import inspect
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
    plot_filter : bool
        Enables plotting of the generated filters to be used.
    waveform_index_range : tuple
        Tuple of values.  If the first is None then the signals will be loaded from the beginning of their default buffer,
        otherwise it will start at this index.  If the second is None then the window will go to the end of the default
        buffer, otherwise it will end in this.  
        Essentially waveforms will be loaded using wf = self.reader.wf(channel)[waveform_index_range[0]:waveform_index_range[1]]
        Bounds will be adjusted based on buffer length (in case of overflow. )
    apply_phase_response : bool
            If True, then the phase response will be included with the filter for each channel.  This hopefully 
            deconvolves the effects of the phase response in the signal. 
    tukey : bool
        If True to loaded wf will have tapered edges of the waveform on the 1% level to help
        with edge effects.  This will be applied before hilbert if hilbert is true, and before
        the filter.
    sine_subtract : bool
        If True then any added sine_subtraction methods will be applied to any loaded waveform.  These must be added to
        the prep object.  
        Example: cor.prep.addSineSubtract(0.03, 0.090, 0.05, max_failed_iterations=3, verbose=True, plot=False)
    '''
    def __init__(self, reader,  upsample=None, n_phi=181, range_phi_deg=(-180,180), n_theta=361, range_theta_deg=(0,180), crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=False, waveform_index_range=(None,None), apply_phase_response=False, tukey=False, sine_subtract=True):
        try:
            n = 1.0003 #Index of refraction of air  #Should use https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf 
            self.c = 299792458.0/n #m/s
            self.min_elevation_linewidth = 0.5
            self.reader = reader
            self.upsample = upsample

            self.apply_tukey = tukey
            self.apply_sine_subtract = sine_subtract
            '''
            Note that the definition of final_corr_length in FFTPrepper and upsample in this are different (off by about 
            a factor of 2).  final_corr_length is not actually being used in correlator, all signal upsampling happens
            internally.  FFTPrepper is being used to apply filters to the loaded waveforms such that any changes to
            filtering only need to happen in a single location/class.
            '''
            self.prep = FFTPrepper(self.reader, final_corr_length=2**10, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=plot_filter,tukey_alpha=0.1,tukey_default=False,apply_phase_response=apply_phase_response)
            self.prepareTimes()

            self.figs = []
            self.axs = []
            self.animations = []

            cable_delays = info.loadCableDelays()
            self.cable_delays = numpy.array([cable_delays['hpol'][0],cable_delays['vpol'][0],cable_delays['hpol'][1],cable_delays['vpol'][1],cable_delays['hpol'][2],cable_delays['vpol'][2],cable_delays['hpol'][3],cable_delays['vpol'][3]])

            self.range_theta_deg = range_theta_deg
            self.n_theta = n_theta
            self.range_phi_deg = range_phi_deg
            self.n_phi = n_phi
            self.thetas_deg = numpy.linspace(min(range_theta_deg),max(range_theta_deg),n_theta) #Zenith angle
            self.phis_deg = numpy.linspace(min(range_phi_deg),max(range_phi_deg),n_phi) #Azimuth angle
            self.thetas_rad = numpy.deg2rad(self.thetas_deg)
            self.phis_rad = numpy.deg2rad(self.phis_deg)

            self.mesh_azimuth_rad, self.mesh_elevation_rad = numpy.meshgrid(numpy.deg2rad(numpy.linspace(min(range_phi_deg),max(range_phi_deg),n_phi+1)), numpy.pi/2.0 - numpy.deg2rad(numpy.linspace(min(range_theta_deg),max(range_theta_deg),n_theta+1)))
            self.mesh_azimuth_deg, self.mesh_elevation_deg = numpy.meshgrid(numpy.linspace(min(range_phi_deg),max(range_phi_deg),n_phi+1), 90.0 - numpy.linspace(min(range_theta_deg),max(range_theta_deg),n_theta+1))
            self.mesh_zenith_deg = 90.0 - self.mesh_elevation_deg

            self.A0_latlonel = info.loadAntennaZeroLocation() #Used for conversion to RA and Dec coordinates.

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
            self.calculateArrayNormalVector()

            if numpy.all(self.prep.filter_original == 1.0):
                self.apply_filter = False
            else:
                self.apply_filter = True
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def overwriteAntennaLocations(self, A0_physical,A1_physical,A2_physical,A3_physical,A0_hpol,A1_hpol,A2_hpol,A3_hpol,A0_vpol,A1_vpol,A2_vpol,A3_vpol,verbose=False):
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
            if A0_physical is not None:
                self.A0_physical = numpy.asarray(A0_physical)
            if A1_physical is not None:
                self.A1_physical = numpy.asarray(A1_physical)
            if A2_physical is not None:
                self.A2_physical = numpy.asarray(A2_physical)
            if A3_physical is not None:
                self.A3_physical = numpy.asarray(A3_physical)

            if A0_hpol is not None:
                self.A0_hpol = numpy.asarray(A0_hpol)
            if A1_hpol is not None:
                self.A1_hpol = numpy.asarray(A1_hpol)
            if A2_hpol is not None:
                self.A2_hpol = numpy.asarray(A2_hpol)
            if A3_hpol is not None:
                self.A3_hpol = numpy.asarray(A3_hpol)

            if A0_vpol is not None:
                self.A0_vpol = numpy.asarray(A0_vpol)
            if A1_vpol is not None:
                self.A1_vpol = numpy.asarray(A1_vpol)
            if A2_vpol is not None:
                self.A2_vpol = numpy.asarray(A2_vpol)
            if A3_vpol is not None:
                self.A3_vpol = numpy.asarray(A3_vpol)
            if verbose:
                print('Rerunning time delay prep with antenna positions.')
            self.generateTimeIndices()
            self.calculateArrayNormalVector()
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateArrayNormalVector(self,plot_map=False,mollweide=False, pol='both'):
        '''
        This function will determine the normal vector to the plane defined by the array.  This can be
        used with a dot product of expected directions to determine which scenario might be more likely.

        I.e. We expect most real signals to result in a positive dot product with this vector (as it points)
        outward toward the sky).  A neigitive dot product is more commonly associate with reflections. 

        Note that because there are 4 antennas, the plane of the array is not well defined.  I have chose to define
        it using the points for a0 and a2, and the midpoint between a1 and a3.  
        '''
        try:
            n_func = lambda v0, v1, v2 : - numpy.cross( v1 - v0 , v2 - v1 )/numpy.linalg.norm(- numpy.cross( v1 - v0 , v2 - v1 ))

            self.n_physical = n_func(self.A0_physical, self.A2_physical, (self.A0_physical + self.A3_physical)/2.0)
            self.n_hpol     = n_func(self.A0_hpol,     self.A2_hpol,     (self.A0_hpol     + self.A3_hpol)/2.0)
            self.n_vpol     = n_func(self.A0_vpol,     self.A2_vpol,     (self.A0_vpol     + self.A3_vpol)/2.0)

            self.generateNormalDotValues()

            if plot_map == True:
                self.plotArrayNormalVector(mollweide=mollweide,pol=pol)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def plotArrayNormalVector(self, pol, mollweide=False):
        '''
        This will plot the values of the dot products of source direction with array norm for a given polarization.
        '''
        try:
            if pol == 'both':
                self.plotArrayNormalVector('hpol', mollweide=mollweide) 
                self.plotArrayNormalVector('vpol', mollweide=mollweide) 
            else:

                fig = plt.figure()
                fig.canvas.set_window_title('Direction Relative to Array Plane')
                if mollweide == True:
                    ax = fig.add_subplot(1,1,1, projection='mollweide')
                else:
                    ax = fig.add_subplot(1,1,1)

                
                if pol == 'hpol':
                    #Calculate minimum angle as a function of azimuth:
                    #import pdb; pdb.set_trace()
                    if mollweide == True:
                        #Automatically converts from rads to degs
                        im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, self.hpol_dot_angle_from_plane_deg, vmin=numpy.min(self.hpol_dot_angle_from_plane_deg), vmax=numpy.max(self.hpol_dot_angle_from_plane_deg),cmap=plt.cm.coolwarm)
                        plt.plot(self.phis_rad, numpy.pi/2 - self.thetas_rad[self.hpol_min_elevation_index_per_az], linewidth = self.min_elevation_linewidth, color='k')
                    else:
                        im = ax.pcolormesh(self.mesh_azimuth_deg, self.mesh_elevation_deg, self.hpol_dot_angle_from_plane_deg, vmin=numpy.min(self.hpol_dot_angle_from_plane_deg), vmax=numpy.max(self.hpol_dot_angle_from_plane_deg),cmap=plt.cm.coolwarm)
                        plt.plot(self.phis_deg, 90.0 - self.thetas_deg[self.hpol_min_elevation_index_per_az], linewidth = self.min_elevation_linewidth, color='k')
                elif pol == 'vpol':
                    min_elevation_index_per_az = numpy.argmin(numpy.abs(self.vpol_dot_angle_from_plane_deg),axis=0)
                    if mollweide == True:
                        #Automatically converts from rads to degs
                        im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, self.vpol_dot_angle_from_plane_deg, vmin=numpy.min(self.vpol_dot_angle_from_plane_deg), vmax=numpy.max(self.vpol_dot_angle_from_plane_deg),cmap=plt.cm.coolwarm)
                        plt.plot(self.phis_rad, numpy.pi/2 - self.thetas_rad[self.vpol_min_elevation_index_per_az], linewidth = self.min_elevation_linewidth, color='k')
                    else:
                        im = ax.pcolormesh(self.mesh_azimuth_deg, self.mesh_elevation_deg, self.vpol_dot_angle_from_plane_deg, vmin=numpy.min(self.vpol_dot_angle_from_plane_deg), vmax=numpy.max(self.vpol_dot_angle_from_plane_deg),cmap=plt.cm.coolwarm)
                        plt.plot(self.phis_deg, 90.0 - self.thetas_deg[self.vpol_min_elevation_index_per_az], linewidth = self.min_elevation_linewidth, color='k')

                cbar = fig.colorbar(im)
                cbar.set_label('Angle off of %s Array Plane'%pol.title())
                plt.xlabel('Azimuth (From East = 0 deg, North = 90 deg)')
                plt.ylabel('Elevation Angle (Degrees)')

                plt.grid(True)

                self.figs.append(fig)
                self.axs.append(ax)
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

    def wf(self, eventid, channels, div_std=False, hilbert=False, apply_filter=False, tukey=False, sine_subtract=False):
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
            eventid = int(eventid)
            self.prep.reader.setEntry(eventid)
            channels = numpy.sort(numpy.asarray(channels)).astype(int)
            temp_waveforms = numpy.zeros((len(channels),self.prep.buffer_length))
            for channel_index, channel in enumerate(channels):
                temp_wf = self.prep.wf(channel,apply_filter=apply_filter,hilbert=hilbert,tukey=tukey,sine_subtract=sine_subtract, return_sine_subtract_info=False)

                if div_std:
                    temp_wf = temp_wf/numpy.std(temp_wf)

                temp_waveforms[channel_index] = temp_wf

            waveforms = scipy.signal.resample(temp_waveforms,self.upsample,axis=1)

            if False:
                plt.figure()
                plt.title('Testing Waveforms')
                plt.plot(self.reader.t(),temp_waveforms[0],label='Hopefully processed non-resampled')
                plt.plot(self.times_resampled,waveforms[0],label='processed resampled')
                plt.plot(self.reader.t(),numpy.asarray(self.reader.wf(0))[self.start_waveform_index:self.end_waveform_index+1],label='no processing')
                plt.legend()
                import pdb; pdb.set_trace()


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

        This will also roll the starting point to zero.  This will not change relative times between signals. 
        '''
        try:
            return self.prep.t()
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
            self.times = self.t()
            self.dt = numpy.diff(self.times)[0]

            self.times_resampled = scipy.signal.resample(numpy.zeros(self.prep.buffer_length),self.upsample,t=self.times)[1]
            self.dt_resampled = numpy.diff(self.times_resampled)[0]
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

            self.t_hpol_0subtract1 = ((hpol_baseline_vectors[0][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[0][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[0][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[0] - self.cable_delays[2]) #ns
            self.t_hpol_0subtract2 = ((hpol_baseline_vectors[1][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[1][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[1][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[0] - self.cable_delays[4]) #ns
            self.t_hpol_0subtract3 = ((hpol_baseline_vectors[2][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[2][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[2][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[0] - self.cable_delays[6]) #ns
            self.t_hpol_1subtract2 = ((hpol_baseline_vectors[3][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[3][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[3][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[2] - self.cable_delays[4]) #ns
            self.t_hpol_1subtract3 = ((hpol_baseline_vectors[4][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[4][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[4][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[2] - self.cable_delays[6]) #ns
            self.t_hpol_2subtract3 = ((hpol_baseline_vectors[5][0]*signal_propogate_direction[:,:,0] + hpol_baseline_vectors[5][1]*signal_propogate_direction[:,:,1] + hpol_baseline_vectors[5][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[4] - self.cable_delays[6]) #ns

            self.t_vpol_0subtract1 = ((vpol_baseline_vectors[0][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[0][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[0][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[1] - self.cable_delays[3]) #ns
            self.t_vpol_0subtract2 = ((vpol_baseline_vectors[1][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[1][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[1][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[1] - self.cable_delays[5]) #ns
            self.t_vpol_0subtract3 = ((vpol_baseline_vectors[2][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[2][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[2][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[1] - self.cable_delays[7]) #ns
            self.t_vpol_1subtract2 = ((vpol_baseline_vectors[3][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[3][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[3][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[3] - self.cable_delays[5]) #ns
            self.t_vpol_1subtract3 = ((vpol_baseline_vectors[4][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[4][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[4][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[3] - self.cable_delays[7]) #ns
            self.t_vpol_2subtract3 = ((vpol_baseline_vectors[5][0]*signal_propogate_direction[:,:,0] + vpol_baseline_vectors[5][1]*signal_propogate_direction[:,:,1] + vpol_baseline_vectors[5][2]*signal_propogate_direction[:,:,2])/self.c)*1.0e9 + (self.cable_delays[5] - self.cable_delays[7]) #ns

            #Should double check when using these via rolling signals.
            #Calculate indices in corr for each direction.
            center = len(self.times_resampled)

            #THESE BREAK IF OUTSIDE OF SIGNAL LENGTH I THINK
            

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

    def generateNormalDotValues(self):
        '''
        For each of the specified directions this will determine the dot product of that vector with the norm
        of the array.  
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

            self.physical_dot_values = (self.n_physical[0]*signal_source_direction[:,:,0] + self.n_physical[1]*signal_source_direction[:,:,1] + self.n_physical[2]*signal_source_direction[:,:,2])
            self.hpol_dot_values = (self.n_hpol[0]*signal_source_direction[:,:,0] + self.n_hpol[1]*signal_source_direction[:,:,1] + self.n_hpol[2]*signal_source_direction[:,:,2])
            self.vpol_dot_values = (self.n_vpol[0]*signal_source_direction[:,:,0] + self.n_vpol[1]*signal_source_direction[:,:,1] + self.n_vpol[2]*signal_source_direction[:,:,2])

            #90 deg off plane for aligned with plane norm (zenith of 0). 
            self.physical_dot_angle_from_plane_deg = numpy.rad2deg(numpy.arcsin(self.physical_dot_values))
            self.hpol_dot_angle_from_plane_deg = numpy.rad2deg(numpy.arcsin(self.hpol_dot_values))
            self.vpol_dot_angle_from_plane_deg = numpy.rad2deg(numpy.arcsin(self.vpol_dot_values))

            self.physical_min_elevation_index_per_az = numpy.argmin(numpy.abs(self.physical_dot_angle_from_plane_deg),axis=0)
            self.hpol_min_elevation_index_per_az = numpy.argmin(numpy.abs(self.hpol_dot_angle_from_plane_deg),axis=0)
            self.vpol_min_elevation_index_per_az = numpy.argmin(numpy.abs(self.vpol_dot_angle_from_plane_deg),axis=0)



        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def fixZenithAngleWrap(self, azimuth_angles_deg, zenith_angles_deg):
        '''
        Given a set of zenith angles, this will make angles > 180 wrap back up, and angles < 0 wrap back down.
        Helpful for when plotting curves from different reference frames.  This is limited in the checks it does.
        It will also wrap the azimuth angles appropriately, containing them between -180, 180.
        '''
        try: 
            _zenith_angles_deg = zenith_angles_deg.copy()
            _azimuth_angles_deg = numpy.tile(azimuth_angles_deg.copy(),(len(self.thetas_deg),1)).T

            _azimuth_angles_deg[numpy.logical_and(_azimuth_angles_deg > 0 , numpy.logical_or(_zenith_angles_deg > 180.0, _zenith_angles_deg < 0.0))] -= 180.0
            _azimuth_angles_deg[numpy.logical_and(_azimuth_angles_deg < 0 , numpy.logical_or(_zenith_angles_deg > 180.0, _zenith_angles_deg < 0.0))] += 180.0 

            _zenith_angles_deg[_zenith_angles_deg > 90.0] = 180.0 - _zenith_angles_deg[_zenith_angles_deg > 90.0]
            _zenith_angles_deg[_zenith_angles_deg < -90.0] = 180.0 + _zenith_angles_deg[_zenith_angles_deg < -90.0]

            return azimuth_angles_deg.copy(), _zenith_angles_deg
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def rotateAaboutBbyTheta(self, a, b, dtheta_rad):
        '''
        This is a utility function that rotates vector a about vector b by dtheta_rad.
        '''
        try:
            a_par  = b * numpy.dot(a,b) / numpy.dot(b,b)
            a_perp = a - a_par
            rot_dir = numpy.cross(b,a_perp)
            x1 = numpy.cos(dtheta_rad)/numpy.linalg.norm(a_perp)
            x2 = numpy.sin(dtheta_rad)/numpy.linalg.norm(rot_dir)

            a_perp_new = numpy.linalg.norm(a_perp)*(x1*a_perp + x2*rot_dir)

            return (a_perp_new + a_par)/numpy.linalg.norm(a_perp_new + a_par)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getPlaneZenithCurves(self, norm, mode, zenith_deg, azimuth_offset_deg=0):
        '''
        Given a normal vector to define the plane this will determine the curve of the plane of the
        particular zenith from that vector.  This could be used to circle spots on the sky, plot
        the array plane, or perhaps be used for plotting the different interferometric circles on
        the map. Mode must be defined so this knows which antenna positions to use.

        The norm vector is assumed to be in the ENU basis of the given mode.  
        '''
        try:
            norm = norm/numpy.linalg.norm(norm)
            if zenith_deg == 0.0:
                thetas      = numpy.ones(self.n_phi) * numpy.rad2deg(numpy.arccos(norm[2]/numpy.linalg.norm(norm)))
                phis        = numpy.ones(self.n_phi) * numpy.rad2deg(numpy.arctan2(norm[1],norm[0]))
                phis -= azimuth_offset_deg
                phis[phis < -180.0] += 360.0
                phis[phis > 180.0] -= 360.0

                #Rotations of az don't matter because the same value.
                return [phis, thetas]
            elif zenith_deg == 180.0:
                thetas      = numpy.ones(self.n_phi) * numpy.rad2deg(numpy.arccos(-norm[2]/numpy.linalg.norm(norm)))
                phis        = numpy.ones(self.n_phi) * numpy.rad2deg(numpy.arctan2(-norm[1],-norm[0]))
                phis -= azimuth_offset_deg
                phis[phis < -180.0] += 360.0
                phis[phis > 180.0] -= 360.0
                #Rotations of az don't matter because the same value.
                return [phis, thetas]
            else:
                dtheta_rad = numpy.deg2rad(numpy.diff(self.phis_deg)[0])

                if mode == 'physical':
                    a0 = self.A0_physical
                    a1 = self.A1_physical
                    a2 = self.A2_physical
                    a3 = self.A3_physical
                elif mode == 'hpol':
                    a0 = self.A0_hpol
                    a1 = self.A1_hpol
                    a2 = self.A2_hpol
                    a3 = self.A3_hpol
                elif mode == 'vpol':
                    a0 = self.A0_vpol
                    a1 = self.A1_vpol
                    a2 = self.A2_vpol
                    a3 = self.A3_vpol

                #Find a vector perpendicular to the norm by performing cross with some vector not exactly parallel to norm.  Any of the antenna position vectors
                #will work assuming they are not parallel to norm.  Because this will be necessarily perpendicular to norm then it is in the plane.  Rotating 
                #norm about it will put it at the appropriate zenith.
                if numpy.linalg.norm(numpy.cross(norm,[1,0,0])) != 0:
                    in_plane_vector = numpy.cross(norm,[1,0,0])/numpy.linalg.norm(numpy.cross(norm,[1,0,0]))

                elif numpy.linalg.norm(numpy.cross(norm,[0,1,0])) != 0:
                    in_plane_vector = numpy.cross(norm,[0,1,0])/numpy.linalg.norm(numpy.cross(norm,[0,1,0]))

                elif numpy.linalg.norm(numpy.cross(norm,[0,0,1])) != 0:
                    in_plane_vector = numpy.cross(norm,[1,0,0])/numpy.linalg.norm(numpy.cross(norm,[1,0,0]))

                else:
                    print('Somehow the norm vector is parallel to ALL unit vectors of the antennas??')

                #Get some initial vector on the cone of the desired curve.
                zenith_vector = self.rotateAaboutBbyTheta(norm,in_plane_vector,numpy.deg2rad(zenith_deg))
                

                debug = False
                if debug:
                    print('SHOULD BE 1 if PERP')
                    print(numpy.cross(norm, in_plane_vector))
                    print(numpy.linalg.norm(numpy.cross(norm, in_plane_vector)))
                    print('zenith_vector')
                    print(zenith_vector)

                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    ax.quiver(0, 0, 0, norm[0], norm[1], norm[2],color='k', normalize=True)
                    ax.quiver(0, 0, 0, in_plane_vector[0], in_plane_vector[1], in_plane_vector[2],color='r', normalize=True)
                    
                    ax.quiver(0, 0, 0, zenith_vector[0], zenith_vector[1], zenith_vector[2], normalize=True,alpha=0.5)

                output_az_degs = numpy.zeros(self.n_phi)
                output_zenith_degs = numpy.zeros(self.n_phi)
                for i in range(self.n_phi):
                    #parellel portion of vector is 0 because A2 is in plane and n is defined as perp to plane. 
                    zenith_vector = self.rotateAaboutBbyTheta(zenith_vector,norm,dtheta_rad)
                    if debug:
                        ax.quiver(0, 0, 0, zenith_vector[0], zenith_vector[1], zenith_vector[2], normalize=True,alpha=0.5)
                    output_az_degs[i] = numpy.rad2deg(numpy.arctan2(zenith_vector[1],zenith_vector[0]))
                    output_zenith_degs[i] = numpy.rad2deg(numpy.arccos(zenith_vector[2]/numpy.linalg.norm(zenith_vector)))

                if debug:
                    ax.set_xlim(-2,2)
                    ax.set_ylim(-2,2)
                    ax.set_zlim(-2,2)
                    plt.figure()#Just to keep the above unmarred
                output_az_degs -= azimuth_offset_deg
                output_az_degs[output_az_degs < -180.0] += 360.0
                output_az_degs[output_az_degs > 180.0] -= 360.0
                out = [numpy.roll(output_az_degs,-numpy.argmin(output_az_degs)) , numpy.roll(output_zenith_degs,-numpy.argmin(output_az_degs))]

                return out
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getArrayPlaneZenithCurves(self, zenith_array_plane_deg, azimuth_offset_deg=0):
        '''
        For the given zenith angle this will return the indices of the ENU thetas that correspond to that zenith.
        These indices can be used to index self.thetas_deg and self.thetas_rad. 

        Azimuth_offset_deg can be used to rotate the results to different perspectives.  Default is 0 which returns
        the azimuth angles with 0deg facing East.  
        '''
        try:
            p = self.getPlaneZenithCurves(self.n_physical.copy(), 'physical', zenith_array_plane_deg, azimuth_offset_deg=azimuth_offset_deg)
            h = self.getPlaneZenithCurves(self.n_hpol.copy(), 'hpol', zenith_array_plane_deg, azimuth_offset_deg=azimuth_offset_deg)
            v = self.getPlaneZenithCurves(self.n_vpol.copy(), 'vpol', zenith_array_plane_deg, azimuth_offset_deg=azimuth_offset_deg)
            return p , h , v
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def interactivePlotter(self, event, mollweide = False, center_dir='E'):
        '''
        This hopefully will make a plot when called by a double clock in the map.
        '''
        if event.dblclick == True:
            try:
                event_ax = event.inaxes
                pol = event_ax.get_title().split('-')[2]
                eventid = int(event_ax.get_title().split('-')[1])
                hilbert = literal_eval(event_ax.get_title().split('-')[3].split('=')[1])

                if center_dir.upper() == 'E':
                    azimuth_offset_rad = 0 #Normally this is subtracted for plotting, but needs to be added here to get back to original orientation for finding the correct time delays.
                    azimuth_offset_deg = 0 #Normally this is subtracted for plotting, but needs to be added here to get back to original orientation for finding the correct time delays.

                elif center_dir.upper() == 'N':
                    azimuth_offset_rad = numpy.pi/2 #Normally this is subtracted for plotting, but needs to be added here to get back to original orientation for finding the correct time delays.
                    azimuth_offset_deg = 90 #Normally this is subtracted for plotting, but needs to be added here to get back to original orientation for finding the correct time delays.
                elif center_dir.upper() == 'W':
                    azimuth_offset_rad = numpy.pi #Normally this is subtracted for plotting, but needs to be added here to get back to original orientation for finding the correct time delays.
                    azimuth_offset_deg = 180 #Normally this is subtracted for plotting, but needs to be added here to get back to original orientation for finding the correct time delays.
                elif center_dir.upper() == 'S':
                    azimuth_offset_rad = -numpy.pi/2 #Normally this is subtracted for plotting, but needs to be added here to get back to original orientation for finding the correct time delays.
                    azimuth_offset_deg = -90 #Normally this is subtracted for plotting, but needs to be added here to get back to original orientation for finding the correct time delays.

                if mollweide == True:
                    #axis coords in radians for mollweide
                    event.xdata = numpy.rad2deg(event.xdata)
                    event.ydata = numpy.rad2deg(event.ydata)

                event.xdata += azimuth_offset_deg #Adding azimuth to orient back to orignal frame of data to determine time delays.
                if event.xdata > 180.0:
                    event.xdata -= 360.0
                elif event.xdata < -180.0:
                    event.xdata += 360.0

                theta_index = numpy.argmin( abs((90.0 - self.thetas_deg) - event.ydata) ) #Data plotted with elevation angles, not zenith.
                phi_index = numpy.argmin( abs(self.phis_deg - (event.xdata)) ) 

                if pol == 'hpol':
                    channels = numpy.array([0,2,4,6])

                    waveforms = self.wf(eventid, channels, div_std=False, hilbert=hilbert, apply_filter=self.apply_filter, tukey=self.apply_tukey, sine_subtract=self.apply_sine_subtract)
                    t_best_0subtract1 = self.t_hpol_0subtract1[theta_index,phi_index]
                    t_best_0subtract2 = self.t_hpol_0subtract2[theta_index,phi_index]
                    t_best_0subtract3 = self.t_hpol_0subtract3[theta_index,phi_index]
                elif pol == 'vpol':
                    channels = numpy.array([1,3,5,7])
                    waveforms = self.wf(eventid, channels, div_std=False, hilbert=hilbert, apply_filter=self.apply_filter, tukey=self.apply_tukey, sine_subtract=self.apply_sine_subtract)
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
                plt.suptitle('ENU %s\nAzimuth = %0.3f, Zenith = %0.3f'%(event_ax.get_title().replace('-',' ').title(), event.xdata,event.ydata))
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
                
                plt.plot(self.times_resampled, waveforms[0]/max(waveforms[0]),label='Ch%i'%channels[0])
                plt.plot(self.times_resampled + t_best_0subtract1, waveforms[1]/max(waveforms[1]),label='Ch%i, shifted %0.2f ns'%(channels[1], t_best_0subtract1))
                plt.plot(self.times_resampled + t_best_0subtract2, waveforms[2]/max(waveforms[2]),label='Ch%i, shifted %0.2f ns'%(channels[2], t_best_0subtract2))
                plt.plot(self.times_resampled + t_best_0subtract3, waveforms[3]/max(waveforms[3]),label='Ch%i, shifted %0.2f ns'%(channels[3], t_best_0subtract3))

                plt.ylabel('Normalized adu')
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

    def mapMax(self, map_values, max_method=0, verbose=False, zenith_cut_ENU=None, zenith_cut_array_plane=None, pol=None):
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
        verbose : bool
            Will print more if True.  Default is False.
        zenith_cut_ENU : list of 2 float values
            Given in degrees, angles within these two values are considered.  If None is given for the first then it is
            assumed to be 0 (overhead), if None is given for the latter then it is assumed to be 180 (straight down).
        zenith_cut_array_plane : list of 2 float values
            Given in degrees, angles within these two values are considered.  If None is given for the first then it is
            assumed to be 0 (overhead), if None is given for the latter then it is assumed to be 180 (straight down).  This is
            polarization dependant because it depends on the calibration of the antennas positions.  So if pol=None then this will
            be ignored.


        Returns
        -------
        row_index : int
            The selected row.
        col_index : int
            The selected column.
        theta_best : float 
            The corresponding values for this parameters for row_index and col_index.
        phi_best : float   
            The corresponding values for this parameters for row_index and col_index.
        t_best_0subtract1 : float
            The corresponding values for this parameters for row_index and col_index.
        t_best_0subtract2 : float
            The corresponding values for this parameters for row_index and col_index.
        t_best_0subtract3 : float
            The corresponding values for this parameters for row_index and col_index.
        t_best_1subtract2 : float
            The corresponding values for this parameters for row_index and col_index.
        t_best_1subtract3 : float
            The corresponding values for this parameters for row_index and col_index.
        t_best_2subtract3 : float
            The corresponding values for this parameters for row_index and col_index.


        '''
        theta_cut = numpy.ones_like(map_values,dtype=bool)

        if zenith_cut_ENU is not None:
            if len(zenith_cut_ENU) != 2:
                print('zenith_cut_ENU must be a 2 valued list.')
                return
            else:
                if zenith_cut_ENU[0] is None:
                    zenith_cut_ENU[0] = 0
                if zenith_cut_ENU[1] is None:
                    zenith_cut_ENU[1] = 180

                theta_cut_1d = numpy.logical_and(self.thetas_deg >= min(zenith_cut_ENU), self.thetas_deg <= max(zenith_cut_ENU))
                theta_cut = numpy.multiply(theta_cut.T,theta_cut_1d).T
        if pol is not None:
            if zenith_cut_array_plane is not None:
                if len(zenith_cut_array_plane) != 2:
                    print('zenith_cut_array_plane must be a 2 valued list.')
                    return
                else:
                    if zenith_cut_array_plane[0] is None:
                        zenith_cut_array_plane[0] = 0
                    if zenith_cut_array_plane[1] is None:
                        zenith_cut_array_plane[1] = 180

                    if pol == 'hpol':
                        theta_cut = numpy.logical_and(theta_cut,numpy.logical_and(90.0 - self.hpol_dot_angle_from_plane_deg >= min(zenith_cut_array_plane), 90.0 - self.hpol_dot_angle_from_plane_deg <= max(zenith_cut_array_plane)))
                    if pol == 'vpol':
                        theta_cut = numpy.logical_and(theta_cut,numpy.logical_and(90.0 - self.vpol_dot_angle_from_plane_deg >= min(zenith_cut_array_plane), 90.0 - self.vpol_dot_angle_from_plane_deg <= max(zenith_cut_array_plane)))

        masked_map_values = numpy.ma.array(map_values.copy(),mask=~theta_cut) #This way the values not in the range are not included in calculations but the dimensions of the map stay the same.

        if max_method == 0:
            row_index, column_index = numpy.unravel_index(masked_map_values.argmax(),numpy.shape(masked_map_values))

        elif max_method == 1:
            #Calculates sum of each point plus surrounding four points to get max.
            rounded_corr_values = (masked_map_values + numpy.roll(masked_map_values,1,axis=0) + numpy.roll(masked_map_values,-1,axis=0) + numpy.roll(masked_map_values,1,axis=1) + numpy.roll(masked_map_values,-1,axis=1))/5.0
            row_index, column_index = numpy.unravel_index(rounded_corr_values.argmax(),numpy.shape(rounded_corr_values))

        theta_best  = self.thetas_deg[row_index]
        phi_best    = self.phis_deg[column_index]

        t_best_0subtract1 = self.t_hpol_0subtract1[row_index,column_index]
        t_best_0subtract2 = self.t_hpol_0subtract2[row_index,column_index]
        t_best_0subtract3 = self.t_hpol_0subtract3[row_index,column_index]
        t_best_1subtract2 = self.t_hpol_1subtract2[row_index,column_index]
        t_best_1subtract3 = self.t_hpol_1subtract3[row_index,column_index]
        t_best_2subtract3 = self.t_hpol_2subtract3[row_index,column_index]

        if verbose == True:
            print("From the correlation plot:")
            print("Best zenith angle:",theta_best)
            print("Best azimuth angle:",phi_best)
            print('Predicted time delays %run between A0 and A1:', t_best_0subtract1)
            print('Predicted time delays between A0 and A2:', t_best_0subtract2)
            print('Predicted time delays between A0 and A3:', t_best_0subtract3)
            print('Predicted time delays between A1 and A2:', t_best_1subtract2)
            print('Predicted time delays between A1 and A3:', t_best_1subtract3)
            print('Predicted time delays between A2 and A3:', t_best_2subtract3)

        return row_index, column_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3

    def altAzToRaDec(eventid, alt,az):
        '''
        This will setEntry to the correct eventid such that it can use time information to understand the current alt-az and how it relates to Ra Dec.

        This function has not yet been completed. 
        '''
        self.reader.setEntry(eventid)

        time = 0

        coords_altaz = SkyCoord(alt= alt * apu.deg , az = az * apu.deg, obstime = time, frame = 'altaz', location = ant0_loc)
        coords_radec = coords_altaz.icrs
        print('THIS FUNCTION IS NOT WORKING YET.')

    def addCurveToMap(self, im, plane_xy,  mollweide=False, *args, **kwargs):
        '''
        This will plot the curves to a map (ax.pcolormesh instance passed as im).

        plane_xy is a set of theta, phi coordinates given in degrees to plot on the map.
        y should be given in zenith angle as it will be converted to elevation internally. 

        The args and kwargs should be plotting things such as color and linestyle.  
        '''
        try:
            plane_xy[1] = 90.0 - plane_xy[1]
            max_diff_deg = numpy.max(numpy.abs(numpy.diff(plane_xy)))

            if mollweide == True:
                plane_xy[0] = numpy.deg2rad(plane_xy[0])
                plane_xy[1] = numpy.deg2rad(plane_xy[1])

            #Plot the center line.
            #print('MAX_DIFF_DEG = ',max_diff_deg)
            if max_diff_deg > 350:
                #Implies the curve is split across left and right side of the plot.
                left_cut = plane_xy[0] <= 0
                right_cut = plane_xy[0] >= 0
                right_cut = numpy.sort(numpy.where(right_cut)[0])
                left_cut = numpy.sort(numpy.where(left_cut)[0])
                left_cut = numpy.roll(left_cut,numpy.argmax(numpy.diff(plane_xy[0][left_cut]))-1)

                plt.plot(plane_xy[0][left_cut], plane_xy[1][left_cut], *args, **kwargs)
                plt.plot(plane_xy[0][right_cut], plane_xy[1][right_cut], *args, **kwargs)
            else:
                if numpy.all([len(numpy.unique(plane_xy[0])) == 1,len(numpy.unique(plane_xy[1])) == 1]):
                    plt.scatter(plane_xy[0][0], plane_xy[1][0], *args, **kwargs)
                else:
                    plt.plot(plane_xy[0], plane_xy[1], *args, **kwargs)

            return im
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def addCircleToMap(self, ax, azimuth, elevation, azimuth_offset_deg=0, mollweide=False, radius = 5.0, crosshair=False, return_circle=False, return_crosshair=False, color='k', *args, **kwargs):
        '''
        This will add a circle to a map return the map.  If return_circle == True then the circle object is returned as
        well such that the position can be redefined later.

        All angles (including radius) should be given in degrees.

        args and kwargs should be plot parameters.  A common set is:
        linewidth=0.5,fill=False
        Note that color will be used for both crosshair and edge_color of the circle if crosshair == True.  Thus it
        has it's own kwarg.   
    
        If return_crosshair is True it will plot the crosshair (even if crosshair = False) and will return a tuple (h,v) where
        h and v are the line2d objects of those lines.  These can then have their xdata and ydata updated. 
        '''

        #Add best circle.
        azimuth = azimuth - azimuth_offset_deg
        if azimuth < -180.0:
            azimuth += 2*180.0
        elif azimuth > 180.0:
            azimuth -= 2*180.0
        
        if mollweide == True:
            radius = numpy.deg2rad(radius)
            elevation = numpy.deg2rad(elevation)
            azimuth = numpy.deg2rad(azimuth)

        circle = plt.Circle((azimuth, elevation), radius, edgecolor=color, *args, **kwargs)

        if crosshair == True or return_crosshair == True:
            h = ax.axhline(elevation,c=color,linewidth=1,alpha=0.5)
            v = ax.axvline(azimuth,c=color,linewidth=1,alpha=0.5)

        ax.add_artist(circle)
        if return_circle == False and return_crosshair == False:
            return ax
        elif return_circle == True and return_crosshair == False:
            return ax,  circle
        elif return_circle == False and return_crosshair == True:
            return ax,  (h, v)
        else:
            return ax, circle, (h, v)

    def getTimeDelayCurves(self, time_delay_dict, mode):
        '''
        This will determine the time delay curves for a particular baseline.  Use addTimeDelayCurves if you just want to plot these
        on an image.  That will call this function and plot them.  These will be returned in ENU degrees azimuth and zenith.

        These will be formatted simila rto time_delay_dict where each pair key will have a list containing the plane corresponding
        to each of the given time delays.
        '''
        try:
            if mode == 'hpol':
                cable_delays = self.cable_delays[0:8:2]
                all_antennas = numpy.vstack((self.A0_hpol,self.A1_hpol,self.A2_hpol,self.A3_hpol))
                time_delays = time_delay_dict['hpol']
            elif mode == 'vpol':
                cable_delays = self.cable_delays[1:8:2]
                all_antennas = numpy.vstack((self.A0_vpol,self.A1_vpol,self.A2_vpol,self.A3_vpol))
                time_delays = time_delay_dict['vpol']

            pairs = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
            baseline_index = -1
            output_dict = {}
            for pair_key, pair_time_delay in time_delays.items():
                baseline_index += 1
                output_dict[pair_key] = {}
                output_dict[pair_key]['zenith_deg'] = []
                output_dict[pair_key]['azimuth_deg'] = []
                pair = numpy.array(pair_key.replace('[','').replace(']','').split(','),dtype=int)
                pair_index = numpy.where(numpy.sum(pair == pairs,axis=1) == 2)[0][0] #Used for consistent coloring.
                i = pair[0]
                j = pair[1]
                A_ji = (all_antennas[j] - all_antennas[i])*(1e9/self.c) #Want these to be time vectors, not distance.  They are currently expressed in ns.
                #Using time delays defined as i - j, to get the result of positive cos(theta) values between source direction and vector between antennas
                #I need to have a negitive somewhere if definine the vector as A_ij.  By using A_ji the cos(theta_source) results in the appropriate sign
                #of dt on the lhs of the equation.   
                for time_delay in pair_time_delay:
                    #Calculate geometric dt by removing cable delays from measured dt:
                    dt = time_delay - (cable_delays[i] - cable_delays[j])

                    #Forcing the vector v to be a unit vector pointing towards the source
                    #below solvers for theta such that dt = |v||A_ij|cos(theta) where dt
                    #is the geometric time delay after the cable delays have been removed
                    #from the measured time delay. Theta is the amount I need to rotate A_ij
                    #in any perpendicular direction to get a vector that would result in the
                    #appropriate time delays.
                    theta_deg = numpy.rad2deg(numpy.arccos(dt/(numpy.linalg.norm(A_ji))))  #Forcing the vector v to be a unit vector pointing towards the source

                    plane_xy = self.getPlaneZenithCurves(A_ji, mode, theta_deg, azimuth_offset_deg=0)
                    output_dict[pair_key]['azimuth_deg'].append(plane_xy[0])
                    output_dict[pair_key]['zenith_deg'].append(plane_xy[1])
            return output_dict
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def addTimeDelayCurves(self, im, time_delay_dict, mode, include_baselines=numpy.array([0,1,2,3,4,5]), mollweide=False, azimuth_offset_deg=0, *args, **kwargs):
        '''
        This will plot the curves to a map (ax.pcolormesh instance passed as im).
        
        time_delays should be a dict as spcified by map.

        These expected time delays should come from raw measured time delays from signals.  
        The cable delays will be accounted for internally and should not have been accounted 
        for already.

        Mode specifies the polarization.

        The args and kwargs should be plotting things such as color and linestyle.  
        '''
        try:
            if mode == 'hpol':
                cable_delays = self.cable_delays[0:8:2]
                all_antennas = numpy.vstack((self.A0_hpol,self.A1_hpol,self.A2_hpol,self.A3_hpol))
                time_delays = time_delay_dict['hpol']
            elif mode == 'vpol':
                cable_delays = self.cable_delays[1:8:2]
                all_antennas = numpy.vstack((self.A0_vpol,self.A1_vpol,self.A2_vpol,self.A3_vpol))
                time_delays = time_delay_dict['vpol']

            pairs = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
            baseline_cm = plt.cm.get_cmap('tab10', 10)
            baseline_colors = baseline_cm(numpy.linspace(0, 1, 10))[0:6] #only want the first 6 colours in this list of 10 colors.
            baseline_index = -1
            for pair_key, pair_time_delay in time_delays.items():
                baseline_index += 1
                if numpy.isin(baseline_index ,include_baselines ):
                    linestyle = '-'
                else:
                    linestyle = '--'

                pair = numpy.array(pair_key.replace('[','').replace(']','').split(','),dtype=int)
                pair_index = numpy.where(numpy.sum(pair == pairs,axis=1) == 2)[0][0] #Used for consistent coloring.
                i = pair[0]
                j = pair[1]
                A_ji = (all_antennas[j] - all_antennas[i])*(1e9/self.c) #Want these to be time vectors, not distance.  They are currently expressed in ns.
                #Using time delays defined as i - j, to get the result of positive cos(theta) values between source direction and vector between antennas
                #I need to have a negitive somewhere if definine the vector as A_ij.  By using A_ji the cos(theta_source) results in the appropriate sign
                #of dt on the lhs of the equation.   
                for time_delay in pair_time_delay:
                    #Calculate geometric dt by removing cable delays from measured dt:
                    dt = time_delay - (cable_delays[i] - cable_delays[j])

                    #Forcing the vector v to be a unit vector pointing towards the source
                    #below solvers for theta such that dt = |v||A_ij|cos(theta) where dt
                    #is the geometric time delay after the cable delays have been removed
                    #from the measured time delay. Theta is the amount I need to rotate A_ij
                    #in any perpendicular direction to get a vector that would result in the
                    #appropriate time delays.

                    theta_deg = numpy.rad2deg(numpy.arccos(dt/(numpy.linalg.norm(A_ji))))  #Forcing the vector v to be a unit vector pointing towards the source
                    plane_xy = self.getPlaneZenithCurves(A_ji, mode, theta_deg, azimuth_offset_deg=azimuth_offset_deg)
                    #Plot array plane 0 elevation curve.
                    im = self.addCurveToMap(im, plane_xy,  mollweide=mollweide, linewidth = 4.0*self.min_elevation_linewidth, color=baseline_colors[pair_index], alpha=0.5, label=pair_key, linestyle=linestyle)
            return im
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def map(self, eventid, pol, include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, plot_corr=False, hilbert=False, interactive=False, max_method=None, waveforms=None, verbose=True, mollweide=False, zenith_cut_ENU=None, zenith_cut_array_plane=None, center_dir='E', circle_zenith=None, circle_az=None, time_delay_dict={}):
        '''
        Makes the cross correlation make for the given event.  center_dir only specifies the center direction when
        plotting and does not modify the output array, which is ENU oriented. 

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
        waveforms : numpy.ndarray
            Giving values for waveforms will supercede loading waveforms for the given eventid.  It should only be used as a workaround
            for specific purposes (like giving fake signals for testing).  If the waveforms are not of reasonable format then this may
            break.
        verbose : bool
            Enables more printing.  Default is False
        mollweide : bool
            Makes the plot with a mollweide projection.  Default is False.
        zenith_cut_ENU : list of floats
            Cuts on the map within the zenith range given for calculations of max value on plot.  This uses ENU zenith.
        zenith_cut_array_plane : list of floats
            The same as zenith_cut_ENU but uses the zenith angle measured relative to the array plane.  
        center_dir : str
            Specifies the center direction when plotting.  By default this is 'E' which is East (ENU standard).
        circle_zenith : list of floats
            List of zenith values to circle on the plot.  These could be known background sources like planes.
            The length of this must match the length of circle_az.  Should be given in degrees.
        circle_az : list of floats
            List of azimuths values to circle on the plot.  These could be known background sources like planes.
            The length of this must match the length of circle_zenith.  Should be given in degrees.
        time_delay_dict : dict of list of floats
            The first level of the dict should specify 'hpol' and/or 'vpol'
            The following key within should have each of the baseline pairs that you wish to plot.  Each of these
            will correspond to a list of floats with all of the time delays for that baseline you want plotted. 
        '''
        try:
            if hilbert == True:
                if verbose == True:
                    print('WARNING! Enabling Hilbert envelopes throws off correlation normalization.')
            
            time_delay_dict = time_delay_dict.copy()
            if ~numpy.isin('hpol', list(time_delay_dict.keys())):
                time_delay_dict['hpol'] = {}
            if ~numpy.isin('vpol', list(time_delay_dict.keys())):
                time_delay_dict['vpol'] = {}
            if pol == 'both':
                hpol_result = self.map(eventid,'hpol', plot_map=plot_map, plot_corr=plot_corr, hilbert=hilbert, mollweide=mollweide, center_dir=center_dir, zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,circle_zenith=circle_zenith,circle_az=circle_az,time_delay_dict=time_delay_dict,include_baselines=include_baselines)
                vpol_result = self.map(eventid,'vpol', plot_map=plot_map, plot_corr=plot_corr, hilbert=hilbert, mollweide=mollweide, center_dir=center_dir, zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,circle_zenith=circle_zenith,circle_az=circle_az,time_delay_dict=time_delay_dict,include_baselines=include_baselines)
                return hpol_result, vpol_result

            elif pol == 'hpol':
                if waveforms is None:
                    waveforms = self.wf(eventid, numpy.array([0,2,4,6]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter,tukey=self.apply_tukey, sine_subtract=self.apply_sine_subtract) #Div by std and resampled waveforms normalizes the correlations

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

                stacked_corr_values = numpy.array([corr_value_0subtract1, corr_value_0subtract2, corr_value_0subtract3, corr_value_1subtract2, corr_value_1subtract3, corr_value_2subtract3] )
                mean_corr_values = numpy.mean(stacked_corr_values[include_baselines],axis=0)
                if plot_map == True:
                    if max_method is not None:
                        row_index, column_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol)
                    else:
                        row_index, column_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol)

            elif pol == 'vpol':
                if waveforms is None:
                    waveforms = self.wf(eventid, numpy.array([1,3,5,7]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter,tukey=self.apply_tukey, sine_subtract=self.apply_sine_subtract) #Div by std and resampled waveforms normalizes the correlations

                corr01 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[1])))/(len(self.times_resampled))
                corr02 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[2])))/(len(self.times_resampled))
                corr03 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[3])))/(len(self.times_resampled))
                corr12 = (numpy.asarray(scipy.signal.correlate(waveforms[1],waveforms[2])))/(len(self.times_resampled))
                corr13 = (numpy.asarray(scipy.signal.correlate(waveforms[1],waveforms[3])))/(len(self.times_resampled))
                corr23 = (numpy.asarray(scipy.signal.correlate(waveforms[2],waveforms[3])))/(len(self.times_resampled))
                try:
                    corr_value_0subtract1 = corr01[self.delay_indices_vpol_0subtract1]
                except:
                    print('Error in corr_value_0subtract1')
                    #import pdb; pdb.set_trace()
                try:
                    corr_value_0subtract2 = corr02[self.delay_indices_vpol_0subtract2]
                except:
                    print('Error in corr_value_0subtract2')
                    #import pdb; pdb.set_trace()
                try:
                    corr_value_0subtract3 = corr03[self.delay_indices_vpol_0subtract3]
                except:
                    print('Error in corr_value_0subtract3')
                    #import pdb; pdb.set_trace()
                try:
                    corr_value_1subtract2 = corr12[self.delay_indices_vpol_1subtract2]
                except:
                    print('Error in corr_value_1subtract2')
                    #import pdb; pdb.set_trace()
                try:
                    corr_value_1subtract3 = corr13[self.delay_indices_vpol_1subtract3]
                except:
                    print('Error in corr_value_1subtract3')
                    #import pdb; pdb.set_trace()
                try:
                    corr_value_2subtract3 = corr23[self.delay_indices_vpol_2subtract3]
                except:
                    print('Error in corr_value_2subtract3')
                    #import pdb; pdb.set_trace()

                stacked_corr_values = numpy.array([corr_value_0subtract1, corr_value_0subtract2, corr_value_0subtract3, corr_value_1subtract2, corr_value_1subtract3, corr_value_2subtract3] )
                mean_corr_values = numpy.mean(stacked_corr_values[include_baselines],axis=0)
                
                if plot_map == True:
                    if max_method is not None:
                        row_index, column_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane)
                    else:
                        row_index, column_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane)

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
                plt.plot(shifts,corr01,alpha=0.7,label='corr01' + [' (Not Included in Map)', ''][numpy.isin( 0, include_baselines).astype(int)])
                plt.plot(shifts,corr02,alpha=0.7,label='corr02' + [' (Not Included in Map)', ''][numpy.isin( 1, include_baselines).astype(int)])
                plt.plot(shifts,corr03,alpha=0.7,label='corr03' + [' (Not Included in Map)', ''][numpy.isin( 2, include_baselines).astype(int)])
                plt.plot(shifts,corr12,alpha=0.7,label='corr12' + [' (Not Included in Map)', ''][numpy.isin( 3, include_baselines).astype(int)])
                plt.plot(shifts,corr13,alpha=0.7,label='corr13' + [' (Not Included in Map)', ''][numpy.isin( 4, include_baselines).astype(int)])
                plt.plot(shifts,corr23,alpha=0.7,label='corr23' + [' (Not Included in Map)', ''][numpy.isin( 5, include_baselines).astype(int)])
                plt.legend()
                self.figs.append(fig)
                self.axs.append(fig.gca())

            if plot_map:
                if ~numpy.all(numpy.isin(include_baselines, numpy.array([0,1,2,3,4,5]))):
                    add_xtext = '\nIncluded baselines = ' + str(numpy.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])[include_baselines])
                else:
                    add_xtext = ''

                if center_dir.upper() == 'E':
                    center_dir_full = 'East'
                    azimuth_offset_rad = 0 #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = 0 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From East = 0 deg, North = 90 deg)' + add_xtext
                    roll = 0
                elif center_dir.upper() == 'N':
                    center_dir_full = 'North'
                    azimuth_offset_rad = numpy.pi/2 #This is subtracted from the xaxis to roll it effectively. 
                    azimuth_offset_deg = 90 #This is subtracted from the xaxis to roll it effectively. 
                    xlabel = 'Azimuth (From North = 0 deg, West = 90 deg)' + add_xtext
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))
                elif center_dir.upper() == 'W':
                    center_dir_full = 'West'
                    azimuth_offset_rad = numpy.pi #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = 180 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From West = 0 deg, South = 90 deg)' + add_xtext
                    roll = len(self.phis_rad)//2
                elif center_dir.upper() == 'S':
                    center_dir_full = 'South'
                    azimuth_offset_rad = -numpy.pi/2 #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = -90 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From South = 0 deg, East = 90 deg)' + add_xtext
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))

                rolled_values = numpy.roll(mean_corr_values,roll,axis=1)

                elevation_best_deg = 90.0 - theta_best

                fig = plt.figure()
                fig.canvas.set_window_title('r%i-e%i-%s Correlation Map'%(self.reader.run,eventid,pol.title()))
                if mollweide == True:
                    ax = fig.add_subplot(1,1,1, projection='mollweide')
                else:
                    ax = fig.add_subplot(1,1,1)
                ax.set_title('%i-%i-%s-Hilbert=%s'%(self.reader.run,eventid,pol,str(hilbert))) #FORMATTING SPECIFIC AND PARSED ELSEWHERE, DO NOT CHANGE. 

                if mollweide == True:
                    #Automatically converts from rads to degs
                    im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, rolled_values, vmin=numpy.min(rolled_values), vmax=numpy.max(rolled_values),cmap=plt.cm.coolwarm)
                else:
                    im = ax.pcolormesh(self.mesh_azimuth_deg, self.mesh_elevation_deg, rolled_values, vmin=numpy.min(rolled_values), vmax=numpy.max(rolled_values),cmap=plt.cm.coolwarm)

                cbar = fig.colorbar(im)
                if hilbert == True:
                    cbar.set_label('Mean Correlation Value (Arb)')
                else:
                    cbar.set_label('Mean Correlation Value')
                plt.xlabel(xlabel,fontsize=18)
                plt.ylabel('Elevation Angle (Degrees)',fontsize=18)
                plt.grid(True)

                #Prepare array cut curves
                if pol is not None:
                    if zenith_cut_array_plane is not None:
                        if len(zenith_cut_array_plane) != 2:
                            print('zenith_cut_array_plane must be a 2 valued list.')
                            return
                        else:
                            if zenith_cut_array_plane[0] is None:
                                zenith_cut_array_plane[0] = 0
                            if zenith_cut_array_plane[1] is None:
                                zenith_cut_array_plane[1] = 180

                #Prepare center line and plot the map.  Prep cut lines as well.
                if pol == 'hpol':
                    selection_index = 1
                elif pol == 'vpol':
                    selection_index = 2 
                
                plane_xy = self.getArrayPlaneZenithCurves(90.0, azimuth_offset_deg=azimuth_offset_deg)[selection_index]
                #Plot array plane 0 elevation curve.
                im = self.addCurveToMap(im, plane_xy,  mollweide=mollweide, linewidth = self.min_elevation_linewidth, color='k')

                if zenith_cut_array_plane is not None:
                    upper_plane_xy = self.getArrayPlaneZenithCurves(zenith_cut_array_plane[0], azimuth_offset_deg=azimuth_offset_deg)[selection_index]
                    lower_plane_xy = self.getArrayPlaneZenithCurves(zenith_cut_array_plane[1], azimuth_offset_deg=azimuth_offset_deg)[selection_index]
                    #Plot upper zenith array cut
                    im = self.addCurveToMap(im, upper_plane_xy,  mollweide=mollweide, linewidth = self.min_elevation_linewidth, color='k',linestyle = '--')
                    #Plot lower zenith array cut
                    im = self.addCurveToMap(im, lower_plane_xy,  mollweide=mollweide, linewidth = self.min_elevation_linewidth, color='k',linestyle = '--')

                #Add curves for time delays if present.
                im = self.addTimeDelayCurves(im, time_delay_dict, pol,  mollweide=mollweide, azimuth_offset_deg=azimuth_offset_deg, include_baselines=include_baselines)


                #Added circles as specified.
                ax, peak_circle = self.addCircleToMap(ax, phi_best, elevation_best_deg, azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius=5.0, crosshair=True, return_circle=True, color='lime', linewidth=0.5,fill=False)

                if circle_az is not None:
                    if circle_zenith is not None:
                        if circle_az is list:
                            circle_az = numpy.array(circle_az)
                        elif circle_az != numpy.ndarray:
                            circle_az = numpy.array([circle_az])

                        _circle_az = circle_az.copy() #These were changing when 'both' was called for pol.  So copying them to ensure original input maintains values.

                        if circle_zenith is list:
                            circle_zenith = numpy.array(circle_zenith)
                        elif circle_zenith != numpy.ndarray:
                            circle_zenith = numpy.array([circle_zenith])

                        _circle_zenith = circle_zenith.copy() #These were changing when 'both' was called for pol.  So copying them to ensure original input maintains values.

                        if len(_circle_zenith) == len(_circle_az):
                            additional_circles = []
                            for i in range(len(_circle_az)):
                                ax, _circ = self.addCircleToMap(ax, _circle_az[i], 90.0-_circle_zenith[i], azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius=5.0, crosshair=False, return_circle=True, color='fuchsia', linewidth=0.5,fill=False)
                                additional_circles.append(_circ)

                if zenith_cut_ENU is not None:
                    if mollweide == True:
                        #Block out simple ENU zenith cut region. 
                        plt.axhspan(numpy.deg2rad(90 - min(zenith_cut_ENU)),numpy.deg2rad(90.0),alpha=0.5)
                        plt.axhspan(numpy.deg2rad(-90) , numpy.deg2rad(90 - max(zenith_cut_ENU)),alpha=0.5)
                    else:
                        #Block out simple ENU zenith cut region. 
                        plt.axhspan(90 - min(zenith_cut_ENU),90.0,alpha=0.5)
                        plt.axhspan(-90 , 90 - max(zenith_cut_ENU),alpha=0.5)

                #Enable Interactive Portion
                if interactive == True:
                    fig.canvas.mpl_connect('button_press_event',lambda event : self.interactivePlotter(event,  mollweide=mollweide, center_dir=center_dir))

                #ax.legend(loc='lower left')
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


    def averagedMap(self, eventids, pol, plot_map=True, hilbert=False, max_method=None, mollweide=False, zenith_cut_ENU=None,zenith_cut_array_plane=None, center_dir='E', circle_zenith=None, circle_az=None, time_delay_dict={}):
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
        time_delay_dict = time_delay_dict.copy()
        if ~numpy.isin('hpol', list(time_delay_dict.keys())):
            time_delay_dict['hpol'] = {}
        if ~numpy.isin('vpol', list(time_delay_dict.keys())):
            time_delay_dict['vpol'] = {}
        if pol == 'both':
            hpol_result = self.averagedMap(eventids, 'hpol', plot_map=plot_map, hilbert=hilbert,mollweide=mollweide, zenith_cut_ENU=zenith_cut_ENU,zenith_cut_array_plane=zenith_cut_array_plane,circle_zenith=circle_zenith, circle_az=circle_az,time_delay_dict=time_delay_dict)
            vpol_result = self.averagedMap(eventids, 'vpol', plot_map=plot_map, hilbert=hilbert,mollweide=mollweide, zenith_cut_ENU=zenith_cut_ENU,zenith_cut_array_plane=zenith_cut_array_plane,circle_zenith=circle_zenith, circle_az=circle_az,time_delay_dict=time_delay_dict)
            return hpol_result, vpol_result

        total_mean_corr_values = numpy.zeros((self.n_theta, self.n_phi))
        for event_index, eventid in enumerate(eventids):
            sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
            sys.stdout.flush()
            total_mean_corr_values += self.map(eventid, pol, plot_map=False, plot_corr=False, hilbert=hilbert, mollweide=False, zenith_cut_ENU=zenith_cut_ENU)/len(eventids)  #Don't need to put center_dir here.  Only for plotting.
        print('')

        if zenith_cut_ENU is not None:
            if len(zenith_cut_ENU) != 2:
                print('zenith_cut_ENU must be a 2 valued list.')
                return
            else:
                if zenith_cut_ENU[0] is None:
                    zenith_cut_ENU[0] = 0
                if zenith_cut_ENU[1] is None:
                    zenith_cut_ENU[1] = 180

                theta_cut = numpy.logical_and(self.thetas_deg >= min(zenith_cut_ENU), self.thetas_deg <= max(zenith_cut_ENU))
        else:
            theta_cut = numpy.ones_like(self.thetas_deg,dtype=bool)

        theta_cut_row_indices = numpy.where(theta_cut)[0]

        if plot_map:
            if center_dir.upper() == 'E':
                center_dir_full = 'East'
                azimuth_offset_rad = 0 #This is subtracted from the xaxis to roll it effectively.
                azimuth_offset_deg = 0 #This is subtracted from the xaxis to roll it effectively.
                xlabel = 'Azimuth (From East = 0 deg, North = 90 deg)'
                roll = 0
            elif center_dir.upper() == 'N':
                center_dir_full = 'North'
                azimuth_offset_rad = numpy.pi/2 #This is subtracted from the xaxis to roll it effectively. 
                azimuth_offset_deg = 90 #This is subtracted from the xaxis to roll it effectively. 
                xlabel = 'Azimuth (From North = 0 deg, West = 90 deg)'
                roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))
            elif center_dir.upper() == 'W':
                center_dir_full = 'West'
                azimuth_offset_rad = numpy.pi #This is subtracted from the xaxis to roll it effectively.
                azimuth_offset_deg = 180 #This is subtracted from the xaxis to roll it effectively.
                xlabel = 'Azimuth (From West = 0 deg, South = 90 deg)'
                roll = len(self.phis_rad)//2
            elif center_dir.upper() == 'S':
                center_dir_full = 'South'
                azimuth_offset_rad = -numpy.pi/2 #This is subtracted from the xaxis to roll it effectively.
                azimuth_offset_deg = -90 #This is subtracted from the xaxis to roll it effectively.
                xlabel = 'Azimuth (From South = 0 deg, East = 90 deg)'
                roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))

            rolled_values = numpy.roll(total_mean_corr_values,roll,axis=1)

            if max_method is not None:
                row_index, column_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(total_mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU,zenith_cut_array_plane=zenith_cut_array_plane,pol=pol)
            else:
                row_index, column_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(total_mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU,zenith_cut_array_plane=zenith_cut_array_plane,pol=pol)        

            elevation_best_deg = 90.0 - theta_best

            if pol == 'hpol':
                t_best_0subtract1  = self.t_hpol_0subtract1[row_index,column_index]
                t_best_0subtract2  = self.t_hpol_0subtract2[row_index,column_index]
                t_best_0subtract3  = self.t_hpol_0subtract3[row_index,column_index]
                t_best_1subtract2  = self.t_hpol_1subtract2[row_index,column_index]
                t_best_1subtract3  = self.t_hpol_1subtract3[row_index,column_index]
                t_best_2subtract3  = self.t_hpol_2subtract3[row_index,column_index]
            elif pol == 'vpol':
                t_best_0subtract1  = self.t_vpol_0subtract1[row_index,column_index]
                t_best_0subtract2  = self.t_vpol_0subtract2[row_index,column_index]
                t_best_0subtract3  = self.t_vpol_0subtract3[row_index,column_index]
                t_best_1subtract2  = self.t_vpol_1subtract2[row_index,column_index]
                t_best_1subtract3  = self.t_vpol_1subtract3[row_index,column_index]
                t_best_2subtract3  = self.t_vpol_2subtract3[row_index,column_index]
            else:
                print('Invalid polarization option.  Returning nothing.')
                return
            fig = plt.figure()
            fig.canvas.set_window_title('r%i %s Averaged Correlation Map'%(self.reader.run,pol.title()))
            if mollweide == True:
                ax = fig.add_subplot(1,1,1, projection='mollweide')
            else:
                ax = fig.add_subplot(1,1,1)

            #im = ax.imshow(total_mean_corr_values, interpolation='none', extent=[min(self.phis_deg),max(self.phis_deg),max(self.thetas_deg),min(self.thetas_deg)],cmap=plt.cm.coolwarm) #cmap=plt.cm.jet)
            if mollweide == True:
                #Automatically converts from rads to degs
                im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, rolled_values, vmin=numpy.min(rolled_values), vmax=numpy.max(rolled_values),cmap=plt.cm.coolwarm)
            else:
                im = ax.pcolormesh(self.mesh_azimuth_deg, self.mesh_elevation_deg, rolled_values, vmin=numpy.min(rolled_values), vmax=numpy.max(rolled_values),cmap=plt.cm.coolwarm)

            cbar = fig.colorbar(im)
            plt.xlabel(xlabel)
            plt.ylabel('Elevation Angle (Degrees)')
            plt.grid(True)
            if hilbert == True:
                cbar.set_label('Mean Correlation Value (Arb)')
            else:
                cbar.set_label('Mean Correlation Value')


            #Prepare array cut curves
            if pol is not None:
                if zenith_cut_array_plane is not None:
                    if len(zenith_cut_array_plane) != 2:
                        print('zenith_cut_array_plane must be a 2 valued list.')
                        return
                    else:
                        if zenith_cut_array_plane[0] is None:
                            zenith_cut_array_plane[0] = 0
                        if zenith_cut_array_plane[1] is None:
                            zenith_cut_array_plane[1] = 180

            #Prepare center line and plot the map.  Prep cut lines as well.
            if pol == 'hpol':
                selection_index = 1
            elif pol == 'vpol':
                selection_index = 2 
            
            #Plot array plane 0 elevation curve.
            plane_xy = self.getArrayPlaneZenithCurves(90.0, azimuth_offset_deg=azimuth_offset_deg)[selection_index]
            im = self.addCurveToMap(im, plane_xy,  mollweide=mollweide, linewidth = self.min_elevation_linewidth, color='k')
            
            if zenith_cut_array_plane is not None:
                #Plot upper zenith array cut
                upper_plane_xy = self.getArrayPlaneZenithCurves(zenith_cut_array_plane[0], azimuth_offset_deg=azimuth_offset_deg)[selection_index]
                im = self.addCurveToMap(im, upper_plane_xy,  mollweide=mollweide, linewidth = self.min_elevation_linewidth, color='k',linestyle = '--')
                #Plot lower zenith array cut
                lower_plane_xy = self.getArrayPlaneZenithCurves(zenith_cut_array_plane[1], azimuth_offset_deg=azimuth_offset_deg)[selection_index]
                im = self.addCurveToMap(im, lower_plane_xy,  mollweide=mollweide, linewidth = self.min_elevation_linewidth, color='k',linestyle = '--')


            #Add curves for time delays if present.
            im = self.addTimeDelayCurves(im, time_delay_dict, pol,  mollweide=mollweide, azimuth_offset_deg=azimuth_offset_deg)

            #Added circles as specified.
            ax, peak_circle = self.addCircleToMap(ax, phi_best, elevation_best_deg, azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius=5.0, crosshair=True, return_circle=True, color='lime', linewidth=0.5,fill=False)

            if circle_az is not None:
                if circle_zenith is not None:
                    if circle_az is list:
                        circle_az = numpy.array(circle_az)
                    elif circle_az != numpy.ndarray:
                        circle_az = numpy.array([circle_az])

                    _circle_az = circle_az.copy() #These were changing when 'both' was called for pol.  So copying them to ensure original input maintains values.

                    if circle_zenith is list:
                        circle_zenith = numpy.array(circle_zenith)
                    elif circle_zenith != numpy.ndarray:
                        circle_zenith = numpy.array([circle_zenith])

                    _circle_zenith = circle_zenith.copy() #These were changing when 'both' was called for pol.  So copying them to ensure original input maintains values.

                    if len(_circle_zenith) == len(_circle_az):
                        additional_circles = []
                        for i in range(len(_circle_az)):
                            ax, _circ = self.addCircleToMap(ax, _circle_az[i], 90.0-_circle_zenith[i], azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius=5.0, crosshair=False, return_circle=True, color='fuchsia', linewidth=0.5,fill=False)
                            additional_circles.append(_circ)

            #Block out simple ENU zenith cut region. 
            if zenith_cut_ENU is not None:
                if mollweide == True:
                    #Block out simple ENU zenith cut region. 
                    plt.axhspan(numpy.deg2rad(90 - min(zenith_cut_ENU)),numpy.deg2rad(90.0),alpha=0.5)
                    plt.axhspan(numpy.deg2rad(-90) , numpy.deg2rad(90 - max(zenith_cut_ENU)),alpha=0.5)
                else:
                    #Block out simple ENU zenith cut region. 
                    plt.axhspan(90 - min(zenith_cut_ENU),90.0,alpha=0.5)
                    plt.axhspan(-90 , 90 - max(zenith_cut_ENU),alpha=0.5)
            ax.legend(loc='lower left')
            self.figs.append(fig)
            self.axs.append(ax)

            return total_mean_corr_values, fig, ax
        else:
            return total_mean_corr_values

    def animatedMap(self, eventids, pol, title, plane_zenith=None, plane_az=None, hilbert=False, max_method=None,center_dir='E',save=True,dpi=300,fps=3):
        '''
        Does the same thing as map, but updates the canvas for each event creating an animation.
        Mostly helpful for repeated sources that are expected to be moving such as planes.

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
        plane_zenith : list of floats
            List of zenith values to circle on the plot.  These could be known background sources like planes.
            The length of this must match the length of plane_az.  Should be given in degrees.
        plane_az : list of floats
            List of azimuths values to circle on the plot.  These could be known background sources like planes.
            The length of this must match the length of plane_zenith.  Should be given in degrees.
        '''
        if pol == 'both':
            hpol_result = self.animatedMap(eventids, 'hpol',title, plane_zenith=plane_zenith, plane_az=plane_az, hilbert=hilbert, max_method=max_method,center_dir=center_dir,save=save)
            vpol_result = self.animatedMap(eventids, 'vpol',title, plane_zenith=plane_zenith, plane_az=plane_az, hilbert=hilbert, max_method=max_method,center_dir=center_dir,save=save)

            return hpol_result, vpol_result

        try:
            print('Performing calculations for %s'%pol)

            all_maps = []# numpy.zeros((self.n_theta, self.n_phi))
            for event_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
                sys.stdout.flush()
                m = self.map(eventid, pol, plot_map=False, plot_corr=False, hilbert=hilbert) #Don't need to pass center_dir because performing rotation below!
                if center_dir.upper() == 'E':
                    center_dir_full = 'East'
                    azimuth_offset_rad = 0 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From East = 0 deg, North = 90 deg)'
                    roll = 0
                elif center_dir.upper() == 'N':
                    center_dir_full = 'North'
                    azimuth_offset_rad = numpy.pi/2 #This is subtracted from the xaxis to roll it effectively. 
                    xlabel = 'Azimuth (From North = 0 deg, West = 90 deg)'
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))
                elif center_dir.upper() == 'W':
                    center_dir_full = 'West'
                    azimuth_offset_rad = numpy.pi #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From West = 0 deg, South = 90 deg)'
                    roll = len(self.phis_rad)//2
                elif center_dir.upper() == 'S':
                    center_dir_full = 'South'
                    azimuth_offset_rad = -numpy.pi/2 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From South = 0 deg, East = 90 deg)'
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))
                azimuth_offset_deg = numpy.rad2deg(azimuth_offset_rad)
                m = numpy.roll(m,roll,axis=1) #orients center.
                all_maps.append(m)
            print('')

            fig = plt.figure(figsize=(16,9))
            ax = fig.add_subplot(1,1,1, projection='mollweide')
            
            fig.canvas.set_window_title('r%i %s Correlation Map Eventid = %i'%(self.reader.run,pol.title(),eventids[0]))
            ax.set_title('r%i %s Correlation Map Eventid = %i'%(self.reader.run,pol.title(),eventids[0]))

            # fig.canvas.set_window_title('Correlation Map Airplane Event %i'%(0))
            # ax.set_title('Correlation Map Airplane Event %i'%(0))


            if hilbert == True:
                #im = ax.imshow(all_maps[0], interpolation='none', vmin=numpy.min(all_maps[0]),vmax=numpy.max(all_maps[0]), extent=extent,cmap=plt.cm.coolwarm) #cmap=plt.cm.jet)
                im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, all_maps[0], vmin=numpy.min(all_maps[0]), vmax=numpy.max(all_maps[0]),cmap=plt.cm.coolwarm)
            else:
                #im = ax.imshow(all_maps[0], interpolation='none', vmin=numpy.concatenate(all_maps).min(),vmax=numpy.concatenate(all_maps).max(), extent=extent,cmap=plt.cm.coolwarm) #cmap=plt.cm.jet)
                im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, all_maps[0], vmin=numpy.concatenate(all_maps).min(), vmax=numpy.concatenate(all_maps).max(),cmap=plt.cm.coolwarm)

            cbar = fig.colorbar(im)
            if hilbert == True:
                cbar.set_label('Mean Correlation Value (Arb)')
            else:
                cbar.set_label('Mean Correlation Value')
            plt.xlabel(xlabel)
            plt.ylabel('Elevation Angle (Degrees)')
            plt.grid(True)


            #Prepare array cut curves

            #Prepare center line and plot the map.  Prep cut lines as well.
            if pol == 'hpol':
                selection_index = 1
            elif pol == 'vpol':
                selection_index = 2 
            
            plane_xy = self.getArrayPlaneZenithCurves(90.0, azimuth_offset_deg=azimuth_offset_deg)[selection_index]
            #Plot array plane 0 elevation curve.
            im = self.addCurveToMap(im, plane_xy,  mollweide=True, linewidth = self.min_elevation_linewidth, color='k')


            #Plot circles
            if plane_az is not None:
                if plane_az is list:
                    plane_az = numpy.array(plane_az)
                _plane_az = plane_az.copy() #These were changing when 'both' was called for pol.  So copying them to ensure original input maintains values.
                _plane_az -= azimuth_offset_deg #handled here so shouldn't be passed to addCircleToMap or similar functions.
                _plane_az[_plane_az < -180.0] += 360.0
                _plane_az[_plane_az > 180.0] -= 360.0


            if plane_zenith is not None:
                if plane_zenith is list:
                    plane_zenith = numpy.array(plane_zenith)
                _plane_zenith = plane_zenith.copy() #These were changing when 'both' was called for pol.  So copying them to ensure original input maintains values.
                plane_elevation = 90.0 - _plane_zenith
            else:
                plane_elevation = None

            if plane_elevation is not None and _plane_az is not None:
                ax, circle, lines = self.addCircleToMap(ax, _plane_az[0], plane_elevation[0], azimuth_offset_deg=0.0, mollweide=True, radius=6.0, crosshair=True, return_circle=True, return_crosshair=True, color='fuchsia', linewidth=2.0,fill=False,zorder=10) #azimuth offset_deg already accounted for as passed.

            def update(frame):
                _frame = frame%len(eventids) #lets it loop multiple times.  i.e. give animation more frames but same content looped.
                # fig.canvas.set_window_title('Correlation Map Airplane Event %i'%(_frame))
                # ax.set_title('Correlation Map Airplane Event %i'%(_frame))
                fig.canvas.set_window_title('r%i %s Correlation Map Eventid = %i'%(self.reader.run,pol.title(),eventids[_frame]))
                ax.set_title('r%i %s Correlation Map Eventid = %i'%(self.reader.run,pol.title(),eventids[_frame]))
                im.set_array(all_maps[_frame].ravel())

                if hilbert == True:
                    im.set_clim(vmin=numpy.min(all_maps[_frame]),vmax=numpy.max(all_maps[_frame]))

                try:
                    azimuth = numpy.deg2rad(_plane_az[_frame]) #Molweide True for animated.
                    elevation = numpy.deg2rad(plane_elevation[_frame])
                    if plane_elevation is not None and _plane_az is not None:
                        circle.center = azimuth, elevation
                        ax.add_artist(circle)

                        lines[0].set_ydata([numpy.deg2rad(plane_elevation[_frame]),numpy.deg2rad(plane_elevation[_frame])])
                        lines[1].set_xdata([numpy.deg2rad(_plane_az[_frame]),numpy.deg2rad(_plane_az[_frame])])
                except Exception as e:
                    pass
                return [im]

            ani = FuncAnimation(fig, update, frames=range(len(eventids)),blit=False,save_count=0)

            if save == True:
                try:
                    #ani.save('./%s_%s_hilbert=%s_%s.mp4'%(title,pol,str(hilbert), center_dir_full + '_centered'), writer='ffmpeg', fps=fps,dpi=dpi)
                    print('Attempting to save animated correlation map as:')
                    save_name = './%s_%s_hilbert=%s_%s.gif'%(title,pol,str(hilbert), center_dir_full + '_centered')
                    print(save_name)
                    print('len(eventids) = ',len(eventids))
                    ani.save(save_name, writer='imagemagick', fps=fps,dpi=dpi)
                    plt.close(fig)
                except Exception as e:
                    plt.close(fig)
                    print('\nError in %s'%inspect.stack()[0][3])
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
            else:
                self.figs.append(fig)
                self.axs.append(ax)
                self.animations.append(ani)
            
            return      
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def histMapPeak(self, eventids, pol, plot_map=True, hilbert=False, max_method=None, use_weight=False, zenith_cut_ENU=None):
        '''
        This will loop over eventids and makes a histogram from of location of maximum correlation
        value in the corresponding correlation maps. 

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
            hpol_hist = self.histMapPeak(eventids, 'hpol', plot_map=plot_map, hilbert=hilbert, max_method=max_method, use_weight=use_weight, zenith_cut_ENU=zenith_cut_ENU)
            vpol_hist = self.histMapPeak(eventids, 'vpol', plot_map=plot_map, hilbert=hilbert, max_method=max_method, use_weight=use_weight, zenith_cut_ENU=zenith_cut_ENU)
            return (hpol_hist,vpol_hist)
        else:
            hist = numpy.zeros((self.n_theta, self.n_phi))
            for event_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
                sys.stdout.flush()
                m = self.map(eventid, pol, plot_map=False, plot_corr=False, hilbert=hilbert)/len(eventids)
                if max_method is not None:
                    row_index, column_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(m,max_method=max_method,verbose=plot_map,zenith_cut_ENU=zenith_cut_ENU)
                else:
                    row_index, column_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(m,verbose=plot_map,zenith_cut_ENU=zenith_cut_ENU)        

                if use_weight == True:
                    hist[row_index,column_index] += m[row_index,column_index]
                else:
                    hist[row_index,column_index] += 1

            if plot_map:
                fig = plt.figure()
                fig.canvas.set_window_title('r%i %s Peak Correlation Map Hist'%(self.reader.run,pol.title()))
                ax = fig.add_subplot(1,1,1)
                im = ax.imshow(hist, interpolation='none', extent=[min(self.phis_deg),max(self.phis_deg),max(self.thetas_deg),min(self.thetas_deg)],cmap=plt.cm.winter,norm=matplotlib.colors.LogNorm()) #cmap=plt.cm.jet)
                cbar = fig.colorbar(im)
                cbar.set_label('Counts (Maybe Weighted)')
                plt.xlabel('Azimuth Angle (Degrees)')
                plt.ylabel('Elevation Angle (Degrees)')

                return hist

    def beamMap(self, eventid, pol, plot_map=True, plot_corr=False, hilbert=False, interactive=False, max_method=None):
        '''
        Makes the beam summed map which can be used to educate the user as to which beam might be good to detect a similar signal
        to the one loaded.  The beamforming algorithm is currently implemented in roughly the same way as the phased array logic:
        16 sample powersums at 8 sample spacing.  

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
            print('This function has not yet been programmed.')
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    def defineMasksForGivenTimeDelays(self):
        '''
        For a set of time delays, this will determine a mask on top of the given map for
        each of the time delays for each of the 6 baseline.
        This could be used for weight cross correlations in the fftmath class.
        These are expected to be sparse masks so maybe just store indices for each time?  Might just decide to pass
        a subset of the times and interpolate?  Definitely don't need as fine resolution as I often use for fftmath.
        '''

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

        crit_freq_high_pass_MHz = 30#None#55
        high_pass_filter_order = 5#None#4
        apply_phase_response = True

        plot_filter=True

        known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
        eventids = {}
        eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])
        eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])
        all_eventids = numpy.sort(numpy.append(eventids['hpol'],eventids['vpol']))

        hpol_eventids_cut = numpy.isin(all_eventids,eventids['hpol'])
        vpol_eventids_cut = numpy.isin(all_eventids,eventids['vpol'])

        cor = Correlator(reader,  upsample=2**15, n_phi=361, n_theta=361, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response)
        if True:
            for mode in ['hpol','vpol']:
                eventid = eventids[mode][0]
                mean_corr_values, fig, ax = cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=False, interactive=True)

                pulser_locations_ENU = info.loadPulserPhaseLocationsENU()[mode]['run%i'%run]
                pulser_locations_ENU = pulser_locations_ENU/numpy.linalg.norm(pulser_locations_ENU)
                pulser_theta = numpy.degrees(numpy.arccos(pulser_locations_ENU[2]))
                pulser_phi = numpy.degrees(numpy.arctan2(pulser_locations_ENU[1],pulser_locations_ENU[0]))
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
            pulser_phi = numpy.degrees(numpy.arctan2(pulser_locations_ENU[1],pulser_locations_ENU[0]))
            print('%s Expected pulser location: Zenith = %0.2f, Az = %0.2f'%(mode.title(), pulser_theta,pulser_phi))

            ax.axvline(pulser_phi,c='r')
            ax.axhline(pulser_theta,c='r')

            all_figs.append(fig)
            all_axs.append(ax)
        all_cors.append(cor)

if __name__=="__main__":
    plt.close('all')

    crit_freq_low_pass_MHz = 100#60 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = 8

    crit_freq_high_pass_MHz = None#30#None
    high_pass_filter_order = None#5#None
    plot_filter=False

    apply_phase_response=True
    n_phi = 360
    n_theta = 360

    upsample = 2**15

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

        cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, n_theta=n_theta, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter,apply_phase_response=apply_phase_response)

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

        if False:

            for run in [1507,1509,1511]:

                if run == 1507:
                    waveform_index_range = (1500,None) #Looking at the later bit of the waveform only, 10000 will cap off.  
                elif run == 1509:
                    waveform_index_range = (2000,3000) #Looking at the later bit of the waveform only, 10000 will cap off.
                elif run == 1511:
                    waveform_index_range = (1250,2000) #Looking at the later bit of the waveform only, 10000 will cap off.  


                reader = Reader(datapath,run)

                known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
                eventids = {}
                eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])[0:5]
                eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])[0:5]
                all_eventids = numpy.sort(numpy.append(eventids['hpol'],eventids['vpol']))

                hpol_eventids_cut = numpy.isin(all_eventids,eventids['hpol'])
                vpol_eventids_cut = numpy.isin(all_eventids,eventids['vpol'])

                cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, n_theta=n_theta, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter,apply_phase_response=apply_phase_response)
                tdc = TimeDelayCalculator(reader, final_corr_length=upsample, crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None,waveform_index_range=(None, None),plot_filters=False,apply_phase_response=True)
                

                if True:
                    for mode in ['hpol','vpol']:

                        site1_measured_time_delays =  {'hpol':{'[0, 1]' : [-33.67217794735314], '[0, 2]': [105.45613390568445], '[0, 3]': [30.139256769961403], '[1, 2]': [139.22210622057898], '[1, 3]': [64.03028824157775], '[2, 3]': [-75.19181797900121]},\
                                                       'vpol':{'[0, 1]' : [-38.04924843261725], '[0, 2]': [101.73562399320996], '[0, 3]': [36.36094981687252], '[1, 2]': [139.78487242582722], '[1, 3]': [74.50399261703114], '[2, 3]': [-65.34340938715698]}}
                        site2_measured_time_delays =  {'hpol':{'[0, 1]' : [-75.67357058767381], '[0, 2]': [40.63430060469886], '[0, 3]': [-44.603959202234826], '[1, 2]': [116.27661403806135], '[1, 3]': [31.225897156995508], '[2, 3]': [-85.14448834399977]},\
                                                       'vpol':{'[0, 1]' : [-79.95580072832284], '[0, 2]': [36.75841347009681], '[0, 3]': [-38.38378549428477], '[1, 2]': [116.62044273548572], '[1, 3]': [41.665786696971985], '[2, 3]': [-75.11094181007027]}}
                        site3_measured_time_delays =  {'hpol':{'[0, 1]' : [-88.02014654064], '[0, 2]': [-143.62662406045482], '[0, 3]': [-177.19680779079835], '[1, 2]': [-55.51270605688091], '[1, 3]': [-88.89534686135659], '[2, 3]': [-33.32012649585307]},\
                                                       'vpol':{'[0, 1]' : [-92.05231944679858], '[0, 2]': [-147.1274253433212], '[0, 3]': [-172.476977489791], '[1, 2]': [-55.10636305083391], '[1, 3]': [-80.36214373436982], '[2, 3]': [-25.34955214646983]}}
                        
                        if run == 1507:
                            time_delay_dict = site1_measured_time_delays
                        elif run == 1509:
                            time_delay_dict = site2_measured_time_delays  
                        elif run == 1511:
                            time_delay_dict = site3_measured_time_delays
                        else:
                            time_delay_dict = None

                        eventid = eventids[mode][0]

                        pulser_locations_ENU = info.loadPulserPhaseLocationsENU()[mode]['run%i'%run]
                        #Do I need to adjust the height of the pulser?  Cosmin thinks there is a 20 m discrepency in alt between the two metrics.
                        #pulser_locations_ENU[2] -= 20.0
                        pulser_locations_ENU = pulser_locations_ENU/numpy.linalg.norm(pulser_locations_ENU)


                        pulser_theta = numpy.degrees(numpy.arccos(pulser_locations_ENU[2]))
                        pulser_phi = numpy.degrees(numpy.arctan2(pulser_locations_ENU[1],pulser_locations_ENU[0]))

                        print('%s Expected pulser location: Zenith = %0.2f, Az = %0.2f'%(mode.title(), pulser_theta,pulser_phi))

                        #ax.axvline(pulser_phi,c='r')
                        #ax.axhline(pulser_theta,c='r')

                        mean_corr_values, fig, ax = cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=False, interactive=True, max_method=max_method,circle_zenith=pulser_theta,circle_az=pulser_phi,time_delay_dict=time_delay_dict)
                        all_figs.append(fig)
                        all_axs.append(ax)
                if False:
                    #SHOULD FIGURE OUT HOW THE BEST MEASURED TIME DELAYS FOR THE PULSERS AND PLOT THEIR RINGS.


                    for mode in ['hpol','vpol']:
                        time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids[mode],align_method=0,hilbert=True,plot=False,hpol_cut=None,vpol_cut=None)
                        
                        td_dict = {'hpol':{},'vpol':{}}
                        for pair_index, pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
                            td_dict['hpol'][str(pair)] = time_shifts[0:12:2]
                            td_dict['vpol'][str(pair)] = time_shifts[1:12:2]

                        pulser_locations_ENU = info.loadPulserPhaseLocationsENU()[mode]['run%i'%run]
                        pulser_locations_ENU = pulser_locations_ENU/numpy.linalg.norm(pulser_locations_ENU)
                        pulser_theta = numpy.degrees(numpy.arccos(pulser_locations_ENU[2]))
                        pulser_phi = numpy.degrees(numpy.arctan2(pulser_locations_ENU[1],pulser_locations_ENU[0]))
                        print('%s Expected pulser location: Zenith = %0.2f, Az = %0.2f'%(mode.title(), pulser_theta,pulser_phi))

                        mean_corr_values, fig, ax = cor.averagedMap(eventids[mode], mode, plot_map=True, hilbert=False, max_method=max_method,center_dir='E',circle_zenith=pulser_theta,circle_az=pulser_phi)


                        #ax.axvline(pulser_phi,c='r')
                        #ax.axhline(pulser_theta,c='r')

                    all_figs.append(fig)
                    all_axs.append(ax)
                all_cors.append(cor)

        if True:
            #Preparing for planes:
            known_planes, calibrated_trigtime, output_tracks = pt.getKnownPlaneTracks()

            plane_polys = {}
            cors = []
            interpolated_plane_locations = {}
            origin = info.loadAntennaZeroLocation(deploy_index = 1)
            plot_baseline_removal_hists = False
            plot_compare_trajectory = False
            plot_individual_maps = False
            plot_residual_hists = True

            all_event_selected_theta_residual = {}
            all_event_selected_phi_residual = {}

            for index, key in enumerate(list(known_planes.keys())):
                print(key)
                # if known_planes[key]['dir'] != 'E':
                #     continue

                enu = pm.geodetic2enu(output_tracks[key]['lat'],output_tracks[key]['lon'],output_tracks[key]['alt'],origin[0],origin[1],origin[2])
                plane_polys[key] = pt.PlanePoly(output_tracks[key]['timestamps'],enu,plot=False)

                interpolated_plane_locations[key] = plane_polys[key].poly(calibrated_trigtime[key])
                normalized_plane_locations = interpolated_plane_locations[key]/(numpy.tile(numpy.linalg.norm(interpolated_plane_locations[key],axis=1),(3,1)).T)

                run = numpy.unique(known_planes[key]['eventids'][:,0])[0]
                calibrated_trigtime[key] = numpy.zeros(len(known_planes[key]['eventids'][:,0]))
                #Mostly expected to have only 1 run per plane.  I.e. not being visible across runs. 
                run_cut = known_planes[key]['eventids'][:,0] == run
                reader = Reader(os.environ['BEACON_DATA'],run)
                eventids = known_planes[key]['eventids'][run_cut,1]


                td_dict = {'hpol':{},'vpol':{}}
                for k in td_dict.keys():
                    pair_cut = numpy.array([pair in known_planes[key]['baselines'][k] for pair in [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]] ]) #Checks which pairs are worth looping over.
                    measured_plane_time_delays = known_planes[key]['time_delays'][k].T[pair_cut]
                    for pair_index, pair in enumerate(known_planes[key]['baselines'][k]):
                        td_dict[k][str(pair)] = [measured_plane_time_delays[pair_index][0]] #Needs to be a list or array.  For now just putting one element.

                plane_zenith = numpy.rad2deg(numpy.arccos(normalized_plane_locations[:,2]))
                plane_az = numpy.rad2deg(numpy.arctan2(normalized_plane_locations[:,1],normalized_plane_locations[:,0]))

                test_az = plane_az[len(plane_az)//2]%360

                if numpy.logical_or(test_az >= 270 + 45, test_az < 45):
                    _dir = 'E'
                elif numpy.logical_and(test_az >= 0 + 45, test_az < 90 + 45):
                    _dir = 'N'
                elif numpy.logical_and(test_az >= 90 + 45, test_az < 180 + 45):
                    _dir = 'W'
                elif numpy.logical_and(test_az >= 180 + 45, test_az < 270 + 45):
                    _dir = 'S'

                cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, n_theta=n_theta, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True)
                if True:
                    cor.prep.addSineSubtract(0.03, 0.090, 0.05, max_failed_iterations=3, verbose=False, plot=False)#Test purposes
                    cor.prep.addSineSubtract(0.09, 0.250, 0.001, max_failed_iterations=3, verbose=False, plot=False)#Test purposes
                    #cor.prep.addSineSubtract(0.03, 0.090, 0.05, max_failed_iterations=3, verbose=True, plot=False)#Test purposes
                #Load Values
                #cor.animatedMap(eventids, 'hpol', key, plane_zenith=plane_zenith,plane_az=plane_az,hilbert=False, max_method=None,center_dir=_dir,save=True,dpi=300)
                hilbert = False
                zenith_cut_array_plane = [0,110]
                print(_dir)
                map_values, fig, ax = cor.map(eventids[0], 'hpol',center_dir=_dir, plot_map=True, plot_corr=False, hilbert=hilbert, interactive=True, max_method=None,mollweide=True,circle_zenith=plane_zenith[0],circle_az=plane_az[0],zenith_cut_array_plane=zenith_cut_array_plane)
                cor.prep.plotEvent(eventids[0], channels=[0,2,4,6], apply_filter=cor.apply_filter, hilbert=hilbert, sine_subtract=cor.apply_sine_subtract, apply_tukey=cor.apply_tukey)
                '''
                test_planes = cor.getTimeDelayCurves(td_dict, 'hpol')

                loop_indices = []
                for angle_index in range(len(test_planes['[0, 1]']['azimuth_deg'][0])):
                    az = test_planes['[0, 1]']['azimuth_deg'][0][angle_index]
                    zen = test_planes['[0, 1]']['zenith_deg'][0][angle_index]

                    loop_indices.append(numpy.argmin(numpy.sqrt(numpy.abs(cor.mesh_azimuth_deg - az)**2 + numpy.abs(cor.mesh_zenith_deg - zen)**2)))

                import pdb; pdb.set_trace()
                '''

                zenith_cut_ENU = None#[0,170]
                '''
                #cor.calculateArrayNormalVector(plot_map=True,mollweide=True, pol='both')
                all_baselines_matrix_minus_1_baseline = numpy.array([[1,2,3,4,5],[0,2,3,4,5],[0,1,3,4,5],[0,1,2,4,5],[0,1,2,3,5],[0,1,2,3,4]])
                all_baselines_matrix_minus_1_antenna = numpy.array([[3,4,5],[1,2,5],[0,2,4],[0,1,3]])
                use_baseline_matrix = all_baselines_matrix_minus_1_antenna


                all_event_selected_theta = []
                all_event_selected_phi = []

                for event_index, eventid in enumerate(eventids):
                    all_theta_best_per_event = []
                    all_phi_best_per_event = []
                    for bl in use_baseline_matrix:
                        if plot_individual_maps:
                            map_values, fig, ax = cor.map(eventid, 'hpol', plot_map=plot_individual_maps, plot_corr=False, hilbert=False, interactive=True, max_method=max_method,mollweide=False,zenith_cut_ENU=zenith_cut_ENU,zenith_cut_array_plane=zenith_cut_array_plane,center_dir=_dir,circle_zenith=plane_zenith[0],circle_az=plane_az[0],time_delay_dict=td_dict.copy(),include_baselines=bl)
                        else:
                            map_values = cor.map(eventid, 'hpol', plot_map=plot_individual_maps, plot_corr=False, hilbert=False, interactive=True, max_method=max_method,mollweide=False,zenith_cut_ENU=zenith_cut_ENU,zenith_cut_array_plane=zenith_cut_array_plane,center_dir=_dir,circle_zenith=plane_zenith[0],circle_az=plane_az[0],time_delay_dict=td_dict.copy(),include_baselines=bl)
                        
                        row_index, column_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = cor.mapMax(map_values, max_method=0, verbose=False, zenith_cut_ENU=None, zenith_cut_array_plane=[0,100], pol='hpol')
                        print(key, bl, theta_best, phi_best)
                        all_theta_best_per_event.append(theta_best)
                        all_phi_best_per_event.append(phi_best)


                    if event_index == 0:
                        all_event_selected_theta = all_theta_best_per_event
                        all_event_selected_phi = all_phi_best_per_event

                        all_event_selected_theta_residual[key] = all_theta_best_per_event - plane_zenith[event_index]
                        all_event_selected_phi_residual[key] = all_phi_best_per_event - plane_az[event_index]
                    else:
                        all_event_selected_theta = numpy.vstack((all_event_selected_theta,all_theta_best_per_event))
                        all_event_selected_phi = numpy.vstack((all_event_selected_phi,all_phi_best_per_event))

                        all_event_selected_theta_residual[key] = numpy.vstack((all_event_selected_theta_residual[key],all_theta_best_per_event  - plane_zenith[event_index] ))
                        all_event_selected_phi_residual[key] = numpy.vstack((all_event_selected_phi_residual[key],all_phi_best_per_event  - plane_az[event_index] ))


                    if plot_baseline_removal_hists == True:
                        plt.figure()
                        plt.suptitle(key + str(eventid))
                        plt.subplot(2,1,1)
                        plt.hist(all_theta_best_per_event,label='Map Max')
                        plt.axvline(plane_zenith[event_index],c='r',linestyle='--',label='Plane zenith')
                        plt.legend()
                        plt.xlabel('theta_best for each subset of baselines')
                        plt.ylabel('Counts')
                        plt.subplot(2,1,2)
                        plt.hist(all_phi_best_per_event,label='Map Max')
                        plt.axvline(plane_az[event_index],c='r',linestyle='--',label='Plane az')
                        plt.legend()
                        plt.xlabel('phi_best for each subset of baselines')
                        plt.ylabel('Counts')

                all_event_selected_theta_means = numpy.mean(all_event_selected_theta,axis=1)
                all_event_selected_phi_means = numpy.mean(all_event_selected_phi,axis=1)

                if plot_compare_trajectory == True:
                    print('mean_thetas = ',all_event_selected_theta_means)
                    print('mean_phis = ',all_event_selected_phi_means)
                    plt.figure()
                    plt.plot(all_event_selected_phi_means,all_event_selected_theta_means)
                    plt.scatter(all_event_selected_phi_means,all_event_selected_theta_means,label='Measured from Maps')

                    plt.plot(plane_az,plane_zenith)
                    plt.scatter(plane_az,plane_zenith,label='Expected for Plane')

                    plt.legend()
                    plt.title(key)
                '''
            '''
            if plot_residual_hists == True:
                bins = numpy.linspace(-15,15,101 )
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                for baseline_index, bl in enumerate(use_baseline_matrix):
                    plt.figure()
                    plt.suptitle('Residuals for All Events in All Planes')
                    for index, key in enumerate(list(known_planes.keys())):
                        if index == 0:
                            theta_res = all_event_selected_theta_residual[key].T[baseline_index]
                            phi_res = all_event_selected_phi_residual[key].T[baseline_index]
                        else:
                            theta_res = numpy.append(theta_res,all_event_selected_theta_residual[key].T[baseline_index])
                            phi_res = numpy.append(phi_res,all_event_selected_phi_residual[key].T[baseline_index])


                    plt.subplot(2,1,1)
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.hist(theta_res,label=str(bl),alpha=0.8,bins=bins,color=colors[baseline_index])
                    plt.legend()
                    plt.xlabel('theta_best for each subset of baselines')
                    plt.ylabel('Counts')
                    plt.subplot(2,1,2)
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.hist(phi_res,label=str(bl),alpha=0.8,bins=bins,color=colors[baseline_index])
                    plt.legend()
                    plt.xlabel('phi_best for each subset of baselines')
                    plt.ylabel('Counts')

                #cor.map(eventids[0], 'hpol', plot_map=True, plot_corr=False, hilbert=False, interactive=True, max_method=max_method,mollweide=True,zenith_cut_ENU=zenith_cut_ENU,zenith_cut_array_plane=zenith_cut_array_plane,center_dir='W',circle_zenith=plane_zenith[0],circle_az=plane_az[0])
                # cor.map(eventids[0], 'hpol', plot_map=True, plot_corr=False, hilbert=True, interactive=True, max_method=max_method,mollweide=True,zenith_cut_ENU=zenith_cut_ENU,center_dir='W',circle_zenith=plane_zenith[0],circle_az=plane_az[0])
                #mean_corr_values, fig, ax = cor.averagedMap(eventids[0:2], 'hpol', plot_map=True, hilbert=False, max_method=max_method,mollweide=True,zenith_cut_ENU=zenith_cut_ENU,center_dir='W',zenith_cut_array_plane=zenith_cut_array_plane,circle_zenith=plane_zenith[0],circle_az=plane_az[0])
                # mean_corr_values, fig, ax = cor.averagedMap(eventids[0:2], 'hpol', plot_map=True, hilbert=False, max_method=max_method,mollweide=True,zenith_cut_ENU=zenith_cut_ENU,center_dir='W')
                # # for eventid in eventids:
                #     cor.map(eventid, 'hpol', plot_map=True, plot_corr=False, hilbert=False, interactive=True, max_method=max_method,mollweide=False,zenith_cut_ENU=zenith_cut_ENU,center_dir='W')
                #cor.animatedMap(eventids, 'hpol', key, plane_zenith=plane_zenith,plane_az=plane_az,hilbert=False, max_method=None,center_dir='W',save=True,dpi=600)

                #TODO: ADD A FEATURE THAT DOES WHAT ZENITH_CUT_ENU DOES BUT FOR ANGLES RELATIVE TO THE ARRAY PLANE. 

                cors.append(cor) #Need to keep references for animations to work. 

            '''

        if False:
            #Test CW Subtraction:
            run = 1650
            eventids = [499,45059,58875]
            mode = 'hpol'
            # run = 1507
            # eventids = [18453]
            # mode = 'vpol'
                        
            hilbert = False

            reader = Reader(os.environ['BEACON_DATA'],run)
            cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, n_theta=n_theta, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True)

            if True:
                cor.prep.addSineSubtract(0.03, 0.090, 0.01, max_failed_iterations=3, verbose=False, plot=False)#Test purposes
                cor.prep.addSineSubtract(0.09, 0.250, 0.001, max_failed_iterations=3, verbose=False, plot=False)#Test purposes
                #cor.prep.addSineSubtract(0.10, 0.13, 0.05, max_failed_iterations=3, verbose=False, plot=False)#Test purposes
                #cor.prep.addSineSubtract(0.117, 0.119, 0.001, max_failed_iterations=3, verbose=False, plot=False)#Test purposes
                #for i in range(10):

            for eventid in eventids:
                for ss in [False, True]:
                    cor.apply_sine_subtract = ss
                    map_values, fig, ax = cor.map(eventid, mode,center_dir='E', plot_map=True, plot_corr=False, hilbert=hilbert, interactive=True, max_method=None,mollweide=True,circle_zenith=None,circle_az=None)

                    if mode == 'hpol':
                        channels = [0,2,4,6]
                    else:
                        channels = [1,3,5,7]

                    print(cor.apply_sine_subtract)
                    cor.prep.plotEvent(eventid, channels=channels, apply_filter=cor.apply_filter, hilbert=hilbert, sine_subtract=cor.apply_sine_subtract, apply_tukey=cor.apply_tukey)
