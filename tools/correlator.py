#!/usr/bin/env python3
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
import itertools

#from    beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
import  beacon.tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import  beacon.tools.info as info
from    beacon.tools.data_handler import getEventTimes
import  beacon.analysis.phase_response as pr
import  beacon.tools.get_plane_tracks as pt
from    beacon.tools.fftmath import FFTPrepper, TimeDelayCalculator
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches 
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats
import scipy.signal
import scipy.stats
from scipy.optimize import curve_fit
def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))
from scipy.linalg import lstsq
import scipy.ndimage
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
    map_source_distance_m : float
        The line of sight distance to the imagined source of a signal for which maps will be produced.  By default this
        is 10^6 m (>600miles), which essentially makes the arrival angles at each antenna identical for each percieved source
        direction.  This is the far-field limit and is the default.  This can be set to a smaller value, allowing for 
        predicted arrival time delays to be more accurate for sources such as pusers (which are only a few hundred
        meters away).  To change the distance call the overwriteSourceDistance() function, which will implement the
        necessary adjustments for the new source distance.
    '''
    def __init__(self, reader,  upsample=None, n_phi=181, range_phi_deg=(-180,180), n_theta=361, range_theta_deg=(0,180), crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=False, waveform_index_range=(None,None), apply_phase_response=True, tukey=True, sine_subtract=True, map_source_distance_m=1e6, deploy_index=None, all_alignments=False):
        try:
            self.conference_mode = False #Enable to apply any temporary adjustments such as fontsizes or title labels. 
            if deploy_index is None:
                self.deploy_index = info.returnDefaultDeploy()
            else:
                self.deploy_index = deploy_index 

            self.all_alignments = all_alignments #Determines whether to plot a single shift of time delays or all.

            n = 1.0003 #Index of refraction of air  #Should use https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf 
            self.c = 299792458.0/n #m/s
            self.min_elevation_linewidth = 0.5
            self.reader = reader

            self.ss_reader_mode = False
            if hasattr(self.reader, "ss_event_file"):
                if self.reader.ss_event_file is not None:
                    print('Sine Subtracted Reader detected and ss_event_file appears to be present.  Any sine subtraction added to this Correlator object will be ignored, assuming that it will be automatically handled via precomputed sine subtraction values.')
                    self.ss_reader_mode = True

            if upsample is None:
                self.upsample = len(self.reader.t())
            else:
                self.upsample = upsample

            self.apply_tukey = tukey
            self.apply_sine_subtract = sine_subtract or self.ss_reader_mode #Used in some places just to have titles portray accurately that sine subtraction was used.  If ss_reader_mode is True then this is largely actually ignored and the ss reader is used with sine subtracted waveforms.
            '''
            Note that the definition of final_corr_length in FFTPrepper and upsample in this are different (off by about 
            a factor of 2).  final_corr_length is not actually being used in correlator, all signal upsampling happens
            internally.  FFTPrepper is being used to apply filters to the loaded waveforms such that any changes to
            filtering only need to happen in a single location/class.
            '''
            self.prep = FFTPrepper(self.reader, final_corr_length=2**10, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=plot_filter,tukey_alpha=0.1,tukey_default=False,apply_phase_response=apply_phase_response)
            self.prepareTimes() 
            if numpy.all(self.prep.filter_original == 1.0):
                self.apply_filter = False
            else:
                self.apply_filter = True

            self.figs = []
            self.axs = []
            self.animations = []

            cable_delays = info.loadCableDelays(deploy_index=self.deploy_index)
            self.cable_delays = numpy.array([cable_delays['hpol'][0],cable_delays['vpol'][0],cable_delays['hpol'][1],cable_delays['vpol'][1],cable_delays['hpol'][2],cable_delays['vpol'][2],cable_delays['hpol'][3],cable_delays['vpol'][3]])

            self.range_theta_deg = range_theta_deg
            self.n_theta = n_theta
            self.range_phi_deg = range_phi_deg
            self.n_phi = n_phi
            self.thetas_deg = numpy.linspace(min(range_theta_deg),max(range_theta_deg),n_theta) #Zenith angle
            self.phis_deg = numpy.linspace(min(range_phi_deg),max(range_phi_deg),n_phi) #Azimuth angle
            self.thetas_rad = numpy.deg2rad(self.thetas_deg)
            self.phis_rad = numpy.deg2rad(self.phis_deg)

            self.mesh_azimuth_rad, self.mesh_elevation_rad = numpy.meshgrid(numpy.deg2rad(numpy.linspace(min(range_phi_deg),max(range_phi_deg),n_phi)), numpy.pi/2.0 - numpy.deg2rad(numpy.linspace(min(range_theta_deg),max(range_theta_deg),n_theta)))
            self.mesh_azimuth_deg, self.mesh_elevation_deg = numpy.meshgrid(numpy.linspace(min(range_phi_deg),max(range_phi_deg),n_phi), 90.0 - numpy.linspace(min(range_theta_deg),max(range_theta_deg),n_theta))
            self.mesh_zenith_deg = 90.0 - self.mesh_elevation_deg
            self.mesh_zenith_rad = numpy.deg2rad(self.mesh_zenith_deg)
            
            self.original_A0_latlonel = numpy.array(info.loadAntennaZeroLocation(deploy_index=self.deploy_index)) #This will not change.  Used as reference point when new antenna positions given. 
            self.A0_latlonel_physical = self.original_A0_latlonel.copy() #Used for looking at planes.  Theoretically this should perfectly align with antenna 0, as antenna 0 can be adjusted, it will move accordingly.  
            self.A0_latlonel_hpol = self.original_A0_latlonel.copy()
            self.A0_latlonel_vpol = self.original_A0_latlonel.copy()

            antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=self.deploy_index, verbose=True)

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

            if map_source_distance_m is None:
                self.map_source_distance_m = 1e6
            else:
                self.map_source_distance_m = map_source_distance_m
            self.recalculateLatLonEl()
            self.generateTimeIndices() #Must be called again if map_source_distance_m is reset
            self.calculateArrayNormalVector()

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def recalculateLatLonEl(self):
        '''
        Given the current physical, hpol, and vpol antenna locations, this will convert those ENU coordinates (with 
        reference to the original input lat lon el) to the shifted lat lon el.  This can be used as the origin for
        calculated direction to airplanes etc.  
        '''
        try:
            self.A0_latlonel_physical   = pm.enu2geodetic(self.A0_physical[0],self.A0_physical[1],self.A0_physical[2],self.original_A0_latlonel[0],self.original_A0_latlonel[1],self.original_A0_latlonel[2])
            self.A0_latlonel_hpol       = pm.enu2geodetic(self.A0_hpol[0],self.A0_hpol[1],self.A0_hpol[2],self.original_A0_latlonel[0],self.original_A0_latlonel[1],self.original_A0_latlonel[2])
            self.A0_latlonel_vpol       = pm.enu2geodetic(self.A0_vpol[0],self.A0_vpol[1],self.A0_vpol[2],self.original_A0_latlonel[0],self.original_A0_latlonel[1],self.original_A0_latlonel[2])
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)        

    def overwriteAntennaLocations(self, A0_physical,A1_physical,A2_physical,A3_physical,A0_hpol,A1_hpol,A2_hpol,A3_hpol,A0_vpol,A1_vpol,A2_vpol,A3_vpol, verbose=True, suppress_time_delay_calculations=False):
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
        verbose : bool
            Enables or disables print statements.
        suppress_time_delay_calculations : bool
            USE WITH CAUTION.  This will disabled the time delay table from being calculated when the antennas are input.
            In most cases this will result incorrect maps.  The only obvious scenario in which this should be used is if
            both antenna positions AND cable delays or source distance are being reset at the same time, in which case 
            the time delay tables would only need to be calculated after both are altered.  Even in this scenario, if 
            time isn't really limited for safety I would always have this as False. 
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
            
            if suppress_time_delay_calculations == False:
                if verbose:
                    print('Rerunning time delay prep with antenna positions.')
                self.recalculateLatLonEl()
                self.generateTimeIndices() #Must be called again if map_source_distance_m is reset
                self.calculateArrayNormalVector()
            else:
                if verbose:
                    print('WARNING!  Time Indices NOT recalculated in overwriteAntennaLocations.')
                    print('WARNING!  Array Normal Vector NOT recalculated in overwriteAntennaLocations.')


        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def fitPlaneGetNorm(self,xyz,verbose=False):
        '''
        Given an array of similarly lengthed x,y,z data, this will fit a plane to the data, and then
        determine the normal vector to that plane and return it.  The xyz data should be stacked such that
        the first column is x, second is y, and third is z.
        '''
        try:
            x = xyz[:,0]
            y = xyz[:,1]
            z = xyz[:,2]

            A = numpy.matrix(numpy.vstack((x,y,numpy.ones_like(x))).T)
            b = numpy.matrix(z).T
            fit, residual, rnk, s = lstsq(A, b)
            if verbose:
                print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
            plane_func = lambda _x, _y : fit[0]*_x + fit[1]*_y + fit[2]
            zero = numpy.array([0,0,plane_func(0,0)[0]]) #2 points in plane per vector, common point at 0,0 between the 2 vectors. 

            v0 = numpy.array([1,0,plane_func(1,0)[0]]) - zero
            v0 = v0/numpy.linalg.norm(v0)
            v1 = numpy.array([0,1,plane_func(0,1)[0]]) - zero
            v1 = v1/numpy.linalg.norm(v1)
            norm = numpy.cross(v0,v1)
            norm = norm/numpy.linalg.norm(norm)
            return norm
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def calculateArrayNormalVector(self,plot_map=False,mollweide=False, pol='both',method='fit',verbose=False):
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

            if method == 'vec':
                if len(included_antennas) < 4:
                    print('WARNING, vec METHOD OF ARRAY PLANE DOES NOT SUPPPORT LIMITING INCLUDED ANTENNAS.')
                self.n_physical = n_func(self.A0_physical, self.A2_physical, (self.A0_physical + self.A3_physical)/2.0)
                self.n_hpol     = n_func(self.A0_hpol,     self.A2_hpol,     (self.A0_hpol     + self.A3_hpol)/2.0)
                self.n_vpol     = n_func(self.A0_vpol,     self.A2_vpol,     (self.A0_vpol     + self.A3_vpol)/2.0)
                self.n_all      = n_func((self.A0_hpol + self.A0_vpol),     (self.A2_hpol + self.A2_vpol),     ((self.A0_hpol + self.A0_vpol)     + (self.A3_hpol + self.A3_vpol))/2.0) #Might be janky to use.

            elif method == 'fit':
                self.n_physical = self.fitPlaneGetNorm(numpy.vstack((self.A0_physical,self.A1_physical,self.A2_physical,self.A3_physical)))
                self.n_hpol     = self.fitPlaneGetNorm(numpy.vstack((self.A0_hpol,self.A1_hpol,self.A2_hpol,self.A3_hpol)))
                self.n_vpol     = self.fitPlaneGetNorm(numpy.vstack((self.A0_vpol,self.A1_vpol,self.A2_vpol,self.A3_vpol)))
                self.n_all      = n_func((self.A0_hpol + self.A0_vpol),     (self.A2_hpol + self.A2_vpol),     ((self.A0_hpol + self.A0_vpol)     + (self.A3_hpol + self.A3_vpol))/2.0) #Might be janky to use.
                
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

    def overwriteCableDelays(self, ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, verbose=True, suppress_time_delay_calculations=False):
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
        suppress_time_delay_calculations : bool
            USE WITH CAUTION.  This will disabled the time delay table from being calculated when the antennas are input.
            In most cases this will result incorrect maps.  The only obvious scenario in which this should be used is if
            both antenna positions AND cable delays or source distance are being reset at the same time, in which case 
            the time delay tables would only need to be calculated after both are altered.  Even in this scenario, if 
            time isn't really limited for safety I would always have this as False. 
        '''
        try:
            self.cable_delays = numpy.array([ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7])

            if suppress_time_delay_calculations == False:
                if verbose:
                    print('Rerunning time delay prep with new cable delays.')
                self.generateTimeIndices() #Must be called again if map_source_distance_m is reset
            else:
                if verbose:
                    print('WARNING!  Time Indices NOT recalculated in overwriteSourceDistance.')

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def overwriteSourceDistance(self, map_source_distance_m, verbose=True, suppress_time_delay_calculations=False, debug=False):
        '''
        Allows user to reset the source distance use when generating maps.
        
        Parameters
        ----------
        map_source_distance_m : float
            The new source distance, given in meters.
        suppress_time_delay_calculations : bool
            USE WITH CAUTION.  This will disabled the time delay table from being calculated when the antennas are input.
            In most cases this will result incorrect maps.  The only obvious scenario in which this should be used is if
            both antenna positions AND cable delays or source distance are being reset at the same time, in which case 
            the time delay tables would only need to be calculated after both are altered.  Even in this scenario, if 
            time isn't really limited for safety I would always have this as False. 
        '''
        try:
            self.map_source_distance_m = map_source_distance_m
            if suppress_time_delay_calculations == False:
                if verbose:
                    print('Rerunning time delay prep with new source distance.')
                self.generateTimeIndices(debug=debug) #Must be called again if map_source_distance_m is reset
            else:
                if verbose:
                    print('WARNING!  Time Indices NOT recalculated in overwriteSourceDistance.')

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
                temp_wf = self.prep.wf(channel,apply_filter=apply_filter,hilbert=hilbert,tukey=tukey,sine_subtract=sine_subtract, return_sine_subtract_info=False) #May apply sine subtraction automatically if the self.prep was handed a sineSubtractedReader instead of the default Reader

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

    def generateTimeIndices(self,debug=False):
        '''
        This is meant to calculate all of the time delays corresponding to each source direction in advance.  Should be
        called again if self.map_source_distance_m changes.

        Note that the phi and theta coordinates used to place distant objects are centered at antenna 0 of that
        polarizations respective phase center.
        '''
        try:
            # self.mesh_azimuth_rad
            # self.mesh_elevation_rad
            # self.mesh_zenith_rad

            #Source direction is the direction FROM BEACON ANTENNA 0 you look to see the source.
            #Position is a vector to a distant source
            #double check shift to center points!!
            signal_source_position_hpol        = numpy.zeros((self.mesh_azimuth_rad.shape[0], self.mesh_azimuth_rad.shape[1], 3))
            signal_source_position_hpol[:,:,0] = self.map_source_distance_m * numpy.multiply( numpy.cos(self.mesh_azimuth_rad) , numpy.sin(self.mesh_zenith_rad) ) + self.A0_hpol[0] #Shifting points to be centered around antenna 0
            signal_source_position_hpol[:,:,1] = self.map_source_distance_m * numpy.multiply( numpy.sin(self.mesh_azimuth_rad) , numpy.sin(self.mesh_zenith_rad) ) + self.A0_hpol[1] #Shifting points to be centered around antenna 0
            signal_source_position_hpol[:,:,2] = self.map_source_distance_m * numpy.cos(self.mesh_zenith_rad) + self.A0_hpol[2] #Shifting points to be centered around antenna 0

            signal_source_position_vpol        = numpy.zeros((self.mesh_azimuth_rad.shape[0], self.mesh_azimuth_rad.shape[1], 3))
            signal_source_position_vpol[:,:,0] = self.map_source_distance_m * numpy.multiply( numpy.cos(self.mesh_azimuth_rad) , numpy.sin(self.mesh_zenith_rad) ) + self.A0_vpol[0] #Shifting points to be centered around antenna 0
            signal_source_position_vpol[:,:,1] = self.map_source_distance_m * numpy.multiply( numpy.sin(self.mesh_azimuth_rad) , numpy.sin(self.mesh_zenith_rad) ) + self.A0_vpol[1] #Shifting points to be centered around antenna 0
            signal_source_position_vpol[:,:,2] = self.map_source_distance_m * numpy.cos(self.mesh_zenith_rad) + self.A0_vpol[2] #Shifting points to be centered around antenna 0

            if debug:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.scatter(signal_source_position_hpol[:,:,0],signal_source_position_hpol[:,:,1],signal_source_position_hpol[:,:,2],marker=',',alpha=0.3)
                ax.scatter(self.A0_hpol[0],self.A0_hpol[1],self.A0_hpol[2],label='A0',c='tab:blue')
                ax.scatter(self.A1_hpol[0],self.A1_hpol[1],self.A1_hpol[2],label='A1',c='tab:orange')
                ax.scatter(self.A2_hpol[0],self.A2_hpol[1],self.A2_hpol[2],label='A2',c='tab:green')
                ax.scatter(self.A3_hpol[0],self.A3_hpol[1],self.A3_hpol[2],label='A3',c='tab:red')
                ax.set_xlabel('E (m)')
                ax.set_ylabel('N (m)')
                ax.set_zlabel('Relative Elevation (m)')
                plt.legend()

            #Calculate the expected readout time for each antenna (including cable delays)
            hpol_arrival_time_ns_0 = self.cable_delays[0] + (numpy.sqrt((signal_source_position_hpol[:,:,0] - self.A0_hpol[0])**2 + (signal_source_position_hpol[:,:,1] - self.A0_hpol[1])**2 + (signal_source_position_hpol[:,:,2] - self.A0_hpol[2])**2 )/self.c)*1.0e9 #ns
            hpol_arrival_time_ns_1 = self.cable_delays[2] + (numpy.sqrt((signal_source_position_hpol[:,:,0] - self.A1_hpol[0])**2 + (signal_source_position_hpol[:,:,1] - self.A1_hpol[1])**2 + (signal_source_position_hpol[:,:,2] - self.A1_hpol[2])**2 )/self.c)*1.0e9 #ns
            hpol_arrival_time_ns_2 = self.cable_delays[4] + (numpy.sqrt((signal_source_position_hpol[:,:,0] - self.A2_hpol[0])**2 + (signal_source_position_hpol[:,:,1] - self.A2_hpol[1])**2 + (signal_source_position_hpol[:,:,2] - self.A2_hpol[2])**2 )/self.c)*1.0e9 #ns
            hpol_arrival_time_ns_3 = self.cable_delays[6] + (numpy.sqrt((signal_source_position_hpol[:,:,0] - self.A3_hpol[0])**2 + (signal_source_position_hpol[:,:,1] - self.A3_hpol[1])**2 + (signal_source_position_hpol[:,:,2] - self.A3_hpol[2])**2 )/self.c)*1.0e9 #ns

            vpol_arrival_time_ns_0 = self.cable_delays[1] + (numpy.sqrt((signal_source_position_vpol[:,:,0] - self.A0_vpol[0])**2 + (signal_source_position_vpol[:,:,1] - self.A0_vpol[1])**2 + (signal_source_position_vpol[:,:,2] - self.A0_vpol[2])**2 )/self.c)*1.0e9 #ns
            vpol_arrival_time_ns_1 = self.cable_delays[3] + (numpy.sqrt((signal_source_position_vpol[:,:,0] - self.A1_vpol[0])**2 + (signal_source_position_vpol[:,:,1] - self.A1_vpol[1])**2 + (signal_source_position_vpol[:,:,2] - self.A1_vpol[2])**2 )/self.c)*1.0e9 #ns
            vpol_arrival_time_ns_2 = self.cable_delays[5] + (numpy.sqrt((signal_source_position_vpol[:,:,0] - self.A2_vpol[0])**2 + (signal_source_position_vpol[:,:,1] - self.A2_vpol[1])**2 + (signal_source_position_vpol[:,:,2] - self.A2_vpol[2])**2 )/self.c)*1.0e9 #ns
            vpol_arrival_time_ns_3 = self.cable_delays[7] + (numpy.sqrt((signal_source_position_vpol[:,:,0] - self.A3_vpol[0])**2 + (signal_source_position_vpol[:,:,1] - self.A3_vpol[1])**2 + (signal_source_position_vpol[:,:,2] - self.A3_vpol[2])**2 )/self.c)*1.0e9 #ns

            self.t_hpol_0subtract1 = hpol_arrival_time_ns_0 - hpol_arrival_time_ns_1
            self.t_hpol_0subtract2 = hpol_arrival_time_ns_0 - hpol_arrival_time_ns_2
            self.t_hpol_0subtract3 = hpol_arrival_time_ns_0 - hpol_arrival_time_ns_3
            self.t_hpol_1subtract2 = hpol_arrival_time_ns_1 - hpol_arrival_time_ns_2
            self.t_hpol_1subtract3 = hpol_arrival_time_ns_1 - hpol_arrival_time_ns_3
            self.t_hpol_2subtract3 = hpol_arrival_time_ns_2 - hpol_arrival_time_ns_3

            self.t_vpol_0subtract1 = vpol_arrival_time_ns_0 - vpol_arrival_time_ns_1
            self.t_vpol_0subtract2 = vpol_arrival_time_ns_0 - vpol_arrival_time_ns_2
            self.t_vpol_0subtract3 = vpol_arrival_time_ns_0 - vpol_arrival_time_ns_3
            self.t_vpol_1subtract2 = vpol_arrival_time_ns_1 - vpol_arrival_time_ns_2
            self.t_vpol_1subtract3 = vpol_arrival_time_ns_1 - vpol_arrival_time_ns_3
            self.t_vpol_2subtract3 = vpol_arrival_time_ns_2 - vpol_arrival_time_ns_3

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

    def generateExpectedTimeDelaysFromDir(self, phi_rad, theta_rad, return_indices=False, debug=False):
        '''
        This calculates the expected time delays for a signal coming from a source defined by theta and phi (in rad).
        This uses the source distance specified in the object.
        '''
        try:
            #Source direction is the direction FROM BEACON ANTENNA 0 you look to see the source.
            #Position is a vector to a distant source
            #double check shift to center points!!
            signal_source_position_hpol    = numpy.zeros(3)
            signal_source_position_hpol[0] = self.map_source_distance_m * numpy.multiply( numpy.cos(phi_rad) , numpy.sin(theta_rad) ) + self.A0_hpol[0] #Shifting points to be centered around antenna 0
            signal_source_position_hpol[1] = self.map_source_distance_m * numpy.multiply( numpy.sin(phi_rad) , numpy.sin(theta_rad) ) + self.A0_hpol[1] #Shifting points to be centered around antenna 0
            signal_source_position_hpol[2] = self.map_source_distance_m * numpy.cos(theta_rad) + self.A0_hpol[2] #Shifting points to be centered around antenna 0

            signal_source_position_vpol    = numpy.zeros(3)
            signal_source_position_vpol[0] = self.map_source_distance_m * numpy.multiply( numpy.cos(phi_rad) , numpy.sin(theta_rad) ) + self.A0_vpol[0] #Shifting points to be centered around antenna 0
            signal_source_position_vpol[1] = self.map_source_distance_m * numpy.multiply( numpy.sin(phi_rad) , numpy.sin(theta_rad) ) + self.A0_vpol[1] #Shifting points to be centered around antenna 0
            signal_source_position_vpol[2] = self.map_source_distance_m * numpy.cos(theta_rad) + self.A0_vpol[2] #Shifting points to be centered around antenna 0

            if debug:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.scatter(signal_source_position_hpol[0],signal_source_position_hpol[1],signal_source_position_hpol[2],marker=',',alpha=0.3)
                ax.scatter(self.A0_hpol[0],self.A0_hpol[1],self.A0_hpol[2],label='A0',c='tab:blue')
                ax.scatter(self.A1_hpol[0],self.A1_hpol[1],self.A1_hpol[2],label='A1',c='tab:orange')
                ax.scatter(self.A2_hpol[0],self.A2_hpol[1],self.A2_hpol[2],label='A2',c='tab:green')
                ax.scatter(self.A3_hpol[0],self.A3_hpol[1],self.A3_hpol[2],label='A3',c='tab:red')
                ax.set_xlabel('E (m)')
                ax.set_ylabel('N (m)')
                ax.set_zlabel('Relative Elevation (m)')
                plt.legend()

            #Calculate the expected readout time for each antenna (including cable delays)
            hpol_arrival_time_ns_0 = self.cable_delays[0] + (numpy.sqrt((signal_source_position_hpol[0] - self.A0_hpol[0])**2 + (signal_source_position_hpol[1] - self.A0_hpol[1])**2 + (signal_source_position_hpol[2] - self.A0_hpol[2])**2 )/self.c)*1.0e9 #ns
            hpol_arrival_time_ns_1 = self.cable_delays[2] + (numpy.sqrt((signal_source_position_hpol[0] - self.A1_hpol[0])**2 + (signal_source_position_hpol[1] - self.A1_hpol[1])**2 + (signal_source_position_hpol[2] - self.A1_hpol[2])**2 )/self.c)*1.0e9 #ns
            hpol_arrival_time_ns_2 = self.cable_delays[4] + (numpy.sqrt((signal_source_position_hpol[0] - self.A2_hpol[0])**2 + (signal_source_position_hpol[1] - self.A2_hpol[1])**2 + (signal_source_position_hpol[2] - self.A2_hpol[2])**2 )/self.c)*1.0e9 #ns
            hpol_arrival_time_ns_3 = self.cable_delays[6] + (numpy.sqrt((signal_source_position_hpol[0] - self.A3_hpol[0])**2 + (signal_source_position_hpol[1] - self.A3_hpol[1])**2 + (signal_source_position_hpol[2] - self.A3_hpol[2])**2 )/self.c)*1.0e9 #ns

            vpol_arrival_time_ns_0 = self.cable_delays[1] + (numpy.sqrt((signal_source_position_vpol[0] - self.A0_vpol[0])**2 + (signal_source_position_vpol[1] - self.A0_vpol[1])**2 + (signal_source_position_vpol[2] - self.A0_vpol[2])**2 )/self.c)*1.0e9 #ns
            vpol_arrival_time_ns_1 = self.cable_delays[3] + (numpy.sqrt((signal_source_position_vpol[0] - self.A1_vpol[0])**2 + (signal_source_position_vpol[1] - self.A1_vpol[1])**2 + (signal_source_position_vpol[2] - self.A1_vpol[2])**2 )/self.c)*1.0e9 #ns
            vpol_arrival_time_ns_2 = self.cable_delays[5] + (numpy.sqrt((signal_source_position_vpol[0] - self.A2_vpol[0])**2 + (signal_source_position_vpol[1] - self.A2_vpol[1])**2 + (signal_source_position_vpol[2] - self.A2_vpol[2])**2 )/self.c)*1.0e9 #ns
            vpol_arrival_time_ns_3 = self.cable_delays[7] + (numpy.sqrt((signal_source_position_vpol[0] - self.A3_vpol[0])**2 + (signal_source_position_vpol[1] - self.A3_vpol[1])**2 + (signal_source_position_vpol[2] - self.A3_vpol[2])**2 )/self.c)*1.0e9 #ns

            t_hpol_0subtract1 = hpol_arrival_time_ns_0 - hpol_arrival_time_ns_1
            t_hpol_0subtract2 = hpol_arrival_time_ns_0 - hpol_arrival_time_ns_2
            t_hpol_0subtract3 = hpol_arrival_time_ns_0 - hpol_arrival_time_ns_3
            t_hpol_1subtract2 = hpol_arrival_time_ns_1 - hpol_arrival_time_ns_2
            t_hpol_1subtract3 = hpol_arrival_time_ns_1 - hpol_arrival_time_ns_3
            t_hpol_2subtract3 = hpol_arrival_time_ns_2 - hpol_arrival_time_ns_3

            t_vpol_0subtract1 = vpol_arrival_time_ns_0 - vpol_arrival_time_ns_1
            t_vpol_0subtract2 = vpol_arrival_time_ns_0 - vpol_arrival_time_ns_2
            t_vpol_0subtract3 = vpol_arrival_time_ns_0 - vpol_arrival_time_ns_3
            t_vpol_1subtract2 = vpol_arrival_time_ns_1 - vpol_arrival_time_ns_2
            t_vpol_1subtract3 = vpol_arrival_time_ns_1 - vpol_arrival_time_ns_3
            t_vpol_2subtract3 = vpol_arrival_time_ns_2 - vpol_arrival_time_ns_3

            ouput_hpol = numpy.array([t_hpol_0subtract1,t_hpol_0subtract2,t_hpol_0subtract3,t_hpol_1subtract2,t_hpol_1subtract3,t_hpol_2subtract3])
            ouput_vpol = numpy.array([t_vpol_0subtract1,t_vpol_0subtract2,t_vpol_0subtract3,t_vpol_1subtract2,t_vpol_1subtract3,t_vpol_2subtract3])

            if return_indices == True:
                #Should double check when using these via rolling signals.
                #Calculate indices in corr for each direction.
                center = len(self.times_resampled)

                delay_indices_hpol_0subtract1 = numpy.rint((t_hpol_0subtract1/self.dt_resampled + center)).astype(int)
                delay_indices_hpol_0subtract2 = numpy.rint((t_hpol_0subtract2/self.dt_resampled + center)).astype(int)
                delay_indices_hpol_0subtract3 = numpy.rint((t_hpol_0subtract3/self.dt_resampled + center)).astype(int)
                delay_indices_hpol_1subtract2 = numpy.rint((t_hpol_1subtract2/self.dt_resampled + center)).astype(int)
                delay_indices_hpol_1subtract3 = numpy.rint((t_hpol_1subtract3/self.dt_resampled + center)).astype(int)
                delay_indices_hpol_2subtract3 = numpy.rint((t_hpol_2subtract3/self.dt_resampled + center)).astype(int)

                delay_indices_vpol_0subtract1 = numpy.rint((t_vpol_0subtract1/self.dt_resampled + center)).astype(int)
                delay_indices_vpol_0subtract2 = numpy.rint((t_vpol_0subtract2/self.dt_resampled + center)).astype(int)
                delay_indices_vpol_0subtract3 = numpy.rint((t_vpol_0subtract3/self.dt_resampled + center)).astype(int)
                delay_indices_vpol_1subtract2 = numpy.rint((t_vpol_1subtract2/self.dt_resampled + center)).astype(int)
                delay_indices_vpol_1subtract3 = numpy.rint((t_vpol_1subtract3/self.dt_resampled + center)).astype(int)
                delay_indices_vpol_2subtract3 = numpy.rint((t_vpol_2subtract3/self.dt_resampled + center)).astype(int)

                ouput2_hpol = numpy.array([delay_indices_hpol_0subtract1,delay_indices_hpol_0subtract2,delay_indices_hpol_0subtract3,delay_indices_hpol_1subtract2,delay_indices_hpol_1subtract3,delay_indices_hpol_2subtract3])
                ouput2_vpol = numpy.array([delay_indices_vpol_0subtract1,delay_indices_vpol_0subtract2,delay_indices_vpol_0subtract3,delay_indices_vpol_1subtract2,delay_indices_vpol_1subtract3,delay_indices_vpol_2subtract3])

                return (ouput_hpol,ouput_vpol), (ouput2_hpol,ouput2_vpol) 
            else:
                return (ouput_hpol,ouput_vpol) 

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def generateTimeIndicesPlaneWave(self):
        '''
        This is a deprecated function that assumes the source is sufficiently distant that a plane wave solution applies.
        '''
        try:
            #Prepare grids of thetas and phis

            thetas  = numpy.tile(self.thetas_rad,(self.n_phi,1)).T  #Each row is a different theta (zenith)
            phis    = numpy.tile(self.phis_rad,(self.n_theta,1))    #Each column is a different phi (azimuth)

            #Source direction is the direction FROM BEACON ANTENNA 0 you look to see the source.
            #Direction is the unit vector from antenna 0 pointing to that source. 
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
        of the array.  These may be slightly sketchy for near field things due to source direction not being
        a single number across the array. 
        '''
        try:
            # self.mesh_azimuth_rad
            # self.mesh_elevation_rad
            # self.mesh_zenith_rad

            #Source direction is the direction FROM BEACON ANTENNA 0 you look to see the source.
            #Position is a vector to a distant source

            #Source direction is the direction FROM BEACON you look to see the source.
            signal_source_direction        = numpy.zeros((self.mesh_azimuth_rad.shape[0], self.mesh_azimuth_rad.shape[1], 3))
            signal_source_direction[:,:,0] = numpy.multiply( numpy.cos(self.mesh_azimuth_rad) , numpy.sin(self.mesh_zenith_rad) )
            signal_source_direction[:,:,1] = numpy.multiply( numpy.sin(self.mesh_azimuth_rad) , numpy.sin(self.mesh_zenith_rad) )
            signal_source_direction[:,:,2] = numpy.cos(self.mesh_zenith_rad)
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

    def generateExpectedStokesParameters(self, antenna_model='uniform', included_antennas=numpy.array([0,1,2,3]), debug=False):
        '''
        Given the local magnetic field vector, this will estimate the percieved Stokes parameters of a purely
        geomagnetic signal as a function of azimuth and elevation. 
        antenna_model : str
            This sets the gain pattern used when comparing the relative powers of the vpol and hpol antennas.  The 
            supported options are 'uniform' and 'dipole', any other input will be interpreted as uniform.  If uniform
            is used then the signals powers will be calculated and compared agnostic of source direction, but if it
            is dipole then the this will attempt to account for the decrease in power due to a simple dipole beam 
            pattern.
        '''
        if antenna_model == 'dipole':
            source_vector = 0#In the same basis as the antennas, assuming source directions and distance relative to antenna 0.


        try:
            signal_source_position_hpol        = numpy.zeros((self.mesh_azimuth_rad.shape[0], self.mesh_azimuth_rad.shape[1], 3))
            signal_source_position_hpol[:,:,0] = self.map_source_distance_m * numpy.multiply( numpy.cos(self.mesh_azimuth_rad) , numpy.sin(self.mesh_zenith_rad) ) + self.A0_hpol[0] #Shifting points to be centered around antenna 0
            signal_source_position_hpol[:,:,1] = self.map_source_distance_m * numpy.multiply( numpy.sin(self.mesh_azimuth_rad) , numpy.sin(self.mesh_zenith_rad) ) + self.A0_hpol[1] #Shifting points to be centered around antenna 0
            signal_source_position_hpol[:,:,2] = self.map_source_distance_m * numpy.cos(self.mesh_zenith_rad) + self.A0_hpol[2] #Shifting points to be centered around antenna 0

            signal_source_position_vpol        = numpy.zeros((self.mesh_azimuth_rad.shape[0], self.mesh_azimuth_rad.shape[1], 3))
            signal_source_position_vpol[:,:,0] = self.map_source_distance_m * numpy.multiply( numpy.cos(self.mesh_azimuth_rad) , numpy.sin(self.mesh_zenith_rad) ) + self.A0_vpol[0] #Shifting points to be centered around antenna 0
            signal_source_position_vpol[:,:,1] = self.map_source_distance_m * numpy.multiply( numpy.sin(self.mesh_azimuth_rad) , numpy.sin(self.mesh_zenith_rad) ) + self.A0_vpol[1] #Shifting points to be centered around antenna 0
            signal_source_position_vpol[:,:,2] = self.map_source_distance_m * numpy.cos(self.mesh_zenith_rad) + self.A0_vpol[2] #Shifting points to be centered around antenna 0

            if debug:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.scatter(signal_source_position_hpol[:,:,0],signal_source_position_hpol[:,:,1],signal_source_position_hpol[:,:,2],marker=',',alpha=0.3)
                ax.scatter(self.A0_hpol[0],self.A0_hpol[1],self.A0_hpol[2],label='A0',c='tab:blue')
                ax.scatter(self.A1_hpol[0],self.A1_hpol[1],self.A1_hpol[2],label='A1',c='tab:orange')
                ax.scatter(self.A2_hpol[0],self.A2_hpol[1],self.A2_hpol[2],label='A2',c='tab:green')
                ax.scatter(self.A3_hpol[0],self.A3_hpol[1],self.A3_hpol[2],label='A3',c='tab:red')
                ax.set_xlabel('E (m)')
                ax.set_ylabel('N (m)')
                ax.set_zlabel('Relative Elevation (m)')
                plt.legend()

            #Calculate the expected readout time for each antenna (including cable delays)
            hpol_beam_angle_0 = None#Here I should calculate the angle of arrival at the antenna in the beam zenith such that I can calculate the beam multiplicative factor.#self.cable_delays[0] + (numpy.sqrt((signal_source_position_hpol[:,:,0] - self.A0_hpol[0])**2 + (signal_source_position_hpol[:,:,1] - self.A0_hpol[1])**2 + (signal_source_position_hpol[:,:,2] - self.A0_hpol[2])**2 )/self.c)*1.0e9 #ns
            hpol_beam_angle_1 = None#Here I should calculate the angle of arrival at the antenna in the beam zenith such that I can calculate the beam multiplicative factor.#self.cable_delays[2] + (numpy.sqrt((signal_source_position_hpol[:,:,0] - self.A1_hpol[0])**2 + (signal_source_position_hpol[:,:,1] - self.A1_hpol[1])**2 + (signal_source_position_hpol[:,:,2] - self.A1_hpol[2])**2 )/self.c)*1.0e9 #ns
            hpol_beam_angle_2 = None#Here I should calculate the angle of arrival at the antenna in the beam zenith such that I can calculate the beam multiplicative factor.#self.cable_delays[4] + (numpy.sqrt((signal_source_position_hpol[:,:,0] - self.A2_hpol[0])**2 + (signal_source_position_hpol[:,:,1] - self.A2_hpol[1])**2 + (signal_source_position_hpol[:,:,2] - self.A2_hpol[2])**2 )/self.c)*1.0e9 #ns
            hpol_beam_angle_3 = None#Here I should calculate the angle of arrival at the antenna in the beam zenith such that I can calculate the beam multiplicative factor.#self.cable_delays[6] + (numpy.sqrt((signal_source_position_hpol[:,:,0] - self.A3_hpol[0])**2 + (signal_source_position_hpol[:,:,1] - self.A3_hpol[1])**2 + (signal_source_position_hpol[:,:,2] - self.A3_hpol[2])**2 )/self.c)*1.0e9 #ns

            vpol_beam_angle_0 = None#Here I should calculate the angle of arrival at the antenna in the beam zenith such that I can calculate the beam multiplicative factor.#self.cable_delays[1] + (numpy.sqrt((signal_source_position_vpol[:,:,0] - self.A0_vpol[0])**2 + (signal_source_position_vpol[:,:,1] - self.A0_vpol[1])**2 + (signal_source_position_vpol[:,:,2] - self.A0_vpol[2])**2 )/self.c)*1.0e9 #ns
            vpol_beam_angle_1 = None#Here I should calculate the angle of arrival at the antenna in the beam zenith such that I can calculate the beam multiplicative factor.#self.cable_delays[3] + (numpy.sqrt((signal_source_position_vpol[:,:,0] - self.A1_vpol[0])**2 + (signal_source_position_vpol[:,:,1] - self.A1_vpol[1])**2 + (signal_source_position_vpol[:,:,2] - self.A1_vpol[2])**2 )/self.c)*1.0e9 #ns
            vpol_beam_angle_2 = None#Here I should calculate the angle of arrival at the antenna in the beam zenith such that I can calculate the beam multiplicative factor.#self.cable_delays[5] + (numpy.sqrt((signal_source_position_vpol[:,:,0] - self.A2_vpol[0])**2 + (signal_source_position_vpol[:,:,1] - self.A2_vpol[1])**2 + (signal_source_position_vpol[:,:,2] - self.A2_vpol[2])**2 )/self.c)*1.0e9 #ns
            vpol_beam_angle_3 = None#Here I should calculate the angle of arrival at the antenna in the beam zenith such that I can calculate the beam multiplicative factor.#self.cable_delays[7] + (numpy.sqrt((signal_source_position_vpol[:,:,0] - self.A3_vpol[0])**2 + (signal_source_position_vpol[:,:,1] - self.A3_vpol[1])**2 + (signal_source_position_vpol[:,:,2] - self.A3_vpol[2])**2 )/self.c)*1.0e9 #ns



        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)



    def calculateStokesParameters(self, eventid):
        '''
        This will calculate the Stokes parameters for a given waveform.  This does NOT account for beam pattern or
        arrival direction.  The "expected" should be calculated in accordance with this such that the match when they
        should.

        Currently has no time windowing, so this is prone to being offset by the noise baseline. 

        Parameters
        ----------
        eventid : int
            The entry number you wish to have calculate the Stokes parameters for.
        included_antennas : numpy.array
            An array of the antennas you want the stokes parameters calculated for.  This uses the lumped antenna
            labelling, where antenna 0 corresponds to channels 0 and 1, antenna 0 to 2 and 3, etc.  This is because
            this calculation definitionally needs both polarizations within a given antenna to calculate the stokes
            parameters.

        References
        ----------
        https://inspirehep.net/files/84e7b61412acf86d5fe2964d55ac1e5e Section 5.2
        https://en.wikipedia.org/wiki/Stokes_parameters
        https://github.com/nichol77/libRootFftwWrapper/blob/master/src/FFTtools.cxx#L2956
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
        https://en.wikipedia.org/wiki/Analytic_signal
        https://inspirehep.net/files/84e7b61412acf86d5fe2964d55ac1e5e



        '''
        included_antennas_channels = numpy.concatenate([[2*i,2*i+1] for i in included_antennas])

        waveforms = self.wf(eventid, included_antennas_channels ,div_std=False,hilbert=False,apply_filter=self.apply_filter,tukey=self.apply_tukey, sine_subtract=self.apply_sine_subtract)
        waveform_times = self.t()

        stokes_parameters = numpy.zeros((len(included_antennas),4))
        freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)

        for antenna_index, antenna in enumerate(included_antennas):
            wf_h = waveforms[antenna_index//2]
            hilbert_h = scipy.signal.hilbert(wf_h)
            wf_v = waveforms[antenna_index//2 + 1]
            hilbert_v = scipy.signal.hilbert(wf_v)

            complex_uv_term = numpy.multiply(wf_h,hilbert_v)

            stokes_I = numpy.sum(numpy.multiply(wf_h,hilbert_h) + numpy.multiply(wf_v,hilbert_v))
            stokes_Q = numpy.sum(numpy.multiply(wf_h,hilbert_h) + numpy.multiply(wf_v,hilbert_v))
            stokes_U = numpy.sum(2.0*numpy.real(complex_uv_term))
            stokes_V = numpy.sum(-2.0*numpy.imag(complex_uv_term))

            stokes_parameters[antenna_index] = numpy.array([stokes_I, stokes_Q, stokes_U, stokes_V])

        return stokes_parameters






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
            n_points = max(100,self.n_phi) #n_phi here is just to set an arbitrary numbers of points on the curve, and has no geometric significance

            if zenith_deg == 0.0:
                thetas      = numpy.ones(n_points) * numpy.rad2deg(numpy.arccos(norm[2]/numpy.linalg.norm(norm))) 
                phis        = numpy.ones(n_points) * numpy.rad2deg(numpy.arctan2(norm[1],norm[0]))
                phis -= azimuth_offset_deg
                phis[phis < -180.0] += 360.0
                phis[phis > 180.0] -= 360.0

                #Rotations of az don't matter because the same value.
                return [phis, thetas]
            elif zenith_deg == 180.0:
                thetas      = numpy.ones(n_points) * numpy.rad2deg(numpy.arccos(-norm[2]/numpy.linalg.norm(norm)))
                phis        = numpy.ones(n_points) * numpy.rad2deg(numpy.arctan2(-norm[1],-norm[0]))
                phis -= azimuth_offset_deg
                phis[phis < -180.0] += 360.0
                phis[phis > 180.0] -= 360.0
                #Rotations of az don't matter because the same value.
                return [phis, thetas]
            else:
                dtheta_rad = numpy.deg2rad(360.0/n_points) #Make full circle, regardless of span given for theta.  

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
                elif mode == 'all':
                    print('Using averaged hpol and vpol phase positions for all mode.')
                    a0 = (self.A0_hpol + self.A0_vpol)/2
                    a1 = (self.A1_hpol + self.A1_vpol)/2
                    a2 = (self.A2_hpol + self.A2_vpol)/2
                    a3 = (self.A3_hpol + self.A3_vpol)/2

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

                output_az_degs = numpy.zeros(n_points)
                output_zenith_degs = numpy.zeros(n_points)
                for i in range(n_points):
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
                out_x = numpy.roll(output_az_degs,-numpy.argmin(output_az_degs)) #Rolled for sorting purposes.  
                out_y = numpy.roll(output_zenith_degs,-numpy.argmin(output_az_degs))
                azimuth_cut = numpy.logical_and(out_x >= min(self.range_phi_deg), out_x <= min(self.range_phi_deg))
                zenith_cut =  numpy.logical_and(out_y >= min(self.range_theta_deg), out_y <= min(self.range_theta_deg))
                out_cut = numpy.logical_and(azimuth_cut,zenith_cut)
                out = [out_x , out_y]

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


    def interactivePlotter(self, event, eventid=None, pol=None, hilbert=None, mollweide = False, center_dir='E', all_alignments=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0):
        '''
        This hopefully will make a plot when called by a double click in the map.
        '''
        if event.dblclick == True:
            try:
                try:
                    plt.close(self.popout_fig)
                except:
                    pass #Just maintaining only one popout at a time.
                event_ax = event.inaxes
                if pol == None:
                    pol = event_ax.get_title().split('-')[2]
                if eventid is None:
                    eventid = int(event_ax.get_title().split('-')[1])
                if hilbert is None:
                    if 'True' in event_ax.get_title().split('Hilbert')[1]:
                        hilbert = True
                    else:
                        hilbert = False

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
                    if shorten_signals == True:
                        waveforms = self.shortenSignals(waveforms,shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)
                    t_best_0subtract1 = self.t_hpol_0subtract1[theta_index,phi_index]
                    t_best_0subtract2 = self.t_hpol_0subtract2[theta_index,phi_index]
                    t_best_0subtract3 = self.t_hpol_0subtract3[theta_index,phi_index]
                    t_best_1subtract2 = self.t_hpol_1subtract2[theta_index,phi_index]
                    t_best_1subtract3 = self.t_hpol_1subtract3[theta_index,phi_index]
                    t_best_2subtract3 = self.t_hpol_2subtract3[theta_index,phi_index]
                elif pol == 'vpol':
                    channels = numpy.array([1,3,5,7])
                    waveforms = self.wf(eventid, channels, div_std=False, hilbert=hilbert, apply_filter=self.apply_filter, tukey=self.apply_tukey, sine_subtract=self.apply_sine_subtract)
                    if shorten_signals == True:
                        waveforms = self.shortenSignals(waveforms,shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)
                    t_best_0subtract1 = self.t_vpol_0subtract1[theta_index,phi_index]
                    t_best_0subtract2 = self.t_vpol_0subtract2[theta_index,phi_index]
                    t_best_0subtract3 = self.t_vpol_0subtract3[theta_index,phi_index]
                    t_best_1subtract2 = self.t_vpol_1subtract2[theta_index,phi_index]
                    t_best_1subtract3 = self.t_vpol_1subtract3[theta_index,phi_index]
                    t_best_2subtract3 = self.t_vpol_2subtract3[theta_index,phi_index]
                elif pol == 'all':
                    channels = numpy.array([0,1,2,3,4,5,6,7])
                    waveforms = self.wf(eventid, channels, div_std=False, hilbert=hilbert, apply_filter=self.apply_filter, tukey=self.apply_tukey, sine_subtract=self.apply_sine_subtract)
                    if shorten_signals == True:
                        waveforms = self.shortenSignals(waveforms,shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)
                    
                    t_best_0subtract1_h = self.t_hpol_0subtract1[theta_index,phi_index]
                    t_best_0subtract2_h = self.t_hpol_0subtract2[theta_index,phi_index]
                    t_best_0subtract3_h = self.t_hpol_0subtract3[theta_index,phi_index]
                    t_best_1subtract2_h = self.t_hpol_1subtract2[theta_index,phi_index]
                    t_best_1subtract3_h = self.t_hpol_1subtract3[theta_index,phi_index]
                    t_best_2subtract3_h = self.t_hpol_2subtract3[theta_index,phi_index]

                    t_best_0subtract1_v = self.t_vpol_0subtract1[theta_index,phi_index]
                    t_best_0subtract2_v = self.t_vpol_0subtract2[theta_index,phi_index]
                    t_best_0subtract3_v = self.t_vpol_0subtract3[theta_index,phi_index]
                    t_best_1subtract2_v = self.t_vpol_1subtract2[theta_index,phi_index]
                    t_best_1subtract3_v = self.t_vpol_1subtract3[theta_index,phi_index]
                    t_best_2subtract3_v = self.t_vpol_2subtract3[theta_index,phi_index]

                # #Determine how many indices to roll each waveform.
                # roll0 = 0
                # roll1 = int(numpy.rint(t_best_0subtract1/self.dt_resampled))
                # roll2 = int(numpy.rint(t_best_0subtract2/self.dt_resampled))
                # roll3 = int(numpy.rint(t_best_0subtract3/self.dt_resampled))

                
                if pol in ['hpol', 'vpol']:
                    self.popout_fig = plt.figure()
                    self.popout_ax = self.popout_fig.gca()
                    plt.suptitle('ENU %s\nAzimuth = %0.3f, Zenith = %0.3f'%(event_ax.get_title().replace('-',' ').title(), event.xdata,event.ydata))
                    if all_alignments == False:
                        #Align signals to antenna 0

                        plt.subplot(2,1,1)
                        
                        plt.plot(self.times_resampled, waveforms[0],label='Ch%i'%channels[0],c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[0]])
                        plt.plot(self.times_resampled, waveforms[1],label='Ch%i'%(channels[1]),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[1]])
                        plt.plot(self.times_resampled, waveforms[2],label='Ch%i'%(channels[2]),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[2]])
                        plt.plot(self.times_resampled, waveforms[3],label='Ch%i'%(channels[3]),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[3]])

                        plt.ylabel('adu')
                        plt.xlabel('Time (ns)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.legend()

                        plt.subplot(2,1,2)
                        
                        plt.plot(self.times_resampled, waveforms[0]/max(waveforms[0]),label='Ch%i'%channels[0],c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[0]])
                        plt.plot(self.times_resampled + t_best_0subtract1, waveforms[1]/max(waveforms[1]),label='Ch%i, shifted %0.2f ns'%(channels[1], t_best_0subtract1),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[1]])
                        plt.plot(self.times_resampled + t_best_0subtract2, waveforms[2]/max(waveforms[2]),label='Ch%i, shifted %0.2f ns'%(channels[2], t_best_0subtract2),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[2]])
                        plt.plot(self.times_resampled + t_best_0subtract3, waveforms[3]/max(waveforms[3]),label='Ch%i, shifted %0.2f ns'%(channels[3], t_best_0subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[3]])

                        plt.ylabel('Normalized adu')
                        plt.xlabel('Time (ns)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.legend()

                    else:
                        #Each waveform aligned to each antenna by that antennas relevant time delays. 
                        plt.subplot(2,2,1)
                        #Antenna 0
                        
                        plt.plot(self.times_resampled                    , waveforms[0]/max(waveforms[0]),label='Ch%i, shifted %0.2f ns'%(channels[0], 0),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[0]])
                        plt.plot(self.times_resampled + t_best_0subtract1, waveforms[1]/max(waveforms[1]),label='Ch%i, shifted %0.2f ns'%(channels[1], t_best_0subtract1),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[1]])
                        plt.plot(self.times_resampled + t_best_0subtract2, waveforms[2]/max(waveforms[2]),label='Ch%i, shifted %0.2f ns'%(channels[2], t_best_0subtract2),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[2]])
                        plt.plot(self.times_resampled + t_best_0subtract3, waveforms[3]/max(waveforms[3]),label='Ch%i, shifted %0.2f ns'%(channels[3], t_best_0subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[3]])

                        plt.ylabel('Normalized adu')
                        plt.xlabel('Time (ns)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.legend()

                        plt.subplot(2,2,2)
                        #Antenna 1
                        
                        plt.plot(self.times_resampled - t_best_0subtract1, waveforms[0]/max(waveforms[0]),label='Ch%i, shifted %0.2f ns'%(channels[0], - t_best_0subtract1),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[0]])
                        plt.plot(self.times_resampled                    , waveforms[1]/max(waveforms[1]),label='Ch%i, shifted %0.2f ns'%(channels[1], 0),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[1]])
                        plt.plot(self.times_resampled + t_best_1subtract2, waveforms[2]/max(waveforms[2]),label='Ch%i, shifted %0.2f ns'%(channels[2], t_best_0subtract2),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[2]])
                        plt.plot(self.times_resampled + t_best_1subtract3, waveforms[3]/max(waveforms[3]),label='Ch%i, shifted %0.2f ns'%(channels[3], t_best_0subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[3]])

                        plt.ylabel('Normalized adu')
                        plt.xlabel('Time (ns)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.legend()

                        plt.subplot(2,2,3)
                        #Antenna 2
                        
                        plt.plot(self.times_resampled - t_best_0subtract2, waveforms[0]/max(waveforms[0]),label='Ch%i, shifted %0.2f ns'%(channels[0], - t_best_0subtract2),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[0]])
                        plt.plot(self.times_resampled - t_best_1subtract2, waveforms[1]/max(waveforms[1]),label='Ch%i, shifted %0.2f ns'%(channels[1], - t_best_1subtract2),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[1]])
                        plt.plot(self.times_resampled                    , waveforms[2]/max(waveforms[2]),label='Ch%i, shifted %0.2f ns'%(channels[2], 0),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[2]])
                        plt.plot(self.times_resampled + t_best_2subtract3, waveforms[3]/max(waveforms[3]),label='Ch%i, shifted %0.2f ns'%(channels[3], t_best_2subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[3]])

                        plt.ylabel('Normalized adu')
                        plt.xlabel('Time (ns)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.legend()

                        plt.subplot(2,2,4)
                        #Antenna 3
                        
                        plt.plot(self.times_resampled - t_best_0subtract3, waveforms[0]/max(waveforms[0]),label='Ch%i, shifted %0.2f ns'%(channels[0], - t_best_0subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[0]])
                        plt.plot(self.times_resampled - t_best_1subtract3, waveforms[1]/max(waveforms[1]),label='Ch%i, shifted %0.2f ns'%(channels[1], - t_best_1subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[1]])
                        plt.plot(self.times_resampled - t_best_2subtract3, waveforms[2]/max(waveforms[2]),label='Ch%i, shifted %0.2f ns'%(channels[2], - t_best_2subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[2]])
                        plt.plot(self.times_resampled                    , waveforms[3]/max(waveforms[3]),label='Ch%i, shifted %0.2f ns'%(channels[3], 0),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[3]])

                        plt.ylabel('Normalized adu')
                        plt.xlabel('Time (ns)')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.legend()
                elif pol == 'all':
                    self.popout_fig = [None]*2
                    self.popout_ax = [None]*2

                    for mode_index, mode in enumerate(['hpol', 'vpol']):
                        self.popout_fig[mode_index] = plt.figure()
                        self.popout_ax[mode_index] = self.popout_fig[mode_index].gca()
                        plt.suptitle('ENU %s (%s)\nAzimuth = %0.3f, Zenith = %0.3f'%(event_ax.get_title().replace('-',' ').title(),mode.title(), event.xdata,event.ydata))

                        if mode == 'hpol':
                            t_best_0subtract1 = t_best_0subtract1_h
                            t_best_0subtract2 = t_best_0subtract2_h
                            t_best_0subtract3 = t_best_0subtract3_h
                            t_best_1subtract2 = t_best_1subtract2_h
                            t_best_1subtract3 = t_best_1subtract3_h
                            t_best_2subtract3 = t_best_2subtract3_h
                        elif mode == 'vpol':
                            t_best_0subtract1 = t_best_0subtract1_v
                            t_best_0subtract2 = t_best_0subtract2_v
                            t_best_0subtract3 = t_best_0subtract3_v
                            t_best_1subtract2 = t_best_1subtract2_v
                            t_best_1subtract3 = t_best_1subtract3_v
                            t_best_2subtract3 = t_best_2subtract3_v

                        if all_alignments == False:
                            #Align signals to antenna 0

                            plt.subplot(2,1,1)
                            
                            plt.plot(self.times_resampled, waveforms[0*2 + int(mode == 'vpol')],label='Ch%i'%channels[0*2 + int(mode == 'vpol')],c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[0*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled, waveforms[1*2 + int(mode == 'vpol')],label='Ch%i'%(channels[1*2 + int(mode == 'vpol')]),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[1*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled, waveforms[2*2 + int(mode == 'vpol')],label='Ch%i'%(channels[2*2 + int(mode == 'vpol')]),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[2*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled, waveforms[3*2 + int(mode == 'vpol')],label='Ch%i'%(channels[3*2 + int(mode == 'vpol')]),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[3*2 + int(mode == 'vpol')]])

                            plt.ylabel('adu')
                            plt.xlabel('Time (ns)')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                            plt.legend()

                            plt.subplot(2,1,2)
                            
                            plt.plot(self.times_resampled, waveforms[0*2 + int(mode == 'vpol')]/max(waveforms[0*2 + int(mode == 'vpol')]),label='Ch%i'%channels[0*2 + int(mode == 'vpol')],c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[0*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled + t_best_0subtract1, waveforms[1*2 + int(mode == 'vpol')]/max(waveforms[1*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[1*2 + int(mode == 'vpol')], t_best_0subtract1),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[1*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled + t_best_0subtract2, waveforms[2*2 + int(mode == 'vpol')]/max(waveforms[2*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[2*2 + int(mode == 'vpol')], t_best_0subtract2),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[2*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled + t_best_0subtract3, waveforms[3*2 + int(mode == 'vpol')]/max(waveforms[3*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[3*2 + int(mode == 'vpol')], t_best_0subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[3*2 + int(mode == 'vpol')]])

                            plt.ylabel('Normalized adu')
                            plt.xlabel('Time (ns)')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                            plt.legend()

                        else:
                            #Each waveform aligned to each antenna by that antennas relevant time delays. 
                            plt.subplot(2,2,1)
                            #Antenna 0
                            
                            plt.plot(self.times_resampled                    , waveforms[0*2 + int(mode == 'vpol')]/max(waveforms[0*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[0*2 + int(mode == 'vpol')], 0),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[0*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled + t_best_0subtract1, waveforms[1*2 + int(mode == 'vpol')]/max(waveforms[1*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[1*2 + int(mode == 'vpol')], t_best_0subtract1),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[1*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled + t_best_0subtract2, waveforms[2*2 + int(mode == 'vpol')]/max(waveforms[2*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[2*2 + int(mode == 'vpol')], t_best_0subtract2),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[2*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled + t_best_0subtract3, waveforms[3*2 + int(mode == 'vpol')]/max(waveforms[3*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[3*2 + int(mode == 'vpol')], t_best_0subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[3*2 + int(mode == 'vpol')]])

                            plt.ylabel('Normalized adu')
                            plt.xlabel('Time (ns)')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                            plt.legend()

                            plt.subplot(2,2,2)
                            #Antenna 1
                            
                            plt.plot(self.times_resampled - t_best_0subtract1, waveforms[0*2 + int(mode == 'vpol')]/max(waveforms[0*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[1*2 + int(mode == 'vpol')], - t_best_0subtract1))
                            plt.plot(self.times_resampled                    , waveforms[1*2 + int(mode == 'vpol')]/max(waveforms[1*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[1*2 + int(mode == 'vpol')], 0))
                            plt.plot(self.times_resampled + t_best_1subtract2, waveforms[2*2 + int(mode == 'vpol')]/max(waveforms[2*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[2*2 + int(mode == 'vpol')], t_best_0subtract2))
                            plt.plot(self.times_resampled + t_best_1subtract3, waveforms[3*2 + int(mode == 'vpol')]/max(waveforms[3*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[3*2 + int(mode == 'vpol')], t_best_0subtract3))

                            plt.ylabel('Normalized adu')
                            plt.xlabel('Time (ns)')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                            plt.legend()

                            plt.subplot(2,2,3)
                            #Antenna 2
                            
                            plt.plot(self.times_resampled - t_best_0subtract2, waveforms[0*2 + int(mode == 'vpol')]/max(waveforms[0*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[0*2 + int(mode == 'vpol')], - t_best_0subtract2),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[0*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled - t_best_1subtract2, waveforms[1*2 + int(mode == 'vpol')]/max(waveforms[1*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[1*2 + int(mode == 'vpol')], - t_best_1subtract2),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[1*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled                    , waveforms[2*2 + int(mode == 'vpol')]/max(waveforms[2*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[2*2 + int(mode == 'vpol')], 0),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[2*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled + t_best_2subtract3, waveforms[3*2 + int(mode == 'vpol')]/max(waveforms[3*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[3*2 + int(mode == 'vpol')], t_best_2subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[3*2 + int(mode == 'vpol')]])

                            plt.ylabel('Normalized adu')
                            plt.xlabel('Time (ns)')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                            plt.legend()

                            plt.subplot(2,2,4)
                            #Antenna 3
                            
                            plt.plot(self.times_resampled - t_best_0subtract3, waveforms[0*2 + int(mode == 'vpol')]/max(waveforms[0*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[0*2 + int(mode == 'vpol')], - t_best_0subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[0*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled - t_best_1subtract3, waveforms[1*2 + int(mode == 'vpol')]/max(waveforms[1*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[1*2 + int(mode == 'vpol')], - t_best_1subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[1*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled - t_best_2subtract3, waveforms[2*2 + int(mode == 'vpol')]/max(waveforms[2*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[2*2 + int(mode == 'vpol')], - t_best_2subtract3),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[2*2 + int(mode == 'vpol')]])
                            plt.plot(self.times_resampled                    , waveforms[3*2 + int(mode == 'vpol')]/max(waveforms[3*2 + int(mode == 'vpol')]),label='Ch%i, shifted %0.2f ns'%(channels[3*2 + int(mode == 'vpol')], 0),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][channels[3*2 + int(mode == 'vpol')]])

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
    
    def generateThetaCutMask(self, pol, shape=None, zenith_cut_ENU=None, zenith_cut_array_plane=None):
        '''
        zenith_cut_ENU : list of 2 float values
            Given in degrees, angles within these two values are considered.  If None is given for the first then it is
            assumed to be 0 (overhead), if None is given for the latter then it is assumed to be 180 (straight down).
        zenith_cut_array_plane : list of 2 float values
            Given in degrees, angles within these two values are considered.  If None is given for the first then it is
            assumed to be 0 (overhead), if None is given for the latter then it is assumed to be 180 (straight down).  This is
            polarization dependant because it depends on the calibration of the antennas positions.  So if pol=None then this will
            be ignored.
        '''
        try:
            if shape is None:
                shape = numpy.shape(self.mesh_azimuth_deg)

            theta_cut = numpy.ones(shape,dtype=bool)

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
                        elif pol == 'vpol':
                            theta_cut = numpy.logical_and(theta_cut,numpy.logical_and(90.0 - self.vpol_dot_angle_from_plane_deg >= min(zenith_cut_array_plane), 90.0 - self.vpol_dot_angle_from_plane_deg <= max(zenith_cut_array_plane)))
                        else:
                            theta_cut = numpy.logical_and(theta_cut,numpy.logical_and(90.0 - self.physical_dot_angle_from_plane_deg >= min(zenith_cut_array_plane), 90.0 - self.physical_dot_angle_from_plane_deg <= max(zenith_cut_array_plane)))
            return theta_cut
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def mapMax(self, map_values, max_method=0, verbose=False, zenith_cut_ENU=None, zenith_cut_array_plane=None, pol=None, return_peak_to_sidelobe=False, theta_cut=None):
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
        linear_max_index : int
            This index in the flattened directional mesh.
        theta_best : float 
            The corresponding values for this parameters for linear_max_index.
        phi_best : float   
            The corresponding values for this parameters for linear_max_index.
        t_best_0subtract1 : float
            The corresponding values for this parameters for linear_max_index.
        t_best_0subtract2 : float
            The corresponding values for this parameters for linear_max_index.
        t_best_0subtract3 : float
            The corresponding values for this parameters for linear_max_index.
        t_best_1subtract2 : float
            The corresponding values for this parameters for linear_max_index.
        t_best_1subtract3 : float
            The corresponding values for this parameters for linear_max_index.
        t_best_2subtract3 : float
            The corresponding values for this parameters for linear_max_index.


        '''
        try:
            if theta_cut is None:
                theta_cut = self.generateThetaCutMask(pol, shape=numpy.shape(map_values),zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane)

            masked_map_values = numpy.ma.array(map_values,mask=~theta_cut) #This way the values not in the range are not included in calculations but the dimensions of the map stay the same.#masked_map_values = numpy.ma.array(map_values.copy(),mask=~theta_cut) #This way the values not in the range are not included in calculations but the dimensions of the map stay the same.

            # if max_method == 0:
            #     row_index, column_index = numpy.unravel_index(masked_map_values.argmax(),numpy.shape(masked_map_values))

            # elif max_method == 1:
            #     #Calculates sum of each point plus surrounding four points to get max.
            #     rounded_corr_values = (masked_map_values + numpy.roll(masked_map_values,1,axis=0) + numpy.roll(masked_map_values,-1,axis=0) + numpy.roll(masked_map_values,1,axis=1) + numpy.roll(masked_map_values,-1,axis=1))/5.0
            #     row_index, column_index = numpy.unravel_index(rounded_corr_values.argmax(),numpy.shape(rounded_corr_values))

            linear_max_index = masked_map_values.argmax()

            if return_peak_to_sidelobe:
                blob_label, num_blobs = scipy.ndimage.label(masked_map_values > 0)
                #max_peak_mask = blob_label == blob_label[numpy.unravel_index(linear_max_index,numpy.shape(blob_label))] #Commented for slightly faster.
                _main_peak_masked_map_values = numpy.ma.array(map_values,mask=numpy.logical_or(~theta_cut , blob_label == blob_label[numpy.unravel_index(linear_max_index,numpy.shape(blob_label))])) #Ignore values in the main peak

                peak_to_sidelobe = masked_map_values.flat[linear_max_index] / _main_peak_masked_map_values.max()

                if False:
                    max_peak_mask = blob_label == blob_label[numpy.unravel_index(linear_max_index,numpy.shape(blob_label))]
                    #For testing and debugging
                    plt.figure()
                    ax = plt.gca()
                    im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, masked_map_values > 0, vmin=numpy.min(masked_map_values > 0), vmax=numpy.max(masked_map_values > 0),cmap=plt.cm.coolwarm)
                    plt.figure()
                    ax = plt.gca()
                    im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, blob_label, vmin=numpy.min(blob_label), vmax=numpy.max(blob_label),cmap=plt.cm.coolwarm)
                    plt.figure()
                    ax = plt.gca()
                    im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, max_peak_mask, vmin=numpy.min(max_peak_mask), vmax=numpy.max(max_peak_mask),cmap=plt.cm.coolwarm)
                    plt.figure()
                    ax = plt.gca()
                    im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, _main_peak_masked_map_values, vmin=numpy.min(_main_peak_masked_map_values), vmax=numpy.max(_main_peak_masked_map_values),cmap=plt.cm.coolwarm)
    
                    plt.figure()
                    ax = plt.subplot(3,1,1)
                    im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, masked_map_values, vmin=numpy.min(masked_map_values), vmax=numpy.max(masked_map_values),cmap=plt.cm.coolwarm)
                    ax = plt.subplot(3,1,2)
                    im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, _main_peak_masked_map_values, vmin=numpy.min(_main_peak_masked_map_values), vmax=numpy.max(_main_peak_masked_map_values),cmap=plt.cm.coolwarm)
                    ax = plt.subplot(3,1,3)
                    im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, blob_label, vmin=numpy.min(blob_label), vmax=numpy.max(blob_label),cmap=plt.cm.coolwarm)

                    plt.figure() #Left blank to ensure the above figure is untouched as this is a bodge

            theta_best  = self.mesh_zenith_deg.flat[linear_max_index]
            phi_best    = self.mesh_azimuth_deg.flat[linear_max_index]

            t_best_0subtract1 = self.t_hpol_0subtract1.flat[linear_max_index]
            t_best_0subtract2 = self.t_hpol_0subtract2.flat[linear_max_index]
            t_best_0subtract3 = self.t_hpol_0subtract3.flat[linear_max_index]
            t_best_1subtract2 = self.t_hpol_1subtract2.flat[linear_max_index]
            t_best_1subtract3 = self.t_hpol_1subtract3.flat[linear_max_index]
            t_best_2subtract3 = self.t_hpol_2subtract3.flat[linear_max_index]

            if verbose == True:
                print("From the correlation plot:")
                print("Best zenith angle:",theta_best)
                print("Best azimuth angle:",phi_best)
                print('Predicted time delays between A0 and A1:', t_best_0subtract1)
                print('Predicted time delays between A0 and A2:', t_best_0subtract2)
                print('Predicted time delays between A0 and A3:', t_best_0subtract3)
                print('Predicted time delays between A1 and A2:', t_best_1subtract2)
                print('Predicted time delays between A1 and A3:', t_best_1subtract3)
                print('Predicted time delays between A2 and A3:', t_best_2subtract3)

            if return_peak_to_sidelobe:
                return linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3, peak_to_sidelobe
            else:
                return linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

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

    def addCurveToMap(self, im, plane_xy, ax=None,  mollweide=False, *args, **kwargs):
        '''
        This will plot the curves to a map (ax.pcolormesh instance passed as im).

        plane_xy is a set of theta, phi coordinates given in degrees to plot on the map.
        y should be given in zenith angle as it will be converted to elevation internally. 

        The args and kwargs should be plotting things such as color and linestyle.  
        '''
        try:
            if ax is None:
                ax = plt.gca()
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

                ax.plot(plane_xy[0][left_cut], plane_xy[1][left_cut], *args, **kwargs)
                ax.plot(plane_xy[0][right_cut], plane_xy[1][right_cut], *args, **kwargs)
            else:
                if numpy.all([len(numpy.unique(plane_xy[0])) == 1,len(numpy.unique(plane_xy[1])) == 1]):
                    #import pdb; pdb.set_trace()
                    ax.scatter(plane_xy[0][0], plane_xy[1][0], *args, **kwargs)
                else:
                    ax.plot(plane_xy[0], plane_xy[1], *args, **kwargs)

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
        try:
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
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def addTimeDelayCurves(self, im, time_delay_dict, mode, ax, include_baselines=numpy.array([0,1,2,3,4,5]), mollweide=False, azimuth_offset_deg=0, *args, **kwargs):
        '''
        This solution attempts to determine the time delay curve using a contour on precalculated time delay values
        rather than doing clever vector math.  

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
                all_antennas = numpy.vstack((self.A0_hpol,self.A1_hpol,self.A2_hpol,self.A3_hpol))
                time_delays = time_delay_dict['hpol']
            elif mode == 'vpol':
                all_antennas = numpy.vstack((self.A0_vpol,self.A1_vpol,self.A2_vpol,self.A3_vpol))
                time_delays = time_delay_dict['vpol']

            pairs = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
            baseline_cm = plt.cm.get_cmap('tab10', 10)
            baseline_colors = baseline_cm(numpy.linspace(0, 1, 10))[0:6] #only want the first 6 colours in this list of 10 colors.

            for pair_key, pair_time_delay in time_delays.items():
                pair = numpy.array(pair_key.replace('[','').replace(']','').split(','),dtype=int)
                pair_index = numpy.where(numpy.sum(pair == pairs,axis=1) == 2)[0][0] #Used for consistent coloring.

                if numpy.isin(pair_index ,include_baselines ):
                    linestyle = '-'
                    linewidth = 4.0*self.min_elevation_linewidth
                else:
                    linestyle = '--'
                    linewidth = self.min_elevation_linewidth

                i = pair[0]
                j = pair[1]

                for time_delay in pair_time_delay:
                    #Attempting to use precalculate expected time delays per direction to derive theta here.
                    x = (180.0 + self.mesh_azimuth_deg - azimuth_offset_deg)%360.0 - 180.0

                    if azimuth_offset_deg == 0:
                        roll = 0
                    elif azimuth_offset_deg == 90:
                        roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))
                    elif azimuth_offset_deg == 180:
                        roll = len(self.phis_rad)//2
                    elif azimuth_offset_deg == -90:
                        roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))
                    else:
                        print("WARNING ONLY CARDINAL DIRECTIONS CURRENTLY SUPPORTED FOR ROLLING IN addTimeDelayCurves")
                    
                    if mollweide == True:
                        if mode == 'hpol':
                            if pair_index == 0:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_hpol_0subtract1,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 1:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_hpol_0subtract2,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 2:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_hpol_0subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 3:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_hpol_1subtract2,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 4:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_hpol_1subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 5:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_hpol_2subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                        else:
                            if pair_index == 0:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_vpol_0subtract1,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 1:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_vpol_0subtract2,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 2:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_vpol_0subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 3:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_vpol_1subtract2,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 4:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_vpol_1subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 5:
                                contours = ax.contour(self.mesh_azimuth_rad, self.mesh_elevation_rad, numpy.roll(self.t_vpol_2subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                    else:
                        if mode == 'hpol':
                            if pair_index == 0:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_hpol_0subtract1,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 1:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_hpol_0subtract2,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 2:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_hpol_0subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 3:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_hpol_1subtract2,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 4:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_hpol_1subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 5:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_hpol_2subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                        else:
                            if pair_index == 0:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_vpol_0subtract1,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 1:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_vpol_0subtract2,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 2:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_vpol_0subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 3:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_vpol_1subtract2,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 4:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_vpol_1subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])
                            elif pair_index == 5:
                                contours = ax.contour(self.mesh_azimuth_deg, self.mesh_elevation_deg, numpy.roll(self.t_vpol_2subtract3,roll,axis=1), levels=[time_delay], linewidths=[linewidth], colors=[baseline_colors[pair_index]],alpha=0.5,linestyles=[linestyle])

                    fmt = str(pair) + ':' + r'%0.2f'
                    ax.clabel(contours, contours.levels, inline=False, fontsize=10, fmt=fmt,manual=False) #manual helpful for presentation quality plots.
            return im, ax
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def addTimeDelayCurvesAttemptedVectorSolution(self, im, time_delay_dict, mode, include_baselines=numpy.array([0,1,2,3,4,5]), mollweide=False, azimuth_offset_deg=0, *args, **kwargs):
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
                all_antennas = numpy.vstack((self.A0_hpol,self.A1_hpol,self.A2_hpol,self.A3_hpol))
                time_delays = time_delay_dict['hpol']
            elif mode == 'vpol':
                all_antennas = numpy.vstack((self.A0_vpol,self.A1_vpol,self.A2_vpol,self.A3_vpol))
                time_delays = time_delay_dict['vpol']

            pairs = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
            baseline_cm = plt.cm.get_cmap('tab10', 10)
            baseline_colors = baseline_cm(numpy.linspace(0, 1, 10))[0:6] #only want the first 6 colours in this list of 10 colors.


            thetas  = numpy.tile(self.thetas_rad,(self.n_phi,1)).T  #Each row is a different theta (zenith)
            phis    = numpy.tile(self.phis_rad,(self.n_theta,1))    #Each column is a different phi (azimuth)

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

                for time_delay in pair_time_delay:
                    #Attempting to use precalculate expected time delays per direction to derive theta here.
                    if mode == 'hpol':
                        if baseline_index == 0:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_hpol_0subtract1 - time_delay))
                        elif baseline_index == 1:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_hpol_0subtract2 - time_delay))
                        elif baseline_index == 2:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_hpol_0subtract3 - time_delay))
                        elif baseline_index == 3:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_hpol_1subtract2 - time_delay))
                        elif baseline_index == 4:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_hpol_1subtract3 - time_delay))
                        elif baseline_index == 5:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_hpol_2subtract3 - time_delay))
                    else:
                        if baseline_index == 0:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_vpol_0subtract1 - time_delay))
                        elif baseline_index == 1:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_vpol_0subtract2 - time_delay))
                        elif baseline_index == 2:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_vpol_0subtract3 - time_delay))
                        elif baseline_index == 3:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_vpol_1subtract2 - time_delay))
                        elif baseline_index == 4:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_vpol_1subtract3 - time_delay))
                        elif baseline_index == 5:
                            flattened_min_index = numpy.argmin(numpy.abs(self.t_vpol_2subtract3 - time_delay))

                    theta_best  = thetas.flat[flattened_min_index] #radians
                    phi_best    = phis.flat[flattened_min_index] #radians
                    #These angles should be from antenna 0 in the given polarization.  
                    source_vector = self.map_source_distance_m * numpy.array([numpy.sin(theta_best)*numpy.cos(phi_best), numpy.sin(theta_best)*numpy.sin(phi_best), numpy.cos(theta_best)]) #vector from ant 0 to the "source"

                    v1 = all_antennas[i] - all_antennas[j]
                    v1_normalized = v1/numpy.linalg.norm(v1)
                    v2 = source_vector - all_antennas[j]
                    cone_angle_rad = numpy.arccos(numpy.dot(v1,v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2)))


                    #NEED TO THINK FURTHER ABOUT WHAT VECTOR I SHOULD BE ROTATING AROUND.  SURELY IT HASE TO BE DEFINED FROM THE ORIGIN FOR THE OUTPUT THETA AND PHI TO MAKE SENSE?

                    plane_xy = self.getPlaneZenithCurves(v1_normalized, mode, numpy.rad2deg(cone_angle_rad), azimuth_offset_deg=azimuth_offset_deg)
                    #Plot array plane 0 elevation curve.
                    im = self.addCurveToMap(im, plane_xy,  mollweide=mollweide, linewidth = 4.0*self.min_elevation_linewidth, color=baseline_colors[pair_index], alpha=0.5, label=pair_key, linestyle=linestyle)
            return im
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def addTimeDelayCurvesPlaneWave(self, im, time_delay_dict, mode, include_baselines=numpy.array([0,1,2,3,4,5]), mollweide=False, azimuth_offset_deg=0, *args, **kwargs):
        '''
        THIS IS DEPRECATED.  WILL PLOT TIME DELAY CURVES GENERATED ASSUMING SAME ARRIVAL ANGLE AT EACH ANTENNA.

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

    def shortenSignals(self, waveforms, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0,shorten_keep_leading=100):
        '''
        Given waveforms this will reduce time window to suround just the main signal pulses.

        waveforms should be a 2d array where each row is a waveform, with timestamp determined by self.dt_resampled.
        '''
        try:
            for wf_index, wf in enumerate(waveforms):
                trigger_index = numpy.where(wf/max(wf) > shorten_thresh)[0][0]
                weights = numpy.ones_like(wf)
                cut = numpy.arange(len(wf)) < trigger_index + int(shorten_delay/self.dt_resampled)
                slope = -1.0/(shorten_length/self.dt_resampled) #Go to zero by 100ns after initial dampening.
                weights[~cut] = numpy.max(numpy.vstack((slope * numpy.arange(sum(~cut)) + 1,numpy.zeros(sum(~cut)))),axis=0) #Max so no negative weights
                cut2 = numpy.arange(len(wf)) < trigger_index - int(shorten_keep_leading/self.dt_resampled)
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

    def map(self, eventid, pol, include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=None, plot_corr=False, hilbert=False, interactive=False, max_method=None, waveforms=None, verbose=True, mollweide=False, zenith_cut_ENU=None, zenith_cut_array_plane=None, center_dir='E', circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=False, circle_map_max=True):
        '''
        Makes the cross correlation make for the given event.  center_dir only specifies the center direction when
        plotting and does not modify the output array, which is ENU oriented.  Note that pol='all' may cause bugs. 

        Parameters
        ----------
        eventid : int
            The entry number you wish to plot the correlation map for.
        pol : str
            The polarization you wish to plot.  Options: 'hpol', 'vpol', 'both', or 'all'.
            'both' will essentially run the function twice, creating seperate 'hpol' and 'vpol' map
            values, while 'all' will create a single map using antennas from both polarizations.
            Note that 'all' may disable plot_corr.
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
        return_max_possible_map_value : bool
            If True, then an additional value will be returned that attempts to predict the maximum possible correlation
            value achievable based upon the cross correlations. 
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
                hpol_result = self.map(eventid,'hpol', plot_map=plot_map, plot_corr=plot_corr, hilbert=hilbert, mollweide=mollweide, center_dir=center_dir, zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,circle_zenith=circle_zenith,circle_az=circle_az,time_delay_dict=time_delay_dict,include_baselines=include_baselines,interactive=interactive,window_title=window_title, plot_peak_to_sidelobe=plot_peak_to_sidelobe)
                vpol_result = self.map(eventid,'vpol', plot_map=plot_map, plot_corr=plot_corr, hilbert=hilbert, mollweide=mollweide, center_dir=center_dir, zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,circle_zenith=circle_zenith,circle_az=circle_az,time_delay_dict=time_delay_dict,include_baselines=include_baselines,interactive=interactive,window_title=window_title, plot_peak_to_sidelobe=plot_peak_to_sidelobe)
                return hpol_result, vpol_result
                # elif pol == 'all':
                #     hpol_result = self.map(eventid,'hpol', plot_map=False, plot_corr=False, hilbert=hilbert, mollweide=mollweide, center_dir=center_dir, zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,circle_zenith=circle_zenith,circle_az=circle_az,time_delay_dict=time_delay_dict,include_baselines=include_baselines,window_title=window_title, plot_peak_to_sidelobe=plot_peak_to_sidelobe)
                #     vpol_result = self.map(eventid,'vpol', plot_map=False, plot_corr=False, hilbert=hilbert, mollweide=mollweide, center_dir=center_dir, zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,circle_zenith=circle_zenith,circle_az=circle_az,time_delay_dict=time_delay_dict,include_baselines=include_baselines,window_title=window_title, plot_peak_to_sidelobe=plot_peak_to_sidelobe)
                #     mean_corr_values = (hpol_result + vpol_result)/2

                #     if plot_map == True:
                #         if max_method is not None:
                #             if plot_peak_to_sidelobe:
                #                 linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3, peak_to_sidelobe = self.mapMax(mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol='all', return_peak_to_sidelobe=plot_peak_to_sidelobe)
                #             else:
                #                 linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol='all', return_peak_to_sidelobe=plot_peak_to_sidelobe)

                #         else:
                #             if plot_peak_to_sidelobe:
                #                 linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3, peak_to_sidelobe = self.mapMax(mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol='all', return_peak_to_sidelobe=plot_peak_to_sidelobe)
                #             else:
                #                 linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane,pol='all', return_peak_to_sidelobe=plot_peak_to_sidelobe)

                #     if plot_corr == True:
                #         print('Disabling plot corr for all antenna map.')
                #         plot_corr = False
                #     if interactive == True:
                #         print('Disabling interactive for all antenna map.')
                #         interactive = False
            elif pol == 'all':
                if waveforms is None:
                    waveforms = self.wf(eventid, numpy.array([0,1,2,3,4,5,6,7]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter,tukey=self.apply_tukey, sine_subtract=self.apply_sine_subtract) #Div by std and resampled waveforms normalizes the correlations

                    if shorten_signals == True:
                        waveforms = self.shortenSignals(waveforms,shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)

                #TODO: Could consider upsample only the cross correlations AFTER, rather than upsampling the waveforms, also consider passing scipy.signal.correlate the desired method

                #Hpol signals:
                corr01_h = (numpy.asarray(scipy.signal.correlate(waveforms[0*2],waveforms[1*2])))/(len(self.times_resampled))
                corr02_h = (numpy.asarray(scipy.signal.correlate(waveforms[0*2],waveforms[2*2])))/(len(self.times_resampled))
                corr03_h = (numpy.asarray(scipy.signal.correlate(waveforms[0*2],waveforms[3*2])))/(len(self.times_resampled))
                corr12_h = (numpy.asarray(scipy.signal.correlate(waveforms[1*2],waveforms[2*2])))/(len(self.times_resampled))
                corr13_h = (numpy.asarray(scipy.signal.correlate(waveforms[1*2],waveforms[3*2])))/(len(self.times_resampled))
                corr23_h = (numpy.asarray(scipy.signal.correlate(waveforms[2*2],waveforms[3*2])))/(len(self.times_resampled))

                #Vpol singals:
                corr01_v = (numpy.asarray(scipy.signal.correlate(waveforms[0*2 + 1],waveforms[1*2 + 1])))/(len(self.times_resampled))
                corr02_v = (numpy.asarray(scipy.signal.correlate(waveforms[0*2 + 1],waveforms[2*2 + 1])))/(len(self.times_resampled))
                corr03_v = (numpy.asarray(scipy.signal.correlate(waveforms[0*2 + 1],waveforms[3*2 + 1])))/(len(self.times_resampled))
                corr12_v = (numpy.asarray(scipy.signal.correlate(waveforms[1*2 + 1],waveforms[2*2 + 1])))/(len(self.times_resampled))
                corr13_v = (numpy.asarray(scipy.signal.correlate(waveforms[1*2 + 1],waveforms[3*2 + 1])))/(len(self.times_resampled))
                corr23_v = (numpy.asarray(scipy.signal.correlate(waveforms[2*2 + 1],waveforms[3*2 + 1])))/(len(self.times_resampled))
                
              
                mean_corr_values = ((corr01_h[self.delay_indices_hpol_0subtract1] if 0 in include_baselines else 0) + (corr02_h[self.delay_indices_hpol_0subtract2] if 1 in include_baselines else 0) + (corr03_h[self.delay_indices_hpol_0subtract3] if 2 in include_baselines else 0) + (corr12_h[self.delay_indices_hpol_1subtract2] if 3 in include_baselines else 0) + (corr13_h[self.delay_indices_hpol_1subtract3] if 4 in include_baselines else 0) + (corr23_h[self.delay_indices_hpol_2subtract3] if 5 in include_baselines else 0) + (corr01_v[self.delay_indices_vpol_0subtract1] if 0 in include_baselines else 0) + (corr02_v[self.delay_indices_vpol_0subtract2] if 1 in include_baselines else 0) + (corr03_v[self.delay_indices_vpol_0subtract3] if 2 in include_baselines else 0) + (corr12_v[self.delay_indices_vpol_1subtract2] if 3 in include_baselines else 0) + (corr13_v[self.delay_indices_vpol_1subtract3] if 4 in include_baselines else 0) + (corr23_v[self.delay_indices_vpol_2subtract3] if 5 in include_baselines else 0))/(2*len(include_baselines))

                if plot_map == True:
                    if max_method is not None:
                        if plot_peak_to_sidelobe:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3, peak_to_sidelobe = self.mapMax(mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)
                        else:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)
                    else:
                        if plot_peak_to_sidelobe:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3, peak_to_sidelobe = self.mapMax(mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)
                        else:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)





            elif pol == 'hpol':
                if waveforms is None:
                    waveforms = self.wf(eventid, numpy.array([0,2,4,6]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter,tukey=self.apply_tukey, sine_subtract=self.apply_sine_subtract) #Div by std and resampled waveforms normalizes the correlations

                    if shorten_signals == True:
                        waveforms = self.shortenSignals(waveforms,shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)

                #TODO: Could consider upsample only the cross correlations AFTER, rather than upsampling the waveforms, also consider passing scipy.signal.correlate the desired method

                corr01 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[1])))/(len(self.times_resampled))
                corr02 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[2])))/(len(self.times_resampled))
                corr03 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[3])))/(len(self.times_resampled))
                corr12 = (numpy.asarray(scipy.signal.correlate(waveforms[1],waveforms[2])))/(len(self.times_resampled))
                corr13 = (numpy.asarray(scipy.signal.correlate(waveforms[1],waveforms[3])))/(len(self.times_resampled))
                corr23 = (numpy.asarray(scipy.signal.correlate(waveforms[2],waveforms[3])))/(len(self.times_resampled))
                
                '''
                # Commented out because it is signicantly faster to do these all as one line of code.                
                # corr_value_0subtract1 = corr01[self.delay_indices_hpol_0subtract1]
                # corr_value_0subtract2 = corr02[self.delay_indices_hpol_0subtract2]
                # corr_value_0subtract3 = corr03[self.delay_indices_hpol_0subtract3]
                # corr_value_1subtract2 = corr12[self.delay_indices_hpol_1subtract2]
                # corr_value_1subtract3 = corr13[self.delay_indices_hpol_1subtract3]
                # corr_value_2subtract3 = corr23[self.delay_indices_hpol_2subtract3]

                # stacked_corr_values = numpy.array([corr_value_0subtract1, corr_value_0subtract2, corr_value_0subtract3, corr_value_1subtract2, corr_value_1subtract3, corr_value_2subtract3] )
                # mean_corr_values = numpy.mean(stacked_corr_values[include_baselines],axis=0)
                '''
                #mean_corr_values = (corr01[self.delay_indices_hpol_0subtract1] + corr02[self.delay_indices_hpol_0subtract2] + corr03[self.delay_indices_hpol_0subtract3] + corr12[self.delay_indices_hpol_1subtract2] + corr13[self.delay_indices_hpol_1subtract3] + corr23[self.delay_indices_hpol_2subtract3])/6.0
                mean_corr_values = ((corr01[self.delay_indices_hpol_0subtract1] if 0 in include_baselines else 0) + (corr02[self.delay_indices_hpol_0subtract2] if 1 in include_baselines else 0) + (corr03[self.delay_indices_hpol_0subtract3] if 2 in include_baselines else 0) + (corr12[self.delay_indices_hpol_1subtract2] if 3 in include_baselines else 0) + (corr13[self.delay_indices_hpol_1subtract3] if 4 in include_baselines else 0) + (corr23[self.delay_indices_hpol_2subtract3] if 5 in include_baselines else 0))/len(include_baselines)

                if plot_map == True:
                    if max_method is not None:
                        if plot_peak_to_sidelobe:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3, peak_to_sidelobe = self.mapMax(mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)
                        else:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)
                    else:
                        if plot_peak_to_sidelobe:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3, peak_to_sidelobe = self.mapMax(mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)
                        else:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)

            elif pol == 'vpol':
                if waveforms is None:
                    waveforms = self.wf(eventid, numpy.array([1,3,5,7]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter,tukey=self.apply_tukey, sine_subtract=self.apply_sine_subtract) #Div by std and resampled waveforms normalizes the correlations

                    if shorten_signals == True:
                        waveforms = self.shortenSignals(waveforms,shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)

                #TODO: Could consider upsample only the cross correlations AFTER, rather than upsampling the waveforms, also consider passing scipy.signal.correlate the desired method

                corr01 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[1])))/(len(self.times_resampled))
                corr02 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[2])))/(len(self.times_resampled))
                corr03 = (numpy.asarray(scipy.signal.correlate(waveforms[0],waveforms[3])))/(len(self.times_resampled))
                corr12 = (numpy.asarray(scipy.signal.correlate(waveforms[1],waveforms[2])))/(len(self.times_resampled))
                corr13 = (numpy.asarray(scipy.signal.correlate(waveforms[1],waveforms[3])))/(len(self.times_resampled))
                corr23 = (numpy.asarray(scipy.signal.correlate(waveforms[2],waveforms[3])))/(len(self.times_resampled))

                '''
                # Commented out because it is signicantly faster to do these all as one line of code.                

                # corr_value_0subtract1 = corr01[self.delay_indices_vpol_0subtract1]
                # corr_value_0subtract2 = corr02[self.delay_indices_vpol_0subtract2]
                # corr_value_0subtract3 = corr03[self.delay_indices_vpol_0subtract3]
                # corr_value_1subtract2 = corr12[self.delay_indices_vpol_1subtract2]
                # corr_value_1subtract3 = corr13[self.delay_indices_vpol_1subtract3]
                # corr_value_2subtract3 = corr23[self.delay_indices_vpol_2subtract3]
                
                # stacked_corr_values = numpy.array([corr_value_0subtract1, corr_value_0subtract2, corr_value_0subtract3, corr_value_1subtract2, corr_value_1subtract3, corr_value_2subtract3] )
                # mean_corr_values = numpy.mean(stacked_corr_values[include_baselines],axis=0)
                '''
                #0 if 0 in include_baselines else corr01[self.delay_indices_vpol_0subtract1]
                mean_corr_values = ((corr01[self.delay_indices_vpol_0subtract1] if 0 in include_baselines else 0) + (corr02[self.delay_indices_vpol_0subtract2] if 1 in include_baselines else 0) + (corr03[self.delay_indices_vpol_0subtract3] if 2 in include_baselines else 0) + (corr12[self.delay_indices_vpol_1subtract2] if 3 in include_baselines else 0) + (corr13[self.delay_indices_vpol_1subtract3] if 4 in include_baselines else 0) + (corr23[self.delay_indices_vpol_2subtract3] if 5 in include_baselines else 0))/len(include_baselines)

                if plot_map == True:
                    if max_method is not None:
                        if plot_peak_to_sidelobe:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3, peak_to_sidelobe = self.mapMax(mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)
                        else:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)
                    else:
                        if plot_peak_to_sidelobe:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3, peak_to_sidelobe = self.mapMax(mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)
                        else:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=plot_peak_to_sidelobe)

            else:
                print('Invalid polarization option.  Returning nothing.')
                return


            if return_max_possible_map_value == True:
                if pol == 'all':
                    max_values_h = numpy.array([numpy.max(corr01_h),numpy.max(corr02_h),numpy.max(corr03_h),numpy.max(corr12_h),numpy.max(corr13_h),numpy.max(corr23_h)])
                    max_values_v = numpy.array([numpy.max(corr01_v),numpy.max(corr02_v),numpy.max(corr03_v),numpy.max(corr12_v),numpy.max(corr13_v),numpy.max(corr23_v)])
                    max_possible_map_value = numpy.mean(numpy.append(max_values_h[include_baselines], max_values_v[include_baselines])) #This would occur if the map samples the cross correlations each at their exact maximum value.   This may never actually occur in a non-ideal calibration.
                else:
                    max_values = numpy.array([numpy.max(corr01),numpy.max(corr02),numpy.max(corr03),numpy.max(corr12),numpy.max(corr13),numpy.max(corr23)])
                    max_possible_map_value = numpy.mean(max_values[include_baselines]) #This would occur if the map samples the cross correlations each at their exact maximum value.   This may never actually occur in a non-ideal calibration.

            if plot_corr:
                if pol == 'all':
                    fig = plt.figure()
                    fig.canvas.set_window_title('HPol Correlations')
                    center = len(self.times_resampled)
                    shifts = (numpy.arange(len(corr01_h)) - center + 1)*self.dt_resampled
                    if True:
                        #To check normalization.  Should be 1 at 0 time. 
                        corr00_h = (numpy.asarray(scipy.signal.correlate_h(waveforms[0],waveforms[0])))/len(self.times_resampled) 
                        plt.plot(shifts,corr00_h,alpha=0.7,label='autocorr00_h')
                    plt.plot(shifts,corr01_h,alpha=0.7,label='corr01_h' + [' (Not Included in Map)', ''][numpy.isin( 0, include_baselines).astype(int)])
                    plt.plot(shifts,corr02_h,alpha=0.7,label='corr02_h' + [' (Not Included in Map)', ''][numpy.isin( 1, include_baselines).astype(int)])
                    plt.plot(shifts,corr03_h,alpha=0.7,label='corr03_h' + [' (Not Included in Map)', ''][numpy.isin( 2, include_baselines).astype(int)])
                    plt.plot(shifts,corr12_h,alpha=0.7,label='corr12_h' + [' (Not Included in Map)', ''][numpy.isin( 3, include_baselines).astype(int)])
                    plt.plot(shifts,corr13_h,alpha=0.7,label='corr13_h' + [' (Not Included in Map)', ''][numpy.isin( 4, include_baselines).astype(int)])
                    plt.plot(shifts,corr23_h,alpha=0.7,label='corr23_h' + [' (Not Included in Map)', ''][numpy.isin( 5, include_baselines).astype(int)])
                    plt.legend()
                    self.figs.append(fig)
                    self.axs.append(fig.gca())

                    fig = plt.figure()
                    fig.canvas.set_window_title('VPol Correlations')
                    center = len(self.times_resampled)
                    shifts = (numpy.arange(len(corr01_v)) - center + 1)*self.dt_resampled
                    if True:
                        #To check normalization.  Should be 1 at 0 time. 
                        corr00_v = (numpy.asarray(scipy.signal.correlate_v(waveforms[0],waveforms[0])))/len(self.times_resampled) 
                        plt.plot(shifts,corr00_v,alpha=0.7,label='autocorr00_v')
                    plt.plot(shifts,corr01_v,alpha=0.7,label='corr01_v' + [' (Not Included in Map)', ''][numpy.isin( 0, include_baselines).astype(int)])
                    plt.plot(shifts,corr02_v,alpha=0.7,label='corr02_v' + [' (Not Included in Map)', ''][numpy.isin( 1, include_baselines).astype(int)])
                    plt.plot(shifts,corr03_v,alpha=0.7,label='corr03_v' + [' (Not Included in Map)', ''][numpy.isin( 2, include_baselines).astype(int)])
                    plt.plot(shifts,corr12_v,alpha=0.7,label='corr12_v' + [' (Not Included in Map)', ''][numpy.isin( 3, include_baselines).astype(int)])
                    plt.plot(shifts,corr13_v,alpha=0.7,label='corr13_v' + [' (Not Included in Map)', ''][numpy.isin( 4, include_baselines).astype(int)])
                    plt.plot(shifts,corr23_v,alpha=0.7,label='corr23_v' + [' (Not Included in Map)', ''][numpy.isin( 5, include_baselines).astype(int)])
                    plt.legend()
                    self.figs.append(fig)
                    self.axs.append(fig.gca())

                else:
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
                    add_text = '\nIncluded baselines = ' + str(numpy.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])[include_baselines])
                else:
                    add_text = '\n' + str(self.deploy_index)

                if center_dir.upper() == 'E':
                    center_dir_full = 'East'
                    azimuth_offset_rad = 0 #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = 0 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From East = 0 deg, North = 90 deg)' + add_text
                    roll = 0
                elif center_dir.upper() == 'N':
                    center_dir_full = 'North'
                    azimuth_offset_rad = numpy.pi/2 #This is subtracted from the xaxis to roll it effectively. 
                    azimuth_offset_deg = 90 #This is subtracted from the xaxis to roll it effectively. 
                    xlabel = 'Azimuth (From North = 0 deg, West = 90 deg)' + add_text
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))
                elif center_dir.upper() == 'W':
                    center_dir_full = 'West'
                    azimuth_offset_rad = numpy.pi #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = 180 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From West = 0 deg, South = 90 deg)' + add_text
                    roll = len(self.phis_rad)//2
                elif center_dir.upper() == 'S':
                    center_dir_full = 'South'
                    azimuth_offset_rad = -numpy.pi/2 #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = -90 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From South = 0 deg, East = 90 deg)' + add_text
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))



                rolled_values = numpy.roll(mean_corr_values,roll,axis=1)




                elevation_best_deg = 90.0 - theta_best

                if map_ax is None:
                    fig = plt.figure()
                    if window_title is None:
                        fig.canvas.set_window_title('r%i-e%i-%s Correlation Map'%(self.reader.run,eventid,pol.title()))
                    else:
                        fig.canvas.set_window_title(window_title)

                    if mollweide == True:
                        map_ax = fig.add_subplot(1,1,1, projection='mollweide')
                    else:
                        map_ax = fig.add_subplot(1,1,1)

                    if True:
                        map_ax.set_title('%i-%i-%s-Hilbert=%s\nSine Subtract %s\nSource Distance = %0.2f m'%(self.reader.run,eventid,pol,str(hilbert), ['Disabled','Enabled'][int(self.apply_sine_subtract)], self.map_source_distance_m)) #FORMATTING SPECIFIC AND PARSED ELSEWHERE, DO NOT CHANGE. 
                    else:
                        map_ax.set_title('%i-%i-%s-Hilbert=%s\nSource Distance = %0.2f m'%(self.reader.run,eventid,pol,str(hilbert), self.map_source_distance_m)) #FORMATTING SPECIFIC AND PARSED ELSEWHERE, DO NOT CHANGE. 
                else:
                    fig = map_ax.figure

                if mollweide == True:
                    #Automatically converts from rads to degs
                    im = map_ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, rolled_values, vmin=numpy.min(rolled_values), vmax=numpy.max(rolled_values),cmap=plt.cm.coolwarm)
                else:
                    im = map_ax.pcolormesh(self.mesh_azimuth_deg, self.mesh_elevation_deg, rolled_values, vmin=numpy.min(rolled_values), vmax=numpy.max(rolled_values),cmap=plt.cm.coolwarm)

                if plot_peak_to_sidelobe and max_method is not None:
                    blank_patch = matplotlib.patches.Patch(color='red', alpha=0.0, label='Peak to Sidelobe: %0.3f'%peak_to_sidelobe)
                    map_ax.legend(handles=[blank_patch], loc='upper right')

                #cbar = fig.colorbar(im)

                if minimal == True:
                    #cbar = plt.colorbar(im, ax=map_ax,fraction=0.046, pad=0.04)
                    map_ax.set_xlabel(pol + ' MV=%0.2f'%(mean_corr_values.flat[linear_max_index]),fontsize=14)
                    map_ax.grid(True)
                else:
                    cbar = plt.colorbar(im, ax=map_ax)
                    if hilbert == True:
                        cbar.set_label('Mean Correlation Value (Arb)')
                    else:
                        cbar.set_label('Mean Correlation Value')
                    map_ax.set_xlabel(xlabel,fontsize=18)
                    map_ax.set_ylabel('Elevation Angle (Degrees)',fontsize=18)
                    map_ax.grid(True)

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
                    plane_xy = self.getPlaneZenithCurves(self.n_hpol.copy(), 'hpol', 90.0, azimuth_offset_deg=azimuth_offset_deg)
                    if zenith_cut_array_plane is not None:
                        upper_plane_xy = self.getPlaneZenithCurves(self.n_hpol.copy(), 'hpol', zenith_cut_array_plane[0], azimuth_offset_deg=azimuth_offset_deg)
                        lower_plane_xy = self.getPlaneZenithCurves(self.n_hpol.copy(), 'hpol', zenith_cut_array_plane[1], azimuth_offset_deg=azimuth_offset_deg)
                elif pol == 'vpol':
                    plane_xy = self.getPlaneZenithCurves(self.n_vpol.copy(), 'vpol', 90.0, azimuth_offset_deg=azimuth_offset_deg)
                    if zenith_cut_array_plane is not None:
                        upper_plane_xy = self.getPlaneZenithCurves(self.n_vpol.copy(), 'vpol', zenith_cut_array_plane[0], azimuth_offset_deg=azimuth_offset_deg)
                        lower_plane_xy = self.getPlaneZenithCurves(self.n_vpol.copy(), 'vpol', zenith_cut_array_plane[1], azimuth_offset_deg=azimuth_offset_deg)
                elif pol == 'all':
                    plane_xy = self.getPlaneZenithCurves(self.n_all.copy(), 'all', 90.0, azimuth_offset_deg=azimuth_offset_deg) #This is janky
                    if zenith_cut_array_plane is not None:
                        upper_plane_xy = self.getPlaneZenithCurves(self.n_all.copy(), 'all', zenith_cut_array_plane[0], azimuth_offset_deg=azimuth_offset_deg)
                        lower_plane_xy = self.getPlaneZenithCurves(self.n_all.copy(), 'all', zenith_cut_array_plane[1], azimuth_offset_deg=azimuth_offset_deg)

                #Plot array plane 0 elevation curve.
                im = self.addCurveToMap(im, plane_xy, ax=map_ax, mollweide=mollweide, linewidth = self.min_elevation_linewidth, color='k')

                if self.conference_mode:
                    ticks_deg = numpy.array([-60,-40,-30,-15,0,15,30,45,60,75])
                    if mollweide == True:
                        plt.set_yticks(numpy.deg2rad(ticks_deg))
                    else:
                        plt.set_yticks(ticks_deg)
                    x = plane_xy[0]
                    y1 = plane_xy[1]
                    if mollweide == True:
                        y2 = -numpy.pi/2 * numpy.ones_like(plane_xy[0])#lower_plane_xy[1]
                    else:
                        y2 = -90 * numpy.ones_like(plane_xy[0])#lower_plane_xy[1]
                    map_ax.fill_between(x, y1, y2, where=y2 <= y1,facecolor='#9DC3E6', interpolate=True,alpha=1)#'#EEC6C7'
                
                if zenith_cut_array_plane is not None:
                    #Plot upper zenith array cut
                    im = self.addCurveToMap(im, upper_plane_xy, ax=map_ax,  mollweide=mollweide, linewidth = self.min_elevation_linewidth, color='k',linestyle = '--')
                    #Plot lower zenith array cut
                    im = self.addCurveToMap(im, lower_plane_xy, ax=map_ax,  mollweide=mollweide, linewidth = self.min_elevation_linewidth, color='k',linestyle = '--')


                if pol != 'all':
                    #Add curves for time delays if present.
                    im, map_ax = self.addTimeDelayCurves(im, time_delay_dict, pol, map_ax, mollweide=mollweide, azimuth_offset_deg=azimuth_offset_deg, include_baselines=include_baselines)


                if circle_map_max:
                    #Added circles as specified.
                    map_ax, peak_circle = self.addCircleToMap(map_ax, phi_best, elevation_best_deg, azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius = radius, crosshair=True, return_circle=True, color='lime', linewidth=0.5,fill=False)

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
                                ax, _circ = self.addCircleToMap(map_ax, _circle_az[i], 90.0-_circle_zenith[i], azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius = radius, crosshair=True, return_circle=True, color='fuchsia', linewidth=0.5,fill=False)
                                additional_circles.append(_circ)

                if add_airplanes:
                    map_ax, airplane_direction_dict = self.addAirplanesToMap([eventid], pol, map_ax, azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius = radius, crosshair=False, color='r', min_approach_cut_km=500,plot_distance_cut_limit=500)
                    if verbose:
                        print('airplane_direction_dict:')
                        print(airplane_direction_dict)
                if zenith_cut_ENU is not None:
                    if mollweide == True:
                        #Block out simple ENU zenith cut region. 
                        plt.axhspan(numpy.deg2rad(90 - min(zenith_cut_ENU)),numpy.deg2rad(90.0),alpha=0.5)
                        plt.axhspan(numpy.deg2rad(-90) , numpy.deg2rad(90 - max(zenith_cut_ENU)),alpha=0.5)
                        # plt.ylim(numpy.deg2rad((90.0 - max(self.range_theta_deg) , 90.0 - min(self.range_theta_deg))  ))
                        # plt.xlim(numpy.deg2rad(self.range_phi_deg))

                    else:
                        #Block out simple ENU zenith cut region. 
                        plt.axhspan(90 - min(zenith_cut_ENU),90.0,alpha=0.5)
                        plt.axhspan(-90 , 90 - max(zenith_cut_ENU),alpha=0.5)
                        plt.ylim(90.0 - max(self.range_theta_deg) , 90.0 - min(self.range_theta_deg)  )
                        plt.xlim(self.range_phi_deg)

                #Enable Interactive Portion
                if interactive == True:
                    print('Map should be interactive')
                    #TODO: Should figure out a way to call  canvas.disconnect on this.   Will need to store the cid somewhere?
                    '''

                    if event.inaxes==self.ax1: self.fig.mpl_connect('pick_event', self._onpick_plot_1)
                    '''
                    # map_ax.figure.canvas.mpl_connect('button_press_event',lambda event : self.interactivePlotter(event, pol=pol, hilbert=hilbert, eventid=eventid, mollweide=mollweide, center_dir=center_dir, all_alignments=self.all_alignments, shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading))
                    map_ax.figure.canvas.mpl_connect('button_press_event',lambda event : self.interactivePlotter(event, pol=pol, hilbert=hilbert, eventid=eventid, mollweide=mollweide, center_dir=center_dir, all_alignments=self.all_alignments, shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading) if event.inaxes == map_ax else None)
                    
                #map_ax.legend(loc='lower left')
                #self.figs.append(fig)
                self.figs.append(map_ax.figure)
                self.axs.append(map_ax)

                if return_max_possible_map_value == True:
                    return mean_corr_values, fig, map_ax, max_possible_map_value
                else:
                    return mean_corr_values, fig, map_ax
            else:
                if return_max_possible_map_value == True:
                    return mean_corr_values, max_possible_map_value
                else:
                    return mean_corr_values

                return mean_corr_values
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def averagedMap(self, eventids, pol, plot_map=True, hilbert=False, max_method=None, mollweide=False, zenith_cut_ENU=None,zenith_cut_array_plane=None, center_dir='E', circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={}):
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
                linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(total_mean_corr_values,max_method=max_method,verbose=True,zenith_cut_ENU=zenith_cut_ENU,zenith_cut_array_plane=zenith_cut_array_plane,pol=pol)
            else:
                linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(total_mean_corr_values,verbose=True,zenith_cut_ENU=zenith_cut_ENU,zenith_cut_array_plane=zenith_cut_array_plane,pol=pol)        

            elevation_best_deg = 90.0 - theta_best

            if pol == 'hpol':
                t_best_0subtract1  = self.t_hpol_0subtract1.flat[linear_max_index]
                t_best_0subtract2  = self.t_hpol_0subtract2.flat[linear_max_index]
                t_best_0subtract3  = self.t_hpol_0subtract3.flat[linear_max_index]
                t_best_1subtract2  = self.t_hpol_1subtract2.flat[linear_max_index]
                t_best_1subtract3  = self.t_hpol_1subtract3.flat[linear_max_index]
                t_best_2subtract3  = self.t_hpol_2subtract3.flat[linear_max_index]
            elif pol == 'vpol':
                t_best_0subtract1  = self.t_vpol_0subtract1.flat[linear_max_index]
                t_best_0subtract2  = self.t_vpol_0subtract2.flat[linear_max_index]
                t_best_0subtract3  = self.t_vpol_0subtract3.flat[linear_max_index]
                t_best_1subtract2  = self.t_vpol_1subtract2.flat[linear_max_index]
                t_best_1subtract3  = self.t_vpol_1subtract3.flat[linear_max_index]
                t_best_2subtract3  = self.t_vpol_2subtract3.flat[linear_max_index]
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
            im = self.addTimeDelayCurves(im, time_delay_dict, pol, ax, mollweide=mollweide, azimuth_offset_deg=azimuth_offset_deg)

            #Added circles as specified.
            ax, peak_circle = self.addCircleToMap(ax, phi_best, elevation_best_deg, azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius=radius, crosshair=True, return_circle=True, color='lime', linewidth=0.5,fill=False)

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
                            ax, _circ = self.addCircleToMap(ax, _circle_az[i], 90.0-_circle_zenith[i], azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius=radius, crosshair=False, return_circle=True, color='fuchsia', linewidth=0.5,fill=False)
                            additional_circles.append(_circ)

            #Block out simple ENU zenith cut region. 
            if zenith_cut_ENU is not None:
                if mollweide == True:
                    #Block out simple ENU zenith cut region. 
                    plt.axhspan(numpy.deg2rad(90 - min(zenith_cut_ENU)),numpy.deg2rad(90.0),alpha=0.5)
                    plt.axhspan(numpy.deg2rad(-90) , numpy.deg2rad(90 - max(zenith_cut_ENU)),alpha=0.5)
                    plt.ylim(numpy.deg2rad((90.0 - max(self.range_theta_deg) , 90.0 - min(self.range_theta_deg))  ))
                    plt.xlim(numpy.deg2rad(self.range_phi_deg))

                else:
                    #Block out simple ENU zenith cut region. 
                    plt.axhspan(90 - min(zenith_cut_ENU),90.0,alpha=0.5)
                    plt.axhspan(-90 , 90 - max(zenith_cut_ENU),alpha=0.5)
                    plt.ylim(90.0 - max(self.range_theta_deg) , 90.0 - min(self.range_theta_deg)  )
                    plt.xlim(self.range_phi_deg)

            ax.legend(loc='lower left')
            self.figs.append(fig)
            self.axs.append(ax)

            return total_mean_corr_values, fig, ax
        else:
            return total_mean_corr_values

    def animatedMap(self, eventids, pol, title, include_baselines=[0,1,2,3,4,5], plane_zenith=None, plane_az=None, map_source_distance_m=None, radius=1.0, hilbert=False, max_method=None,center_dir='E',save=True,dpi=300,fps=3):
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
        map_source_distance_m : list of floats
            This must be the same length as eventids (with distance corresponding to each eventid).  For each frame of 
            this animation, this will recalculate expected time delays based on this distance.  This is currently not 
            implemented but is intended to allow for accurate video maps of airplane trajectories. 
        '''
        if pol == 'both':
            hpol_result = self.animatedMap(eventids, 'hpol', title,include_baselines=include_baselines, plane_zenith=plane_zenith, plane_az=plane_az, hilbert=hilbert, max_method=max_method,center_dir=center_dir,save=save, map_source_distance_m=None)
            vpol_result = self.animatedMap(eventids, 'vpol', title,include_baselines=include_baselines, plane_zenith=plane_zenith, plane_az=plane_az, hilbert=hilbert, max_method=max_method,center_dir=center_dir,save=save, map_source_distance_m=None)

            return hpol_result, vpol_result

        try:
            print('Performing calculations for %s'%pol)

            if map_source_distance_m is None:
                change_per_frame = False
                #Keep source distance as None for default handling.
            elif numpy.size(map_source_distance_m) == 1:
                change_per_frame = False
                map_source_distance_m = float(map_source_distance_m)
                self.overwriteSourceDistance(map_source_distance_m, verbose=False, suppress_time_delay_calculations=False)
            elif numpy.size(map_source_distance_m) == numpy.size(eventids):
                change_per_frame = True
            else:
                change_per_frame = False
                #Keep source distance as None for default handling.

            all_maps = []# numpy.zeros((self.n_theta, self.n_phi))
            for event_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
                sys.stdout.flush()
                if change_per_frame == True:
                    self.overwriteSourceDistance(map_source_distance_m[event_index], verbose=False, suppress_time_delay_calculations=False)
                m = self.map(eventid, pol, include_baselines=include_baselines, plot_map=False, plot_corr=False, hilbert=hilbert) #Don't need to pass center_dir because performing rotation below!
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
            ax.set_title('r%i %s Correlation Map Eventid = %i\nSource Distance = %0.2f m'%(self.reader.run,pol.title(),eventids[0],self.map_source_distance_m))

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

            # plt.ylim(90.0 - max(self.range_theta_deg) , 90.0 - min(self.range_theta_deg)  )
            # plt.xlim(self.range_phi_deg)


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
                
                if numpy.size(plane_az) == 1:
                    _plane_az = plane_az.copy() #These were changing when 'both' was called for pol.  So copying them to ensure original input maintains values.
                    _plane_az -= azimuth_offset_deg #handled here so shouldn't be passed to addCircleToMap or similar functions.
                    _plane_az = float(_plane_az)*numpy.ones_like(eventids,dtype=float)
                else:
                    _plane_az = plane_az.copy() #These were changing when 'both' was called for pol.  So copying them to ensure original input maintains values.
                    _plane_az -= azimuth_offset_deg #handled here so shouldn't be passed to addCircleToMap or similar functions.
                
                _plane_az[_plane_az < -180.0] += 360.0
                _plane_az[_plane_az > 180.0] -= 360.0


            if plane_zenith is not None:
                if plane_zenith is list:
                    plane_zenith = numpy.array(plane_zenith)
                _plane_zenith = plane_zenith.copy() #These were changing when 'both' was called for pol.  So copying them to ensure original input maintains values.
                _plane_zenith = float(_plane_zenith)*numpy.ones_like(eventids,dtype=float)
                plane_elevation = 90.0 - _plane_zenith
            else:
                plane_elevation = None

            if plane_elevation is not None and _plane_az is not None:
                ax, circle, lines = self.addCircleToMap(ax, _plane_az[0], plane_elevation[0], azimuth_offset_deg=0.0, mollweide=True, radius=radius, crosshair=True, return_circle=True, return_crosshair=True, color='fuchsia', linewidth=2.0,fill=False,zorder=10) #azimuth offset_deg already accounted for as passed.

            if self.conference_mode:
                ticks_deg = numpy.array([-60,-40,-30,-15,0,15,30,45,60,75])
                plt.yticks(numpy.deg2rad(ticks_deg))
                x = plane_xy[0]
                y1 = plane_xy[1]
                y2 = -numpy.pi/2 * numpy.ones_like(plane_xy[0])#lower_plane_xy[1]
                y1_interp = scipy.interpolate.interp1d(x,y1)
                ax.fill_between(x, y1, y2, where=y2 <= y1,facecolor='#9DC3E6', interpolate=True,alpha=1)#'#EEC6C7'
                #plt.plot(plane_xy[0], plane_xy[1],linestyle='-',linewidth=6,color='#41719C')

            def update(frame):
                _frame = frame%len(eventids) #lets it loop multiple times.  i.e. give animation more frames but same content looped.
                # fig.canvas.set_window_title('Correlation Map Airplane Event %i'%(_frame))
                # ax.set_title('Correlation Map Airplane Event %i'%(_frame))
                fig.canvas.set_window_title('r%i %s Correlation Map Eventid = %i'%(self.reader.run,pol.title(),eventids[_frame]))
                if change_per_frame:
                    ax.set_title('r%i %s Correlation Map Eventid = %i\nSource Distance = %0.2f km'%(self.reader.run,pol.title(),eventids[_frame],map_source_distance_m[_frame]/1000.0))
                else:
                    ax.set_title('r%i %s Correlation Map Eventid = %i\nSource Distance = %0.2f km'%(self.reader.run,pol.title(),eventids[_frame],self.map_source_distance_m/1000.0))
                im.set_array(all_maps[_frame][:-1,:-1].ravel()) #Some obscure bug as noted here: https://stackoverflow.com/questions/29009743/using-set-array-with-pyplot-pcolormesh-ruins-figure

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

                    if self.conference_mode:
                        ax.fill_between(x, y1, y2, where=y2 <= y1,facecolor='#9DC3E6', interpolate=True,alpha=1)#'#EEC6C7'

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

    def histMapPeak(self, eventids, pol, initial_hist=None, initial_thetas=None, initial_phis=None, plot_map=True, return_fig=False, hilbert=False, max_method=None, plot_max=True, use_weight=False, mollweide=False, include_baselines=[0,1,2,3,4,5], center_dir='E', zenith_cut_ENU=None, zenith_cut_array_plane=None, circle_zenith=None, circle_az=None, window_title=None,radius=1.0,iterate_sub_baselines=None, shift_1d_hists=False, acceptable_fit_range=None, return_max_possible_map_values=False, initial_max_possible_map_values=None, return_map_peaks=False, return_peak_to_sidelobe=False, initial_peaks=None, initial_peak_to_sidelobes=None, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0):
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
        return_fig : bool
            If True AND plot_map is True then the matplotlib figure will be returned.  Note that this does not return
            the correlation plots if plot_corr is enabled (though they will still be plotted).
        plot_corr : bool
            Plot the cross-correlations for each baseline.
        hilbert : bool
            Enables performing calculations with Hilbert envelopes of waveforms. 
        max_method : bool
            Determines how the most probable source direction is from the map.
        center_dir : str
            Specifies the center direction when plotting.  By default this is 'E' which is East (ENU standard).
        mollweide : bool
            Makes the plot with a mollweide projection.  Default is False.
        zenith_cut_ENU : list of 2 float values
            Given in degrees, angles within these two values are considered.  If None is given for the first then it is
            assumed to be 0 (overhead), if None is given for the latter then it is assumed to be 180 (straight down).
        zenith_cut_array_plane : list of 2 float values
            Given in degrees, angles within these two values are considered.  If None is given for the first then it is
            assumed to be 0 (overhead), if None is given for the latter then it is assumed to be 180 (straight down).  This is
            polarization dependant because it depends on the calibration of the antennas positions.  So if pol=None then this will
            be ignored.
        circle_zenith : list of floats
            List of zenith values to circle on the plot.  These could be known background sources like planes.
            The length of this must match the length of circle_az.  Should be given in degrees.
        circle_az : list of floats
            List of azimuths values to circle on the plot.  These could be known background sources like planes.
            The length of this must match the length of circle_zenith.  Should be given in degrees.
        initial_hist : None or 2d array
            This allows for the user to pass a "starting" condition for this histogram.  This can be used to loop over
            histograms from multiple runs, and plot the summed histogram (by consecutively feed each run with the past)
            runs output.  This requries them all to have the same binning.  
        iterate_sub_baselines : int
            This is the subset size of subsets to include when generating the histogram.  Using this subset size, each
            possible set of baselines within the predefined include_baselines is iterated over, with the max peak
            being selected.  This allows for the user to have a spread visible that is induced by non-overlapping baselines
            even if the signal itself is very consistent in form (and thus wouldn't have a large spread in histogram, 
            even if it is inaccurate).  Default is None which results in matching the length of the original 
            include_baselines (resulting in only the 1 full map being used).  You can use any subset you like, but 3 is
            recommended if you want to have some actual level of pointing among the subsets of baselines. 
        '''
        try:
            if initial_hist is None:
                hist = numpy.zeros_like(self.mesh_azimuth_rad,dtype=int)
            else:
                hist = initial_hist.copy()

            all_theta_best = numpy.zeros(len(eventids))
            all_phi_best = numpy.zeros(len(eventids))
            if return_max_possible_map_values == True:
                all_max_possible_map_values = numpy.zeros(len(eventids))
            if return_peak_to_sidelobe == True:
                all_peak_to_sidelobes = numpy.zeros(len(eventids))
            if return_map_peaks == True:
                all_map_peaks = numpy.zeros(len(eventids))
            if iterate_sub_baselines is None:
                iterate_sub_baselines = len(include_baselines)
            elif iterate_sub_baselines > len(include_baselines):
                print('WARNING, include_baselines > len(include_baselines, setting to len(include_baselines)')
                iterate_sub_baselines = len(include_baselines)
            include_baselines_all =  numpy.array(list(itertools.combinations(include_baselines,iterate_sub_baselines))) #Every subset of baselines matching the specified length within the allowable set. 
                


            for event_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
                sys.stdout.flush()
                for _include_baselines in include_baselines_all:
                    if return_max_possible_map_values:
                        m, max_possible_map_value = self.map(eventid, pol, verbose=False, plot_map=False, plot_corr=False, include_baselines=_include_baselines, hilbert=hilbert, return_max_possible_map_value=True, shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)
                    else:
                        m = self.map(eventid, pol, verbose=False, plot_map=False, plot_corr=False, include_baselines=_include_baselines, hilbert=hilbert, return_max_possible_map_value=False, shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length, shorten_keep_leading=shorten_keep_leading)

                    if return_peak_to_sidelobe == True:
                        if max_method is not None:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3, peak_to_sidelobe = self.mapMax(m,max_method=max_method,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=return_peak_to_sidelobe)
                        else:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3, peak_to_sidelobe = self.mapMax(m,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol, return_peak_to_sidelobe=return_peak_to_sidelobe)

                        all_peak_to_sidelobes[event_index] = peak_to_sidelobe
                    else:
                        if max_method is not None:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(m,max_method=max_method,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol)
                        else:
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(m,verbose=False,zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, pol=pol)

                    all_theta_best[event_index] = theta_best
                    all_phi_best[event_index] = phi_best

                    if return_max_possible_map_values == True:
                        all_max_possible_map_values[event_index] = max_possible_map_value

                    if return_map_peaks == True:
                        all_map_peaks[event_index] = m.flat[linear_max_index]

                    if use_weight == True:
                        hist.flat[linear_max_index] += m.flat[linear_max_index]/len(eventids)
                    else:
                        hist.flat[linear_max_index] += 1

            #After all events completed, run once more on histogram.
            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = self.mapMax(hist,pol=pol, max_method=0)

            if initial_thetas is not None:
                all_theta_best = numpy.append(initial_thetas,all_theta_best)
            if initial_phis is not None:
                all_phi_best = numpy.append(initial_phis,all_phi_best)

            if return_max_possible_map_values:
                if initial_max_possible_map_values is not None:
                    all_max_possible_map_values = numpy.append(initial_max_possible_map_values,all_max_possible_map_values)
            if return_map_peaks == True:
                if initial_peaks is not None:
                    all_map_peaks = numpy.append(initial_peaks,all_map_peaks)
            if return_peak_to_sidelobe == True:
                if initial_peak_to_sidelobes is not None:
                    all_peak_to_sidelobes = numpy.append(initial_peak_to_sidelobes,all_peak_to_sidelobes)


            if plot_map:
                if numpy.logical_or(~numpy.all(numpy.isin(include_baselines, numpy.array([0,1,2,3,4,5]))),iterate_sub_baselines != len(include_baselines)):
                        add_text = '\nIncluded baselines (subsets of len %i) = '%iterate_sub_baselines + str(numpy.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])[include_baselines])
                else:
                    add_text = ''
                if center_dir.upper() == 'E':
                    center_dir_full = 'East'
                    azimuth_offset_rad = 0 #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = 0 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From East = 0 deg, North = 90 deg)' + add_text
                    roll = 0
                elif center_dir.upper() == 'N':
                    center_dir_full = 'North'
                    azimuth_offset_rad = numpy.pi/2 #This is subtracted from the xaxis to roll it effectively. 
                    azimuth_offset_deg = 90 #This is subtracted from the xaxis to roll it effectively. 
                    xlabel = 'Azimuth (From North = 0 deg, West = 90 deg)' + add_text
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))
                elif center_dir.upper() == 'W':
                    center_dir_full = 'West'
                    azimuth_offset_rad = numpy.pi #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = 180 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From West = 0 deg, South = 90 deg)' + add_text
                    roll = len(self.phis_rad)//2
                elif center_dir.upper() == 'S':
                    center_dir_full = 'South'
                    azimuth_offset_rad = -numpy.pi/2 #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = -90 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From South = 0 deg, East = 90 deg)' + add_text
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))

                rolled_values = numpy.roll(hist,roll,axis=1)
                if plot_map:
                    fig = plt.figure()

                    if window_title is None:
                        window_title = 'r%i %s Peak Correlation Map Hist'%(self.reader.run,pol.title())
                    fig.canvas.set_window_title(window_title)

                    if mollweide == True:
                        ax = fig.add_subplot(2,1,1, projection='mollweide')
                    else:
                        ax = fig.add_subplot(2,1,1)                    

                    if mollweide == True:
                        #Automatically converts from rads to degs
                        im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, rolled_values, vmin=0.1, vmax=numpy.max(rolled_values),cmap=plt.cm.Reds,norm=matplotlib.colors.LogNorm())
                    else:
                        im = ax.pcolormesh(self.mesh_azimuth_deg, self.mesh_elevation_deg, rolled_values, vmin=0.1, vmax=numpy.max(rolled_values),cmap=plt.cm.Reds,norm=matplotlib.colors.LogNorm())

                    mean_phi = numpy.mean(all_phi_best)
                    mean_theta = numpy.mean(all_theta_best)

                    stacked_variables = numpy.vstack((all_phi_best,all_theta_best)) #x,y
                    covariance_matrix = numpy.cov(stacked_variables)
                    sig_phi = numpy.sqrt(covariance_matrix[0,0])
                    sig_theta = numpy.sqrt(covariance_matrix[1,1])
                    rho_phi_theta = covariance_matrix[0,1]/(sig_phi*sig_theta)

                    #import pdb; pdb.set_trace()
                    
                    
                    cbar = fig.colorbar(im)
                    if use_weight == False:
                        cbar.set_label('Counts')
                    else:
                        cbar.set_label('Counts (Weighted)')

                    plt.xlabel('Azimuth Angle (Degrees)')
                    plt.ylabel('Elevation Angle (Degrees)')
                    plt.grid(True)



                    if plot_max:
                        #Added circles as specified.
                        if self.conference_mode == False:
                            ax, peak_circle = self.addCircleToMap(ax, mean_phi, 90.0 - mean_theta, azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius=radius, crosshair=True, return_circle=True, color='lime', linewidth=0.5,fill=False)
                        else:
                            pass

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
                                    ax, _circ = self.addCircleToMap(ax, _circle_az[i], 90.0-_circle_zenith[i], azimuth_offset_deg=azimuth_offset_deg, mollweide=mollweide, radius=radius, crosshair=True, return_circle=True, color='fuchsia', linewidth=0.5,fill=False)
                                    additional_circles.append(_circ)
                            plt.title('%s Difference from reconstructed \nand expected peaks is %0.2f'%( window_title,  numpy.sqrt( (_circle_az[i] - phi_best)**2 + (_circle_zenith[i] - theta_best)**2 ) ))
                            print('%s\nDifference in degrees from reconstructed and expected peaks is %0.2f'%( window_title,  numpy.sqrt( (_circle_az[i] - phi_best)**2 + (_circle_zenith[i] - theta_best)**2 ) ))
                            
                            if False:
                                string = 'Covariance Matrix:\n' + str(covariance_matrix) 
                                ax.text(0.45, 0.85, string, fontsize=12, horizontalalignment='center', verticalalignment='top',transform=plt.gcf().transFigure) #Need to reset the x and y here to be appropriate for the values in the plot. 
                            else:                                
                                if False:
                                    string = 'sig_phi = %0.3f\nsig_theta = %0.3f\nrho_phi,theta = %0.3f\nReconstruction Offset = %0.3f deg'%(sig_phi, sig_phi, rho_phi_theta, numpy.sqrt( (_circle_az[i] - phi_best)**2 + (_circle_zenith[i] - theta_best)**2 ) )
                                    ax.text(0.45, 0.85, string, fontsize=12, horizontalalignment='center', verticalalignment='top',transform=plt.gcf().transFigure) #Need to reset the x and y here to be appropriate for the values in the plot. 
                                else:
                                    string = '$\\sigma_\\phi$ = %0.3f\n$\\sigma_\\theta$ = %0.3f\n$\\rho_{\\phi,\\theta}$ = %0.3f\nReconstruction Offset = %0.3f deg'%(sig_phi, sig_theta, rho_phi_theta, numpy.sqrt( (_circle_az[i] - phi_best)**2 + (_circle_zenith[i] - theta_best)**2 ) )
                                    ax.text(0.45, 0.85, string, fontsize=14, horizontalalignment='center', verticalalignment='top',transform=plt.gcf().transFigure,usetex=True) #Need to reset the x and y here to be appropriate for the values in the plot. 

                    if self.conference_mode:
                        angular_range = 1
                        if mollweide == True:
                            angular_range = numpy.deg2rad(angular_range)
                        ax.set_xlim(mean_phi - angular_range, mean_phi + angular_range)
                        ax.set_ylim(90.0 - mean_theta - angular_range, 90.0 - mean_theta + angular_range)
                        plt.figure()
                        plt.subplot(1,2,1)
                    else:
                        plt.subplot(2,2,3)
                    plt.ylabel('Counts')
                    
                    if shift_1d_hists == True:
                        plt.xlabel('Azimuth Distribution (Degrees)\nCentered on Mean')
                        #ax.text(0.45, 0.85, 'Mean $\\phi$$ = %0.3f\n$\\sigma_\\phi$$ = %0.3f'%(mean_phi,sig_phi), fontsize=14, horizontalalignment='center', verticalalignment='top',transform=plt.gcf().transFigure,usetex=True) #Need to reset the x and y here to be appropriate for the values in the plot. 
                        phi_n, phi_bins, phi_patches = plt.hist(all_phi_best - mean_phi, bins=self.phis_deg-mean_phi, log=False, edgecolor='black', linewidth=1.0,label='Mean = %0.3f\nSigma = %0.3f'%(mean_phi,sig_phi),density=False)
                        
                        #Fit Gaussian
                        x = (phi_bins[1:len(phi_bins)] + phi_bins[0:len(phi_bins)-1] )/2.0
                        if acceptable_fit_range is not None:
                            x_cut = numpy.abs(x - numpy.mean(all_phi_best - mean_phi)) <= acceptable_fit_range
                        else:
                            x_cut = numpy.ones(len(x),dtype=bool)

                        try:
                            popt, pcov = curve_fit(gaus,x[x_cut],phi_n[x_cut],p0=[numpy.max(phi_n[x_cut]),numpy.mean(all_phi_best - mean_phi),1.5*numpy.std(all_phi_best - mean_phi)])
                            popt[2] = abs(popt[2]) #I want positive sigma.

                            if acceptable_fit_range is not None:
                                plt.axvline(min(x[x_cut]),linestyle='-',c='b',label='Range Considered in Fit %0.2f +- %0.2f'%(mean_phi,acceptable_fit_range))
                                plt.axvline(max(x[x_cut]),linestyle='-',c='b')
                                plt.xlim(-4.0, 4.0)
                                #plt.xlim(min(x[x_cut]) - 1.0, max(x[x_cut]) + 1.0)

                            plot_x = numpy.linspace(min(x[x_cut]),max(x[x_cut]),1000)
                            plt.plot(plot_x,gaus(plot_x,*popt),'-',c='k',label='Fit Sigma = %f ns'%popt[2])
                            plt.axvline(popt[1],linestyle='-',c='k',label='Fit Center = %f'%(popt[1] + mean_phi))
                        except Exception as e:
                            #print('Failed to fit histogram')
                            #print(e)
                            #print('Trying to add info without fit.')
                            try:
                                if acceptable_fit_range is not None:
                                    range_cut = numpy.abs(numpy.mean(all_phi_best) - all_phi_best) <= acceptable_fit_range
                                    plt.axvline(numpy.mean(all_phi_best[range_cut]),linestyle='-',c='k',label='Fit Failed\nstd = %f ns\nmean= %f ns\nRange Considered in Fit %0.2f +- %0.2f'%(numpy.std(all_phi_best[range_cut]) , numpy.mean(all_phi_best[range_cut]), mean_phi,acceptable_fit_range))
                                else:
                                    plt.axvline(numpy.mean(all_phi_best),linestyle='-',c='k',label='Fit Failed\nstd = %f ns\nmean= %f ns'%(numpy.std(all_phi_best) , numpy.mean(all_phi_best)))    
                            except Exception as e:
                                #print('Failed here too.')
                                print(e)                         

                        if circle_az is not None:
                            if len(circle_az) == 1:
                                plt.axvline(circle_az[0] - mean_phi,color='fuchsia',label='Highlighted Azimuth')



                    else:
                        plt.xlabel('Azimuth Distribution (Degrees)')
                        #ax.text(0.45, 0.85, 'Mean $\\phi$$ = %0.3f\n$\\sigma_\\phi$$ = %0.3f'%(mean_phi,sig_phi), fontsize=14, horizontalalignment='center', verticalalignment='top',transform=plt.gcf().transFigure,usetex=True) #Need to reset the x and y here to be appropriate for the values in the plot. 
                        phi_n, phi_bins, phi_patches = plt.hist(all_phi_best, bins=self.phis_deg, log=False, edgecolor='black', linewidth=1.0,label='Mean = %0.3f\nSigma = %0.3f'%(mean_phi,sig_phi),density=False)
                        
                        #Fit Gaussian
                        x = (phi_bins[1:len(phi_bins)] + phi_bins[0:len(phi_bins)-1] )/2.0


                        try:
                            if acceptable_fit_range is not None:
                                x_cut = numpy.abs(x - numpy.mean(all_phi_best)) <= acceptable_fit_range
                            else:
                                x_cut = numpy.ones(len(x),dtype=bool)

                            popt, pcov = curve_fit(gaus,x[x_cut],phi_n[x_cut],p0=[numpy.max(phi_n[x_cut]),numpy.mean(all_phi_best),1.5*numpy.std(all_phi_best)])
                            popt[2] = abs(popt[2]) #I want positive sigma.
                            
                            if acceptable_fit_range is not None:
                                plt.axvline(min(x[x_cut]),linestyle='-',c='b',label='Range Considered in Fit %0.2f +- %0.2f'%(mean_phi,acceptable_fit_range))
                                plt.axvline(max(x[x_cut]),linestyle='-',c='b')
                                plt.xlim(mean_phi - 4.0, mean_phi + 4.0)
                                #plt.xlim(min(x[x_cut]) - 1.0, max(x[x_cut]) + 1.0)

                            plot_x = numpy.linspace(min(x[x_cut]),max(x[x_cut]),1000)
                            plt.plot(plot_x,gaus(plot_x,*popt),'-',c='k',label='Fit Sigma = %f ns'%popt[2])
                            plt.axvline(popt[1],linestyle='-',c='k',label='Fit Center = %f'%popt[1])
                            

                        except Exception as e:
                            print('Failed to fit histogram')
                            print(e)
                            print('Trying to add info without fit.')
                            try:
                                if acceptable_fit_range is not None:
                                    range_cut = numpy.abs(numpy.mean(all_phi_best) - all_phi_best) <= acceptable_fit_range
                                    plt.axvline(popt[1],linestyle='-',c='k',label='Fit Failed\nstd = %f ns\nmean= %f ns\nRange Considered in Fit %0.2f +- %0.2f'%(numpy.std(all_phi_best[range_cut]) , numpy.mean(all_phi_best[range_cut]), mean_phi,acceptable_fit_range))
                                else:
                                    plt.axvline(popt[1],linestyle='-',c='k',label='Fit Failed\nstd = %f ns\nmean= %f ns'%(numpy.std(all_phi_best) , numpy.mean(all_phi_best)))
                            except Exception as e:
                                print('Failed here too.')
                                print(e)
                                
                        if circle_az is not None:
                            if len(circle_az) == 1:
                                plt.axvline(circle_az[0],color='fuchsia',label='Highlighted Azimuth')

                    plt.legend(loc = 'upper right',fontsize=10)
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                    if self.conference_mode:
                        plt.subplot(1,2,2)
                    else:
                        plt.subplot(2,2,4)
                    #plt.ylabel('Counts (PDF)')
                    if shift_1d_hists == True:
                        plt.xlabel('Zenith Distribution (Degrees)\nCentered on Mean')
                        #ax.text(0.45, 0.85, 'Mean $\\theta$$ = %0.3f\n$\\sigma_\\theta$$ = %0.3f'%(mean_theta,sig_theta), fontsize=14, horizontalalignment='center', verticalalignment='top',transform=plt.gcf().transFigure,usetex=True) #Need to reset the x and y here to be appropriate for the values in the plot. 
                        theta_n, theta_bins, theta_patches = plt.hist(all_theta_best - mean_theta, bins=self.thetas_deg-mean_theta, log=False, edgecolor='black', linewidth=1.0,label='Mean = %0.3f\nSigma = %0.3f'%(mean_theta,sig_theta),density=False)
                        
                        #Fit Gaussian
                        x = (theta_bins[1:len(theta_bins)] + theta_bins[0:len(theta_bins)-1] )/2.0
                        if acceptable_fit_range is not None:
                            x_cut = numpy.abs(x - numpy.mean(all_theta_best - mean_theta)) <= acceptable_fit_range
                            #plt.xlim(min(x[x_cut]) - 1.0, max(x[x_cut]) + 1.0)
                        else:
                            x_cut = numpy.ones(len(x),dtype=bool)

                        try:
                            popt, pcov = curve_fit(gaus,x[x_cut],theta_n[x_cut],p0=[numpy.max(theta_n[x_cut]),numpy.mean(all_theta_best - mean_theta),1.5*numpy.std(all_theta_best - mean_theta)])
                            popt[2] = abs(popt[2]) #I want positive sigma.

                            if acceptable_fit_range is not None:
                                plt.axvline(min(x[x_cut]),linestyle='-',c='b',label='Range Considered in Fit %0.2f +- %0.2f'%(mean_theta,acceptable_fit_range))
                                plt.axvline(max(x[x_cut]),linestyle='-',c='b')
                                plt.xlim(-4.0, 4.0)

                            plot_x = numpy.linspace(min(x[x_cut]),max(x[x_cut]),1000)
                            plt.plot(plot_x,gaus(plot_x,*popt),'-',c='k',label='Fit Sigma = %f ns'%popt[2])
                            plt.axvline(popt[1],linestyle='-',c='k',label='Fit Center = %f'%(popt[1] + mean_theta))
                        except Exception as e:
                            print('Failed to fit histogram')
                            print(e)
                            print('Trying to add info without fit.')
                            try:
                                if acceptable_fit_range is not None:
                                    range_cut = numpy.abs(numpy.mean(all_theta_best) - all_theta_best) <= acceptable_fit_range
                                    plt.axvline(popt[1],linestyle='-',c='k',label='Fit Failed\nstd = %f ns\nmean= %f ns\nRange Considered in Fit %0.2f +- %0.2f'%(numpy.std(all_theta_best[range_cut]) , numpy.mean(all_theta_best[range_cut]), mean_theta,acceptable_fit_range))
                                else:
                                    plt.axvline(popt[1],linestyle='-',c='k',label='Fit Failed\nstd = %f ns\nmean= %f ns'%(numpy.std(all_theta_best) , numpy.mean(all_theta_best)))
                            except Exception as e:
                                print('Failed here too.')
                                print(e)
                            # x = numpy.linspace(min(self.thetas_deg-mean_theta),max(self.thetas_deg-mean_theta),10*self.n_theta)                               
                        
                        # y = scipy.stats.norm.pdf(x,0,sig_theta)
                        # #y = y*len(all_theta_best)/numpy.sum(y) # y*numpy.diff(self.thetas_deg)[0]*len(all_theta_best)
                        # y = y*numpy.diff(self.thetas_deg)[0]*len(all_theta_best)

                        # plt.plot(x,scipy.stats.norm.pdf(x,0,sig_theta)*numpy.diff(self.thetas_deg)[0]*len(all_theta_best),label='Gaussian Fit')
                        
                        #plt.xlim(min(all_theta_best - mean_theta) - 1.0,max(all_theta_best - mean_theta) + 1.0)
                        if circle_zenith is not None:
                            if len(circle_zenith) == 1:
                                plt.axvline(circle_zenith[0] - mean_theta,color='fuchsia',label='Highlighted Zenith')

                    else:
                        plt.xlabel('Zenith Distribution (Degrees)')
                        #ax.text(0.45, 0.85, 'Mean $\\theta$$ = %0.3f\n$\\sigma_\\theta$$ = %0.3f'%(mean_theta,sig_theta), fontsize=14, horizontalalignment='center', verticalalignment='top',transform=plt.gcf().transFigure,usetex=True) #Need to reset the x and y here to be appropriate for the values in the plot. 
                        theta_n, theta_bins, theta_patches = plt.hist(all_theta_best, bins=self.thetas_deg, log=False, edgecolor='black', linewidth=1.0,label='Mean = %0.3f\nSigma = %0.3f'%(mean_theta,sig_theta),density=False)
                        
                        #Fit Gaussian
                        x = (theta_bins[1:len(theta_bins)] + theta_bins[0:len(theta_bins)-1] )/2.0
                        if acceptable_fit_range is not None:
                            x_cut = numpy.abs(x - numpy.mean(all_theta_best)) <= acceptable_fit_range
                        else:
                            x_cut = numpy.ones(len(x),dtype=bool)

                        try:
                            popt, pcov = curve_fit(gaus,x[x_cut],theta_n[x_cut],p0=[numpy.max(theta_n[x_cut]),numpy.mean(all_theta_best),1.5*numpy.std(all_theta_best)])
                            popt[2] = abs(popt[2]) #I want positive sigma.

                            if acceptable_fit_range is not None:
                                plt.axvline(min(x[x_cut]),linestyle='-',c='b',label='Range Considered in Fit %0.2f +- %0.2f'%(mean_theta,acceptable_fit_range))
                                plt.axvline(max(x[x_cut]),linestyle='-',c='b')
                                plt.xlim(mean_theta - 4.0, mean_theta + 4.0)
                                #plt.xlim(min(x[x_cut]) - 1.0, max(x[x_cut]) + 1.0)

                            plot_x = numpy.linspace(min(x[x_cut]),max(x[x_cut]),1000)
                            plt.plot(plot_x,gaus(plot_x,*popt),'-',c='k',label='Fit Sigma = %f ns'%popt[2])
                            plt.axvline(popt[1],linestyle='-',c='k',label='Fit Center = %f'%popt[1])
                        except Exception as e:
                            print('Failed to fit histogram')
                            print(e)
                            print('Trying to add info without fit.')
                            try:
                                if acceptable_fit_range is not None:
                                    range_cut = numpy.abs(numpy.mean(all_theta_best) - all_theta_best) <= acceptable_fit_range
                                    plt.axvline(popt[1],linestyle='-',c='k',label='Fit Failed\nstd = %f ns\nmean= %f ns\nRange Considered in Fit %0.2f +- %0.2f'%(numpy.std(all_theta_best[range_cut]) , numpy.mean(all_theta_best[range_cut]), mean_theta,acceptable_fit_range))
                                else:
                                    plt.axvline(popt[1],linestyle='-',c='k',label='Fit Failed\nstd = %f ns\nmean= %f ns'%(numpy.std(all_theta_best) , numpy.mean(all_theta_best)))
                            except Exception as e:
                                print('Failed here too.')
                                print(e)
                                
                        # x = numpy.linspace(min(self.thetas_deg),max(self.thetas_deg),10*self.n_theta)

                        # y = scipy.stats.norm.pdf(x,mean_theta,sig_theta)
                        # #y = y*len(all_theta_best)/numpy.sum(y) # y*numpy.diff(self.thetas_deg)[0]*len(all_theta_best)
                        # y = y*numpy.diff(self.thetas_deg)[0]*len(all_theta_best)

                        # plt.plot(x,y,label='Gaussian Fit')
                        #plt.xlim(min(all_theta_best) - 1.0,max(all_theta_best) + 1.0)
                        if circle_zenith is not None:
                            if len(circle_zenith) == 1:
                                plt.axvline(circle_zenith[0],color='fuchsia',label='Highlighted Zenith')

                    
                    plt.legend(loc = 'upper right',fontsize=10)
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)

            if plot_map == True and return_fig == True:
                if return_max_possible_map_values == True:
                    if return_map_peaks == True:
                        if return_peak_to_sidelobe == True:
                            return hist, all_phi_best, all_theta_best, all_max_possible_map_values, all_map_peaks, all_peak_to_sidelobes, fig
                        else:
                            return hist, all_phi_best, all_theta_best, all_max_possible_map_values, all_map_peaks, fig
                    else:
                        if return_peak_to_sidelobe == True:
                            return hist, all_phi_best, all_theta_best, all_max_possible_map_values, all_peak_to_sidelobes, fig
                        else:
                            return hist, all_phi_best, all_theta_best, all_max_possible_map_values, fig
                else:
                    if return_map_peaks == True:
                        if return_peak_to_sidelobe == True:
                            return hist, all_phi_best, all_theta_best, all_map_peaks, all_peak_to_sidelobes, fig
                        else:
                            return hist, all_phi_best, all_theta_best, all_map_peaks, fig
                    else:
                        if return_peak_to_sidelobe == True:
                            return hist, all_phi_best, all_theta_best, all_peak_to_sidelobes, fig
                        else:
                            return hist, all_phi_best, all_theta_best, fig
            else:
                if return_max_possible_map_values == True:
                    if return_map_peaks == True:
                        if return_peak_to_sidelobe == True:
                            return hist, all_phi_best, all_theta_best, all_max_possible_map_values, all_map_peaks, all_peak_to_sidelobes
                        else:
                            return hist, all_phi_best, all_theta_best, all_max_possible_map_values, all_map_peaks
                    else:
                        if return_peak_to_sidelobe == True:
                            return hist, all_phi_best, all_theta_best, all_max_possible_map_values, all_peak_to_sidelobes
                        else:
                            return hist, all_phi_best, all_theta_best, all_max_possible_map_values
                else:
                    if return_map_peaks == True:
                        if return_peak_to_sidelobe == True:
                            return hist, all_phi_best, all_theta_best, all_map_peaks, all_peak_to_sidelobes
                        else:
                            return hist, all_phi_best, all_theta_best, all_map_peaks
                    else:
                        if return_peak_to_sidelobe == True:
                            return hist, all_phi_best, all_theta_best, all_peak_to_sidelobes
                        else:
                            return hist, all_phi_best, all_theta_best
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def generateTimeDelayOverlapMap(self, pol, time_delay_dict, window_ns, value_mode='distance', plot_map=False, mollweide=False,center_dir='E',window_title=None, include_baselines=[0,1,2,3,4,5]):
        '''
        For this given time delays (to be filtered to match the specified included baselines), this will generate a map
        that can be used to determine if the given time delays are capable overlapping given the current calibration.
        It does so by creating a weighted window around the given time delays.  Time delays within the window will be
        given a value of 1 (or some other weighting as specified).  This is done for each baseline, then the maps are
        summed.  If there is a point on that map that has a value equal to the total number of baselines specified, then
        that can be interpreted as a point in the sky where all of the given time delays overlap (within the specified
        window/tolerance).

        Parameters
        ----------

        pol : str
            The polarization you wish to plot.  Options: 'hpol', 'vpol', 'both'
        time_delay_dict : dict of list of floats
            The first level of the dict should specify 'hpol' and/or 'vpol'
            The following key within should have each of the baseline pairs that you wish to plot.  Each of these
            will correspond to a list containing a single float that is the time delay for that baseline.
        window : float
            Given in ns, this represents the one sided window for which time delays will be considered as matching. 
            For example a map of expected time delays coming from each source direction defined as "map_baseline" will
            undergo the calculation numpy.logical_and(numpy.abs(map_baseline - time_delay_baseline) < window_ns ).
        plot_map : bool
            Whether to actually plot the results.  
        '''
        try:
            if pol == 'hpol':
                all_antennas = numpy.vstack((self.A0_hpol,self.A1_hpol,self.A2_hpol,self.A3_hpol))
                time_delays = time_delay_dict['hpol']
            elif pol == 'vpol':
                all_antennas = numpy.vstack((self.A0_vpol,self.A1_vpol,self.A2_vpol,self.A3_vpol))
                time_delays = time_delay_dict['vpol']

            pairs = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
            baseline_cm = plt.cm.get_cmap('tab10', 10)
            baseline_colors = baseline_cm(numpy.linspace(0, 1, 10))[0:6] #only want the first 6 colours in this list of 10 colors.

            if value_mode == 'box':
                overlap_map = numpy.zeros_like(self.mesh_azimuth_deg,dtype=int)
            elif value_mode == 'distance':
                overlap_map = numpy.zeros_like(self.mesh_azimuth_deg,dtype=float)
            elif value_mode == 'gaus':
                overlap_map = numpy.zeros_like(self.mesh_azimuth_deg,dtype=float)


            for pair_key, pair_time_delay in time_delays.items():
                for time_delay in pair_time_delay: 
                    #time_delay = pair_time_delay[0]

                    pair = numpy.array(pair_key.replace('[','').replace(']','').split(','),dtype=int)
                    pair_index = numpy.where(numpy.sum(pair == pairs,axis=1) == 2)[0][0] #Used for consistent coloring.

                    if numpy.isin(pair_index ,include_baselines ):
                        i = pair[0]
                        j = pair[1]

                        #Attempting to use precalculate expected time delays per direction to derive theta here.
                        if value_mode == 'box':
                            if pol == 'hpol':
                                if pair_index == 0:
                                    overlap_map +=  (numpy.abs(self.t_hpol_0subtract1 - time_delay) < window_ns).astype(int)
                                elif pair_index == 1:
                                    overlap_map +=  (numpy.abs(self.t_hpol_0subtract2 - time_delay) < window_ns).astype(int)
                                elif pair_index == 2:
                                    overlap_map +=  (numpy.abs(self.t_hpol_0subtract3 - time_delay) < window_ns).astype(int)
                                elif pair_index == 3:
                                    overlap_map +=  (numpy.abs(self.t_hpol_1subtract2 - time_delay) < window_ns).astype(int)
                                elif pair_index == 4:
                                    overlap_map +=  (numpy.abs(self.t_hpol_1subtract3 - time_delay) < window_ns).astype(int)
                                elif pair_index == 5:
                                    overlap_map +=  (numpy.abs(self.t_hpol_2subtract3 - time_delay) < window_ns).astype(int)
                            else:
                                if pair_index == 0:
                                    overlap_map +=  (numpy.abs(self.t_vpol_0subtract1 - time_delay) < window_ns).astype(int)
                                elif pair_index == 1:
                                    overlap_map +=  (numpy.abs(self.t_vpol_0subtract2 - time_delay) < window_ns).astype(int)
                                elif pair_index == 2:
                                    overlap_map +=  (numpy.abs(self.t_vpol_0subtract3 - time_delay) < window_ns).astype(int)
                                elif pair_index == 3:
                                    overlap_map +=  (numpy.abs(self.t_vpol_1subtract2 - time_delay) < window_ns).astype(int)
                                elif pair_index == 4:
                                    overlap_map +=  (numpy.abs(self.t_vpol_1subtract3 - time_delay) < window_ns).astype(int)
                                elif pair_index == 5:
                                    overlap_map +=  (numpy.abs(self.t_vpol_2subtract3 - time_delay) < window_ns).astype(int)
                        elif value_mode == 'distance':
                            if pol == 'hpol':
                                if pair_index == 0:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_hpol_0subtract1 - time_delay), (numpy.abs(self.t_hpol_0subtract1 - time_delay) < window_ns).astype(int))
                                elif pair_index == 1:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_hpol_0subtract2 - time_delay), (numpy.abs(self.t_hpol_0subtract2 - time_delay) < window_ns).astype(int))
                                elif pair_index == 2:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_hpol_0subtract3 - time_delay), (numpy.abs(self.t_hpol_0subtract3 - time_delay) < window_ns).astype(int))
                                elif pair_index == 3:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_hpol_1subtract2 - time_delay), (numpy.abs(self.t_hpol_1subtract2 - time_delay) < window_ns).astype(int))
                                elif pair_index == 4:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_hpol_1subtract3 - time_delay), (numpy.abs(self.t_hpol_1subtract3 - time_delay) < window_ns).astype(int))
                                elif pair_index == 5:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_hpol_2subtract3 - time_delay), (numpy.abs(self.t_hpol_2subtract3 - time_delay) < window_ns).astype(int))
                            else:
                                if pair_index == 0:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_vpol_0subtract1 - time_delay), (numpy.abs(self.t_vpol_0subtract1 - time_delay) < window_ns).astype(int))
                                elif pair_index == 1:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_vpol_0subtract2 - time_delay), (numpy.abs(self.t_vpol_0subtract2 - time_delay) < window_ns).astype(int))
                                elif pair_index == 2:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_vpol_0subtract3 - time_delay), (numpy.abs(self.t_vpol_0subtract3 - time_delay) < window_ns).astype(int))
                                elif pair_index == 3:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_vpol_1subtract2 - time_delay), (numpy.abs(self.t_vpol_1subtract2 - time_delay) < window_ns).astype(int))
                                elif pair_index == 4:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_vpol_1subtract3 - time_delay), (numpy.abs(self.t_vpol_1subtract3 - time_delay) < window_ns).astype(int))
                                elif pair_index == 5:
                                    overlap_map +=  numpy.multiply(window_ns - numpy.abs(self.t_vpol_2subtract3 - time_delay), (numpy.abs(self.t_vpol_2subtract3 - time_delay) < window_ns).astype(int))
                        elif value_mode == 'gaus':
                            #Here window will be treated as FWHM
                            sigma = window_ns / 2.355 #conversion to standard deviation.
                            scale_factor = 1# Don't want normalized area, want normalized peak value.  #1/(sigma * numpy.sqrt(2*numpy.pi)) #So only compute once.
                            denom = 2*(sigma**2)
                            g = lambda x, x0 : scale_factor * numpy.exp(-((x-x0)**2)/denom)

                            if pol == 'hpol':
                                if pair_index == 0:
                                    overlap_map +=  g(self.t_hpol_0subtract1, time_delay)
                                elif pair_index == 1:
                                    overlap_map +=  g(self.t_hpol_0subtract2, time_delay)
                                elif pair_index == 2:
                                    overlap_map +=  g(self.t_hpol_0subtract3, time_delay)
                                elif pair_index == 3:
                                    overlap_map +=  g(self.t_hpol_1subtract2, time_delay)
                                elif pair_index == 4:
                                    overlap_map +=  g(self.t_hpol_1subtract3, time_delay)
                                elif pair_index == 5:
                                    overlap_map +=  g(self.t_hpol_2subtract3, time_delay)
                            else:
                                if pair_index == 0:
                                    overlap_map +=  g(self.t_vpol_0subtract1, time_delay)
                                elif pair_index == 1:
                                    overlap_map +=  g(self.t_vpol_0subtract2, time_delay)
                                elif pair_index == 2:
                                    overlap_map +=  g(self.t_vpol_0subtract3, time_delay)
                                elif pair_index == 3:
                                    overlap_map +=  g(self.t_vpol_1subtract2, time_delay)
                                elif pair_index == 4:
                                    overlap_map +=  g(self.t_vpol_1subtract3, time_delay)
                                elif pair_index == 5:
                                    overlap_map +=  g(self.t_vpol_2subtract3, time_delay)



            if plot_map:
                if ~numpy.all(numpy.isin(include_baselines, numpy.array([0,1,2,3,4,5]))):
                    add_text = '\nIncluded baselines = ' + str(numpy.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])[include_baselines])
                else:
                    add_text = ''

                if center_dir.upper() == 'E':
                    center_dir_full = 'East'
                    azimuth_offset_rad = 0 #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = 0 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From East = 0 deg, North = 90 deg)' + add_text
                    roll = 0
                elif center_dir.upper() == 'N':
                    center_dir_full = 'North'
                    azimuth_offset_rad = numpy.pi/2 #This is subtracted from the xaxis to roll it effectively. 
                    azimuth_offset_deg = 90 #This is subtracted from the xaxis to roll it effectively. 
                    xlabel = 'Azimuth (From North = 0 deg, West = 90 deg)' + add_text
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))
                elif center_dir.upper() == 'W':
                    center_dir_full = 'West'
                    azimuth_offset_rad = numpy.pi #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = 180 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From West = 0 deg, South = 90 deg)' + add_text
                    roll = len(self.phis_rad)//2
                elif center_dir.upper() == 'S':
                    center_dir_full = 'South'
                    azimuth_offset_rad = -numpy.pi/2 #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = -90 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From South = 0 deg, East = 90 deg)' + add_text
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))

                rolled_values = numpy.roll(overlap_map,roll,axis=1)

                fig = plt.figure()
                if window_title is None:
                    fig.canvas.set_window_title('Overlap Map')
                else:
                    fig.canvas.set_window_title(window_title)
                if mollweide == True:
                    ax = fig.add_subplot(1,1,1, projection='mollweide')
                else:
                    ax = fig.add_subplot(1,1,1)                    

                if mollweide == True:
                    #Automatically converts from rads to degs
                    im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, rolled_values, vmin=numpy.min(rolled_values), vmax=numpy.max(rolled_values),cmap=plt.cm.coolwarm)
                else:
                    im = ax.pcolormesh(self.mesh_azimuth_deg, self.mesh_elevation_deg, rolled_values, vmin=numpy.min(rolled_values), vmax=numpy.max(rolled_values),cmap=plt.cm.coolwarm)

                cbar = fig.colorbar(im)
                cbar.set_label('N Overlapping Time Delays')
                plt.xlabel(xlabel,fontsize=18)
                plt.ylabel('Elevation Angle (Degrees)',fontsize=18)
                plt.grid(True)
                im, ax = self.addTimeDelayCurves(im, time_delay_dict, pol, ax, mollweide=mollweide, azimuth_offset_deg=azimuth_offset_deg, include_baselines=include_baselines)
            if plot_map == True:
                return self.mesh_azimuth_deg, self.mesh_elevation_deg, overlap_map, im, ax
            else:
                return self.mesh_azimuth_deg, self.mesh_elevation_deg, overlap_map
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def plotPointingResolution(self, pol, snr=5, bw=50e6, plot_map=True, mollweide=False,center_dir='E', window_title=None, include_baselines=[0,1,2,3,4,5]):
        '''
        This will use the equation dTheta ~ c/L*SNR*bw to estimate the bandwith of the array in various directions.
        '''
        try:
            if pol == 'hpol':
                baseline_vectors = numpy.array([    self.A0_hpol - self.A1_hpol,\
                                                    self.A0_hpol - self.A2_hpol,\
                                                    self.A0_hpol - self.A3_hpol,\
                                                    self.A1_hpol - self.A2_hpol,\
                                                    self.A1_hpol - self.A3_hpol,\
                                                    self.A2_hpol - self.A3_hpol])
            elif pol == 'vpol':
                baseline_vectors = numpy.array([    self.A0_vpol - self.A1_vpol,\
                                                    self.A0_vpol - self.A2_vpol,\
                                                    self.A0_vpol - self.A3_vpol,\
                                                    self.A1_vpol - self.A2_vpol,\
                                                    self.A1_vpol - self.A3_vpol,\
                                                    self.A2_vpol - self.A3_vpol])

            pairs = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
            baseline_cm = plt.cm.get_cmap('tab10', 10)
            baseline_colors = baseline_cm(numpy.linspace(0, 1, 10))[0:6] #only want the first 6 colours in this list of 10 colors.

            thetas  = numpy.tile(self.thetas_rad,(self.n_phi,1)).T  #Each row is a different theta (zenith)
            phis    = numpy.tile(self.phis_rad,(self.n_theta,1))    #Each column is a different phi (azimuth)


            #To get the resolution I need baseline in each perceived direction.  To get that I need to dot with the unit vectors.
            theta_hat = numpy.zeros((self.n_theta, self.n_phi, 3))
            theta_hat[:,:,0] = numpy.cos(thetas)*numpy.cos(phis)
            theta_hat[:,:,1] = numpy.cos(thetas)*numpy.sin(phis)
            theta_hat[:,:,2] = -numpy.sin(thetas)

            phi_hat = numpy.zeros((self.n_theta, self.n_phi, 3))
            phi_hat[:,:,0] = -numpy.sin(phis)
            phi_hat[:,:,1] = numpy.cos(phis)
            #phi_hat[:,:,2] = zeros

            '''
            Dot each baseline with the unit vector to get the percieved baseline in that direction.  Take max among baselines.
            '''
            max_theta_baseline = numpy.zeros_like(phis) #Just for shape
            max_phi_baseline = numpy.zeros_like(phis) #Just for shape
            for pair_index in include_baselines:
                i = pairs[pair_index][0]
                j = pairs[pair_index][1]

                dot_values = numpy.abs(baseline_vectors[pair_index][0]*theta_hat[:,:,0] + baseline_vectors[pair_index][1]*theta_hat[:,:,1] + baseline_vectors[pair_index][2]*theta_hat[:,:,2]) #m
                cut = dot_values > max_theta_baseline
                max_theta_baseline[cut] = dot_values[cut]

                #import pdb; pdb.set_trace()
                dot_values = numpy.abs(baseline_vectors[pair_index][0]*phi_hat[:,:,0] + baseline_vectors[pair_index][1]*phi_hat[:,:,1] + baseline_vectors[pair_index][2]*phi_hat[:,:,2]) #m
                cut = dot_values > max_phi_baseline
                max_phi_baseline[cut] = dot_values[cut]


            debug = False
            if debug:
                theta_resolution = max_theta_baseline
                phi_resolution = max_phi_baseline
            else:
                theta_resolution = numpy.rad2deg(self.c / (snr * bw * max_theta_baseline)) #theta_hat[:,:,2]
                phi_resolution = numpy.rad2deg(self.c / (snr * bw * max_phi_baseline)) #phi_hat[:,:,0]
                reality_cut = phi_resolution > 10#360.0
                phi_resolution[reality_cut] = 10#360.0
                reality_cut = theta_resolution > 10#90.0
                theta_resolution[reality_cut] = 10#90.0



            if plot_map:
                if ~numpy.all(numpy.isin(include_baselines, numpy.array([0,1,2,3,4,5]))):
                    add_text = '\nIncluded baselines = ' + str(numpy.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])[include_baselines])
                else:
                    add_text = ''

                if center_dir.upper() == 'E':
                    center_dir_full = 'East'
                    azimuth_offset_rad = 0 #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = 0 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From East = 0 deg, North = 90 deg)' + add_text
                    roll = 0
                elif center_dir.upper() == 'N':
                    center_dir_full = 'North'
                    azimuth_offset_rad = numpy.pi/2 #This is subtracted from the xaxis to roll it effectively. 
                    azimuth_offset_deg = 90 #This is subtracted from the xaxis to roll it effectively. 
                    xlabel = 'Azimuth (From North = 0 deg, West = 90 deg)' + add_text
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))
                elif center_dir.upper() == 'W':
                    center_dir_full = 'West'
                    azimuth_offset_rad = numpy.pi #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = 180 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From West = 0 deg, South = 90 deg)' + add_text
                    roll = len(self.phis_rad)//2
                elif center_dir.upper() == 'S':
                    center_dir_full = 'South'
                    azimuth_offset_rad = -numpy.pi/2 #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = -90 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From South = 0 deg, East = 90 deg)' + add_text
                    roll = numpy.argmin(abs(self.phis_rad - azimuth_offset_rad))

                rolled_values_theta = numpy.roll(theta_resolution,roll,axis=1)
                rolled_values_phi = numpy.roll(phi_resolution,roll,axis=1)
                same_colors = False

                if same_colors:
                    vmin = numpy.min((numpy.min(rolled_values_phi),numpy.min(rolled_values_theta)))
                    vmax = numpy.max((numpy.max(rolled_values_phi),numpy.max(rolled_values_theta)))
                else:
                    #Just phi right now
                    vmin = numpy.min(rolled_values_phi)
                    vmax = numpy.max(rolled_values_phi)

                #import pdb; pdb.set_trace()

                fig = plt.figure()
                if window_title is None:
                    fig.canvas.set_window_title('%s Resolution Map'%(pol.title()))
                else:
                    fig.canvas.set_window_title(window_title)

                plt.suptitle('%s Resolution Map\nBW = %0.2f MHz, SNR = %0.1f sigma'%(pol.title(),bw/1e6,snr))

                if mollweide == True:
                    ax = fig.add_subplot(1,2,1, projection='mollweide')
                else:
                    ax = fig.add_subplot(1,2,1)                    

                if mollweide == True:
                    #Automatically converts from rads to degs
                    im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, rolled_values_phi, vmin=vmin, vmax=vmax,cmap=plt.cm.coolwarm)
                else:
                    im = ax.pcolormesh(self.mesh_azimuth_deg, self.mesh_elevation_deg, rolled_values_phi, vmin=vmin, vmax=vmax,cmap=plt.cm.coolwarm)

                cbar = fig.colorbar(im)
                cbar.set_label('Azimuthal Resolution (Deg)')
                plt.xlabel(xlabel,fontsize=18)
                plt.ylabel('Elevation Angle (Degrees)',fontsize=18)
                plt.grid(True)

                #Prepare center line and plot the map.  Prep cut lines as well.
                if pol == 'hpol':
                    selection_index = 1
                elif pol == 'vpol':
                    selection_index = 2 
                plane_xy = self.getArrayPlaneZenithCurves(90.0, azimuth_offset_deg=azimuth_offset_deg)[selection_index]
                im = self.addCurveToMap(im, plane_xy,  mollweide=mollweide, linewidth = self.min_elevation_linewidth, color='k')


                if mollweide == True:
                    ax = fig.add_subplot(1,2,2, projection='mollweide')
                else:
                    ax = fig.add_subplot(1,2,2)                    

                if same_colors == False:
                    #Just theta right now
                    vmin = numpy.min(rolled_values_theta)
                    vmax = numpy.max(rolled_values_theta)



                if mollweide == True:
                    #Automatically converts from rads to degs
                    im = ax.pcolormesh(self.mesh_azimuth_rad, self.mesh_elevation_rad, rolled_values_theta, vmin=vmin, vmax=vmax,cmap=plt.cm.coolwarm)
                else:
                    im = ax.pcolormesh(self.mesh_azimuth_deg, self.mesh_elevation_deg, rolled_values_theta, vmin=vmin, vmax=vmax,cmap=plt.cm.coolwarm)

                cbar = fig.colorbar(im)
                cbar.set_label('Elevation Resolution (Deg)')
                plt.xlabel(xlabel,fontsize=18)
                plt.ylabel('Elevation Angle (Degrees)',fontsize=18)
                plt.grid(True)

                plane_xy = self.getArrayPlaneZenithCurves(90.0, azimuth_offset_deg=azimuth_offset_deg)[selection_index]
                im = self.addCurveToMap(im, plane_xy,  mollweide=mollweide, linewidth = self.min_elevation_linewidth, color='k')



            if plot_map == True:
                return self.mesh_azimuth_deg, self.mesh_elevation_deg, phi_resolution, theta_resolution, im, ax
            else:
                return self.mesh_azimuth_deg, self.mesh_elevation_deg, phi_resolution, theta_resolution,

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)



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

    def getEventTimes(self,plot=False,smooth_window=101):
        '''
        Uses the function from the data handler.
        This will hopefully do the appropriate math to determine the real time
        of each event in a run.

        Smoothing will be performed on the rates using the specified smooth window.
        To disable this set the smooth window to None.
        '''
        actual_event_time_seconds = getEventTimes(self.reader,plot=plot,smooth_window=smooth_window)
        return actual_event_time_seconds

    def getENUTrackDict(self, *args, **kwargs):
        '''
        This will return a dict with the trajectories of each plane observed in the 
        period of time specified (given in UTC timestamps).

        Calls pt.getENUTrackDict(start,stop,min_approach_cut_km,hour_window = 12,flights_of_interest=[])
        return flight_tracks_ENU, all_vals
        return {}, []

        '''
        return pt.getENUTrackDict(*args, **kwargs)

    def addAirplanesToMap(self, eventids, pol, ax, azimuth_offset_deg=0, mollweide=False, radius = 1.0, crosshair=False, color='k', min_approach_cut_km=100,plot_distance_cut_limit=100, time_window_s = 60.0):
        '''
        This will add circles to a map for every airplane that is expected to be present in the sky at the time of the
        event.  If a range of events are given then this will plot trajectories of planes ovr the map.

        args and kwargs should be plot parameters.  A common set is:
        linewidth=0.5,fill=False
        Note that color will be used for both crosshair and edge_color of the circle if crosshair == True.  Thus it
        has it's own kwarg.   
    
        Parameters
        ----------
        time_window_s : float
            Half of this is added to both the start and stop time (such that the full search window is expanded by the
            full value).  This means that if a single event is given, the window search will be this span, centered
            at the events trigger time.
        eventids : numpy.array of ints
            The entries to be used for determining the time window appropriate to see if airplanes are visible.
        pol : str
            The polarization you wish to plot.  Used for determining direction of planes based off of antenna locations.
            Either 'hpol' or 'vpol'.
        ax : matplotlib.pyplot.axes
            The axes inwhich the current map is plotted.  This will be modified and returned. 
        azimuth_offset_deg : float
            This is used when the centir_dir is not set to East, and allows for the center of the plot to align properly
            with added plot components. 
        mollweide : bool
            This must match the axes given, as it is used to determine if radians or degrees are used when plotting. 
        radius : float
            The radius of the circle to be added around suspected airplane locations.
        crosshair : bool
            Enables a crosshair centered at the airplane locations.
        color : str
            The color string that the crosshair will be colored. 
        *args : *args
            The list of args for plotting style.
        *kwargs : *kwargs
            The list of kwargs for plotting style.
        '''
        try:
            vmin = 0
            vmax = plot_distance_cut_limit
            norm = matplotlib.colors.Normalize(vmin, vmax)
            cmap = matplotlib.cm.jet
            # Determine if planes are in the sky in the time window of the event.
            event_times = self.getEventTimes()[eventids]
            start = min(event_times) - time_window_s/2
            stop = max(event_times) +  time_window_s/2
            if pol == 'hpol':
                origin = self.A0_latlonel_hpol
            elif pol == 'vpol':
                origin = self.A0_latlonel_vpol
            else:
                origin = self.A0_latlonel_physical

            flight_tracks_ENU, all_vals = self.getENUTrackDict(start,stop,min_approach_cut_km,hour_window=2,flights_of_interest=[],origin=origin)

            # Get interpolated airplane trajectories. 
            airplane_direction_dict = {}

            #TODO: Refresh yourself on what is happening here in the airplane plotting

            for plane_index, key in enumerate(list(flight_tracks_ENU.keys())):
                airplane_direction_dict[key] = {}

                original_norms = numpy.sqrt(flight_tracks_ENU[key][:,0]**2 + flight_tracks_ENU[key][:,1]**2 + flight_tracks_ENU[key][:,2]**2 )
                # import pdb; pdb.set_trace()
                #cut = numpy.logical_and(original_norms/1000.0 < plot_distance_cut_limit,numpy.logical_and(min(flight_tracks_ENU[key][:,3]) <= start ,max(flight_tracks_ENU[key][:,3]) >= stop))
                cut = numpy.logical_and(original_norms/1000.0 < plot_distance_cut_limit,numpy.logical_and(flight_tracks_ENU[key][:,3] >= start ,flight_tracks_ENU[key][:,3]) <= stop)
                if numpy.sum(cut) == 0:
                    continue
                poly = pt.PlanePoly(flight_tracks_ENU[key][cut,3],(flight_tracks_ENU[key][cut,0],flight_tracks_ENU[key][cut,1],flight_tracks_ENU[key][cut,2]),order=5,plot=False)
                interpolated_airplane_locations = poly.poly(event_times)
                if len(poly.poly_funcs) > 0:
                    
                    # Geometry
                    # Calculate phi and theta
                    norms = numpy.sqrt(interpolated_airplane_locations[:,0]**2 + interpolated_airplane_locations[:,1]**2 + interpolated_airplane_locations[:,2]**2 )
                    azimuths = numpy.rad2deg(numpy.arctan2(interpolated_airplane_locations[:,1],interpolated_airplane_locations[:,0]))
                    #azimuths[azimuths < 0] = azimuths[azimuths < 0]%360
                    zeniths = numpy.rad2deg(numpy.arccos(interpolated_airplane_locations[:,2]/norms))

                    airplane_direction_dict[key]['azimuth'] = azimuths
                    airplane_direction_dict[key]['zenith'] = zeniths
                    airplane_direction_dict[key]['distance_km'] = norms/1000

                    # plot each circle
                    for direction_index in range(len(azimuths)):
                        azimuth = azimuths[direction_index]
                        zenith = zeniths[direction_index]
                        elevation = 90.0 - zenith

                        # Add best circle.
                        azimuth = azimuth - azimuth_offset_deg
                        if azimuth < -180.0:
                            azimuth += 2*180.0
                        elif azimuth > 180.0:
                            azimuth -= 2*180.0
                        
                        if mollweide == True:
                            radius = numpy.deg2rad(radius)
                            elevation = numpy.deg2rad(elevation)
                            azimuth = numpy.deg2rad(azimuth)

                        circle = plt.Circle((azimuth, elevation), radius, edgecolor=cmap(norm(norms[direction_index]/1000)),fill=False)

                        if crosshair == True:
                            h = ax.axhline(elevation,c=color,linewidth=1,alpha=0.5)
                            v = ax.axvline(azimuth,c=color,linewidth=1,alpha=0.5)

                        ax.add_artist(circle)

            sc = plt.scatter([0,0],[0,0],s=0,c=[vmin,vmax], cmap='jet', vmin = vmin, vmax = vmax, facecolors='none')
            #import pdb; pdb.set_trace()
            cbar = plt.colorbar(sc)
            cbar.set_label('Airplane Distance (km)', rotation=90, labelpad=10)
            
            return ax, airplane_direction_dict
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

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

        crit_freq_low_pass_MHz = 100#70 #This new pulser seems to peak in the region of 85 MHz or so
        low_pass_filter_order = 8#12

        crit_freq_high_pass_MHz = None#55
        high_pass_filter_order = None#4

        apply_phase_response = True
        sine_subtract = True
        plot_filter = True


        known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
        eventids = {}
        eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])
        eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])
        all_eventids = numpy.sort(numpy.append(eventids['hpol'],eventids['vpol']))

        hpol_eventids_cut = numpy.isin(all_eventids,eventids['hpol'])
        vpol_eventids_cut = numpy.isin(all_eventids,eventids['vpol'])

        cor = Correlator(reader,  upsample=2**15, n_phi=361, n_theta=361, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=True)
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
    #'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0'

    if False:
        crit_freq_low_pass_MHz = 100#60 #This new pulser seems to peak in the region of 85 MHz or so
        low_pass_filter_order = 8

        crit_freq_high_pass_MHz = None#30#None
        high_pass_filter_order = None#5#None
        
        sine_subtract = True
        sine_subtract_min_freq_GHz = 0.03
        sine_subtract_max_freq_GHz = 0.250
        sine_subtract_percent = 0.05
    elif False:
        crit_freq_low_pass_MHz = 85
        low_pass_filter_order = 6

        crit_freq_high_pass_MHz = 25
        high_pass_filter_order = 8

        sine_subtract = True
        sine_subtract_min_freq_GHz = 0.03
        sine_subtract_max_freq_GHz = 0.13
        sine_subtract_percent = 0.03
    elif True:
        crit_freq_low_pass_MHz = 85
        low_pass_filter_order = 6

        crit_freq_high_pass_MHz = 25
        high_pass_filter_order = 8

        sine_subtract = True
        sine_subtract_min_freq_GHz = 0.00
        sine_subtract_max_freq_GHz = 0.25
        sine_subtract_percent = 0.03
    else:
        crit_freq_low_pass_MHz = 80
        low_pass_filter_order = 14

        crit_freq_high_pass_MHz = 20
        high_pass_filter_order = 4

        sine_subtract = False
        sine_subtract_min_freq_GHz = 0.02
        sine_subtract_max_freq_GHz = 0.15
        sine_subtract_percent = 0.01

    plot_filter=True

    apply_phase_response=True

    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

    range_phi_deg = (min_phi, max_phi)
    range_theta_deg = (min_theta, max_theta)

    # n_phi = 720
    # n_theta = 1080
    upsample = 2**16
    max_method = 0


    
    if len(sys.argv) >= 3:
        run = int(sys.argv[1])
        eventid = int(sys.argv[2])
        try:
            deploy_index = str(sys.argv[3])
        except:
            deploy_index = info.returnDefaultDeploy()
    
        datapath = os.environ['BEACON_DATA']

        all_figs = []
        all_axs = []
        all_cors = []
        map_source_distance_m = info.returnDefaultSourceDistance()
        if run == 1507:
            waveform_index_range = (1500,None) #Looking at the later bit of the waveform only, 10000 will cap off.  
        elif run == 1509:
            waveform_index_range = (2000,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
        elif run == 1511:
            waveform_index_range = (1250,2000) #Looking at the later bit of the waveform only, 10000 will cap off.  
        elif run == 5630:
                waveform_index_range = (1250,2274)
                print('Using waveform_index_range of ', str(waveform_index_range))
                map_source_distance_m = 521.62
        else:
            waveform_index_range = info.returnDefaultWaveformIndexRange()
            #waveform_index_range = (None,None)
        # if True:
        # shorten_signals = False
        # index_window_dict = {'hpol': {'d2sa': (1250, 2274),
        #                               'd3sa': (2273, 3297),
        #                               'd3sb': (2426, 3450),
        #                               'd3sc': (1889, 2913),
        #                               'd4sa': (2060, 3084),
        #                               'd4sb': (1097, 2121)},
        #                      'vpol': {'d2sa': (1233, 2257),
        #                               'd3sa': (2246, 3270),
        #                               'd3sb': (2445, 3469),
        #                               'd3sc': (1941, 2965),
        #                               'd4sa': (2074, 3098),
        #                               'd4sb': (1063, 2087)}}
        #     else:
        #         index_window_dict = {'hpol':{},'vpol':{}} 

        reader = Reader(datapath,run)


        if False:
            for map_source_distance_m in [1e3,1e4,1e5,1e6]:
                cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter,apply_phase_response=apply_phase_response, deploy_index=deploy_index, map_source_distance_m=map_source_distance_m)
                

                if sine_subtract:
                    cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                for mode in ['hpol']:
                    mean_corr_values, fig, ax, max_possible_map_value = cor.map(eventid, mode, include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=None, plot_corr=False, hilbert=False, interactive=True, max_method=0, waveforms=None, verbose=True, mollweide=False, zenith_cut_ENU=None, zenith_cut_array_plane=[0.0,90.0], center_dir='E', circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=True, return_max_possible_map_value=True, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=False, circle_map_max=True)
                    #mean_corr_values, fig, ax = cor.map(eventid, mode, include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, plot_corr=False, hilbert=False,interactive=True, max_method=0, waveforms=None, verbose=True, mollweide=False, zenith_cut_ENU=None, zenith_cut_array_plane=[0,90.0], center_dir='E', circle_zenith=None, circle_az=None, time_delay_dict={},window_title=None,add_airplanes=True)
                    fig.set_size_inches(16, 9)
                    plt.sca(ax)
                    plt.tight_layout()
                    all_figs.append(fig)
                    all_axs.append(ax)
                    if False:
                        cor.plotPointingResolution(mode, snr=5, bw=50e6, plot_map=True, mollweide=False,center_dir='E', window_title=None, include_baselines=[0,1,2,3,4,5])
                        # for baseline in [0,1,2,3,4,5]:
                        #     cor.plotPointingResolution(mode, snr=5, bw=50e6, plot_map=True, mollweide=False,center_dir='E', window_title=None, include_baselines=[baseline])
        else:
            cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter,apply_phase_response=apply_phase_response, deploy_index=deploy_index, map_source_distance_m=map_source_distance_m)
            

            if sine_subtract:
                cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

            for mode in ['hpol','vpol','all']:
                mean_corr_values, fig, ax, max_possible_map_value = cor.map(eventid, mode, include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=None, plot_corr=False, hilbert=False, interactive=True, max_method=0, waveforms=None, verbose=True, mollweide=False, zenith_cut_ENU=None, zenith_cut_array_plane=[0.0,90.0], center_dir='E', circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=True, return_max_possible_map_value=True, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=False, circle_map_max=True)
                #mean_corr_values, fig, ax = cor.map(eventid, mode, include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, plot_corr=False, hilbert=False,interactive=True, max_method=0, waveforms=None, verbose=True, mollweide=False, zenith_cut_ENU=None, zenith_cut_array_plane=[0,90.0], center_dir='E', circle_zenith=None, circle_az=None, time_delay_dict={},window_title=None,add_airplanes=True)
                fig.set_size_inches(16, 9)
                plt.sca(ax)
                plt.tight_layout()
                all_figs.append(fig)
                all_axs.append(ax)
                if False:
                    cor.plotPointingResolution(mode, snr=5, bw=50e6, plot_map=True, mollweide=False,center_dir='E', window_title=None, include_baselines=[0,1,2,3,4,5])
                    # for baseline in [0,1,2,3,4,5]:
                    #     cor.plotPointingResolution(mode, snr=5, bw=50e6, plot_map=True, mollweide=False,center_dir='E', window_title=None, include_baselines=[baseline])


            
        all_cors.append(cor)
    else:
        plt.close('all')
        datapath = os.environ['BEACON_DATA']

        all_figs = []
        all_axs = []
        all_cors = []

        if True:

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
                        site3_measured_time_delays =  {'hpol':{'[0, 1]' : [-88.02014654064],    '[0, 2]': [-143.62662406045482], '[0, 3]': [-177.19680779079835], '[1, 2]': [-55.51270605688091], '[1, 3]': [-88.89534686135659], '[2, 3]': [-33.32012649585307]},\
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

                        mean_corr_values, fig, ax = cor.map(eventid, mode, plot_map=True, plot_corr=False, center_dir='W',hilbert=False, interactive=True, max_method=max_method,circle_zenith=pulser_theta,circle_az=pulser_phi,time_delay_dict=time_delay_dict)
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

        if False:
            #Preparing for planes:
            known_planes, calibrated_trigtime, output_tracks = pt.getKnownPlaneTracks()

            plane_polys = {}
            cors = []
            interpolated_airplane_locations = {}
            origin = info.loadAntennaZeroLocation()
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

                interpolated_airplane_locations[key] = plane_polys[key].poly(calibrated_trigtime[key])
                normalized_plane_locations = interpolated_airplane_locations[key]/(numpy.tile(numpy.linalg.norm(interpolated_airplane_locations[key],axis=1),(3,1)).T)

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
