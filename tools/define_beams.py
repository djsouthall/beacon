'''
This script is meant to be used to determine good beams for the BEACON trigger.
'''
import os
import sys
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
from tools.profiler import profile
from tools.correlator import Correlator

import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import numpy
import sys
import math
import matplotlib
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


class BeamMaker(Correlator):
    '''
    Uses some of the framework of the correlator class,
    while adding specific tools for making beams.

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        This is the reader for the selected run.
    upsample : int
        Forced to None.
    n_phi : int
        The number of azimuthal angles to probe in the specified range.
    range_phi_deg : tuple of floats with len = 2 
        The specified range of azimuthal angles to probe.
    n_theta : int 
        The number of zenith angles to probe in the specified range.
    range_theta_deg : tuple of floats with len = 2  
        The specified range of zenith angles to probe.
    
    '''
    def __init__(self, reader,  n_phi=181, range_phi_deg=(-180,180), n_theta=361, range_theta_deg=(0,180), crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=False, waveform_index_range=(None,None)):
        try:
            super().__init__(reader,  upsample=None, n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter, waveform_index_range=waveform_index_range)
            self.prepInterpFunc()


        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)



    def prepInterpFunc(self):
        '''
        When called this will generate the interpolation functions used by interpolateTimeGrid.  Should be called if positions change (or similar).
        '''
        try:
            self.t_hpol_0subtract1_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_hpol_0subtract1)
            self.t_hpol_0subtract2_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_hpol_0subtract2)
            self.t_hpol_0subtract3_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_hpol_0subtract3)

            self.t_vpol_0subtract1_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_vpol_0subtract1)
            self.t_vpol_0subtract2_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_vpol_0subtract2)
            self.t_vpol_0subtract3_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_vpol_0subtract3)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def interpolateTimeGrid(self, mode, thetas ,phis):
        '''
        Will determine appropriate time delays for the given thetas and phis using the predifined grid (created by generateTimeIndices).
        Expects angles in radians. 
        '''
        try:
            if mode == 'hpol':
                t_0subtract1 = self.t_hpol_0subtract1_interp(thetas,phis,grid=False)
                t_0subtract2 = self.t_hpol_0subtract2_interp(thetas,phis,grid=False)
                t_0subtract3 = self.t_hpol_0subtract3_interp(thetas,phis,grid=False)
            elif mode == 'vpol':
                t_0subtract1 = self.t_vpol_0subtract1_interp(thetas,phis,grid=False)
                t_0subtract2 = self.t_vpol_0subtract2_interp(thetas,phis,grid=False)
                t_0subtract3 = self.t_vpol_0subtract3_interp(thetas,phis,grid=False)

            return t_0subtract1, t_0subtract2, t_0subtract3
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def roundToNearestSample(self,t,dt=None):
        '''
        Given some time, it will round it to the nearest value using dt value in ns. 
        '''
        try:
            if dt is None:
                dt = numpy.diff(self.reader.t())[0]
            return numpy.round(t/dt)*dt
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def makeFakeSignalFromDir(self, eventid, pol, theta, phi, plot=False, hilbert=False):
        '''
        This load the first waveform of the specified polarization and replicate it and roll
        it to make it appear to be coming from a specified direction. 

        Theta and phi should be given in radians.
        '''
        try:
            if pol == 'hpol':
                waveforms = self.wf(eventid, numpy.array([0,0,0,0]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter) #Div by std and resampled waveforms normalizes the correlations
            elif pol == 'vpol':
                waveforms = self.wf(eventid, numpy.array([1,1,1,1]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter) #Div by std and resampled waveforms normalizes the correlations

            t_0subtract1, t_0subtract2, t_0subtract3 = bm.roundToNearestSample(numpy.asarray(bm.interpolateTimeGrid(pol, theta,phi)))

            waveforms[1] = numpy.roll(waveforms[1],numpy.round(-t_0subtract1/self.dt_resampled).astype(int))
            waveforms[2] = numpy.roll(waveforms[2],numpy.round(-t_0subtract2/self.dt_resampled).astype(int))
            waveforms[3] = numpy.roll(waveforms[3],numpy.round(-t_0subtract3/self.dt_resampled).astype(int))

            if plot == True:
                plt.figure()
                for i,waveform in enumerate(waveforms):
                    plt.plot(self.t(), waveform,label=str(i))
                plt.legend()
                plt.ylabel('adu')
                plt.xlabel('ns')
            
            return waveforms
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def makeFakeBeamMap(self, eventid, pol, beam_theta, beam_phi, plot_map=True, plot_corr=False, hilbert=False, normalize=False, use_single_wf=True):
        '''
        This will generate fake signals from each of the directions normally available in the correlation map.  It will then apply beamforming assuming 
        the beam is centered on theta, phi. 
        '''
        try:
            #Prepare beam's time delays.  The ones that are assumed for this beam.
            t_0subtract1, t_0subtract2, t_0subtract3 = bm.roundToNearestSample(numpy.asarray(bm.interpolateTimeGrid(pol, beam_theta, beam_phi)))
            rolls = [0, (t_0subtract1/self.dt_resampled).astype(int) , (t_0subtract2/self.dt_resampled).astype(int) , (t_0subtract3/self.dt_resampled).astype(int)]


            #Prepare fake waveform.  Will be rolled into position (to appear to be coming from a particular direction) and then rolled back (assumed to be 
            #from the beam direction) before experiencing the power sum.  These rolls are done as one motion in the faster version of the code. 
            if pol == 'hpol':
                if use_single_wf == True:
                    raw_wf = self.wf(eventid, numpy.array([0,0,0,0]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter) #Div by std and resampled waveforms normalizes the correlations
                else:
                    raw_wf = self.wf(eventid, numpy.array([0,2,4,6]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter) #Div by std and resampled waveforms normalizes the correlations
            elif pol == 'vpol':
                if use_single_wf == True:
                    raw_wf = self.wf(eventid, numpy.array([1,1,1,1]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter) #Div by std and resampled waveforms normalizes the correlations
                else:
                    raw_wf = self.wf(eventid, numpy.array([1,3,5,7]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter) #Div by std and resampled waveforms normalizes the correlations                
            
            #Prepare for power sum
            max_powers = numpy.zeros((self.n_theta, self.n_phi))
            power_sum_step = 8
            N_original = int(len(self.t()))
            N_new = int(numpy.ceil(N_original/power_sum_step)*power_sum_step)

            padded_wf = numpy.zeros((4,N_new))

            for waveform_index, waveform in enumerate(raw_wf):
                padded_wf[waveform_index][0:N_original] = waveform
            
            binned_8_indices_A = numpy.arange(N_new).reshape((-1,power_sum_step)).astype(int)
            binned_8_indices_B = numpy.roll(numpy.arange(N_new).reshape((-1,power_sum_step)),-1,axis=0).astype(int)

            #Preparing for the waveform that will actually be the rolled padded and summed version.
            temp_waveforms = numpy.zeros_like(padded_wf)
            temp_waveforms[0] = padded_wf[0]

            #looping
            for theta_index, theta in enumerate(self.thetas_rad):
                sys.stdout.write('theta = %0.2f\r'%(numpy.rad2deg(theta)))
                sys.stdout.flush()
                for phi_index, phi in enumerate(self.phis_rad):
                    temp_t_0subtract1, temp_t_0subtract2, temp_t_0subtract3 = bm.roundToNearestSample(numpy.asarray(bm.interpolateTimeGrid(pol, theta,phi)))
                    temp_waveforms[1] = numpy.roll(padded_wf[1],numpy.round(-temp_t_0subtract1/self.dt_resampled).astype(int) + rolls[1])
                    temp_waveforms[2] = numpy.roll(padded_wf[2],numpy.round(-temp_t_0subtract2/self.dt_resampled).astype(int) + rolls[2])
                    temp_waveforms[3] = numpy.roll(padded_wf[3],numpy.round(-temp_t_0subtract3/self.dt_resampled).astype(int) + rolls[3])

                    summed_waveforms = numpy.sum(temp_waveforms,axis=0)
                    power = summed_waveforms**2
                    power_sum = numpy.sum(power[binned_8_indices_A],axis=1) + numpy.sum(power[binned_8_indices_B],axis=1)
                    max_powers[theta_index,phi_index] = numpy.max(power_sum)

            if normalize == True:
                max_powers /= numpy.max(max_powers)

            if plot_map:
                fig = plt.figure()
                fig.canvas.set_window_title('Zenith = %0.2f, Az = %0.2f Beam Map'%(numpy.rad2deg(beam_theta),numpy.rad2deg(beam_phi)))
                ax = fig.add_subplot(1,1,1)
                im = ax.imshow(max_powers, interpolation='none', extent=[min(self.phis_deg),max(self.phis_deg),max(self.thetas_deg),min(self.thetas_deg)],cmap=plt.cm.coolwarm) #cmap=plt.cm.jet)
                cbar = fig.colorbar(im)
                plt.title('Sensitivity Map For Beam Centered at Zenith = %0.2f, Az = %0.2f\nwith time delays: t_0subtract1 = %0.2f, t_0subtract2 = %0.2f, t_0subtract3 = %0.2f'%(numpy.rad2deg(beam_theta),numpy.rad2deg(beam_theta),t_0subtract1,t_0subtract2,t_0subtract3))
                if normalize == True:
                    cbar.set_label('Normalized Beam Powersum')
                else:
                    cbar.set_label('Beam Powersum')
                plt.xlabel('Azimuth Angle (Degrees)')
                plt.ylabel('Zenith Angle (Degrees)')

                radius = 5.0 #Degrees I think?  Should eventually represent error. 
                circle = plt.Circle((numpy.rad2deg(beam_phi), numpy.rad2deg(beam_theta)), radius, edgecolor='lime',linewidth=2,fill=False)
                ax.axvline(numpy.rad2deg(beam_phi),c='lime',linewidth=1)
                ax.axhline(numpy.rad2deg(beam_theta),c='lime',linewidth=1)

                ax.add_artist(circle)

                self.figs.append(fig)
                self.axs.append(ax)
            return max_powers
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

if __name__=="__main__":
    #plt.close('all')
    use_pulser=False
    max_method = 0
    
    datapath = os.environ['BEACON_DATA']
    if use_pulser == True:
        run = 1509
        known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
        eventids = {}
        eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])
        eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])
        waveform_index_range = (2000,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
    else:
        run = 1645
        eventids = {}
        eventids['hpol'] = [45]
        eventids['vpol'] = [48]
        waveform_index_range = (None,None) #Looking at the later bit of the waveform only, 10000 will cap off.  

    reader = Reader(datapath,run)

    all_fig = []
    all_ax = []

    for mode in ['hpol']:
        print('Mode = %s'%mode)
        bm = BeamMaker(reader, n_phi=360, n_theta=180, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=False)
        eventid = eventids[mode][0]


        max_powers = bm.makeFakeBeamMap(eventid, mode, numpy.deg2rad(100.0), numpy.deg2rad(0.0), plot_map=True, plot_corr=False, hilbert=False)
        #max_powers = bm.makeFakeBeamMap(eventid, mode, numpy.deg2rad(100.0), numpy.deg2rad(3.24), plot_map=True, plot_corr=False, hilbert=False)
        #max_powers = bm.makeFakeBeamMap(eventid, mode, numpy.deg2rad(100.0), numpy.deg2rad(0.0), plot_map=True, plot_corr=False, hilbert=True)

        #max_powers = bm.makeFakeBeamMap(eventid, mode, numpy.deg2rad(70.0), numpy.deg2rad(25.0), plot_map=True, plot_corr=False, hilbert=False)
        #max_powers = bm.makeFakeBeamMap(eventid, mode, numpy.deg2rad(70.0), numpy.deg2rad(25.0), plot_map=True, plot_corr=False, hilbert=True)

        '''
        mean_corr_values, fig, ax = bm.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=False, interactive=True)
        row_index, column_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = bm.mapMax(mean_corr_values, max_method=0, verbose=True)
        t_0subtract1, t_0subtract2, t_0subtract3 = bm.interpolateTimeGrid(mode, numpy.deg2rad(theta_best),numpy.deg2rad(phi_best))
        t_0subtract1 = bm.roundToNearestSample(t_0subtract1)
        t_0subtract2 = bm.roundToNearestSample(t_0subtract2)
        t_0subtract3 = bm.roundToNearestSample(t_0subtract3)
        print('From Correlation Map:')
        print('t_best_0subtract1 = %0.2f, t_best_0subtract2 = %0.2f, t_best_0subtract3 = %0.2f'%(t_best_0subtract1, t_best_0subtract2, t_best_0subtract3))

        print('From Interpolator (theta = %0.2f, phi = %0.2f:'%(theta_best, phi_best))
        print('t_0subtract1 = %0.2f, t_0subtract2 = %0.2f, t_0subtract3 = %0.2f'%(t_0subtract1, t_0subtract2, t_0subtract3))

        mean_corr_values, fig, ax = bm.makeFakeSignalsMap(eventid, mode, t_0subtract1, t_0subtract2, t_0subtract3, plot_map=True, plot_corr=False, hilbert=False, interactive=False, max_method=None)
        all_fig.append(fig)
        all_ax.append(ax)
        '''