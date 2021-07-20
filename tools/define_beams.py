'''
This script is meant to be used to map the current beams for the BEACON trigger.
'''
import os
import sys
from pathlib import Path
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.correlator import Correlator
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
from tools.profiler import profile

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
plt.ioff()
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

    For a detailed parameter list look at the correlator class, as it is kept
    more up to date. 

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
    def __init__(self, reader,  upsample=None, n_phi=181, range_phi_deg=(-180,180), n_theta=361, range_theta_deg=(0,180), crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, 
                    low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=False, waveform_index_range=(None,None), apply_phase_response=False, tukey=False, sine_subtract=True, 
                    map_source_distance_m=1e6, deploy_index=None):
        try:
            super().__init__(reader,  upsample=upsample, n_phi=n_phi, range_phi_deg=range_phi_deg, 
                                n_theta=n_theta, range_theta_deg=range_theta_deg, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, 
                                low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter, 
                                waveform_index_range=waveform_index_range, apply_phase_response=apply_phase_response, tukey=tukey, sine_subtract=sine_subtract, 
                                map_source_distance_m=map_source_distance_m, deploy_index=deploy_index)
            self.prepInterpFunc()


        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def prepInterpFunc(self):
        '''
        When called this will generate the interpolation functions used by interpolateTimeGrid.  Should be called if 
        positions change (or similar).  This will return arrival times (at board, including cable delays) for antennas 
        1,2,3 relative to antenna 0.  
        '''
        try:
        
            print(self.t_hpol_0subtract1)
            print(self.t_hpol_0subtract2)
            print(self.t_hpol_0subtract3)
            
            self.t_hpol_0subtract1_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_hpol_0subtract1) #self.t_hpol_0subtract1 already includes cable delays.  Results in the expected measured time difference in signals from each direction.  
            self.t_hpol_0subtract2_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_hpol_0subtract2) #self.t_hpol_0subtract2 already includes cable delays.  Results in the expected measured time difference in signals from each direction.  
            self.t_hpol_0subtract3_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_hpol_0subtract3) #self.t_hpol_0subtract3 already includes cable delays.  Results in the expected measured time difference in signals from each direction.  

            self.t_vpol_0subtract1_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_vpol_0subtract1) #self.t_vpol_0subtract1 already includes cable delays.  Results in the expected measured time difference in signals from each direction.  
            self.t_vpol_0subtract2_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_vpol_0subtract2) #self.t_vpol_0subtract2 already includes cable delays.  Results in the expected measured time difference in signals from each direction.  
            self.t_vpol_0subtract3_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , self.t_vpol_0subtract3) #self.t_vpol_0subtract3 already includes cable delays.  Results in the expected measured time difference in signals from each direction.  
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    
    def overwriteDelays(self, pol, t_0subtract1, t_0subtract2, t_0subtract3):
        '''
        Overwrites the delays relative to antenna0 for either vpol or hpol. 
        pol is 'hpol' or 'vpol'. 
        '''
        
        if pol == 'hpol':
            self.t_hpol0subtract1 = t_0subtract1
            self.t_hpol0subtract2 = t_0subtract2
            self.t_hpol0subtract3 = t_0subtract3
        elif pol == 'vpol':
            self.t_vpol0subtract1 = t_0subtract1
            self.t_vpol0subtract2 = t_0subtract2
            self.t_vpol0subtract3 = t_0subtract3
        
        self.prepInterpFunc()
        
    def interpolateTimeGrid(self, mode, thetas, phis):
        '''
        Will determine appropriate time delays for the given thetas and phis using the predifined grid (created by 
        generateTimeIndices).  Expects angles in radians. 
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

    def makeFakeBeamMap(self, eventid, pol, beam_theta, beam_phi, plot_map=True, plot_corr=False, hilbert=False, normalize=False, 
                                savefig=False, savefig_text=None, turnoff_ants=None, forced_deltats = None):
        '''
        This will generate fake signals from each of the directions normally available in the correlation map.  It will 
        then apply beamforming assuming the beam is centered on theta, phi. 
        '''
        try:
            if not (forced_deltats is None):
                t_1, t_2, t_3 = forced_deltats
            else:
                t_1, t_2, t_3 = bm.interpolateTimeGrid(pol, beam_theta, beam_phi)
            t_0 = 0.0
            t_1, t_2, t_3 = bm.roundToNearestSample(numpy.array([t_1, t_2, t_3]))
            
            '''
            Prepare beam rolls.  These are the values that would actually be set in firmware, and have the roll of 
            counteracting time delays.  So give the expected time delays from a source direction, they should
            roll the signals back by that amount.  To roll "back" (to the left) requires a positive index value,
            and is equivalent to pushing a signal back in time, or holding that signal while all other signals move
            forward in time. 

            t_i here represent the expected arrival time at antenna i relative to antenna 0.  t_i represents 
            arrival_0 - arrival_i, so if it positive when arrival_i < arrival 0 (signal arrives at antenna i first).
            
            In the case that the signal arrives at i before 0, we want to hold it until 0 is expected to arrive.  This
            is equivalent to rolling signal i forward in time by the expected time offset.  
            
            To roll signal i forward, a positive roll integer required.

            In the above case where i arrives before 0, t_1 = arrival_0 - arrival_i is positive.

            So in this case the beam roll values should be integer versions of the expected arrival time delays,
            with the same sign.
            '''

            beam_rolls = numpy.array([0, (t_1/self.dt_resampled).astype(int) , (t_2/self.dt_resampled).astype(int) , (t_3/self.dt_resampled).astype(int)])

            #Prepare fake waveform.  Will be rolled into position (to appear to be coming from a particular direction) and then rolled back (assumed to be 
            #from the beam direction) before experiencing the power sum.  These rolls are done as one motion in the faster version of the code. 
            if pol == 'hpol':
                raw_wf = self.wf(eventid, numpy.array([0,0,0,0]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter) #Div by std and resampled waveforms normalizes the correlations
            elif pol == 'vpol':
                raw_wf = self.wf(eventid, numpy.array([1,1,1,1]),div_std=True,hilbert=hilbert,apply_filter=self.apply_filter) #Div by std and resampled waveforms normalizes the correlations
            if turnoff_ants is not None:
                for ant in turnoff_ants:
                    raw_wf[ant] *= 0

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
                if theta_index % 10 == 0:
                    sys.stdout.write('theta = %0.2f\r'%(numpy.rad2deg(theta)))
                    sys.stdout.flush()
                for phi_index, phi in enumerate(self.phis_rad):
                    #Below is the times you would expect the signals to appear (after digitization).
                    temp_t_1, temp_t_2, temp_t_3 = bm.interpolateTimeGrid(pol, theta,phi)
                    temp_t_1, temp_t_2, temp_t_3 = bm.roundToNearestSample(numpy.array([temp_t_1, temp_t_2, temp_t_3]))

                    '''
                    If I want a signal to appear to be arriving at a later time, I need to roll it a positive integer.
                    I would want a signal to appear later when temp_t_i < 0 (so that arrival time 1 > arrival time 0)

                    So to simulate signals with arrival times temp_t_i, I roll signals from antenna 0 by an integer version
                    of that time delay with the opposite sign.


                    Here is the logic behind generating the fake signals using temp_t's:

                    t1 = arrival_time0 - arrival_time1
                    So if starting with signals at arrival_time0 and want to replicated to make one simulate arrival_time1
                    "rolling" signal 1 by t1 (in indices) will push the peak to the right (forward in time) by t1 (or back if
                    t1 is negative).  To roll a signal to the right requires a negative roll integer.  

                    Behaviour I want:  if t1 is negative, then arrival_time1 > arrival_time0, so I want fake signal 1 pushed
                    right, which occurs will roll with a positive integer roll.  So the roll I want is negative (integer) of the
                    expected time delays.  
                    '''
                    roll_make_simulated_signal_1 = -numpy.round(temp_t_1/self.dt_resampled).astype(int)
                    roll_make_simulated_signal_2 = -numpy.round(temp_t_2/self.dt_resampled).astype(int)
                    roll_make_simulated_signal_3 = -numpy.round(temp_t_3/self.dt_resampled).astype(int)


                    '''
                    The signals here are rolled twice.  roll_make_simulated_signal_i rolling rolls the signal to make it 
                    appear to be coming from the simulated direction theta, phi.  Then it is rolled by beam_rolls, which 
                    are firmware roll values indicating how long to delay the signal for that beam such that a signal 
                    from that beams intended direction would align perfectly. For a given direction roll and 
                    roll_make_simulated_signal roll should be equal but opposite.
                    '''

                    temp_waveforms[1] = numpy.roll(padded_wf[1], roll_make_simulated_signal_1 + beam_rolls[1])
                    temp_waveforms[2] = numpy.roll(padded_wf[2], roll_make_simulated_signal_2 + beam_rolls[2])
                    temp_waveforms[3] = numpy.roll(padded_wf[3], roll_make_simulated_signal_3 + beam_rolls[3])

                    summed_waveforms = numpy.sum(temp_waveforms,axis=0)
                    power = summed_waveforms**2
                    power_sum = numpy.sum(power[binned_8_indices_A],axis=1) + numpy.sum(power[binned_8_indices_B],axis=1)
                    max_powers[theta_index,phi_index] = numpy.max(power_sum)

            integer_delays = (beam_rolls - min(beam_rolls)).astype(int) #You can sanity check these: A signal from East would hit antenna 0 and 2 first, so those should be delayed the most (largest positive integer), a signal from north would need 0 and 1 to be delayed the most (so should be delayed largest integer number).
            
            if normalize == True:
                max_powers /= numpy.max(max_powers)

            additional_text = ''
            if turnoff_ants is not None:
                additional_text += '_suppressing_ants_%s'%str(turnoff_ants)
            if hilbert == True:
                additional_text += '_hilbert-True'
            else:
                additional_text += '_hilbert-False'
            if savefig_text is None:
                savefig_text = '%s_zen%0.2f_az%0.2f%s.jpg'%(pol, numpy.rad2deg(beam_theta), numpy.rad2deg(beam_phi),additional_text)
            else:
                savefig_text = '%s_%s_zen%0.2f_az%0.2f%s.jpg'%(pol, savefig_text,numpy.rad2deg(beam_theta), numpy.rad2deg(beam_phi),additional_text)
            save_map_text = savefig_text.replace('.jpg','_map_data.csv')
            header = '%s\nthetas_deg [axis=0] = ,%s,\nself.phis_deg [axis=1] = ,%s\n\n'%(savefig_text.replace('.jpg',''),str(list(self.thetas_deg)).replace('[','').replace(']',''),str(list(self.phis_deg)).replace('[','').replace(']',''))
            numpy.savetxt(save_map_text,max_powers,delimiter=',',header=header)

            if plot_map:
                fig = plt.figure(figsize=(16,9))
                fig.canvas.set_window_title('Zenith = %0.2f, Az = %0.2f Beam Map'%(numpy.rad2deg(beam_theta),numpy.rad2deg(beam_phi)))
                ax = fig.add_subplot(1,1,1)
                if normalize == True:
                    im = ax.imshow(max_powers, interpolation='none', extent=[min(self.phis_deg),max(self.phis_deg),max(self.thetas_deg),min(self.thetas_deg)],cmap='plasma',vmin=0,vmax=1) #cmap=plt.cm.jet)
                else:
                    im = ax.imshow(max_powers, interpolation='none', extent=[min(self.phis_deg),max(self.phis_deg),max(self.thetas_deg),min(self.thetas_deg)],cmap='plasma') #cmap=plt.cm.jet)
                cbar = fig.colorbar(im)
                plt.title('Sensitivity Map For Beam Centered at Zenith = %0.2f, Az = %0.2f\nwith time delays: t_1 = %0.2f, t_2 = %0.2f, t_3 = %0.2f\nHilbert=%s'%(numpy.rad2deg(beam_theta),numpy.rad2deg(beam_phi),t_1,t_2,t_3,str(hilbert)))
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
                if savefig == True:

                    
                    fig.savefig(savefig_text)
                    plt.close(fig)

                self.figs.append(fig)
                self.axs.append(ax)

            return max_powers, integer_delays[0], integer_delays[1], integer_delays[2], integer_delays[3]
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateBeams(self,reader,eventid,mode='hpol',finame_hpol_delays=None):
        '''
        This is not a real function, really.  It is storing relevant code for calculating beams from an event
        using pre-saved beams.  This code was pulled from low_snr_search.py
        '''
        if mode == 'hpol':
            pol = 0
        elif mode == 'vpol':
            pol = 1
        hpol_beam_delays = info.loadBeamDelays(finame_hpol_delays=finame_hpol_delays, finame_vpol_delays=finame_hpol_delays)[pol]
        reader.setEntry(eventid)
        t = reader.t()

        max_powers = numpy.zeros(hpol_beam_delays.shape[0])

        power_sum_step = 8
        N_original = int(len(t))
        N_new = int(numpy.ceil(N_original/power_sum_step)*power_sum_step)
        padded_wf = numpy.zeros((4,N_new))
        new_t = numpy.arange(N_new)*(t[1]-t[0])

        for i in range(4):
            padded_wf[i][0:N_original] = reader.wf(2*i)

        binned_8_indices_A = numpy.arange(N_new).reshape((-1,power_sum_step)).astype(int)
        binned_8_indices_B = numpy.roll(numpy.arange(N_new).reshape((-1,power_sum_step)),-1,axis=0).astype(int)


        rolled_wf = numpy.zeros_like(padded_wf)
        plt.figure()
        for beam_index, beam in enumerate(hpol_beam_delays):
            #delay_ant0 = beam[3]
            #delay_ant1 = beam[4]
            #delay_ant2 = beam[5]
            #delay_ant3 = beam[6]

            for i in range(4):
                rolled_wf[i] = numpy.roll(padded_wf[i],int(beam[3 + i]))

            summed_waveforms = numpy.sum(rolled_wf,axis=0)
            power = summed_waveforms**2
            power_sum = numpy.sum(power[binned_8_indices_A],axis=1) + numpy.sum(power[binned_8_indices_B],axis=1)
            max_powers[beam_index] = numpy.max(power_sum)
            plt.plot(power_sum,label='Beam %i\nZen = %0.1f deg\nAz = %0.1f deg'%(beam_index,hpol_beam_delays[beam_index,1],hpol_beam_delays[beam_index,2]))

        plt.xlabel('Power Sum Bins')
        plt.ylabel('Summed Power (adu^2)')
        plt.legend()

        for beam_index in numpy.argsort(max_powers)[::-1][0:5]:
            print('Beam %i\tPowerSum = %i\nZen = %0.1f deg\tAz = %0.1f deg'%(beam_index,max_powers[beam_index],hpol_beam_delays[beam_index,1],hpol_beam_delays[beam_index,2]))

        best_beam = numpy.argmax(max_powers)

        beam_index = best_beam
        beam = hpol_beam_delays[best_beam]
        for i in range(4):
            rolled_wf[i] = numpy.roll(padded_wf[i],int(beam[3 + i]))

        summed_waveforms = numpy.sum(rolled_wf,axis=0)
        power = summed_waveforms**2
        power_sum = numpy.sum(power[binned_8_indices_A],axis=1) + numpy.sum(power[binned_8_indices_B],axis=1)
        max_powers[beam_index] = numpy.max(power_sum)

        fig = plt.figure()
        plt.suptitle('Eventid: %i, Triggered on Beam: %i'%(eventid,best_beam))
        plt.subplot(3,1,1)
        for wf_index, wf in enumerate(padded_wf):
            plt.plot(new_t,wf,label='hpol %i'%wf_index)

        extra_t_to_fit_legend = 400
        plt.xlabel('ns')
        plt.ylabel('adu')
        plt.xlim(min(t),max(t) + extra_t_to_fit_legend)
        plt.legend(loc='upper right')


        plt.subplot(3,1,2)
        for wf_index, wf in enumerate(rolled_wf):
            plt.plot(new_t,wf,label='Aligned hpol %i'%wf_index)

        plt.xlabel('ns')
        plt.ylabel('adu')
        plt.xlim(min(t),max(t) + extra_t_to_fit_legend)
        plt.legend(loc='upper right')

        plt.subplot(3,1,3)
        plt.plot(power_sum,label='Beam %i\nZen = %0.1f deg\nAz = %0.1f deg'%(best_beam,hpol_beam_delays[best_beam,1],hpol_beam_delays[best_beam,2]))

        plt.xlabel('Power Sum Bins')
        plt.ylabel('Summed Power (adu^2)')
        plt.xlim(0,len(power_sum)* (1+extra_t_to_fit_legend/max(t)))
        plt.legend(loc='upper right')
        
        return fig

        
if __name__=="__main__":
    #plt.close('all')
    '''
    There are arguments here at the top that control how this script runs.  Additionally, to define the beams you want
    set you must scroll to the DEFINE BEAMS HERE section below and ensure the correct scenario is being run.  Each
    scenario has an associated 'mode' label.  If that mode label is included in 'modes' then it will be run.
    '''
    modes = ['hpol_2021']#['hpol_2018','vpol_2018']#['vpol_no_ant01']:#['hpol','vpol','vpol_no_ant0','vpol_no_ant01']:

    use_pulser=False
    max_method = 0
    
    circle_beam_centers = True #Will add circles to the maps centered on the selected beams.
    radius = 2.5 #Degrees I think?  Should eventually represent error. 
    circle_color = 'lime'
    circle_linewidth = 2
    cmap = 'plasma'
    

    #Normalize normalizes beams BEFORE they are comapared and in power units NOT linear.  EVERY beam peaks at 1.
    normalize = False #IF TRUE This will result in the max value if 1, with all other values scaled in by that.  Puts each beam on portrayed equal footing, which may not be accurate. 
    post_normalize = True #This normalizes AFTER beams are comared/summed.  So if some beams are worse, they should maintain that relative worseness.
    display_type = 'power'#'linear' #'power' is default, and will plot the resulting power sum as would be output by the daq algorithm.  linear will take those power units (which may already be normalized) and 

    '''
    Recommended settings are linear, not normalized before, post_normalize = True.  This makes beam coverage scale in
    linear units, and has relative coverage across beams be consistent rather than normalized per beam.  
    '''

    
    if display_type == 'linear':
        #The below list of conversion formulas can be used to determine how 'linear' is calculated from the power units.
        # conversion_formula = lambda power_sum, n_summed_antennas : numpy.sqrt(power_sum) # n_summed_antenna not used, just here for consistent formatting.
        # conversion_formula = lambda power_sum, n_summed_antennas : numpy.sqrt(power_sum/(16*n_summed_antennas)) #Below there is an option "turnoff_ants" that sets the signal in certain antennas to 0.  This lets you not divide by 4 if not all 4 antennas are being summed. 
        conversion_formula = lambda power_sum, n_summed_antennas : numpy.sqrt(power_sum/(16*4)) #This assumes we should always say 4 summed antennas, even if some aren't working.  n_summed_antenna not used, just here for consistent formatting.


    # BeamMaker is built on the Correlator class, and by default will use "default_deploy" as set in info.py as the calibration of choice. You can also set it as a kwarg here.  None will go to default deploy. 
    deploy_index = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'config/rtk-gps-day3-june22-2021.json')#0#None


    for hilbert in [False]:
        datapath = os.environ['BEACON_DATA']
        if use_pulser == True:
            run = 1509
            known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
            eventids = {}
            eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])
            eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])
            waveform_index_range = (2000,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
        else:
            # run = 1645
            # eventids = {}
            # eventids['hpol'] = [45]
            # eventids['vpol'] = [48]
            run = 1650
            eventids = {}
            eventids['hpol'] = [57507] #The 75th percentile in impulsivity_h event for events with impulsivity in BOTH h and v > 0.5 for run 1650
            eventids['vpol'] = [72789] #The 75th percentile in impulsivity_v event for events with impulsivity in BOTH h and v > 0.5 for run 1650

            waveform_index_range = (None,None) #Looking at the later bit of the waveform only, 10000 will cap off.  

        reader = Reader(datapath,run)
        date = dt.datetime.today().strftime('%Y-%m-%d') #appended to file names.
        # BeamMaker is built on the Correlator class, and by default will use "default_deploy" as set in info.py as the calibration of choice. You can also set it as a kwarg here.  None will go to default deploy. 
        bm = BeamMaker(reader, n_phi=4*360,range_phi_deg=(-180,180), n_theta=4*180, range_theta_deg=(0,180), waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=False,deploy_index=deploy_index)
        print('Using calibration %s'%str(bm.deploy_index)) #note that if you manually reset antenna positions this is no longer the case obviously.             

        deploy_root_name = os.path.split(str(bm.deploy_index))[1].replace('.json','')

        all_fig = []
        all_ax = []
        center_dir = 'E'
        sample_delays = {}
        for mode in modes:
            print('\nMode = %s\n'%mode)
            #beam_thetas_deg = numpy.linspace(5,110,4)

            #########################
            ### DEFINE BEAMS HERE ###
            #########################

            if False:
                n_beams_total = 48 #This is an aim (lower bound),not a guarentee
                n_theta = 4
                up_theta = 5
                down_theta = 105
                linspace_cos_vals = numpy.linspace(numpy.cos(numpy.deg2rad(up_theta)),numpy.cos(numpy.deg2rad(down_theta)),n_theta)
                thetas_deg = numpy.rad2deg(numpy.arccos(linspace_cos_vals))
                phis_per_theta = numpy.ceil(n_beams_total*(numpy.sin(numpy.deg2rad(thetas_deg))/sum(numpy.sin(numpy.deg2rad(thetas_deg)))))
            else:
                if mode == 'hpol':
                    thetas_deg = numpy.array([2,10,25,40,68,90,100])#numpy.array([2,15,30,50,80,100])
                    phis_per_theta = numpy.array([1,5,7,9,12,2,15])#numpy.array([1,5,8,10,8,15])
                    phis_angular_ranges = numpy.array([[-48,48],[-45,45],[-47,47],[-48,48],[-49,49],[-48,48],[-42,42]])
                    turnoff_ants=None
                    year = 'any'

                    outfile_name = '%s_beam_sample_delays_hilbert-%s_deploy-%s_%s_%s.csv'%(mode,str(hilbert),str(deploy_root_name),display_type,date)
                    savefig_text = '%s_coverage_map_%i_beams_hilbert-%s_deploy-%s_%s_%s.jpg'%(mode,sum(phis_per_theta),str(hilbert),str(deploy_root_name),display_type,date)
                    '''
                    thetas_deg = numpy.array([100])#
                    phis_per_theta = numpy.array([1])
                    phis_angular_ranges = numpy.array([[-42,42]])
                    '''
                elif mode == 'vpol':
                    thetas_deg = numpy.array([2,10,25,40,68,90,100])#numpy.array([2,15,30,50,80,100])
                    phis_per_theta = numpy.array([1,5,7,9,12,2,15])#numpy.array([1,5,8,10,8,15])
                    phis_angular_ranges = numpy.array([[-48,48],[-45,45],[-47,47],[-48,48],[-49,49],[-48,48],[-42,42]])
                    turnoff_ants=None
                    year = 'any'
                    
                    outfile_name = '%s_beam_sample_delays_hilbert-%s_deploy-%s_%s_%s.csv'%(mode,str(hilbert),str(deploy_root_name),display_type,date)
                    savefig_text = '%s_coverage_map_%i_beams_hilbert-%s_deploy-%s_%s_%s.jpg'%(mode,sum(phis_per_theta),str(hilbert),str(deploy_root_name),display_type,date)
                elif mode == 'vpol_no_ant0':
                    thetas_deg = numpy.array([2,10,25,40,68,90,100])#numpy.array([2,15,30,50,80,100])
                    phis_per_theta = numpy.array([1,5,7,9,12,2,15])#numpy.array([1,5,8,10,8,15])
                    phis_angular_ranges = numpy.array([[-48,48],[-45,45],[-47,47],[-48,48],[-49,49],[-48,48],[-42,42]])
                    turnoff_ants=[0]
                    year = 'any'
                    
                    outfile_name = '%s_beam_sample_delays_hilbert-%s_deploy-%s_%s_%s.csv'%(mode,str(hilbert),str(deploy_root_name),display_type,date)
                    savefig_text = '%s_coverage_map_%i_beams_no_ant0_hilbert-%s_deploy-%s_%s_%s.jpg'%(mode,sum(phis_per_theta),str(hilbert),str(deploy_root_name),display_type,date)
                    mode = 'vpol'
                elif mode == 'vpol_no_ant01':
                    thetas_deg = numpy.array([2,10,25,40,68,90,100])#numpy.array([2,15,30,50,80,100])
                    phis_per_theta = numpy.array([1,5,7,9,12,2,15])#numpy.array([1,5,8,10,8,15])
                    phis_angular_ranges = numpy.array([[-48,48],[-45,45],[-47,47],[-48,48],[-49,49],[-48,48],[-42,42]])
                    turnoff_ants=[0,1]
                    year = 'any'
                    
                    outfile_name = '%s_beam_sample_delays_hilbert-%s_deploy-%s_%s_%s.csv'%(mode,str(hilbert),str(deploy_root_name),display_type,date)
                    savefig_text = '%s_coverage_map_%i_beams_no_ant01_hilbert-%s_deploy-%s_%s_%s.jpg'%(mode,sum(phis_per_theta),str(hilbert),str(deploy_root_name),display_type,date)
                    mode = 'vpol'
                elif mode == 'hpol_2018':
                    thetas_deg = numpy.array([-40,-20, 0, 20, 40]) + 90.
                    phis_per_theta = numpy.array([4,4,4,4,4])
                    phis_angular_ranges = numpy.array([[-40,40], [-40,40], [-40,40], [-40,40], [-40,40]])
                    turnoff_ants = None
                    year = 'any'
                    
                    outfile_name = '%s_beam_sample_delays_hilbert-%s_deploy-%s_%s_%s.csv'%(mode,str(hilbert),str(deploy_root_name),display_type,date)
                    savefig_text = '%s_coverage_map_%i_beams_hilbert-%s_deploy-%s_%s_%s.jpg'%(mode,sum(phis_per_theta),str(hilbert),str(deploy_root_name),display_type,date)
                    mode = 'hpol'
                elif mode == 'vpol_2018':
                    thetas_deg = numpy.array([-40,-20, 0, 20, 40]) + 90.
                    phis_per_theta = numpy.array([4,4,4,4,4])
                    phis_angular_ranges = numpy.array([[-40,40], [-40,40], [-40,40], [-40,40], [-40,40]])
                    turnoff_ants = None
                    year = 'any'
                    
                    outfile_name = '%s_beam_sample_delays_hilbert-%s_deploy-%s_%s_%s.csv'%(mode,str(hilbert),str(deploy_root_name),display_type,date)
                    savefig_text = '%s_coverage_map_%i_beams_hilbert-%s_deploy-%s_%s_%s.jpg'%(mode,sum(phis_per_theta),str(hilbert),str(deploy_root_name),display_type,date)
                    mode = 'vpol'
                elif mode == 'hpol_2019':
                    thetas_deg = numpy.array([-40,-20, 0, 20, 40])+ 90.
                    phis_per_theta = numpy.array([4,4,4,4,4])
                    phis_angular_ranges = numpy.array([[-40,40], [-40,40], [-40,40], [-40,40], [-40,40]])
                    #thetas_deg = numpy.array([ 0])+90.
                    #phis_per_theta = numpy.array([1])
                    #phis_angular_ranges = numpy.array([[-40,40]])
                    
                    turnoff_ants = None
                    
                    outfile_name = '%s_beam_sample_delays_hilbert-%s_deploy-%s_%s_%s.csv'%(mode,str(hilbert),str(deploy_root_name),display_type,date)
                    savefig_text = '%s_coverage_wfs_%i_beams_hilbert-%s_deploy-%s_%s_%s.jpg'%(mode,sum(phis_per_theta),str(hilbert),str(deploy_root_name),display_type,date)
                    savefig_text1 = '%s_coverage_map_%i_beams_hilbert-%s_deploy-%s_%s_%s.jpg'%(mode,sum(phis_per_theta),str(hilbert),str(deploy_root_name),display_type,date)
                    
                    mode = 'hpol'
                    year = '2019'
                    pol = 0
                    
                elif mode == 'vpol_2019':
                    thetas_deg = numpy.array([-40,-20, 0, 20, 40])+ 90.
                    phis_per_theta = numpy.array([4,4,4,4,4])
                    phis_angular_ranges = numpy.array([[-40,40], [-40,40], [-40,40], [-40,40], [-40,40]])
                    #thetas_deg = numpy.array([ 0])+90.
                    #phis_per_theta = numpy.array([1])
                    #phis_angular_ranges = numpy.array([[-40,40]])
                    
                    turnoff_ants = None
                    
                    outfile_name = '%s_beam_sample_delays_hilbert-%s_deploy-%s_%s_%s.csv'%(mode,str(hilbert),str(deploy_root_name),display_type,date)
                    savefig_text = '%s_coverage_wfs_%i_beams_hilbert-%s_deploy-%s_%s_%s.jpg'%(mode,sum(phis_per_theta),str(hilbert),str(deploy_root_name),display_type,date)
                    savefig_text1 = '%s_coverage_map_%i_beams_hilbert-%s_deploy-%s_%s_%s.jpg'%(mode,sum(phis_per_theta),str(hilbert),str(deploy_root_name),display_type,date)
                    
                    mode = 'vpol'
                    year = '2019'
                    pol = 1

                elif mode == 'hpol_2021':
                    elev_deg = numpy.concatenate([numpy.linspace(10,70,7),numpy.linspace(15,60,4),[0]])
                    thetas_deg = 90 - elev_deg
                    phis_per_theta = numpy.concatenate([numpy.ones(7), 2*numpy.ones(4), [5]]).astype(int)
                    phis_angular_ranges = numpy.vstack((numpy.tile([-10,10],(7,1)),numpy.tile([-20,20],(4,1)) , [-50, 50] ))
                    turnoff_ants=None
                    mode = 'hpol'
                    year = '2021'

                    outfile_name = '%s_beam_sample_delays_hilbert-%s_deploy-%s_%s_%s.csv'%(mode,str(hilbert),deploy_root_name,display_type,date)
                    savefig_text = '%s_coverage_map_%i_beams_hilbert-%s_deploy-%s_%s_%s.jpg'%(mode,sum(phis_per_theta),str(hilbert),deploy_root_name,display_type,date)

            # Path(outfile_name).touch()
            # Path(savefig_text).touch()

            beam_thetas_deg = numpy.array([])#numpy.zeros(sum(phis_per_theta))
            beam_phis_deg = numpy.array([])#numpy.zeros(sum(phis_per_theta))
            for theta_index, theta in enumerate(thetas_deg):
                beam_thetas_deg = numpy.append(beam_thetas_deg,numpy.array([theta]*int(phis_per_theta[theta_index])))
                if int(phis_per_theta[theta_index]) == 1:
                    beam_phis_deg = numpy.append(beam_phis_deg,0.0)
                else:
                    beam_phis_deg = numpy.append(beam_phis_deg,numpy.linspace(phis_angular_ranges[theta_index][0],phis_angular_ranges[theta_index][1],int(phis_per_theta[theta_index])))

            print(beam_thetas_deg)
            print(beam_phis_deg)
            
            all_max_powers = numpy.zeros((len(beam_thetas_deg), bm.n_theta, bm.n_phi))

            sample_delays[mode] = numpy.zeros((len(beam_thetas_deg), 7))
            sample_delays[mode][:,0] = numpy.arange(len(beam_thetas_deg))
            sample_delays[mode][:,1] = beam_thetas_deg
            sample_delays[mode][:,2] = beam_phis_deg
            
            
            eventid = eventids[mode][0]
            
            if year == '2019':
                finame_beam_delays = os.environ['BEACON_ANALYSIS_DIR'] + 'tools/beam_definitions/delays_current.csv'
                beam_delays = info.loadBeamDelays(finame_hpol_delays=finame_beam_delays, finame_vpol_delays=finame_beam_delays, reset_rel_to_ant0=True)[pol]
                nbeams = numpy.shape(beam_delays)[0]
                
                figy = bm.calculateBeams(reader, eventid, mode=mode,finame_hpol_delays=finame_beam_delays)
                figy.savefig(savefig_text)
                
                
                for i in range(nbeams):
                    bm.overwriteDelays(mode, beam_delays[i,4], beam_delays[i,5], beam_delays[i,6])
                    
                    mean_corr_values, figy1, axy1 = bm.map(eventid, mode, include_baselines=numpy.array([0,1,2,3,4,5]), 
                        plot_map=True, plot_corr=True, hilbert=False, interactive=False, 
                        max_method=None, waveforms=None, verbose=True, mollweide=False, zenith_cut_ENU=None, 
                        zenith_cut_array_plane=None, center_dir=center_dir, circle_zenith=None, circle_az=None, 
                        radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False)
                    figy1.savefig(savefig_text1)

                    all_fig.append(figy1)
                    all_ax.append(axy1)
            
            for beam_index in range(len(beam_thetas_deg)):
                print('%i/%i               '%(beam_index + 1,len(beam_thetas_deg)))
                forced_deltats = None 
                if year == '2019':
                    # this sends the time delays as read from the file (and adjusted to be 0 at ant0)
                    # to the BeamMaker map
                    forced_deltats = beam_delays[beam_index,4:]
                
                all_max_powers[beam_index], \
                sample_delays[mode][beam_index][3], \
                sample_delays[mode][beam_index][4], \
                sample_delays[mode][beam_index][5], \
                sample_delays[mode][beam_index][6] = bm.makeFakeBeamMap(eventid, mode, \
                                                        numpy.deg2rad(beam_thetas_deg[beam_index]), numpy.deg2rad(beam_phis_deg[beam_index]), \
                                                        plot_map=False, plot_corr=False, hilbert=hilbert, \
                                                        normalize=normalize,savefig=True, \
                                                        savefig_text='beam%i'%beam_index,turnoff_ants=turnoff_ants,
                                                        forced_deltats = forced_deltats)
                    #print('delay_ch0_by: %i\ndelay_ch1_by: %i\ndelay_ch2_by: %i\ndelay_ch3_by: %i'%(sample_delays[mode][beam_index][3], sample_delays[mode][beam_index][4], sample_delays[mode][beam_index][5], sample_delays[mode][beam_index][6]))
        

            numpy.savetxt(outfile_name,sample_delays[mode], delimiter=',',fmt=['%i','%.3f','%.3f','%i','%i','%i','%i'],header='beam_index , zenith, azimuth, delay_ant0_by, delay_ant1_by, delay_ant2_by, delay_ant3_by')

            coverage = numpy.max(all_max_powers,axis=0)

            if display_type == 'linear':
                if turnoff_ants is None:
                    turnoff_ants = []
                coverage = conversion_formula(coverage,4 - len(turnoff_ants)) #conversion formula can change, see definition above. 
            if post_normalize:
                coverage = coverage/numpy.max(coverage)


            fig = plt.figure(figsize=(16,9))
            fig.canvas.set_window_title('Beam Coverage Map')
            ax = fig.add_subplot(1,1,1)
            if normalize == True or post_normalize == True:
                im = ax.imshow(coverage, interpolation='none', extent=[min(bm.phis_deg),max(bm.phis_deg),90 - max(bm.thetas_deg),90 - min(bm.thetas_deg)],cmap=cmap,vmin=0,vmax=1) #cmap=plt.cm.jet)
            else:
                im = ax.imshow(coverage, interpolation='none', extent=[min(bm.phis_deg),max(bm.phis_deg),90 - max(bm.thetas_deg),90 - min(bm.thetas_deg)],cmap=cmap) #cmap=plt.cm.jet)
            cbar = fig.colorbar(im)
            plt.title('Beam Coverage Map')
            cbar_label = 'Max Beam Sensitivity'

            if normalize == True:
                cbar_label += '\nEach Beam Normalized Before Combined'
            if post_normalize == True:
                cbar_label += 'Normalized After Beams Combined'
            cbar.set_label(cbar_label)

            plt.xlabel('Azimuth Angle (Degrees)')
            plt.ylabel('Zenith Angle (Degrees)')

            add_ground_mask = True
            if add_ground_mask:
                mollweide = False
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
                    roll = numpy.argmin(abs(bm.phis_rad - azimuth_offset_rad))
                elif center_dir.upper() == 'W':
                    center_dir_full = 'West'
                    azimuth_offset_rad = numpy.pi #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = 180 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From West = 0 deg, South = 90 deg)'
                    roll = len(bm.phis_rad)//2
                elif center_dir.upper() == 'S':
                    center_dir_full = 'South'
                    azimuth_offset_rad = -numpy.pi/2 #This is subtracted from the xaxis to roll it effectively.
                    azimuth_offset_deg = -90 #This is subtracted from the xaxis to roll it effectively.
                    xlabel = 'Azimuth (From South = 0 deg, East = 90 deg)'
                    roll = numpy.argmin(abs(bm.phis_rad - azimuth_offset_rad))

                if mode == 'hpol':
                    plane_xy = bm.getPlaneZenithCurves(bm.n_hpol.copy(), 'hpol', 90.0, azimuth_offset_deg=azimuth_offset_deg)
                elif mode == 'vpol':
                    plane_xy = bm.getPlaneZenithCurves(bm.n_vpol.copy(), 'vpol', 90.0, azimuth_offset_deg=azimuth_offset_deg)
                elif mode == 'all':
                    plane_xy = bm.getPlaneZenithCurves(bm.n_all.copy(), 'all', 90.0, azimuth_offset_deg=azimuth_offset_deg) #This is janky

                #Plot array plane 0 elevation curve.
                im = bm.addCurveToMap(im, plane_xy,  mollweide=mollweide, linewidth = bm.min_elevation_linewidth, color='k')

                x = plane_xy[0]
                y1 = plane_xy[1]

                y2 = -90 * numpy.ones_like(plane_xy[0])#lower_plane_xy[1]

                ax.fill_between(x, y1, y2, where=y2 <= y1,facecolor='#9DC3E6', interpolate=True,alpha=1)#'#EEC6C7'
                ax.set_ylim(-90,90)


            if circle_beam_centers == True:
                for beam_index in range(len(beam_thetas_deg)):
                    circle = plt.Circle((beam_phis_deg[beam_index], 90 - beam_thetas_deg[beam_index]), radius, edgecolor=circle_color,linewidth=circle_linewidth,fill=False)
                    ax.add_artist(circle)


            fig.savefig(savefig_text.replace('.jpg','.png'),dpi=108*4, figsize=(16,9))

            all_fig.append(fig)
            all_ax.append(ax)