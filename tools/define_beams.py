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
        This will return arrival times (at board, including cable delays) for antennas 1,2,3 relative to antenna 0.  So if ant 1 signal arrives
        later than ant 0 it will arive at a positive number of that many ns.   
        '''
        try:
            cable_delays = info.loadCableDelays()['hpol']
            cable_delays = cable_delays - cable_delays[0] #Relative to antenna 0, how long it takes for the signal to arrive. 
            #cable_delays = numpy.zeros(4)# THIS WILL IGNORE CABLE DELAYS AND SHOULD BE DONE FOR TESTING PURPOSES ONLY.
            self.t_hpol_0subtract1_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , -self.t_hpol_0subtract1 + cable_delays[1])
            self.t_hpol_0subtract2_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , -self.t_hpol_0subtract2 + cable_delays[2])
            self.t_hpol_0subtract3_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , -self.t_hpol_0subtract3 + cable_delays[3])

            cable_delays = info.loadCableDelays()['vpol']
            cable_delays = cable_delays - cable_delays[0] #Relative to antenna 0, how long it takes for the signal to arrive. 
            #cable_delays = numpy.zeros(4)# THIS WILL IGNORE CABLE DELAYS AND SHOULD BE DONE FOR TESTING PURPOSES ONLY.
            self.t_vpol_0subtract1_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , -self.t_vpol_0subtract1 + cable_delays[1])
            self.t_vpol_0subtract2_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , -self.t_vpol_0subtract2 + cable_delays[2])
            self.t_vpol_0subtract3_interp = scipy.interpolate.RectBivariateSpline( self.thetas_rad , self.phis_rad , -self.t_vpol_0subtract3 + cable_delays[3])
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

    def makeFakeBeamMap(self, eventid, pol, beam_theta, beam_phi, plot_map=True, plot_corr=False, hilbert=False, normalize=False, savefig=False, savefig_text=None, turnoff_ants=None):
        '''
        This will generate fake signals from each of the directions normally available in the correlation map.  It will then apply beamforming assuming 
        the beam is centered on theta, phi. 
        '''
        try:
            t_1, t_2, t_3 = bm.interpolateTimeGrid(pol, beam_theta, beam_phi)
            t_0 = 0.0
            t_1, t_2, t_3 = bm.roundToNearestSample(numpy.array([t_1, t_2, t_3]))
            
            rolls = numpy.array([0, -(t_1/self.dt_resampled).astype(int) , -(t_2/self.dt_resampled).astype(int) , -(t_3/self.dt_resampled).astype(int)]) #How much to roll a signal to account for arrival direction of beam.  (negative how many you expect them to be delayed)

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

                    temp_waveforms[1] = numpy.roll(padded_wf[1],numpy.round(temp_t_1/self.dt_resampled).astype(int) + rolls[1])
                    temp_waveforms[2] = numpy.roll(padded_wf[2],numpy.round(temp_t_2/self.dt_resampled).astype(int) + rolls[2])
                    temp_waveforms[3] = numpy.roll(padded_wf[3],numpy.round(temp_t_3/self.dt_resampled).astype(int) + rolls[3])

                    summed_waveforms = numpy.sum(temp_waveforms,axis=0)
                    power = summed_waveforms**2
                    power_sum = numpy.sum(power[binned_8_indices_A],axis=1) + numpy.sum(power[binned_8_indices_B],axis=1)
                    max_powers[theta_index,phi_index] = numpy.max(power_sum)

            integer_delays = (rolls - min(rolls)).astype(int)
            
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

if __name__=="__main__":
    #plt.close('all')
    use_pulser=False
    max_method = 0
    normalize = True

    for hilbert in [False,True]:
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

        sample_delays = {}
        for mode in ['hpol','vpol','vpol_no_ant0']:
            print('\nMode = %s\n'%mode)
            #beam_thetas_deg = numpy.linspace(5,110,4)
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
                    outfile_name = '%s_beam_sample_delays_hilbert-%s.csv'%(mode,str(hilbert))
                    savefig_text = '%s_coverage_map_%i_beams_hilbert-%s.jpg'%(mode,sum(phis_per_theta),str(hilbert))
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
                    outfile_name = '%s_beam_sample_delays_hilbert-%s.csv'%(mode,str(hilbert))
                    savefig_text = '%s_coverage_map_%i_beams_hilbert-%s.jpg'%(mode,sum(phis_per_theta),str(hilbert))
                elif mode == 'vpol_no_ant0':
                    thetas_deg = numpy.array([2,10,25,40,68,90,100])#numpy.array([2,15,30,50,80,100])
                    phis_per_theta = numpy.array([1,5,7,9,12,2,15])#numpy.array([1,5,8,10,8,15])
                    phis_angular_ranges = numpy.array([[-48,48],[-45,45],[-47,47],[-48,48],[-49,49],[-48,48],[-42,42]])
                    turnoff_ants=[0]
                    outfile_name = '%s_beam_sample_delays_hilbert-%s.csv'%(mode,str(hilbert))
                    savefig_text = '%s_coverage_map_%i_beams_no_ant0_hilbert-%s.jpg'%(mode,sum(phis_per_theta),str(hilbert))
                    mode = 'vpol'

            beam_thetas_deg = numpy.array([])#numpy.zeros(sum(phis_per_theta))
            beam_phis_deg = numpy.array([])#numpy.zeros(sum(phis_per_theta))
            for theta_index, theta in enumerate(thetas_deg):
                beam_thetas_deg = numpy.append(beam_thetas_deg,numpy.array([theta]*int(phis_per_theta[theta_index])))
                if int(phis_per_theta[theta_index]) == 1:
                    beam_phis_deg = numpy.append(beam_phis_deg,0.0)
                else:
                    beam_phis_deg = numpy.append(beam_phis_deg,numpy.linspace(phis_angular_ranges[theta_index][0],phis_angular_ranges[theta_index][1],int(phis_per_theta[theta_index])))

            bm = BeamMaker(reader, n_phi=360,range_phi_deg=(-180,180), n_theta=180, range_theta_deg=(0,180), waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=False)
            all_max_powers = numpy.zeros((len(beam_thetas_deg), bm.n_theta, bm.n_phi))

            sample_delays[mode] = numpy.zeros((len(beam_thetas_deg), 7))
            sample_delays[mode][:,0] = numpy.arange(len(beam_thetas_deg))
            sample_delays[mode][:,1] = beam_thetas_deg
            sample_delays[mode][:,2] = beam_phis_deg
            
            
            eventid = eventids[mode][0]

            for beam_index in range(len(beam_thetas_deg)):
                print('%i/%i               '%(beam_index + 1,len(beam_thetas_deg)))
                all_max_powers[beam_index], sample_delays[mode][beam_index][3], sample_delays[mode][beam_index][4], sample_delays[mode][beam_index][5], sample_delays[mode][beam_index][6] = bm.makeFakeBeamMap(eventid, mode, numpy.deg2rad(beam_thetas_deg[beam_index]), numpy.deg2rad(beam_phis_deg[beam_index]), plot_map=False, plot_corr=False, hilbert=hilbert, normalize=normalize,savefig=True, savefig_text='beam%i'%beam_index,turnoff_ants=turnoff_ants)
                #print('delay_ch0_by: %i\ndelay_ch1_by: %i\ndelay_ch2_by: %i\ndelay_ch3_by: %i'%(sample_delays[mode][beam_index][3], sample_delays[mode][beam_index][4], sample_delays[mode][beam_index][5], sample_delays[mode][beam_index][6]))
            

            numpy.savetxt(outfile_name,sample_delays[mode], delimiter=',',fmt=['%i','%.3f','%.3f','%i','%i','%i','%i'],header='beam_index , zenith, azimuth, delay_ant0_by, delay_ant1_by, delay_ant2_by, delay_ant3_by')

            coverage = numpy.max(all_max_powers,axis=0)

            fig = plt.figure(figsize=(16,9))
            fig.canvas.set_window_title('Beam Coverage Map')
            ax = fig.add_subplot(1,1,1)
            if normalize == True:
                im = ax.imshow(coverage, interpolation='none', extent=[min(bm.phis_deg),max(bm.phis_deg),max(bm.thetas_deg),min(bm.thetas_deg)],cmap='plasma',vmin=0,vmax=1) #cmap=plt.cm.jet)
            else:
                im = ax.imshow(coverage, interpolation='none', extent=[min(bm.phis_deg),max(bm.phis_deg),max(bm.thetas_deg),min(bm.thetas_deg)],cmap='plasma') #cmap=plt.cm.jet)
            cbar = fig.colorbar(im)
            plt.title('Beam Coverage Map')
            if normalize == True:
                cbar.set_label('Max Beam Sensitivity (Each Beam Normalized In Advance)')
            else:
                cbar.set_label('Max Beam Sensitivity')
            plt.xlabel('Azimuth Angle (Degrees)')
            plt.ylabel('Zenith Angle (Degrees)')


            radius = 2.5 #Degrees I think?  Should eventually represent error. 

            for beam_index in range(len(beam_thetas_deg)):
                circle = plt.Circle((beam_phis_deg[beam_index], beam_thetas_deg[beam_index]), radius, edgecolor='lime',linewidth=2,fill=False)
                ax.add_artist(circle)

                        
            fig.savefig(savefig_text)

            all_fig.append(fig)
            all_ax.append(ax)