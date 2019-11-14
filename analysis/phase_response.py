'''
This file is intended to determine the phase response of the boards based on data
that Eric took before deployment. 


I was working on this to see if the phase response could be used to give a better
time delay measurement.  But I was having trouble trying to resample the phase response
properly and eventually gave up as the difference is so small between channels that it
shouldn't make a difference. 

Clearly I was doing something wrong.  
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv
import inspect
import glob

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
import tools.field_fox as ff
import tools.constants as constants
from tools.correlator import Correlator
import matplotlib.pyplot as plt
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

def loadResponse2ndStage():
    '''
    This will load the phase response for the 2nd stage.
    '''
    try:
        response = []
        for channel in numpy.arange(8):
            phase_filename = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/' + 'beacon_2ndStage_ch%i_s21phase.csv'%channel
            mag_filename = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/' + 'beacon_2ndStage_ch%i_s21mag.csv'%channel
            freqs, phase = ff.readerFieldFox(phase_filename,header=17) #Phase in degrees
            freqs, logmag = ff.readerFieldFox(mag_filename,header=17) 
            mag = ff.logMagToLin(logmag)
            real,imag = ff.magPhaseToReIm(mag,phase) #phase in degrees
            response.append(numpy.add(real,imag*1j))
        return freqs, response 
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def loadInterpolatedResponse2ndStage(goal_freqs, plot=False):
    '''
    This does the same as loadResponse2ndStage but will perform interpolation
    to match the frequencies with the given freqs.
    '''
    try:
        freqs, response = loadResponse2ndStage()
        
        if max(goal_freqs) > max(freqs):
            #To help with interpolation later, I am zero padding the current response so that
            #It has a larger BW than the goal (slightly)
            N_add = numpy.ceil((max(goal_freqs) - max(freqs))/(freqs[1] - freqs[0])).astype(int)
            freqs = numpy.arange(len(freqs) + N_add)*(freqs[1] - freqs[0])
            temp_response = numpy.zeros((8,response.shape[1]+N_add))
            temp_response[:,0:response.shape[1]] = response
            response = temp_response


        T_goal = 1.0/numpy.diff(goal_freqs)[0]
        dT_goal = 1.0/(2.0*goal_freqs[-1])
        N_goal = T_goal/dT_goal
        
        T_current = 1.0/numpy.diff(freqs)[0]
        dT_current = 1.0/(2.0*freqs[-1])
        N_current = T_current/dT_current


        if T_goal > T_current:
            #Zeropad T_current to get matching frequency
            response_td = numpy.fft.irfft(response,axis=1)
            N_add = numpy.floor((T_goal-T_current)/dT_current).astype(int)

            new_response_td = numpy.zeros((8,response_td.shape[1]+N_add))
            new_times = numpy.arange(response_td.shape[1]+N_add)*dT_current
            tk = scipy.signal.tukey(response_td.shape[1], alpha=0.02) 


            new_response_td[:,0:response_td.shape[1]] = numpy.multiply(response_td,tk)

            new_response = numpy.fft.rfft(new_response_td,axis=1) #Probably need some normalization. 
            new_freqs = numpy.fft.rfftfreq(len(new_times),new_times[1]-new_times[0])
            
            plt.figure()
            for channel in range(8):
                plt.subplot(3,1,1)
                plt.plot(new_times*1e9,new_response_td[channel])
                plt.subplot(3,1,2)
                plt.plot(freqs/1e6,abs(response[channel]))
                plt.scatter(new_freqs/1e6,abs(new_response[channel]))

                plt.subplot(3,1,3)
                plt.plot(freqs/1e6,numpy.unwrap(numpy.angle(response[channel]))*180./numpy.pi)
                plt.scatter(new_freqs/1e6,numpy.unwrap(numpy.angle(new_response[channel]))*180./numpy.pi)


            import pdb; pdb.set_trace()
            #import pdb; pdb.set_trace()

            out_response = numpy.zeros((8,len(goal_freqs)))
            for channel in range(8):
                out_response[channel] = scipy.interpolate.interp1d(new_freqs,new_response[channel])(goal_freqs) #To account for slight mismatches

        else:
            print('Not handled yet.')


        if plot == True:
            for channel in range(8):
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(freqs/1e6,abs(response[channel]),label='Ch%i Original Sampling'%channel)
                plt.plot(goal_freqs/1e6,abs(out_response[channel]),label='Ch%i New Sampling'%channel)
                plt.xlabel('MHz')
                plt.ylabel('Linear Mag')
                plt.legend()

                plt.subplot(2,1,2)
                plt.plot(freqs/1e6,numpy.unwrap(numpy.angle(response[channel]))*180./numpy.pi,label='Ch%i Original Sampling'%channel)
                plt.plot(goal_freqs/1e6,numpy.unwrap(numpy.angle(out_response[channel]))*180./numpy.pi,label='Ch%i New Sampling'%channel)
                plt.xlabel('MHz')
                plt.ylabel('Unwrapped Phase')
                plt.legend()

        return goal_freqs, out_response 
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)




def loadInterpolatedPhaseResponse2ndStage(goal_freqs, plot=False):
    '''
    This does the same as loadPhaseResponse2ndStage but will perform interpolation
    to match the frequencies with the given freqs.

    This can then be called by the filter creation tool and applied as part of that I think.

    Then it is applied in the same way as a filter.  Though with e^-itheta.
    '''
    try:
        freqs, response = loadInterpolatedResponse2ndStage(goal_freqs, plot=True)
        phase = numpy.zeros(response.shape)
        for channel in range(8):
            phase[channel] = numpy.unwrap(numpy.angle(response[channel]))
            phase[channel] = phase[channel] - phase[channel][freqs/1e6 >= 50][0] #Align at 50Mhz

        return freqs, phase

    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)



if __name__ == '__main__':
    try:
        plt.close('all')
        datapath = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/'
        plot = True
        if plot == True:
            plt.figure()
            plt.subplot(2,1,1)
        components = ['2ndStage','preamp']
        for comp_index, component in enumerate(components):
            #s21_infiles = glob.glob(datapath + '/%s*s21*.csv'%component)
            #s21_files_groups = numpy.unique([f.replace('mag.csv','.csv').replace('phase.csv','.csv') for f in s21_infiles])
            if component == '2ndStage':
                channels = numpy.arange(8)
            elif component == 'preamp':
                channels = numpy.arange(1,9)

            for channel in channels:
                if component == '2ndStage':
                    filename = datapath + 'beacon_2ndStage_ch%i_s21phase.csv'%channel
                    label = 'Channel %i'%channel
                elif component == 'preamp':
                    filename = datapath + 'beacon_preamp_%i_s21phase.csv'%(channel)
                    label = 'Board %i'%channel

                freqs, phase = ff.readerFieldFox(filename,header=17) #
                phase = numpy.rad2deg(numpy.unwrap(numpy.deg2rad(phase)))
                phase = phase - phase[freqs/1e6 >= 50][0] #Align at 50Mhz

                if plot == True:
                    plt.subplot(2,1,comp_index+1)
                    plt.plot(freqs/1e6, phase,label=label)
                    plt.ylabel('Unwrapped Phase')
                    plt.legend(loc='upper right')
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        datapath = os.environ['BEACON_DATA']
        runs = numpy.array([1645])#numpy.arange(1645,1700)

        for run in runs:
            run = int(run)

            reader = Reader(datapath,run)
            upsample = 2**12
            cor = Correlator(reader,  upsample=upsample, n_phi=180, n_theta=180, waveform_index_range=(None,None),crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=True)
            t = cor.t()/1e9
            goal_freqs = numpy.fft.rfftfreq(len(t),t[1]-t[0])
            freqs, phase = loadInterpolatedPhaseResponse2ndStage(goal_freqs, plot=True)



    except Exception as e:
        print('main()')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
