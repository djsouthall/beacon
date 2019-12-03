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
import matplotlib.pyplot as plt
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()


def loadInterpolatedPhaseResponseMeanPreamp(goal_freqs,  plot=False):
    '''
    This will load the phase response for the preamp.

    Because we don't know which board goes to which ant, I am just
    taking the mean.  They are all similar anyways.  Hopefully overall
    this will make the signals more impulsive. 

    Returns response in rads
    '''
    try:
        phase_response = numpy.zeros((8,len(goal_freqs)))
        if plot == True:
            plt.figure()
        for channel in numpy.arange(8):
            #Below is loading in the signals and making them into complex numbers
            phase_filename = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/' + 'beacon_preamp_%i_s21phase.csv'%(channel+1)
            mag_filename = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/' + 'beacon_preamp_%i_s21mag.csv'%(channel+1)
            freqs, phase = ff.readerFieldFox(phase_filename,header=17) #Phase in degrees
            freqs, logmag = ff.readerFieldFox(mag_filename,header=17)
            if False:
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(freqs,phase)
                plt.subplot(2,1,2)
                plt.plot(freqs,logmag)
                import pdb; pdb.set_trace() 
            phase = numpy.unwrap(numpy.deg2rad(phase))
            mag = ff.logMagToLin(logmag)
            real,imag = ff.magPhaseToReIm(mag,numpy.rad2deg(phase)) #needs phase in degrees

            response_fft = numpy.add(real,imag*1j)

            #Now with the response I try to extend it to zero (added ~ 3 sample, as lowest freq is 2Mhz and sampling of 0.62 MHz)
            freqs_to_zero = numpy.linspace(0,freqs[-1],len(freqs)+3,endpoint=True) #The original did not include 0 Mhz, which I think messes up ffts. 
            response_fft_to_zero = scipy.interpolate.interp1d(numpy.append(0,freqs),numpy.append(0,response_fft),fill_value=0.0,bounds_error=False,kind='linear')(freqs_to_zero)
            #tukey = scipy.signal.tukey(len(response_fft_to_zero), alpha=0.005, sym=True)
            #response_fft_to_zero = numpy.multiply(tukey,response_fft_to_zero)

            output_phase = numpy.unwrap(numpy.angle(response_fft_to_zero)) #radians
            #output_phase = output_phase - output_phase[freqs_to_zero/1e6 >= 50][0] #Align at 50Mhz
            output_phase = scipy.interpolate.interp1d(freqs_to_zero,output_phase,fill_value=0.0,bounds_error=False,kind='linear')(goal_freqs)
            phase_response[channel] = output_phase #radians
            
            if plot == True:
                #phase = phase - phase[freqs/1e6 >= 50][0] #Align at 50Mhz
                plt.plot(freqs,numpy.rad2deg(phase),label='Board %i Raw'%(channel+1))
                #plt.plot(freqs_to_zero,output_phase,label='Board %i Raw'%(channel+1))
                #plot_phase = output_phase - output_phase[goal_freqs/1e6 >= 50][0] #Align at 50Mhz
                plot_phase=output_phase
                plt.scatter(goal_freqs,numpy.rad2deg(plot_phase),s=4,label='Board %i Interp'%(channel+1))
                plt.xlim(0,max(freqs))

        phase_response = numpy.mean(phase_response,axis=0)
        if plot == True:
            plt.plot(goal_freqs,numpy.rad2deg(phase_response),label='Average')
            plt.xlim(0,max(freqs))
            plt.legend()

        return goal_freqs, phase_response 
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def loadInterpolatedPhaseResponse2ndStage(goal_freqs, upsample = 2**14, plot=False):
    '''
    This will load the phase response for the 2nd stage.

    Returns response in rads
    '''
    try:
        phase_response = numpy.zeros((8,len(goal_freqs)))
        if plot == True:
            plt.figure()
        for channel in numpy.arange(8):
            
            #Below is loading in the signals and making them into complex numbers
            phase_filename = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/' + 'beacon_2ndStage_ch%i_s21phase.csv'%channel
            mag_filename = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/' + 'beacon_2ndStage_ch%i_s21mag.csv'%channel
            freqs, phase = ff.readerFieldFox(phase_filename,header=17) #Phase in degrees
            freqs, logmag = ff.readerFieldFox(mag_filename,header=17)
            if False:
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(freqs,phase)
                plt.subplot(2,1,2)
                plt.plot(freqs,logmag)
                import pdb; pdb.set_trace() 
            phase = numpy.unwrap(numpy.deg2rad(phase))
            mag = ff.logMagToLin(logmag)
            real,imag = ff.magPhaseToReIm(mag,numpy.rad2deg(phase)) #needs phase in degrees

            response_fft = numpy.add(real,imag*1j)

            #Now with the response I try to extend it to zero (added ~ 3 sample, as lowest freq is 2Mhz and sampling of 0.62 MHz)
            freqs_to_zero = numpy.linspace(0,freqs[-1],len(freqs)+3,endpoint=True) #The original did not include 0 Mhz, which I think messes up ffts. 
            response_fft_to_zero = scipy.interpolate.interp1d(numpy.append(0,freqs),numpy.append(0,response_fft),fill_value=0.0,bounds_error=False,kind='linear')(freqs_to_zero)
            tukey = scipy.signal.tukey(len(response_fft_to_zero), alpha=0.005, sym=True)
            response_fft_to_zero = numpy.multiply(tukey,response_fft_to_zero)

            output_phase = numpy.unwrap(numpy.angle(response_fft_to_zero)) #radians
            output_phase = scipy.interpolate.interp1d(freqs_to_zero,output_phase,fill_value=0.0,bounds_error=False,kind='linear')(goal_freqs)
            phase_response[channel] = output_phase #radians
            
            if plot == True:
                phase = phase - phase[freqs/1e6 >= 50][0] #Align at 50Mhz
                plt.plot(freqs,numpy.rad2deg(phase),label='Ch%i Raw'%channel)
                #plt.plot(freqs_to_zero,output_phase,label='Ch%i Raw'%channel)
                plot_phase = output_phase - output_phase[goal_freqs/1e6 >= 50][0] #Align at 50Mhz
                plt.scatter(goal_freqs,numpy.rad2deg(plot_phase),s=4,label='Ch%i Interp'%channel)
                plt.xlim(0,max(freqs))

        return goal_freqs, phase_response 
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)



if __name__ == '__main__':
    try:
        plt.close('all')
        plot_residual = True

        datapath = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/'
        plot = True
        if plot == True:
            plt.figure()
            plt.subplot(2,1,1)
        components = ['2ndStage','preamp']

        #Phase response plots
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
                    label = '2nd Stage Channel %i'%channel
                elif component == 'preamp':
                    filename = datapath + 'beacon_preamp_%i_s21phase.csv'%(channel)
                    label = 'Board %i'%channel



                freqs, phase = ff.readerFieldFox(filename,header=17) #
                phase = numpy.rad2deg(numpy.unwrap(numpy.deg2rad(phase)))
                phase = phase - phase[freqs/1e6 >= 50][0] #Align at 50Mhz

                if plot_residual == True:
                    if channel == channels[0]:
                        template_phase = phase.copy()
                    phase -= template_phase
                    ylabel = 'Unwrapped Phase Residual (Deg)'
                else :
                    ylabel = 'Unwrapped Phase (Deg)'



                if plot == True:
                    plt.subplot(2,1,comp_index+1)
                    plt.plot(freqs/1e6, phase,label=label)
                    plt.ylabel(ylabel)
                    plt.legend(loc='upper right',fontsize=10)
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        #Group delay plots
        for plot_residual in [True,False]:

            plt.figure()
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
                        label = '2nd Stage Channel %i'%channel
                    elif component == 'preamp':
                        filename = datapath + 'beacon_preamp_%i_s21phase.csv'%(channel)
                        label = 'Board %i'%channel



                    freqs, phase = ff.readerFieldFox(filename,header=17) #
                    phase = numpy.unwrap(numpy.deg2rad(phase)) #radians
                    phase = phase - phase[freqs/1e6 >= 50][0] #Align at 50Mhz


                    #I should calculate the group delays for my interpolated phase response values and see how that looks.
                    #Is my interpoaltion reproducing the same differences I see in the original data.

                    group_delay_freqs = numpy.diff(freqs) + freqs[0:len(freqs)-1]
                    omega = 2.0*numpy.pi*freqs            
                    group_delay = (-numpy.diff(phase)/numpy.diff(omega)) * 1e9

                    if plot_residual == True:
                        ylabel = 'Group Delay Residual (ns)'
                        if channel == channels[0]:
                            template_delay = group_delay.copy()
                        group_delay -= template_delay
                    else :
                        ylabel = 'Group Delay (ns)'



                    if plot == True:
                        plt.subplot(2,1,comp_index+1)
                        plt.plot(group_delay_freqs/1e6, group_delay,label=label)
                        plt.ylabel(ylabel)
                        plt.legend(loc='upper right',fontsize=10)
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        if True:

            datapath = os.environ['BEACON_DATA']
            runs = numpy.array([1507])#numpy.arange(1645,1700)
            eventid = 18001
            for run in runs:
                run = int(run)

                reader = Reader(datapath,run)
                upsample = 2**14
                from tools.correlator import Correlator
                cor = Correlator(reader,  upsample=upsample, n_phi=180, n_theta=180, waveform_index_range=(None,None),crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=True)
                t = cor.t()/1e9
                t = t[-1]*numpy.arange(upsample)/upsample
                goal_freqs = numpy.fft.rfftfreq(len(t),t[1]-t[0])
                loadInterpolatedPhaseResponseMeanPreamp(goal_freqs,  plot=False)
                freqs, phase_response_2nd_stage = loadInterpolatedPhaseResponse2ndStage(goal_freqs, plot=True)
                freqs, phase_response_preamp = loadInterpolatedPhaseResponseMeanPreamp(goal_freqs, plot=True)

                for channel in range(8):
                    wf = cor.wf(eventid,[channel])[0]
                    wf_fft = numpy.fft.rfft(wf)
                    wf_fft = numpy.multiply( wf_fft , numpy.exp(-1j*(phase_response_2nd_stage[channel] + phase_response_preamp)) )
                    wf_dedispered = numpy.fft.irfft(wf_fft)

                    plt.figure()
                    plt.plot(t*1e9,wf,label='original')
                    plt.plot(t*1e9,wf_dedispered,label='dedispered')

                    plt.xlim(t[numpy.argmax(wf_dedispered)]*1e9 - 100, min(max(t*1e9),t[numpy.argmax(wf_dedispered)]*1e9+300))

                    plt.legend()
                    plt.ylabel('adu')
                    plt.xlabel('t (ns)')

                plt.figure()
                for channel in range(8):
                    plt.plot(goal_freqs, numpy.exp(-1j*(phase_response_2nd_stage[channel] + phase_response_preamp)))
                    plt.legend()
                    plt.ylabel('phase response')
                    plt.xlabel('t (ns)')


    except Exception as e:
        print('main()')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
