'''
The purpose of this script is to provide the code used to generate CR signals.  Eventually this should entail actual CR 
models/physics, however as an initial starting place a bipolar delta function convolved with the
system response will be used. 
'''

import sys
import os
import inspect

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.info as info
import tools.field_fox as ff

import numpy
import scipy
import scipy.signal
import matplotlib.pyplot as plt

datapath = os.environ['BEACON_DATA']

plt.ion()


def loadInterpolatedMeanResponse(mode, return_domain='time', upsample_factor=16, plot=False):
    '''
    This function uses code originally use in loadInterpolatedPhaseResponseMeanPreamp, but is seperated such that the
    system response (at least the portions we have measured) can be used seperately. 

    Because we don't know which board goes to which ant, I am just taking the mean.  They are all similar anyways. 
    These responses are intended to be used for making a crude CR template and should be double checked if needed for a
    more precise application.   

    Parameters
    ----------
    mode : str
        This determines which response to load.  Options: 'preamp', 'stage2', 'all'.  When all is selected, this function
        will call itself recursively and output all responses in the order listed in Options. 
    return_domain : str
        Either 'time' or 'freq', determines if the output is freq, response_fft or t, response
    upsample_factor : int
        This will be used to multiply the length of the time domain responses using scipy.signal.resample.  This should 
        be a factor of 2.  This is used to allow for better precision in the time domain aligning of signals before 
        averaging.  The responses for both preamp and stage2 are both relatively well lined up by default so this could
        be low, with a smaller change in the output.  By default this is set to 16. 
    plot : bool
        If True then the preamp response will be plotted.

    Returns
    ----------
    If return_domain == 'freq' then:
        (mode_freq_Hz, mode_response_fft) or (modeA_freq_Hz, modeA_response_fft, modeB_freq_Hz, modeB_response_fft) if 
        mode == 'all'.
    else:
        (mode_t_s, mode_response) or (modeA_t_s, modeA_response, modeB_t_s, modeB_response) if 
        mode == 'all'.
    '''
    try:
        if mode in ['preamp','stage2']:
            # Pseudo Code:

            # Load in responses as is.  
            for channel in numpy.arange(8):
                #Below is loading in the signals and making them into complex numbers
                if mode == 'preamp':
                    phase_filename = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/' + 'beacon_preamp_%i_s21phase.csv'%(channel+1) #These should probably not be hardcoded, and added to info.py somewhere. 
                    mag_filename = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/' + 'beacon_preamp_%i_s21mag.csv'%(channel+1)
                elif mode == 'stage2':
                    phase_filename = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/' + 'beacon_2ndStage_ch%i_s21phase.csv'%channel
                    mag_filename = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/beacon_amp_chain_sparams/' + 'beacon_2ndStage_ch%i_s21mag.csv'%channel
                freqs, phase = ff.readerFieldFox(phase_filename,header=17) #Phase in degrees
                freqs, logmag = ff.readerFieldFox(mag_filename,header=17)

                phase = numpy.unwrap(numpy.deg2rad(phase))
                mag = ff.logMagToLin(logmag)
                real,imag = ff.magPhaseToReIm(mag,numpy.rad2deg(phase)) #needs phase in degrees
                response_fft = numpy.add(real,imag*1.0j)
                #import pdb; pdb.set_trace()

                #Extend measured response down to 0 MHz. 
                #Now with the response I try to extend it to zero (added ~ 3 sample, as lowest freq is 2Mhz and sampling of 0.62 MHz)
                freqs_to_zero = numpy.linspace(0,freqs[-1],len(freqs)+3,endpoint=True) #The original did not include 0 Mhz, which I think messes up ffts. This is a hard coded solution to the data taken.
                response_fft_to_zero = scipy.interpolate.interp1d(numpy.append(0,freqs),numpy.append(0,response_fft),fill_value=0.0,bounds_error=False,kind='linear')(freqs_to_zero)


                if channel == 0:
                    responses_fft = numpy.zeros((8,len(response_fft)),dtype='complex')
                responses_fft[channel] = response_fft

                if channel == 0:
                    responses_fft_to_zero = numpy.zeros((8,len(response_fft_to_zero)),dtype='complex')
                responses_fft_to_zero[channel] = response_fft_to_zero


            if plot == True:
                plt.figure()
                plt.suptitle('%s : Comparing Extended Response Magnitudes'%mode)
                plt.subplot(2,1,1)
                for channel in range(8):
                    plt.plot(freqs/1e6, 10*numpy.log10(numpy.abs(responses_fft[channel])),label='channel %i'%channel)
                    plt.ylabel('dBish')
                    plt.xlabel('Freq (MHz)')
                    #plt.ylim(-35,12)
                    plt.xlim(0,250)

                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.legend()


                plt.subplot(2,1,2)
                for channel in range(8):
                    plt.plot(freqs_to_zero/1e6, 10*numpy.log10(numpy.abs(responses_fft_to_zero[channel])),label='channel %i extended'%channel)
                    plt.ylabel('dBish')
                    plt.xlabel('Freq (MHz)')
                    #plt.ylim(-35,12)
                    plt.xlim(0,250)

                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.legend()

                plt.figure()
                plt.suptitle('%s : Comparing Extended Response Real'%mode)
                plt.subplot(2,1,1)
                for channel in range(8):
                    plt.plot(freqs/1e6, numpy.real(responses_fft[channel]),label='channel %i'%channel)
                    plt.ylabel('Real(Response)')
                    plt.xlabel('Freq (MHz)')
                    ##plt.ylim(-35,12)
                    plt.xlim(0,250)

                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.legend()


                plt.subplot(2,1,2)
                for channel in range(8):
                    plt.plot(freqs_to_zero/1e6, numpy.real(responses_fft_to_zero[channel]),label='channel %i extended'%channel)
                    plt.ylabel('Real(Response)')
                    plt.xlabel('Freq (MHz)')
                    ##plt.ylim(-35,12)
                    plt.xlim(0,250)

                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.legend()

                plt.figure()
                plt.suptitle('%s : Comparing Extended Response Imag'%mode)
                plt.subplot(2,1,1)
                for channel in range(8):
                    plt.plot(freqs/1e6, numpy.imag(responses_fft[channel]),label='channel %i'%channel)
                    plt.ylabel('imag(Response)')
                    plt.xlabel('Freq (MHz)')
                    ##plt.ylim(-35,12)
                    plt.xlim(0,250)

                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.legend()


                plt.subplot(2,1,2)
                for channel in range(8):
                    plt.plot(freqs_to_zero/1e6, numpy.imag(responses_fft_to_zero[channel]),label='channel %i extended'%channel)
                    plt.ylabel('imag(Response)')
                    plt.xlabel('Freq (MHz)')
                    ##plt.ylim(-35,12)
                    plt.xlim(0,250)

                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.legend()




            # Convert to time domain.
            #freqs_to_zero
            #response_fft_to_zero

            #import pdb; pdb.set_trace()

            responses = numpy.fft.fftshift(numpy.fft.irfft(responses_fft_to_zero,axis=1))
            N = numpy.shape(responses)[1] #Number of samples in time
            df = freqs_to_zero[1]-freqs_to_zero[0] #frequency step in Hz.  df = 1/T = 1/(N*dt)
            dt = 1.0/(N*df) #Time step in seconds
            t = numpy.arange(N)*dt

            # Plot time domain responses to ensure they are causal. 
            if plot == True:
                plt.figure()
                plt.suptitle('%s : Time Domain Response'%mode)
                for channel in range(8):
                    plt.plot(t*1e9, responses[channel],label='channel %i'%channel)
                    plt.ylabel('Response')
                    plt.xlabel('ns')

                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.legend()

            # Align time domain responses.

            #upsample for more precise alignment
            upsample_factor = 16
            upsampled_responses, upsampled_time = scipy.signal.resample(responses,upsample_factor*N,t=t,axis=1)

            if plot == True:
                plt.figure()
                plt.suptitle('%s : Time Domain Response'%mode)
                for channel in range(8):
                    plt.plot(upsampled_time*1e9, upsampled_responses[channel]/numpy.max(upsampled_responses[channel]),label='channel %i'%channel)
                    plt.ylabel('Normalized Response')
                    plt.xlabel('ns')

                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.legend()
                ax = plt.gca()

            #align all signals to 0 based on cross correlation
            # if plot == True:
            #     plt.figure()
            #     plt.suptitle('%s : Time Domain Response'%mode)

            for channel in range(numpy.shape(upsampled_responses)[0] - 1):
                #Using channel + 1 because cross correlating and aligning every signal to channel zero.
                c = numpy.correlate(upsampled_responses[0],upsampled_responses[channel + 1],mode='full') 
                roll_amount = numpy.argmax(c) - upsample_factor*N + 1 #This makes it so zero with zero would result in 0 roll as desired.  
                #print('channel = %i, roll_amount = %i'%(channel,roll_amount))

                upsampled_responses[channel + 1] = numpy.roll(upsampled_responses[channel + 1],roll_amount)
                # if plot == True:
                #     plt.plot(c)

            # Average waveforms in time domain.  

            #import pdb; pdb.set_trace()
            averaged_upsampled_response = numpy.mean(upsampled_responses,axis=0)
            
            if plot == True:
                plt.figure()
                plt.suptitle('%s : Aligned Time Domain Response'%mode)
                plt.subplot(1,1,1,sharex=ax,sharey=ax)
                plt.plot(upsampled_time*1e9,averaged_upsampled_response/numpy.max(averaged_upsampled_response),label='Averaged',linewidth=4,c='k',alpha=0.8)
                for channel in range(8):
                    plt.plot(upsampled_time*1e9, upsampled_responses[channel]/numpy.max(upsampled_responses[channel]),label='channel %i'%channel)

                plt.ylabel('Normalized Response')
                plt.xlabel('ns')

                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.legend()

            #Downsample to original measurement sampling
            downsampled_response, downsampled_time = scipy.signal.resample(averaged_upsampled_response,N,t=upsampled_time)
            
            if plot == True:
                plt.scatter(upsampled_time*1e9,averaged_upsampled_response/numpy.max(averaged_upsampled_response),label='Averaged Downsampled',s=20,c='r')

            if return_domain == 'time':
                return downsampled_time, downsampled_response
            else:
                #Return Frequency domain response with original sampling.
                out_response_fft = numpy.fft.rfft(downsampled_response)
                out_freq_Hz = numpy.fft.rfftfreq(len(downsampled_time),downsampled_time[1]-downsampled_time[0])

                return out_freq_Hz, out_response_fft

            # Leave upsampling to another function?
        elif mode == 'all':
            preamp_x, preamp_y = loadInterpolatedMeanResponse('preamp',plot=plot)
            stage2_x, stage2_y = loadInterpolatedMeanResponse('stage2',plot=plot)

            return preamp_x, preamp_y, stage2_x, stage2_y

        else:
            print('Given mode not in list of acceptable modes.  Please give any of: "peamp", "stage2", or "all"')
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


class CosmicRayGenerator():
    """
    Given a particular model in init, this will produce CR's of specific Energies.

    The models are intended to be given in Electric field such that they can be convolved with the system response and 
    scaled to produce adu signals.

    This functionality is anticipated to be filled out in the future.  Early versions    will simply produce a bipolar 
    signal, and convolve it with the system response.  The bipolar signal will be scaled to have visible magnitude above
    normal noise levels. 

    Parameters
    ----------
    model : str
        This is the model you wish to use.  Currently the options include:
        'bi-delta' : A bi polar delta function convolved with the system response.
    """

    def __init__(self, model='bi-delta'):
        try:
            self.accepted_model_list = ['bi-delta']
            if not type(model) == str:
                print('ERROR')
                print('Model parameter incorrectly given.  Must be a string, not a %s'%str(type(model)))
                return
            elif model not in self.accepted_model_list:
                print('ERROR')
                print('Given model [%s] no in list of allowable models:\n'%model)
                print(self.accepted_model_list)
                return
            else:
                self.model = model


            #One-time preparation required for each model can be performed or called below.
            if self.model == 'bi-delta':
                preamp_t_s, preamp_response, stage2_t_s, stage2_response = loadInterpolatedMeanResponse('all', return_domain='time', plot=False)
                self.preamp_response = preamp_response
                self.stage2_response = stage2_response
                self.response_t_s = preamp_t_s #Same for preamp and stage2
                self.response_t_ns = self.response_t_s*1.0e9

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
        

    def eFieldGenerator(self,t_ns,t_offset=0,extent_ns=5,plot=False):
        '''
        For a given set of time data this will produce a signal for the set model.

        Parameters
        ----------
        t_ns : array of float
            This should be the time series of the electric field to be output.  Expected in ns.
        t_offset : float
            This is the time offset within the given time serious the signal should start.  What "start" means will be 
            model dependant likely.  Descriptions of meanings for each model are:
            'bi-delta' : The initial rise time value will occur at the time step closest to this offset (rounded up).
        extent_ns : float
            How long the entire bipolar signal will take to return to no signal (roughly).  Only used in certain models
            (model == 'bi-delta').  Given in ns.
        '''
        try:
            if self.model == 'bi-delta':
                #Generate "Electric field" portion of bipolar signal (signal before response)
                half_extent_index = int(numpy.ceil(extent_ns/(2*(t_ns[1] - t_ns[0])))) #The number of indices corresponding to half of the extent.  I.e. the extent in time (indices) of each pol of the bipolar signal.
                print(half_extent_index)
                efield = numpy.zeros_like(t_ns)
                start_index = numpy.where(t_ns >= t_offset)[0][0]
                efield[start_index:start_index+half_extent_index] = 1.0
                efield[start_index + half_extent_index:start_index+2*half_extent_index] = -1.0

                # self.preamp_response
                # self.stage2_response
                # self.response_t_s

                #Compare time of response v.s. expected time.  Might be best to abstract this from the individual calling
                #of the function and to put it in the setup.  Surely this is a stable request.
                resample_factor = (self.response_t_ns[1] - self.response_t_ns[0])/(t_ns[1] - t_ns[0])
                #If resample_factor > 1 then responses need to be upsampled by this factor to have the same time step.
                #Less concerned about overall length, more concerned about matching time step for convolution.
                stage2_response_resampled, response_t_s_resampled  = scipy.signal.resample(self.stage2_response,int(len(self.response_t_s)*resample_factor),t=self.response_t_s)
                preamp_response_resampled, response_t_s_resampled  = scipy.signal.resample(self.preamp_response,int(len(self.response_t_s)*resample_factor),t=self.response_t_s)

                #Use impulse response to get signal.
                efield_convolved = numpy.convolve(numpy.convolve(efield,stage2_response_resampled,mode='same'),1.0e6*preamp_response_resampled,mode='same')
                if len(response_t_s_resampled) > len(t_ns):
                    #convolve mode 'same' will match max length of input array.  Can cut down the signal after matching.
                    #should ultimately match len(t_ns)
                    efield_convolved_t_ns = response_t_s_resampled*1e9
                else:
                    efield_convolved_t_ns = t_ns



                #Plot efield signal.
                if plot == True:
                    plt.figure()
                    
                    plt.plot(response_t_s_resampled*1e9,stage2_response_resampled/numpy.max(stage2_response_resampled),label='stage2')
                    plt.plot(response_t_s_resampled*1e9,preamp_response_resampled/numpy.max(preamp_response_resampled),label='preamp')
                    plt.plot(t_ns,efield/numpy.max(efield),label='bipolar delta')
            
                    plt.plot(efield_convolved_t_ns,efield_convolved/numpy.max(efield_convolved),c='r',linestyle='--',label='Resultant Convolved "E Field" Signal')#Where the output convolved signal will be plotted.  

                    plt.ylabel('Normalized Responses and Signals')
                    plt.xlabel('ns')
                    plt.legend()
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            else:
                print('Model not yet programmed in function: eFieldGenerator')

            return efield
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)




if __name__ == '__main__':
    try:
        plt.close('all')
        #Get timing info from real BEACON data for testing.
        run = 1509
        known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
        eventid = known_pulser_ids['run%i'%run]['hpol'][0]
        reader = Reader(datapath,run)
        reader.setEntry(eventid)
        test_t = reader.t()
        test_pulser_adu = reader.wf(0)

        #Creating test signal
        cr_gen = CosmicRayGenerator(model='bi-delta')

        test_e = cr_gen.eFieldGenerator(test_t,t_offset=800.0,plot=True)


        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(test_t,test_pulser_adu,label='Pulser Signal')
        plt.ylabel('E (adu)')
        plt.xlabel('t (ns)')

        plt.legend()
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        plt.subplot(2,1,2)
        plt.plot(test_t,test_e,label='Test CR Signal')
        plt.scatter(test_t,test_e,c='r')
        plt.ylabel('E (adu)')
        plt.xlabel('t (ns)')

        plt.legend()
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
