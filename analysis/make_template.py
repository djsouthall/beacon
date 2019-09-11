'''
This script uses the events found by find_good_saturating_signals.py to determine some good
expected time delays between antennas.  These can be used then as educated guesses for time
differences in the antenna_timings.py script. 
'''

import numpy
import scipy.spatial
import scipy.signal
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


#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.
known_pulser_ids = info.loadPulserEventids()
ignorable_pulser_ids = info.loadPulserIgnorableEventids()
clock_rates = info.loadClockRates()

def rfftWrapper(waveform_times, *args, **kwargs):
    spec = numpy.fft.rfft(*args, **kwargs)
    real_power_multiplier = 2.0*numpy.ones_like(spec) #The factor of 2 because rfft lost half of the power except for dc and Nyquist bins (handled below).
    if len(numpy.shape(spec)) != 1:
        real_power_multiplier[:,[0,-1]] = 1.0
    else:
        real_power_multiplier[[0,-1]] = 1.0
    spec_dbish = 10.0*numpy.log10( real_power_multiplier*spec * numpy.conj(spec) / len(waveform_times)) #10 because doing power in log.  Dividing by N to match monutau. 
    freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
    return freqs, spec_dbish

def makeFilter(waveform_times,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order, plot_filter=False):
    dt = waveform_times[1] - waveform_times[0]
    freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
    b, a = scipy.signal.butter(filter_order, crit_freq_low_pass_MHz*1e6, 'low', analog=True)
    d, c = scipy.signal.butter(filter_order, crit_freq_high_pass_MHz*1e6, 'high', analog=True)

    filter_x_low_pass, filter_y_low_pass = scipy.signal.freqs(b, a,worN=freqs)
    filter_x_high_pass, filter_y_high_pass = scipy.signal.freqs(d, c,worN=freqs)
    filter_x = freqs
    filter_y = numpy.multiply(filter_y_low_pass,filter_y_high_pass)
    if plot_filter == True:
        plt.figure()
        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y)),color='k',label='final filter')
        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_low_pass)),color='r',linestyle='--',label='low pass')
        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_high_pass)),color='orange',linestyle='--',label='high pass')
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(crit_freq_low_pass_MHz, color='magenta',label='LP Crit') # cutoff frequency
        plt.axvline(crit_freq_high_pass_MHz, color='cyan',label='HP Crit') # cutoff frequency
        plt.xlim(0,200)
        plt.ylim(-50,10)
        plt.legend()
    return filter_y, freqs

def alignToTemplate(eventids,_upsampled_waveforms,_waveforms_corr, _template, _final_corr_length, _waveform_times_corr, _filter_y_corr=None, align_method=1, plot_wf=False, template_pre_filtered=False):
    '''
    _waveforms_corr should be a 2d array where each row is a waveform of a different event, already zero padded for cross correlation
    (i.e. must be over half zeros on the right side).

    _upsampled_waveforms are the waveforms that have been upsampled (BUT WERE NOT ORIGINALLY PADDED BY A FACTOR OF 2).  These are
    the waveforms that will be aligned based on the time delays from correlations performed with _waveforms_corr.
        
    If a filter is given then it will be applied to all signals.  It is assumed the upsampled signals are already filtered.

    This given _template must be in the same form upsampled nature as _waveforms_corr. 
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
    plt.close('all')

    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine

    #Filter settings
    final_corr_length = 2**16 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = 80
    crit_freq_high_pass_MHz = 20
    filter_order = 6
    use_filter = False
    align_method = 2
    #1. Looks for best alignment within window after cfd trigger, cfd applied on hilbert envelope.
    #2. Of the peaks in the correlation plot within range of the max peak, choose the peak with the minimum index.
    #3. Align to maximum correlation value.


    #Plotting info
    plot = True

    #Initial Template Selection Parameters
    manual_template = False #If true, templates waveforms will be printed out until the user chooses one as a started template for that channel.
    initial_std_precentage_window = 0.2 #The inital percentage of the window that the std will be calculated for.  Used for selecting interesting intial templates (ones that that start in the middle of the waveform and have low std)
    intial_std_threshold = 10.0 #The std in the first % defined above must be below this value for the event to be considered as a template.  
    
    #Template loop parameters
    iterate_limit = 1

    #Output
    save_templates = False

    #General Prep
    channels = numpy.arange(8,dtype=int)
    
    #Main loop
    for run_index, run in enumerate(runs):
        if 'run%i'%run in list(known_pulser_ids.keys()):
            try:
                if 'run%i'%run in list(ignorable_pulser_ids.keys()):
                    eventids = numpy.sort(known_pulser_ids['run%i'%run][~numpy.isin(known_pulser_ids['run%i'%run],ignorable_pulser_ids['run%i'%run])])
                else:
                    eventids = numpy.sort(known_pulser_ids['run%i'%run])

                reader = Reader(datapath,run)
                reader.setEntry(eventids[0])
                
                waveform_times = reader.t()
                dt = waveform_times[1]-waveform_times[0]
                waveform_times_padded_to_power2 = numpy.arange(2**(numpy.ceil(numpy.log2(len(waveform_times)))))*dt #Rounding up to a factor of 2 of the len of the waveforms  USED FOR WAVEFORMS
                waveform_times_corr = numpy.arange(2*len(waveform_times_padded_to_power2))*dt #multiplying by 2 for cross correlation later. USED FOR CORRELATIONS
                
                if use_filter:
                    filter_y_corr,freqs_corr = makeFilter(waveform_times_corr,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order,plot_filter=True)
                    filter_y_wf,freqs_wf = makeFilter(waveform_times_padded_to_power2,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order,plot_filter=False)
                else:
                    freqs_corr = numpy.fft.rfftfreq(len(waveform_times_corr), d=(waveform_times_corr[1] - waveform_times_corr[0])/1.0e9)
                    freqs_wf = numpy.fft.rfftfreq(len(waveform_times_padded_to_power2), d=(waveform_times_padded_to_power2[1] - waveform_times_padded_to_power2[0])/1.0e9)

                df_corr = freqs_corr[1] - freqs_corr[0] #Note that this is the df for the padded correlation ffts and would not be the same as the one for the normal waveform ffts which have not been doubled in length. 
                final_dt_corr = 1e9/(2*(final_corr_length//2 + 1)*df_corr) #ns #This is the time step resulting from the cross correlation.  

                time_shifts_corr = numpy.arange(-(final_corr_length-1)//2,(final_corr_length-1)//2 + 1)*final_dt_corr #This results in the maxiumum of an autocorrelation being located at a time shift of 0.0

                #Load in waveforms:
                print('Loading Waveforms for Template:\n')
                exclude_eventids = []
                waveforms_corr = {}
                upsampled_waveforms = {}
                for channel in channels:
                    channel=int(channel)
                    waveforms_corr['ch%i'%channel] = numpy.zeros((len(eventids),len(waveform_times_corr)))
                    upsampled_waveforms['ch%i'%channel] = numpy.zeros((len(eventids),final_corr_length//2))

                for event_index, eventid in enumerate(eventids):
                    sys.stdout.write('(%i/%i)\r'%(event_index,len(eventids)))
                    sys.stdout.flush()
                    reader.setEntry(eventid)
                    event_times = reader.t()
                    for channel in channels:
                        channel=int(channel)
                        waveforms_corr['ch%i'%channel][event_index][0:reader.header().buffer_length] = reader.wf(channel)
                        #Below are the actual time domain waveforms_corr and should not have the factor of 2 padding.  The small rounding padding sticks around, so using waveform_times_padded_to_power2 times,
                        if use_filter:
                            upsampled_waveforms['ch%i'%channel][event_index] = numpy.fft.irfft(numpy.multiply(filter_y_wf,numpy.fft.rfft(waveforms_corr['ch%i'%channel][event_index][0:len(waveform_times_padded_to_power2)])),n=final_corr_length//2) * ((final_corr_length//2)/len(waveform_times_padded_to_power2))
                        else:
                            upsampled_waveforms['ch%i'%channel][event_index] = numpy.fft.irfft(numpy.fft.rfft(waveforms_corr['ch%i'%channel][event_index][0:len(waveform_times_padded_to_power2)]),n=final_corr_length//2) * ((final_corr_length//2)/len(waveform_times_padded_to_power2))
                        #upsampled_waveforms['ch%i'%channel][event_index], upsampled_times = scipy.signal.resample(waveforms_corr['ch%i'%channel][event_index][0:len(waveform_times_padded_to_power2)],final_corr_length//2,t=waveform_times_padded_to_power2)
                    
                '''
                print('Upsampling waveforms_corr')
                for channel in channels:
                    print(channel)
                    channel=int(channel)
                    upsampled_waveforms['ch%i'%channel], upsampled_times = scipy.signal.resample(waveforms_corr['ch%i'%channel],2*(final_corr_length//2 + 1),t=waveform_times_corr,axis = 1)
                '''
                print('\n')

                print('Making Templates')
                
                max_corrs = {}
                index_delays = {}
                templates = {}
                for channel in channels:
                    channel=int(channel)
                    sys.stdout.write('(%i/%i)\n'%(channel,len(channels)))
                    sys.stdout.flush()

                    #Find template to correlate to:
                    ratios = []
                    early_std = []
                    total_std = []
                    for index, waveform in enumerate(waveforms_corr['ch%i'%channel]): 
                        ratios.append(numpy.max(numpy.abs(scipy.signal.hilbert(waveform[0:len(waveform_times)])))/numpy.std(waveform[0:len(waveform_times)]))
                        early_std.append(numpy.std(waveform[0:len(waveform_times)][0:int(initial_std_precentage_window*len(waveform[0:len(waveform_times)]))]))
                        total_std.append(numpy.std(waveform[0:len(waveform_times)]))
                    early_std = numpy.array(early_std)
                    total_std = numpy.array(total_std)

                    exclude_eventids.append(eventids[numpy.logical_or(total_std < 20.0, total_std > 30.0)])

                    sorted_template_indices = numpy.argsort(ratios)[::-1][(early_std < intial_std_threshold)[numpy.argsort(ratios)[::-1]]] #Indices sorted from large to small in ratio, then cut on those templates with small enough initial std

                    if manual_template == True:
                        template_selected = False
                        template_index = 0
                        while template_selected == False:
                            fig = plt.figure()
                            plt.plot(waveform_times,waveforms_corr['ch%i'%channel][sorted_template_indices[template_index]][0:len(waveform_times)])
                            plt.show()
                            
                            acceptable_response = False
                            while acceptable_response == False:
                                response = input('Is this a good waveform to start?  (y/n)')
                                if response == 'y':
                                    acceptable_response = True
                                    template_selected = True
                                    plt.close(fig)
                                elif response == 'n':
                                    acceptable_response = True
                                    template_selected = False
                                    template_index += 1
                                    plt.close(fig)
                                    if template_index >= len(waveforms_corr['ch%i'%channel]):
                                        print('All possible starting templates cycled through.  Defaulting to first option.')
                                        template_selected = True
                                        template_index = 0
                                else:
                                    print('Response not accepted.  Please type either y or n')
                                    acceptable_response = False
                    else:
                        template_selected = True
                        template_index = 0

                    template_event_index = sorted_template_indices[template_index]
                    template_eventid = eventids[template_event_index]
                    template_waveform = waveforms_corr['ch%i'%channel][template_event_index]

                    #At this point there is an initial template, which is zero padded 2 some factor of two but is not significantly upsampled.  
                    #For less computation perhaps the template should be the conjugated one in the correlation?  Maybe not worth the confusion of shifting
                    #waveforms_corr before averaging.

                    if use_filter:
                        index_delays['ch%i'%channel], max_corrs['ch%i'%channel], downsampled_template_out = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],waveforms_corr['ch%i'%channel][template_event_index],final_corr_length,waveform_times_corr,_filter_y_corr = filter_y_corr,plot_wf = False)
                    else:
                        index_delays['ch%i'%channel], max_corrs['ch%i'%channel], downsampled_template_out = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],waveforms_corr['ch%i'%channel][template_event_index],final_corr_length,waveform_times_corr,_filter_y_corr = None,plot_wf = False)

                    #The above used the first pass at a template to align.  Now I use the produced templates iteratively, aligning signals to them to make a new template, repeat.

                    print('Using initial template for alignment of new template:')
                    template_count = 0
                    while template_count <= iterate_limit:
                        #I go one over iterate limit.  The last time it is just getting the correlation times of the events with the final template, and not using the output as a new template.
                        if template_count < iterate_limit:
                            sys.stdout.write('\t\t(%i/%i)\r'%(template_count+1,iterate_limit))
                            sys.stdout.flush()
                            if use_filter:
                                downsampled_template_out = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],downsampled_template_out,final_corr_length,waveform_times_corr,template_pre_filtered=True, _filter_y_corr = filter_y_corr, align_method=align_method,plot_wf = False)[2]
                            else:
                                downsampled_template_out = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],downsampled_template_out,final_corr_length,waveform_times_corr,template_pre_filtered=True, _filter_y_corr = None, align_method=align_method,plot_wf = False)[2]
                        else:
                            if use_filter:
                                index_delays['ch%i'%channel], max_corrs['ch%i'%channel], temp = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],downsampled_template_out,final_corr_length,waveform_times_corr,template_pre_filtered=True, _filter_y_corr = filter_y_corr,align_method=align_method,plot_wf = True)
                            else:
                                index_delays['ch%i'%channel], max_corrs['ch%i'%channel], temp = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],downsampled_template_out,final_corr_length,waveform_times_corr,template_pre_filtered=True, _filter_y_corr = None,align_method=align_method,plot_wf = True)
                        template_count += 1

                    templates['ch%i'%channel] = downsampled_template_out[0:len(waveform_times)] #cutting off the additional zero padding

                #Plotting

                if False:
                    plt.figure()
                    plt.suptitle('Averaged Templates for Run %i'%(run),fontsize=20)
                    for channel in channels:
                        channel=int(channel)
                        if channel == 0:
                            ax = plt.subplot(4,2,channel+1)
                        else:
                            plt.subplot(4,2,channel+1,sharex=ax)
                        plt.plot(waveform_times,templates['ch%i'%channel],label='ch%i'%channel)
                        plt.ylabel('Adu',fontsize=16)
                        plt.xlabel('Time (ns)',fontsize=16)
                        plt.legend(fontsize=16)

                if False:
                    plt.figure()
                    plt.suptitle('Averaged Templates for Run %i'%(run),fontsize=20)
                    plt.subplot(2,1,1)
                    for channel in channels:
                        channel=int(channel)
                        if channel == 0:
                            ax = plt.subplot(2,1,1)
                        elif channel %2 == 0:
                            plt.subplot(2,1,1)
                        elif channel %2 == 1:
                            plt.subplot(2,1,2)

                        template_freqs, template_fft_dbish = rfftWrapper(waveform_times,templates['ch%i'%channel])
                        plt.plot(template_freqs/1e6,template_fft_dbish,label='ch%i'%channel)
                    
                    plt.subplot(2,1,1)
                    plt.ylabel('dBish',fontsize=16)
                    plt.xlabel('Freq (MHz)',fontsize=16)
                    plt.legend(fontsize=16)
                    plt.xlim(0,250)
                    plt.ylim(-50,100)
                    plt.subplot(2,1,2)
                    plt.ylabel('dBish',fontsize=16)
                    plt.xlabel('Freq (MHz)',fontsize=16)
                    plt.legend(fontsize=16)
                    plt.xlim(0,250)
                    plt.ylim(-50,100)

                if True:
                    plt.figure()
                    plt.suptitle('Max Correlation Values for Run %i'%(run),fontsize=20)
                    for channel in channels:
                        channel=int(channel)
                        if channel == 0:
                            ax = plt.subplot(4,2,channel+1)
                        else:
                            plt.subplot(4,2,channel+1,sharex=ax)
                        plt.hist(max_corrs['ch%i'%channel],bins=100,range=(-1.05,1.05),label='ch%i'%channel)
                        plt.ylabel('Counts',fontsize=16)
                        plt.xlabel('Correlation Value',fontsize=16)
                        plt.legend(fontsize=16)

                if False:            
                    times, subtimes, trigtimes, all_eventids = cc.getTimes(reader)

                    eventids_in_template = eventids
                    indices_of_eventids_in_template = numpy.where(numpy.isin(all_eventids, eventids_in_template))[0]

                    if 'run%i'%run in list(clock_rates.keys()):
                        adjusted_trigtimes = trigtimes%clock_rates['run%i'%run]
                    else:
                        adjusted_trigtimes = trigtimes%clock_rates['default']
                        print('No adjusted clock rate present, using nominal value of %f'%clock_rates['default'])
                    fig = plt.figure()
                    plt.scatter(times,adjusted_trigtimes,c='b',marker=',',s=(72./fig.dpi)**2)
                    plt.scatter(times[indices_of_eventids_in_template],adjusted_trigtimes[indices_of_eventids_in_template],c='r',marker=',',s=(72./fig.dpi)**2,label='In Template')
                    plt.ylabel('Trig times')
                    plt.xlabel('Times')
                    plt.legend()

                    for channel in channels:
                        channel=int(channel)
                        fig = plt.figure()
                        plt.title('Channel %i'%channel)
                        plt.scatter(times[indices_of_eventids_in_template],adjusted_trigtimes[indices_of_eventids_in_template],c=max_corrs['ch%i'%channel],marker=',',s=(72./fig.dpi)**2)
                        plt.ylabel('Trig times')
                        plt.xlabel('Times')
                        cbar = plt.colorbar()
                        cbar.set_label('Max Correlation Value')

                template_dir = '/home/dsouthall/Projects/Beacon/beacon/analysis/templates'
                if save_templates:
                    dir_made = False
                    attempt = 0
                    while dir_made == False:
                        try:
                            os.mkdir(template_dir + '/run793_%i'%attempt)
                            template_dir = template_dir + '/run793_%i'%attempt
                            dir_made = True
                            print('Templates being saved in ' + template_dir)
                        except Exception as e:
                            print('Dir exists, altering')
                            attempt += 1

                        for channel in channels:
                            channel=int(channel)
                            y = templates['ch%i'%channel]
                            x = waveform_times
                            with open(template_dir + '/ch%i.csv'%channel, mode='w') as file:
                                writer = csv.writer(file, delimiter=',')
                                for i in range(len(x)):
                                    writer.writerow([x[i],y[i]])

            except Exception as e:
                print('Error in main loop.')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


