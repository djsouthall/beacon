'''
This script uses the events found by find_good_saturating_signals.py to attempt to make a
template of the signal. 
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
from objects.fftmath import TemplateCompareTool


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
ignorable_pulser_ids = info.loadIgnorableEventids()
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

if __name__ == '__main__':
    plt.close('all')

    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine

    #Filter settings
    final_corr_length = 2**16 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = 80
    crit_freq_high_pass_MHz = 20
    filter_order = 6 #outdated
    low_pass_filter_order = 6
    high_pass_filter_order = 6
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
                #reader.setEntry(eventids[0])
                tct = TemplateCompareTool(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, initial_template_id=eventids[0])

                if manual_template == True:
                    upsampled_waveforms = tct.loadFilteredWaveformsMultiple(eventids)
                    templates = {}
                    for channel in range(8):
                        template_selected = False
                        template_index = 0
                        while template_selected == False:
                            fig = plt.figure()
                            plt.plot(upsampled_waveforms['ch%i'%channel][template_index])
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
                                    if template_index >= len(upsampled_waveforms['ch%i'%channel]):
                                        print('All possible starting templates cycled through.  Defaulting to first option.')
                                        template_selected = True
                                        template_index = 0
                                else:
                                    print('Response not accepted.  Please type either y or n')
                                    acceptable_response = False
                        templates['ch%i'%channel] = upsampled_waveforms['ch%i'%channel][template_index]
                    templates_for_corr = numpy.zeros((8,tct.final_corr_length))
                    for channel in range(8):
                        templates_for_corr[channel,0:len(templates['ch%i'%channel])] = templates['ch%i'%channel]
                    tct.setTemplateToCustom(templates_for_corr)

                continue
                #WORKING ENDED HERE
                #I have the templates loaded at this point.  Just need to write the overhead for using tct.alignToTemplate

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


