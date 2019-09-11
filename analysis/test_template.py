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
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.
template_dirs = {
    'run793':{  'dir':'/home/dsouthall/Projects/Beacon/beacon/analysis/templates/run793_4',
                'resample_factor' : 200,
                'crit_freq_low_pass_MHz' : 75,
                'crit_freq_high_pass_MHz' : 15,
                'filter_order' : 6,
    }
}

known_pulser_ids = info.loadPulserEventids()
ignorable_pulser_ids = info.loadPulserIgnorableEventids()

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

def loadTemplates(template_path):
    waveforms = {}
    for channel in range(8):
        with open(template_path + '/ch%i.csv'%channel) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            x = []
            y = []
            for row in csv_reader:
                x.append(float(row[0]))
                y.append(float(row[1]))
                line_count += 1
            waveforms['ch%i'%channel] = numpy.array(y)
    return x,waveforms


if __name__ == '__main__':
    #plt.close('all')
    # If your data is elsewhere, pass it as an argument
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine
    event_limit = 1
    save_fig = True

    for run_index, run in enumerate(runs):
        reader = Reader(datapath,run)
        eventids = cc.getTimes(reader)[3]

        if event_limit is not None:
            if event_limit < len(eventids):
                eventids = eventids[0:event_limit]

        waveform_times, templates = loadTemplates(template_dirs['run%i'%run]['dir'])
        original_wf_len = int(reader.header().buffer_length)
        upsample_wf_len = original_wf_len*template_dirs['run%i'%run]['resample_factor']
        corr_delay_times = numpy.arange(-upsample_wf_len+1,upsample_wf_len)
        #Setup Filter
        freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
        b, a = scipy.signal.butter(template_dirs['run%i'%run]['filter_order'], template_dirs['run%i'%run]['crit_freq_low_pass_MHz']*1e6, 'low', analog=True)
        d, c = scipy.signal.butter(template_dirs['run%i'%run]['filter_order'], template_dirs['run%i'%run]['crit_freq_high_pass_MHz']*1e6, 'high', analog=True)

        filter_x_low_pass, filter_y_low_pass = scipy.signal.freqs(b, a,worN=freqs)
        filter_x_high_pass, filter_y_high_pass = scipy.signal.freqs(d, c,worN=freqs)
        filter_x = freqs
        filter_y = numpy.multiply(filter_y_low_pass,filter_y_high_pass)
        filter_y = numpy.tile(filter_y,(8,1))
        templates_scaled = {} #precomputing
        len_template = len(waveform_times)

        templates_scaled_2d = numpy.zeros((8,len_template))
        for channel in range(8):
            templates_scaled['ch%i'%channel] = templates['ch%i'%channel]/numpy.std(templates['ch%i'%channel])
            templates_scaled_2d[channel] = templates['ch%i'%channel]/numpy.std(templates['ch%i'%channel])

        if True:
            plt.figure()
            plt.suptitle('Averaged Templates for Run %i'%(run),fontsize=20)
            for channel in range(8):
                if channel == 0:
                    ax = plt.subplot(4,2,channel+1)
                else:
                    plt.subplot(4,2,channel+1,sharex=ax)
                plt.plot(waveform_times,templates['ch%i'%channel],label='ch%i'%channel)
                plt.ylabel('Adu',fontsize=16)
                plt.xlabel('Time (ns)',fontsize=16)
                plt.legend(fontsize=16)

        if True:
            plt.figure()
            plt.suptitle('Averaged Templates for Run %i'%(run),fontsize=20)
            plt.subplot(2,1,1)
            for channel in range(8):
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
            plt.xlim(0,250)
            plt.ylim(-50,100)
            plt.legend(fontsize=16)
            plt.subplot(2,1,2)
            plt.ylabel('dBish',fontsize=16)
            plt.xlabel('Freq (MHz)',fontsize=16)
            plt.legend(fontsize=16)
            plt.xlim(0,250)
            plt.ylim(-50,100)
        '''
        max_corrs = {}
        delays = {}
        for channel in range(8):
            max_corrs['ch%i'%channel] = numpy.zeros(len(eventids))
            delays['ch%i'%channel] = numpy.zeros(len(eventids))
        '''
        
        max_corrs = numpy.zeros((len(eventids),8))
        delays = numpy.zeros((len(eventids),8))

        for event_index, eventid in enumerate(eventids):
            sys.stdout.write('\r(%i/%i)'%(event_index+1,len(eventids)))
            sys.stdout.flush()
            reader.setEntry(eventid) 

            wfs = numpy.zeros((8,original_wf_len))
            for channel in range(8):
                wfs[channel] = reader.wf(channel)
            
            wfs = scipy.signal.resample(wfs,upsample_wf_len,axis=1)

            #Apply filter
            wfs = numpy.fft.irfft(numpy.multiply(filter_y,numpy.fft.rfft(wfs,axis=1)),axis=1)

            #Can't find a way to vectorize corr
            scaled_wfs = wfs/numpy.tile(numpy.std(wfs,axis=1),(numpy.shape(wfs)[1],1)).T
            corr = numpy.zeros((8,numpy.shape(wfs)[1]*2 - 1))
            for channel in range(8):
                corr[channel] = scipy.signal.correlate(templates_scaled_2d[channel],scaled_wfs[channel])/(len_template) #should be roughly normalized between -1,1
            
            max_corrs[event_index] = numpy.max(corr,axis=1)
            delays[event_index] = corr_delay_times[numpy.argmax(corr,axis=1)]

        if 'run%i'%run in list(known_pulser_ids.keys()):
            print('Testing against known pulser events')
            pulser_max_corrs = numpy.zeros((len(known_pulser_ids['run%i'%run]),8))
            pulser_delays = numpy.zeros((len(known_pulser_ids['run%i'%run]),8))

            for event_index, eventid in enumerate(known_pulser_ids['run%i'%run]):
                sys.stdout.write('\r(%i/%i)'%(event_index+1,len(known_pulser_ids['run%i'%run])))
                sys.stdout.flush()
                reader.setEntry(eventid) 
                wfs = numpy.zeros((8,original_wf_len))
                for channel in range(8):
                    wfs[channel] = reader.wf(channel)

                wfs = scipy.signal.resample(wfs,upsample_wf_len,axis=1)

                #Apply filter
                wfs = numpy.fft.irfft(numpy.multiply(filter_y,numpy.fft.rfft(wfs,axis=1)),axis=1)

                #Can't find a way to vectorize corr
                scaled_wfs = wfs/numpy.tile(numpy.std(wfs,axis=1),(numpy.shape(wfs)[1],1)).T
                corr = numpy.zeros((8,numpy.shape(wfs)[1]*2 - 1))
                for channel in range(8):
                    corr[channel] = scipy.signal.correlate(templates_scaled_2d[channel],scaled_wfs[channel])/(len_template) #should be roughly normalized between -1,1
                pulser_max_corrs[event_index] = numpy.max(corr,axis=1)
                pulser_delays[event_index] = corr_delay_times[numpy.argmax(corr,axis=1)]





        if True:
            fig = plt.figure(figsize=(16,12))
            plt.suptitle('Max Correlation Values for Run %i'%(run),fontsize=20)

            times, subtimes, trigtimes, all_eventids = cc.getTimes(reader)


            for channel in range(8):
                if channel == 0:
                    ax = plt.subplot(4,2,channel+1)
                else:
                    plt.subplot(4,2,channel+1,sharex=ax)
                max_bin = numpy.max(numpy.histogram(max_corrs[:,channel],bins=100,range=(-1.05,1.05))[0])
                plt.hist(max_corrs[:,channel],weights=numpy.ones_like(max_corrs[:,channel])/max_bin,bins=100,range=(-1.05,1.05),label='ch%i all events'%channel,alpha=0.5)
                if 'run%i'%run in list(known_pulser_ids.keys()):
                    max_bin = numpy.max(numpy.histogram(pulser_max_corrs[:,channel],bins=100,range=(-1.05,1.05))[0])
                    plt.hist(pulser_max_corrs[:,channel],weights=numpy.ones_like(pulser_max_corrs[:,channel])/max_bin,bins=100,range=(-1.05,1.05),label='ch%i pulser events'%channel,alpha=0.5)
                plt.ylabel('Counts',fontsize=16)
                plt.xlabel('Correlation Value',fontsize=16)
                plt.legend(fontsize=16)


                plt.figure()
                plt.plot(times[numpy.isin(all_eventids,known_pulser_ids['run%i'%run])],pulser_max_corrs[:,channel])

                plt.figure()
                plt.subplot(2,1,1)
                for index in numpy.where(max_corrs[:,channel] < 0.7):
                    plt.plot(waveform_times,waveforms['ch%i'%channel][index],alpha=0.5)        
                plt.subplot(2,1,2)
                for index in numpy.where(max_corrs[:,channel] > 0.8):
                    plt.plot(waveform_times,waveforms['ch%i'%channel][index],alpha=0.5)        


            if save_fig:
                fig_saved = False
                attempt = 0
                while fig_saved == False:
                    filename = 'template_search_%i.png'%attempt
                    if os.path.exists(filename):
                        print('%s exists, altering name.'%filename)
                        attempt += 1
                    else:
                        try:
                            fig.savefig(filename)
                            print('%s saved'%filename)
                            fig_saved = True

                        except Exception as e:
                            print('Error while saving figure.')
                            print(e)
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)

                        

