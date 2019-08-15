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

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
plt.ion()

#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.
template_dirs = {
    'run793':'/home/dsouthall/Projects/Beacon/beacon/analysis/templates/run793_0'
}

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
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine
    event_limit = 0

    for run_index, run in enumerate(runs):
        reader = Reader(datapath,run)
        N = reader.N()
        waveform_times, templates = loadTemplates(template_dirs['run%i'%run])

        templates_scaled = {} #precomputing
        len_template = len(waveform_times)
        for channel in range(8):
            templates_scaled['ch%i'%channel] = templates['ch%i'%channel]/numpy.std(templates['ch%i'%channel])

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
            plt.legend(fontsize=16)
            plt.subplot(2,1,2)
            plt.ylabel('dBish',fontsize=16)
            plt.xlabel('Freq (MHz)',fontsize=16)
            plt.legend(fontsize=16)

        max_corrs = {}
        delays = {}
        for channel in range(8):
            max_corrs['ch%i'%channel] = numpy.zeros(N if event_limit == None else event_limit)
            delays['ch%i'%channel] = numpy.zeros(N if event_limit == None else event_limit)

        for event_index, eventid in enumerate(range(N if event_limit == None else event_limit)):
            sys.stdout.write('\r(%i/%i)'%(eventid+1,N if event_limit == None else event_limit))
            sys.stdout.flush()
            reader.setEntry(eventid) 

            for channel in range(8):
                wf = reader.wf(channel)
                corr = scipy.signal.correlate(templates_scaled['ch%i'%channel],wf/numpy.std(wf))/(len_template) #should be roughly normalized between -1,1
                max_corrs['ch%i'%channel][event_index] = numpy.max(corr)
                delays['ch%i'%channel][event_index] = int(numpy.argmax((corr))-numpy.size(corr)/2.)

        if True:
            plt.figure()
            plt.suptitle('Max Correlation Values for Run %i'%(run),fontsize=20)
            for channel in range(8):
                if channel == 0:
                    ax = plt.subplot(4,2,channel+1)
                else:
                    plt.subplot(4,2,channel+1,sharex=ax)
                plt.hist(max_corrs['ch%i'%channel],bins=100,range=(-1.05,1.05),label='ch%i'%channel)
                plt.ylabel('Counts',fontsize=16)
                plt.xlabel('Correlation Value',fontsize=16)
                plt.legend(fontsize=16)


