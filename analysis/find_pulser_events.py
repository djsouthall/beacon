'''
This script is a quick working script that I am using to find and label the different subset of pulser events corresponding
to different settings on the pulsing day.  This is not meant to be a general file.
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info
from objects.fftmath import TimeDelayCalculator

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()


if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([1509])

    event_min = 0

    for run_index, run in enumerate(runs):
        print('Run %i'%run)
        reader = Reader(datapath,run)
        times, subtimes, trigtimes, eventids = cc.getTimes(reader,trigger_type=3)
        rtimes = times - times[0]



        if False:
            #USE THIS TO SAVE ONCE YOU HAVE SELECTED THE CORRECT EVENT RANGE.  
            #cut = numpy.logical_and(eventids > 4157, eventids < 5794)
            cut = numpy.logical_and(eventids > 4208, eventids < 6033)
            extra_text = 'site_2_bicone_vpol_17dB'
            numpy.savetxt('./run%i_pulser_eventids_%s.csv'%(run,extra_text), numpy.sort(eventids[cut]), delimiter=",")

        meas = {}
        for channel in range(8):
            meas[channel] = []

        for event_index, eventid in enumerate(eventids):
            if eventid < event_min:
                continue
            sys.stdout.write('(%i/%i)\r'%(event_index,len(eventids)))
            sys.stdout.flush()
            reader.setEntry(eventid)
            event_times = reader.t()
            for channel in range(8):
                channel=int(channel)
                meas[channel].append(numpy.max(reader.wf(channel)) - numpy.min(reader.wf(channel)))

        if run == 1507:
            lines_of_interest = numpy.array([1035,1275,1755,10034,11714,12314,12914,15711,17354])
        elif run == 1509:
            lines_of_interest = numpy.array([721,1201,1801,2401,3722,4082,4201])
        elif run == 1511:
            lines_of_interest = numpy.array([892,2690,3892,5812])

        plt.figure()
        plt.title(run)


        for channel in range(8):
            label = str(channel // 2) + ['H','V'][channel % 2]
            linestyle = ['-','-.'][channel % 2]
            plt.plot(eventids[eventids >= event_min],meas[channel],linestyle=linestyle,label=label)

        for line in lines_of_interest:
            plt.axvline(line,label=line)
        plt.legend()
        plt.xlabel('Time from start of run in seconds')
        plt.ylabel('Measured Metric of signal (adu)')

