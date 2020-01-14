#!/usr/bin/env python3
'''
This script is meant determine the SNR of each RF signal (mean across all channels, hpol and vpol sperately)
then list the eventids of the RF triggers with lowest SNR.  The purpose is to demonstrate low SNR triggering
capability. 
'''
import os
import sys
import h5py
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
from objects.fftmath import TemplateCompareTool, TimeDelayCalculator
from tools.data_handler import createFile
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
import inspect
from ast import literal_eval
import astropy.units as apu
from astropy.coordinates import SkyCoord
import itertools
plt.ion()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 16})
from scipy.optimize import curve_fit
def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))


def plotEvent(reader,eventid,snr=None):
    reader.setEntry(eventid)
    plt.figure()
    t = reader.t()
    if snr is not None:
        plt.suptitle('Run %i, Eventid: %i, Mean Hpol vSNR = %0.3f'%(reader.run,eventid,snr))
    else:
        plt.suptitle('Run %i, Eventid: %i'%(reader.run,eventid))
    for i in range(8):
        plt.subplot(2,4,i + 1)
        plt.plot(t,reader.wf(i),label=str(i))
        plt.legend()
        plt.ylim(-10,10)
        plt.ylabel('adu')
        plt.xlabel('ns')


if __name__=="__main__":

    run = 1790
    datapath = os.environ['BEACON_DATA']

    crit_freq_low_pass_MHz = None#95 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = None#10

    crit_freq_high_pass_MHz = None#50#None
    high_pass_filter_order = None#4#None

    apply_phase_response = False
    hilbert=False

    try:
        run = int(run)
        reader = Reader(datapath,run)

        N = reader.head_tree.Draw("Entry$","trigger_type==%i"%2,"goff") 
        eventids = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N).astype(int)


        filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.

        if filename is not None:
            with h5py.File(filename, 'r') as file:
                rf_cut = file['trigger_type'][...] == 2
                peak = file['inband_peak_freq_MHz'][...]
                eventids = eventids[numpy.isin(eventids,numpy.where(~numpy.any(numpy.logical_and(peak > 48, peak < 49),axis=1))[0])]

                p2p = file['p2p'][...][eventids]

                hpol_snrs = numpy.zeros(len(eventids))
                vpol_snrs = numpy.zeros(len(eventids))
                total_snrs = numpy.zeros(len(eventids))

                snrs = numpy.zeros(8)

                good_antennas = numpy.array([0,2,3,4,5,6,7])

                for event_index, eventid in enumerate(eventids):
                    sys.stdout.write('\r%i/%i'%(event_index+1,len(eventids)))
                    sys.stdout.flush()
                    reader.setEntry(eventid)

                    for i in range(8):
                        wf = reader.wf(i)
                        snrs[i] = p2p[event_index,i]/numpy.std(wf)

                    hpol_snrs[event_index] = numpy.max(snrs[0::2])
                    vpol_snrs[event_index] = numpy.max(snrs[numpy.array([3,5,7])])
                    total_snrs[event_index] = numpy.max(snrs[good_antennas])

                plt.figure()
                plt.hist(hpol_snrs,bins=100)
                plt.ylabel('Bins')
                plt.xlabel('Max SNR Across Hpol Channels per Signal')

                res = scipy.stats.cumfreq(hpol_snrs,numbins=100)
                x = res.lowerlimit + numpy.linspace(0, res.binsize*res.cumcount.size,res.cumcount.size)
                fig = plt.figure(figsize=(10, 4))
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                ax1.hist(hpol_snrs, bins=100)
                ax1.set_title('Histogram')
                ax2.bar(x, res.cumcount, width=res.binsize)
                ax2.set_title('Cumulative histogram')
                ax2.set_xlim([x.min(), x.max()])

                best_eventids_indices = numpy.argsort(hpol_snrs)

                for event_index in best_eventids_indices[0:5]:
                    plotEvent(reader,eventids[event_index],hpol_snrs[event_index])

                interesting_events_indices = numpy.where(numpy.logical_and(hpol_snrs < 6 , hpol_snrs > 5.5))[0]


                for event_index in interesting_events_indices[0:5]:
                    plotEvent(reader,eventids[event_index],hpol_snrs[event_index])



    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)

    sys.exit(0)