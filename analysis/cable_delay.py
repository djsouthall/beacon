'''
This file is intended to calculate cable lengths from s21 measurements. 
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


if __name__ == '__main__':
    try:
        plt.close('all')
        datapath = os.environ['BEACON_ANALYSIS_DIR'] + 'data/calibration/'
        infiles = glob.glob(datapath + '/300FT*.csv')

        files_groups = numpy.unique([f.replace('LM.csv','.csv').replace('P.csv','.csv') for f in infiles])

        fig = plt.figure()
        #fig.canvas.set_window_title(root.split('/')[-1].replace('_.csv','').replace('_',' '))
        cable_delay_range = [30e6,None] #The range of frequencies to average over (you should ignore low frequencies)

        cable_delays = {}
        cable_delays['hpol'] = numpy.zeros(4)
        cable_delays['vpol'] = numpy.zeros(4)

        plot = True

        for root in files_groups:
            pol = root.split('_')[-2][1].lower() + 'pol'
            ant = int(root.split('_')[-2][0])

            freqs, lm = ff.readerFieldFox(root.replace('.csv','LM.csv'),header=18) 
            freqs, phase = ff.readerFieldFox(root.replace('.csv','P.csv'),header=18) 
            phase = numpy.unwrap(numpy.pi*phase/180.0)

            group_delay_freqs = numpy.diff(freqs) + freqs[0:len(freqs)-1]
            omega = 2.0*numpy.pi*freqs            
            group_delay = (-numpy.diff(phase)/numpy.diff(omega)) * 1e9

            if cable_delay_range[0] is None:
                cable_delay_range[0] = min(freqs)
            if cable_delay_range[1] is None:
                cable_delay_range[1] = max(freqs)

            cut = numpy.logical_and(group_delay_freqs >= cable_delay_range[0], group_delay_freqs <= cable_delay_range[1])
            cable_delays[pol][ant] = numpy.mean(group_delay[cut])


            if plot == True:

                #plt.suptitle(root.split('/')[-1].replace('_.csv','').replace('_',' '))
                plt.subplot(4,1,1)
                plt.plot(freqs/1e6, lm,label=root.split('/')[-1].replace('_.csv','').replace('_',' '))
                plt.ylabel('dB')
                plt.legend()
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.subplot(4,1,2)
                plt.plot(freqs/1e6, numpy.unwrap(phase),label=root.split('/')[-1].replace('_.csv','').replace('_',' '))
                plt.ylabel('Phase')
                plt.legend()
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                plt.subplot(4,1,3)
                plt.plot(group_delay_freqs/1e6, group_delay,label=root.split('/')[-1].replace('_.csv','').replace('_',' '))
                plt.ylabel('Group Delay (ns)')
                plt.legend()
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                plt.subplot(4,1,4)
                plt.plot(group_delay_freqs/1e6, group_delay*0.84*constants.speed_light,label=root.split('/')[-1].replace('_.csv','').replace('_',' '))
                plt.ylabel('Cable Length (m)')
                plt.legend()
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        pprint(cable_delays)


    except Exception as e:
        print('main()')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
