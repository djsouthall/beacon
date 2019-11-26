'''
This script uses the time delay calculators to look for signals bouncing off of planes.
The characteristic of this signals that we look for is their progression through the sky,
i.e. a repeated signal that moves gradually with time in a sensible flight path.
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
from tools.data_handler import createFile
from objects.fftmath import TimeDelayCalculator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
import h5py
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

cable_delays = info.loadCableDelays()


#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.

known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
ignorable_pulser_ids = info.loadIgnorableEventids()

def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))

if __name__ == '__main__':
    plt.close('all')
    datapath = os.environ['BEACON_DATA']
    run = 1645

    hpol_pairs  = numpy.array(list(itertools.combinations((0,2,4,6), 2)))
    vpol_pairs  = numpy.array(list(itertools.combinations((1,3,5,7), 2)))
    pairs       = numpy.vstack((hpol_pairs,vpol_pairs)) 

    cm = plt.cm.get_cmap('plasma')

    try:
        reader = Reader(datapath,run)
        filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.

        with h5py.File(filename, 'a') as file:
            try:
                rf_cut = file['trigger_type'][...] == 2
                inband_cut = ~numpy.logical_and(numpy.any(file['inband_peak_freq_MHz'][...],axis=1) < 49, file['trigger_type'][...] > 48)
                total_cut = numpy.logical_and(rf_cut,inband_cut)

                eventids = file['eventids'][total_cut]

                td_01 = file['hpol_t_%isubtract%i'%(0,1)][total_cut]
                td_02 = file['hpol_t_%isubtract%i'%(0,2)][total_cut]
                td_03 = file['hpol_t_%isubtract%i'%(0,3)][total_cut]


                '''
                Here I will work with the data.

                First I need to figure out how to get more accurate arrival raw_approx_trigger_time.
                This should be possible from getting the gps trigger time and 
                interpolating that as a function of raw_trig_time.  
                '''
                #ultimately should use an interpolated time, but temporarily I am using raw_approx_trigger_time
                raw_approx_trigger_time = file['raw_approx_trigger_time'][total_cut] 

                fig = plt.figure()
                scatter = plt.scatter(td_01,td_02,c=raw_approx_trigger_time-min(raw_approx_trigger_time),cmap=cm)
                cbar = fig.colorbar(scatter)




            except Exception as e:
                file.close()
                print('Error in plotting.')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

    except Exception as e:
        print('Error in plotting.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
