#!/usr/bin/env python3
'''
This is meant to be a simple script that tracks indicators of antenna performance over time.  Hopefully
this will make it clear which runs contain downed antennas, etc.  
'''

import sys
import os
import inspect
import h5py

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_handler import createFile

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.ion()

import numpy
import scipy
import scipy.signal
import scipy.signal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

datapath = os.environ['BEACON_DATA']


if __name__ == '__main__':
    plt.close('all')
    
    runs = numpy.arange(1770,1810)#numpy.append(numpy.arange(1500,1700),numpy.arange(3530,3555))#numpy.arange(1500,1700)
    
    #Prepare for getting data
    mean_std = numpy.zeros((len(runs),8))

    event_type_counts = numpy.zeros((len(runs),8))


    for run_index, run in enumerate(runs):
        try:
            sys.stdout.write('\n(%i/%i)\tRun %i\t'%(run_index+1,len(runs),run))
            sys.stdout.flush()

            reader = Reader(datapath,run)
            filename = createFile(reader)
            if filename is not None:
                with h5py.File(filename, 'r') as file:
                    N = reader.head_tree.Draw("trigger_type","","goff") 
                    trigger_type = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N).astype(int)
                    event_type_counts[run_index,0] = sum(trigger_type == 1) # [1] Software
                    event_type_counts[run_index,1] = sum(trigger_type == 2) # [2] RF
                    event_type_counts[run_index,2] = sum(trigger_type == 3) # [3] GPS
                    

                    std = file['std'][...][trigger_type == 2]
                    mean_std[run_index,:] = numpy.mean(std,axis=0)
                    file.close()
            else:
                print('Run Data absent, filled with 0 value.')
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)    
    #Plot Data

    fig = plt.figure()
    
    #Plot hpol
    ax_h = plt.subplot(3,1,1)

    plt.plot(runs,mean_std[:,0::2],alpha=0.5)
    for channel in [0,2,4,6]:
        plt.scatter(runs,mean_std[:,channel])

    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylabel('Mean Time Domain Std (adu)\nRFI Triggers Only')
    plt.xlabel('Run Number')
    plt.legend(['Channel 0','Channel 2','Channel 4','Channel 6'],loc='upper left')

    #Plot vpol
    ax_v = plt.subplot(3,1,2, sharex = ax_h)

    plt.plot(runs,mean_std[:,1::2],alpha=0.5)
    for channel in [1,3,5,7]:
        plt.scatter(runs,mean_std[:,channel])

    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylabel('Mean Time Domain Std (adu)\nRFI Triggers Only')
    plt.xlabel('Run Number')
    plt.legend(['Channel 1','Channel 3','Channel 5','Channel 7'],loc='upper left')

    #Plot Data

    ax_counts = plt.subplot(3,1,3, sharex = ax_h)

    plt.plot(runs,event_type_counts[:,0],linewidth=3,label='[1] Software')
    plt.plot(runs,event_type_counts[:,1],linewidth=3,label='[2] RF')
    plt.plot(runs,event_type_counts[:,2],linestyle='--',linewidth=3,label='[3] GPS')

    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.ylabel('Counts In Run')
    plt.xlabel('Run Number')
    plt.legend(loc='upper left')
