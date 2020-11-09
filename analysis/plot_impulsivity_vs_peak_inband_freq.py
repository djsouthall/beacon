'''
This is meant to plot the impulsivity as a function of peak inband frequency.
If it seems sensible this may performa  2dhist with these as the axis.
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
from tools.fftmath import TimeDelayCalculator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
import h5py
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

cm = plt.cm.get_cmap('plasma')

if __name__ == '__main__':
    plt.close('all')
    datapath = os.environ['BEACON_DATA']
    #run = 1652

    runs = numpy.arange(1600,1700) #No RF triggers before 1642, but 1642,43,44,45 don't have pointing?

    colormap_mode = 0
    similarity_count_cut_limit = 1000

    #filter_string = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_32768-align_0'
    align_method = 8#[0,4,8]

    filter_string = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_32768-align_%i'%align_method

    for run_index, run in enumerate(runs):
        try:
            reader = Reader(datapath,run)
            filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.  
            if filename is not None:
                with h5py.File(filename, 'r') as file:
                    try:
                        #print(list(file['map_direction'].keys()) )
                        dsets = list(file.keys()) #Existing datasets
                        n_events_total = len(file['eventids'][...])
                        pol = 'hpol'

                        rf_cut = file['trigger_type'][...] == 2 #This is RF triggers.
                        #inband_cut = ~numpy.logical_and(numpy.any(file['inband_peak_freq_MHz'][...],axis=1) < 49, file['trigger_type'][...] > 48) #Cutting out known CW
                        total_loading_cut = rf_cut#numpy.logical_and(numpy.logical_and(rf_cut,inband_cut),rough_dir_cut)

                        eventids = file['eventids'][total_loading_cut]
                        calibrated_trigtime = file['calibrated_trigtime'][total_loading_cut]
                        
                        if run_index == 0:
                            print(list(file['impulsivity'].keys()))
                            impulsivity_hpol = file['impulsivity'][filter_string]['hpol'][...][total_loading_cut]
                            impulsivity_vpol = file['impulsivity'][filter_string]['vpol'][...][total_loading_cut]
                            peak_freq = {}
                            for channel in range(8):
                                peak_freq[channel] = file['inband_peak_freq_MHz'][...][total_loading_cut,channel]
                        else:
                            impulsivity_hpol = numpy.append(impulsivity_hpol,file['impulsivity'][filter_string]['hpol'][...][total_loading_cut])
                            impulsivity_vpol = numpy.append(impulsivity_vpol,file['impulsivity'][filter_string]['vpol'][...][total_loading_cut])
                            for channel in range(8):
                                peak_freq[channel] = numpy.append(peak_freq[channel],file['inband_peak_freq_MHz'][...][total_loading_cut,channel])
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


    try:
        for channel in range(8):
            fig,ax = plt.subplots()
            unique_freqs = numpy.unique(peak_freq[channel])
            step = scipy.stats.mode(numpy.diff(unique_freqs))[0][0]
            #freq_bins = numpy.arange(min(unique_freqs) - step/2,max(unique_freqs)+step,step)
            freq_bins = numpy.arange(10 - step/2,100+step,step)
            if channel%2 == 0:
                h = plt.hist2d(peak_freq[channel],impulsivity_hpol,bins=[freq_bins,1000],norm=LogNorm())
                plt.ylabel('Hpol Impulsivity')
            else:
                h = plt.hist2d(peak_freq[channel],impulsivity_vpol,bins=[freq_bins,1000],norm=LogNorm())
                plt.ylabel('vpol Impulsivity')

            plt.xlabel('Peak Frequency for Ch %i'%channel)
            plt.colorbar(h[3], ax=ax)

    except Exception as e:
        print('Error in plotting.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    

