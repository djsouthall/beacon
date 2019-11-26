#!/usr/bin/env python3
'''
This uses the values calculated by rf_bg_search to make histograms of source locations.
'''
import os
import sys
import h5py
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
from tools.data_handler import createFile

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
plt.ion()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tools.correlator import Correlator


font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 16})


if __name__=="__main__":
    try:
        runs = numpy.arange(1645,1646)
        datapath = os.environ['BEACON_DATA']

        if True:
            highlight_roi = True
            d = 2.0 #Angular radius to cut on.
            az_angle_cut_edges = {}
            az_angle_cut_edges['hpol'] = [41.0-d,41.0+d]
            az_angle_cut_edges['vpol'] = [37.0-d,37.0+d]

            zen_angle_cut_edges = {}
            zen_angle_cut_edges['hpol'] = [151.0-d,151.0+d]
            zen_angle_cut_edges['vpol'] = [105.0-d,105.0+d]

            selected_events = {}
            selected_events['run'] = numpy.array([])
            selected_events['eventid'] = numpy.array([])
        else:
            highlight_roi = False


        az_edges = numpy.linspace(-180,180,181)
        az_centers = numpy.diff(az_edges)[0]/2.0 + az_edges[0:len(az_edges)-1]
        zen_edges = numpy.linspace(0,180,91)
        zen_centers = numpy.diff(zen_edges)[0]/2.0 + zen_edges[0:len(zen_edges)-1]

        hists = {}
        hists['hpol'] = numpy.zeros((len(zen_edges)-1,len(az_edges)-1))
        hists['vpol'] = numpy.zeros((len(zen_edges)-1,len(az_edges)-1))

        for run_index, run in enumerate(runs):
            try:
                run = int(run)

                reader = Reader(datapath,run)
                filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
                

                with h5py.File(filename, 'a') as file:
                    cut = file['trigger_type'][:] == 2
                    cut_ids = numpy.where(cut)[0]

                    current_selected_events = numpy.array([])

                    for mode in ['hpol','vpol']:
                        zen_values = file['%s_max_corr_dir_ENU_zenith'%mode][cut]
                        az_values = file['%s_max_corr_dir_ENU_azimuth'%mode][cut]
                        hists[mode] += numpy.histogram2d(az_values, zen_values, bins=(az_edges, zen_edges))[0].T

                        if highlight_roi == True:
                            zen_cut = numpy.logical_and(zen_values > zen_angle_cut_edges[mode][0],zen_values < zen_angle_cut_edges[mode][1])
                            az_cut = numpy.logical_and(az_values > az_angle_cut_edges[mode][0],az_values < az_angle_cut_edges[mode][1])
                            roi_cut = numpy.logical_and(zen_cut,az_cut)
                            current_selected_events = numpy.append(current_selected_events, cut_ids[roi_cut])
                    if highlight_roi == True:
                        sorted_events = numpy.sort(numpy.unique(current_selected_events)).astype(int)
                        selected_events['eventid'] = numpy.append(selected_events['eventid'], sorted_events)
                        selected_events['run'] = numpy.append(selected_events['run'], numpy.ones_like(sorted_events)*run)

                    file.close()
                        
            except Exception as e:
                print('\nError in %s'%inspect.stack()[0][3])
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
        
        selected_events['run'] = numpy.array(selected_events['run']).flatten()
        selected_events['eventid'] = numpy.array(selected_events['eventid']).flatten()

        plot_thresh = 1000
        masks = {}
        masks['hpol'] = hists['hpol'] > plot_thresh
        masks['vpol'] = hists['vpol'] > plot_thresh

        fig = plt.figure()
        fig.canvas.set_window_title('Hpol RF Source Directions')
        ax = fig.add_subplot(1,1,1)
        ax.set_title('Hpol RF Source Directions')
        im = ax.imshow(numpy.multiply(hists['hpol'],masks['hpol'].astype(int)),interpolation='none', extent=[min(az_edges),max(az_edges),max(zen_edges),min(zen_edges)],cmap='cool',norm=LogNorm()) #cmap=plt.cm.jet)
        cbar = fig.colorbar(im)
        cbar.set_label('Mean Correlation Value')
        plt.xlabel('Azimuth Angle (Degrees)')
        plt.ylabel('Zenith Angle (Degrees)')   

        row_index, column_index = numpy.unravel_index(hists['hpol'].argmax(),numpy.shape(hists['hpol']))
        print('hpol az = ',az_centers[column_index])
        print('hpol zen = ',zen_centers[row_index])
        print('This accounts for %0.2f percent of all RF triggers'%(100*hists['hpol'][row_index,column_index]/numpy.sum(hists['hpol'])))

        fig = plt.figure()
        fig.canvas.set_window_title('Vpol RF Source Directions')
        ax = fig.add_subplot(1,1,1)
        ax.set_title('Vpol RF Source Directions')
        im = ax.imshow(numpy.multiply(hists['vpol'],masks['vpol'].astype(int)),interpolation='none', extent=[min(az_edges),max(az_edges),max(zen_edges),min(zen_edges)],cmap='cool',norm=LogNorm()) #cmap=plt.cm.jet)
        cbar = fig.colorbar(im)
        cbar.set_label('Mean Correlation Value')
        plt.xlabel('Azimuth Angle (Degrees)')
        plt.ylabel('Zenith Angle (Degrees)')   

        row_index, column_index = numpy.unravel_index(hists['vpol'].argmax(),numpy.shape(hists['vpol']))
        print('vpol az = ',az_centers[column_index])
        print('vpol zen = ',zen_centers[row_index])
        print('This accounts for %0.2f percent of all RF triggers'%(100*hists['vpol'][row_index,column_index]/numpy.sum(hists['vpol'])))


        if highlight_roi:
            all_figs = []
            all_axs = []
            all_corrs = []
            for run in numpy.unique(selected_events['run']).astype(int):
                reader = Reader(datapath,run)
                run_cut = selected_events['run'] == run
                eventids = selected_events['eventid'][run_cut]
                upsample = 2**12
                cor = Correlator(reader,  upsample=upsample, n_phi=420, n_theta=420, waveform_index_range=(None,None),crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=False)
                for mode in ['hpol','vpol']:
                    mean_corr_values, fig, ax = cor.map(eventids[0], mode, plot_map=True, hilbert=False,interactive=True, max_method=0)
                    all_figs.append(fig)
                    all_axs.append(ax)
                #hpol, vpol = cor.averagedMap(eventids, 'both', plot_map=True, hilbert=False, max_method=0)


    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

