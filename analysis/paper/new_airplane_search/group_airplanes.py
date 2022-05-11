#!/usr/bin/env python3
'''
This script is intended to look at the events that construct best above horizon in allsky maps.  I want to view the
peak to sidelobe values and other parameters for the belowhorizon and abovehorizon maps and determine if there is
an obvious cut for which sidelobed above horizon events can be discriminated. 
'''

import sys
import os
import gc
import inspect
import h5py
import copy
from pprint import pprint
import textwrap
from multiprocessing import Process

import numpy
import scipy
import scipy.signal
import scipy.signal

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.line_of_sight import circleSource
from beacon.tools.get_sun_coords_from_timestamp import getSunElWeightsFromRunDict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, FuncFormatter
import matplotlib.dates as mdates
import time
from datetime import datetime
import pytz
import pandas as pd
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

def saveprint(*args, outfile=None):
    print(*args) #Print once to command line
    if outfile is not None:
        print(*args, file=open(outfile,'a')) #Print once to file


start_time = time.time()

raw_datapath = os.environ['BEACON_DATA']

gc.enable()

processed_datapath = os.environ['BEACON_PROCESSED_DATA']



class SelectFromCollection(object):
    """
    This is an adapted bit of code that prints out information
    about the lasso'd data points. 

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.
    """

    def __init__(self, ax, collection, eventids, run_numbers):
        self.groups = []
        self.canvas = ax.figure.canvas
        self.eventids = eventids
        self.run_numbers = run_numbers
        self.id = numpy.array(list(zip(self.run_numbers,self.eventids)))
        self.collection = collection

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []


    def onselect(self, verts):
        path = Path(verts)
        self.ind = numpy.nonzero(path.contains_points(self.xys))[0]
        print('Selected run/eventids:')
        print(repr(self.id[self.ind]))
        event_info = self.id[self.ind]
        self.groups.append(event_info)
        #print(event_info)
        print('Coordinates:')
        print(repr(self.xys[self.ind]))
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()


def selectorGroupsToDict(selector, savename=None):
    '''
    Will take the event selector group and turn it into an eventids_dict
    '''
    eventids_dict = {}
    for g in selector.groups:
        if len(g) != 0:
            for run in numpy.unique(g[:,0]):
                run_cut = g[:,0] == run
                if run in eventids_dict.keys():
                    eventids_dict[run] = numpy.append(eventids_dict[run] ,g[:,1][run_cut])
                    eventids_dict[run] = numpy.sort(numpy.unique(eventids_dict[run]))
                else:
                    eventids_dict[run] = numpy.sort(g[:,1][run_cut])

    if savename is not None:
        numpy.save(savename, eventids_dict, allow_pickle=True)
    return eventids_dict





if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length
    
    run_batches = {}
    run_batches['batch_0'] = numpy.arange(5733,5974) # September data, should setup to auto add info to the "notes" section based off of existing sorting, and run this one those events for consistency
    run_batches['batch_1'] = numpy.arange(5974,6073)
    run_batches['batch_2'] = numpy.arange(6074,6173)
    run_batches['batch_3'] = numpy.arange(6174,6273)
    run_batches['batch_4'] = numpy.arange(6274,6373)
    run_batches['batch_5'] = numpy.arange(6374,6473)
    run_batches['batch_6'] = numpy.arange(6474,6573)
    run_batches['batch_7'] = numpy.arange(6574,6641)
    
    if False:
        runs = numpy.array([])
        for k in (run_batches.keys()):
            runs = numpy.append(runs,run_batches[k])
    else:
        batch = 0
        runs = run_batches['batch_%i'%batch]


    # runs = numpy.arange(6027,6037)

    runs = runs.astype(int)

    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, \
                    low_ram_mode=True,\
                    analysis_data_dir=processed_datapath, trigger_types=[2], remove_incomplete_runs=True)

    new_cut_dict = new_cut_dict = numpy.load( os.path.join( '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/cuts_run5733-run6640_1652152119' , 'pass_all_cuts_eventids_dict.npy')  , allow_pickle=True)[()]

    old_cut_dicts = {}
    for i in range(8):
        f = os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/eventid_dicts', 'stage_2_eventids_dict_batch_%i.npy'%i)
        out = numpy.load( f , allow_pickle=True)[()]
        for run in list(out.keys()):
            old_cut_dicts[run] = out[run]

    new_cut_array = ds.organizeEventDict(new_cut_dict)
    old_cut_array = ds.organizeEventDict(old_cut_dicts)

    only_new_dict               = ds.returnEventsAWithoutB(new_cut_dict, old_cut_dicts)
    only_new_array              = ds.organizeEventDict(only_new_dict)
    matching_cut_dict           = ds.returnCommonEvents(new_cut_dict, old_cut_dicts)
    matching_array              = ds.organizeEventDict(matching_cut_dict)
    

    only_new_azimuth            = ds.getDataArrayFromParam('phi_best_choice', eventids_dict=only_new_dict)
    only_new_elevation          = ds.getDataArrayFromParam('elevation_best_choice', eventids_dict=only_new_dict)
    only_new_times              = ds.getDataArrayFromParam('calibrated_trigtime', eventids_dict=only_new_dict)
    only_new_impulsivity        = ds.getDataArrayFromParam('impulsivity_hSLICERADDimpulsivity_v', eventids_dict=only_new_dict)

    only_matching_azimuth       = ds.getDataArrayFromParam('phi_best_choice', eventids_dict=matching_cut_dict)
    only_matching_elevation     = ds.getDataArrayFromParam('elevation_best_choice', eventids_dict=matching_cut_dict)
    only_matching_times         = ds.getDataArrayFromParam('calibrated_trigtime', eventids_dict=matching_cut_dict)
    only_matching_impulsivity   = ds.getDataArrayFromParam('impulsivity_hSLICERADDimpulsivity_v', eventids_dict=matching_cut_dict)


    markersize = 40

    vmin = min(min(only_new_impulsivity), min(only_matching_impulsivity))
    vmax = max(max(only_new_impulsivity), max(only_matching_impulsivity))
    min_t = min(min(only_new_times), min(only_matching_times))

    normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    fig1 = plt.figure(figsize=(16,9))
    ax1 = plt.gca()
    
    s_a = plt.scatter( (only_new_times - min_t)/3600.0, only_new_azimuth, c=only_new_impulsivity, norm=normalize, marker='o', label='New', cmap='inferno')
    time_selector_only_new = SelectFromCollection(ax1, s_a, only_new_array['eventid'], only_new_array['run'])

    s_b = plt.scatter( (only_matching_times - min_t)/3600.0, only_matching_azimuth, c=only_matching_impulsivity, norm=normalize, marker=',', label='Matching', cmap='inferno')
    time_selector_only_old = SelectFromCollection(ax1, s_b, only_new_array['eventid'], only_new_array['run'])

    plt.ylabel('Azimuth (deg)')
    plt.xlabel('Trigger Time (hours)')
    plt.legend()


    fig2 = plt.figure(figsize=(16,9))
    ax2 = plt.gca()
    
    s_c = plt.scatter( only_new_azimuth, only_new_elevation, c=only_new_impulsivity, norm=normalize, marker='o', label='New', cmap='inferno')
    space_selector_only_new = SelectFromCollection(ax2, s_c, only_new_array['eventid'], only_new_array['run'])

    s_d = plt.scatter( only_matching_azimuth, only_matching_elevation, c=only_matching_impulsivity, norm=normalize, marker=',', label='Matching', cmap='inferno')
    space_selector_only_old = SelectFromCollection(ax2, s_d, only_matching_array['eventid'], only_new_array['run'])

    plt.ylabel('Azimuth (deg)')
    plt.xlabel('Trigger Time (hours)')
    plt.legend()


    if False:
        # only makes sense to run after code circles made.
        selectorGroupsToDict(time_selector_only_new, savename='./likely_airplanes_batch_%i_%i.npy'%(batch, start_time))
        selectorGroupsToDict(space_selector_only_new, savename='./likely_sidelobes_batch_%i_%i.npy'%(batch, start_time))



    # #--------------------#


    # fig2 = plt.figure(figsize=(16,9))
    # ax2 = plt.gca()

    # s1 = plt.scatter( (only_new_times - min_t)/3600.0, only_new_azimuth, c=only_new_impulsivity, norm=normalize, marker='o', label='New', cmap='inferno')
    # s2 = plt.scatter( (only_matching_times - min_t)/3600.0, only_matching_azimuth, c=only_matching_impulsivity, norm=normalize, marker='*', label='Matching', cmap='inferno')

    # plt.ylabel('Elevation (deg)')
    # plt.xlabel('Trigger Time (hours)')
    # plt.legend()