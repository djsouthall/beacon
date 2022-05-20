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

    def __init__(self, ax, collection, eventids, run_numbers, ds):
        self.ds = ds
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

        self.eventids_dict = {}
        for run in numpy.unique(event_info[:,0]):
            run_cut = event_info[:,0] == run
            self.eventids_dict[int(run)] = numpy.asarray(event_info[:,1][run_cut]).astype(int)

        print(self.eventids_dict)
        self.ds.eventInspector(self.eventids_dict, savedir='/home/dsouthall/Projects/Beacon/beacon/analysis/paper/new_airplane_search/figures/', azimuth_range=[-120,120])
        ds.inspector_mpl['fig1'].set_size_inches((24,16))


        self.groups.append(event_info)
        #print(event_info)
        print('Coordinates:')
        print(repr(self.xys[self.ind]))
        self.canvas.draw_idle()

        print('%i Circled Events'%len(event_info))
        for run in numpy.unique(event_info[:,0]):
            run_cut = event_info[:,0] == run
            for eventid in numpy.asarray(event_info[:,1][run_cut]).astype(int):
                print(run, ' - ', eventid)


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


def giveGroupKey(group, key, note='cluster', excel_filename='/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/new-cut-event-info_master.xlsx'):
    '''
    Will write a key to the group generated by selectors.
    '''

    with pd.ExcelFile(excel_filename) as xls:
        for sheet_index, sheet_name in enumerate(xls.sheet_names):
            if sheet_name != 'passing all cuts':
                continue
            else:
                df = xls.parse(sheet_name).copy(deep=True)

    for row_index, row in df.iterrows():
        run_cut = group[:,0] == int(row['run'])
        if int(row['eventid']) in group[:,1][run_cut]:
            if df['key'][row_index] != 'unsorted' and df['key'][row_index] != key:
                answer = None
                while answer not in ['y', 'n']:
                    answer = input("Change df['key'][%i] (%i, %i) from %s to %s?\ny/n"%(row_index, row['run'], row['eventid'], df['key'][row_index], key))
                if answer == 'y':
                    df['key'][row_index] = key
                    df['notes'][row_index] = str(df['notes'].fillna('')[row_index]) + str(note)
                else:
                    print('Skipping')
            else:
                df['key'][row_index] = key
                df['notes'][row_index] = str(df['notes'].fillna('')[row_index]) + str(note)

    with pd.ExcelWriter(excel_filename, engine="openpyxl", mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer,sheet_name='passing all cuts',float_format="%0.10f", index=False)
    
    return df

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
    
    if True:
        runs = numpy.array([])
        for k in (run_batches.keys()):
            runs = numpy.append(runs,run_batches[k])
    else:
        batch = 7
        runs = run_batches['batch_%i'%batch]


    # runs = numpy.arange(6027,6037)

    runs = runs.astype(int)

    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, \
                    low_ram_mode=True,\
                    analysis_data_dir=processed_datapath, trigger_types=[2], remove_incomplete_runs=True, preload_readers=False)

    excel_filename = '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/new-cut-event-info_master.xlsx'
    with pd.ExcelFile(excel_filename) as xls:
        for sheet_index, sheet_name in enumerate(xls.sheet_names):
            if sheet_name != 'passing all cuts':
                continue
            else:
                df = xls.parse(sheet_name)
                df = df.sort_values(by = ['run', 'eventid'], ascending = [True, True], na_position = 'last')
     #Df should exist

    new_cut_dict = numpy.load( os.path.join( '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/cuts_run5733-run6640_1652152119' , 'pass_all_cuts_eventids_dict.npy')  , allow_pickle=True)[()]

    old_cut_dicts = {}
    for i in range(8):
        f = os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/eventid_dicts', 'stage_2_eventids_dict_batch_%i.npy'%i)
        out = numpy.load( f , allow_pickle=True)[()]
        for run in list(out.keys()):
            old_cut_dicts[run] = out[run]

    print('Getting new and old cut arrays')
    new_cut_array = ds.organizeEventDict(new_cut_dict)
    old_cut_array = ds.organizeEventDict(old_cut_dicts)

    print('Sorting matching')
    only_new_dict               = ds.returnEventsAWithoutB(new_cut_dict, old_cut_dicts)
    only_new_array              = ds.organizeEventDict(only_new_dict)
    only_matching_cut_dict           = ds.returnCommonEvents(new_cut_dict, old_cut_dicts)
    only_matching_array              = ds.organizeEventDict(only_matching_cut_dict)


    print('Loading pre sorted clusters')

    file_airplane_roots = ['likely_airplanes_batch_1_1652289117.npy','likely_airplanes_batch_6_1652293161.npy','likely_airplanes_batch_7_1652293564.npy']
    file_sidelobe_roots = ['likely_sidelobes_batch_1_1652289117.npy','likely_sidelobes_batch_6_1652293161.npy','likely_sidelobes_batch_7_1652293564.npy']
    
    time_clustered_dicts = {}
    for fname in file_airplane_roots:
        f = os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/new_airplane_search', fname)
        out = numpy.load( f , allow_pickle=True)[()]
        for run in list(out.keys()):
            time_clustered_dicts[run] = out[run]

    space_clustered_dicts = {}
    for fname in file_sidelobe_roots:
        f = os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/new_airplane_search', fname)
        out = numpy.load( f , allow_pickle=True)[()]
        for run in list(out.keys()):
            space_clustered_dicts[run] = out[run]

    clustered_dict = ds.returnUniqueEvents(time_clustered_dicts,space_clustered_dicts)
    clustered_array = ds.organizeEventDict(clustered_dict)

    need_to_inspect_dict = ds.returnEventsAWithoutB(only_new_dict, clustered_dict)
    need_to_inspect_array = ds.organizeEventDict(need_to_inspect_dict)



    if False:
        print('Getting map values and impulsivity')
        if not numpy.all(df['eventid'].to_numpy() == new_cut_array['eventid']):
            print('WARNING NEED TO FIX SORTING')
            import pdb; pdb.set_trace()

        only_new_cut = numpy.zeros(len(df['run']),dtype=bool)
        only_matching_cut = numpy.zeros(len(df['run']),dtype=bool)

        df_runs = df['run'].to_numpy()
        for run in numpy.unique(df_runs):
            df_cut = df_runs == run
            only_new_cut[df_cut]        = numpy.isin(df[df_cut]['eventid'] , only_new_array['eventid'][only_new_array['run'] == run])
            only_matching_cut[df_cut]   = numpy.isin(df[df_cut]['eventid'] , only_matching_array['eventid'][only_matching_array['run'] == run])

        only_new_azimuth            = df['phi_best_choice'].to_numpy()[only_new_cut]#ds.getDataArrayFromParam('phi_best_choice', eventids_dict=only_new_dict)
        only_new_elevation          = df['elevation_best_choice'].to_numpy()[only_new_cut]#ds.getDataArrayFromParam('elevation_best_choice', eventids_dict=only_new_dict)
        only_new_times              = df['calibrated_trigtime'].to_numpy()[only_new_cut]#ds.getDataArrayFromParam('calibrated_trigtime', eventids_dict=only_new_dict)
        only_new_impulsivity        = df['impulsivity_hSLICERADDimpulsivity_v'].to_numpy()[only_new_cut]#ds.getDataArrayFromParam('impulsivity_hSLICERADDimpulsivity_v', eventids_dict=only_new_dict)

        only_matching_azimuth       = df['phi_best_choice'].to_numpy()[only_matching_cut]#ds.getDataArrayFromParam('phi_best_choice', eventids_dict=only_matching_cut_dict)
        only_matching_elevation     = df['elevation_best_choice'].to_numpy()[only_matching_cut]#ds.getDataArrayFromParam('elevation_best_choice', eventids_dict=only_matching_cut_dict)
        only_matching_times         = df['calibrated_trigtime'].to_numpy()[only_matching_cut]#ds.getDataArrayFromParam('calibrated_trigtime', eventids_dict=only_matching_cut_dict)
        only_matching_impulsivity   = df['impulsivity_hSLICERADDimpulsivity_v'].to_numpy()[only_matching_cut]#ds.getDataArrayFromParam('impulsivity_hSLICERADDimpulsivity_v', eventids_dict=only_matching_cut_dict)

        markersize = 40

        vmin = min(min(only_new_impulsivity), min(only_matching_impulsivity))
        vmax = max(max(only_new_impulsivity), max(only_matching_impulsivity))
        min_t = min(min(only_new_times), min(only_matching_times))
        max_t = max(max(only_new_times), max(only_matching_times))

        impulsivity_normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        time_normalize = matplotlib.colors.Normalize(vmin=min_t, vmax=max_t + 0.3*(max_t - min_t)) #Don't like top end of this cmap
        
        print('Ready to plot')

        fig1 = plt.figure(figsize=(16,9))
        ax1 = plt.gca()

        '''
        Want to add a different sorting such that events which have been sorted are obvious and ones that haven't been sorted are dots.  Rather than new v.s. not, or on top of it.
        Likely on top of it.  Color based on sorted or not.  Green sorted, red not. 
        colors = [cm.Set1(x) for x in numpy.linspace(0, 0.8, len(list(self.roi.keys())))]
        '''

        c_vals = numpy.zeros((len(only_new_times),4))
        c_vals[numpy.where(df['key'].to_numpy()[only_new_cut] == 'good')[0], :]         = numpy.ones((sum(df['key'].to_numpy()[only_new_cut] == 'good'),4))     *cm.Set1(0.25) #Green
        c_vals[numpy.where(df['key'].to_numpy()[only_new_cut] == 'bad')[0], :]          = numpy.ones((sum(df['key'].to_numpy()[only_new_cut] == 'bad'),4))      *cm.Set1(0.95) #Grey
        c_vals[numpy.where(df['key'].to_numpy()[only_new_cut] == 'ambiguous')[0], :]    = numpy.ones((sum(df['key'].to_numpy()[only_new_cut] == 'ambiguous'),4))*cm.Set1(0.45) #Orange
        c_vals[numpy.where(df['key'].to_numpy()[only_new_cut] == 'unsorted')[0], :]     = numpy.ones((sum(df['key'].to_numpy()[only_new_cut] == 'unsorted'),4)) *cm.Set1(0.05) #Red
        

        if True:
            unsorted_cut = numpy.where(df['key'].to_numpy()[only_new_cut] == 'unsorted')[0]
            sorted_cut = numpy.where(df['key'].to_numpy()[only_new_cut] != 'unsorted')[0]
            s_a = plt.scatter( (only_new_times[unsorted_cut] - min_t)/3600.0, only_new_azimuth[unsorted_cut], c=c_vals[unsorted_cut], marker='1', label='New')
            s_a2 = plt.scatter( (only_new_times[sorted_cut] - min_t)/3600.0, only_new_azimuth[sorted_cut], c=c_vals[sorted_cut], marker='o', s=(72./fig1.dpi)**2, label='sorted')

            time_selector_only_new = SelectFromCollection(ax1, s_a, df['eventid'].to_numpy()[only_new_cut][unsorted_cut], df['run'].to_numpy()[only_new_cut][unsorted_cut], ds)
        else:
            s_a = plt.scatter( (only_new_times - min_t)/3600.0, only_new_azimuth, c=only_new_impulsivity, norm=impulsivity_normalize, marker='1', label='New', cmap='inferno')
            time_selector_only_new = SelectFromCollection(ax1, s_a, df['eventid'].to_numpy()[only_new_cut], df['run'].to_numpy()[only_new_cut], ds)


        #Runs still off by one, not sure why




        c_vals = numpy.zeros((len(only_matching_times),4))
        c_vals[numpy.where(df['key'].to_numpy()[only_matching_cut] == 'good')[0], :]         = numpy.ones((sum(df['key'].to_numpy()[only_matching_cut] == 'good'),4))     *cm.Set1(0.25)
        c_vals[numpy.where(df['key'].to_numpy()[only_matching_cut] == 'bad')[0], :]          = numpy.ones((sum(df['key'].to_numpy()[only_matching_cut] == 'bad'),4))      *cm.Set1(0.95)
        c_vals[numpy.where(df['key'].to_numpy()[only_matching_cut] == 'ambiguous')[0], :]    = numpy.ones((sum(df['key'].to_numpy()[only_matching_cut] == 'ambiguous'),4))*cm.Set1(0.45)
        c_vals[numpy.where(df['key'].to_numpy()[only_matching_cut] == 'unsorted')[0], :]     = numpy.ones((sum(df['key'].to_numpy()[only_matching_cut] == 'unsorted'),4)) *cm.Set1(0.05)

        if True:
            s_b = plt.scatter( (only_matching_times - min_t)/3600.0, only_matching_azimuth, c=c_vals, marker='o', s=(72./fig1.dpi)**2, label='Matching')
        else:
            s_b = plt.plot( (only_matching_times - min_t)/3600.0, only_matching_azimuth, c='k', marker=',', linestyle='None', label='Matching')

        # s_b = plt.scatter( (only_matching_times - min_t)/3600.0, only_matching_azimuth, c=only_matching_impulsivity, norm=impulsivity_normalize, marker='.', label='Matching', cmap='inferno')
        # time_selector_only_old = SelectFromCollection(ax1, s_b, only_new_array['eventid'], only_new_array['run'])

        plt.ylabel('Azimuth (deg)')
        plt.xlabel('Trigger Time (hours)')
        plt.legend()


        fig2 = plt.figure(figsize=(16,9))
        ax2 = plt.gca()



        c_vals = numpy.zeros((len(only_new_times),4))
        c_vals[numpy.where(df['key'].to_numpy()[only_new_cut] == 'good')[0], :]         = numpy.ones((sum(df['key'].to_numpy()[only_new_cut] == 'good'),4))     *cm.Set1(0.25) #Green
        c_vals[numpy.where(df['key'].to_numpy()[only_new_cut] == 'bad')[0], :]          = numpy.ones((sum(df['key'].to_numpy()[only_new_cut] == 'bad'),4))      *cm.Set1(0.95) #Grey
        c_vals[numpy.where(df['key'].to_numpy()[only_new_cut] == 'ambiguous')[0], :]    = numpy.ones((sum(df['key'].to_numpy()[only_new_cut] == 'ambiguous'),4))*cm.Set1(0.45) #Orange
        c_vals[numpy.where(df['key'].to_numpy()[only_new_cut] == 'unsorted')[0], :]     = numpy.ones((sum(df['key'].to_numpy()[only_new_cut] == 'unsorted'),4)) *cm.Set1(0.05) #Red

        if True:
            unsorted_cut = numpy.where(df['key'].to_numpy()[only_new_cut] == 'unsorted')[0]
            sorted_cut = numpy.where(df['key'].to_numpy()[only_new_cut] != 'unsorted')[0]
            s_c = plt.scatter( only_new_azimuth[unsorted_cut], only_new_elevation[unsorted_cut], c=c_vals[unsorted_cut], marker='1', label='New')
            s_c2 = plt.scatter( only_new_azimuth[sorted_cut], only_new_elevation[sorted_cut], c=c_vals[sorted_cut], marker='o', s=(72./fig1.dpi)**2, label='sorted')
            space_selector_only_new = SelectFromCollection(ax2, s_c, df['eventid'].to_numpy()[only_new_cut][unsorted_cut], df['run'].to_numpy()[only_new_cut][unsorted_cut], ds)
        else:
            s_c = plt.scatter( only_new_azimuth, only_new_elevation, c=only_new_times, norm=time_normalize, marker='1', label='New', cmap='inferno')
            space_selector_only_new = SelectFromCollection(ax2, s_c, df['eventid'].to_numpy()[only_new_cut], df['run'].to_numpy()[only_new_cut], ds)




        c_vals = numpy.zeros((len(only_matching_times),4))
        c_vals[numpy.where(df['key'].to_numpy()[only_matching_cut] == 'good')[0], :]         = numpy.ones((sum(df['key'].to_numpy()[only_matching_cut] == 'good'),4))     *cm.Set1(0.25)
        c_vals[numpy.where(df['key'].to_numpy()[only_matching_cut] == 'bad')[0], :]          = numpy.ones((sum(df['key'].to_numpy()[only_matching_cut] == 'bad'),4))      *cm.Set1(0.95)
        c_vals[numpy.where(df['key'].to_numpy()[only_matching_cut] == 'ambiguous')[0], :]    = numpy.ones((sum(df['key'].to_numpy()[only_matching_cut] == 'ambiguous'),4))*cm.Set1(0.45)
        c_vals[numpy.where(df['key'].to_numpy()[only_matching_cut] == 'unsorted')[0], :]     = numpy.ones((sum(df['key'].to_numpy()[only_matching_cut] == 'unsorted'),4)) *cm.Set1(0.05)


        if True:
            s_d = plt.scatter( only_matching_azimuth, only_matching_elevation, c=c_vals, marker='o', s=(72./fig1.dpi)**2, label='Matching')
        else:
            s_d = plt.plot( only_matching_azimuth, only_matching_elevation, c='k', marker=',', linestyle='None', label='Matching')

        # s_d = plt.scatter( only_matching_azimuth, only_matching_elevation, c=only_matching_times, norm=time_normalize, marker='.', label='Matching', cmap='inferno')
        # space_selector_only_old = SelectFromCollection(ax2, s_d, only_matching_array['eventid'], only_new_array['run'])

        plt.ylabel('Elevation (deg)')
        plt.xlabel('Azimuth (deg)')
        plt.legend()


        if False:
            # only makes sense to run after code circles made.
            selectorGroupsToDict(time_selector_only_new, savename='./likely_airplanes_batch_%i_%i.npy'%(batch, start_time))
            selectorGroupsToDict(space_selector_only_new, savename='./likely_sidelobes_batch_%i_%i.npy'%(batch, start_time))
    else:
        '''
        For reviewing things after the fact.
        '''
        print('Getting map values and impulsivity')
        if not numpy.all(df['eventid'].to_numpy() == new_cut_array['eventid']):
            print('WARNING NEED TO FIX SORTING')
            import pdb; pdb.set_trace()


        markersize = 40
        
        print('Ready to plot')

        '''
        Want to add a different sorting such that events which have been sorted are obvious and ones that haven't been sorted are dots.  Rather than new v.s. not, or on top of it.
        Likely on top of it.  Color based on sorted or not.  Green sorted, red not. 
        colors = [cm.Set1(x) for x in numpy.linspace(0, 0.8, len(list(self.roi.keys())))]
        '''


        times = df['calibrated_trigtime'].to_numpy()
        azimuth = df['phi_best_choice'].to_numpy()
        elevation = df['elevation_best_choice'].to_numpy()
        min_t = min(times)

        c_vals = numpy.zeros((len(times),4))

        good_cut = numpy.where(numpy.logical_and(df['key'].to_numpy() == 'good', df['suspected_airplane_icao24'].fillna('').to_numpy() == ''))[0]
        bad_cut = numpy.where(df['key'].to_numpy() == 'bad')[0]
        ambiguous_cut =numpy.where(df['key'].to_numpy() == 'ambiguous')[0] 
        unsorted_cut = numpy.where(df['key'].to_numpy() == 'unsorted')[0]
        airplane_cut = numpy.where(df['suspected_airplane_icao24'].fillna('').to_numpy() != '')[0]



        c_vals[good_cut, :]      = numpy.ones((sum(numpy.logical_and(df['key'].to_numpy() == 'good', df['suspected_airplane_icao24'].fillna('').to_numpy() == '')),4))     *cm.Set1(0.25) #Green
        c_vals[bad_cut, :]       = numpy.ones((sum(df['key'].to_numpy() == 'bad'),4))      *cm.Set1(0.95) #Grey
        c_vals[ambiguous_cut, :] = numpy.ones((sum(df['key'].to_numpy() == 'ambiguous'),4))*cm.Set1(0.45) #Orange
        c_vals[unsorted_cut, :]  = numpy.ones((sum(df['key'].to_numpy() == 'unsorted'),4)) *cm.Set1(0.05) #Red
        c_vals[airplane_cut, :]  = numpy.ones((sum(df['suspected_airplane_icao24'].fillna('').to_numpy() != ''),4))      *cm.Set1(0.85) #Pink

        fig1 = plt.figure(figsize=(16,9))
        ax1 = plt.gca()
        for cut, name in [[good_cut, 'Good'], [airplane_cut, 'Airplane'], [ambiguous_cut, 'Ambiguous'], [bad_cut, 'Bad']]:
            if name == 'Good':
                s1 = plt.scatter( (times[cut] - min_t)/3600.0, azimuth[cut], c=c_vals[cut], marker='o', label=name)
                selector1 = SelectFromCollection(ax1, s1, df['eventid'].to_numpy()[cut], df['run'].to_numpy()[cut], ds)
            else:
                plt.scatter( (times[cut] - min_t)/3600.0, azimuth[cut], c=c_vals[cut], marker='o', s=(72./fig1.dpi)**2, label=name)
        plt.ylabel('Azimuth (deg)')
        plt.xlabel('Trigger Time (hours)')
        plt.legend()

        fig2 = plt.figure(figsize=(16,9))
        ax2 = plt.gca()
        for cut, name in [[good_cut, 'Good'], [airplane_cut, 'Airplane'], [ambiguous_cut, 'Ambiguous'], [bad_cut, 'Bad']]:
            if name == 'Good':
                s2 = plt.scatter( azimuth[cut], elevation[cut], c=c_vals[cut], marker='o', label=name)
                selector2 = SelectFromCollection(ax2, s2, df['eventid'].to_numpy()[cut], df['run'].to_numpy()[cut], ds)
            else:
                plt.scatter( azimuth[cut], elevation[cut], c=c_vals[cut], marker='o', s=(72./fig2.dpi)**2, label=name)
        plt.ylabel('Elevation (deg)')
        plt.xlabel('Azimuth (deg)')
        plt.legend()
