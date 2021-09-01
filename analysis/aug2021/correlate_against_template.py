#!/usr/bin/env python3
'''
Using the eventids highlighted by the parse pulsing runs, this will look at each event from the specified source
and compare it to the selected template event.  This template event will be used for a reference anchor of zero time
delay.  Both the correlation value and best alignment time will be stored for each event.
'''
import os
import sys
import numpy
import pymap3d as pm

from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
import beacon.tools.interpret #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_slicer import dataSlicerSingleRun,dataSlicer
from beacon.tools.correlator import Correlator
from beacon.tools.fftmath import TemplateCompareTool
import beacon.tools.info as info

import csv
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
from pprint import pprint
import itertools
import warnings
import h5py
import inspect
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import pandas as pd

plt.ion()

# import beacon.tools.info as info
from beacon.tools.data_handler import createFile, getTimes, loadTriggerTypes, getEventTimes
# from beacon.tools.fftmath import FFTPrepper
# from beacon.analysis.background_identify_60hz import plotSubVSec, plotSubVSecHist, alg1, diffFromPeriodic, get60HzEvents2, get60HzEvents3, get60HzEvents
from beacon.tools.fftmath import TemplateCompareTool

from beacon.analysis.find_pulsing_signals_june2021_deployment import Selector

from beacon.analysis.aug2021.parse_pulsing_runs import PulserInfo, predictAlignment


deploy_index = info.returnDefaultDeploy()

#Filter settings
final_corr_length = 2**15 #Should be a factor of 2 for fastest performance
apply_phase_response = True

# crit_freq_low_pass_MHz = [100,75,75,75,75,75,75,75]
# low_pass_filter_order = [8,12,14,14,14,14,14,14]

# crit_freq_high_pass_MHz = None#30#None#50
# high_pass_filter_order = None#5#None#8

crit_freq_low_pass_MHz = None#[80,70,70,70,70,70,60,70] #Filters here are attempting to correct for differences in signals from pulsers.
low_pass_filter_order = None#[0,8,8,8,10,8,3,8]

crit_freq_high_pass_MHz = None#65
high_pass_filter_order = None#12

sine_subtract = False
sine_subtract_min_freq_GHz = 0.03
sine_subtract_max_freq_GHz = 0.13
sine_subtract_percent = 0.03

plot_filters = False
plot_multiple = False

hilbert = False #Apply hilbert envelope to wf before correlating
align_method = 0

shorten_signals = True
shorten_thresh = 0.7
shorten_delay = 10.0
shorten_length = 90.0

class Selector(object):
    def __init__(self,ax,scatter_data, event_info, deploy_index):
        self.ax = ax
        self.xys = scatter_data.get_offsets()
        self.event_info = event_info
        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.readers = {}
        self.cors = {}
        for run in numpy.unique(self.event_info['run']):
            self.readers[run] = Reader(os.environ['BEACON_DATA'],run)
            self.cors[run] = Correlator(self.readers[run],upsample=len(self.readers[run].t())*8, n_phi=180*4 + 1, n_theta=360*4 + 1,deploy_index=deploy_index)

    def onselect(self,verts):
        #print(verts)
        path = Path(verts)
        ind = numpy.nonzero(path.contains_points(self.xys))[0]

        if type(self.event_info) == numpy.ma.core.MaskedArray:
            print('Only printing non-masked events:')
            eventids = self.event_info[ind][~self.event_info.mask[[ind]]]
            pprint(numpy.asarray(eventids[['run','eventid']]))
        else:
            eventids = self.event_info[ind]
            pprint(eventids[['run','eventid','attenuation_dB']])
        if len(eventids) == 1:
            eventid = eventids[0]['eventid']
            run = eventids[0]['run']
            fig = plt.figure()
            plt.title('Run %i, Eventid %i'%(run, eventid))
            self.readers[run].setEntry(eventid)
            ax = plt.subplot(2,1,1)
            for channel in [0,2,4,6]:
                plt.plot(self.readers[run].t(),self.readers[run].wf(channel))
            plt.ylabel('HPol Adu')
            plt.xlabel('t (ns)')
            plt.subplot(2,1,2,sharex=ax)
            plt.ylabel('VPol Adu')
            plt.xlabel('t (ns)')
            for channel in [1,3,5,7]:
                plt.plot(self.readers[run].t(),self.readers[run].wf(channel))
            self.cors[run].map(eventid,'both',interactive=True)


if __name__ == '__main__':
    plt.close('all')
    try:
        origin = info.loadAntennaZeroLocation()
        pulser_info = PulserInfo()

        sites_day2 = ['d2sa']
        sites_day3 = ['d3sa','d3sb','d3sc']
        sites_day4 = ['d4sa','d4sb']
        cors_list = [] #To keep interactive live
        lassos = []
        for site in sites_day4:
            #Prepare correlators for future use on a per event basis
            source_latlonel = pulser_info.getPulserLatLonEl(site)
            
            # Prepare expected angle and arrival times
            enu = pm.geodetic2enu(source_latlonel[0],source_latlonel[1],source_latlonel[2],origin[0],origin[1],origin[2])
            source_distance_m = numpy.sqrt(enu[0]**2 + enu[1]**2 + enu[2]**2)
            azimuth_deg = numpy.rad2deg(numpy.arctan2(enu[1],enu[0]))
            zenith_deg = numpy.rad2deg(numpy.arccos(enu[2]/source_distance_m))
            
            
            runs = pulser_info.returnRuns(site)

            cors = {}
            search_box_radius = 5
            for run in runs:
                reader = Reader(os.environ['BEACON_DATA'],run)
                cors[run] = Correlator(reader,upsample=len(reader.t())*4, n_phi=10, range_phi_deg=(azimuth_deg - search_box_radius, azimuth_deg + search_box_radius), n_theta=10, range_theta_deg=(zenith_deg - search_box_radius, zenith_deg + search_box_radius), deploy_index=deploy_index, map_source_distance_m = source_distance_m)

            #Setup template compare tools
            tcts = {}
            for run in runs:
                reader = Reader(os.environ['BEACON_DATA'],run)
                tcts[run] = TemplateCompareTool(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=(None,None), plot_filters=plot_filters, initial_template_id=None,apply_phase_response=apply_phase_response,sine_subtract=sine_subtract)
                if sine_subtract == True:
                    tcts[run].addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

            
            for pol in ['hpol', 'vpol']:
                reference_event = pulser_info.getPulserReferenceEvent(site, pol)
                
                #Set appropriate template for reference event run
                tcts[int(reference_event['run'])].setTemplateToEvent(int(reference_event['eventid']),sine_subtract=sine_subtract)
                tcts[int(reference_event['run'])].plotEvent(int(reference_event['eventid']), channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=hilbert, sine_subtract=False, apply_tukey=None, additional_title_text=None, time_delays=None, verbose=False)
                
                #Set that template for all other tcts
                for run in runs:
                    if run == reference_event['run']:
                        continue
                    else:
                        tcts[run].setTemplateToCustom(tcts[int(reference_event['run'])].template_ffts_filtered)
                
                # Prepare cor used for determining expeced time delays
                cor_reader = Reader(os.environ['BEACON_DATA'],reference_event['run'][0])
                cor = Correlator(cor_reader,upsample=len(cor_reader.t())*8, n_phi=180*4 + 1, n_theta=360*4 + 1,deploy_index=deploy_index, map_source_distance_m = source_distance_m)
                cors_list.append(cor) #Needed for interactive plotter. 

                pulser_candidates = pulser_info.getPulserCandidates(site, pol, attenuation=None)
                expected_time_delays = predictAlignment(azimuth_deg, zenith_deg, cor, pol=pol)
                time_delay_dict = {pol:{'[0, 1]' : [expected_time_delays[0]], '[0, 2]': [expected_time_delays[1]], '[0, 3]': [expected_time_delays[2]], '[1, 2]': [expected_time_delays[3]], '[1, 3]': [expected_time_delays[4]], '[2, 3]': [expected_time_delays[5]]}}
                
                #Plot map of the reference event in the specific polarization.
                cor.map(int(reference_event['eventid']), pol, include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, plot_corr=False, hilbert=hilbert, interactive=True, max_method=0, waveforms=None, verbose=True, mollweide=False, zenith_cut_ENU=[90,180], zenith_cut_array_plane=[0,91], center_dir='E', circle_zenith=zenith_deg, circle_az=azimuth_deg, radius=1.0, time_delay_dict=time_delay_dict,window_title='Map ' + str(reference_event),add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True)

                for run_index, run in enumerate(numpy.unique(pulser_candidates['run'])):
                    print('(%i/%i)\r'%(run_index + 1,len(numpy.unique(pulser_candidates['run']))))
                    run_cut = pulser_candidates['run'] == run
                    _max_corrs = numpy.zeros((sum(run_cut),8))
                    _map_max = numpy.zeros(sum(run_cut))
                    for event_index, eventid in enumerate(pulser_candidates[run_cut]['eventid']):
                        sys.stdout.write('(%i/%i)\r'%(event_index + 1,len(pulser_candidates[run_cut]['eventid'])))
                        sys.stdout.flush()
                        corrs, corrs_fft = tcts[run].crossCorrelateWithTemplate(eventid, load_upsampled_waveforms=False,sine_subtract=sine_subtract)
                        _max_corrs[event_index] = numpy.max(corrs,axis=1)
                        _map_max[event_index] = numpy.max(cors[run].map(int(eventid), pol, include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=False, plot_corr=False, hilbert=False, interactive=False, max_method=0, waveforms=None, verbose=False, mollweide=False, zenith_cut_ENU=[90,180], zenith_cut_array_plane=[0,91], center_dir='E'))
                        if False:
                            #Sanity check
                            if run == int(reference_event['run']) and eventid == int(reference_event['eventid']):
                                print('REFERENCE EVENT REACHED:')
                                print(_max_corrs[event_index])
                                print(_map_max[event_index])
                                cors[run].map(int(eventid), pol, include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, plot_corr=False, hilbert=False, interactive=False, max_method=0, waveforms=None, verbose=False, mollweide=False, zenith_cut_ENU=[90,180], zenith_cut_array_plane=[0,91], center_dir='E')
                                import pdb; pdb.set_trace()
                    if run_index == 0:
                        max_corrs = _max_corrs
                        map_max = _map_max
                    else:
                        max_corrs = numpy.vstack((max_corrs,_max_corrs))
                        map_max = numpy.append(map_max,_map_max)

                if pol == 'hpol':
                    channels = [0,2,4,6]
                else:
                    channels = [1,3,5,7]

                if False:
                    plt.figure()
                    fig.canvas.set_window_title('Corr %s'%(pol))
                    plt.suptitle(site + '\n' + pol)
                    bins = numpy.arange(0,101,1)/100.0
                    for attenuation in ['0dB','3dB','6dB','10dB','13dB','20dB']:
                        att = int(attenuation.replace('dB',''))
                        cut = numpy.where(pulser_candidates['attenuation_dB'] == att)[0]
                        for ch in channels:
                            plt.subplot(4,1,ch//2 + 1)
                            plt.hist(max_corrs[cut,ch],alpha=0.7,label='%s'%(str(attenuation)),bins=bins) #, color = plt.rcParams['axes.prop_cycle'].by_key()['color'][ch]
                            plt.ylabel('ch %i'%ch)
                    
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.legend()
                    plt.xlabel('Max Corr with Template Event\nrun %i, eventid %i'%(int(reference_event['run']),int(reference_event['eventid'])))

                if False:
                    fig = plt.figure()
                    fig.canvas.set_window_title('Map Max %s'%(pol))
                    plt.suptitle(site + '\n' + pol)
                    bins = numpy.linspace(0,1,100)
                    for attenuation in ['0dB','3dB','6dB','10dB','13dB','20dB']:
                        att = int(attenuation.replace('dB',''))
                        cut = numpy.where(pulser_candidates['attenuation_dB'] == att)[0]
                        plt.hist(map_max[cut],alpha=0.7,label='%s'%(str(attenuation)),bins=bins)
                        plt.ylabel('Counts')
                    plt.legend()
                    plt.xlabel('Max Map Correlation Value within %0.1f Degree\nSided Box Centered At Expected Source Direction'%(2*search_box_radius))
                    
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                if False:
                    for attenuation in ['0dB','3dB','6dB','10dB','13dB','20dB']:
                        att = int(attenuation.replace('dB',''))
                        cut = numpy.where(pulser_candidates['attenuation_dB'] == att)[0]
                        if sum(cut) == 0:
                            continue
                        fig, ax = plt.subplots()
                        fig.canvas.set_window_title('2D Hist %s %s'%(pol, attenuation))
                        
                        hist = plt.hist2d(numpy.max(max_corrs[cut], axis=1),   map_max[cut], bins = (numpy.linspace(0,1,100),numpy.linspace(0,1,100)), cmap='cool', norm=LogNorm(vmin = 0.1))

                        cbar = fig.colorbar(hist[3], ax=ax)
                        cbar.set_label('Counts')
                        plt.xlabel('Max Corr with Template Event\nrun %i, eventid %i'%(int(reference_event['run']),int(reference_event['eventid'])))
                        plt.ylabel('Max Map Correlation Value within %0.1f Degree\nSided Box Centered At Expected Source Direction'%(2*search_box_radius))
                        plt.suptitle(site + '\n' + pol + ' , ' + attenuation)

                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                if True:
                    for attenuation in ['0dB','3dB','6dB','10dB','13dB','20dB']:
                        att = int(attenuation.replace('dB',''))
                        cut = numpy.where(pulser_candidates['attenuation_dB'] == att)[0]
                        if sum(cut) == 0:
                            continue
                        fig, ax = plt.subplots()
                        fig.canvas.set_window_title('Scatter %s %s'%(pol, attenuation))
                        
                        scatter = plt.scatter(numpy.max(max_corrs[cut], axis=1),   map_max[cut])
                        _s = Selector(ax,scatter,pulser_candidates[cut],deploy_index)
                        lassos.append(_s)
                        plt.xlabel('Max Corr with Template Event\nrun %i, eventid %i'%(int(reference_event['run']),int(reference_event['eventid'])))
                        plt.ylabel('Max Map Correlation Value within %0.1f Degree\nSided Box Centered At Expected Source Direction'%(2*search_box_radius))
                        plt.suptitle(site + '\n' + pol + ' , ' + attenuation)

                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

