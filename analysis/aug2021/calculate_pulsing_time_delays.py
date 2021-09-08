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
from beacon.tools.fftmath import TimeDelayCalculator

from beacon.analysis.find_pulsing_signals_june2021_deployment import Selector

from beacon.analysis.aug2021.parse_pulsing_runs import PulserInfo, predictAlignment


deploy_index = info.returnDefaultDeploy()

#Filter settings
final_corr_length = 2**16 #Should be a factor of 2 for fastest performance
apply_phase_response = True

# crit_freq_low_pass_MHz = [100,75,75,75,75,75,75,75]
# low_pass_filter_order = [8,12,14,14,14,14,14,14]

# crit_freq_high_pass_MHz = None#30#None#50
# high_pass_filter_order = None#5#None#8

crit_freq_low_pass_MHz = None#90#[80,70,70,70,70,70,60,70] #Filters here are attempting to correct for differences in signals from pulsers.
low_pass_filter_order = None#6#[0,8,8,8,10,8,3,8]

crit_freq_high_pass_MHz = None#30#65
high_pass_filter_order = None#12#12

sine_subtract = True
sine_subtract_min_freq_GHz = 0.02
sine_subtract_max_freq_GHz = 0.15
sine_subtract_percent = 0.03

plot_filters = False
plot_multiple = False

hilbert = False #Apply hilbert envelope to wf before correlating
align_method = 9


shorten_signals = False
shorten_thresh = 0.7
shorten_delay = 10.0
shorten_length = 90.0


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

        known_pulser_ids = info.load2021PulserEventids()

        for site in ['d3sa']:
            all_event_info = numpy.append(known_pulser_ids[site]['hpol'],known_pulser_ids[site]['vpol'])
            runs = numpy.sort(numpy.unique(all_event_info['run']))
            hpol_cut = numpy.append(numpy.ones(len(known_pulser_ids[site]['hpol']),dtype=bool),numpy.zeros(len(known_pulser_ids[site]['vpol']),dtype=bool))
            vpol_cut = ~hpol_cut

            #Prepare correlators for future use on a per event basis
            source_latlonel = pulser_info.getPulserLatLonEl(site)
            
            # Prepare expected angle and arrival times
            enu = pm.geodetic2enu(source_latlonel[0],source_latlonel[1],source_latlonel[2],origin[0],origin[1],origin[2])
            source_distance_m = numpy.sqrt(enu[0]**2 + enu[1]**2 + enu[2]**2)
            azimuth_deg = numpy.rad2deg(numpy.arctan2(enu[1],enu[0]))
            zenith_deg = numpy.rad2deg(numpy.arccos(enu[2]/source_distance_m))
            
            #Calculate Expected Time Delays
            cor_reader = Reader(os.environ['BEACON_DATA'],runs[0])
            cor = Correlator(cor_reader,upsample=len(cor_reader.t())*8, n_phi=180*4 + 1, n_theta=360*4 + 1,deploy_index=deploy_index)

            if True:
                expected_time_delays_hpol = predictAlignment(azimuth_deg, zenith_deg, cor, pol='hpol')
                expected_time_delays_vpol = predictAlignment(azimuth_deg, zenith_deg, cor, pol='vpol')
                align_method_10_window_ns = 20
            elif site == 'd2sa':
                #USE IF YOU HAVE ALREADY RUN THIS ONCE AND SUSPECT YOU KNOW BETTER
                align_method_10_window_ns = 5
                expected_time_delays_hpol = numpy.array([-63.1,99.4,15.8,162.49,78.4,-84.4])
                expected_time_delays_vpol = numpy.array([-70.0,86.8,11.54,156.7,81.4,-75])
            elif site == 'd3sa':
                #USE IF YOU HAVE ALREADY RUN THIS ONCE AND SUSPECT YOU KNOW BETTER
                align_method_10_window_ns = 5
                expected_time_delays_hpol = numpy.array([])
                expected_time_delays_vpol = numpy.array([])

            time_delay_dict = {'hpol':{'[0, 1]' : [expected_time_delays_hpol[0]], '[0, 2]': [expected_time_delays_hpol[1]], '[0, 3]': [expected_time_delays_hpol[2]], '[1, 2]': [expected_time_delays_hpol[3]], '[1, 3]': [expected_time_delays_hpol[4]], '[2, 3]': [expected_time_delays_hpol[5]]},
                               'vpol':{'[0, 1]' : [expected_time_delays_vpol[0]], '[0, 2]': [expected_time_delays_vpol[1]], '[0, 3]': [expected_time_delays_vpol[2]], '[1, 2]': [expected_time_delays_vpol[3]], '[1, 3]': [expected_time_delays_vpol[4]], '[2, 3]': [expected_time_delays_vpol[5]]}}
            align_method_10_estimates = numpy.append(expected_time_delays_hpol,expected_time_delays_vpol)


            #Setup template compare tools
            tdcs = {}
            for run in runs:
                reader = Reader(os.environ['BEACON_DATA'],run)
                tdcs[run] = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=(None,None), plot_filters=plot_filters, apply_phase_response=apply_phase_response)
                if sine_subtract == True:
                    tdcs[run].addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

            all_time_delays = numpy.zeros((12,len(all_event_info)))
            all_max_corrs = numpy.zeros((12,len(all_event_info)))
            for run_index, run in enumerate(runs):
                run_cut = numpy.where(all_event_info['run'] == run)[0]
                timeshifts, max_corrs, pairs = tdcs[run].calculateMultipleTimeDelays(all_event_info['eventid'][run_cut], align_method=align_method, hilbert=hilbert, plot=False, hpol_cut=hpol_cut, vpol_cut=vpol_cut, colors=None, align_method_10_estimates=align_method_10_estimates, align_method_10_window_ns=align_method_10_window_ns, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0,sine_subtract=False, crosspol_delays=False)
                all_time_delays[:,run_cut] = timeshifts
                all_max_corrs[:,run_cut] = max_corrs

            for pair_index, pair in enumerate(pairs):
                if pair[0]%2 == 0:
                    pol = 'hpol'
                else:
                    pol = 'vpol'

                fig = plt.figure()
                fig.canvas.set_window_title('%s %s'%(pol, str(pair//2)))
                plt.suptitle('Time Delays for Channels %s\n%s Antennas %s'%(str(pair),pol, str(pair//2)))
                
                ax1 = plt.subplot(2,1,1)
                dt = tdcs[runs[0]].dt_ns_upsampled
                bins = numpy.arange(align_method_10_estimates[pair_index] - align_method_10_window_ns, align_method_10_estimates[pair_index] + align_method_10_window_ns + dt, dt)
                plt.hist(all_time_delays[pair_index], bins=bins, color='k', alpha = 0.8)
                plt.axvline(align_method_10_estimates[pair_index],linestyle='--',c='k',label='Expected = %0.2f ns'%align_method_10_estimates[pair_index])
                plt.ylabel('Counts')

                ax2 = plt.subplot(2,1,2, sharex=ax1)
                for db in [0,3,6,10,13,20]:
                    db_cut = numpy.where(all_event_info['attenuation_dB'] == db)[0]
                    plt.hist(all_time_delays[pair_index,db_cut], bins=bins, label='%i dB attenuation'%db, alpha = 0.7, density=True)
                plt.axvline(align_method_10_estimates[pair_index],linestyle='--',c='k',label='Expected = %0.2f ns'%align_method_10_estimates[pair_index])
                plt.legend()
                plt.ylabel('Counts (Density)')
                plt.xlabel('Time Delay (ns)')

                

    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

