#!/usr/bin/env python3
'''
This is intended to calculate time delays after first averaging each waveform from a given set/channel.  The goal here 
is to generate the cleanest possibly waveform for each channel to remove any time delay ambiguity, and then to gain 
statistical error based on how each waveform aligns with that template.
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
final_corr_length = 2**18
apply_phase_response = True

crit_freq_low_pass_MHz = 80
low_pass_filter_order = 14

crit_freq_high_pass_MHz = 20
high_pass_filter_order = 4

sine_subtract = False
sine_subtract_min_freq_GHz = 0.02
sine_subtract_max_freq_GHz = 0.15
sine_subtract_percent = 0.01
max_failed_iterations = 3

plot_filters = False
plot_multiple = False

hilbert = False #Apply hilbert envelope to wf before correlating
align_method = 1 #0,1,2 


shorten_signals = False
shorten_thresh = 0.7
shorten_delay = 10.0
shorten_length = 90.0



def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))

'''
Baselines for Site d3sa hpol:
Delays in ns:
array([-100.9643034 ,   39.12872542,  -52.79778144,  140.02931026,
         48.10385574,  -92.04129368])
Sigma:
array([0.06557082, 0.07901542, 0.05177623, 0.07434778, 0.05871251,
       0.06644926])
Baselines:
array([[0, 2],
       [0, 4],
       [0, 6],
       [2, 4],
       [2, 6],
       [4, 6]])


Event 2918 set as template
Optimal parameters not found: Number of calls to function has reached maxfev = 800.
Error trying to fit gaussian to data for Channels [1, 3] vpol d3sa
Event 2918 set as template
Event 2918 set as template  
Event 2918 set as template  
Event 2918 set as template  
Event 2918 set as template  
(510/510)           

Baselines for Site d3sa vpol:
Delays in ns:
array([-107.24171021,   26.75577314,  -56.59551099,  134.310878  ,
         50.92560667,  -83.4251287 ])
Sigma:
array([0.00387661, 0.11545836, 0.16632272, 0.14901272, 0.26207502,
       0.18229291])
Baselines:
array([[1, 3],
       [1, 5],
       [1, 7],
       [3, 5],
       [3, 7],
       [5, 7]])
'''

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

        #Set Baseline
        sites = [sites_day3[0]]
        pols = ['vpol']
        limit_eventids = 50000
        baseline_antennas = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]

        #The attenuations to include for each site and polarization
        attenuations_dict = {'hpol':{   'd2sa' : [10],
                                        'd3sa' : [10],
                                        'd3sb' : [6],
                                        'd3sc' : [10],
                                        'd4sa' : [6],
                                        'd4sb' : [6]
                                    },
                             'vpol':{   'd2sa' : [10],
                                        'd3sa' : [6],
                                        'd3sb' : [20],
                                        'd3sc' : [10],
                                        'd4sa' : [6],
                                        'd4sb' : [6]
                                    }
                            }


        known_pulser_ids = info.load2021PulserEventids()

        for site in sites:
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

            time_delay_dict = {'hpol':{'[0, 1]' : [expected_time_delays_hpol[0]], '[0, 2]': [expected_time_delays_hpol[1]], '[0, 3]': [expected_time_delays_hpol[2]], '[1, 2]': [expected_time_delays_hpol[3]], '[1, 3]': [expected_time_delays_hpol[4]], '[2, 3]': [expected_time_delays_hpol[5]]},
                               'vpol':{'[0, 1]' : [expected_time_delays_vpol[0]], '[0, 2]': [expected_time_delays_vpol[1]], '[0, 3]': [expected_time_delays_vpol[2]], '[1, 2]': [expected_time_delays_vpol[3]], '[1, 3]': [expected_time_delays_vpol[4]], '[2, 3]': [expected_time_delays_vpol[5]]}}
            align_method_10_estimates = numpy.append(expected_time_delays_hpol,expected_time_delays_vpol)


            #Setup template compare tools
            tcts = {}
            for run in runs:
                reader = Reader(os.environ['BEACON_DATA'],run)
                tcts[run] = TemplateCompareTool(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=(None,None), plot_filters=plot_filters, initial_template_id=None,apply_phase_response=apply_phase_response,sine_subtract=sine_subtract)
                if sine_subtract == True:
                    tcts[run].addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=max_failed_iterations, verbose=False, plot=False)

            
            for pol in pols:
                finalized_time_delays = []
                finalized_sigma = []
                reference_event = pulser_info.getPulserReferenceEvent(site, pol)
                for baseline in baseline_antennas:
                    if pol == 'hpol':
                        channels = [min(baseline)*2, max(baseline)*2]
                    elif pol == 'vpol':
                        channels = [min(baseline)*2 + 1, max(baseline)*2 + 1]
                    
                    #Set appropriate template for reference event run
                    tcts[int(reference_event['run'])].setTemplateToEvent(int(reference_event['eventid']),sine_subtract=sine_subtract)
                    if False:
                        if sine_subtract == True:
                            tcts[int(reference_event['run'])].plotEvent(int(reference_event['eventid']), channels=channels, apply_filter=True, hilbert=hilbert, sine_subtract=False, apply_tukey=None, additional_title_text=None, time_delays=None, verbose=False)
                            tcts[int(reference_event['run'])].plotEvent(int(reference_event['eventid']), channels=channels, apply_filter=True, hilbert=hilbert, sine_subtract=True, apply_tukey=None, additional_title_text=None, time_delays=None, verbose=False)
                        else:
                            tcts[int(reference_event['run'])].plotEvent(int(reference_event['eventid']), channels=channels, apply_filter=True, hilbert=hilbert, sine_subtract=False, apply_tukey=None, additional_title_text=None, time_delays=None, verbose=False)
                    
                    #Set that template for all other tcts
                    for run in runs:
                        if run == reference_event['run']:
                            continue
                        else:
                            tcts[run].setTemplateToCustom(tcts[int(reference_event['run'])].template_ffts_filtered)
                    
                    if pol == 'hpol':
                        cut = numpy.logical_and(numpy.isin(all_event_info['attenuation_dB'],attenuations_dict[pol][site]), hpol_cut)
                        
                    else:
                        cut = numpy.logical_and(numpy.isin(all_event_info['attenuation_dB'],attenuations_dict[pol][site]), vpol_cut)
                    #Cut is boolean array to here
                    if limit_eventids is not None:
                        cut[numpy.cumsum(cut) > limit_eventids] = False #will always choose the first 100
                        cut = numpy.sort(numpy.random.choice(numpy.where(cut)[0], numpy.min([limit_eventids,sum(cut)]),replace=False))
                    else:
                        cut = numpy.where(cut)[0]
                    #Cut is index array after here

                    run_0 = all_event_info[cut]['run'][0]
                    averaged_waveforms = numpy.zeros((len(channels),tcts[run_0].final_corr_length//2))
                    times = numpy.arange(tcts[run_0].final_corr_length//2)*tcts[run_0].dt_ns_upsampled

                    if len(cut) > 100:
                        plot_averaged = False
                    else:
                        plot_averaged = True

                    if plot_averaged == True:
                        figs = []
                        axs = []
                        for channel in channels:
                            fig = plt.figure()
                            fig.canvas.set_window_title('Ch %i Aligned Waveforms'%channel)
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                            plt.xlabel('t (ns)')
                            plt.ylabel('Ch %i Amp (Adu)'%channel)
                            figs.append(fig)
                            axs.append(plt.gca())

                    for event_index, event_info in enumerate(all_event_info[cut]):
                        sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(cut)))
                        max_corrs, upsampled_waveforms, rolled_wfs = tcts[event_info['run']].alignToTemplate(event_info['eventid'], channels=channels, align_method=align_method, sine_subtract=sine_subtract)
                        averaged_waveforms += rolled_wfs/len(cut)

                        if plot_averaged == True:
                            for channel_index, channel in enumerate(channels):
                                ax = axs[channel_index]
                                ax.plot(times, rolled_wfs[channel_index],alpha=0.2)

                    if plot_averaged == True:
                        #Repeat for template and plot on top. 
                        max_corrs, upsampled_waveforms, rolled_wfs = tcts[int(reference_event['run'])].alignToTemplate(int(reference_event['eventid']), channels=channels, align_method=align_method)

                        for channel_index, channel in enumerate(channels):
                            ax = axs[channel_index]
                            ax.plot(times, rolled_wfs[channel_index],linestyle='--',c='b',label=str(channel)+' template')
                            ax.legend()

                        for channel_index, channel in enumerate(channels):
                            ax = axs[channel_index]
                            ax.plot(times, averaged_waveforms[channel_index],linestyle='--',c='r',label=str(channel)+' avg')
                            ax.legend()

                        fig = plt.figure()
                        fig.canvas.set_window_title('Average Waveforms')
                        plt.minorticks_on()
                        plt.grid(b=True, which='major', color='k', linestyle='-')
                        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                        plt.xlabel('t (ns)')
                        plt.ylabel('Adu')
                        for channel_index, channel in enumerate(channels):
                            plt.plot(times, averaged_waveforms[channel_index],alpha=0.7,label=str(channel))
                        plt.legend()

                    # Resample the new waveforms and set them as the template event
                    resampled_wf = numpy.zeros((len(channels),len(tcts[run_0].waveform_times_corr)))
                    
                    resampled_wf[:,0:tcts[run_0].buffer_length], resampled_t = scipy.signal.resample(averaged_waveforms,len(tcts[run_0].waveform_times_original),t=times,domain='time',axis=1) #Because of how the tool works it must be resampled to the original window sampling before being zero padded to the required amount, all before storing it to as the FFT'd template
                    
                    ffts = numpy.fft.rfft(resampled_wf,axis=1)
                    template_ffts = tcts[int(reference_event['run'])].template_ffts_filtered
                    template_ffts[channels] = ffts
                    
                    #Set that template for all other tcts
                    for run in runs:
                        tcts[run].setTemplateToCustom(template_ffts)


                    if plot_averaged == True:
                        figs2 = []
                        axs2 = []
                        for channel in channels:
                            fig = plt.figure()
                            fig.canvas.set_window_title('Ch %i Internal Align'%channel)
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                            plt.xlabel('t (ns)')
                            plt.ylabel('Ch %i Amp (Adu)'%channel)
                            figs2.append(fig)
                            axs2.append(plt.gca())

                    #I now have a template for each channel, now I must either set it as a template event for the tcts, or just do it myself outside if the tct class
                    all_time_shifts = numpy.zeros((len(channels) , len(cut)))
                    all_max_corrs = numpy.zeros((len(channels) , len(cut)))
                    for event_index, event_info in enumerate(all_event_info[cut]):
                        sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(cut)))
                        max_corrs, upsampled_waveforms, rolled_wfs, index_delays, time_shifts = tcts[event_info['run']].alignToTemplate(event_info['eventid'], channels=channels,align_method=align_method,sine_subtract=sine_subtract, return_delays=True)
                        all_time_shifts[:,event_index] = time_shifts
                        all_max_corrs[:,event_index] = max_corrs

                        if plot_averaged == True:
                            for channel_index, channel in enumerate(channels):
                                ax = axs2[channel_index]
                                ax.plot(times, rolled_wfs[channel_index],alpha=0.2)

                    if plot_averaged == True:
                        #Repeat for template and plot on top. 
                        for channel_index, channel in enumerate(channels):
                            ax = axs2[channel_index]
                            ax.plot(times, averaged_waveforms[channel_index],linestyle='--',c='b',label=str(channel)+' template')
                            ax.legend()


                    template_indices, template_corr_time_shifts, template_max_corrs, template_pairs, template_corrs = tcts[event_info['run']].returnTemplateTimeDelays()

                    template_baseline_time_delay = template_corr_time_shifts[numpy.where(numpy.all(template_pairs == channels,axis=1))[0]]
                    all_time_delays = template_baseline_time_delay + all_time_shifts[0] - all_time_shifts[1] #sign may be off of difference here but it just flips the error so doesn't matter

                    #REMOVE OUTLIERS
                    all_time_delays = all_time_delays[numpy.abs(all_time_delays - numpy.median(all_time_delays)) < 20]

                    predicted_time_delay = numpy.append(expected_time_delays_hpol,expected_time_delays_vpol)[numpy.where(numpy.all(template_pairs == channels,axis=1))[0]]

                    fig = plt.figure(figsize=(16,9))
                    plt.title('Channels %s Time Delays\n%s %s'%(str(channels), pol, site))
                    fig.canvas.set_window_title('%s %s %s Time Delays'%(str(channels), pol, site))
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.xlabel('t (ns)')
                    plt.ylabel('Adu')

                    half_window = numpy.min((2,numpy.abs(template_baseline_time_delay - predicted_time_delay) + 1.0))

                    dt = tcts[run_0].dt_ns_upsampled
                    bins = numpy.arange(template_baseline_time_delay - half_window, template_baseline_time_delay + half_window + dt, dt) - dt/2.0
                    n, bins, patches = plt.hist(all_time_delays,bins=bins,color='#3DB2FF',label=str(channels),alpha=0.8, edgecolor='black', linewidth=1.0)
                    plt.axvline(predicted_time_delay,c='#FF2442',label='Predicted from Current Calibration\n%0.3f ns'%predicted_time_delay, linewidth=2.0)
                    plt.axvline(template_baseline_time_delay,c='#FFB830',label='Template Time Delay\n%0.3f ns'%template_baseline_time_delay, linewidth=2.0)
                    plt.axvline(numpy.mean(all_time_delays),c='#AE00FB',label='Mean Time Delay\n%0.3f ns'%numpy.mean(all_time_delays), linewidth=2.0)

                    try:
                        #Fit Gaussian
                        x = (bins[1:len(bins)] + bins[0:len(bins)-1] )/2.0
                        popt, pcov = curve_fit(gaus,x,n,p0=[numpy.max(n),numpy.mean((numpy.mean(all_time_delays),template_baseline_time_delay[0])),1.5*numpy.std(all_time_delays)])
                        popt[2] = abs(popt[2]) #I want positive sigma.

                        plot_x = numpy.linspace(min(x),max(x),1000)
                        plt.plot(plot_x,gaus(plot_x,*popt),'-',c='k',label='Fit Sigma = %f ns'%popt[2])

                        plt.axvline(popt[1],linestyle='-',c='k',label='Fit Center = %f'%popt[1])


                        finalized_time_delays.append(popt[1])
                        finalized_sigma.append(popt[2])
                    except Exception as e:
                        print(e)
                        print('Error trying to fit gaussian to data for Channels %s %s %s'%(str(channels), pol, site))
                        finalized_time_delays.append(-999)
                        finalized_sigma.append(-999)

                    plt.legend(loc='upper right')
                    plt.tight_layout()
                    fig.savefig('time_delay_%s_%s_baseline_%i-%i.png'%(site,pol,channels[0],channels[1]),dpi=90,transparent=True)
                print('\n\nBaselines for Site %s %s:'%(site,pol))
                print('Delays in ns:')
                pprint(numpy.array(finalized_time_delays))
                print('Sigma:')
                pprint(numpy.array(finalized_sigma))
                print('Baselines:')
                pprint(numpy.array(baseline_antennas)*2 + int(pol == 'vpol'))
                print('\n')




    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

