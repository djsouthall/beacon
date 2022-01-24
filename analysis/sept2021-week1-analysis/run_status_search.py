#!/usr/bin/env python3
'''
This is intended to look for any relationship between the runs that seem to readily be showing above horizon events
and any other measurable we have, like weather or trigger thresholds.
'''

import sys
import os
import inspect
import h5py
import copy
import csv
from pprint import pprint

import numpy
import scipy
import scipy.signal
from datetime import datetime
import pytz

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_handler import getTimes
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.flipbook_reader import flipbookToDict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=cm.get_cmap(plt.get_cmap('Pastel1'))(numpy.linspace(0,1,9)))



plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_jan21_2022')
#processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)

def weatherReader(filename):
    '''
    Given a filename of data taken by 
    https://dendra.science/orgs/ucnrs/datastreams?faceted=true&scheme=dq&selectStationId=58e68cacdf5ce600012602dc
    this will read it into a useable format.
    '''
    with open(filename, mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        data = []
        for row_index, row in enumerate(csv_reader):
            if row_index == 0:
                list_of_column_names = row
            else:
                data.append(row)
    data = numpy.asarray(data)
    out_dict = {}
    for column_index, column in enumerate(list_of_column_names):
        if column == 'Time':
            out_dict[column] = data[:,column_index] #Need to convert to something not string
            out_dict['datetime'] = numpy.array([datetime.strptime(t, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.timezone('America/Los_Angeles')) for t in data[:,column_index]])
            out_dict['timestamp'] = numpy.array([t.timestamp() for t in out_dict['datetime']])
        else:
            try:
                if ~numpy.all(data[:,column_index] == ''):
                    out_dict[column] = data[:,column_index].astype(float)
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()
    return out_dict

def getRunStartTimes(runs):
    run_start_times_utc = []
    run_start_times_datetime = []
    print('\nGetting Run Start Times')
    for run_index, run in enumerate(runs):
        sys.stdout.write('Run %i/%i\r'%(run_index+1,len(runs)))
        sys.stdout.flush()

        reader = Reader(raw_datapath,run)
        raw_approx_trigger_time, raw_approx_trigger_time_nsecs, trig_time, eventids = getTimes(reader)
        run_start_times_utc.append(raw_approx_trigger_time[0])
        run_start_times_datetime.append(datetime.fromtimestamp(raw_approx_trigger_time[0]).replace(tzinfo=pytz.timezone('America/Los_Angeles')))

    return numpy.asarray(run_start_times_utc), numpy.asarray(run_start_times_datetime)

def getTriggerRates(runs, eventid_interval=100):
    times_utc = numpy.array([])
    rates = numpy.array([])
    print('\nGetting Trigger Rates')
    for run_index, run in enumerate(runs):
        sys.stdout.write('Run %i/%i\r'%(run_index+1,len(runs)))
        sys.stdout.flush()

        reader = Reader(raw_datapath,run)
        raw_approx_trigger_time, raw_approx_trigger_time_nsecs, trig_time, eventids = getTimes(reader)
        t = numpy.asarray(raw_approx_trigger_time + raw_approx_trigger_time_nsecs/1e9)[0:-1:eventid_interval]
        t_centers = (t[1:] + t[:-1])/2
        rate = eventid_interval/((t[1:] - t[:-1]))

        times_utc = numpy.append(times_utc,t_centers)
        rates = numpy.append(rates,rate)

        if len(times_utc) != len(rates):
            import pdb; pdb.set_trace()

    datetimes = numpy.array([datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')) for t in times_utc])

    return numpy.asarray(rates), numpy.asarray(datetimes)

def getGlobalScalerTimeTrace(runs, sample_every=10):
    # beam_scalers = numpy.array([])
    # trigger_thresholds = numpy.array([])
    # readout_time_utc = numpy.array([])
    # # readout_time_datetime = []
    
    print('\nGetting Global Scalers Times')
    for run_index, run in enumerate(runs):
        sys.stdout.write('Run %i/%i\r'%(run_index+1,len(runs)))
        sys.stdout.flush()
        reader = Reader(raw_datapath,run)
        _beam_scalers, _trigger_thresholds, _readout_time = reader.returnBeamScalers(plot=False)
        if run_index == 0:
            beam_scalers        = _beam_scalers[:,0:-1:sample_every]
            trigger_thresholds  = _trigger_thresholds[:,0:-1:sample_every]
            readout_time_utc    = _readout_time[0:-1:sample_every]
        else:
            beam_scalers        = numpy.append(beam_scalers         , _beam_scalers[:,0:-1:sample_every], axis=1)
            trigger_thresholds  = numpy.append(trigger_thresholds   , _trigger_thresholds[:,0:-1:sample_every], axis=1)
            readout_time_utc    = numpy.append(readout_time_utc     , _readout_time[0:-1:sample_every])


    readout_time_datetime = numpy.array([datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')) for t in readout_time_utc])

    return beam_scalers, trigger_thresholds, readout_time_utc, readout_time_datetime

def getTriggerTimes(eventids_dict):
    out_dict = {}
    for run in list(eventids_dict.keys()):
        reader = Reader(raw_datapath,run)
        N = reader.head_tree.Draw("raw_approx_trigger_time_nsecs:raw_approx_trigger_time:trig_time:Entry$","","goff") 
        #ROOT.gSystem.ProcessEvents()
        raw_approx_trigger_time_nsecs = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N)
        raw_approx_trigger_time = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N) 
        trig_time = numpy.frombuffer(reader.head_tree.GetV3(), numpy.dtype('float64'), N)
        eventids = numpy.frombuffer(reader.head_tree.GetV4(), numpy.dtype('float64'), N).astype(int)
        out_dict[run] = raw_approx_trigger_time[eventids_dict[run].astype(int)] + raw_approx_trigger_time_nsecs[eventids_dict[run].astype(int)]/1e9
    return out_dict




if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length

    runs = numpy.arange(5733,5974)
    flipbook_path = '/home/dsouthall/scratch-midway2/event_flipbook_1642725413'
    weather_data_path = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'data', 'barcroft-weather-data-06-01-2021_01-21-2022.csv')

    print("Preparing dataSlicer")

    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)



    if False:
        ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                        n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)

    sorted_dict = flipbookToDict(flipbook_path)

    good_eventids_dict = sorted_dict['good']['eventids_dict']

    good_eventids_times_dict = getTriggerTimes(good_eventids_dict)

    plot_weather = True


    run_start_times_utc, run_start_times_datetime = getRunStartTimes(runs)

    ####
    # Plot Weather Events
    ####

    
    if plot_weather:
        weather_data = weatherReader(weather_data_path)


        first_plot = True    
        for key_index, key in enumerate(list(weather_data.keys())):
            if 'time' in key.lower():
                continue
            fig = plt.figure()
            if first_plot:
                ax = plt.subplot(1,1,1)
                ax.minorticks_on()
                ax.grid(b=True, which='major', color='k', linestyle='-',alpha=0.7)
                ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)
                first_plot = False
            else:
                _ax = plt.subplot(1,1,1,sharex=ax)
                _ax.minorticks_on()
                _ax.grid(b=True, which='major', color='k', linestyle='-',alpha=0.7)
                _ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)

            plt.plot(weather_data['datetime'] , weather_data[key], label=[key])

            first_run = True
            first_good_event = True
            for run_index, run in enumerate(runs):
                if run in list(good_eventids_dict.keys()):
                    c = 'r'
                    linewidth = 1
                    alpha = 0.8 
                    for t in good_eventids_times_dict[run]:
                        if first_good_event:
                            plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha,label='Passed Event Times')
                            first_good_event = False
                        else:
                            plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha)
                c = 'k'
                linewidth = 0.5
                alpha = 0.2
                if first_run:
                    plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha, label='Run Starts')
                    first_run = False
                else:
                    plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha)

            plt.xlim(min(run_start_times_datetime),max(run_start_times_datetime))
            plt.ylabel(key.split(' ')[-1])
            plt.xlabel('Date')
            plt.xticks(rotation = 45)
            plt.legend()
            plt.tight_layout()
            #Want to do similar for trigger rates. 
    else:
        ax = None

    ####
    # Plot Beam Scalers and Trigger Thresholds
    ####

    beam_scalers, trigger_thresholds, readout_time_utc, readout_time_datetime = getGlobalScalerTimeTrace(runs, sample_every=100)
    plt.figure()
    ax_beams = plt.subplot(2,1,1, sharex = ax)
    for beam_index, beam in enumerate(beam_scalers):
        plt.plot(readout_time_datetime, beam, linewidth=0.5)#, label='beam %i'%beam_index
    plt.plot(readout_time_datetime, numpy.mean(beam_scalers,axis=0), linewidth=1.0, c='k', label='Mean')
    plt.plot(readout_time_datetime, numpy.min(beam_scalers,axis=0), linewidth=1.0, c='#1f77b4', label='Min')
    plt.plot(readout_time_datetime, numpy.max(beam_scalers,axis=0), linewidth=1.0, c='#ff7f0e', label='Max')
    plt.ylabel('Beam Scaler')
    ax_beams.minorticks_on()
    ax_beams.grid(b=True, which='major', color='k', linestyle='-',alpha=0.7)
    ax_beams.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)

    first_run = True
    first_good_event = True
    for run_index, run in enumerate(runs):
        if run in list(good_eventids_dict.keys()):
            c = 'r'
            linewidth = 1
            alpha = 0.8 
            for t in good_eventids_times_dict[run]:
                if first_good_event:
                    plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha,label='Passed Event Times')
                    first_good_event = False
                else:
                    plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha)
        c = 'k'
        linewidth = 0.5
        alpha = 0.2
        if first_run:
            plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha, label='Run Starts')
            first_run = False
        else:
            plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha)
    plt.legend()
    # ax_beams.get_xaxis().set_ticklabels([])
    plt.ylim(0,100)

    ax_thresh = plt.subplot(2,1,2, sharex=ax_beams)
    for beam_index, beam in enumerate(trigger_thresholds):
        plt.plot(readout_time_datetime, beam, linewidth=0.5)#, label='beam %i'%beam_index)
    plt.plot(readout_time_datetime, numpy.mean(trigger_thresholds,axis=0), linewidth=1.0, c='k', label='Mean')
    plt.plot(readout_time_datetime, numpy.min(trigger_thresholds,axis=0), linewidth=1.0, c='#1f77b4', label='Min')
    plt.plot(readout_time_datetime, numpy.max(trigger_thresholds,axis=0), linewidth=1.0, c='#ff7f0e', label='Max')
    plt.ylabel('Trigger Threshold')
    plt.xlabel('Date')
    ax_thresh.minorticks_on()
    ax_thresh.grid(b=True, which='major', color='k', linestyle='-',alpha=0.7)
    ax_thresh.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)

    first_run = True
    first_good_event = True
    for run_index, run in enumerate(runs):
        if run in list(good_eventids_dict.keys()):
            c = 'r'
            linewidth = 1
            alpha = 0.8 
            for t in good_eventids_times_dict[run]:
                if first_good_event:
                    plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha,label='Passed Event Times')
                    first_good_event = False
                else:
                    plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha)
        c = 'k'
        linewidth = 0.5
        alpha = 0.2
        if first_run:
            plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha, label='Run Starts')
            first_run = False
        else:
            plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha)
    plt.xticks(rotation = 45)
    plt.tight_layout()


    ####
    # Plot Rates
    ####


    rates, rate_timestamps = getTriggerRates(runs, eventid_interval=5000)
    plt.figure()
    ax_rates = plt.subplot(1,1,1, sharex = ax)
    plt.plot(rate_timestamps, rates, linewidth=0.5,label='Event Rate')#, label='beam %i'%beam_index
    plt.ylabel('Events/s')
    plt.xlabel('Date')
    ax_rates.minorticks_on()
    ax_rates.grid(b=True, which='major', color='k', linestyle='-',alpha=0.7)
    ax_rates.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)

    first_run = True
    first_good_event = True
    for run_index, run in enumerate(runs):
        if run in list(good_eventids_dict.keys()):
            c = 'r'
            linewidth = 1
            alpha = 0.8 
            for t in good_eventids_times_dict[run]:
                if first_good_event:
                    plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha,label='Passed Event Times')
                    first_good_event = False
                else:
                    plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha)
        c = 'k'
        linewidth = 0.5
        alpha = 0.2
        if first_run:
            plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha, label='Run Starts')
            first_run = False
        else:
            plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha)
    plt.legend()
    plt.xticks(rotation = 45)
    plt.tight_layout()

    ####
    # Plot Combined
    ####



    if plot_weather:
        plt.figure()
        ax_beams = plt.subplot(3,1,1, sharex = ax)
        plt.plot(readout_time_datetime, numpy.mean(trigger_thresholds,axis=0), linewidth=1.0, c='k', label='Mean')
        plt.plot(readout_time_datetime, numpy.min(trigger_thresholds,axis=0), linewidth=1.0, c='#1f77b4', label='Min')
        plt.plot(readout_time_datetime, numpy.max(trigger_thresholds,axis=0), linewidth=1.0, c='#ff7f0e', label='Max')
        plt.ylabel('Trigger Threshold')
        ax_beams.minorticks_on()
        ax_beams.grid(b=True, which='major', color='k', linestyle='-',alpha=0.7)
        ax_beams.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)

        first_run = True
        first_good_event = True
        for run_index, run in enumerate(runs):
            if run in list(good_eventids_dict.keys()):
                c = 'r'
                linewidth = 1
                alpha = 0.8 
                for t in good_eventids_times_dict[run]:
                    if first_good_event:
                        plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha,label='Passed Event Times')
                        first_good_event = False
                    else:
                        plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha)
            c = 'k'
            linewidth = 0.5
            alpha = 0.2
            if first_run:
                plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha, label='Run Starts')
                first_run = False
            else:
                plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha)
        plt.legend(loc='lower left')
        # ax_beams.get_xaxis().set_ticklabels([])

        ax_rates = plt.subplot(3,1,2, sharex = ax)
        plt.plot(rate_timestamps, rates, linewidth=0.5,label='Event Rate')#, label='beam %i'%beam_index
        plt.ylabel('Events/s')
        ax_rates.minorticks_on()
        ax_rates.grid(b=True, which='major', color='k', linestyle='-',alpha=0.7)
        ax_rates.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)

        first_run = True
        first_good_event = True
        for run_index, run in enumerate(runs):
            if run in list(good_eventids_dict.keys()):
                c = 'r'
                linewidth = 1
                alpha = 0.8 
                for t in good_eventids_times_dict[run]:
                    if first_good_event:
                        plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha,label='Passed Event Times')
                        first_good_event = False
                    else:
                        plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha)
            c = 'k'
            linewidth = 0.5
            alpha = 0.2
            if first_run:
                plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha, label='Run Starts')
                first_run = False
            else:
                plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha)
        plt.legend(loc='lower left')
        # ax_rates.get_xaxis().set_ticklabels([])



        _ax = plt.subplot(3,1,3, sharex=ax_beams)
        _ax.minorticks_on()
        _ax.grid(b=True, which='major', color='k', linestyle='-',alpha=0.7)
        _ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.3)


        for key in ['WhiteMt Barcroft Air Temp Avg degC', 'WhiteMt Barcroft Wind Speed Avg m/s']:
            plt.plot(weather_data['datetime'] , weather_data[key], label=[key])

        first_run = True
        first_good_event = True
        for run_index, run in enumerate(runs):
            if run in list(good_eventids_dict.keys()):
                c = 'r'
                linewidth = 1
                alpha = 0.8 
                for t in good_eventids_times_dict[run]:
                    if first_good_event:
                        plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha,label='Passed Event Times')
                        first_good_event = False
                    else:
                        plt.axvline(datetime.fromtimestamp(t).replace(tzinfo=pytz.timezone('America/Los_Angeles')),linewidth=linewidth,c=c, alpha=alpha)
            c = 'k'
            linewidth = 0.5
            alpha = 0.2
            if first_run:
                plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha, label='Run Starts')
                first_run = False
            else:
                plt.axvline(run_start_times_datetime[run_index],linewidth=linewidth,c=c, alpha=alpha)

        plt.xlim(min(run_start_times_datetime),max(run_start_times_datetime))
        plt.ylabel('Weather Variable')
        plt.ylim(-8,18)
        plt.xlabel('Date')
        plt.xticks(rotation = 45)
        plt.legend(loc='lower left')
        plt.tight_layout()
