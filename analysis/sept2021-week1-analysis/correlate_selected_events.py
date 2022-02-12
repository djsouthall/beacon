#!/usr/bin/env python3
'''
This script is intended to look at the events that construct best above horizon in allsky maps.  I want to view the
peak to sidelobe values and other parameters for the belowhorizon and abovehorizon maps and determine if there is
an obvious cut for which sidelobed above horizon events can be discriminated. 
'''

import sys
import os
import inspect
import h5py
import copy
from pprint import pprint

import numpy
import scipy
import scipy.signal
import time

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TimeDelayCalculator
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.flipbook_reader import flipbookToDict
import  beacon.tools.get_plane_tracks as pt
from tools.airplane_traffic_loader import getDataFrames

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)


def enu2Spherical(enu):
    '''
    2d array like ((e_0, n_0, u_0), (e_1, n_1, u_1), ... , (e_i, n_i, u_i))

    Return in degrees
    '''
    r = numpy.linalg.norm(enu, axis=1)
    theta = numpy.degrees(numpy.arccos(enu[:,2]/r))
    phi = numpy.degrees(numpy.arctan2(enu[:,1],enu[:,0]))
    # import pdb; pdb.set_trace()
    return numpy.vstack((r,phi,theta)).T

if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length

    # _runs = numpy.arange(5733,5974)#[0:100]
    # bad_runs = numpy.array([5775])
    #_runs = numpy.arange(5733,5800)

    flipbook_path = '/home/dsouthall/Projects/Beacon/beacon/analysis/sept2021-week1-analysis/airplane_event_flipbook_1643947072'#'/home/dsouthall/scratch-midway2/event_flipbook_1643154940'#'/home/dsouthall/scratch-midway2/event_flipbook_1642725413'
    sorted_dict = flipbookToDict(flipbook_path)
    
    if True:
        good_dict = sorted_dict['no-obvious-airplane']['eventids_dict']
        runs = list(good_dict.keys())
    elif False:
        good_dict = {5805:[11079]}
        runs = numpy.array([list(good_dict.keys())[0]])
    else:
        good_dict = {5903:[86227]}
        runs = numpy.array([5903])

    datapath = os.environ['BEACON_DATA']
    align_method_13_n = 2

    crit_freq_low_pass_MHz = 80
    low_pass_filter_order = 14

    crit_freq_high_pass_MHz = 20
    high_pass_filter_order = 4

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.00
    sine_subtract_max_freq_GHz = 0.25
    sine_subtract_percent = 0.03

    apply_phase_response = True

    shorten_signals = False
    shorten_thresh = 0.7
    shorten_delay = 10.0
    shorten_length = 90.0

    notch_tv = True
    misc_notches = True
    # , notch_tv=notch_tv, misc_notches=misc_notches


    hilbert=False
    final_corr_length = 2**17
    waveform_index_range = [None,None]

    # map_resolution_theta = 0.25 #degrees
    # min_theta   = 0
    # max_theta   = 120
    # n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    # map_resolution_phi = 0.1 #degrees
    # min_phi     = -180
    # max_phi     = 180
    # n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)



    # ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
    #                 cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
    #                 std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
    #                 snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
    #                 n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)



    tdcs = {}
    for run in runs:
        reader = Reader(datapath,run)
        tdcs[run] = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches)
        if sine_subtract:
            tdcs[run].addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

    n_events = 0
    for run in runs:
        for eventid in good_dict[run]:
            n_events += 1
    max_ccs = {}
    max_ccs['hpol'] = numpy.zeros((n_events,n_events))
    max_ccs['vpol'] = numpy.zeros((n_events,n_events))
    max_ccs['all'] = numpy.zeros((n_events,n_events))

    channels = numpy.arange(8)
    hpol_channels = channels[::2]
    vpol_channels = hpol_channels = channels[::2]

    for channel in channels:
        fig = plt.figure()
        plt.title('Channel %i'%channel)
        fig.canvas.set_window_title('Channel %i'%channel)

        index = 0
        xticks = []
        for run in runs:
            for eventid in good_dict[run]:
                xticks.append('%i-%i'%(run,eventid))
                tdcs[run].setEntry(eventid)
                t = tdcs[run].t()#tdcs[run].dt_ns_upsampled
                wf = tdcs[run].wf(channel, apply_filter=True, hilbert=False, tukey=True, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)
                # plt.plot(t,wf, label='%i-%i'%(run,eventid))

                if index == 0:
                    waveforms = wf
                else:
                    waveforms = numpy.vstack((waveforms, wf))

                index += 1
        
        max_cc = numpy.zeros((len(waveforms), len(waveforms)))
        mask = numpy.zeros((len(waveforms), len(waveforms)))
        for i in range(len(waveforms)):
            for j in range(len(waveforms)):
                cc = numpy.correlate(waveforms[i]/numpy.std(waveforms[i]),waveforms[j]/numpy.std(waveforms[j]),mode='Full')/len(t)
                max_index = numpy.argmax(numpy.abs(cc))
                max_cc[i,j] = cc[max_index]
                mask[i,j] = i == j

        # x = numpy.arange(len(len(waveforms)))
        # y = numpy.arange(len(len(waveforms)))

        # max_cc[1,1] = 0.0

        max_cc = numpy.ma.masked_array(max_cc,mask=mask)
        max_value = numpy.abs(max_cc).max()

        im = plt.pcolormesh(max_cc, vmin=-max_value, vmax=max_value,cmap='coolwarm')#,shading='nearest'
        plt.colorbar(im)
        plt.xticks(numpy.arange(len(xticks)) + 0.5, xticks, size='small', rotation='vertical')
        plt.yticks(numpy.arange(len(xticks)) + 0.5, xticks, size='small')

        ax = plt.gca()
        ax.set_aspect('equal')

        if channel%2 == 0:
            max_ccs['hpol'] += max_cc/len(hpol_channels)
        else:
            max_ccs['vpol'] += max_cc/len(vpol_channels)

        max_ccs['all'] += max_cc/len(channels)


    for pol in ['hpol', 'vpol', 'all']:
        z = max_ccs[pol]
        z = numpy.ma.masked_array(z,mask=mask)
        fig = plt.figure()
        plt.title('Averaged %s'%pol)
        fig.canvas.set_window_title('Averaged %s'%pol)

        max_value = numpy.abs(z).max()

        im = plt.pcolormesh(z, vmin=-max_value, vmax=max_value,cmap='coolwarm')#,shading='nearest'
        plt.colorbar(im)
        plt.xticks(numpy.arange(len(xticks)) + 0.5, xticks, size='small', rotation='vertical')
        plt.yticks(numpy.arange(len(xticks)) + 0.5, xticks, size='small')

        ax = plt.gca()
        ax.set_aspect('equal')




# wf1 = tdcs[run].wf(0, apply_filter=True, hilbert=False, tukey=True, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)
# wf2 = tdcs[run].wf(2, apply_filter=True, hilbert=False, tukey=True, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)

# max(numpy.correlate(wf1/numpy.std(wf1),wf2/numpy.std(wf2), mode='full')/len(wf2))