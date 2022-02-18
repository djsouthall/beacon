#!/usr/bin/env python3
'''
This will make a map, then look at the waveforms aligned at the N top sidelobes.
'''

import sys
import os
import subprocess
import inspect
import h5py
import copy
from pprint import pprint
import pyperclip as pc

import numpy
import scipy
import scipy.signal
import time



#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.flipbook_reader import flipbookToDict
import  beacon.tools.get_plane_tracks as pt
from tools.airplane_traffic_loader import getDataFrames, getFileNamesFromTimestamps

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


'''
'impulsivity_h','impulsivity_v', 'cr_template_search_h', 'cr_template_search_v', 'std_h', 'std_v', 'p2p_h', 'p2p_v', 'snr_h', 'snr_v',\
'time_delay_0subtract1_h','time_delay_0subtract2_h','time_delay_0subtract3_h','time_delay_1subtract2_h','time_delay_1subtract3_h','time_delay_2subtract3_h',\
'time_delay_0subtract1_v','time_delay_0subtract2_v','time_delay_0subtract3_v','time_delay_1subtract2_v','time_delay_1subtract3_v','time_delay_2subtract3_v',
'cw_present','cw_freq_Mhz','cw_linear_magnitude','cw_dbish','theta_best_h','theta_best_v','elevation_best_h','elevation_best_v','phi_best_h','phi_best_v',\
'calibrated_trigtime','triggered_beams','beam_power','hpol_peak_to_sidelobe','vpol_peak_to_sidelobe','hpol_max_possible_map_value','vpol_max_possible_map_value',\
'map_max_time_delay_0subtract1_h','map_max_time_delay_0subtract2_h','map_max_time_delay_0subtract3_h',\
'map_max_time_delay_1subtract2_h','map_max_time_delay_1subtract3_h','map_max_time_delay_2subtract3_h',\
'map_max_time_delay_0subtract1_v','map_max_time_delay_0subtract2_v','map_max_time_delay_0subtract3_v',\
'map_max_time_delay_1subtract2_v','map_max_time_delay_1subtract3_v','map_max_time_delay_2subtract3_v'
'''

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
    
    if len(sys.argv) >= 3:
        runs = [int(sys.argv[1])]
        eventids = [int(sys.argv[2])]
        good_dict = {}
        good_dict[runs[0]] = eventids
        single_event = True
    elif False:
        good_dict = sorted_dict['no-obvious-airplane']['eventids_dict']
        # maybe_dict = sorted_dict['maybe']['eventids_dict']
        # bad_dict = sorted_dict['bad']['eventids_dict']
        runs = list(good_dict.keys())
        single_event = False
    elif True:
        # r5927e105291
        good_dict = {5927:[105291]}
        runs = numpy.array([list(good_dict.keys())[0]])
        single_event = True
    elif False:
        good_dict = {5805:[11079]}
        runs = numpy.array([list(good_dict.keys())[0]])
        single_event = True
    else:
        good_dict = {5903:[86227]}
        runs = numpy.array([5903])
        single_event = True

    print("Preparing dataSlicer")

    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)


    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                    n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)

    ds.prepareCorrelator()

    elevation_best_choice = ds.getDataFromParam(good_dict, 'elevation_best_choice')
    phi_best_choice = ds.getDataFromParam(good_dict, 'phi_best_choice')

    theta_cut = None
    zenith_cut_ENU = None
    zenith_cut_array_plane = [0,91]
    plot_focused_map = False #This makes a map using the waveforms delayed and summed based on time delays for each beam.  It basically makes a map assuming each beam is true.  It is not particularly useful as far as I can tell.

    n_peaks = 10

    for run in runs:
        run_index = numpy.where(numpy.isin(ds.runs, run))[0][0]
        if ds.cor.reader.run != run:
            ds.cor.setReader(ds.data_slicers[run_index].reader, verbose=False)

        eventids = good_dict[run]
        event_times = ds.cor.getEventTimes()[eventids]

        for event_index, eventid in enumerate(eventids):
            best_elevation = float(elevation_best_choice[run][numpy.array(good_dict[run]) == eventid][0])
            best_phi = float(phi_best_choice[run][numpy.array(good_dict[run]) == eventid][0])

            ds.cor.reader.setEntry(eventid)

            map_values, fig, ax = ds.cor.map(eventid, 'hpol', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=None, plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=None, verbose=False, mollweide=False, zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, center_dir='E', circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=False, circle_map_max=False, override_to_time_window=(None,None))
            ax.set_xlim(-90,90)
            ax.set_ylim(-40,90)
            
            # linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = ds.cor.mapMax(map_values, max_method=0, verbose=False, zenith_cut_ENU=[0,80], zenith_cut_array_plane=[0,90], pol='hpol', return_peak_to_sidelobe=False, theta_cut=None)

            if theta_cut is None:
                theta_cut = ds.cor.generateThetaCutMask('hpol', shape=numpy.shape(map_values),zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane)

            masked_map_values = numpy.ma.array(map_values,mask=~theta_cut) #This way the values not in the range are not included in calculations but the dimensions of the map stay the same.#masked_map_values = numpy.ma.array(map_values.copy(),mask=~theta_cut) #This way the values not in the range are not included in calculations but the dimensions of the map stay the same.

            # if max_method == 0:
            #     row_index, column_index = numpy.unravel_index(masked_map_values.argmax(),numpy.shape(masked_map_values))

            # elif max_method == 1:
            #     #Calculates sum of each point plus surrounding four points to get max.
            #     rounded_corr_values = (masked_map_values + numpy.roll(masked_map_values,1,axis=0) + numpy.roll(masked_map_values,-1,axis=0) + numpy.roll(masked_map_values,1,axis=1) + numpy.roll(masked_map_values,-1,axis=1))/5.0
            #     row_index, column_index = numpy.unravel_index(rounded_corr_values.argmax(),numpy.shape(rounded_corr_values))

            linear_max_index = masked_map_values.argmax()

            threshold = masked_map_values.max()*0.65

            blob_labels, num_blobs = scipy.ndimage.label(masked_map_values >= threshold)
            blob_labels = numpy.ma.masked_array(blob_labels,mask=~theta_cut)



            if num_blobs > 1:
                
                blob_max_indices = numpy.zeros(num_blobs)
                blob_max_values = numpy.zeros(num_blobs)
                for blob_label in range(num_blobs):
                    blob_max_indices[blob_label] = numpy.ma.array(map_values,mask=numpy.logical_or(~theta_cut , blob_labels.data != blob_label)).argmax()
                    blob_max_values[blob_label] = numpy.ma.array(map_values,mask=numpy.logical_or(~theta_cut , blob_labels.data != blob_label)).max()

                blob_max_values[numpy.isnan(blob_max_values)] = 0

                selected_blobs = numpy.argsort(blob_max_values)[::-1][0:min(n_peaks,num_blobs)]#numpy.arange(num_blobs)[~numpy.isnan(blob_max_values)][numpy.argsort(blob_max_values)[~numpy.isnan(blob_max_values)]][::-1][0:6]
                selected_blob_maxes = blob_max_values[selected_blobs]

                if plot_focused_map == True:
                    focused_map_max = {}

                blob_values = []
                blob_thetas = []
                blob_phis = []
                for blob_index, blob_label in enumerate(selected_blobs):
                    peak_masked_map_values = numpy.ma.array(map_values,mask=numpy.logical_or(~theta_cut , blob_labels != blob_label)) #Select values in current peak.

                    linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = ds.cor.mapMax(map_values, max_method=0, verbose=False, zenith_cut_ENU=None, zenith_cut_array_plane=None, pol='hpol', return_peak_to_sidelobe=False, theta_cut=~peak_masked_map_values.mask)
                    
                    print('Blob %i'%blob_index)
                    theta_best  = ds.cor.mesh_zenith_deg.flat[linear_max_index]
                    print('theta_best = %0.2f deg'%theta_best)
                    phi_best    = ds.cor.mesh_azimuth_deg.flat[linear_max_index]
                    print('phi_best = %0.2f deg'%phi_best)

                    ax.scatter(phi_best,90.0 - theta_best,label='Blob Index = %i'%(blob_index))
                    ax.legend(loc='lower left',fontsize=10)

                    fig, all_ax = plt.subplots(6,2)
                    fig.canvas.set_window_title('r%ie%i-peak%s'%(run, eventid, blob_index))

                    for pol in ['vpol', 'hpol']:
                        if pol is not None:
                            if pol == 'vpol':
                                channels = numpy.array([1,3,5,7])
                                waveforms = ds.cor.wf(eventid, channels, div_std=False, hilbert=False, apply_filter=ds.cor.apply_filter, tukey=ds.cor.apply_tukey, sine_subtract=ds.cor.apply_sine_subtract)

                                t_best_0subtract1 = ds.cor.t_vpol_0subtract1.flat[linear_max_index]
                                t_best_0subtract2 = ds.cor.t_vpol_0subtract2.flat[linear_max_index]
                                t_best_0subtract3 = ds.cor.t_vpol_0subtract3.flat[linear_max_index]
                                t_best_1subtract2 = ds.cor.t_vpol_1subtract2.flat[linear_max_index]
                                t_best_1subtract3 = ds.cor.t_vpol_1subtract3.flat[linear_max_index]
                                t_best_2subtract3 = ds.cor.t_vpol_2subtract3.flat[linear_max_index]
                            else:
                                #Default to hpol times.
                                channels = numpy.array([0,2,4,6])
                                waveforms = ds.cor.wf(eventid, channels, div_std=False, hilbert=False, apply_filter=ds.cor.apply_filter, tukey=ds.cor.apply_tukey, sine_subtract=ds.cor.apply_sine_subtract)

                                t_best_0subtract1 = ds.cor.t_hpol_0subtract1.flat[linear_max_index]
                                t_best_0subtract2 = ds.cor.t_hpol_0subtract2.flat[linear_max_index]
                                t_best_0subtract3 = ds.cor.t_hpol_0subtract3.flat[linear_max_index]
                                t_best_1subtract2 = ds.cor.t_hpol_1subtract2.flat[linear_max_index]
                                t_best_1subtract3 = ds.cor.t_hpol_1subtract3.flat[linear_max_index]
                                t_best_2subtract3 = ds.cor.t_hpol_2subtract3.flat[linear_max_index]


                        else:
                            channels = numpy.array([0,2,4,6])
                            waveforms = ds.cor.wf(eventid, channels, div_std=False, hilbert=False, apply_filter=ds.cor.apply_filter, tukey=ds.cor.apply_tukey, sine_subtract=ds.cor.apply_sine_subtract)

                            t_best_0subtract1 = ds.cor.t_hpol_0subtract1.flat[linear_max_index]
                            t_best_0subtract2 = ds.cor.t_hpol_0subtract2.flat[linear_max_index]
                            t_best_0subtract3 = ds.cor.t_hpol_0subtract3.flat[linear_max_index]
                            t_best_1subtract2 = ds.cor.t_hpol_1subtract2.flat[linear_max_index]
                            t_best_1subtract3 = ds.cor.t_hpol_1subtract3.flat[linear_max_index]
                            t_best_2subtract3 = ds.cor.t_hpol_2subtract3.flat[linear_max_index]

                        time_delay_dict = {}
                        time_delay_dict['t_best_0subtract1'] = t_best_0subtract1
                        time_delay_dict['t_best_0subtract2'] = t_best_0subtract2
                        time_delay_dict['t_best_0subtract3'] = t_best_0subtract3
                        time_delay_dict['t_best_1subtract2'] = t_best_1subtract2
                        time_delay_dict['t_best_1subtract3'] = t_best_1subtract3
                        time_delay_dict['t_best_2subtract3'] = t_best_2subtract3

                        time_delays = [t_best_0subtract1,t_best_0subtract2,t_best_0subtract3,t_best_1subtract2,t_best_1subtract3,t_best_2subtract3]

                        blob_value = 0
                        for pair_index, pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
                            i, j = pair

                            w_i = waveforms[i]/max(waveforms[i])
                            w_j = waveforms[j]/max(waveforms[j])

                            plt.sca(all_ax[pair_index,int(pol == 'vpol')])

                            plt.plot(ds.cor.times_resampled,w_i,label='Ch%i'%channels[i],c=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
                            plt.plot(ds.cor.times_resampled + time_delays[pair_index], w_j,label='Ch%i, shifted %0.2f ns'%(channels[j], time_delays[pair_index]),c=plt.rcParams['axes.prop_cycle'].by_key()['color'][j])

                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                            #plt.ylabel('Normalized adu\n%i shifted to %i\n%0.2f ns'%(j,i,time_delays[pair_index]))
                            plt.ylabel('%i shifted to %i\n%0.2f ns'%(j,i,time_delays[pair_index]))
                            # plt.legend()
                            xmin = 100
                            xmax = 900
                            plt.xlim(xmin,xmax)
                            if pair_index == 5:
                                plt.xlabel('Time (ns)')
                            cut_i = numpy.logical_and(ds.cor.times_resampled >= xmin , ds.cor.times_resampled < xmax)
                            cut_j = numpy.arange(sum(cut_i)) + numpy.where(numpy.logical_and(ds.cor.times_resampled + time_delays[pair_index] >= xmin , ds.cor.times_resampled + time_delays[pair_index] < xmax))[0].min()
                            
                            blob_value += numpy.sum(numpy.abs(w_i[cut_i] + w_j[cut_j]))

                        blob_values.append(blob_value)
                        blob_thetas.append(theta_best)
                        blob_phis.append(phi_best)

                        plt.suptitle('Blob %i, El = %0.2f, Az = %0.2f\nBlob Value = %.2f'%(blob_index, 90.0 - theta_best, phi_best, blob_value))
                        print('blob_value = %0.2f'%blob_value)

                        cut_i = numpy.logical_and(ds.cor.times_resampled >= xmin , ds.cor.times_resampled < xmax)
                        summed_waveforms = numpy.zeros((4,len(cut_i)))


                        for i in range(4):
                            for j in range(4):
                                if i == j:
                                    cut_j = cut_i
                                else:
                                    if i < j:
                                        t_best_isubtractj = time_delay_dict['t_best_%isubtract%i'%(i,j)]
                                    else:
                                        t_best_isubtractj = -time_delay_dict['t_best_%isubtract%i'%(j,i)]
                                    cut_j = numpy.arange(sum(cut_i)) + numpy.where(numpy.logical_and(ds.cor.times_resampled + t_best_isubtractj >= xmin , ds.cor.times_resampled + t_best_isubtractj < xmax))[0].min()

                                summed_waveforms[i][cut_i] += waveforms[j][cut_j]/4

                        map_wf = numpy.zeros_like(summed_waveforms)
                        for i in range(4):
                            map_wf[i] = summed_waveforms[i]/numpy.std(summed_waveforms[i])

                        if plot_focused_map == True:
                            fig_ds = plt.figure()
                            fig_ds.canvas.set_window_title('Delay Sum')
                            map_ax = fig_ds.add_subplot(1,2,2)
                            match_this_format = ds.cor.wf(eventid, numpy.array([0,2,4,6]),div_std=True,hilbert=False,apply_filter=ds.cor.apply_filter,tukey=ds.cor.apply_tukey, sine_subtract=ds.cor.apply_sine_subtract)
                            if numpy.all(numpy.shape(map_wf) == numpy.shape(match_this_format)):
                                map_values2, map_fig, map_ax = ds.cor.map(eventid, pol, include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=map_ax, plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=map_wf, verbose=False, mollweide=False, zenith_cut_ENU=zenith_cut_ENU, zenith_cut_array_plane=zenith_cut_array_plane, center_dir='E', circle_zenith=None, circle_az=None, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=False, circle_map_max=False, override_to_time_window=(None,None))
                                map_ax.set_xlim(-90,90)
                                map_ax.set_ylim(-40,90)

                            for i, wf in enumerate(summed_waveforms):
                                if plot_focused_map == True:
                                    ax_wf = plt.subplot(4, 2, 2*i+1) 
                                else:
                                    ax_wf = plt.subplot(4, 1, i+1) 
                                ax_wf.plot(ds.cor.times_resampled,waveforms[i],label='Raw Ch%i'%channels[i],c='k',linestyle='--')
                                ax_wf.plot(ds.cor.times_resampled,wf,label='Averaged Ch%i'%channels[i],c=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
                                ax_wf.minorticks_on()
                                ax_wf.grid(b=True, which='major', color='k', linestyle='-')
                                ax_wf.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                                ax_wf.set_ylabel('adu waveforms averaged\naligned to ch %i'%(i))

                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = ds.cor.mapMax(map_values, max_method=0, verbose=False, zenith_cut_ENU=None, zenith_cut_array_plane=None, pol=pol, return_peak_to_sidelobe=False, theta_cut=~peak_masked_map_values.mask)
                            focused_map_max[blob_index] = {}
                            focused_map_max[blob_index]['max'] = map_values2.flat[linear_max_index]
                            focused_map_max[blob_index]['theta'] = theta_best
                            focused_map_max[blob_index]['phi'] = phi_best

                if plot_focused_map == True:
                    pprint(focused_map_max)

                print('Blob Values')
                for i, bv in enumerate(blob_values):
                    print('blob: ', i, '\tbv: ', bv, '\tphi: ', blob_phis[i],'\tel: ', 90.0 - blob_thetas[i])

    if single_event:
        #Ready the potential next command if inspecting the event.
        pc.copy(r'%run ' + os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'tools','event_info.py') + ' ' + str(run) + ' ' + str(eventid) + ' ' + 'True')

    # if single_event:
    #     subprocess.run([os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'tools','event_info.py'), str(run), str(eventid), 'True'])
    # else:
    #     import pdb; pdb.set_trace()
