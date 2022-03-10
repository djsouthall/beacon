#!/usr/bin/env python3
'''
This is meant to calculate the time delays for each event and then save them to file.
'''
import os
import sys
import h5py
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
#from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
from tools.fftmath import TimeDelayCalculator
from tools.data_handler import createFile

import matplotlib.pyplot as plt
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
import itertools
plt.ion()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 16})

if __name__=="__main__":
    default_align_method = 0
    if len(sys.argv) > 1:
        run = int(sys.argv[1])
        if len(sys.argv) > 2:
            align_method = int(sys.argv[2])
        else:
            align_method = default_align_method#4#0#4#8
        print('Using align_method = %i'%align_method)
    else:
        run = 1703
        align_method = default_align_method#4#0#4#8

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

    filter_string = ''

    if crit_freq_low_pass_MHz is None:
        filter_string += 'LPf_%s-'%('None')
    else:
        filter_string += 'LPf_%0.1f-'%(crit_freq_low_pass_MHz)

    if low_pass_filter_order is None:
        filter_string += 'LPo_%s-'%('None')
    else:
        filter_string += 'LPo_%i-'%(low_pass_filter_order)

    if crit_freq_high_pass_MHz is None:
        filter_string += 'HPf_%s-'%('None')
    else:
        filter_string += 'HPf_%0.1f-'%(crit_freq_high_pass_MHz)

    if high_pass_filter_order is None:
        filter_string += 'HPo_%s-'%('None')
    else:
        filter_string += 'HPo_%i-'%(high_pass_filter_order)

    if apply_phase_response is None:
        filter_string += 'Phase_%s-'%('None')
    else:
        filter_string += 'Phase_%i-'%(apply_phase_response)

    if hilbert is None:
        filter_string += 'Hilb_%s-'%('None')
    else:
        filter_string += 'Hilb_%i-'%(hilbert)

    if final_corr_length is None:
        filter_string += 'corlen_%s-'%('None')
    else:
        filter_string += 'corlen_%i-'%(final_corr_length)

    if align_method is None:
        filter_string += 'align_%s-'%('None')
    else:
        filter_string += 'align_%i-'%(align_method)

    if shorten_signals is None:
        filter_string += 'shortensignals-%s-'%('None')
    else:
        filter_string += 'shortensignals-%i-'%(shorten_signals)
    if shorten_thresh is None:
        filter_string += 'shortenthresh-%s-'%('None')
    else:
        filter_string += 'shortenthresh-%0.2f-'%(shorten_thresh)
    if shorten_delay is None:
        filter_string += 'shortendelay-%s-'%('None')
    else:
        filter_string += 'shortendelay-%0.2f-'%(shorten_delay)
    if shorten_length is None:
        filter_string += 'shortenlength-%s-'%('None')
    else:
        filter_string += 'shortenlength-%0.2f-'%(shorten_length)

    filter_string += 'sinesubtract_%i'%(int(sine_subtract))

    print(filter_string)

    plot_filter = False
    plot_multiple = False


    hpol_pairs  = numpy.array(list(itertools.combinations((0,2,4,6), 2)))
    vpol_pairs  = numpy.array(list(itertools.combinations((1,3,5,7), 2)))
    pairs       = numpy.vstack((hpol_pairs,vpol_pairs)) 

    #known_pulser_runs = numpy.array([5630, 5631, 5632, 5638, 5639, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5656, 5657, 5659, 5660], dtype=int)
    known_pulser_ids = info.load2021PulserEventids()
    known_pulser_runs = numpy.unique(numpy.concatenate([numpy.append(numpy.unique(known_pulser_ids[site]['hpol']['run']),numpy.unique(known_pulser_ids[site]['vpol']['run'])) for site in list(known_pulser_ids.keys())]))


    try:
        run = int(run)

        reader = Reader(datapath,run)
        filename = createFile(reader, check_defaults=True) #Creates an analysis file if one does not exist.  Returns filename to load file.
        if filename is not None:
            with h5py.File(filename, 'a') as file:
                eventids = file['eventids'][...]


                if numpy.isin(run, known_pulser_runs):
                    print('Given run is known to be a pulsing run, including calculations for force triggers.')
                    rf_cut = numpy.ones_like(file['trigger_type'][...] == 2)
                    waveform_index_range = (None, None)
                else:
                    waveform_index_range = info.returnDefaultWaveformIndexRange()
                    rf_cut = file['trigger_type'][...] == 2
                print('USING WAVEFORM_INDEX_RANGE OF ', str(waveform_index_range))



                # print('TEMPORARY HARDCODE') #debugging an error that occurred in run 1663 due to a zero valued waveform
                # rf_cut[numpy.cumsum(file['trigger_type'][...] == 2) < 10000] = False #TEMPORARY

                print('Total Events: %i'%file.attrs['N'])
                print('[1] Software Events: %i'%sum(file['trigger_type'][...] == 1))
                print('[2] RF Events: %i'%sum(file['trigger_type'][...] == 2))
                print('[3] GPS Events: %i'%sum(file['trigger_type'][...] == 3))

                dsets = list(file.keys()) #Existing datasets

                if not numpy.isin('time_delays',dsets):
                    file.create_group('time_delays')
                else:
                    print('time_delays group already exists in file %s'%filename)

                time_delay_groups = list(file['time_delays'].keys()) #Existing datasets

                if not numpy.isin(filter_string,time_delay_groups):
                    file['time_delays'].create_group(filter_string)
                else:
                    print('%s group already exists in file %s'%(filter_string,filename))

                time_delay_dsets = list(file['time_delays'][filter_string].keys()) #Existing datasets

                file['time_delays'][filter_string].attrs['final_corr_length'] = final_corr_length

                if crit_freq_low_pass_MHz is not None:
                    file['time_delays'][filter_string].attrs['crit_freq_low_pass_MHz'] = crit_freq_low_pass_MHz 
                else:
                    file['time_delays'][filter_string].attrs['crit_freq_low_pass_MHz'] = 0

                if low_pass_filter_order is not None:
                    file['time_delays'][filter_string].attrs['low_pass_filter_order'] = low_pass_filter_order 
                else:
                    file['time_delays'][filter_string].attrs['low_pass_filter_order'] = 0

                if crit_freq_high_pass_MHz is not None:
                    file['time_delays'][filter_string].attrs['crit_freq_high_pass_MHz'] = crit_freq_high_pass_MHz 
                else:
                    file['time_delays'][filter_string].attrs['crit_freq_high_pass_MHz'] = 0

                if high_pass_filter_order is not None:
                    file['time_delays'][filter_string].attrs['high_pass_filter_order'] = high_pass_filter_order 
                else:
                    file['time_delays'][filter_string].attrs['high_pass_filter_order'] = 0

                file['time_delays'][filter_string].attrs['sine_subtract_min_freq_GHz'] = sine_subtract_min_freq_GHz 
                file['time_delays'][filter_string].attrs['sine_subtract_max_freq_GHz'] = sine_subtract_max_freq_GHz 
                file['time_delays'][filter_string].attrs['sine_subtract_percent'] = sine_subtract_percent
                file['time_delays'][filter_string].attrs['sine_subtract'] = sine_subtract
                file['time_delays'][filter_string].attrs['waveform_index_range_min'] = min(waveform_index_range)
                file['time_delays'][filter_string].attrs['waveform_index_range_max'] = max(waveform_index_range)
                
                #eventids with rf trigger
                if numpy.size(eventids) != 0:
                    print('run = ',run)

                #Time Delays
                #01
                file.attrs['align_method_13_n'] = align_method_13_n
                if align_method == 13:
                    time_delay_dimensions = (file.attrs['N'],file.attrs['align_method_13_n'])
                else:
                    time_delay_dimensions = (file.attrs['N'],)

                if not numpy.isin('hpol_t_0subtract1',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_t_0subtract1', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_t_0subtract1 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_t_0subtract1',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_t_0subtract1', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_t_0subtract1 of %s will be overwritten by this analysis script.'%filename)



                #02
                if not numpy.isin('hpol_t_0subtract2',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_t_0subtract2', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_t_0subtract2 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_t_0subtract2',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_t_0subtract2', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_t_0subtract2 of %s will be overwritten by this analysis script.'%filename)



                #03
                if not numpy.isin('hpol_t_0subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_t_0subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_t_0subtract3 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_t_0subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_t_0subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_t_0subtract3 of %s will be overwritten by this analysis script.'%filename)



                #12
                if not numpy.isin('hpol_t_1subtract2',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_t_1subtract2', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_t_1subtract2 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_t_1subtract2',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_t_1subtract2', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_t_1subtract2 of %s will be overwritten by this analysis script.'%filename)



                #13
                if not numpy.isin('hpol_t_1subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_t_1subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_t_1subtract3 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_t_1subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_t_1subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_t_1subtract3 of %s will be overwritten by this analysis script.'%filename)



                #23
                if not numpy.isin('hpol_t_2subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_t_2subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_t_2subtract3 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_t_2subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_t_2subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_t_2subtract3 of %s will be overwritten by this analysis script.'%filename)



                #Correlation values.
                #01
                if not numpy.isin('hpol_max_corr_0subtract1',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_max_corr_0subtract1', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_max_corr_0subtract1 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_max_corr_0subtract1',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_max_corr_0subtract1', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_max_corr_0subtract1 of %s will be overwritten by this analysis script.'%filename)
                


                #02
                if not numpy.isin('hpol_max_corr_0subtract2',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_max_corr_0subtract2', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_max_corr_0subtract2 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_max_corr_0subtract2',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_max_corr_0subtract2', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_max_corr_0subtract2 of %s will be overwritten by this analysis script.'%filename)



                #03
                if not numpy.isin('hpol_max_corr_0subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_max_corr_0subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_max_corr_0subtract3 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_max_corr_0subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_max_corr_0subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_max_corr_0subtract3 of %s will be overwritten by this analysis script.'%filename)



                #12
                if not numpy.isin('hpol_max_corr_1subtract2',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_max_corr_1subtract2', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_max_corr_1subtract2 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_max_corr_1subtract2',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_max_corr_1subtract2', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_max_corr_1subtract2 of %s will be overwritten by this analysis script.'%filename)



                #13
                if not numpy.isin('hpol_max_corr_1subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_max_corr_1subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_max_corr_1subtract3 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_max_corr_1subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_max_corr_1subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_max_corr_1subtract3 of %s will be overwritten by this analysis script.'%filename)



                #23
                if not numpy.isin('hpol_max_corr_2subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('hpol_max_corr_2subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in hpol_max_corr_2subtract3 of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('vpol_max_corr_2subtract3',time_delay_dsets):
                    file['time_delays'][filter_string].create_dataset('vpol_max_corr_2subtract3', time_delay_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in vpol_max_corr_2subtract3 of %s will be overwritten by this analysis script.'%filename)


                if sum(rf_cut) > 0:
                    #rf_cut = numpy.multiply(rf_cut ,numpy.cumsum(rf_cut) < 10) #limits it to the first 100 Trues for testing.
                    tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=plot_filter,apply_phase_response=apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches)
                    if sine_subtract:
                        tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids[rf_cut],align_method=align_method,hilbert=hilbert,plot=plot_multiple,hpol_cut=None,vpol_cut=None,shorten_signals=shorten_signals,shorten_thresh=shorten_thresh,shorten_delay=shorten_delay,shorten_length=shorten_length,sine_subtract=sine_subtract)

                    if align_method == 13:
                        rf_cut = numpy.where(rf_cut)[0]

                    for pair_index, pair in enumerate(pairs):
                        if align_method != 13:
                            #The cut is different because the dimensions are different.
                            if numpy.all(pair == numpy.array([0,2])):
                                file['time_delays'][filter_string]['hpol_t_0subtract1'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_0subtract1'][rf_cut] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([0,4])):
                                file['time_delays'][filter_string]['hpol_t_0subtract2'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_0subtract2'][rf_cut] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([0,6])):
                                file['time_delays'][filter_string]['hpol_t_0subtract3'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_0subtract3'][rf_cut] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([2,4])):
                                file['time_delays'][filter_string]['hpol_t_1subtract2'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_1subtract2'][rf_cut] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([2,6])):
                                file['time_delays'][filter_string]['hpol_t_1subtract3'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_1subtract3'][rf_cut] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([4,6])):
                                file['time_delays'][filter_string]['hpol_t_2subtract3'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_2subtract3'][rf_cut] = corrs[pair_index]

                            elif numpy.all(pair == numpy.array([1,3])):
                                file['time_delays'][filter_string]['vpol_t_0subtract1'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_0subtract1'][rf_cut] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([1,5])):
                                file['time_delays'][filter_string]['vpol_t_0subtract2'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_0subtract2'][rf_cut] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([1,7])):
                                file['time_delays'][filter_string]['vpol_t_0subtract3'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_0subtract3'][rf_cut] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([3,5])):
                                file['time_delays'][filter_string]['vpol_t_1subtract2'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_1subtract2'][rf_cut] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([3,7])):
                                file['time_delays'][filter_string]['vpol_t_1subtract3'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_1subtract3'][rf_cut] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([5,7])):
                                file['time_delays'][filter_string]['vpol_t_2subtract3'][rf_cut] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_2subtract3'][rf_cut] = corrs[pair_index]
                        else:
                            if numpy.all(pair == numpy.array([0,2])):
                                file['time_delays'][filter_string]['hpol_t_0subtract1'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_0subtract1'][rf_cut,:] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([0,4])):
                                file['time_delays'][filter_string]['hpol_t_0subtract2'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_0subtract2'][rf_cut,:] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([0,6])):
                                file['time_delays'][filter_string]['hpol_t_0subtract3'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_0subtract3'][rf_cut,:] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([2,4])):
                                file['time_delays'][filter_string]['hpol_t_1subtract2'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_1subtract2'][rf_cut,:] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([2,6])):
                                file['time_delays'][filter_string]['hpol_t_1subtract3'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_1subtract3'][rf_cut,:] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([4,6])):
                                file['time_delays'][filter_string]['hpol_t_2subtract3'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['hpol_max_corr_2subtract3'][rf_cut,:] = corrs[pair_index]

                            elif numpy.all(pair == numpy.array([1,3])):
                                file['time_delays'][filter_string]['vpol_t_0subtract1'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_0subtract1'][rf_cut,:] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([1,5])):
                                file['time_delays'][filter_string]['vpol_t_0subtract2'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_0subtract2'][rf_cut,:] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([1,7])):
                                file['time_delays'][filter_string]['vpol_t_0subtract3'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_0subtract3'][rf_cut,:] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([3,5])):
                                file['time_delays'][filter_string]['vpol_t_1subtract2'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_1subtract2'][rf_cut,:] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([3,7])):
                                file['time_delays'][filter_string]['vpol_t_1subtract3'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_1subtract3'][rf_cut,:] = corrs[pair_index]
                            elif numpy.all(pair == numpy.array([5,7])):
                                file['time_delays'][filter_string]['vpol_t_2subtract3'][rf_cut,:] = time_shifts[pair_index]
                                file['time_delays'][filter_string]['vpol_max_corr_2subtract3'][rf_cut,:] = corrs[pair_index]
                else:
                    print('No RF signals, skipping calculation.')
                file.close()
        else:
            print('filename is None, indicating empty tree.  Skipping run %i'%run)
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)

    sys.exit(0)