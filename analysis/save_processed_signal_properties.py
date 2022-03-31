#!/usr/bin/env python3
'''
This will load waveforms using the same signal processing used for time delays, and then save the values for certain
signal properties after they have been processed.  
'''
import os
import sys
import h5py
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
#from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from examples.beacon_data_reader import Reader as RawReader #Without sine subtraction

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
    align_method = default_align_method#4#0#4#8
    if len(sys.argv) > 1:
        run = int(sys.argv[1])
    else:
        run = 1703

    datapath = os.environ['BEACON_DATA']

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

    #numpy.array(['eventids','trigger_type','raw_approx_trigger_time','raw_approx_trigger_time_nsecs','trig_time','calibrated_trigtime','p2p','std'])
    #Want to calculate max, min, p2p, std, t_max (time of max value)

    try:
        run = int(run)

        reader = Reader(datapath,run)
        raw_reader = RawReader(datapath,run)
        filename = createFile(raw_reader, check_defaults=True) #Creates an analysis file if one does not exist.  Returns filename to load file.
        filename = createFile(reader, check_defaults=False) 
        if filename is not None:
            with h5py.File(filename, 'a') as file:
                eventids = file['eventids'][...]


                if numpy.isin(run, known_pulser_runs):
                    print('Given run is known to be a pulsing run, including calculations for force triggers.')
                    waveform_index_range = (None, None)
                else:
                    waveform_index_range = info.returnDefaultWaveformIndexRange()
                print('USING WAVEFORM_INDEX_RANGE OF ', str(waveform_index_range))

                print('Total Events: %i'%file.attrs['N'])
                print('[1] Software Events: %i'%sum(file['trigger_type'][...] == 1))
                print('[2] RF Events: %i'%sum(file['trigger_type'][...] == 2))
                print('[3] GPS Events: %i'%sum(file['trigger_type'][...] == 3))

                dsets = list(file.keys()) #Existing datasets

                if not numpy.isin('processed_properties',dsets):
                    file.create_group('processed_properties')
                else:
                    print('processed_properties group already exists in file %s'%filename)

                properties_groups = list(file['processed_properties'].keys()) #Existing datasets

                if not numpy.isin(filter_string,properties_groups):
                    file['processed_properties'].create_group(filter_string)
                else:
                    print('%s group already exists in file %s'%(filter_string,filename))

                properties_dsets = list(file['processed_properties'][filter_string].keys()) #Existing datasets

                file['processed_properties'][filter_string].attrs['final_corr_length'] = final_corr_length

                if crit_freq_low_pass_MHz is not None:
                    file['processed_properties'][filter_string].attrs['crit_freq_low_pass_MHz'] = crit_freq_low_pass_MHz 
                else:
                    file['processed_properties'][filter_string].attrs['crit_freq_low_pass_MHz'] = 0

                if low_pass_filter_order is not None:
                    file['processed_properties'][filter_string].attrs['low_pass_filter_order'] = low_pass_filter_order 
                else:
                    file['processed_properties'][filter_string].attrs['low_pass_filter_order'] = 0

                if crit_freq_high_pass_MHz is not None:
                    file['processed_properties'][filter_string].attrs['crit_freq_high_pass_MHz'] = crit_freq_high_pass_MHz 
                else:
                    file['processed_properties'][filter_string].attrs['crit_freq_high_pass_MHz'] = 0

                if high_pass_filter_order is not None:
                    file['processed_properties'][filter_string].attrs['high_pass_filter_order'] = high_pass_filter_order 
                else:
                    file['processed_properties'][filter_string].attrs['high_pass_filter_order'] = 0

                file['processed_properties'][filter_string].attrs['sine_subtract_min_freq_GHz'] = sine_subtract_min_freq_GHz 
                file['processed_properties'][filter_string].attrs['sine_subtract_max_freq_GHz'] = sine_subtract_max_freq_GHz 
                file['processed_properties'][filter_string].attrs['sine_subtract_percent'] = sine_subtract_percent
                file['processed_properties'][filter_string].attrs['sine_subtract'] = sine_subtract
                file['processed_properties'][filter_string].attrs['waveform_index_range_min'] = min(waveform_index_range)
                file['processed_properties'][filter_string].attrs['waveform_index_range_max'] = max(waveform_index_range)
                
                #eventids with rf trigger
                if numpy.size(eventids) != 0:
                    print('run = ',run)

                #Time Delays
                #01
                output_dimensions = (file.attrs['N'],8)
                #Want to calculate max, min, p2p, std, t_max (time of max value)

                if not numpy.isin('std',properties_dsets):
                    file['processed_properties'][filter_string].create_dataset('std', output_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in processed_properties : std of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('min',properties_dsets):
                    file['processed_properties'][filter_string].create_dataset('min', output_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in processed_properties : min of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('max',properties_dsets):
                    file['processed_properties'][filter_string].create_dataset('max', output_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in processed_properties : max of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('p2p',properties_dsets):
                    file['processed_properties'][filter_string].create_dataset('p2p', output_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in processed_properties : p2p of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('t_max',properties_dsets):
                    file['processed_properties'][filter_string].create_dataset('t_max', output_dimensions, dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in processed_properties : t_max of %s will be overwritten by this analysis script.'%filename)


                tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=plot_filter,apply_phase_response=apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches)
                if sine_subtract:
                    tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                #Run test event and get the times/shape of output.
                waveform_ffts_filtered_corr, upsampled_waveforms = tdc.loadFilteredFFTs(0, channels=[0,1,2,3,4,5,6,7], hilbert=False, load_upsampled_waveforms=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, sine_subtract=True)
                times = tdc.dt_ns_upsampled*numpy.arange(len(upsampled_waveforms[0]))

                #Acutally run over data.
                for eventid in eventids:
                    if eventid%1000 == 0:
                        sys.stdout.write('(%i/%i)\t\t\t\r'%(eventid,len(eventids)))
                    waveform_ffts_filtered_corr, upsampled_waveforms = tdc.loadFilteredFFTs(eventid, channels=[0,1,2,3,4,5,6,7], hilbert=False, load_upsampled_waveforms=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, sine_subtract=True)

                    file['processed_properties'][filter_string]['std'][eventid,:] = numpy.std(upsampled_waveforms, axis=1)
                    min_vals = numpy.min(upsampled_waveforms, axis=1)
                    max_vals = numpy.max(upsampled_waveforms, axis=1)
                    p2p_vals = max_vals - min_vals
                    file['processed_properties'][filter_string]['min'][eventid,:] = min_vals
                    file['processed_properties'][filter_string]['max'][eventid,:] = max_vals
                    file['processed_properties'][filter_string]['p2p'][eventid,:] = p2p_vals
                    file['processed_properties'][filter_string]['t_max'][eventid,:] = times[numpy.argmax(upsampled_waveforms, axis=1)]
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