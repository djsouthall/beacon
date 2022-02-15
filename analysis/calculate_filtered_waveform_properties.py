#!/usr/bin/env python3
'''
This will process the waveforms in the same way as for impulsivity etc, but will then calculate the signal properties
of: min, max, p2p, std
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
    debug = True

    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 5732

    datapath = os.environ['BEACON_DATA']
    crit_freq_low_pass_MHz = 80
    low_pass_filter_order = 14

    crit_freq_high_pass_MHz = 20
    high_pass_filter_order = 4
    
    apply_phase_response = True
    plot_filter=False
    hilbert=False
    plot_multiple = False
    final_corr_length = 2**17
    align_method = 0
    max_method = 0
    impulsivity_window = 400

    notch_tv = True
    misc_notches = True
    # , notch_tv=notch_tv, misc_notches=misc_notches

    hpol_pairs  = numpy.array(list(itertools.combinations((0,2,4,6), 2)))
    vpol_pairs  = numpy.array(list(itertools.combinations((1,3,5,7), 2)))
    pairs       = numpy.vstack((hpol_pairs,vpol_pairs)) 

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.00
    sine_subtract_max_freq_GHz = 0.25
    sine_subtract_percent = 0.03

    try:
        run = int(run)

        reader = Reader(datapath,run)
        try:
            print(reader.status())
        except Exception as e:
            print('Status Tree not present.  Returning Error.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(1)

        filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.

        known_pulser_ids = info.load2021PulserEventids()
        known_pulser_runs = numpy.unique(numpy.concatenate([numpy.append(numpy.unique(known_pulser_ids[site]['hpol']['run']),numpy.unique(known_pulser_ids[site]['vpol']['run'])) for site in list(known_pulser_ids.keys())]))
        if numpy.isin(run, known_pulser_runs):
            waveform_index_range = (None, None)
        else:
            waveform_index_range = info.returnDefaultWaveformIndexRange()

        print('USING WAVEFORM_INDEX_RANGE OF ', str(waveform_index_range))

        if filename is not None:
            tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=plot_filter,apply_phase_response=apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches)
            if sine_subtract:
                tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)


            if debug == True:
                plt.figure()

                tdc.setEntry(1000)
                for channel in range(8):
                    wf = tdc.wf(channel, apply_filter=True, hilbert=False, tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)
                    plt.plot(tdc.t(), wf)

            else:
                with h5py.File(filename, 'a') as file:
                    eventids = file['eventids'][...]
                    dsets = list(file.keys()) #Existing datasets

                    for key in ['filtered_min', 'filtered_max', 'filtered_p2p', 'filtered_std']:
                        if not numpy.isin(key,dsets):
                            file.create_dataset(key, (file.attrs['N'],8), dtype=numpy.float64, compression='gzip', compression_opts=9, shuffle=True)
                        else:
                            print('%s group already exists in file %s'%(key,filename))

                        file[key].attrs['waveform_index_range_min'] = min(waveform_index_range)
                        file[key].attrs['waveform_index_range_max'] = max(waveform_index_range)
                        file[key].attrs['sine_subtract_min_freq_GHz'] = sine_subtract_min_freq_GHz 
                        file[key].attrs['sine_subtract_max_freq_GHz'] = sine_subtract_max_freq_GHz 
                        file[key].attrs['sine_subtract_percent'] = sine_subtract_percent
                        file[key].attrs['sine_subtract'] = sine_subtract

                    channels = numpy.arange(8)
                    for eventid in eventids: 
                        if eventid%500 == 0:
                            sys.stdout.write('(%i/%i)\r'%(eventid,len(eventids)))
                            sys.stdout.flush()
                        try:
                            tdc.setEntry(eventid)
                            for channel in channels:
                                wf = tdc.wf(channel, apply_filter=True, hilbert=False, tukey=None, sine_subtract=True, return_sine_subtract_info=False, ss_first=True, attempt_raw_reader=False)
                                
                                file['filtered_min'][eventid,channel] = numpy.min(wf)
                                file['filtered_max'][eventid,channel] = numpy.max(wf)
                                file['filtered_p2p'][eventid,channel] = max_values[channel] - min_values[channel]
                                file['filtered_std'][eventid,channel] = numpy.std(wf)

                    except Exception as e:
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                file.close()
        else:
            print('filename is None, indicating empty tree.  Skipping run %i'%run)
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

