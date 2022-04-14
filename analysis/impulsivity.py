#!/usr/bin/env python3
'''
This is meant to calculate the impulsivity for each event and then save them to file.
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
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 1701

    #TODO
    #Want this to save impulsivity data calculated with and without sine subtract.  Could be a good handle on whether the signal is contaminated or just purely CW.

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
    impulsivity_window = 400 #ns

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
            with h5py.File(filename, 'a') as file:
                eventids = file['eventids'][...]
                dsets = list(file.keys()) #Existing datasets

                if not numpy.isin('time_delays',dsets):
                    print('time delays not present to perform impulsivity in run %i'%run)
                    file.close()
                else:

                    if not numpy.isin('impulsivity',dsets):
                        file.create_group('impulsivity')
                    else:
                        print('impulsivity group already exists in file %s'%filename)

                    time_delays_dsets = list(file['time_delays'].keys())
                    impulsivity_dsets = list(file['impulsivity'].keys())

                    for tdset in time_delays_dsets:#["LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_262144-align_8"]:#
                        align_method = int(tdset.split('align_')[1].split('-')[0])
                        if align_method == 13:
                            print('Similarity currently not designed to work with align method 13, which results in multiple alignment times.')
                            print('Skipping %s'%tdset)
                            continue
                        if not numpy.isin(tdset,impulsivity_dsets):
                            file['impulsivity'].create_group(tdset)
                        else:
                            print('impulsivity["%s"] group already exists in file %s'%(tdset,filename))
                        idsets = list(file['impulsivity'][tdset].keys())

                        all_time_delays = numpy.vstack((file['time_delays'][tdset]['%s_t_%isubtract%i'%('hpol',0,1)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%('hpol',0,2)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%('hpol',0,3)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%('hpol',1,2)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%('hpol',1,3)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%('hpol',2,3)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%('vpol',0,1)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%('vpol',0,2)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%('vpol',0,3)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%('vpol',1,2)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%('vpol',1,3)][...],file['time_delays'][tdset]['%s_t_%isubtract%i'%('vpol',2,3)][...])).T

                        #eventids with rf trigger
                        if numpy.size(eventids) != 0:
                            print('run = ',run)

                        file.attrs['impulsivity_window_ns'] = impulsivity_window
                        file['impulsivity'][tdset].attrs['waveform_index_range_min'] = min(waveform_index_range)
                        file['impulsivity'][tdset].attrs['waveform_index_range_max'] = max(waveform_index_range)
                        file['impulsivity'][tdset].attrs['sine_subtract_min_freq_GHz'] = sine_subtract_min_freq_GHz 
                        file['impulsivity'][tdset].attrs['sine_subtract_max_freq_GHz'] = sine_subtract_max_freq_GHz 
                        file['impulsivity'][tdset].attrs['sine_subtract_percent'] = sine_subtract_percent
                        file['impulsivity'][tdset].attrs['sine_subtract'] = sine_subtract



                        if not numpy.isin('hpol',idsets):
                            file['impulsivity'][tdset].create_dataset('hpol', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                        else:
                            print('Values in hpol of %s will be overwritten by this analysis script.'%filename)

                        if not numpy.isin('vpol',idsets):
                            file['impulsivity'][tdset].create_dataset('vpol', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                        else:
                            print('Values in vpol of %s will be overwritten by this analysis script.'%filename)

                        #Add two functions to TimeDelayCalculator: calculateImpulsivityFromDelay and calculateImpulsivityFromEventid, these can then be use below to get the impulsivity.

                        tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=plot_filter,apply_phase_response=apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches)
                        if sine_subtract:
                            tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)



                        for eventid in eventids: 
                            if eventid%500 == 0:
                                sys.stdout.write('(%i/%i)\r'%(eventid,len(eventids)))
                                sys.stdout.flush()
                            try:
                                delays = -all_time_delays[eventid] #Unsure why I need to invert this.
                                file['impulsivity'][tdset]['hpol'][eventid], file['impulsivity'][tdset]['vpol'][eventid] = tdc.calculateImpulsivityFromTimeDelays(eventid, delays, upsampled_waveforms=None,return_full_corrs=False, align_method=0, hilbert=False,plot=False,impulsivity_window=impulsivity_window,sine_subtract=sine_subtract) 
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

