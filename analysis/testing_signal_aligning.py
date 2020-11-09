#!/usr/bin/env python3
'''
This script is meant to generate a template of event the event type characterized by a peak aroud 77-80 MHz.
I may then try and cross correlate with this to get some metric of similarity with this template.
'''
import os
import sys
import h5py
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
from tools.fftmath import TemplateCompareTool, TimeDelayCalculator
from tools.data_handler import createFile
from tools.correlator import Correlator

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
    if len(sys.argv) > 1:
        run = int(sys.argv[1])
        if len(sys.argv) > 2:
            align_method = int(sys.argv[2])
        else:
            align_method = 8#4#0#4#8
        print('Using align_method = %i'%align_method)
    else:
        run = 1657
        align_method = 8#4#4#0#4#8

    datapath = os.environ['BEACON_DATA']

    crit_freq_low_pass_MHz = None#95 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = None#10

    crit_freq_high_pass_MHz = None#50#None
    high_pass_filter_order = None#4#None

    apply_phase_response = True
    hilbert=False
    final_corr_length = 2**16

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
        filter_string += 'align_%s'%('None')
    else:
        filter_string += 'align_%i'%(align_method)

    print(filter_string)


    plot_filter = False
    plot_multiple = False
    plot_aligned_wf = False
    plot_averaged = True
    plot_maps = False

    waveform_index_range = (None,None)


    hpol_pairs  = numpy.array(list(itertools.combinations((0,2,4,6), 2)))
    vpol_pairs  = numpy.array(list(itertools.combinations((1,3,5,7), 2)))
    pairs       = numpy.vstack((hpol_pairs,vpol_pairs)) 

    try:
        run = int(run)

        reader = Reader(datapath,run)

        try:
            print(reader.status())
        except Exception as e:
            print('Status Tree not present.  Returning Error.')
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(1)
        
        tct = TemplateCompareTool(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False,apply_phase_response=True)
        
        filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
        


        if filename is not None:
            with h5py.File(filename, 'r') as file:
                rf_cut = file['trigger_type'][...] == 2
                peak = file['inband_peak_freq_MHz'][...]
                peak_cut = numpy.logical_and(peak > 76, peak < 81)
                p2p = file['p2p'][...]

                for event_type in [0]:
                    eventids = file['eventids'][...]
                    if event_type == 1:
                        #Pick crosspol
                        p2p_cut = numpy.all(p2p[: , 1::2] >= 16,axis=1)
                        event_cut = numpy.where(numpy.logical_and(p2p_cut,numpy.sum(peak_cut,axis=1) == numpy.max(numpy.sum(peak_cut,axis=1))))[0]
                        template_eventid = eventids[numpy.argmax(numpy.sum(peak_cut,axis=1))] #Pick a crosspol event
                        eventids = eventids[event_cut]
                        pol = 'both'

                    elif event_type == 0:
                        #Pick hpol
                        p2p_cut = numpy.all(p2p[: , 1::2] < 16 ,axis=1)
                        event_cut = numpy.where(numpy.logical_and(p2p_cut,numpy.logical_and(numpy.all(peak_cut[:,0::2],axis=1),numpy.all(~peak_cut[:,1::2],axis=1))))[0]
                        template_eventid = event_cut[0]
                        eventids = eventids[event_cut]
                        pol = 'hpol'
                    #eventids = eventids[numpy.logical_and(numpy.any(peak_cut,axis=1),rf_cut)]

                    eventid = eventids[numpy.random.randint(len(eventids))]
                    tct.setEntry(eventid)
                    waveform_ffts_filtered_corr, upsampled_waveforms = tct.loadFilteredFFTs(eventid, hilbert=False, load_upsampled_waveforms=True)
                    cor02 = scipy.signal.correlate(upsampled_waveforms[0],upsampled_waveforms[2])/(numpy.std(upsampled_waveforms[0])*numpy.std(upsampled_waveforms[2]))
                    dcor02 = scipy.signal.correlate(numpy.diff(upsampled_waveforms[0]),numpy.diff(upsampled_waveforms[2]))/(numpy.std(numpy.diff(upsampled_waveforms[0]))*numpy.std(numpy.diff(upsampled_waveforms[2])))

                    mult = scipy.interpolate.interp1d(numpy.arange(len(cor02)),cor02)(numpy.arange(len(dcor02))+0.5)*dcor02/len(dcor02)**2
                    add = (scipy.interpolate.interp1d(numpy.arange(len(cor02)),cor02)(numpy.arange(len(dcor02))+0.5)+dcor02)/(2*len(dcor02))
                    plt.figure()
                    plt.subplot(2,1,1)
                    plt.plot(upsampled_waveforms[0])
                    plt.plot(upsampled_waveforms[1])
                    plt.subplot(2,1,2)
                    plt.plot(cor02/len(cor02),label='corr')
                    plt.plot(numpy.arange(len(dcor02))+0.5,dcor02/len(dcor02),label='corr of diff')
                    plt.plot(numpy.arange(len(dcor02))+0.5,mult,label='mult')
                    plt.plot(numpy.arange(len(dcor02))+0.5,add,label='add')
                    plt.legend()



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