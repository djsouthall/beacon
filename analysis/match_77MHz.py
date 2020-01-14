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
from objects.fftmath import TemplateCompareTool, TimeDelayCalculator
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
from scipy.optimize import curve_fit
def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))


if __name__=="__main__":
    align_default = 0
    if len(sys.argv) > 1:
        run = int(sys.argv[1])
        if len(sys.argv) > 2:
            align_method = int(sys.argv[2])
        else:
            align_method = align_default#4#0#4#8
        print('Using align_method = %i'%align_method)
    else:
        run = 1657
        align_method = align_default#4#4#0#4#8

    datapath = os.environ['BEACON_DATA']

    crit_freq_low_pass_MHz = None#95 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = None#10

    crit_freq_high_pass_MHz = None#50#None
    high_pass_filter_order = None#4#None

    apply_phase_response = True
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
        filter_string += 'align_%s'%('None')
    else:
        filter_string += 'align_%i'%(align_method)

    print(filter_string)


    plot_filter = False
    plot_multiple = False
    plot_aligned_wf = True
    plot_averaged = True
    plot_maps = False
    plot_td = True

    waveform_index_range = (None,500)


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
        
        tct = TemplateCompareTool(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=(None,None), plot_filters=False,apply_phase_response=True)
        
        filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
        


        if filename is not None:
            with h5py.File(filename, 'r') as file:
                print(list(file['time_delays'].keys()))
                rf_cut = file['trigger_type'][...] == 2
                peak = file['inband_peak_freq_MHz'][...]
                peak_cut = numpy.logical_and(peak > 76, peak < 81)
                p2p = file['p2p'][...]


                '''
                These are extremely consistent.  I should look at the time delays and see how consistent they are and if they can
                easily be cut on.  

                Cross pol ones look like they are stronger in general but have some reflection?
                '''
                if plot_td:
                    for pair_index in range(12):
                        plt.figure(pair_index)

                for event_type in [1]:
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

                    tct.setTemplateToEvent(template_eventid)
                    choice_events = numpy.sort(numpy.random.choice(eventids,size=numpy.min((1000,len(eventids))),replace=False))
                    times, averaged_waveforms = tct.averageAlignedSignalsPerChannel( choice_events, align_method=0, template_eventid=None, plot=plot_aligned_wf,event_type=None)
                    

                    resampled_averaged_waveforms = numpy.zeros((8,len(tct.waveform_times_corr)))
                    for channel in range(8):
                        #Resampling averaged waveforms to be more compatible with cross correlation framework. 
                        resampled_averaged_waveforms[channel] = scipy.interpolate.interp1d(times,averaged_waveforms[channel],kind='cubic',bounds_error=False,fill_value=0)(tct.waveform_times_corr)

                    #numpy.savetxt('./template_77MHz_type%i.csv'%(event_type), delimiter=",")
                    
                    #FFTs of resampled templates which will be used when performing cross correlation.
                    averaged_waveforms_ffts = numpy.fft.rfft(resampled_averaged_waveforms,axis=1)
                    averaged_waveforms_ffts_freqs = numpy.fft.rfftfreq(len(tct.waveform_times_corr),(tct.waveform_times_corr[1]-tct.waveform_times_corr[0])/1.0e9)

                    #Set up time delay calculator used to determine time delays between averaged waveforms and later all waveforms internally.
                    tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False,apply_phase_response=apply_phase_response)
                    indices, corr_time_shifts, max_corrs, pairs, corrs = tdc.calculateTimeDelays(averaged_waveforms_ffts, averaged_waveforms, return_full_corrs=True, align_method=align_method)


                    if plot_averaged == True:                    
                        plt.figure()
                        for channel in range(8):
                            print('Channel %i'%channel)
                            wfs = averaged_waveforms[channel]
                            ffts = numpy.fft.rfft(wfs)
                            freqs = numpy.fft.rfftfreq(len(times),times[1]*1e-9)/1e6
                            plt.plot(freqs,10*numpy.log10(abs(ffts)),label='Ch%i'%channel)

                        plt.xlabel('MHz')
                        plt.ylabel('dBish')
                        plt.title('Event Classification Type == %i'%event_type)
                        plt.legend()

                    if plot_maps == True:
                        cor = Correlator(reader,  upsample=final_corr_length//2, n_phi=360, n_theta=360, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter)
                        cor.averagedMap(eventids, pol, plot_map=True, hilbert=False, max_method=None)

                    time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids,align_method=align_method,hilbert=hilbert,plot=plot_multiple,hpol_cut=None,vpol_cut=None)
                    #bins = numpy.arange(-1000,1000)#numpy.linspace(min(numpy.min(time_shifts)*.99,numpy.min(time_shifts)*1.01),numpy.max(time_shifts)*1.01,1000)
                    

                    if plot_td:

                        fit_time_delays = []

                        for pair_index, pair in enumerate(pairs):
                            if event_type == 0 and pair in vpol_pairs:
                                continue
                            fig = plt.figure(pair_index)
                            n, bins, patches = plt.hist(time_shifts[pair_index],bins = 100,label='Event_type %i, pair %i'%(event_type, pair_index))
                            x = (bins[1:len(bins)] + bins[0:len(bins)-1] )/2.0
                            #Fit Gaussian
                            popt, pcov = curve_fit(gaus,x,n,p0=[100,numpy.mean(time_shifts[pair_index]),1.0])
                            popt[2] = abs(popt[2]) #I want positive sigma.
                            plot_x = numpy.linspace(min(x),max(x),1000)

                            plt.plot(plot_x,gaus(plot_x,*popt),'--',label='fit')

                            plt.axvline(popt[1],c='y',linestyle='--',label='Fit Center = %f'%popt[1])
                            #plt.axvline(corr_time_shifts[pair_index],color = 'r', linestyle='--',label='Averaged WF TD = %f'%(corr_time_shifts[pair_index]))
                            plt.legend()
                            plt.ylabel('Counts')
                            plt.xlabel('Time Delays (ns)')
                            fit_time_delays.append(popt[1])
                    fit_time_delays = numpy.array(fit_time_delays)
                    print(fit_time_delays)
                    #numpy.savetxt('./time_delays_77MHz.csv',fit_time_delays, delimiter=",")

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