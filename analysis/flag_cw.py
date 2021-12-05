'''
This is a script load waveforms using the sine subtraction method, and save any identified CW present in events.
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv
import h5py

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info
from tools.fftmath import FFTPrepper
from tools.data_handler import createFile

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
plt.ion()

datapath = os.environ['BEACON_DATA']



if __name__=="__main__":
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 1701

    datapath = os.environ['BEACON_DATA']

    crit_freq_low_pass_MHz = None#85
    low_pass_filter_order = None#6

    crit_freq_high_pass_MHz = None#25
    high_pass_filter_order = None#8

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.00
    sine_subtract_max_freq_GHz = 0.25
    sine_subtract_percent = 0.03

    hilbert=False
    final_corr_length = 2**13




    try:
        run = int(run)

        reader = Reader(datapath,run)
        prep = FFTPrepper(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=(None,None), plot_filters=False)
        prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=5, verbose=False, plot=False)
        
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
        
        if filename is not None:
            with h5py.File(filename, 'a') as file:
                eventids = file['eventids'][...]
                dsets = list(file.keys()) #Existing datasets

                if not numpy.isin('cw',dsets):
                    file.create_group('cw')
                else:
                    print('cw group already exists in file %s'%filename)

                cw_dsets = list(file['cw'].keys())

                #Add attributes for future replicability. 

                file['cw'].attrs['final_corr_length'] = final_corr_length
                file.attrs['t_ns'] = prep.t() #len of this is needed to get linear magnitude to dbish.

                if crit_freq_low_pass_MHz is not None:
                    file['cw'].attrs['crit_freq_low_pass_MHz'] = crit_freq_low_pass_MHz 
                else:
                    file['cw'].attrs['crit_freq_low_pass_MHz'] = 0

                if low_pass_filter_order is not None:
                    file['cw'].attrs['low_pass_filter_order'] = low_pass_filter_order 
                else:
                    file['cw'].attrs['low_pass_filter_order'] = 0

                if crit_freq_high_pass_MHz is not None:
                    file['cw'].attrs['crit_freq_high_pass_MHz'] = crit_freq_high_pass_MHz 
                else:
                    file['cw'].attrs['crit_freq_high_pass_MHz'] = 0

                if high_pass_filter_order is not None:
                    file['cw'].attrs['high_pass_filter_order'] = high_pass_filter_order 
                else:
                    file['cw'].attrs['high_pass_filter_order'] = 0

                file['cw'].attrs['sine_subtract_min_freq_GHz'] = sine_subtract_min_freq_GHz 
                file['cw'].attrs['sine_subtract_max_freq_GHz'] = sine_subtract_max_freq_GHz 
                file['cw'].attrs['sine_subtract_percent'] = sine_subtract_percent 

                if not numpy.isin('freq_hz',cw_dsets):
                    file['cw'].create_dataset('freq_hz', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in cw[\'freq_hz\'] of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('linear_magnitude',cw_dsets):
                    file['cw'].create_dataset('linear_magnitude', (file.attrs['N'],), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in cw[\'linear_magnitude\'] of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('has_cw',cw_dsets):
                    file['cw'].create_dataset('has_cw', (file.attrs['N'],), dtype=bool, compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in cw[\'has_cw\'] of %s will be overwritten by this analysis script.'%filename)

                if not numpy.isin('dbish',cw_dsets):
                    file['cw'].create_dataset('dbish', (file.attrs['N'],), dtype=float, compression='gzip', compression_opts=4, shuffle=True)
                else:
                    print('Values in cw[\'dbish\'] of %s will be overwritten by this analysis script.'%filename)


                for eventid in eventids: 
                    if eventid%500 == 0:
                        sys.stdout.write('(%i/%i)\r'%(eventid,len(eventids)))
                        sys.stdout.flush()
                    try:
                        prep.setEntry(eventid)
                        t_ns = prep.t()
                        peak_freqs = numpy.array([])
                        peak_magnitude = numpy.array([])

                        for channel in range(8):
                            wf, ss_freqs, n_fits = prep.wf(channel,apply_filter=False,hilbert=False,tukey=None,sine_subtract=True, return_sine_subtract_info=True)
                            freqs, spec_dbish, spec = prep.rfftWrapper(t_ns, wf)

                            #Without sine_subtract to plot what the old signal spectral peak value.
                            raw_wf = prep.wf(channel,apply_filter=False,hilbert=False,tukey=None,sine_subtract=False, return_sine_subtract_info=True)
                            raw_freqs, raw_spec_dbish, raw_spec = prep.rfftWrapper(t_ns, raw_wf)

                            for ss_n in range(len(n_fits)):
                                unique_peak_indices = numpy.unique(numpy.argmin(numpy.abs(numpy.tile(raw_freqs,(n_fits[ss_n],1)).T - 1e9*ss_freqs[0]),axis=0)) #Gets indices of freq of peaks in non-upsampled spectrum.
                                unique_peak_freqs = raw_freqs[unique_peak_indices]
                                unique_peak_magnitude = numpy.abs(raw_spec[unique_peak_indices]).astype(numpy.double)

                                peak_freqs = numpy.append(peak_freqs,unique_peak_freqs)
                                peak_magnitude = numpy.append(peak_magnitude,unique_peak_magnitude) #divided by 2 when plotting

                        if len(peak_magnitude) > 0:
                            max_index = numpy.argmax(peak_magnitude)
                            max_freq = peak_freqs[max_index]
                            max_magnitude = peak_magnitude[max_index]

                            file['cw']['has_cw'][eventid] = True
                            file['cw']['freq_hz'][eventid] = max_freq
                            file['cw']['linear_magnitude'][eventid] = max_magnitude
                            #file['cw']['dbish'][eventid] = 
                        else:
                            file['cw']['has_cw'][eventid] = False
                            file['cw']['freq_hz'][eventid] = 0.0
                            file['cw']['freq_hz'][eventid] = 0.0

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








'''












if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    run = 1650
    eventids = [499,45059,58875]
    print('Run %i'%run)
    final_corr_length = 2**12 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = None#80 #This new pulser seems to peak in the region of 85 MHz or so
    crit_freq_high_pass_MHz = None#65
    low_pass_filter_order = None#3
    high_pass_filter_order = None#6
    plot_filters = False

    enable_plots = True
    
    for val in [False,True]:

        reader = Reader(datapath,run)
        prep = FFTPrepper(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=(None,None), plot_filters=plot_filters)
        if val:
            prep.addSineSubtract(0.03, 0.090, 0.05, max_failed_iterations=3, verbose=False, plot=False)#Test purposes
        for event_index, eventid in enumerate(eventids):
            if enable_plots:
                plt.figure()
                plt.title('Run %i, eventid %i'%(run,eventid))
                plt.subplot(2,1,1)
                plt.ylabel('adu')
                plt.xlabel('ns')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.subplot(2,1,2)
                plt.ylabel('dBish')
                plt.xlabel('freq')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                

            prep.setEntry(eventid)
            t_ns = prep.t()
            print(eventid)
            for channel in [0,1,2,3,4,5,6,7]:
                channel=int(channel)
                wf, ss_freqs, n_fits = prep.wf(channel,apply_filter=False,hilbert=False,tukey=None,sine_subtract=True, return_sine_subtract_info=True)
                freqs, spec_dbish, spec = prep.rfftWrapper(t_ns, wf)
                
                #Without sine_subtract to plot what the old signal looked like.
                raw_wf = prep.wf(channel,apply_filter=False,hilbert=False,tukey=None,sine_subtract=False, return_sine_subtract_info=True)
                raw_freqs, raw_spec_dbish, raw_spec = prep.rfftWrapper(t_ns, raw_wf)

                peak_freqs = numpy.array([])
                peak_magnitude = numpy.array([])

                for ss_n in range(len(n_fits)):
                    freqs_tiled = numpy.tile(freqs,(n_fits[ss_n],))
                    unique_peak_indices = numpy.unique(numpy.argmin(numpy.abs(numpy.tile(raw_freqs,(n_fits[ss_n],1)).T - 1e9*ss_freqs[0]),axis=0)) #Gets indices of freq of peaks in non-upsampled spectrum.
                    unique_peak_freqs = raw_freqs[unique_peak_indices]
                    unique_peak_magnitude = raw_spec[unique_peak_indices]
                    unique_peak_db = raw_spec_dbish[unique_peak_indices].astype(numpy.double)


                    peak_freqs = numpy.append(peak_freqs,unique_peak_freqs)
                    peak_db = numpy.append(peak_db,unique_peak_db) #divided by 2 when plotting

                print(n_fits)
                print(ss_freqs)

                if enable_plots:
                    plt.subplot(2,1,1)
                    plt.plot(t_ns,wf)

                    plt.subplot(2,1,2)
                    plt.plot(freqs/1e6,spec_dbish/2.0,label='Ch %i'%channel)
                    # if len(peak_freqs) > 0:
                    #     plt.plot(raw_freqs/1e6,raw_spec_dbish/2.0,alpha=0.5,linestyle='--',label='Ch %i'%channel)
                    if len(peak_freqs) > 0:
                        plt.scatter(peak_freqs/1e6,peak_db/2.0,label='Ch %i Removed Peak Max'%channel)
                    plt.legend(loc = 'upper right')
                    plt.xlim(10,110)
                    plt.ylim(-10,30)
                
'''