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
from tools.correlator import Correlator
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
    #TODO: Add these parameters to the 2d data slicer.
    plt.close('all')
    if len(sys.argv) == 2:
        run = int(sys.argv[1])
    else:
        run = 1650

    datapath = os.environ['BEACON_DATA']
    crit_freq_low_pass_MHz = 100 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = 8

    crit_freq_high_pass_MHz = None
    high_pass_filter_order = None

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.03
    sine_subtract_max_freq_GHz = 0.09
    sine_subtract_percent = 0.03

    hilbert=False
    final_corr_length = 2**10

    trigger_types = [2]#[1,2,3]
    db_subset_plot_ranges = [[0,30],[30,40],[40,50]] #Used as bin edges.  
    plot_maps = True

    subset_cm = plt.cm.get_cmap('autumn', 10)
    subset_colors = subset_cm(numpy.linspace(0, 1, len(db_subset_plot_ranges)))[0:len(db_subset_plot_ranges)]

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
        
        if filename is not None:
            with h5py.File(filename, 'r') as file:
                eventids = file['eventids'][...]

                trigger_type_cut = numpy.isin(file['trigger_type'][...], trigger_types)

                dsets = list(file.keys()) #Existing datasets

                if not numpy.isin('cw',dsets):
                    print('cw dataset does not exist for this run.')
                    file.create_group('cw')
                else:
                    cw_dsets = list(file['cw'].keys())
                    print(list(file['cw'].attrs))
                    prep = FFTPrepper(reader, final_corr_length=int(file['cw'].attrs['final_corr_length']), crit_freq_low_pass_MHz=float(file['cw'].attrs['crit_freq_low_pass_MHz']), crit_freq_high_pass_MHz=float(file['cw'].attrs['crit_freq_high_pass_MHz']), low_pass_filter_order=float(file['cw'].attrs['low_pass_filter_order']), high_pass_filter_order=float(file['cw'].attrs['high_pass_filter_order']), waveform_index_range=(None,None), plot_filters=False)
                    prep.addSineSubtract(file['cw'].attrs['sine_subtract_min_freq_GHz'], file['cw'].attrs['sine_subtract_max_freq_GHz'], file['cw'].attrs['sine_subtract_percent'], max_failed_iterations=3, verbose=False, plot=False)
                    if plot_maps:
                        cor = Correlator(reader,  upsample=2**16, n_phi=720, n_theta=720, waveform_index_range=(None,None),crit_freq_low_pass_MHz=float(file['cw'].attrs['crit_freq_low_pass_MHz']), crit_freq_high_pass_MHz=float(file['cw'].attrs['crit_freq_high_pass_MHz']), low_pass_filter_order=float(file['cw'].attrs['low_pass_filter_order']), high_pass_filter_order=float(file['cw'].attrs['high_pass_filter_order']), plot_filter=False,apply_phase_response=True, tukey=False, sine_subtract=True)
                        cor.prep.addSineSubtract(file['cw'].attrs['sine_subtract_min_freq_GHz'], file['cw'].attrs['sine_subtract_max_freq_GHz'], file['cw'].attrs['sine_subtract_percent'], max_failed_iterations=3, verbose=False, plot=False)
                    #Add attributes for future replicability. 
                    
                    raw_freqs = prep.rfftWrapper(prep.t(), numpy.ones_like(prep.t()))[0]
                    df = raw_freqs[1] - raw_freqs[0]
                    freq_bins = (numpy.append(raw_freqs,raw_freqs[-1]+df) - df/2)/1e6 #MHz

                    freq_hz = file['cw']['freq_hz'][...][trigger_type_cut]
                    linear_magnitude = file['cw']['linear_magnitude'][...][trigger_type_cut]
                    binary_cw_cut = file['cw']['has_cw'][...][trigger_type_cut]
                    if not numpy.isin('dbish',cw_dsets):
                        dbish = 10.0*numpy.log10( linear_magnitude[binary_cw_cut]**2 / len(prep.t()))
                    else:
                        dbish = file['cw']['dbish'][...][trigger_type_cut][binary_cw_cut]

                    fig = plt.figure()
                    plt.suptitle('Run %i - Considering trigger types: %s'%(run, str(trigger_types)))

                    ax1 = plt.subplot(3,1,1)
                    plt.hist(freq_hz/1e6,bins=freq_bins)#400,range=[1000*float(file['cw'].attrs['sine_subtract_min_freq_GHz']),1000*float(file['cw'].attrs['sine_subtract_max_freq_GHz'])])
                    plt.xlim(1000*float(file['cw'].attrs['sine_subtract_min_freq_GHz']),1000*float(file['cw'].attrs['sine_subtract_max_freq_GHz']))
                    plt.yscale('log', nonposy='clip')
                    plt.grid(which='both', axis='both')
                    ax1.minorticks_on()
                    ax1.grid(b=True, which='major', color='k', linestyle='-')
                    ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.xlabel('Freq (MHz)')

                    ax2 = plt.subplot(3,1,2)
                    plt.hist(dbish,bins=50) #I think this factor of 2 makes it match monutau?
                    plt.yscale('log', nonposy='clip')
                    plt.grid(which='both', axis='both')
                    ax2.minorticks_on()
                    ax2.grid(b=True, which='major', color='k', linestyle='-')
                    ax2.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.xlabel('Power (dBish)')
                    plt.ylabel('Counts')

                    ax3 = plt.subplot(3,1,3)
                    plt.hist(binary_cw_cut.astype(int),bins=3,weights=numpy.ones(len(binary_cw_cut))/len(binary_cw_cut)) 
                    plt.grid(which='both', axis='both')
                    ax3.minorticks_on()
                    ax3.grid(b=True, which='major', color='k', linestyle='-')
                    ax3.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                    plt.xlabel('Without (0) --- With CW (1)')
                    plt.ylabel('Percent of Counts')

                    print('Max CW Eventid: %i'%numpy.where(trigger_type_cut)[0][numpy.where(binary_cw_cut)[0][numpy.argmax(dbish)]])
                    print('Approximate power is %0.3f dBish, at %0.2f MHz'%(numpy.max(dbish), freq_hz[numpy.where(binary_cw_cut)[0][numpy.argmax(dbish)]]/1e6))

                    if len(db_subset_plot_ranges) > 0:
                        for subset_index, subrange in enumerate(db_subset_plot_ranges):
                            cut = numpy.logical_and(dbish > subrange[0], dbish <= subrange[1])
                            eventid = numpy.random.choice(numpy.where(trigger_type_cut)[0][numpy.where(binary_cw_cut)[0][cut]])
                            prep.plotEvent(eventid, channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=False, sine_subtract=False, apply_tukey=None)
                            prep.plotEvent(eventid, channels=[0,1,2,3,4,5,6,7], apply_filter=True, hilbert=False, sine_subtract=True, apply_tukey=None)

                            ax1.axvline(freq_hz[numpy.where(eventids[trigger_type_cut] == eventid)[0][0] ]/1e6,color=subset_colors[subset_index],label='r%ie%i'%(run, eventid))
                            ax1.legend(loc='upper left')
                            ax2.axvline(dbish[numpy.where(eventids[trigger_type_cut][binary_cw_cut] == eventid)[0][0] ],color=subset_colors[subset_index],label='r%ie%i'%(run, eventid))
                            ax2.legend(loc='upper left')
                            if plot_maps:
                                hpol_result, vpol_result = cor.map(eventid, 'both', center_dir='E', plot_map=True, plot_corr=False, hilbert=hilbert, interactive=True, max_method=0,mollweide=True,circle_zenith=None,circle_az=None)
                file.close()
        else:
            print('filename is None, indicating empty tree.  Skipping run %i'%run)
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


