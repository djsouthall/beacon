'''
The point of this script is to quantify and visualize how measured time delays vary for an
impulsive source as a function of frequency.  This is going to be done through the proxy
of calculating time delays and plotting them as a windows bandwidth varies. 
'''

#General Imports
import numpy
import itertools
import os
import sys
import csv
import scipy
import scipy.interpolate
import pymap3d as pm
from iminuit import Minuit
import inspect
import h5py

#Personal Imports
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_handler import createFile, getTimes
import tools.station as bc
import tools.info as info
import tools.get_plane_tracks as pt
from tools.fftmath import TimeDelayCalculator
from tools.data_slicer import dataSlicerSingleRun
from tools.correlator import Correlator

#Plotting Imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


#Settings
from pprint import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()
datapath = os.environ['BEACON_DATA']

n = 1.0003 #Index of refraction of air  #Should use https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf 
c = 299792458/n #m/s




if __name__ == '__main__':
    try:
        plt.close('all')
        
        use_sources = [   'Solar Plant']

        # use_sources   = [   'Solar Plant',\
        #                     'Tonopah KTPH',\
        #                     'Dyer Cell Tower',\
        #                     'West Dyer Substation',\
        #                     'East Dyer Substation',\
        #                     'Beatty Substation',\
        #                     'Palmetto Cell Tower',\
        #                     'Black Mountain',\
        #                     'Cedar Peak',\
        #                     'Goldfield Hill Tower',\
        #                     'Silver Peak Substation']

        # Azimuths
        # Northern Cell Tower : 49.393
        # Quarry Substation : 46.095
        # Solar Plant : 43.441
        # Tonopah KTPH : 30.265
        # Dyer Cell Tower : 29.118
        # Tonopah Vortac : 25.165
        # Silver Peak Lithium Mine : 19.259
        # Silver Peak Substation : 19.112
        # Silver Peak Town Antenna : 18.844
        # Past SP Substation : 18.444
        # Goldfield Hill Tower : 10.018
        # Goldield Town Tower : 8.967
        # Goldfield KGFN-FM : 8.803
        # Cedar Peak : 4.992
        # West Dyer Substation : 3.050
        # Black Mountain : -13.073
        # Palmetto Cell Tower : -13.323
        # East Dyer Substation : -17.366
        # Beatty Substation : -29.850
        # Beatty Airport Antenna : -31.380
        # Beatty Airport Vortac : -33.040

        #only_plot = ['East Dyer Substation','West Dyer Substation','Northern Cell Tower','Solar Plant','Quarry Substation','Tonopah KTPH','Dyer Cell Tower','Tonopah Vortac','Beatty Airport Vortac','Palmetto Cell Tower','Cedar Peak','Goldfield Hill Tower','Goldield Town Tower','Goldfield KGFN-FM','Silver Peak Town Antenna','Silver Peak Lithium Mine','Past SP Substation','Silver Peak Substation']#['Solar Plant','Tonopah KTPH','Beatty Airport Vortac','Palmetto Cell Tower','Goldfield Hill Tower','Silver Peak Substation']
        impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        map_direction_dset_key = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1'

        plot_residuals = True

        final_corr_length = 2**17

        filter_bandwidth = 10
        n_steps = 10
        band_centers = numpy.linspace(35,80,n_steps)
        filter_order_low = 20
        filter_order_high = 20


        limit_events = 10

        sine_subtract = False
        sine_subtract_min_freq_GHz = 0.03
        sine_subtract_max_freq_GHz = 0.13
        sine_subtract_percent = 0.03
        normalize_filters=True

        #waveform_index_range = (None,None)

        apply_phase_response = True
        hilbert = False

        print('Potential RFI Source Locations.')
        sources_ENU, data_slicer_cut_dict = info.loadValleySourcesENU()

        print('Source Directions based on ENU, sorted North to South are:')

        keys = list(sources_ENU.keys())
        azimuths = []

        for source_key in keys:
            azimuths.append(numpy.rad2deg(numpy.arctan2(sources_ENU[source_key][1],sources_ENU[source_key][0])))


        sort_cut = numpy.argsort(azimuths)[::-1]
        for index in sort_cut:
            print('%s : %0.3f'%(keys[index], azimuths[index]))

        run = 1650

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        #DETERMINE THE TIME DELAYS TO BE USED IN THE ACTUAL CALCULATION

        print('Calculating time delays from info.py')
        reader = Reader(datapath,run)

            
        ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                    curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

        ds.addROI('Simple Template V > 0.7',{'cr_template_search_v':[0.7,1.0]})# Adding 2 ROI in different rows and appending as below allows for "OR" instead of "AND"
        ds.addROI('Simple Template H > 0.7',{'cr_template_search_h':[0.7,1.0]})
        #Done for OR condition
        _eventids = numpy.sort(numpy.unique(numpy.append(ds.getCutsFromROI('Simple Template H > 0.7',load=False,save=False),ds.getCutsFromROI('Simple Template V > 0.7',load=False,save=False))))

        time_delay_dict = {}
        for source_key, cut_dict in data_slicer_cut_dict.items():
            if not(source_key in use_sources):
                continue #Skipping calculating that one.

            ds.addROI(source_key,cut_dict)
            roi_eventids = numpy.intersect1d(ds.getCutsFromROI(source_key),_eventids)
            roi_impulsivity = ds.getDataFromParam(roi_eventids,'impulsivity_h')
            roi_impulsivity_sort = numpy.argsort(roi_impulsivity)[::-1] #Reverse sorting so high numbers are first.
            
            if len(roi_eventids) > limit_events:
                print('LIMITING TIME DELAY CALCULATION TO %i MOST IMPULSIVE EVENTS'%limit_events)
                roi_eventids = numpy.sort(roi_eventids[roi_impulsivity_sort[0:limit_events]])

            argmaxs = []
            reader.setEntry(roi_eventids[0])
            for channel in range(8):
                argmaxs.append(numpy.argmax(reader.wf(channel)))

            waveform_index_range = (int(numpy.min(argmaxs) - 40), int(numpy.max(argmaxs) + 100))



            tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
            if sine_subtract:
                tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
            time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(roi_eventids,align_method=0,hilbert=hilbert,plot=False, sine_subtract=sine_subtract)
            unfiltered_time_delays = numpy.mean(time_shifts,axis=1)

            tdc.plotEvent(roi_eventids[0], channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=False, sine_subtract=sine_subtract, apply_tukey=None, additional_title_text=None, time_delays=None, verbose=False)


            time_delays = numpy.zeros((n_steps,12))


            plt.figure()
            plt.subplot(3,1,1)
            ax1 = plt.gca()
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            if plot_residuals:
                plt.ylabel('Residual from\nUnfiltered Signal (ns)')
            else:
                plt.ylabel('Time Delay (ns)')


            ax2 = plt.subplot(3,1,2,sharex=ax1)
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            ax3 = plt.subplot(3,1,3,sharex=ax2)
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            if normalize_filters:
                plt.ylabel('Normalized\nFilter Magnitude')
            else:
                plt.ylabel('Filter Magnitude')
            plt.xlabel('Band Center (MHz)\n%0.2f MHz Band'%(filter_bandwidth))



            for band_index, band_center in enumerate(band_centers):
                crit_freq_low_pass_MHz = band_center - filter_bandwidth/2#; print(crit_freq_low_pass_MHz)
                crit_freq_high_pass_MHz = band_center + filter_bandwidth/2#; print(crit_freq_high_pass_MHz)
                low_pass_filter_order = filter_order_low#; print(low_pass_filter_order)
                high_pass_filter_order = filter_order_high#; print(high_pass_filter_order)
        
                tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=float(crit_freq_low_pass_MHz), crit_freq_high_pass_MHz=float(crit_freq_high_pass_MHz), low_pass_filter_order=int(low_pass_filter_order), high_pass_filter_order=int(high_pass_filter_order),waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
                if sine_subtract:
                    tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
                
                tdc.plotEvent(roi_eventids[0], channels=[0], apply_filter=True, hilbert=False, sine_subtract=sine_subtract, apply_tukey=None, additional_title_text='band_center=%0.2f'%band_center, time_delays=None, verbose=False)

                time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(roi_eventids,align_method=0,hilbert=hilbert,plot=False, sine_subtract=sine_subtract)
                time_delays[band_index] = numpy.mean(time_shifts,axis=1)

                if normalize_filters:
                    ax3.plot(tdc.freqs_original/1e6, numpy.abs(tdc.filter_original[0])/numpy.max(numpy.abs(tdc.filter_original[0])),alpha=0.5)
                else:
                    ax3.plot(tdc.freqs_original/1e6, numpy.abs(tdc.filter_original[0]),alpha=0.5)
                ax3.axvline(band_center,alpha=0.5, color=ax3.lines[-1].get_color())




            for baseline_index in range(12):
                if baseline_index < 6:
                    ax = ax1
                else:
                    ax = ax2
                if plot_residuals:
                    ax.plot(band_centers,time_delays[:,baseline_index] - unfiltered_time_delays[baseline_index],label=str(pairs[baseline_index]))
                else:
                    ax.plot(band_centers,time_delays[:,baseline_index],label=str(pairs[baseline_index]))
                ax.legend()

            ax1.set_xlim(0,100)
    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






