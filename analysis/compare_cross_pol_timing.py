'''
This is designed to perform cross correlations using antennas on the same mast, with the goal of highlighting
potential cable delay descrepencies.  Ideally these time delays should be near zero, as the antennas are located
in roughly the same location. 
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
from beacon.tools.data_handler import createFile, getTimes
import beacon.tools.station as bc
import beacon.tools.info as info
import beacon.tools.get_plane_tracks as pt
from beacon.tools.fftmath import TimeDelayCalculator
from beacon.tools.data_slicer import dataSlicerSingleRun
from beacon.tools.correlator import Correlator

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

print('IF USING TIME DELAYS FOR PULSERS, I TRUST CURRENT BY EYE VERSION')

if __name__ == '__main__':
    try:
        plt.close('all')
        if len(sys.argv) == 2:
            deploy_index = int(sys.argv[2])
        else:
            deploy_index = info.returnDefaultDeploy()

        print('Plotting for Deploy Index %i '%deploy_index)

        #### PULSERS ####
        if False:
            included_pulsers =        [ 'run1507',\
                                        'run1509',\
                                        'run1511']
        else:
            included_pulsers = []

        #### VALLEY SOURCES ####                      
        valley_source_run = 1650
        impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        map_direction_dset_key = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1'


        # 'Solar Plant'               :{'time_delay_0subtract1_h':[-127,-123],'time_delay_0subtract2_h':[-127,-123.5]},\
        # # 'A'                         :{'time_delay_0subtract1_h':[-127,-123],'time_delay_0subtract2_h':[-127,-123.5]},\
        # 'Booker Antenna'            :{'time_delay_0subtract1_h':[-140.5,-137],'time_delay_0subtract2_h':[-90,-83.5],'time_delay_0subtract3_h':[-167,-161],'time_delay_1subtract2_h':[46,55]},\
        # # 'C'                         :{'time_delay_0subtract1_h':[-140.5,-137],'time_delay_0subtract2_h':[-90,-83.5],'time_delay_0subtract3_h':[-167,-161],'time_delay_1subtract2_h':[46,55]},\
        # 'Tonopah AFS GATR Site'     :{'time_delay_0subtract1_h':[-135,-131],'time_delay_0subtract2_h':[-111,-105]},\
        # # 'B'                         :{'time_delay_0subtract1_h':[-135,-131],'time_delay_0subtract2_h':[-111,-105]},\
        # 'KNKN223'                   :{'time_delay_0subtract1_h':[-127,-123],'time_delay_0subtract2_h':[-127,-123.5]},\
        # 'Dyer Cell Tower'           :{'time_delay_0subtract1_h':[-140.5,-137],'time_delay_0subtract2_h':[-90,-83.5],'time_delay_0subtract3_h':[-167,-161],'time_delay_1subtract2_h':[46,55]},\
        # 'Beatty Airport Antenna'    :{'time_delay_0subtract1_h':[-124.5,-121],'time_delay_0subtract2_h':[22.5,28.5]},\
        # # 'E'                         :{'time_delay_0subtract1_h':[-124.5,-121],'time_delay_0subtract2_h':[22.5,28.5]}
        # 'Palmetto Cell Tower'       :{'time_delay_0subtract1_h':[-138,-131.7],'time_delay_0subtract2_h':[-7,-1]},\
        # 'Cedar Peak'                :{'time_delay_0subtract1_h':[-143,-140],'time_delay_0subtract2_h':[-60.1,-57.4]},\
        # # 'D'                         :{'time_delay_0subtract1_h':[-143,-140],'time_delay_0subtract2_h':[-60.1,-57.4]},\
        # 'Silver Peak Substation'    :{'time_delay_0subtract1_h':[-140.5,-137],'time_delay_0subtract2_h':[-90,-83.5],'time_delay_0subtract3_h':[-167,-161],'time_delay_1subtract2_h':[46,55]},\


        if False:
            included_valley_sources = ['Solar Plant','Booker Antenna','Tonopah AFS GATR Site','KNKN223','Dyer Cell Tower','Beatty Airport Antenna','Palmetto Cell Tower','Cedar Peak','Silver Peak Substation']
            # included_valley_sources = [ 'Northern Cell Tower',\
            #                             'Booker Antenna',\
            #                             'Tonopah AFS GATR Site',\
            #                             'Miller Substation',\
            #                             'Dyer Cell Tower',\
            #                             'Beatty Airport Antenna',\
            #                             'Palmetto Cell Tower',\
            #                             'Cedar Peak']

            # [ 'Tonopah AFS GATR Site',\
            #     'Tonopah Vortac',\
            #     'Tonopah Airport Antenna',\
            #     'Dyer Cell Tower',\
            #     'West Dyer Substation',\
            #     'East Dyer Substation',\
            #     'Oasis',\
            #     'Beatty Substation',\
            #     'Palmetto Cell Tower',\
            #     'Cedar Peak',\
            #     'Dome Thing',\
            #     'Goldfield Hill Tower',\
            #     'Silver Peak Substation',\
            #     'Silver Peak Town Antenna',\
            #     'Silver Peak Lithium Mine',\
            #     'Past SP Substation']
        else:
            included_valley_sources = ['A','B','C','D','E','Cedar Peak']

        #### AIRPLANES ####
        plot_animated_airplane = False #Otherwise plots first event from each plane.  
        if False:
            included_airplanes =      [ '1728-62026',\
                                        '1773-14413',\
                                        '1773-63659',\
                                        '1774-178',\
                                        '1774-88800',\
                                        '1783-28830',\
                                        '1784-7166']


        else:
            included_airplanes =      []#['1728-62026']

        plot_time_delay_calculations = False
        plot_time_delays_on_maps = True
        limit_events = 1e6#10 #Number of events use for time delay calculation


        plot_residuals = False
        plot_histograms = False
        iterate_sub_baselines = 3 #The lower this is the higher the time it will take to plot.  Does combinatoric subsets of baselines with this length. 

        final_corr_length = 2**17
        cor_upsample = final_corr_length

        crit_freq_low_pass_MHz = 85#None#[80,70,70,70,70,70,60,70]#90#
        low_pass_filter_order = 6#None#[0,8,8,8,10,8,3,8]#8#

        crit_freq_high_pass_MHz = 25#70#None#60
        high_pass_filter_order = 8#6#None#8

        sine_subtract = False
        sine_subtract_min_freq_GHz = 0.03
        sine_subtract_max_freq_GHz = 0.13
        sine_subtract_percent = 0.03

        waveform_index_range = (None,None)

        apply_phase_response = True
        hilbert = False

        included_antennas_channels = numpy.arange(8)

        cable_delays = info.loadCableDelays(deploy_index=deploy_index)
        cable_delays_differences = cable_delays['hpol'] - cable_delays['vpol']#numpy.array([cable_delays['hpol'][0],cable_delays['vpol'][0],cable_delays['hpol'][1],cable_delays['vpol'][1],cable_delays['hpol'][2],cable_delays['vpol'][2],cable_delays['hpol'][3],cable_delays['vpol'][3]])


        #### PULSERS ####

        if len(included_pulsers) > 0:            
            pulser_locations_ENU = info.loadPulserLocationsENU(deploy_index=deploy_index)
            known_pulser_ids = info.loadPulserEventids(remove_ignored=True)


            for pulser_key in included_pulsers:
                run = int(pulser_key.replace('run',''))
                reader = Reader(datapath,run)
                tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
                hpol_eventids = known_pulser_ids[pulser_key]['hpol']
                hpol_eventids = hpol_eventids[0:min(len(hpol_eventids),limit_events)]
                vpol_eventids = known_pulser_ids[pulser_key]['vpol']
                vpol_eventids = vpol_eventids[0:min(len(vpol_eventids),limit_events)]

                #Calculate old and new geometries
                #Distance needed when calling correlator, as it uses that distance.
                pulser_ENU = numpy.array([pulser_locations_ENU[pulser_key][0] , pulser_locations_ENU[pulser_key][1] , pulser_locations_ENU[pulser_key][2]])
                distance_m = numpy.linalg.norm(pulser_ENU)
                zenith_deg = numpy.rad2deg(numpy.arccos(pulser_ENU[2]/distance_m))
                elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(pulser_ENU[2]/distance_m))
                azimuth_deg = numpy.rad2deg(numpy.arctan2(pulser_ENU[1],pulser_ENU[0]))

                #tdc.plotEvent(hpol_eventids[0], channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=False, sine_subtract=True, apply_tukey=None, additional_title_text=None, time_delays=None, verbose=False)
                #indices, time_shifts, max_corrs, pairs, corrs = tdc.calculateTimeDelaysFromEvent(hpol_eventids[0], return_full_corrs=True, align_method=0, hilbert=False, align_method_10_estimate=None, align_method_10_window_ns=8, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0,sine_subtract=True, crosspol_delays=True)
                timeshifts, max_corrs, pairs = tdc.calculateMultipleTimeDelays(hpol_eventids, align_method=0, hilbert=False, plot=False, hpol_cut=None, vpol_cut=None, colors=None, align_method_10_estimates=None, align_method_10_window_ns=8, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0,sine_subtract=True, crosspol_delays=True)
                print(pulser_key)
                print('Signal Delays:')
                print(numpy.mean(timeshifts,axis=1))
                print('Cable Delays:')
                print(cable_delays_differences)

        #### VALLEY SOURCES ####

        fig = plt.figure()
        axs = [plt.subplot(2,2,i+1) for i in range(4)]
        for ax in axs:
            ax.set_ylabel('Normalized Counts')
            ax.set_xlabel('H-V Time Delay')
        half_window = 100
        bins = numpy.linspace(-half_window,half_window,4*half_window+1)#bins = numpy.arange(min(tdc.corr_time_shifts),max(tdc.corr_time_shifts),1000)


        if len(included_valley_sources) > 0:

            map_resolution = 0.25 #degrees
            range_phi_deg = (-90, 90)
            range_theta_deg = (0,180)
            n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
            n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
            
            sources_ENU, data_slicer_cut_dict = info.loadValleySourcesENU(deploy_index=deploy_index) #Plot all potential sources

            run = valley_source_run
            reader = Reader(datapath,run)

            tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
            if sine_subtract:
                tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

            ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                        curve_choice=0, trigger_types=[2],included_antennas=included_antennas_channels,include_test_roi=False,\
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


            valley_sources_time_delay_dict = {}
            for valley_source_key in included_valley_sources:
                ds.addROI(valley_source_key,data_slicer_cut_dict[valley_source_key])
                roi_eventids = numpy.intersect1d(ds.getCutsFromROI(valley_source_key),_eventids)
                roi_impulsivity = ds.getDataFromParam(roi_eventids,'impulsivity_h')
                roi_impulsivity_sort = numpy.argsort(roi_impulsivity)[::-1] #Reverse sorting so high numbers are first.
                
                if len(roi_eventids) > limit_events:
                    print('LIMITING TIME DELAY CALCULATION TO %i MOST IMPULSIVE EVENTS'%limit_events)
                    roi_eventids = numpy.sort(roi_eventids[roi_impulsivity_sort[0:limit_events]])
            
                print('Calculating time delays for %s'%valley_source_key)


                #Calculate old and new geometries
                #Distance needed when calling correlator, as it uses that distance.
                valley_source_ENU = numpy.array([sources_ENU[valley_source_key][0] , sources_ENU[valley_source_key][1] , sources_ENU[valley_source_key][2]])
                distance_m = numpy.linalg.norm(valley_source_ENU)
                zenith_deg = numpy.rad2deg(numpy.arccos(valley_source_ENU[2]/distance_m))
                elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(valley_source_ENU[2]/distance_m))
                azimuth_deg = numpy.rad2deg(numpy.arctan2(valley_source_ENU[1],valley_source_ENU[0]))

                #tdc.plotEvent(roi_eventids[0], channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=False, sine_subtract=True, apply_tukey=None, additional_title_text=None, time_delays=None, verbose=False)
                #indices, time_shifts, max_corrs, pairs, corrs = tdc.calculateTimeDelaysFromEvent(roi_eventids[0], return_full_corrs=True, align_method=0, hilbert=False, align_method_10_estimate=None, align_method_10_window_ns=8, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0,sine_subtract=True, crosspol_delays=True)
                timeshifts, max_corrs, pairs = tdc.calculateMultipleTimeDelays(roi_eventids, align_method=0, hilbert=False, plot=False, hpol_cut=None, vpol_cut=None, colors=None, align_method_10_estimates=None, align_method_10_window_ns=8, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0,sine_subtract=True, crosspol_delays=True)

                for pair_index in numpy.arange(4):
                    label = valley_source_key + '\n%i events'%(numpy.shape(timeshifts)[1])
                    axs[pair_index].hist(timeshifts[pair_index], alpha=0.7, label=label,bins=bins, weights = numpy.ones(numpy.shape(timeshifts)[1])/numpy.shape(timeshifts)[1] )
                    axs[pair_index].set_ylim(0,1)
                print(valley_source_key)
                print('Signal Delays:')
                print(numpy.mean(timeshifts,axis=1))
                print('Cable Delays:')
                print(cable_delays_differences)
            plt.legend(fontsize=8)

        #### AIRPLANES ####

        if len(included_airplanes) > 0:
            print('Loading known plane locations.')
            known_planes, calibrated_trigtime, output_tracks = pt.getKnownPlaneTracks(ignore_planes=[]) # ['1728-62026','1773-14413','1773-63659','1774-88800','1783-28830','1784-7166']#'1774-88800','1728-62026'
            origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
            antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=deploy_index)

            plane_polys = {}
            interpolated_plane_locations = {}


            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            for index, key in enumerate(list(calibrated_trigtime.keys())):
                if key not in included_airplanes:
                    continue

                run = int(key.split('-')[0])
                reader = Reader(datapath,run)
                eventids = known_planes[key]['eventids'][:,1]

                enu = numpy.array(pm.geodetic2enu(output_tracks[key]['lat'],output_tracks[key]['lon'],output_tracks[key]['alt'],origin[0],origin[1],origin[2]))
                
                unique_enu_indices = numpy.sort(numpy.unique(enu,axis=1,return_index=True)[1])
                distance = numpy.sqrt(enu[0]**2 + enu[0]**2 + enu[0]**2)
                distance_cut_indices = numpy.where(distance/1000.0 < 100)
                unique_enu_indices = unique_enu_indices[numpy.isin(unique_enu_indices,distance_cut_indices)]
                
                plane_polys[key] = pt.PlanePoly(output_tracks[key]['timestamps'][unique_enu_indices],enu[:,unique_enu_indices],plot=False)

                interpolated_plane_locations[key] = plane_polys[key].poly(calibrated_trigtime[key])

                source_distance_m = numpy.sqrt(interpolated_plane_locations[key][:,0]**2 + interpolated_plane_locations[key][:,1]**2 + interpolated_plane_locations[key][:,2]**2)
                initial_source_distance_m = source_distance_m[0] #m
                azimuth_deg = numpy.rad2deg(numpy.arctan2(interpolated_plane_locations[key][:,1],interpolated_plane_locations[key][:,0]))
                zenith_deg = numpy.rad2deg(numpy.arccos(interpolated_plane_locations[key][:,2]/source_distance_m))


    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






