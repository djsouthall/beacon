'''
This is intended to be maintained as a simple script that plots the unknown sources for a given deploy_index (defaulting
to default is set to None). 
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

print('IF USING TIME DELAYS FOR PULSERS, I TRUST CURRENT BY EYE VERSION')

# Site 1, Run 1507
#By eye unfiltered
site1_measured_time_delays_hpol =  [((0, 1), -40.31873095761775), ((0, 2), 105.45784514533067), ((0, 3), 30.135050738939075), ((1, 2), 145.61788076163322), ((1, 3), 70.58430198762721), ((2, 3), -75.18303704634458)]
site1_measured_time_delays_errors_hpol =  [((0, 1), 0.05390602875058418), ((0, 2), 0.044986501385526165), ((0, 3), 0.04766074318950998), ((1, 2), 0.028887890879186494), ((1, 3), 0.03252021044637105), ((2, 3), 0.03225428496045381)]
site1_measured_time_delays_vpol =  [((0, 1), -37.997470132440384), ((0, 2), 101.68917969316817), ((0, 3), 36.40724396999149), ((1, 2), 139.68684427937058), ((1, 3), 74.50112508391489), ((2, 3), -65.25487641338805)]
site1_measured_time_delays_errors_vpol =  [((0, 1), 0.03011827820152272), ((0, 2), 0.03019736343475925), ((0, 3), 0.031375107623472516), ((1, 2), 0.032447697682718275), ((1, 3), 0.03109667967812645), ((2, 3), 0.031271228554630597)]

# Site 2, Run 1509
#By eye with no filtering
site2_measured_time_delays_hpol =  [((0, 1), -82.36300952477696), ((0, 2), 40.532241178254594), ((0, 3), -44.62321034858617), ((1, 2), 122.63983003115614), ((1, 3), 37.84008150548918), ((2, 3), -85.10254711335975)]
site2_measured_time_delays_errors_hpol =  [((0, 1), 0.06805633356582455), ((0, 2), 0.06374115194036371), ((0, 3), 0.061932184861914215), ((1, 2), 0.05385988241825421), ((1, 3), 0.06333087652874915), ((2, 3), 0.04576242749045018)]
site2_measured_time_delays_vpol =  [((0, 1), -79.95386249566438), ((0, 2), 36.76801256475748), ((0, 3), -38.345785064895686), ((1, 2), 116.64786424329235), ((1, 3), 41.7113458359634), ((2, 3), -75.08797122311996)]
site2_measured_time_delays_errors_vpol =  [((0, 1), 0.042075284218763505), ((0, 2), 0.04115999677119886), ((0, 3), 0.03731577983391688), ((1, 2), 0.04697815701549882), ((1, 3), 0.037159371366881265), ((2, 3), 0.03363016431350404)]

# Site 3, Run 1511
#By eye with no filtering
site3_measured_time_delays_hpol =  [((0, 1), -94.60387463826103), ((0, 2), -143.57675835202508), ((0, 3), -177.08153009135634), ((1, 2), -49.07507885269845), ((1, 3), -82.25996187697748), ((2, 3), -33.23172107457927)]
site3_measured_time_delays_errors_hpol =  [((0, 1), 0.050722621463835826), ((0, 2), 0.0668571644522086), ((0, 3), 0.054427178270967415), ((1, 2), 0.0460384615661204), ((1, 3), 0.03794985949273057), ((2, 3), 0.06075860733801835)]
site3_measured_time_delays_vpol =  [((0, 1), -92.0411188138037), ((0, 2), -147.12574991663104), ((0, 3), -172.41237810521784), ((1, 2), -55.093716852263405), ((1, 3), -80.26283429217344), ((2, 3), -25.26536153179327)]
site3_measured_time_delays_errors_vpol =  [((0, 1), 0.026686488662462645), ((0, 2), 0.029657613098382718), ((0, 3), 0.03176185523762114), ((1, 2), 0.029839183454581412), ((1, 3), 0.02891810009497307), ((2, 3), 0.030919678248785214)]

print('IF USING HILBERT TIME DELAYS FOR PULSERS, I DONT TRUST CURRENT VERSION')

# Site 1, Run 1507
#Using hilbert enevelopes:
site1_measured_time_delays_hilbert_hpol =  [((0, 1), -58.987476762558636), ((0, 2), 85.65448689743364), ((0, 3), 8.320951597921127), ((1, 2), 144.48438341182535), ((1, 3), 67.16676381542445), ((2, 3), -77.3644536438967)]
site1_measured_time_delays_hilbert_errors_hpol =  [((0, 1), 0.8377448773505135), ((0, 2), 0.3594185551190815), ((0, 3), 0.594635989964824), ((1, 2), 0.7881229445755795), ((1, 3), 0.655179094065839), ((2, 3), 0.5535397424928563)]
site1_measured_time_delays_hilbert_vpol =  [((0, 1), -41.7904293292381), ((0, 2), 97.99566415134592), ((0, 3), 32.98832571271674), ((1, 2), 139.79330184195436), ((1, 3), 74.75836427079234), ((2, 3), -65.04301010744241)]
site1_measured_time_delays_hilbert_errors_vpol =  [((0, 1), 0.4735699881401701), ((0, 2), 0.4590523544465075), ((0, 3), 0.42013177070803437), ((1, 2), 0.6012971718190752), ((1, 3), 0.47273363718795347), ((2, 3), 0.4704350387698477)]

# Site 2, Run 1509
#Using hilbert envelopes:
site2_measured_time_delays_hilbert_hpol =  [((0, 1), -100.60440248833073), ((0, 2), 18.252306266584625), ((0, 3), -68.1176724682709), ((1, 2), 118.43971845333104), ((1, 3), 32.27161618415478), ((2, 3), -86.74537359631644)]
site2_measured_time_delays_hilbert_errors_hpol =  [((0, 1), 0.8957178121617457), ((0, 2), 1.016165870957765), ((0, 3), 0.7663702873722121), ((1, 2), 1.400179720774765), ((1, 3), 1.0146094407051107), ((2, 3), 0.9757739190044133)]
site2_measured_time_delays_hilbert_vpol =  [((0, 1), -83.84057479839511), ((0, 2), 29.220940734890714), ((0, 3), -42.61904475811045), ((1, 2), 113.34806469515324), ((1, 3), 41.375706449073355), ((2, 3), -71.86012025765733)]
site2_measured_time_delays_hilbert_errors_vpol =  [((0, 1), 0.6408498537052976), ((0, 2), 0.6485697868214796), ((0, 3), 0.5952727184008998), ((1, 2), 0.7658668502667069), ((1, 3), 0.6880201767149771), ((2, 3), 0.6428620412909642)]

# Site 3, Run 1511
#Using hilbert envelopes:
site3_measured_time_delays_hilbert_hpol =  [((0, 1), -109.18511106059067), ((0, 2), -158.10857451624145), ((0, 3), -191.41010115062258), ((1, 2), -48.875126824337244), ((1, 3), -82.20492349641286), ((2, 3), -33.23548270662396)]
site3_measured_time_delays_hilbert_errors_hpol =  [((0, 1), 0.052428389377522186), ((0, 2), 0.07196071815111829), ((0, 3), 0.057841508523156095), ((1, 2), 0.05836839659836438), ((1, 3), 0.06144960035833733), ((2, 3), 0.08874875693196994)]
site3_measured_time_delays_hilbert_vpol =  [((0, 1), -95.30265032203047), ((0, 2), -150.3612627337264), ((0, 3), -175.88155062409197), ((1, 2), -55.06965132126639), ((1, 3), -80.55266965411217), ((2, 3), -25.524379719980214)]
site3_measured_time_delays_hilbert_errors_vpol =  [((0, 1), 0.03654878810011606), ((0, 2), 0.044076823274307404), ((0, 3), 0.04115323593715623), ((1, 2), 0.039281585194972476), ((1, 3), 0.04240838492773853), ((2, 3), 0.04449409255674249)]







if __name__ == '__main__':
    try:
        plt.close('all')
        if len(sys.argv) == 2:
            if str(sys.argv[1]) in ['vpol', 'hpol']:
                mode = str(sys.argv[1])
            else:
                print('Given mode not in options.  Defaulting to hpol')
                mode = 'hpol'
            deploy_index = info.returnDefaultDeploy()
        elif len(sys.argv) == 3:
            if str(sys.argv[1]) in ['vpol', 'hpol']:
                mode = str(sys.argv[1])
            else:
                print('Given mode not in options.  Defaulting to hpol')
                mode = 'hpol'

            deploy_index = int(sys.argv[2])
        else:
            print('No mode given.  Defaulting to hpol')
            mode = 'hpol'
            deploy_index = info.returnDefaultDeploy()
        print('Plotting for Deploy Index %i '%deploy_index)


        #### PULSERS ####
        if True:
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

        if True:
            included_valley_sources = [ 'Tonopah AFS GATR Site',\
                                        'Tonopah Vortac',\
                                        'Tonopah Airport Antenna',\
                                        'Dyer Cell Tower',\
                                        'West Dyer Substation',\
                                        'East Dyer Substation',\
                                        'Oasis',\
                                        'Beatty Substation',\
                                        'Palmetto Cell Tower',\
                                        'Cedar Peak',\
                                        'Dome Thing',\
                                        'Goldfield Hill Tower',\
                                        'Silver Peak Substation',\
                                        'Silver Peak Town Antenna',\
                                        'Silver Peak Lithium Mine',\
                                        'Past SP Substation']
        else:
            included_valley_sources = []

        #### AIRPLANES ####
        plot_animated_airplane = True #Otherwise plots first event from each plane.  
        if True:
            included_airplanes =      [ '1728-62026',\
                                        '1773-14413',\
                                        '1773-63659',\
                                        '1774-178',\
                                        '1774-88800',\
                                        '1783-28830',\
                                        '1784-7166']


        else:
            included_airplanes =      []

        plot_predicted_time_shifts = False
        plot_airplane_tracks = True
        plot_time_delay_calculations = False
        plot_time_delays_on_maps = True
        plot_expected_direction = True
        limit_events = 10 #Number of events use for time delay calculation


        plot_residuals = False
        plot_histograms = False
        iterate_sub_baselines = 3 #The lower this is the higher the time it will take to plot.  Does combinatoric subsets of baselines with this length. 

        final_corr_length = 2**17
        cor_upsample = final_corr_length

        crit_freq_low_pass_MHz = None#[80,70,70,70,70,70,60,70]#90#
        low_pass_filter_order = None#[0,8,8,8,10,8,3,8]#8#

        crit_freq_high_pass_MHz = None#70#None#60
        high_pass_filter_order = None#6#None#8

        sine_subtract = False
        sine_subtract_min_freq_GHz = 0.03
        sine_subtract_max_freq_GHz = 0.13
        sine_subtract_percent = 0.03

        waveform_index_range = (None,None)

        apply_phase_response = True
        hilbert = False

        included_antennas_lumped = [0,1,2,3] #If an antenna is not in this list then it will not be included in the chi^2 (regardless of if it is fixed or not)  Lumped here imlies that antenna 0 in this list means BOTH channels 0 and 1 (H and V of crossed dipole antenna 0).
        included_antennas_channels = numpy.concatenate([[2*i,2*i+1] for i in included_antennas_lumped])
        include_baselines = [0,1,2,3,4,5] #Basically sets the starting condition of which baselines to include, then the lumped channels and antennas will cut out further from that.  The above options of excluding antennas will override this to exclude baselines, but if both antennas are included but the baseline is not then it will not be included.  Overwritten when antennas removed.

        #### PULSERS ####

        if len(included_pulsers) > 0:

            map_resolution = 0.25 #degrees
            range_phi_deg = (-90, 90)
            range_theta_deg = (0,180)
            n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
            n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
            
            pulser_time_delay_dict = {}
            pulser_time_delay_error_dict = {}

            if mode == 'hpol':
                if hilbert == True:
                    pulser_time_delay_dict['run1507'] = site1_measured_time_delays_hilbert_hpol
                    pulser_time_delay_error_dict['run1507'] = site1_measured_time_delays_hilbert_errors_hpol

                    pulser_time_delay_dict['run1509'] = site2_measured_time_delays_hilbert_hpol
                    pulser_time_delay_error_dict['run1509'] = site2_measured_time_delays_hilbert_errors_hpol

                    pulser_time_delay_dict['run1511'] = site3_measured_time_delays_hilbert_hpol
                    pulser_time_delay_error_dict['run1511'] = site3_measured_time_delays_hilbert_errors_hpol
                else:
                    pulser_time_delay_dict['run1507'] = site1_measured_time_delays_hpol
                    pulser_time_delay_error_dict['run1507'] = site1_measured_time_delays_errors_hpol

                    pulser_time_delay_dict['run1509'] = site2_measured_time_delays_hpol
                    pulser_time_delay_error_dict['run1509'] = site2_measured_time_delays_errors_hpol

                    pulser_time_delay_dict['run1511'] = site3_measured_time_delays_hpol
                    pulser_time_delay_error_dict['run1511'] = site3_measured_time_delays_errors_hpol
            else:
                if hilbert == True:
                    pulser_time_delay_dict['run1507'] = site1_measured_time_delays_hilbert_vpol
                    pulser_time_delay_error_dict['run1507'] = site1_measured_time_delays_hilbert_errors_vpol

                    pulser_time_delay_dict['run1509'] = site2_measured_time_delays_hilbert_vpol
                    pulser_time_delay_error_dict['run1509'] = site2_measured_time_delays_hilbert_errors_vpol

                    pulser_time_delay_dict['run1511'] = site3_measured_time_delays_hilbert_vpol
                    pulser_time_delay_error_dict['run1511'] = site3_measured_time_delays_hilbert_errors_vpol
                else:
                    pulser_time_delay_dict['run1507'] = site1_measured_time_delays_vpol
                    pulser_time_delay_error_dict['run1507'] = site1_measured_time_delays_errors_vpol

                    pulser_time_delay_dict['run1509'] = site2_measured_time_delays_vpol
                    pulser_time_delay_error_dict['run1509'] = site2_measured_time_delays_errors_vpol

                    pulser_time_delay_dict['run1511'] = site3_measured_time_delays_vpol
                    pulser_time_delay_error_dict['run1511'] = site3_measured_time_delays_errors_vpol



            pulser_locations_ENU = info.loadPulserLocationsENU(deploy_index=deploy_index)
            known_pulser_ids = info.loadPulserEventids(remove_ignored=True)


            for pulser_key in included_pulsers:
                run = int(pulser_key.replace('run',''))
                reader = Reader(datapath,run)

                if plot_predicted_time_shifts:
                    if run == 1507:
                            _waveform_index_range = (1500,2000) #Looking at the later bit of the waveform only, 10000 will cap off.  
                    elif run == 1509:
                            _waveform_index_range = (2500,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
                    elif run == 1511:
                            _waveform_index_range = (1250,1750) #Looking at the later bit of the waveform only, 10000 will cap off.  
                    reader = Reader(datapath,run)
                    tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=_waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
                    if sine_subtract:
                        tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
                    _time_delays = numpy.append(numpy.array(pulser_time_delay_dict['run%i'%run])[:,1],numpy.array(pulser_time_delay_dict['run%i'%run])[:,1]) #Just duplicating, I am only plotting the ones for this polarization anyways.
                    if mode == 'hpol':
                        _eventid = numpy.random.choice(known_pulser_ids['run%i'%run]['hpol'])
                        _fig, _ax = tdc.plotEvent(_eventid, channels=[0,2,4,6], apply_filter=True, hilbert=hilbert, sine_subtract=True, apply_tukey=None, additional_title_text='Sample Event shifted by input time delays', time_delays=_time_delays)
                    else:
                        _eventid = numpy.random.choice(known_pulser_ids['run%i'%run]['vpol'])
                        _fig, _ax = tdc.plotEvent(_eventid, channels=[1,3,5,7], apply_filter=True, hilbert=hilbert, sine_subtract=True, apply_tukey=None, additional_title_text='Sample Event shifted by input time delays', time_delays=_time_delays)

                #Calculate old and new geometries
                #Distance needed when calling correlator, as it uses that distance.
                pulser_ENU = numpy.array([pulser_locations_ENU[pulser_key][0] , pulser_locations_ENU[pulser_key][1] , pulser_locations_ENU[pulser_key][2]])
                distance_m = numpy.linalg.norm(pulser_ENU)
                zenith_deg = numpy.rad2deg(numpy.arccos(pulser_ENU[2]/distance_m))
                elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(pulser_ENU[2]/distance_m))
                azimuth_deg = numpy.rad2deg(numpy.arctan2(pulser_ENU[1],pulser_ENU[0]))
                
                cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=distance_m, deploy_index=deploy_index)
                cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                if plot_expected_direction == False:
                    zenith_deg = None
                    azimuth_deg = None

                if plot_time_delays_on_maps:
                    td_dict = {mode:{'[0, 1]' :  [pulser_time_delay_dict[pulser_key][0][1]], '[0, 2]' : [pulser_time_delay_dict[pulser_key][1][1]], '[0, 3]' : [pulser_time_delay_dict[pulser_key][2][1]], '[1, 2]' : [pulser_time_delay_dict[pulser_key][3][1]], '[1, 3]' : [pulser_time_delay_dict[pulser_key][4][1]], '[2, 3]' : [pulser_time_delay_dict[pulser_key][5][1]]}}
                else:
                    td_dict = {}

                eventid = numpy.random.choice(known_pulser_ids[pulser_key][mode]) #For plotting single map
                
                mean_corr_values, fig, ax = cor.map(eventid, mode, include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=False, radius=1.0,zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90], interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg, time_delay_dict=td_dict,window_title=pulser_key)

                if plot_histograms:
                    map_resolution = 0.1 #degrees
                    range_phi_deg=(azimuth_deg - 10, azimuth_deg + 10)
                    range_theta_deg=(zenith_deg - 10,zenith_deg + 10)
                    n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
                    n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
                    
                    cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tupulser_key=False, sine_subtract=True,map_source_distance_m=distance_m)
                    cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    eventids = numpy.sort(numpy.random.choice(known_pulser_ids[pulser_key][mode],min(limit_events,len(known_pulser_ids[pulser_key][mode])))) #For plotting multiple events in a histogram
                    hist = cor.histMapPeak(eventids, mode, plot_map=True, hilbert=False, max_method=0, use_weight=False, mollweide=False, center_dir='E', radius=1.0,zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90],circle_zenith=zenith_deg, circle_az=azimuth_deg, window_title='Hist ' + pulser_key, include_baselines=include_baselines,iterate_sub_baselines=iterate_sub_baselines)

        #### VALLEY SOURCES ####

        if len(included_valley_sources) > 0:

            map_resolution = 0.25 #degrees
            range_phi_deg = (-90, 90)
            range_theta_deg = (0,180)
            n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
            n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
            
            valley_sources_time_delay_dict = {}
            valley_sources_time_delay_error_dict = {}

            if mode == 'hpol':
                if hilbert == True:
                    valley_sources_time_delay_dict['run1507'] = site1_measured_time_delays_hilbert_hpol
                    valley_sources_time_delay_error_dict['run1507'] = site1_measured_time_delays_hilbert_errors_hpol

                    valley_sources_time_delay_dict['run1509'] = site2_measured_time_delays_hilbert_hpol
                    valley_sources_time_delay_error_dict['run1509'] = site2_measured_time_delays_hilbert_errors_hpol

                    valley_sources_time_delay_dict['run1511'] = site3_measured_time_delays_hilbert_hpol
                    valley_sources_time_delay_error_dict['run1511'] = site3_measured_time_delays_hilbert_errors_hpol
                else:
                    valley_sources_time_delay_dict['run1507'] = site1_measured_time_delays_hpol
                    valley_sources_time_delay_error_dict['run1507'] = site1_measured_time_delays_errors_hpol

                    valley_sources_time_delay_dict['run1509'] = site2_measured_time_delays_hpol
                    valley_sources_time_delay_error_dict['run1509'] = site2_measured_time_delays_errors_hpol

                    valley_sources_time_delay_dict['run1511'] = site3_measured_time_delays_hpol
                    valley_sources_time_delay_error_dict['run1511'] = site3_measured_time_delays_errors_hpol
            else:
                if hilbert == True:
                    valley_sources_time_delay_dict['run1507'] = site1_measured_time_delays_hilbert_vpol
                    valley_sources_time_delay_error_dict['run1507'] = site1_measured_time_delays_hilbert_errors_vpol

                    valley_sources_time_delay_dict['run1509'] = site2_measured_time_delays_hilbert_vpol
                    valley_sources_time_delay_error_dict['run1509'] = site2_measured_time_delays_hilbert_errors_vpol

                    valley_sources_time_delay_dict['run1511'] = site3_measured_time_delays_hilbert_vpol
                    valley_sources_time_delay_error_dict['run1511'] = site3_measured_time_delays_hilbert_errors_vpol
                else:
                    valley_sources_time_delay_dict['run1507'] = site1_measured_time_delays_vpol
                    valley_sources_time_delay_error_dict['run1507'] = site1_measured_time_delays_errors_vpol

                    valley_sources_time_delay_dict['run1509'] = site2_measured_time_delays_vpol
                    valley_sources_time_delay_error_dict['run1509'] = site2_measured_time_delays_errors_vpol

                    valley_sources_time_delay_dict['run1511'] = site3_measured_time_delays_vpol
                    valley_sources_time_delay_error_dict['run1511'] = site3_measured_time_delays_errors_vpol

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

                time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(roi_eventids,align_method=0,hilbert=hilbert,plot=plot_time_delay_calculations, sine_subtract=sine_subtract)

                if mode == 'hpol':
                    valley_sources_time_delay_dict[valley_source_key] = numpy.mean(time_shifts[0:6,:],axis=1) #perhaps I should be fitting.  Assuming only using sources with consistent time delays
                else:
                    valley_sources_time_delay_dict[valley_source_key] = numpy.mean(time_shifts[6:12,:],axis=1)
                
                all_tds = numpy.mean(time_shifts,axis=1)

                eventid = numpy.random.choice(roi_eventids) #Used for all 1 off event plotting here
                if plot_predicted_time_shifts:
                        if mode == 'hpol':
                            _fig, _ax = tdc.plotEvent(eventid, channels=[0,2,4,6], apply_filter=True, hilbert=hilbert, sine_subtract=True, apply_tukey=None, additional_title_text='%s Sample Event shifted by input time delays'%valley_source_key, time_delays=all_tds)
                        else:
                            _fig, _ax = tdc.plotEvent(eventid, channels=[1,3,5,7], apply_filter=True, hilbert=hilbert, sine_subtract=True, apply_tukey=None, additional_title_text='%s Sample Event shifted by input time delays'%valley_source_key, time_delays=all_tds)
                
                # print('valley_sources_time_delay_dict[%s] = '%valley_source_key)
                # print(valley_sources_time_delay_dict[valley_source_key])

                if plot_time_delays_on_maps:
                    td_dict = {mode:{'[0, 1]' :  [valley_sources_time_delay_dict[valley_source_key][0]], '[0, 2]' : [valley_sources_time_delay_dict[valley_source_key][1]], '[0, 3]' : [valley_sources_time_delay_dict[valley_source_key][2]], '[1, 2]' : [valley_sources_time_delay_dict[valley_source_key][3]], '[1, 3]' : [valley_sources_time_delay_dict[valley_source_key][4]], '[2, 3]' : [valley_sources_time_delay_dict[valley_source_key][5]]}}
                else:
                    td_dict = {}


                #Calculate old and new geometries
                #Distance needed when calling correlator, as it uses that distance.
                valley_source_ENU = numpy.array([sources_ENU[valley_source_key][0] , sources_ENU[valley_source_key][1] , sources_ENU[valley_source_key][2]])
                distance_m = numpy.linalg.norm(valley_source_ENU)
                zenith_deg = numpy.rad2deg(numpy.arccos(valley_source_ENU[2]/distance_m))
                elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(valley_source_ENU[2]/distance_m))
                azimuth_deg = numpy.rad2deg(numpy.arctan2(valley_source_ENU[1],valley_source_ENU[0]))
                
                cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=distance_m, deploy_index=deploy_index)
                cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                if plot_expected_direction == False:
                    zenith_deg = None
                    azimuth_deg = None
                
                mean_corr_values, fig, ax = cor.map(eventid, mode, include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=False, radius=1.0,zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90], interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg, time_delay_dict=td_dict,window_title=valley_source_key)

                if plot_histograms:
                    map_resolution = 0.1 #degrees
                    range_phi_deg=(azimuth_deg - 10, azimuth_deg + 10)
                    range_theta_deg=(zenith_deg - 10,zenith_deg + 10)
                    n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
                    n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
                    
                    cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tuvalley_source_key=False, sine_subtract=True,map_source_distance_m=distance_m)
                    cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    eventids = numpy.sort(numpy.random.choice(roi_eventids,min(limit_events,len(roi_eventids)))) #For plotting multiple events in a histogram
                    hist = cor.histMapPeak(eventids, mode, plot_map=True, hilbert=False, max_method=0, use_weight=False, mollweide=False, center_dir='E', radius=1.0,zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90],circle_zenith=zenith_deg, circle_az=azimuth_deg, window_title='Hist ' + valley_source_key, include_baselines=include_baselines,iterate_sub_baselines=iterate_sub_baselines)


        #### AIRPLANES ####

        if len(included_airplanes) > 0:

            map_resolution = 0.5 #degrees
            range_phi_deg = (-180, 180)
            range_theta_deg = (0,180)
            n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
            n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
            try_to_use_precalculated_time_delays_airplanes = True
            try_to_use_precalculated_time_delays_airplanes_but_just_as_guess_for_real_time_delays_why_is_this_so_long = False

            print('Loading known plane locations.')
            known_planes, calibrated_trigtime, output_tracks = pt.getKnownPlaneTracks(ignore_planes=[]) # ['1728-62026','1773-14413','1773-63659','1774-88800','1783-28830','1784-7166']#'1774-88800','1728-62026'
            origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
            antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=deploy_index)
            if mode == 'hpol':
                antennas_phase_start = antennas_phase_hpol
            else:
                antennas_phase_start = antennas_phase_vpol

            plane_polys = {}
            interpolated_plane_locations = {}
            measured_plane_time_delays = {}

            if plot_airplane_tracks == True:
                plane_fig = plt.figure()
                plane_fig.canvas.set_window_title('3D Plane Tracks')
                plane_ax = plane_fig.add_subplot(111, projection='3d')
                plane_ax.scatter(0,0,0,label='Antenna 0',c='k')
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            if plot_animated_airplane == True:
                cors = [] #So animations don't reset
            for index, key in enumerate(list(calibrated_trigtime.keys())):
                if key not in included_airplanes:
                    continue

                pair_cut = numpy.array([pair in known_planes[key]['baselines'][mode] for pair in [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]] ]) #Checks which pairs are worth looping over.
                pair_cut = numpy.logical_and(pair_cut,numpy.isin([0,1,2,3,4,5],include_baselines)) #To include the ability to disable baselines from this scripts settings.

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
                

                if plot_airplane_tracks == True:
                    #Alpha ranges from 0.5 to 1, where 0.5 is the earliest times and 1 is the later times.
                    #alpha = 0.5 + (calibrated_trigtime[key] - min(calibrated_trigtime[key]))/(2*(max(calibrated_trigtime[key]) - min(calibrated_trigtime[key])))
                    arrow_index = len(enu[0][unique_enu_indices])//2
                    arrow_dir = numpy.array([(enu[0][unique_enu_indices][arrow_index+1] - enu[0][unique_enu_indices][arrow_index])/1000.0,(enu[1][unique_enu_indices][arrow_index+1] - enu[1][unique_enu_indices][arrow_index])/1000.0,(enu[2][unique_enu_indices][arrow_index+1] - enu[2][unique_enu_indices][arrow_index])/1000.0])
                    arrow_dir = 3*arrow_dir/numpy.linalg.norm(arrow_dir)
                    plane_ax.plot(enu[0][unique_enu_indices]/1000.0,enu[1][unique_enu_indices]/1000.0,enu[2][unique_enu_indices]/1000.0,label=key + ' : ' + known_planes[key]['known_flight'],color=colors[index])
                    plane_ax.quiver(enu[0][unique_enu_indices][arrow_index]/1000.0,enu[1][unique_enu_indices][arrow_index]/1000.0,enu[2][unique_enu_indices][arrow_index]/1000.0, arrow_dir[0],arrow_dir[1],arrow_dir[2],color=colors[index])
                    plane_ax.scatter(interpolated_plane_locations[key][:,0]/1000.0,interpolated_plane_locations[key][:,1]/1000.0,interpolated_plane_locations[key][:,2]/1000.0,label=key + ' : ' + known_planes[key]['known_flight'],color=colors[index])

                    plt.grid()
                    plt.show()


                if try_to_use_precalculated_time_delays_airplanes == True and numpy.logical_and('time_delays' in list(known_planes[key].keys()),'max_corrs' in list(known_planes[key].keys())):
                    print('Using precalculated time delays from info.py')
                    measured_plane_time_delays[key] = known_planes[key]['time_delays'][mode].T[pair_cut]

                elif try_to_use_precalculated_time_delays_airplanes_but_just_as_guess_for_real_time_delays_why_is_this_so_long == True:

                    guess_time_delays = numpy.vstack((known_planes[key]['time_delays']['hpol'].T,known_planes[key]['time_delays']['vpol'].T)).T

                    print('Calculating time delays from info.py')
                    run = int(key.split('-')[0])
                    reader = Reader(datapath,run)
                    tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
                    if sine_subtract:
                        tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    
                    time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids,align_method=10,hilbert=hilbert, align_method_10_estimates=guess_time_delays, align_method_10_window_ns=8, sine_subtract=sine_subtract)

                    if mode == 'hpol':
                        measured_plane_time_delays[key] = time_shifts[0:6,:][pair_cut]
                    else:
                        measured_plane_time_delays[key] = time_shifts[6:12,:][pair_cut]

                else:
                    print('Calculating time delays from info.py')
                    run = int(key.split('-')[0])
                    reader = Reader(datapath,run)
                    tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
                    if sine_subtract:
                        tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
                    eventids = known_planes[key]['eventids'][:,1]
                    time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids,align_method=0,hilbert=hilbert, sine_subtract=sine_subtract)

                    if mode == 'hpol':
                        measured_plane_time_delays[key] = time_shifts[0:6,:][pair_cut]
                    else:
                        measured_plane_time_delays[key] = time_shifts[6:12,:][pair_cut]



                if plot_expected_direction == False:
                    zenith_deg = None
                    azimuth_deg = None
                    source_distance_m = None
                    initial_source_distance_m = 1e6 #m
                else:
                    source_distance_m = numpy.sqrt(interpolated_plane_locations[key][:,0]**2 + interpolated_plane_locations[key][:,1]**2 + interpolated_plane_locations[key][:,2]**2)
                    initial_source_distance_m = source_distance_m[0] #m
                    azimuth_deg = numpy.rad2deg(numpy.arctan2(interpolated_plane_locations[key][:,1],interpolated_plane_locations[key][:,0]))
                    zenith_deg = numpy.rad2deg(numpy.arccos(interpolated_plane_locations[key][:,2]/source_distance_m))

                cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=initial_source_distance_m, deploy_index=deploy_index)
                cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                if plot_animated_airplane == True:
                    cor.animatedMap(eventids, mode, 'deploy_index_%i'%deploy_index,include_baselines=include_baselines,map_source_distance_m = source_distance_m,  plane_zenith=zenith_deg,plane_az=azimuth_deg,hilbert=False, max_method=None,center_dir='W',save=False,dpi=300)
                else:
                    event_index = 0
                    td_dict = {mode:{'[0, 1]' : [ measured_plane_time_delays[key][0][event_index]], '[0, 2]' : [measured_plane_time_delays[key][1][event_index]], '[0, 3]' : [measured_plane_time_delays[key][2][event_index]], '[1, 2]' : [measured_plane_time_delays[key][3][event_index]], '[1, 3]' : [measured_plane_time_delays[key][4][event_index]], '[2, 3]' : [measured_plane_time_delays[key][5][event_index]]}}
                    cor.overwriteSourceDistance(source_distance_m[event_index], verbose=False, suppress_time_delay_calculations=False)
                    mean_corr_values, fig, ax = cor.map(eventids[event_index], mode, include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=False, radius=1.0,zenith_cut_ENU=[0,90],zenith_cut_array_plane=[0,95], interactive=True,circle_zenith=zenith_deg[event_index], circle_az=azimuth_deg[event_index], time_delay_dict=td_dict,window_title=key)

                cors.append(cor)
            if plot_airplane_tracks == True:
                plane_ax.legend(loc='upper right')
                plane_ax.set_xlabel('East (km)',linespacing=10)
                plane_ax.set_ylabel('North (km)',linespacing=10)
                plane_ax.set_zlabel('Up (km)',linespacing=10)
                plane_ax.dist = 10

            
    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






