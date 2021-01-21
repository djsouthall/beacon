'''
This script assumes the antenna positions have already roughly been set/determined
using find_phase_centers.py  .  It will assume these as starting points of the antennas
and then vary all 4 of their locations to match measured time delays from RFI sources in the valley. 
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
        if len(sys.argv) == 2:
            if str(sys.argv[1]) in ['vpol', 'hpol']:
                mode = str(sys.argv[1])
            else:
                print('Given mode not in options.  Defaulting to hpol')
                mode = 'hpol'
        else:
            print('No mode given.  Defaulting to hpol')
            mode = 'hpol'

        #palmetto is a reflection?
        #Need to label which ones work for vpol and hpol
        #use_sources = ['Solar Plant','Quarry Substation','Tonopah KTPH','Dyer Cell Tower','Beatty Airport Vortac','Palmetto Cell Tower','Cedar Peak','Goldfield Hill Tower','Goldield Town Tower','Goldfield KGFN-FM','Silver Peak Town Antenna','Silver Peak Lithium Mine','Past SP Substation','Silver Peak Substation']
        use_sources = ['Quarry Substation','Tonopah KTPH','Silver Peak Substation','Palmetto Cell Tower','Beatty Airport Vortac']#['Quarry Substation','Palmetto Cell Tower','Silver Peak Substation']#['Solar Plant']#['Quarry Substation']#,'Beatty Airport Vortac','Palmetto Cell Tower','Silver Peak Substation']#['Quarry Substation','Beatty Airport Vortac','Palmetto Cell Tower','Silver Peak Substation']

        impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        map_direction_dset_key = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1'

        plot_residuals = False
        #plot_time_delays = False

        final_corr_length = 2**16

        crit_freq_low_pass_MHz = 100 #This new pulser seems to peak in the region of 85 MHz or so
        low_pass_filter_order = 8

        crit_freq_high_pass_MHz = None
        high_pass_filter_order = None

        sine_subtract = True
        sine_subtract_min_freq_GHz = 0.03
        sine_subtract_max_freq_GHz = 0.09
        sine_subtract_percent = 0.03

        waveform_index_range = (None,None)#(150,400)

        apply_phase_response = True
        hilbert = False

        #Limits 
        cable_delay_guess_range = None #ns
        antenna_position_guess_range = 3 #Limit to how far from input phase locations to limit the parameter space to
        fix_ant0_x = True
        fix_ant0_y = True
        fix_ant0_z = True
        fix_ant1_x = False
        fix_ant1_y = False
        fix_ant1_z = False
        fix_ant2_x = False
        fix_ant2_y = False
        fix_ant2_z = False#mode == 'vpol' #Most vpols don't have antenna 0, so the array would float in optimization otherwise.
        fix_ant3_x = False
        fix_ant3_y = False
        fix_ant3_z = False
        fix_cable_delay0 = True
        fix_cable_delay1 = True
        fix_cable_delay2 = True
        fix_cable_delay3 = True

        #I think adding an absolute time offset for each antenna and letting that vary could be interesting.  It could be used to adjust the cable delays.
        cable_delays = info.loadCableDelays()[mode]
        print('Potential RFI Source Locations.')
        sources_ENU, data_slicer_cut_dict = info.loadValleySourcesENU()

        for key in list(sources_ENU.keys()):
            if key in use_sources:
                print('Using key %s'%key)
                continue
            else:
                del sources_ENU[key]
                del data_slicer_cut_dict[key]

        #Note the above ROI assumes you have already cut out events that are below a certain correlation with a template.
        # ds.addROI('Simple Template V > 0.7',{'cr_template_search_v':[0.7,1.0]})# Adding 2 ROI in different rows and appending as below allows for "OR" instead of "AND"
        # ds.addROI('Simple Template H > 0.7',{'cr_template_search_h':[0.7,1.0]})
        # #Done for OR condition
        # _eventids = numpy.sort(numpy.unique(numpy.append(ds.getCutsFromROI('Simple Template H > 0.7',load=False,save=False),ds.getCutsFromROI('Simple Template V > 0.7',load=False,save=False))))
        # roi_eventids = numpy.intersect1d(ds.getCutsFromROI(roi_key),_eventids)
        source_event_run = 1650
        origin = info.loadAntennaZeroLocation()
        antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU()

        if mode == 'hpol':
            antennas_phase_start = antennas_phase_hpol
        else:
            antennas_phase_start = antennas_phase_vpol

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        #DETERMINE THE TIME DELAYS TO BE USED IN THE ACTUAL CALCULATION

        print('Calculating time delays from info.py')
        run = 1650
        reader = Reader(datapath,run)
        tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
        if sine_subtract:
            tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
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
            ds.addROI(source_key,cut_dict)
            roi_eventids = numpy.intersect1d(ds.getCutsFromROI(source_key),_eventids)
            roi_impulsivity = ds.getDataFromParam(roi_eventids,'impulsivity_h')
            roi_impulsivity_sort = numpy.argsort(roi_impulsivity)[::-1] #Reverse sorting so high numbers are first.
            limit_events = 10
            if len(roi_eventids) > limit_events:
                print('LIMITING TIME DELAY CALCULATION TO %i MOST IMPULSIVE EVENTS'%limit_events)
                roi_eventids = numpy.sort(roi_eventids[roi_impulsivity_sort[0:limit_events]])
        
            print('Calculating time delays for %s'%source_key)
            time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(roi_eventids,align_method=0,hilbert=hilbert,plot=False, sine_subtract=sine_subtract)

            if mode == 'hpol':
                # for event_index, eventid in enumerate(roi_eventids):
                #     print(eventid)
                #     print(time_shifts[0:6,event_index])
                time_delay_dict[source_key] = numpy.mean(time_shifts[0:6,:],axis=1)
            else:
                time_delay_dict[source_key] = numpy.mean(time_shifts[6:12,:],axis=1)
            


        if cable_delay_guess_range is not None:

            limit_cable_delay0 = (cable_delays[0] - cable_delay_guess_range , cable_delays[0] + cable_delay_guess_range)
            limit_cable_delay1 = (cable_delays[1] - cable_delay_guess_range , cable_delays[1] + cable_delay_guess_range)
            limit_cable_delay2 = (cable_delays[2] - cable_delay_guess_range , cable_delays[2] + cable_delay_guess_range)
            limit_cable_delay3 = (cable_delays[3] - cable_delay_guess_range , cable_delays[3] + cable_delay_guess_range)
        else:
            limit_cable_delay0 = None#(cable_delays[0] , None)
            limit_cable_delay1 = None#(cable_delays[1] , None)
            limit_cable_delay2 = None#(cable_delays[2] , None)
            limit_cable_delay3 = None#(cable_delays[3] , None)

        if antenna_position_guess_range is not None:

            ant0_physical_limits_x = (antennas_phase_start[0][0] - antenna_position_guess_range ,antennas_phase_start[0][0] + antenna_position_guess_range)
            ant0_physical_limits_y = (antennas_phase_start[0][1] - antenna_position_guess_range ,antennas_phase_start[0][1] + antenna_position_guess_range)
            ant0_physical_limits_z = (antennas_phase_start[0][2] - antenna_position_guess_range ,antennas_phase_start[0][2] + antenna_position_guess_range)

            ant1_physical_limits_x = (antennas_phase_start[1][0] - antenna_position_guess_range ,antennas_phase_start[1][0] + antenna_position_guess_range)
            ant1_physical_limits_y = (antennas_phase_start[1][1] - antenna_position_guess_range ,antennas_phase_start[1][1] + antenna_position_guess_range)
            ant1_physical_limits_z = (antennas_phase_start[1][2] - antenna_position_guess_range ,antennas_phase_start[1][2] + antenna_position_guess_range)

            ant2_physical_limits_x = (antennas_phase_start[2][0] - antenna_position_guess_range ,antennas_phase_start[2][0] + antenna_position_guess_range)
            ant2_physical_limits_y = (antennas_phase_start[2][1] - antenna_position_guess_range ,antennas_phase_start[2][1] + antenna_position_guess_range)
            ant2_physical_limits_z = (antennas_phase_start[2][2] - antenna_position_guess_range ,antennas_phase_start[2][2] + antenna_position_guess_range)

            ant3_physical_limits_x = (antennas_phase_start[3][0] - antenna_position_guess_range ,antennas_phase_start[3][0] + antenna_position_guess_range)
            ant3_physical_limits_y = (antennas_phase_start[3][1] - antenna_position_guess_range ,antennas_phase_start[3][1] + antenna_position_guess_range)
            ant3_physical_limits_z = (antennas_phase_start[3][2] - antenna_position_guess_range ,antennas_phase_start[3][2] + antenna_position_guess_range)

        else:
            ant0_physical_limits_x = None#None 
            ant0_physical_limits_y = None#None
            ant0_physical_limits_z = None#None

            ant1_physical_limits_x = None#(None,0.0) #Forced west of 0
            ant1_physical_limits_y = None#None
            ant1_physical_limits_z = None#(10.0,None) #Forced above 0

            ant2_physical_limits_x = None#None
            ant2_physical_limits_y = None#(None,0.0) #Forced south of 0 
            ant2_physical_limits_z = (antennas_phase_start[2][2] - 2.5 ,antennas_phase_start[2][2] + 2.5)#None#None

            ant3_physical_limits_x = None#(None,0.0) #Forced West of 0
            ant3_physical_limits_y = None#(None,0.0) #Forced South of 0
            ant3_physical_limits_z = None#(10.0,None) #Forced above 0


        ##########
        # Define Chi^2
        ##########

        chi2_fig = plt.figure()
        chi2_fig.canvas.set_window_title('Initial Positions')
        chi2_ax = chi2_fig.add_subplot(111, projection='3d')
        chi2_ax.scatter(antennas_phase_start[0][0], antennas_phase_start[0][1], antennas_phase_start[0][2],c='r',alpha=0.5,label='Initial Ant0')
        chi2_ax.scatter(antennas_phase_start[1][0], antennas_phase_start[1][1], antennas_phase_start[1][2],c='g',alpha=0.5,label='Initial Ant1')
        chi2_ax.scatter(antennas_phase_start[2][0], antennas_phase_start[2][1], antennas_phase_start[2][2],c='b',alpha=0.5,label='Initial Ant2')
        chi2_ax.scatter(antennas_phase_start[3][0], antennas_phase_start[3][1], antennas_phase_start[3][2],c='m',alpha=0.5,label='Initial Ant3')

        chi2_ax.set_xlabel('East (m)',linespacing=10)
        chi2_ax.set_ylabel('North (m)',linespacing=10)
        chi2_ax.set_zlabel('Up (m)',linespacing=10)
        
        if False:
            for key, enu in sources_ENU.items():
                chi2_ax.scatter(enu[0], enu[1], enu[2],alpha=0.5,label=key)
        else:
            chi2_ax.dist = 10
        plt.legend()


        chi2_fig = plt.figure()
        chi2_fig.canvas.set_window_title('Both')
        chi2_ax = chi2_fig.add_subplot(111, projection='3d')
        chi2_ax.scatter(antennas_phase_start[0][0], antennas_phase_start[0][1], antennas_phase_start[0][2],c='r',alpha=0.5,label='Initial Ant0')
        chi2_ax.scatter(antennas_phase_start[1][0], antennas_phase_start[1][1], antennas_phase_start[1][2],c='g',alpha=0.5,label='Initial Ant1')
        chi2_ax.scatter(antennas_phase_start[2][0], antennas_phase_start[2][1], antennas_phase_start[2][2],c='b',alpha=0.5,label='Initial Ant2')
        chi2_ax.scatter(antennas_phase_start[3][0], antennas_phase_start[3][1], antennas_phase_start[3][2],c='m',alpha=0.5,label='Initial Ant3')

        pairs = numpy.array(list(itertools.combinations((0,1,2,3), 2)))
        def rawChi2(ant0_x, ant0_y, ant0_z,ant1_x, ant1_y, ant1_z, ant2_x, ant2_y, ant2_z, ant3_x, ant3_y, ant3_z, cable_delay0, cable_delay1, cable_delay2, cable_delay3):
            '''
            This is a chi^2 that loops over locations from potential RFI, calculating expected time delays for those locations.  Then
            it will compares those to the calculated time delays for suspected corresponding events.  
            '''
            try:
                #chi2_ax.scatter(ant0_x, ant0_y, ant0_z,label='Antenna 0',c='r',alpha=0.5)
                #chi2_ax.scatter(ant1_x, ant1_y, ant1_z,label='Antenna 1',c='g',alpha=0.5)
                #chi2_ax.scatter(ant2_x, ant2_y, ant2_z,label='Antenna 2',c='b',alpha=0.5)
                #chi2_ax.scatter(ant3_x, ant3_y, ant3_z,label='Antenna 3',c='m',alpha=0.5)

                #Calculate distances (already converted to ns) from pulser to each antenna
                chi_2 = 0.0

                for key in list(sources_ENU.keys()):
                        d0 = (numpy.sqrt((sources_ENU[key][0] - ant0_x)**2 + (sources_ENU[key][1] - ant0_y)**2 + (sources_ENU[key][2] - ant0_z)**2 )/c)*1.0e9 #ns
                        d1 = (numpy.sqrt((sources_ENU[key][0] - ant1_x)**2 + (sources_ENU[key][1] - ant1_y)**2 + (sources_ENU[key][2] - ant1_z)**2 )/c)*1.0e9 #ns
                        d2 = (numpy.sqrt((sources_ENU[key][0] - ant2_x)**2 + (sources_ENU[key][1] - ant2_y)**2 + (sources_ENU[key][2] - ant2_z)**2 )/c)*1.0e9 #ns
                        d3 = (numpy.sqrt((sources_ENU[key][0] - ant3_x)**2 + (sources_ENU[key][1] - ant3_y)**2 + (sources_ENU[key][2] - ant3_z)**2 )/c)*1.0e9 #ns

                        d = [d0,d1,d2,d3]

                        _cable_delays = [cable_delay0,cable_delay1,cable_delay2,cable_delay3]

                        for pair_index, pair in enumerate(pairs):
                            geometric_time_delay = (d[pair[0]] + _cable_delays[0]) - (d[pair[1]] + _cable_delays[1])
                            
                            vals = ((geometric_time_delay - time_delay_dict[key][pair_index])**2) #Assumes time delays are accurate
                            chi_2 += numpy.sum(vals)
                return chi_2
            except Exception as e:
                print('Error in rawChi2')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


        initial_step = 0.1 #m
        #-12 ft on pulser locations relative to antennas to account for additional mast elevation.
        
        
        #rawChi2(ant0_x=antennas_phase_start[0][0], ant0_y=antennas_phase_start[0][1], ant0_z=antennas_phase_start[0][2], ant1_x=antennas_phase_start[1][0], ant1_y=antennas_phase_start[1][1], ant1_z=antennas_phase_start[1][2], ant2_x=antennas_phase_start[2][0], ant2_y=antennas_phase_start[2][1], ant2_z=antennas_phase_start[2][2], ant3_x=antennas_phase_start[3][0], ant3_y=antennas_phase_start[3][1], ant3_z=antennas_phase_start[3][2])

        
        m = Minuit(     rawChi2,\
                        ant0_x=antennas_phase_start[0][0],\
                        ant0_y=antennas_phase_start[0][1],\
                        ant0_z=antennas_phase_start[0][2],\
                        ant1_x=antennas_phase_start[1][0],\
                        ant1_y=antennas_phase_start[1][1],\
                        ant1_z=antennas_phase_start[1][2],\
                        ant2_x=antennas_phase_start[2][0],\
                        ant2_y=antennas_phase_start[2][1],\
                        ant2_z=antennas_phase_start[2][2],\
                        ant3_x=antennas_phase_start[3][0],\
                        ant3_y=antennas_phase_start[3][1],\
                        ant3_z=antennas_phase_start[3][2],\
                        cable_delay0=cable_delays[0],\
                        cable_delay1=cable_delays[1],\
                        cable_delay2=cable_delays[2],\
                        cable_delay3=cable_delays[3],\
                        error_ant0_x=initial_step,\
                        error_ant0_y=initial_step,\
                        error_ant0_z=initial_step,\
                        error_ant1_x=initial_step,\
                        error_ant1_y=initial_step,\
                        error_ant1_z=initial_step,\
                        error_ant2_x=initial_step,\
                        error_ant2_y=initial_step,\
                        error_ant2_z=initial_step,\
                        error_ant3_x=initial_step,\
                        error_ant3_y=initial_step,\
                        error_ant3_z=initial_step,\
                        error_cable_delay0=0.5,\
                        error_cable_delay1=0.5,\
                        error_cable_delay2=0.5,\
                        error_cable_delay3=0.5,\
                        errordef = 1.0,\
                        limit_ant0_x=ant0_physical_limits_x,\
                        limit_ant0_y=ant0_physical_limits_y,\
                        limit_ant0_z=ant0_physical_limits_z,\
                        limit_ant1_x=ant1_physical_limits_x,\
                        limit_ant1_y=ant1_physical_limits_y,\
                        limit_ant1_z=ant1_physical_limits_z,\
                        limit_ant2_x=ant2_physical_limits_x,\
                        limit_ant2_y=ant2_physical_limits_y,\
                        limit_ant2_z=ant2_physical_limits_z,\
                        limit_ant3_x=ant3_physical_limits_x,\
                        limit_ant3_y=ant3_physical_limits_y,\
                        limit_ant3_z=ant3_physical_limits_z,\
                        limit_cable_delay0=limit_cable_delay0,\
                        limit_cable_delay1=limit_cable_delay1,\
                        limit_cable_delay2=limit_cable_delay2,\
                        limit_cable_delay3=limit_cable_delay3,\
                        fix_ant0_x=fix_ant0_x,\
                        fix_ant0_y=fix_ant0_y,\
                        fix_ant0_z=fix_ant0_z,\
                        fix_ant1_x=fix_ant1_x,\
                        fix_ant1_y=fix_ant1_y,\
                        fix_ant1_z=fix_ant1_z,\
                        fix_ant2_x=fix_ant2_x,\
                        fix_ant2_y=fix_ant2_y,\
                        fix_ant2_z=fix_ant2_z,\
                        fix_ant3_x=fix_ant3_x,\
                        fix_ant3_y=fix_ant3_y,\
                        fix_ant3_z=fix_ant3_z,\
                        fix_cable_delay0=fix_cable_delay0,\
                        fix_cable_delay1=fix_cable_delay1,\
                        fix_cable_delay2=fix_cable_delay2,\
                        fix_cable_delay3=fix_cable_delay3)


        result = m.migrad(resume=False)

        m.hesse()
        m.minos()
        pprint(m.get_fmin())
        print(result)

        #12 variables
        ant0_phase_x = m.values['ant0_x']
        ant0_phase_y = m.values['ant0_y']
        ant0_phase_z = m.values['ant0_z']

        ant1_phase_x = m.values['ant1_x']
        ant1_phase_y = m.values['ant1_y']
        ant1_phase_z = m.values['ant1_z']

        ant2_phase_x = m.values['ant2_x']
        ant2_phase_y = m.values['ant2_y']
        ant2_phase_z = m.values['ant2_z']

        ant3_phase_x = m.values['ant3_x']
        ant3_phase_y = m.values['ant3_y']
        ant3_phase_z = m.values['ant3_z']



        chi2_ax.plot([antennas_phase_start[0][0],ant0_phase_x], [antennas_phase_start[0][1],ant0_phase_y], [antennas_phase_start[0][2],ant0_phase_z],c='r',alpha=0.5,linestyle='--')
        chi2_ax.plot([antennas_phase_start[1][0],ant1_phase_x], [antennas_phase_start[1][1],ant1_phase_y], [antennas_phase_start[1][2],ant1_phase_z],c='g',alpha=0.5,linestyle='--')
        chi2_ax.plot([antennas_phase_start[2][0],ant2_phase_x], [antennas_phase_start[2][1],ant2_phase_y], [antennas_phase_start[2][2],ant2_phase_z],c='b',alpha=0.5,linestyle='--')
        chi2_ax.plot([antennas_phase_start[3][0],ant3_phase_x], [antennas_phase_start[3][1],ant3_phase_y], [antennas_phase_start[3][2],ant3_phase_z],c='m',alpha=0.5,linestyle='--')

        chi2_ax.scatter(ant0_phase_x, ant0_phase_y, ant0_phase_z,marker='*',c='r',alpha=0.5,label='Final Ant0')
        chi2_ax.scatter(ant1_phase_x, ant1_phase_y, ant1_phase_z,marker='*',c='g',alpha=0.5,label='Final Ant1')
        chi2_ax.scatter(ant2_phase_x, ant2_phase_y, ant2_phase_z,marker='*',c='b',alpha=0.5,label='Final Ant2')
        chi2_ax.scatter(ant3_phase_x, ant3_phase_y, ant3_phase_z,marker='*',c='m',alpha=0.5,label='Final Ant3')
        
        chi2_ax.set_xlabel('East (m)',linespacing=10)
        chi2_ax.set_ylabel('North (m)',linespacing=10)
        chi2_ax.set_zlabel('Up (m)',linespacing=10)
        chi2_ax.dist = 10
        plt.legend()



        chi2_fig = plt.figure()
        chi2_fig.canvas.set_window_title('Final Positions')
        chi2_ax = chi2_fig.add_subplot(111, projection='3d')
        chi2_ax.scatter(ant0_phase_x, ant0_phase_y, ant0_phase_z,marker='*',c='r',alpha=0.5,label='Final Ant0')
        chi2_ax.scatter(ant1_phase_x, ant1_phase_y, ant1_phase_z,marker='*',c='g',alpha=0.5,label='Final Ant1')
        chi2_ax.scatter(ant2_phase_x, ant2_phase_y, ant2_phase_z,marker='*',c='b',alpha=0.5,label='Final Ant2')
        chi2_ax.scatter(ant3_phase_x, ant3_phase_y, ant3_phase_z,marker='*',c='m',alpha=0.5,label='Final Ant3')

        chi2_ax.set_xlabel('East (m)',linespacing=10)
        chi2_ax.set_ylabel('North (m)',linespacing=10)
        chi2_ax.set_zlabel('Up (m)',linespacing=10)
        chi2_ax.dist = 10
        plt.legend()

        cor = Correlator(reader,  upsample=2**16, n_phi=720, n_theta=720, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=True, tukey=False, sine_subtract=True)
        cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

        adjusted_cor = Correlator(reader,  upsample=2**16, n_phi=720, n_theta=720, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=True, tukey=False, sine_subtract=True)
        adjusted_cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
        
        ant0_ENU = numpy.array([ant0_phase_x, ant0_phase_y, ant0_phase_z])
        ant1_ENU = numpy.array([ant1_phase_x, ant1_phase_y, ant1_phase_z])
        ant2_ENU = numpy.array([ant2_phase_x, ant2_phase_y, ant2_phase_z])
        ant3_ENU = numpy.array([ant3_phase_x, ant3_phase_y, ant3_phase_z])


        if mode == 'hpol':
            adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,adjusted_cor.A0_vpol,adjusted_cor.A1_vpol,adjusted_cor.A2_vpol,adjusted_cor.A3_vpol,verbose=False)
        else:
            adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,adjusted_cor.A0_hpol,adjusted_cor.A1_hpol,adjusted_cor.A2_hpol,adjusted_cor.A3_hpol,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,verbose=False)


        for source_key, cut_dict in data_slicer_cut_dict.items():
            ds.addROI(source_key,cut_dict)
            roi_eventids = numpy.intersect1d(ds.getCutsFromROI(source_key),_eventids)
            roi_impulsivity = ds.getDataFromParam(roi_eventids,'impulsivity_h')
            roi_impulsivity_sort = numpy.argsort(roi_impulsivity) #NOT REVERSED
            eventid = roi_eventids[roi_impulsivity_sort[-1]]
            
            #mean_corr_values, fig, ax = cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=False, zenith_cut_array_plane=None, interactive=True)
            adjusted_mean_corr_values, adjusted_fig, adjusted_ax = adjusted_cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=False, zenith_cut_array_plane=[70,91], interactive=True)
        
        print('Copy-Paste Prints:\n------------')
        print('')
        print('antennas_phase_%s = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%(mode, m.values['ant0_x'],m.values['ant0_y'],m.values['ant0_z'] ,  m.values['ant1_x'],m.values['ant1_y'],m.values['ant1_z'],  m.values['ant2_x'],m.values['ant2_y'],m.values['ant2_z'],  m.values['ant3_x'],m.values['ant3_y'],m.values['ant3_z']))
        print('antennas_phase_%s_hesse = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%(mode, m.errors['ant0_x'],m.errors['ant0_y'],m.errors['ant0_z'] ,  m.errors['ant1_x'],m.errors['ant1_y'],m.errors['ant1_z'],  m.errors['ant2_x'],m.errors['ant2_y'],m.errors['ant2_z'],  m.errors['ant3_x'],m.errors['ant3_y'],m.errors['ant3_z']))


    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






