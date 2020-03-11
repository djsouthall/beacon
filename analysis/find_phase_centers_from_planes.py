'''
This script assumes the antenna positions have already roughly been set/determined
using find_phase_centers.py  .  It will assume these as starting points of the antennas
and then vary all 4 of their locations to match measured time delays from planes. 
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
import objects.station as bc
import tools.info as info
import tools.get_plane_tracks as pt
from objects.fftmath import TimeDelayCalculator

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
        plot_residuals = True
        plot_planes = True
        plot_interps = False
        plot_time_delays = True
        use_interpolated_tracks = True
        plot_az_res = True

        final_corr_length = 2**18
        crit_freq_low_pass_MHz = None#60 #This new pulser seems to peak in the region of 85 MHz or so
        low_pass_filter_order = None#5

        crit_freq_high_pass_MHz = None#60
        high_pass_filter_order = None#6

        waveform_index_range = (None,None)#(150,400)

        apply_phase_response = True
        hilbert = False

        try_to_use_precalculated_time_delays = False
        try_to_use_precalculated_time_delays_but_just_as_guess_for_real_time_delays_why_is_this_so_long = True

        if len(sys.argv) == 2:
            if str(sys.argv[1]) in ['vpol', 'hpol']:
                mode = str(sys.argv[1])
            else:
                print('Given mode not in options.  Defaulting to hpol')
                mode = 'hpol'
        else:
            print('No mode given.  Defaulting to hpol')
            mode = 'hpol'

        print('Loading known plane locations.')
        known_planes, calibrated_trigtime, output_tracks = pt.getKnownPlaneTracks(ignore_planes=[]) # ['1728-62026','1773-14413','1773-63659','1774-88800','1783-28830','1784-7166']#'1774-88800','1728-62026'
        origin = info.loadAntennaZeroLocation(deploy_index = 1)
        antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU()
        if mode == 'hpol':
            antennas_phase_start = antennas_phase_hpol
        else:
            antennas_phase_start = antennas_phase_vpol

        print('Loading in cable delays.')
        cable_delays = info.loadCableDelays()[mode]

        if plot_planes == True:
            plane_fig = plt.figure()
            plane_fig.canvas.set_window_title('3D Plane Tracks')
            plane_ax = plane_fig.add_subplot(111, projection='3d')
            plane_ax.scatter(0,0,0,label='Antenna 0',c='k')

        plane_polys = {}
        interpolated_plane_locations = {}
        measured_plane_time_delays = {}
        measured_plane_time_delays_weights = {}

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for index, key in enumerate(list(calibrated_trigtime.keys())):
            enu = numpy.array(pm.geodetic2enu(output_tracks[key]['lat'],output_tracks[key]['lon'],output_tracks[key]['alt'],origin[0],origin[1],origin[2]))
            
            unique_enu_indices = numpy.sort(numpy.unique(enu,axis=1,return_index=True)[1])
            distance = numpy.sqrt(enu[0]**2 + enu[0]**2 + enu[0]**2)
            distance_cut_indices = numpy.where(distance/1000.0 < 100)
            unique_enu_indices = unique_enu_indices[numpy.isin(unique_enu_indices,distance_cut_indices)]
            

            plane_polys[key] = pt.PlanePoly(output_tracks[key]['timestamps'][unique_enu_indices],enu[:,unique_enu_indices],plot=plot_interps)
            #import pdb; pdb.set_trace()

            interpolated_plane_locations[key] = plane_polys[key].poly(calibrated_trigtime[key])
            
            if plot_planes == True:
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


            # enu_az = numpy.rad2deg(numpy.arctan2(interpolated_plane_locations[key][:,1],interpolated_plane_locations[key][:,0]))
            # enu_elevation_angle = numpy.rad2deg(numpy.arctan2(interpolated_plane_locations[key][:,2],numpy.sqrt(interpolated_plane_locations[key][:,0]**2+interpolated_plane_locations[key][:,1]**2)))
            pair_cut = numpy.array([pair in known_planes[key]['baselines'][mode] for pair in [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]] ]) #Checks which pairs are worth looping over.
           
            if try_to_use_precalculated_time_delays == True and numpy.logical_and('time_delays' in list(known_planes[key].keys()),'max_corrs' in list(known_planes[key].keys())):
                print('Using precalculated time delays from info.py')
                measured_plane_time_delays[key] = known_planes[key]['time_delays'][mode].T[pair_cut]
                measured_plane_time_delays_weights[key] = known_planes[key]['max_corrs'][mode].T[pair_cut] #might need something better than this. 

            elif try_to_use_precalculated_time_delays_but_just_as_guess_for_real_time_delays_why_is_this_so_long == True:

                guess_time_delays = numpy.vstack((known_planes[key]['time_delays']['hpol'].T,known_planes[key]['time_delays']['vpol'].T)).T

                print('Calculating time delays from info.py')
                run = int(key.split('-')[0])
                reader = Reader(datapath,run)
                tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
                eventids = known_planes[key]['eventids'][:,1]
                
                time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids,align_method=10,hilbert=hilbert, align_method_10_estimates=guess_time_delays, align_method_10_window_ns=8)

                if mode == 'hpol':
                    measured_plane_time_delays[key] = time_shifts[0:6,:][pair_cut]
                    measured_plane_time_delays_weights[key] = corrs[0:6,:][pair_cut] #might need something better than this. 
                else:
                    measured_plane_time_delays[key] = time_shifts[6:12,:][pair_cut]
                    measured_plane_time_delays_weights[key] = corrs[6:12,:][pair_cut] #might need something better than this.

            else:
                print('Calculating time delays from info.py')
                run = int(key.split('-')[0])
                reader = Reader(datapath,run)
                tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
                eventids = known_planes[key]['eventids'][:,1]
                time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids,align_method=0,hilbert=hilbert)

                if mode == 'hpol':
                    measured_plane_time_delays[key] = time_shifts[0:6,:][pair_cut]
                    measured_plane_time_delays_weights[key] = corrs[0:6,:][pair_cut] #might need something better than this. 
                else:
                    measured_plane_time_delays[key] = time_shifts[6:12,:][pair_cut]
                    measured_plane_time_delays_weights[key] = corrs[6:12,:][pair_cut] #might need something better than this. 


        if plot_planes == True:
            plt.legend(loc='upper right')
            plane_ax.set_xlabel('East (km)',linespacing=10)
            plane_ax.set_ylabel('North (km)',linespacing=10)
            plane_ax.set_zlabel('Up (km)',linespacing=10)
            plane_ax.dist = 10


        #Limits 
        guess_range = None #Limit to how far from input phase locations to limit the parameter space to
        fix_ant0_x = True
        fix_ant0_y = True
        fix_ant0_z = True
        fix_ant1_x = False
        fix_ant1_y = False
        fix_ant1_z = False
        fix_ant2_x = False
        fix_ant2_y = False
        fix_ant2_z = False
        fix_ant3_x = False
        fix_ant3_y = False
        fix_ant3_z = False

        if guess_range is not None:

            ant0_physical_limits_x = (antennas_phase_start[0][0] - guess_range ,antennas_phase_start[0][0] + guess_range)
            ant0_physical_limits_y = (antennas_phase_start[0][1] - guess_range ,antennas_phase_start[0][1] + guess_range)
            ant0_physical_limits_z = (antennas_phase_start[0][2] - guess_range ,antennas_phase_start[0][2] + guess_range)

            ant1_physical_limits_x = (antennas_phase_start[1][0] - guess_range ,antennas_phase_start[1][0] + guess_range)
            ant1_physical_limits_y = (antennas_phase_start[1][1] - guess_range ,antennas_phase_start[1][1] + guess_range)
            ant1_physical_limits_z = (antennas_phase_start[1][2] - guess_range ,antennas_phase_start[1][2] + guess_range)

            ant2_physical_limits_x = (antennas_phase_start[2][0] - guess_range ,antennas_phase_start[2][0] + guess_range)
            ant2_physical_limits_y = (antennas_phase_start[2][1] - guess_range ,antennas_phase_start[2][1] + guess_range)
            ant2_physical_limits_z = (antennas_phase_start[2][2] - guess_range ,antennas_phase_start[2][2] + guess_range)

            ant3_physical_limits_x = (antennas_phase_start[3][0] - guess_range ,antennas_phase_start[3][0] + guess_range)
            ant3_physical_limits_y = (antennas_phase_start[3][1] - guess_range ,antennas_phase_start[3][1] + guess_range)
            ant3_physical_limits_z = (antennas_phase_start[3][2] - guess_range ,antennas_phase_start[3][2] + guess_range)

        else:
            ant0_physical_limits_x = None#None 
            ant0_physical_limits_y = None#None
            ant0_physical_limits_z = None#None

            ant1_physical_limits_x = None#(None,0.0) #Forced west of 0
            ant1_physical_limits_y = None#None
            ant1_physical_limits_z = None#(10.0,None) #Forced above 0

            ant2_physical_limits_x = None#None
            ant2_physical_limits_y = None#(None,0.0) #Forced south of 0 
            ant2_physical_limits_z = None#None

            ant3_physical_limits_x = None#(None,0.0) #Forced West of 0
            ant3_physical_limits_y = None#(None,0.0) #Forced South of 0
            ant3_physical_limits_z = None#(0.0,None) #Forced above 0


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
        chi2_ax.dist = 10
        plt.legend()


        chi2_fig = plt.figure()
        chi2_fig.canvas.set_window_title('Both')
        chi2_ax = chi2_fig.add_subplot(111, projection='3d')
        chi2_ax.scatter(antennas_phase_start[0][0], antennas_phase_start[0][1], antennas_phase_start[0][2],c='r',alpha=0.5,label='Initial Ant0')
        chi2_ax.scatter(antennas_phase_start[1][0], antennas_phase_start[1][1], antennas_phase_start[1][2],c='g',alpha=0.5,label='Initial Ant1')
        chi2_ax.scatter(antennas_phase_start[2][0], antennas_phase_start[2][1], antennas_phase_start[2][2],c='b',alpha=0.5,label='Initial Ant2')
        chi2_ax.scatter(antennas_phase_start[3][0], antennas_phase_start[3][1], antennas_phase_start[3][2],c='m',alpha=0.5,label='Initial Ant3')

        def rawChi2(ant0_x, ant0_y, ant0_z,ant1_x, ant1_y, ant1_z, ant2_x, ant2_y, ant2_z, ant3_x, ant3_y, ant3_z):
            '''
            This is a chi^2 that loops over locations from planes, calculating expected time delays for those locations.  Then
            it will compares those to the calculated time delays for known plane events.  
            '''
            try:
                #chi2_ax.scatter(ant0_x, ant0_y, ant0_z,label='Antenna 0',c='r',alpha=0.5)
                #chi2_ax.scatter(ant1_x, ant1_y, ant1_z,label='Antenna 1',c='g',alpha=0.5)
                #chi2_ax.scatter(ant2_x, ant2_y, ant2_z,label='Antenna 2',c='b',alpha=0.5)
                #chi2_ax.scatter(ant3_x, ant3_y, ant3_z,label='Antenna 3',c='m',alpha=0.5)

                #Calculate distances (already converted to ns) from pulser to each antenna
                chi_2 = 0.0

                for key in list(known_planes.keys()):
                    # print(interpolated_plane_locations[key][:,0] - ant0_x)
                    # print(interpolated_plane_locations[key][:,1] - ant0_y)
                    # print(interpolated_plane_locations[key][:,2] - ant0_z)
                    # print(interpolated_plane_locations[key][:,0] - ant1_x)
                    # print(interpolated_plane_locations[key][:,1] - ant1_y)
                    # print(interpolated_plane_locations[key][:,2] - ant1_z)
                    # print(interpolated_plane_locations[key][:,0] - ant2_x)
                    # print(interpolated_plane_locations[key][:,1] - ant2_y)
                    # print(interpolated_plane_locations[key][:,2] - ant2_z)
                    # print(interpolated_plane_locations[key][:,0] - ant3_x)
                    # print(interpolated_plane_locations[key][:,1] - ant3_y)
                    # print(interpolated_plane_locations[key][:,2] - ant3_z)



                    d0 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant0_x)**2 + (interpolated_plane_locations[key][:,1] - ant0_y)**2 + (interpolated_plane_locations[key][:,2] - ant0_z)**2 )/c)*1.0e9 #ns
                    d1 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant1_x)**2 + (interpolated_plane_locations[key][:,1] - ant1_y)**2 + (interpolated_plane_locations[key][:,2] - ant1_z)**2 )/c)*1.0e9 #ns
                    d2 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant2_x)**2 + (interpolated_plane_locations[key][:,1] - ant2_y)**2 + (interpolated_plane_locations[key][:,2] - ant2_z)**2 )/c)*1.0e9 #ns
                    d3 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant3_x)**2 + (interpolated_plane_locations[key][:,1] - ant3_y)**2 + (interpolated_plane_locations[key][:,2] - ant3_z)**2 )/c)*1.0e9 #ns

                    d = [d0,d1,d2,d3]

                    for pair_index, pair in enumerate(known_planes[key]['baselines'][mode]):
                        geometric_time_delay = (d[pair[0]] + cable_delays[pair[0]]) - (d[pair[1]] + cable_delays[pair[1]])
                        #Right now these seem reversed from what I would expect based on the plot?  In time I mean. 

                        
                        vals = ((geometric_time_delay - measured_plane_time_delays[key][pair_index])**2)/(1.0001-measured_plane_time_delays_weights[key][pair_index])**2 #1-max(corr) makes the optimizer see larger variations when accurate time delays aren't well matched.  i.e the smaller max(corr) the smaller (1-max(corr))**2 is, and the larger that chi^2 is.  
                        #chi_2 += numpy.mean(vals)/numpy.std(vals) #This was a completely dumb calculation that biased against planes that actually travelled.  I missed the point.
                        #chi_2 += numpy.mean(vals)#Weights each plane equally
                        chi_2 += numpy.sum(vals)#Weights each event equally, planes with more events weighted more.
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
                        error_ant0_x=initial_step,\
                        error_ant0_y=initial_step,\
                        error_ant0_z=5.0*initial_step,\
                        error_ant1_x=initial_step,\
                        error_ant1_y=initial_step,\
                        error_ant1_z=5.0*initial_step,\
                        error_ant2_x=initial_step,\
                        error_ant2_y=initial_step,\
                        error_ant2_z=5.0*initial_step,\
                        error_ant3_x=initial_step,\
                        error_ant3_y=initial_step,\
                        error_ant3_z=5.0*initial_step,\
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
                        fix_ant3_z=fix_ant3_z)


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


        if plot_time_delays == True:
            if plot_az_res:
                az_fig = plt.figure()
                az_fig.canvas.set_window_title('%s Res v.s. Az'%(mode))
                az_ax = plt.gca()
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.ylabel('Time Delay (ns)')
                plt.xlabel('Azimuth Angle (Deg)')

            loc_dict = {0:[m.values['ant0_x'],m.values['ant0_y'],m.values['ant0_z']],1:[m.values['ant1_x'],m.values['ant1_y'],m.values['ant1_z']],2:[m.values['ant2_x'],m.values['ant2_y'],m.values['ant2_z']],3:[m.values['ant3_x'],m.values['ant3_y'],m.values['ant3_z']]}
            
            for plane in list(known_planes.keys()):
                d0 = (numpy.sqrt((interpolated_plane_locations[plane][:,0] - m.values['ant0_x'])**2 + (interpolated_plane_locations[plane][:,1] - m.values['ant0_y'])**2 + (interpolated_plane_locations[plane][:,2] - m.values['ant0_z'])**2 )/c)*1.0e9 #ns
                d1 = (numpy.sqrt((interpolated_plane_locations[plane][:,0] - m.values['ant1_x'])**2 + (interpolated_plane_locations[plane][:,1] - m.values['ant1_y'])**2 + (interpolated_plane_locations[plane][:,2] - m.values['ant1_z'])**2 )/c)*1.0e9 #ns
                d2 = (numpy.sqrt((interpolated_plane_locations[plane][:,0] - m.values['ant2_x'])**2 + (interpolated_plane_locations[plane][:,1] - m.values['ant2_y'])**2 + (interpolated_plane_locations[plane][:,2] - m.values['ant2_z'])**2 )/c)*1.0e9 #ns
                d3 = (numpy.sqrt((interpolated_plane_locations[plane][:,0] - m.values['ant3_x'])**2 + (interpolated_plane_locations[plane][:,1] - m.values['ant3_y'])**2 + (interpolated_plane_locations[plane][:,2] - m.values['ant3_z'])**2 )/c)*1.0e9 #ns

                d = [d0,d1,d2,d3]
                if plot_az_res == True:
                    #SHOULD MAKE THIS V.S. ELEVATION AND ARRAY ELEVATION
                    azimuths = numpy.rad2deg(numpy.arctan2(interpolated_plane_locations[plane][:,1],interpolated_plane_locations[plane][:,0]))
                    azimuths[azimuths < 0] = azimuths[azimuths < 0]%360
                for pair_index, pair in enumerate(known_planes[plane]['baselines'][mode]):
                    geometric_time_delay = (d[pair[0]] + cable_delays[pair[0]]) - (d[pair[1]] + cable_delays[pair[1]])
                    #Right now these seem reversed from what I would expect based on the plot?  In time I mean. 
                    if pair_index == 0:
                        geometric_time_delays = geometric_time_delay
                    else:
                        geometric_time_delays = numpy.vstack((geometric_time_delays,geometric_time_delay))   

                run = int(plane.split('-')[0])
                eventids = known_planes[plane]['eventids'][:,1]
                reader = Reader(datapath,run)
                filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.  

                with h5py.File(filename, 'r') as file:
                    try:
                        load_cut = numpy.isin(file['eventids'][...],eventids)
                        times = file['calibrated_trigtime'][load_cut]
                    except Exception as e:
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                min_timestamp = min(times)
                max_timestamp = max(times)

                flight_tracks_ENU, all_vals = pt.getENUTrackDict(min_timestamp,max_timestamp,100,hour_window = 0,flights_of_interest=[known_planes[plane]['known_flight']])
                #Get the corresponding calibrated trig times for the particular track.  



                time_delay_fig = plt.figure()
                time_delay_fig.canvas.set_window_title('%s , %s Delays'%(plane, mode))
                plt.ylabel('Time Delay (ns)')
                plt.xlabel('Readout Time (s)')
                td_ax = plt.gca()
                td_ax.minorticks_on()
                td_ax.grid(b=True, which='major', color='k', linestyle='-')
                td_ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                python_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

                track = flight_tracks_ENU[known_planes[plane]['known_flight']]

                if use_interpolated_tracks == True:
                    track[:,0:3] = plane_polys[plane].poly(track[:,3])#E N U t


                if mode  == 'hpol':       
                    tof, dof, dt = pt.getTimeDelaysFromTrack(track, adjusted_antennas_physical=None,adjusted_antennas_phase_hpol=loc_dict,adjusted_antennas_phase_vpol=None)
                else:
                    tof, dof, dt = pt.getTimeDelaysFromTrack(track, adjusted_antennas_physical=None,adjusted_antennas_phase_hpol=None,adjusted_antennas_phase_vpol=loc_dict)

                distance = numpy.sqrt(numpy.sum(track[:,0:3]**2,axis=1))/1000 #km

                plot_distance_cut_limit = None
                if plot_distance_cut_limit is not None:
                    plot_distance_cut = distance <= plot_distance_cut_limit
                else:
                    plot_distance_cut = numpy.ones_like(distance,dtype=bool)

                x = track[plot_distance_cut,3]

                for pair_index, pair in enumerate(known_planes[plane]['baselines'][mode]):
                    if plot_residuals == True:
                        y = measured_plane_time_delays[plane][pair_index] - geometric_time_delays[pair_index]

                        #PLOTTING FIT/IDENTIFIED MEASURED FLIGHT TRACKS
                        td_ax.plot(times, y,c=python_colors[pair_index],linestyle = '--',alpha=0.8)
                        td_ax.scatter(times, y,c=python_colors[pair_index],label='Residuals for A%i and A%i'%(pair[0],pair[1]))
                    else:
                        y = measured_plane_time_delays[plane][pair_index]

                        #PLOTTING FIT/IDENTIFIED MEASURED FLIGHT TRACKS
                        td_ax.plot(times, y,c=python_colors[pair_index],linestyle = '--',alpha=0.8)
                        td_ax.scatter(times, y,c=python_colors[pair_index],label='Measured Time Delays for A%i and A%i'%(pair[0],pair[1]))

                        #PLOTTING EXPECTED FLIGHT TRACKS FOR THE KNOWN CORRELATED FLIGHT
                        y = dt['expected_time_differences_%s'%mode][(pair[0], pair[1])][plot_distance_cut]
                        td_ax.plot(x,y,c=python_colors[pair_index],linestyle = '--',alpha=0.5,label='Flight %s TD: A%i and A%i'%(known_planes[plane]['known_flight'],pair[0],pair[1]))
                        td_ax.scatter(x,y,facecolors='none', edgecolors=python_colors[pair_index],alpha=0.4)

                        text_color = plt.gca().lines[-1].get_color()

                        #Attempt at avoiding overlapping text.
                        text_loc = numpy.array([numpy.mean(x)-5,numpy.mean(y)])
                        td_ax.text(text_loc[0],text_loc[1], 'A%i and A%i'%(pair[0],pair[1]),color=text_color,withdash=True)
                    
                    if plot_az_res == True:
                        python_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                        y = measured_plane_time_delays[plane][pair_index] - geometric_time_delays[pair_index]
                        az_ax.scatter(azimuths, y,c=python_colors[pair_index],label='Residuals A%i and A%i'%(pair[0],pair[1]))

                if plot_residuals:
                    td_ax.legend()


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





