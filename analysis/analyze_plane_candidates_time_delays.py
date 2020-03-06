'''
This is meant to generate plots for plane candidates as identified by the
plane_hunt.py script.  My hopes for this script are to:

1 : Plot time delays curves for all baselines
2 : Plot some directional information.  Perhaps estimated with simple montecarlo?
3 : Determine the max correlation between signals in the same channel to give some
    metric that the events are the same in nature.  Not sure how to do this combinatorically
    but it might be fine just to do it for each pair. 
4 : Plot correlation maps that scan across events in plane.  Maybe with interpolated time delay
    steps for smooth animation?
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv
import glob
import pymap3d as pm

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info
from tools.data_handler import createFile, getTimes
from objects.fftmath import TimeDelayCalculator, TemplateCompareTool
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import pprint
import itertools
import warnings
import h5py
import pandas as pd
import tools.get_plane_tracks as pt
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

n = 1.0003 #Index of refraction of air  #Should use https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf 
c = 299792458/n #m/s

datapath = os.environ['BEACON_DATA']

cable_delays = info.loadCableDelays()


#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.

known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
ignorable_pulser_ids = info.loadIgnorableEventids()
cm = plt.cm.get_cmap('plasma')

# all_candidates = {\
# '1651-49':{     'eventids':numpy.array([[1651,49],[1651,6817],[1651,12761]]),\
#                 'known_flight':None},\
# '1661-34206':{  'eventids':numpy.array([[1661,34206],[1661,35542],[1661,36609]]),\
#                 'known_flight':None},\
# '1662-58427':{  'eventids':numpy.array([[1662,58427],[1662,58647]]),\
#                 'known_flight':None},\
# '1718-25660':{  'eventids':numpy.array([[1718,25660],[1718,26777],[1718,27756],[1718,28605]]),\
#                 'known_flight':None},\
# '1728-62026':{  'eventids':numpy.array([[1728,62026],[1728,62182],[1728,62370],[1728,62382],[1728,62552],[1728,62577]]),\
#                 'known_flight':'a44585'},\
# '1773-14413':{  'eventids':numpy.array([[1773,14413],[1773,14540],[1773,14590]]),\
#                 'known_flight':'aa8c39'},\
# '1773-63659':{  'eventids':numpy.array([[1773,63659],[1773,63707],[1773,63727],[1773,63752],[1773,63757]]),\
#                 'known_flight':'a28392'},\
# '1774-88800':{  'eventids':numpy.array([[1774,88800],[1774,88810],[1774,88815],[1774,88895],[1774,88913],[1774,88921],[1774,88923],[1774,88925],[1774,88944],[1774,88955],[1774,88959],[1774,88988],[1774,88993],[1774,89029],[1774,89030],[1774,89032],[1774,89034],[1774,89041],[1774,89043],[1774,89052],[1774,89172],[1774,89175],[1774,89181],[1774,89203],[1774,89204],[1774,89213]]),\
#                 'known_flight':'ab5f43'},\
# '1783-28830':{  'eventids':numpy.array([[1783,28830],[1783,28832],[1783,28861]]),\
#                 'known_flight':'a52e4f'},\
# '1783-35725':{  'eventids':numpy.array([[1783,35725],[1783,34730],[1783,34738],[1783,34778],[1783,34793]]),\
#                 'known_flight':None},\
# '1784-7166':{   'eventids':numpy.array([[1784,7166],[1784,7176],[1784,7179],[1784,7195],[1784,7244],[1784,7255]]),\
#                 'known_flight':'acf975'}\
# }

#Now all means only the ones that are planes! 
all_candidates = {\
'1728-62026':{  'eventids':numpy.array([[1728,62026],[1728,62182],[1728,62370],[1728,62382],[1728,62552],[1728,62577]]),\
                'known_flight':'a44585',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]}},\
'1773-14413':{  'eventids':numpy.array([[1773,14413],[1773,14540],[1773,14590]]),\
                'known_flight':'aa8c39',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1773-63659':{  'eventids':numpy.array([[1773,63659],[1773,63707],[1773,63727],[1773,63752],[1773,63757]]),\
                'known_flight':'a28392',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1774-88800':{  'eventids':numpy.array([[1774,88800],[1774,88810],[1774,88815],[1774,88895],[1774,88913],[1774,88921],[1774,88923],[1774,88925],[1774,88944],[1774,88955],[1774,88959],[1774,88988],[1774,88993],[1774,89029],[1774,89030],[1774,89032],[1774,89034],[1774,89041],[1774,89043],[1774,89052],[1774,89172],[1774,89175],[1774,89181],[1774,89203],[1774,89204],[1774,89213]]),\
                'known_flight':'ab5f43',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1783-28830':{  'eventids':numpy.array([[1783,28830],[1783,28832],[1783,28861]]),\
                'known_flight':'a52e4f',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1784-7166':{   'eventids':numpy.array([[1784,7166],[1784,7176],[1784,7179],[1784,7195],[1784,7244],[1784,7255]]),\
                'known_flight':'acf975',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}}\
}


candidates = all_candidates
# test_track = '1784-7166'#'1728-62026'
# candidates = {test_track:all_candidates[test_track]}

if __name__ == '__main__':
    #################################
    # Align method can be one of a few:
    # 0: argmax of corrs (default)
    # 1: argmax of hilbert of corrs
    # 2: Average of argmin and argmax
    # 3: argmin of corrs
    # 4: Pick the largest max peak preceding the max of the hilbert of corrs
    # 5: Pick the average indices of values > 95% peak height in corrs
    # 6: Pick the average indices of values > 98% peak height in hilbert of corrs
    # 7: Gets argmax of abs(corrs) and then finds highest positive peak before this value
    # 8: Apply cfd to waveforms to get first pass guess at delays, then pick the best correlation near that. 
    # 9: Slider method.  By eye inspection of each time delay. 
    
    if len(sys.argv) == 2:
        if str(sys.argv[1]) in ['vpol', 'hpol']:
            mode = str(sys.argv[1])
        else:
            print('Given mode not in options.  Defaulting to hpol')
            mode = 'hpol'
    else:
        print('No mode given.  Defaulting to hpol')
        mode = 'hpol'

    final_corr_length = 2**18

    #FILTER STRING USED IF ABOVE IS FALSE
    default_align_method=0 #WILL BE CHANGED IF GIVEN ABOVE
    filter_string = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_%i-align_%i'%(final_corr_length,default_align_method)


    crit_freq_low_pass_MHz = None#60 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = None#5

    crit_freq_high_pass_MHz = None#60
    high_pass_filter_order = None#6

    waveform_index_range = (None,None)#(150,400)

    apply_phase_response = True
    hilbert = False
    use_interpolated_tracks = True


    plot_filter = False
    plot_multiple = False
    plot_averaged_waveforms = False
    get_averaged_waveforms = True #If you want those calculations done but not plotted
    plot_averaged_waveforms_aligned = False
    plot_time_delays = True
    plot_planes = False
    plot_interps = False
    plot_residuals = True

    known_planes, calibrated_trigtime, output_tracks = pt.getKnownPlaneTracks()

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

    ant0_x=antennas_phase_start[0][0]
    ant0_y=antennas_phase_start[0][1]
    ant0_z=antennas_phase_start[0][2]
    ant1_x=antennas_phase_start[1][0]
    ant1_y=antennas_phase_start[1][1]
    ant1_z=antennas_phase_start[1][2]
    ant2_x=antennas_phase_start[2][0]
    ant2_y=antennas_phase_start[2][1]
    ant2_z=antennas_phase_start[2][2]
    ant3_x=antennas_phase_start[3][0]
    ant3_y=antennas_phase_start[3][1]
    ant3_z=antennas_phase_start[3][2]
    loc_dict = {0:[ant0_x,ant0_y,ant0_z],1:[ant1_x,ant1_y,ant1_z],2:[ant2_x,ant2_y,ant2_z],3:[ant3_x,ant3_y,ant3_z]}

    try:

        for key in list(calibrated_trigtime.keys()):
            pair_cut = numpy.array([pair in known_planes[key]['baselines'][mode] for pair in [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]] ]) #Checks which pairs are worth looping over.
            measured_plane_time_delays[key] = known_planes[key]['time_delays'][mode].T[pair_cut]
            measured_plane_time_delays_weights[key] = known_planes[key]['max_corrs'][mode].T[pair_cut] #might need something better than this. 
            enu = pm.geodetic2enu(output_tracks[key]['lat'],output_tracks[key]['lon'],output_tracks[key]['alt'],origin[0],origin[1],origin[2])

            plane_polys[key] = pt.PlanePoly(output_tracks[key]['timestamps'],enu,plot=plot_interps)

            interpolated_plane_locations[key] = plane_polys[key].poly(calibrated_trigtime[key])
            
            d0 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant0_x)**2 + (interpolated_plane_locations[key][:,1] - ant0_y)**2 + (interpolated_plane_locations[key][:,2] - ant0_z)**2 )/c)*1.0e9 #ns
            d1 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant1_x)**2 + (interpolated_plane_locations[key][:,1] - ant1_y)**2 + (interpolated_plane_locations[key][:,2] - ant1_z)**2 )/c)*1.0e9 #ns
            d2 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant2_x)**2 + (interpolated_plane_locations[key][:,1] - ant2_y)**2 + (interpolated_plane_locations[key][:,2] - ant2_z)**2 )/c)*1.0e9 #ns
            d3 = (numpy.sqrt((interpolated_plane_locations[key][:,0] - ant3_x)**2 + (interpolated_plane_locations[key][:,1] - ant3_y)**2 + (interpolated_plane_locations[key][:,2] - ant3_z)**2 )/c)*1.0e9 #ns

            d = [d0,d1,d2,d3]

            for pair_index, pair in enumerate(known_planes[key]['baselines'][mode]):
                geometric_time_delay = (d[pair[0]] + cable_delays[pair[0]]) - (d[pair[1]] + cable_delays[pair[1]])
                if pair_index == 0:
                    geometric_time_delays = geometric_time_delay
                else:
                    geometric_time_delays = numpy.vstack((geometric_time_delays,geometric_time_delay))

            if plot_planes == True:
                plane_ax.plot(enu[0]/1000.0,enu[1]/1000.0,enu[2]/1000.0,label=key + ' : ' + known_planes[key]['known_flight'])
                plane_ax.scatter(interpolated_plane_locations[key][:,0]/1000.0,interpolated_plane_locations[key][:,1]/1000.0,interpolated_plane_locations[key][:,2]/1000.0,label=key + ' : ' + known_planes[key]['known_flight'])


            if plot_time_delays:
                plane = key

                min_timestamp = min(calibrated_trigtime[key])
                max_timestamp = max(calibrated_trigtime[key])

                flight_tracks_ENU, all_vals = pt.getENUTrackDict(min_timestamp,max_timestamp,100,hour_window = 0,flights_of_interest=[known_planes[plane]['known_flight']])

                time_delay_fig = plt.figure()
                time_delay_fig.canvas.set_window_title('%s %s Delays'%(key, mode))
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.title(mode + ' ' + plane + ', ' + known_planes[plane]['known_flight'] + '\nAdjusted Antenna Positions')
                plt.ylabel('Time Delay (ns)')
                plt.xlabel('Readout Time (s)')

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
                        #PLOTTING FIT/IDENTIFIED MEASURED FLIGHT TRACKS
                        y = measured_plane_time_delays[plane][pair_index] - geometric_time_delays[pair_index]
                        plt.plot(calibrated_trigtime[key], y,c=python_colors[pair_index],linestyle = '--',alpha=0.8)
                        plt.scatter(calibrated_trigtime[key], y,c=python_colors[pair_index],label='Measured Time Delays for A%i and A%i'%(pair[0],pair[1]))
                        text_color = plt.gca().lines[-1].get_color()

                        #Attempt at avoiding overlapping text.
                        text_loc = numpy.array([numpy.mean(x)-5,numpy.mean(y)])
                        plt.text(text_loc[0],text_loc[1], 'A%i and A%i'%(pair[0],pair[1]),color=text_color,withdash=True)

                    else:
                        #PLOTTING FIT/IDENTIFIED MEASURED FLIGHT TRACKS
                        y = measured_plane_time_delays[plane][pair_index]
                        plt.plot(calibrated_trigtime[key], y,c=python_colors[pair_index],linestyle = '--',alpha=0.8)
                        plt.scatter(calibrated_trigtime[key], y,c=python_colors[pair_index],label='Measured Time Delays for A%i and A%i'%(pair[0],pair[1]))

                        #PLOTTING EXPECTED FLIGHT TRACKS FOR THE KNOWN CORRELATED FLIGHT
                        y = dt['expected_time_differences_%s'%mode][(pair[0], pair[1])][plot_distance_cut]
                        plt.plot(x,y,c=python_colors[pair_index],linestyle = '--',alpha=0.5,label='Flight %s TD: A%i and A%i'%(known_planes[plane]['known_flight'],pair[0],pair[1]))
                        plt.scatter(x,y,facecolors='none', edgecolors=python_colors[pair_index],alpha=0.4)

                        text_color = plt.gca().lines[-1].get_color()

                        #Attempt at avoiding overlapping text.
                        text_loc = numpy.array([numpy.mean(x)-5,numpy.mean(y)])
                        plt.text(text_loc[0],text_loc[1], 'A%i and A%i'%(pair[0],pair[1]),color=text_color,withdash=True)

                        
    except Exception as e:
        print('Error in plotting.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    
