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
from pprint import pprint
import itertools
import warnings
import h5py
import pandas as pd
import tools.get_plane_tracks as pt
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

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
                'align_method':0},\
'1773-14413':{  'eventids':numpy.array([[1773,14413],[1773,14540],[1773,14590]]),\
                'known_flight':'aa8c39',\
                'align_method':0},\
'1773-63659':{  'eventids':numpy.array([[1773,63659],[1773,63707],[1773,63727],[1773,63752],[1773,63757]]),\
                'known_flight':'a28392',\
                'align_method':0},\
'1774-88800':{  'eventids':numpy.array([[1774,88800],[1774,88810],[1774,88815],[1774,88895],[1774,88913],[1774,88921],[1774,88923],[1774,88925],[1774,88944],[1774,88955],[1774,88959],[1774,88988],[1774,88993],[1774,89029],[1774,89030],[1774,89032],[1774,89034],[1774,89041],[1774,89043],[1774,89052],[1774,89172],[1774,89175],[1774,89181],[1774,89203],[1774,89204],[1774,89213]]),\
                'known_flight':'ab5f43',\
                'align_method':9},\
'1783-28830':{  'eventids':numpy.array([[1783,28830],[1783,28832],[1783,28861]]),\
                'known_flight':'a52e4f',\
                'align_method':0},\
'1784-7166':{   'eventids':numpy.array([[1784,7166],[1784,7176],[1784,7179],[1784,7195],[1784,7244],[1784,7255]]),\
                'known_flight':'acf975',\
                'align_method':9}\
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
    
    final_corr_length = 2**18

    calculate_time_delays = True # Or just use precalculated time delays.

    #FILTER STRING USED IF ABOVE IS FALSE
    default_align_method=0 #WILL BE CHANGED IF GIVEN ABOVE
    filter_string = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_%i-align_%i'%(final_corr_length,default_align_method)


    crit_freq_low_pass_MHz = None#60 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = None#5

    crit_freq_high_pass_MHz = None#60
    high_pass_filter_order = None#6

    waveform_index_range = (None,None)#(150,400)

    apply_phase_response = False
    hilbert = False


    plot_filter = False
    plot_multiple = False
    plot_averaged_waveforms = False
    get_averaged_waveforms = True #If you want those calculations done but not plotted
    plot_averaged_waveforms_aligned = False
    plot_time_delays = True

    plot_plane_tracks = True #if True then plot_time_delays is set to True

    try:
        for candidates_key in list(candidates.keys()):
            track = candidates[candidates_key]['eventids']
            known_flight = candidates[candidates_key]['known_flight']
            if 'align_method' in list(candidates[candidates_key].keys()):
                if candidates[candidates_key]['align_method'] is not None:
                    align_method = candidates[candidates_key]['align_method']
                else:
                    align_method = default_align_method
            else:
                align_method = default_align_method

            runs = numpy.unique(track[:,0])
            for run_index, run in enumerate(runs):
                reader = Reader(datapath,run)
                eventids = numpy.sort(track[:,1][track[:,0] == run])
                filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.  

                with h5py.File(filename, 'r') as file:
                    try:
                        load_cut = numpy.isin(file['eventids'][...],eventids)
                        calibrated_trigtime = file['calibrated_trigtime'][load_cut]
                        print('\nProcessing Run %i'%run)

                        if numpy.any([plot_averaged_waveforms, get_averaged_waveforms, plot_averaged_waveforms_aligned]):
                            tct = TemplateCompareTool(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False,apply_phase_response=apply_phase_response)

                            #First pass alignment to make templates.  
                            times, averaged_waveforms = tct.averageAlignedSignalsPerChannel(eventids, template_eventid=eventids[-1], align_method=0, plot=plot_averaged_waveforms_aligned)

                            if plot_averaged_waveforms:
                                plt.figure()
                                plt.suptitle('Averaged Waveforms for Candidate %i-%i'%(run,eventids[0]))
                                plt.subplot(4,2,1)
                                for index, waveform in enumerate(averaged_waveforms):
                                    plt.subplot(4,2,index+1)
                                    plt.plot(times, waveform)

                                    plt.minorticks_on()
                                    plt.grid(b=True, which='major', color='k', linestyle='-')
                                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                                    plt.ylabel('Adu (averaged)')
                                    plt.xlabel('Time (ns)')

                        if calculate_time_delays == True:
                            tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=plot_filter,apply_phase_response=apply_phase_response)
                            time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids,align_method=align_method,hilbert=hilbert,plot=plot_multiple,hpol_cut=None,vpol_cut=None, colors=calibrated_trigtime)
                            hpol_delays = time_shifts[0:6,:].T
                            vpol_delays = time_shifts[6:12,:].T
                            hpol_corrs = corrs[0:6,:].T
                            vpol_corrs = corrs[6:12,:].T
                        else:
                            hpol_delays = numpy.vstack((file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,1)][load_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,2)][load_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,3)][load_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(1,2)][load_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(1,3)][load_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(2,3)][load_cut])).T.T
                            vpol_delays = numpy.vstack((file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,1)][load_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,2)][load_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,3)][load_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(1,2)][load_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(1,3)][load_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(2,3)][load_cut])).T.T
                            hpol_corrs = numpy.vstack((file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(0,1)][load_cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(0,2)][load_cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(0,3)][load_cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(1,2)][load_cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(1,3)][load_cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(2,3)][load_cut])).T.T
                            vpol_corrs = numpy.vstack((file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(0,1)][load_cut],file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(0,2)][load_cut],file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(0,3)][load_cut],file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(1,2)][load_cut],file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(1,3)][load_cut],file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(2,3)][load_cut])).T.T
                        print('\n\n')
                        print(known_flight)
                        print(hpol_delays)
                        print(vpol_delays)
                        print(hpol_corrs)
                        print(vpol_corrs)


                    except Exception as e:
                        file.close()
                        print('Error in plotting.')
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)

                if run_index == 0:
                    times = calibrated_trigtime
                    total_hpol_delays = hpol_delays
                    total_vpol_delays = vpol_delays

                    total_hpol_corrs = hpol_corrs
                    total_vpol_corrs = vpol_corrs

                else:
                    times =  numpy.append(times,calibrated_trigtime)
                    total_hpol_delays = numpy.vstack((total_hpol_delays,hpol_delays))
                    total_vpol_delays = numpy.vstack((total_vpol_delays,vpol_delays))

                    total_hpol_corrs = numpy.vstack((total_hpol_corrs,hpol_corrs))
                    total_vpol_corrs = numpy.vstack((total_vpol_corrs,vpol_corrs))





            if plot_time_delays or plot_plane_tracks:
                if plot_plane_tracks == True:
                    try:
                        min_timestamp = min(times)
                        max_timestamp = max(times)
                        if known_flight is not None:
                            flight_tracks_ENU, all_vals = pt.getENUTrackDict(min_timestamp,max_timestamp,100,hour_window = 0,flights_of_interest=[known_flight])
                        else:
                            flight_tracks_ENU, all_vals = pt.getENUTrackDict(min_timestamp,max_timestamp,100,hour_window = 0,flights_of_interest=[])
                    except Exception as e:
                        print('Error in Getting Plane Tracks.')
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                

                for pol in ['Hpol','Vpol']:
                    time_delay_fig = plt.figure()
                    time_delay_fig.canvas.set_window_title('%s Delays'%pol)
                    plt.minorticks_on()
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                    plt.title(pol + ' ' + candidates_key)
                    plt.ylabel('Time Delay (ns)')
                    plt.xlabel('Readout Time (s)')

                    thresh = 0.1

                    python_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

                    for pair_index, pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
                        cut = total_hpol_corrs[:,pair_index] > thresh
                        if pol == 'Hpol':
                            y = total_hpol_delays[cut,pair_index]
                        else:
                            y = total_vpol_delays[cut,pair_index]

                        plt.plot(times[cut], y,c=python_colors[pair_index],linestyle = '--',alpha=0.8)
                        plt.scatter(times[cut], y,c=python_colors[pair_index],label='Measured Time Delays for A%i and A%i'%(pair[0],pair[1]))

                        if plot_plane_tracks == True:
                            plot_distance_cut_limit = None
                            existing_locations_A = numpy.array([])
                            existing_locations_B = numpy.array([])

                            for flight in list(flight_tracks_ENU.keys()):
                                track = flight_tracks_ENU[flight]
                                tof, dof, dt = pt.getTimeDelaysFromTrack(track)
                                distance = numpy.sqrt(numpy.sum(track[:,0:3]**2,axis=1))/1000 #km

                                if plot_distance_cut_limit is not None:
                                    plot_distance_cut = distance <= plot_distance_cut_limit
                                else:
                                    plot_distance_cut = numpy.ones_like(distance,dtype=bool)

                                x = track[plot_distance_cut,3]

                                #cable_delays = info.loadCableDelays()[mode]


                                y = dt['expected_time_differences_hpol'][(pair[0], pair[1])][plot_distance_cut]
                                if known_flight is not None:
                                    plt.plot(x,y,c=python_colors[pair_index],linestyle = '--',alpha=0.5,label='Flight %s TD: A%i and A%i'%(known_flight,pair[0],pair[1]))
                                    plt.scatter(x,y,facecolors='none', edgecolors=python_colors[pair_index],alpha=0.4)
                                else:
                                    plt.plot(x,y,c=python_colors[pair_index],linestyle = '--',alpha=0.2)
                                    plt.scatter(x,y,facecolors='none', edgecolors=python_colors[pair_index],alpha=0.2)

                                text_color = plt.gca().lines[-1].get_color()

                                #Attempt at avoiding overlapping text.
                                text_loc = numpy.array([numpy.mean(x)-5,numpy.mean(y)])
                                if existing_locations_A.size != 0:
                                    if len(numpy.shape(existing_locations_A)) == 1:
                                        dist = numpy.sqrt((text_loc[0]-existing_locations_A[0])**2 + (text_loc[1]-existing_locations_A[1])**2) #weird units but works
                                        while dist < 15:
                                            text_loc[1] -= 1
                                            dist = numpy.sqrt((text_loc[0]-existing_locations_A[0])**2 + (text_loc[1]-existing_locations_A[1])**2) #weird units but works
                                    else:
                                        dist = min(numpy.sqrt((existing_locations_A[:,0] - text_loc[0])**2 + (existing_locations_A[:,1] - text_loc[1])**2))
                                        while dist < 15:
                                            text_loc[1] -= 1
                                            dist = min(numpy.sqrt((existing_locations_A[:,0] - text_loc[0])**2 + (existing_locations_A[:,1] - text_loc[1])**2)) #weird units but works
                                    existing_locations_A = numpy.vstack((existing_locations_A,text_loc))
                                else:
                                    existing_locations_A = text_loc           


                                plt.text(text_loc[0],text_loc[1],flight,color=text_color,withdash=True)

                                if known_flight is None:
                                    pass
                                    #plt.xlim(min(times) - 300, max(times) + 300)

                    plt.legend(loc='upper right')


                        
    except Exception as e:
        print('Error in plotting.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    


'''
#ON FRIDAY I RAN THIS SCRIPT ATTEMPTING TO GET TIME DELAYS.  I GOT UP TO 16/27.  NEED TO DO THE REST BY EYE.
#HERE IS THE PRINTOUT

l_corrs, align_method, print_warning)
    768                         self.persistent_object[fig_index].canvas.draw_idle()
    769 
--> 770 
    771                     slider_roll.on_changed(update)
    772 

KeyboardInterrupt: 

In [29]:                                                                       

In [29]:                                                                       

In [29]: plt.close('all')                                                      

In [30]:                                                                       

In [30]:                                                                       

In [30]: %run analyze_plane_candidates.py                                      
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1728_analysis_data.h5 already exists, checking if setup is up to date.

Processing Run 1728
None given as eventid, setting event 0 as template.
Event 62577 set as template
Calculating time delays:
(6/6)           



a44585
[[  99.36190078   25.44653118   96.07746836  -73.43052481   -2.83086794
    70.61529703]
 [  91.29158112    8.25800151   78.10693097  -82.84589775  -13.15336983
    70.09917193]
 [  69.94277039  -26.74466399   37.41124928  -96.76563515  -32.45332034
    64.24975419]
 [  68.09723217  -29.07504699   34.68986241  -97.21919963  -33.14148713
    63.68670863]
 [  37.28612804  -65.26636423  -13.93537755 -102.66197335  -50.20489556
    51.69071023]
 [  31.93719524  -69.58304684  -20.25399992 -101.70792393  -51.92531254
    49.12572491]]
[[ 102.52121196   23.36639064  102.63069304  -79.07662054    0.12512124
    79.32686301]
 [  94.66985446    5.77121697   84.08146995  -88.88299734  -10.61966482
    78.31025298]
 [  73.3992445   -29.96653579   43.11990563 -103.28757952  -30.27933888
    73.03952095]
 [  71.64754721  -32.15615741   40.3515983  -103.80370462  -31.45235046
    72.44519509]
 [  38.80322302  -67.95647078   -9.36845247 -106.74405364  -48.10911487
    58.556738  ]
 [  33.72017284  -72.27315339  -16.18755978 -105.99332623  -49.84517201
    56.03867314]]
[[0.54296347 0.66401425 0.5290622  0.46341469 0.57353087 0.68803661]
 [0.72755695 0.73327106 0.67828298 0.71050678 0.740176   0.7255736 ]
 [0.78951981 0.87576484 0.91596379 0.79068938 0.80368587 0.8066493 ]
 [0.73420813 0.86623596 0.79277576 0.75192662 0.77947303 0.72376367]
 [0.55723343 0.85686536 0.74134153 0.63097564 0.71972242 0.70229345]
 [0.62062809 0.83165972 0.74880425 0.61628435 0.74726736 0.75501151]]
[[0.74859039 0.72094946 0.68496901 0.77969161 0.85019343 0.77799921]
 [0.8244373  0.83915634 0.8180681  0.84242355 0.89382184 0.83890872]
 [0.88868985 0.90940108 0.89158618 0.88425718 0.87083196 0.89345349]
 [0.88184143 0.92427453 0.87074308 0.89623232 0.86348606 0.88157825]
 [0.91547578 0.93912025 0.90280934 0.91923303 0.92003877 0.89160619]
 [0.9403921  0.94107688 0.90800336 0.93372543 0.91330014 0.90518039]]
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1773_analysis_data.h5 already exists, checking if setup is up to date.

Processing Run 1773
None given as eventid, setting event 0 as template.
Event 14590 set as template
Calculating time delays:
(3/3)           



aa8c39
[[ 91.08825911 109.19955788 144.42118554  18.90894665  53.00448319
   33.67325238]
 [108.0265463   99.76854479 150.20804266  -8.05467951  42.29097744
   50.34565695]
 [111.38917949  97.04715793 149.94216004 -14.2951011   38.70938209
   52.69168011]]
[[ -14.06049879 -254.52787238   40.21083691   10.44762313   55.74151021
    14.51406326]
 [-212.42457679  -50.09541448 -132.1280242   -13.68513508   44.94980369
    59.1041434 ]
 [  12.76236597  222.55939683   77.87232866  -21.06728795   55.00642295
  -237.54266472]]
[[0.90272377 0.60064537 0.87741312 0.67375529 0.89613162 0.59279454]
 [0.9031536  0.88038772 0.93706705 0.88307895 0.94245707 0.896344  ]
 [0.71921489 0.74597444 0.79102785 0.7537876  0.63843518 0.79467544]]
[[0.10380397 0.06419393 0.0760841  0.43932523 0.47531169 0.51327703]
 [0.06753853 0.08494523 0.06685715 0.50694484 0.47081591 0.42630021]
 [0.06910693 0.09430286 0.0840499  0.24675524 0.32500253 0.19712771]]
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1773_analysis_data.h5 already exists, checking if setup is up to date.

Processing Run 1773
None given as eventid, setting event 0 as template.
Event 63757 set as template
Calculating time delays:
(5/5)           



a28392
[[ 93.40300196  12.60596443  77.8566885  -80.84395799 -15.5306733
   65.28200438]
 [ 94.54473323  12.88748721  79.01405993 -81.71980663 -15.59323392
   66.14221287]
 [ 95.07649848  13.09080922  79.54582518 -82.03260972 -15.54631345
   66.51757658]
 [ 95.8897865   13.24721076  80.07759042 -82.61129543 -15.85911654
   67.00242136]
 [ 95.82722589  13.16900999  80.09323058 -82.67385605 -15.71835515
   67.00242136]]
[[ 442.74149019  355.8291523   429.71324159  -87.74126607  -13.51309338
    74.16561207]
 [  10.02533896  596.48420774  672.37023679  -88.60147456  -13.38797215
    75.08838118]
 [ 297.75725908  590.44710815  284.30672632  -88.85171703  -13.38797215
    75.44810473]
 [  12.96568798   71.71010783 -156.35462334  -89.2740012   -12.82492659
    90.13420969]
 [  -5.94325866  198.31715754  232.099891    -89.33656182  -13.30977138
    75.91730936]]
[[0.95098351 0.95833601 0.94619827 0.93922142 0.93210956 0.90940208]
 [0.92373364 0.9104255  0.94496063 0.91447291 0.93351235 0.90544071]
 [0.95289233 0.94974068 0.93237005 0.94604289 0.92418265 0.90493278]
 [0.83174037 0.7827824  0.88097779 0.79032702 0.86531262 0.80427036]
 [0.87302171 0.88326716 0.8874286  0.88267858 0.91460527 0.89029115]]
[[0.06289992 0.05902408 0.06221534 0.94319635 0.89536839 0.89388475]
 [0.09225186 0.08122064 0.08065197 0.86999957 0.85958327 0.84102569]
 [0.06772418 0.07353492 0.08277247 0.92582262 0.89727331 0.88631299]
 [0.07894339 0.06906414 0.09807917 0.45253002 0.52531616 0.36415181]
 [0.06454302 0.05635923 0.07649677 0.82547947 0.82546989 0.76452701]]
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1774_analysis_data.h5 already exists, checking if setup is up to date.

Processing Run 1774
None given as eventid, setting event 0 as template.
Event 89213 set as template
Calculating time delays:
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)
QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)
QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)
QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)
QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)
QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)
QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1847 chosen representing time delay of -28.887 ns
Corresponding correlation value of 0.630 (max = 0.727)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1091 chosen representing time delay of 17.063 ns
Corresponding correlation value of 0.777 (max = 0.777)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -753 chosen representing time delay of -11.777 ns
Corresponding correlation value of 0.752 (max = 0.752)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2912 chosen representing time delay of 45.544 ns
Corresponding correlation value of 0.767 (max = 0.767)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1061 chosen representing time delay of 16.594 ns
Corresponding correlation value of 0.600 (max = 0.600)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1850 chosen representing time delay of -28.934 ns
Corresponding correlation value of 0.648 (max = 0.683)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -23678 chosen representing time delay of -370.328 ns
Corresponding correlation value of 0.092 (max = 0.092)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 39697 chosen representing time delay of 620.867 ns
Corresponding correlation value of 0.075 (max = 0.075)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 18692 chosen representing time delay of 292.346 ns
Corresponding correlation value of 0.076 (max = 0.076)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2483 chosen representing time delay of 38.835 ns
Corresponding correlation value of 0.629 (max = 0.629)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1276 chosen representing time delay of 19.957 ns
Corresponding correlation value of 0.683 (max = 0.683)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1191 chosen representing time delay of -18.627 ns
Corresponding correlation value of 0.684 (max = 0.684)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1638 chosen representing time delay of -25.619 ns
Corresponding correlation value of 0.577 (max = 0.723)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1009 chosen representing time delay of 15.781 ns
Corresponding correlation value of 0.706 (max = 0.706)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 131 chosen representing time delay of 2.049 ns
Corresponding correlation value of 0.576 (max = 0.662)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1821 chosen representing time delay of 28.481 ns
Corresponding correlation value of 0.803 (max = 0.803)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 929 chosen representing time delay of 14.530 ns
Corresponding correlation value of 0.695 (max = 0.740)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -788 chosen representing time delay of -12.324 ns
Corresponding correlation value of 0.540 (max = 0.661)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 0 chosen representing time delay of 0.000 ns
Corresponding correlation value of 0.075 (max = 0.075)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2294 chosen representing time delay of 35.879 ns
Corresponding correlation value of 0.062 (max = 0.062)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3848 chosen representing time delay of 60.183 ns
Corresponding correlation value of 0.095 (max = 0.095)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2323 chosen representing time delay of 36.332 ns
Corresponding correlation value of 0.795 (max = 0.802)
If satisfied with current slider location, press Enter to lock it down.
int of 1150 chosen representing time delay of 17.986 ns
Corresponding correlation value of 0.603 (max = 0.667)
If satisfied with current slider location, press Enter to lock it down.
int of -1167 chosen representing time delay of -18.252 ns
Corresponding correlation value of 0.577 (max = 0.677)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1614 chosen representing time delay of -25.243 ns
Corresponding correlation value of 0.673 (max = 0.809)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1000 chosen representing time delay of 15.640 ns
Corresponding correlation value of 0.774 (max = 0.803)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -641 chosen representing time delay of -10.025 ns
Corresponding correlation value of 0.794 (max = 0.794)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2573 chosen representing time delay of 40.242 ns
Corresponding correlation value of 0.853 (max = 0.853)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1786 chosen representing time delay of 27.933 ns
Corresponding correlation value of 0.807 (max = 0.807)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1634 chosen representing time delay of -25.556 ns
Corresponding correlation value of 0.829 (max = 0.829)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 77970 chosen representing time delay of 1219.463 ns
Corresponding correlation value of 0.077 (max = 0.077)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -7274 chosen representing time delay of -113.766 ns
Corresponding correlation value of 0.059 (max = 0.059)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -4611 chosen representing time delay of -72.117 ns
Corresponding correlation value of 0.081 (max = 0.081)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2187 chosen representing time delay of 34.205 ns
Corresponding correlation value of 0.747 (max = 0.800)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1181 chosen representing time delay of 18.471 ns
Corresponding correlation value of 0.632 (max = 0.632)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1016 chosen representing time delay of -15.890 ns
Corresponding correlation value of 0.590 (max = 0.636)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -42 chosen representing time delay of -0.657 ns
Corresponding correlation value of 0.789 (max = 0.789)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 873 chosen representing time delay of 13.654 ns
Corresponding correlation value of 0.849 (max = 0.849)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 425 chosen representing time delay of 6.647 ns
Corresponding correlation value of 0.829 (max = 0.829)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 909 chosen representing time delay of 14.217 ns
Corresponding correlation value of 0.831 (max = 0.831)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 491 chosen representing time delay of 7.679 ns
Corresponding correlation value of 0.843 (max = 0.843)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -445 chosen representing time delay of -6.960 ns
Corresponding correlation value of 0.860 (max = 0.860)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2973 chosen representing time delay of -46.498 ns
Corresponding correlation value of 0.062 (max = 0.062)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2453 chosen representing time delay of -38.365 ns
Corresponding correlation value of 0.071 (max = 0.071)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -17935 chosen representing time delay of -280.506 ns
Corresponding correlation value of 0.083 (max = 0.083)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2012 chosen representing time delay of -31.468 ns
Corresponding correlation value of 0.097 (max = 0.331)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 652 chosen representing time delay of 10.197 ns
Corresponding correlation value of 0.380 (max = 0.380)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 143 chosen representing time delay of 2.237 ns
Corresponding correlation value of 0.238 (max = 0.238)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 196 chosen representing time delay of 3.065 ns
Corresponding correlation value of 0.842 (max = 0.842)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 844 chosen representing time delay of 13.200 ns
Corresponding correlation value of 0.841 (max = 0.841)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 607 chosen representing time delay of 9.494 ns
Corresponding correlation value of 0.887 (max = 0.887)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 637 chosen representing time delay of 9.963 ns
Corresponding correlation value of 0.850 (max = 0.850)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 414 chosen representing time delay of 6.475 ns
Corresponding correlation value of 0.849 (max = 0.849)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -239 chosen representing time delay of -3.738 ns
Corresponding correlation value of 0.833 (max = 0.833)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1147 chosen representing time delay of -17.939 ns
Corresponding correlation value of 0.102 (max = 0.102)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2036 chosen representing time delay of -31.843 ns
Corresponding correlation value of 0.079 (max = 0.079)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3331 chosen representing time delay of -52.097 ns
Corresponding correlation value of 0.099 (max = 0.099)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 7777 chosen representing time delay of 121.633 ns
Corresponding correlation value of 0.212 (max = 0.212)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2100 chosen representing time delay of 32.844 ns
Corresponding correlation value of 0.302 (max = 0.302)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 5097 chosen representing time delay of 79.718 ns
Corresponding correlation value of 0.241 (max = 0.241)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 370 chosen representing time delay of 5.787 ns
Corresponding correlation value of 0.772 (max = 0.826)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 855 chosen representing time delay of 13.372 ns
Corresponding correlation value of 0.778 (max = 0.778)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 726 chosen representing time delay of 11.355 ns
Corresponding correlation value of 0.863 (max = 0.863)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 451 chosen representing time delay of 7.054 ns
Corresponding correlation value of 0.806 (max = 0.806)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 370 chosen representing time delay of 5.787 ns
Corresponding correlation value of 0.831 (max = 0.843)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -111 chosen representing time delay of -1.736 ns
Corresponding correlation value of 0.773 (max = 0.773)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 299 chosen representing time delay of 4.676 ns
Corresponding correlation value of 0.082 (max = 0.082)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 20960 chosen representing time delay of 327.818 ns
Corresponding correlation value of 0.057 (max = 0.057)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 27664 chosen representing time delay of 432.669 ns
Corresponding correlation value of 0.086 (max = 0.086)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -10218 chosen representing time delay of -159.811 ns
Corresponding correlation value of 0.206 (max = 0.206)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2194 chosen representing time delay of -34.314 ns
Corresponding correlation value of 0.405 (max = 0.405)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 12498 chosen representing time delay of 195.471 ns
Corresponding correlation value of 0.200 (max = 0.200)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 357 chosen representing time delay of 5.584 ns
Corresponding correlation value of 0.758 (max = 0.758)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 844 chosen representing time delay of 13.200 ns
Corresponding correlation value of 0.757 (max = 0.757)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 721 chosen representing time delay of 11.277 ns
Corresponding correlation value of 0.740 (max = 0.740)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -441 chosen representing time delay of -6.897 ns
Corresponding correlation value of 0.706 (max = 0.720)
If satisfied with current slider location, press Enter to lock it down.
int of 359 chosen representing time delay of 5.615 ns
Corresponding correlation value of 0.673 (max = 0.756)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -109 chosen representing time delay of -1.705 ns
Corresponding correlation value of 0.666 (max = 0.666)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2943 chosen representing time delay of 46.029 ns
Corresponding correlation value of 0.075 (max = 0.075)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -8453 chosen representing time delay of -132.206 ns
Corresponding correlation value of 0.085 (max = 0.085)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -25104 chosen representing time delay of -392.630 ns
Corresponding correlation value of 0.085 (max = 0.085)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -951 chosen representing time delay of -14.874 ns
Corresponding correlation value of 0.358 (max = 0.358)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2994 chosen representing time delay of -46.827 ns
Corresponding correlation value of 0.416 (max = 0.416)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1128 chosen representing time delay of -17.642 ns
Corresponding correlation value of 0.220 (max = 0.325)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 444 chosen representing time delay of 6.944 ns
Corresponding correlation value of 0.638 (max = 0.750)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 856 chosen representing time delay of 13.388 ns
Corresponding correlation value of 0.840 (max = 0.840)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 840 chosen representing time delay of 13.138 ns
Corresponding correlation value of 0.865 (max = 0.865)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 340 chosen representing time delay of 5.318 ns
Corresponding correlation value of 0.815 (max = 0.815)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 307 chosen representing time delay of 4.802 ns
Corresponding correlation value of 0.661 (max = 0.754)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -5 chosen representing time delay of -0.078 ns
Corresponding correlation value of 0.796 (max = 0.796)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2344 chosen representing time delay of -36.661 ns
Corresponding correlation value of 0.055 (max = 0.055)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 269 chosen representing time delay of 4.207 ns
Corresponding correlation value of 0.074 (max = 0.074)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 38670 chosen representing time delay of 604.805 ns
Corresponding correlation value of 0.087 (max = 0.087)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -904 chosen representing time delay of -14.139 ns
Corresponding correlation value of 0.592 (max = 0.592)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 545 chosen representing time delay of 8.524 ns
Corresponding correlation value of 0.624 (max = 0.624)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1461 chosen representing time delay of 22.850 ns
Corresponding correlation value of 0.559 (max = 0.607)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 966 chosen representing time delay of 15.108 ns
Corresponding correlation value of 0.748 (max = 0.789)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 811 chosen representing time delay of 12.684 ns
Corresponding correlation value of 0.761 (max = 0.761)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1214 chosen representing time delay of 18.987 ns
Corresponding correlation value of 0.747 (max = 0.858)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -214 chosen representing time delay of -3.347 ns
Corresponding correlation value of 0.819 (max = 0.819)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 185 chosen representing time delay of 2.893 ns
Corresponding correlation value of 0.839 (max = 0.849)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 443 chosen representing time delay of 6.929 ns
Corresponding correlation value of 0.656 (max = 0.803)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 9 chosen representing time delay of 0.141 ns
Corresponding correlation value of 0.078 (max = 0.078)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 22799 chosen representing time delay of 356.580 ns
Corresponding correlation value of 0.054 (max = 0.054)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -13309 chosen representing time delay of -208.155 ns
Corresponding correlation value of 0.086 (max = 0.086)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -818 chosen representing time delay of -12.794 ns
Corresponding correlation value of 0.057 (max = 0.339)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -529 chosen representing time delay of -8.274 ns
Corresponding correlation value of 0.045 (max = 0.413)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1217 chosen representing time delay of 19.034 ns
Corresponding correlation value of 0.403 (max = 0.403)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1384 chosen representing time delay of 21.646 ns
Corresponding correlation value of 0.852 (max = 0.852)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 760 chosen representing time delay of 11.887 ns
Corresponding correlation value of 0.818 (max = 0.818)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1467 chosen representing time delay of 22.944 ns
Corresponding correlation value of 0.863 (max = 0.863)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -633 chosen representing time delay of -9.900 ns
Corresponding correlation value of 0.863 (max = 0.863)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 78 chosen representing time delay of 1.220 ns
Corresponding correlation value of 0.849 (max = 0.849)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 707 chosen representing time delay of 11.058 ns
Corresponding correlation value of 0.845 (max = 0.845)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3197 chosen representing time delay of 50.002 ns
Corresponding correlation value of 0.085 (max = 0.085)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2974 chosen representing time delay of 46.514 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 9220 chosen representing time delay of 144.202 ns
Corresponding correlation value of 0.108 (max = 0.108)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -117 chosen representing time delay of -1.830 ns
Corresponding correlation value of 0.190 (max = 0.190)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3710 chosen representing time delay of -58.025 ns
Corresponding correlation value of 0.241 (max = 0.241)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -82 chosen representing time delay of -1.282 ns
Corresponding correlation value of 0.225 (max = 0.225)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1419 chosen representing time delay of 22.193 ns
Corresponding correlation value of 0.817 (max = 0.832)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 787 chosen representing time delay of 12.309 ns
Corresponding correlation value of 0.758 (max = 0.758)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1485 chosen representing time delay of 23.226 ns
Corresponding correlation value of 0.870 (max = 0.870)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -639 chosen representing time delay of -9.994 ns
Corresponding correlation value of 0.840 (max = 0.840)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 71 chosen representing time delay of 1.110 ns
Corresponding correlation value of 0.878 (max = 0.878)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 702 chosen representing time delay of 10.979 ns
Corresponding correlation value of 0.820 (max = 0.820)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -17045 chosen representing time delay of -266.586 ns
Corresponding correlation value of 0.107 (max = 0.107)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -18255 chosen representing time delay of -285.511 ns
Corresponding correlation value of 0.062 (max = 0.062)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -17680 chosen representing time delay of -276.518 ns
Corresponding correlation value of 0.114 (max = 0.114)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -472 chosen representing time delay of -7.382 ns
Corresponding correlation value of 0.228 (max = 0.341)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 180 chosen representing time delay of 2.815 ns
Corresponding correlation value of 0.139 (max = 0.333)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 696 chosen representing time delay of 10.886 ns
Corresponding correlation value of 0.480 (max = 0.480)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2307 chosen representing time delay of 36.082 ns
Corresponding correlation value of 0.732 (max = 0.732)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 717 chosen representing time delay of 11.214 ns
Corresponding correlation value of 0.770 (max = 0.770)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2965 chosen representing time delay of 46.373 ns
Corresponding correlation value of 0.857 (max = 0.857)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1603 chosen representing time delay of -25.071 ns
Corresponding correlation value of 0.856 (max = 0.856)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -156 chosen representing time delay of -2.440 ns
Corresponding correlation value of 0.683 (max = 0.770)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2284 chosen representing time delay of 35.722 ns
Corresponding correlation value of 0.741 (max = 0.775)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 914 chosen representing time delay of 14.295 ns
Corresponding correlation value of 0.076 (max = 0.076)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 25241 chosen representing time delay of 394.773 ns
Corresponding correlation value of 0.072 (max = 0.072)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 21002 chosen representing time delay of 328.475 ns
Corresponding correlation value of 0.118 (max = 0.118)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2032 chosen representing time delay of -31.781 ns
Corresponding correlation value of 0.236 (max = 0.236)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 8656 chosen representing time delay of 135.381 ns
Corresponding correlation value of 0.254 (max = 0.254)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2771 chosen representing time delay of -43.339 ns
Corresponding correlation value of 0.195 (max = 0.195)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2376 chosen representing time delay of 37.161 ns
Corresponding correlation value of 0.767 (max = 0.767)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 692 chosen representing time delay of 10.823 ns
Corresponding correlation value of 0.766 (max = 0.774)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3016 chosen representing time delay of 47.171 ns
Corresponding correlation value of 0.868 (max = 0.868)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1676 chosen representing time delay of -26.213 ns
Corresponding correlation value of 0.850 (max = 0.850)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -182 chosen representing time delay of -2.847 ns
Corresponding correlation value of 0.685 (max = 0.763)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1504 chosen representing time delay of 23.523 ns
Corresponding correlation value of 0.674 (max = 0.777)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -388 chosen representing time delay of -6.068 ns
Corresponding correlation value of 0.076 (max = 0.076)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -14018 chosen representing time delay of -219.244 ns
Corresponding correlation value of 0.073 (max = 0.073)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 16392 chosen representing time delay of 256.373 ns
Corresponding correlation value of 0.121 (max = 0.121)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3015 chosen representing time delay of -47.155 ns
Corresponding correlation value of 0.220 (max = 0.220)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -28 chosen representing time delay of -0.438 ns
Corresponding correlation value of 0.278 (max = 0.278)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2769 chosen representing time delay of -43.308 ns
Corresponding correlation value of 0.279 (max = 0.279)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3446 chosen representing time delay of 53.896 ns
Corresponding correlation value of 0.788 (max = 0.788)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 707 chosen representing time delay of 11.058 ns
Corresponding correlation value of 0.826 (max = 0.826)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2967 chosen representing time delay of 46.404 ns
Corresponding correlation value of 0.875 (max = 0.875)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2739 chosen representing time delay of -42.838 ns
Corresponding correlation value of 0.875 (max = 0.875)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -476 chosen representing time delay of -7.445 ns
Corresponding correlation value of 0.844 (max = 0.844)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2260 chosen representing time delay of 35.347 ns
Corresponding correlation value of 0.855 (max = 0.855)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 5257 chosen representing time delay of 82.220 ns
Corresponding correlation value of 0.097 (max = 0.097)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -14022 chosen representing time delay of -219.306 ns
Corresponding correlation value of 0.082 (max = 0.082)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 38399 chosen representing time delay of 600.566 ns
Corresponding correlation value of 0.072 (max = 0.072)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3132 chosen representing time delay of -48.985 ns
Corresponding correlation value of 0.341 (max = 0.341)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 641 chosen representing time delay of 10.025 ns
Corresponding correlation value of 0.311 (max = 0.311)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 1428 chosen representing time delay of 22.334 ns
Corresponding correlation value of 0.274 (max = 0.274)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3399 chosen representing time delay of 53.161 ns
Corresponding correlation value of 0.846 (max = 0.866)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 701 chosen representing time delay of 10.964 ns
Corresponding correlation value of 0.847 (max = 0.847)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2946 chosen representing time delay of 46.076 ns
Corresponding correlation value of 0.933 (max = 0.933)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2729 chosen representing time delay of -42.682 ns
Corresponding correlation value of 0.940 (max = 0.940)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)
 
int of -470 chosen representing time delay of -7.351 ns
Corresponding correlation value of 0.928 (max = 0.928)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2249 chosen representing time delay of 35.175 ns
Corresponding correlation value of 0.891 (max = 0.891)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 66618 chosen representing time delay of 1041.916 ns
Corresponding correlation value of 0.047 (max = 0.047)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 51470 chosen representing time delay of 804.999 ns
Corresponding correlation value of 0.056 (max = 0.056)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 41221 chosen representing time delay of 644.703 ns
Corresponding correlation value of 0.071 (max = 0.071)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3130 chosen representing time delay of -48.954 ns
Corresponding correlation value of 0.734 (max = 0.734)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -307 chosen representing time delay of -4.802 ns
Corresponding correlation value of 0.687 (max = 0.687)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2802 chosen representing time delay of 43.824 ns
Corresponding correlation value of 0.719 (max = 0.719)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3521 chosen representing time delay of 55.069 ns
Corresponding correlation value of 0.863 (max = 0.871)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 695 chosen representing time delay of 10.870 ns
Corresponding correlation value of 0.851 (max = 0.851)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3010 chosen representing time delay of 47.077 ns
Corresponding correlation value of 0.934 (max = 0.934)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2824 chosen representing time delay of -44.168 ns
Corresponding correlation value of 0.933 (max = 0.933)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -494 chosen representing time delay of -7.726 ns
Corresponding correlation value of 0.933 (max = 0.933)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2320 chosen representing time delay of 36.285 ns
Corresponding correlation value of 0.897 (max = 0.897)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 18444 chosen representing time delay of 288.467 ns
Corresponding correlation value of 0.064 (max = 0.064)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 5292 chosen representing time delay of 82.768 ns
Corresponding correlation value of 0.067 (max = 0.067)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -11273 chosen representing time delay of -176.311 ns
Corresponding correlation value of 0.099 (max = 0.099)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3197 chosen representing time delay of -50.002 ns
Corresponding correlation value of 0.710 (max = 0.729)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -312 chosen representing time delay of -4.880 ns
Corresponding correlation value of 0.656 (max = 0.706)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2880 chosen representing time delay of 45.044 ns
Corresponding correlation value of 0.735 (max = 0.735)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3538 chosen representing time delay of 55.335 ns
Corresponding correlation value of 0.843 (max = 0.843)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 692 chosen representing time delay of 10.823 ns
Corresponding correlation value of 0.842 (max = 0.842)

If satisfied with current slider location, press Enter to lock it down.
int of 3024 chosen representing time delay of 47.296 ns
Corresponding correlation value of 0.924 (max = 0.924)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2856 chosen representing time delay of -44.668 ns
Corresponding correlation value of 0.919 (max = 0.919)



If satisfied with current slider location, press Enter to lock it down.




int of -509 chosen representing time delay of -7.961 ns
Corresponding correlation value of 0.910 (max = 0.910)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2340 chosen representing time delay of 36.598 ns
Corresponding correlation value of 0.885 (max = 0.885)
If satisfied with current slider location, press Enter to lock it down.
int of 30970 chosen representing time delay of 484.376 ns
Corresponding correlation value of 0.057 (max = 0.057)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 41100 chosen representing time delay of 642.810 ns
Corresponding correlation value of 0.067 (max = 0.067)

If satisfied with current slider location, press Enter to lock it down.
int of 13062 chosen representing time delay of 204.292 ns
Corresponding correlation value of 0.080 (max = 0.080)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3241 chosen representing time delay of -50.690 ns
Corresponding correlation value of 0.424 (max = 0.424)




If satisfied with current slider location, press Enter to lock it down.
int of 627 chosen representing time delay of 9.806 ns
Corresponding correlation value of 0.408 (max = 0.408)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1 chosen representing time delay of -0.016 ns
Corresponding correlation value of 0.350 (max = 0.350)
(18/26)         

If satisfied with current slider location, press Enter to lock it down.

int of 4490 chosen representing time delay of 70.224 ns
Corresponding correlation value of 0.893 (max = 0.893)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)


int of 696 chosen representing time delay of 10.886 ns
Corresponding correlation value of 0.862 (max = 0.862)

If satisfied with current slider location, press Enter to lock it down.
int of 3094 chosen representing time delay of 48.391 ns
Corresponding correlation value of 0.950 (max = 0.950)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)





int of -3787 chosen representing time delay of -59.229 ns
Corresponding correlation value of 0.942 (max = 0.942)
If satisfied with current slider location, press Enter to lock it down.



int of -521 chosen representing time delay of -8.149 ns
Corresponding correlation value of 0.936 (max = 0.936)


If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2407 chosen representing time delay of 37.646 ns
Corresponding correlation value of 0.886 (max = 0.886)

If satisfied with current slider location, press Enter to lock it down.
int of 22515 chosen representing time delay of 352.138 ns
Corresponding correlation value of 0.069 (max = 0.069)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)


int of 67205 chosen representing time delay of 1051.097 ns
Corresponding correlation value of 0.075 (max = 0.075)

If satisfied with current slider location, press Enter to lock it down.
int of -22018 chosen representing time delay of -344.365 ns
Corresponding correlation value of 0.086 (max = 0.086)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -4234 chosen representing time delay of -66.220 ns
Corresponding correlation value of 0.787 (max = 0.787)





If satisfied with current slider location, press Enter to lock it down.

int of -354 chosen representing time delay of -5.537 ns
Corresponding correlation value of 0.786 (max = 0.786)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)


int of 2969 chosen representing time delay of 46.436 ns
Corresponding correlation value of 0.789 (max = 0.789)
If satisfied with current slider location, press Enter to lock it down.
int of 3706 chosen representing time delay of 57.962 ns
Corresponding correlation value of 0.904 (max = 0.904)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 703 chosen representing time delay of 10.995 ns
Corresponding correlation value of 0.882 (max = 0.882)

If satisfied with current slider location, press Enter to lock it down.
int of 3158 chosen representing time delay of 49.392 ns
Corresponding correlation value of 0.936 (max = 0.936)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3012 chosen representing time delay of -47.108 ns
Corresponding correlation value of 0.950 (max = 0.950)
If satisfied with current slider location, press Enter to lock it down.
int of -544 chosen representing time delay of -8.508 ns
Corresponding correlation value of 0.937 (max = 0.937)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2462 chosen representing time delay of 38.506 ns
Corresponding correlation value of 0.906 (max = 0.906)
If satisfied with current slider location, press Enter to lock it down.
int of 24446 chosen representing time delay of 382.339 ns
Corresponding correlation value of 0.067 (max = 0.067)













If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 22017 chosen representing time delay of 344.349 ns
Corresponding correlation value of 0.059 (max = 0.059)
If satisfied with current slider location, press Enter to lock it down.
int of 26883 chosen representing time delay of 420.454 ns
Corresponding correlation value of 0.077 (max = 0.077)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3420 chosen representing time delay of -53.489 ns
Corresponding correlation value of 0.736 (max = 0.736)
If satisfied with current slider location, press Enter to lock it down.
int of -387 chosen representing time delay of -6.053 ns
Corresponding correlation value of 0.720 (max = 0.720)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3021 chosen representing time delay of 47.249 ns
Corresponding correlation value of 0.760 (max = 0.760)
(20/26)         
If satisfied with current slider location, press Enter to lock it down.

int of 4663 chosen representing time delay of 72.930 ns
Corresponding correlation value of 0.878 (max = 0.878)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 694 chosen representing time delay of 10.854 ns
Corresponding correlation value of 0.877 (max = 0.877)
If satisfied with current slider location, press Enter to lock it down.
int of 4101 chosen representing time delay of 64.140 ns
Corresponding correlation value of 0.909 (max = 0.909)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3108 chosen representing time delay of -48.610 ns
Corresponding correlation value of 0.931 (max = 0.931)



If satisfied with current slider location, press Enter to lock it down.
int of -562 chosen representing time delay of -8.790 ns
Corresponding correlation value of 0.920 (max = 0.920)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 2537 chosen representing time delay of 39.679 ns
Corresponding correlation value of 0.900 (max = 0.900)

If satisfied with current slider location, press Enter to lock it down.
int of 132 chosen representing time delay of 2.065 ns
Corresponding correlation value of 0.063 (max = 0.063)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 40956 chosen representing time delay of 640.558 ns
Corresponding correlation value of 0.066 (max = 0.066)
If satisfied with current slider location, press Enter to lock it down.
int of 44039 chosen representing time delay of 688.777 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3489 chosen representing time delay of -54.568 ns
Corresponding correlation value of 0.769 (max = 0.769)
If satisfied with current slider location, press Enter to lock it down.
int of 521 chosen representing time delay of 8.149 ns
Corresponding correlation value of 0.750 (max = 0.750)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3085 chosen representing time delay of 48.250 ns
Corresponding correlation value of 0.790 (max = 0.790)
(21/26)         
If satisfied with current slider location, press Enter to lock it down.

int of 6342 chosen representing time delay of 99.190 ns
Corresponding correlation value of 0.608 (max = 0.608)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 672 chosen representing time delay of 10.510 ns
Corresponding correlation value of 0.760 (max = 0.760)

If satisfied with current slider location, press Enter to lock it down.
int of 4532 chosen representing time delay of 70.881 ns
Corresponding correlation value of 0.633 (max = 0.633)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -4796 chosen representing time delay of -75.010 ns
Corresponding correlation value of 0.666 (max = 0.666)

If satisfied with current slider location, press Enter to lock it down.
int of -964 chosen representing time delay of -15.077 ns
Corresponding correlation value of 0.583 (max = 0.583)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3836 chosen representing time delay of 59.996 ns
Corresponding correlation value of 0.658 (max = 0.658)

If satisfied with current slider location, press Enter to lock it down.


int of 120 chosen representing time delay of 1.877 ns
Corresponding correlation value of 0.093 (max = 0.093)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)


int of -5157 chosen representing time delay of -80.656 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.
int of 25353 chosen representing time delay of 396.525 ns
Corresponding correlation value of 0.083 (max = 0.083)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -5140 chosen representing time delay of -80.390 ns
Corresponding correlation value of 0.327 (max = 0.327)

If satisfied with current slider location, press Enter to lock it down.
int of -785 chosen representing time delay of -12.278 ns
Corresponding correlation value of 0.415 (max = 0.415)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4365 chosen representing time delay of 68.269 ns
Corresponding correlation value of 0.359 (max = 0.359)
If satisfied with current slider location, press Enter to lock it down.
int of 6388 chosen representing time delay of 99.909 ns
Corresponding correlation value of 0.715 (max = 0.715)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 680 chosen representing time delay of 10.635 ns
Corresponding correlation value of 0.892 (max = 0.892)
If satisfied with current slider location, press Enter to lock it down.
int of 4565 chosen representing time delay of 71.397 ns
Corresponding correlation value of 0.722 (max = 0.722)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)














int of -4855 chosen representing time delay of -75.933 ns
Corresponding correlation value of 0.820 (max = 0.820)
If satisfied with current slider location, press Enter to lock it down.

int of -981 chosen representing time delay of -15.343 ns
Corresponding correlation value of 0.801 (max = 0.801)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)


int of 3876 chosen representing time delay of 60.621 ns
Corresponding correlation value of 0.782 (max = 0.782)
If satisfied with current slider location, press Enter to lock it down.

int of -4786 chosen representing time delay of -74.854 ns
Corresponding correlation value of 0.069 (max = 0.069)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -11662 chosen representing time delay of -182.395 ns
Corresponding correlation value of 0.083 (max = 0.083)
If satisfied with current slider location, press Enter to lock it down.
int of 56085 chosen representing time delay of 877.178 ns
Corresponding correlation value of 0.086 (max = 0.086)


If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)


int of -5220 chosen representing time delay of -81.642 ns
Corresponding correlation value of 0.761 (max = 0.761)
If satisfied with current slider location, press Enter to lock it down.
int of -802 chosen representing time delay of -12.543 ns
Corresponding correlation value of 0.825 (max = 0.825)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4418 chosen representing time delay of 69.098 ns
Corresponding correlation value of 0.733 (max = 0.733)
(23/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 7275 chosen representing time delay of 113.782 ns
Corresponding correlation value of 0.717 (max = 0.717)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 687 chosen representing time delay of 10.745 ns
Corresponding correlation value of 0.832 (max = 0.832)
If satisfied with current slider location, press Enter to lock it down.
int of 6311 chosen representing time delay of 98.705 ns
Corresponding correlation value of 0.735 (max = 0.735)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -4862 chosen representing time delay of -76.042 ns
Corresponding correlation value of 0.783 (max = 0.783)

If satisfied with current slider location, press Enter to lock it down.
int of -977 chosen representing time delay of -15.280 ns
Corresponding correlation value of 0.730 (max = 0.730)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3890 chosen representing time delay of 60.840 ns
Corresponding correlation value of 0.787 (max = 0.787)
If satisfied with current slider location, press Enter to lock it down.
int of 6180 chosen representing time delay of 96.656 ns
Corresponding correlation value of 0.079 (max = 0.079)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -25844 chosen representing time delay of -404.204 ns
Corresponding correlation value of 0.066 (max = 0.066)
If satisfied with current slider location, press Enter to lock it down.
int of 26382 chosen representing time delay of 412.619 ns
Corresponding correlation value of 0.082 (max = 0.082)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)


int of -5229 chosen representing time delay of -81.782 ns
Corresponding correlation value of 0.755 (max = 0.755)

If satisfied with current slider location, press Enter to lock it down.
int of -783 chosen representing time delay of -12.246 ns
Corresponding correlation value of 0.754 (max = 0.754)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4440 chosen representing time delay of 69.442 ns
Corresponding correlation value of 0.727 (max = 0.727)
(24/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 6553 chosen representing time delay of 102.490 ns
Corresponding correlation value of 0.788 (max = 0.788)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 740 chosen representing time delay of 11.574 ns
Corresponding correlation value of 0.843 (max = 0.843)
If satisfied with current slider location, press Enter to lock it down.
int of 5546 chosen representing time delay of 86.740 ns
Corresponding correlation value of 0.786 (max = 0.786)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -4950 chosen representing time delay of -77.419 ns
Corresponding correlation value of 0.801 (max = 0.801)

If satisfied with current slider location, press Enter to lock it down.
int of -999 chosen representing time delay of -15.625 ns
Corresponding correlation value of 0.755 (max = 0.755)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3944 chosen representing time delay of 61.685 ns
Corresponding correlation value of 0.808 (max = 0.808)

If satisfied with current slider location, press Enter to lock it down.

int of 41849 chosen representing time delay of 654.525 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 36471 chosen representing time delay of 570.412 ns
Corresponding correlation value of 0.058 (max = 0.058)


If satisfied with current slider location, press Enter to lock it down.
int of 53256 chosen representing time delay of 832.932 ns
Corresponding correlation value of 0.070 (max = 0.070)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)


int of -5365 chosen representing time delay of -83.909 ns
Corresponding correlation value of 0.930 (max = 0.930)
If satisfied with current slider location, press Enter to lock it down.
int of -823 chosen representing time delay of -12.872 ns
Corresponding correlation value of 0.916 (max = 0.916)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4540 chosen representing time delay of 71.006 ns
Corresponding correlation value of 0.907 (max = 0.907)
If satisfied with current slider location, press Enter to lock it down.
int of 7402 chosen representing time delay of 115.768 ns
Corresponding correlation value of 0.727 (max = 0.727)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 745 chosen representing time delay of 11.652 ns
Corresponding correlation value of 0.848 (max = 0.848)
If satisfied with current slider location, press Enter to lock it down.
int of 6414 chosen representing time delay of 100.316 ns
Corresponding correlation value of 0.777 (max = 0.777)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)


int of -4941 chosen representing time delay of -77.278 ns
Corresponding correlation value of 0.785 (max = 0.785)
If satisfied with current slider location, press Enter to lock it down.
int of -986 chosen representing time delay of -15.421 ns
Corresponding correlation value of 0.739 (max = 0.739)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3079 chosen representing time delay of 48.156 ns
Corresponding correlation value of 0.767 (max = 0.767)

If satisfied with current slider location, press Enter to lock it down.
int of 66223 chosen representing time delay of 1035.738 ns
Corresponding correlation value of 0.095 (max = 0.095)

If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 61733 chosen representing time delay of 965.514 ns
Corresponding correlation value of 0.107 (max = 0.107)
If satisfied with current slider location, press Enter to lock it down.

int of 66314 chosen representing time delay of 1037.161 ns
Corresponding correlation value of 0.124 (max = 0.124)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -5362 chosen representing time delay of -83.863 ns
Corresponding correlation value of 0.908 (max = 0.908)

If satisfied with current slider location, press Enter to lock it down.
int of -813 chosen representing time delay of -12.715 ns
Corresponding correlation value of 0.894 (max = 0.894)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4551 chosen representing time delay of 71.178 ns
Corresponding correlation value of 0.890 (max = 0.890)
(26/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 5740 chosen representing time delay of 89.774 ns
Corresponding correlation value of 0.782 (max = 0.782)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 763 chosen representing time delay of 11.933 ns
Corresponding correlation value of 0.884 (max = 0.884)
If satisfied with current slider location, press Enter to lock it down.
int of 6474 chosen representing time delay of 101.254 ns
Corresponding correlation value of 0.765 (max = 0.765)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -4980 chosen representing time delay of -77.888 ns
Corresponding correlation value of 0.830 (max = 0.830)
If satisfied with current slider location, press Enter to lock it down.
int of -1002 chosen representing time delay of -15.671 ns
Corresponding correlation value of 0.834 (max = 0.834)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 3112 chosen representing time delay of 48.672 ns
Corresponding correlation value of 0.770 (max = 0.770)
If satisfied with current slider location, press Enter to lock it down.
int of 39310 chosen representing time delay of 614.814 ns
Corresponding correlation value of 0.077 (max = 0.077)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 34801 chosen representing time delay of 544.293 ns
Corresponding correlation value of 0.066 (max = 0.066)

If satisfied with current slider location, press Enter to lock it down.
int of 34840 chosen representing time delay of 544.903 ns
Corresponding correlation value of 0.067 (max = 0.067)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -5430 chosen representing time delay of -84.926 ns
Corresponding correlation value of 0.943 (max = 0.943)
If satisfied with current slider location, press Enter to lock it down.
int of -834 chosen representing time delay of -13.044 ns
Corresponding correlation value of 0.939 (max = 0.939)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4593 chosen representing time delay of 71.835 ns
Corresponding correlation value of 0.929 (max = 0.929)




ab5f43
[[-2.88873651e+01  1.70634084e+01 -1.17770362e+01  4.55441296e+01
   1.65942038e+01 -2.89342856e+01]
 [-2.56185729e+01  1.57809158e+01  2.04886022e+00  2.84807211e+01
   1.45297034e+01 -1.23244417e+01]
 [-2.52432092e+01  1.56401544e+01 -1.00253390e+01  4.02421172e+01
   2.79333157e+01 -2.55560123e+01]
 [-6.56886484e-01  1.36538548e+01  6.64706561e+00  1.42169003e+01
   7.67931580e+00 -6.95986870e+00]
 [ 3.06547026e+00  1.32002903e+01  9.49357371e+00  9.96277834e+00
   6.47502391e+00 -3.73799690e+00]
 [ 5.78685712e+00  1.33723320e+01  1.13547521e+01  7.05370963e+00
   5.78685712e+00 -1.73605714e+00]
 [ 5.58353511e+00  1.32002903e+01  1.12765513e+01 -6.89730808e+00
   5.61481542e+00 -1.70477683e+00]
 [ 6.94422854e+00  1.33879721e+01  1.31377297e+01  5.31765249e+00
   4.80152739e+00 -7.82007719e-02]
 [ 1.51083891e+01  1.26841652e+01  1.89871474e+01 -3.34699304e+00
   2.89342856e+00  6.92858839e+00]
 [ 2.16459737e+01  1.18865173e+01  2.29441065e+01 -9.90021772e+00
   1.21993204e+00  1.10575891e+01]
 [ 2.21933791e+01  1.23088015e+01  2.32256293e+01 -9.99405865e+00
   1.11045096e+00  1.09793884e+01]
 [ 3.60818362e+01  1.12139907e+01  4.63730577e+01 -2.50711675e+01
  -2.43986408e+00  3.57221126e+01]
 [ 3.71610068e+01  1.08229868e+01  4.71707056e+01 -2.62128987e+01
  -2.84650810e+00  2.35227922e+01]
 [ 5.38959720e+01  1.10575891e+01  4.64043380e+01 -4.28383828e+01
  -7.44471348e+00  3.53467489e+01]
 [ 5.31608847e+01  1.09637482e+01  4.60758948e+01 -4.26819813e+01
  -7.35087256e+00  3.51747072e+01]
 [ 5.50689836e+01  1.08699073e+01  4.70768647e+01 -4.41677960e+01
  -7.72623626e+00  3.62851582e+01]
 [ 5.53348662e+01  1.08229868e+01  4.72958268e+01 -4.46682809e+01
  -7.96083858e+00  3.65979612e+01]
 [ 7.02242932e+01  1.08855474e+01  4.83906377e+01 -5.92292646e+01
  -8.14852043e+00  3.76458516e+01]
 [ 5.79624121e+01  1.09950285e+01  4.93916075e+01 -4.71081450e+01
  -8.50824398e+00  3.85060601e+01]
 [ 7.29300399e+01  1.08542671e+01  6.41402731e+01 -4.86095998e+01
  -8.78976676e+00  3.96790717e+01]
 [ 9.91898591e+01  1.05101837e+01  7.08811796e+01 -7.50101804e+01
  -1.50771088e+01  5.99956322e+01]
 [ 9.99093062e+01  1.06353050e+01  7.13973047e+01 -7.59329495e+01
  -1.53429914e+01  6.06212384e+01]
 [ 1.13782123e+02  1.07447861e+01  9.87050143e+01 -7.60424306e+01
  -1.52804308e+01  6.08402005e+01]
 [ 1.02489932e+02  1.15737142e+01  8.67402962e+01 -7.74187642e+01
  -1.56245142e+01  6.16847689e+01]
 [ 1.15768423e+02  1.16519150e+01  1.00315950e+02 -7.72780028e+01
  -1.54211922e+01  4.81560353e+01]
 [ 8.97744861e+01  1.19334378e+01  1.01254359e+02 -7.78879688e+01
  -1.56714347e+01  4.86721604e+01]]
[[-3.70327575e+02  6.20867208e+02  2.92345766e+02  3.88345033e+01
   1.99568370e+01 -1.86274239e+01]
 [ 0.00000000e+00  3.58785141e+01  6.01833141e+01  3.63320786e+01
   1.79861775e+01 -1.82520602e+01]
 [ 1.21946284e+03 -1.13766483e+02 -7.21167518e+01  3.42050176e+01
   1.84710223e+01 -1.58903968e+01]
 [-4.64981790e+01 -3.83652987e+01 -2.80506169e+02 -3.14679906e+01
   1.01973807e+01  2.23654208e+00]
 [-1.79392571e+01 -3.18433543e+01 -5.20973542e+01  1.21633481e+02
   3.28443242e+01  7.97178669e+01]
 [ 4.67640616e+00  3.27817636e+02  4.32669231e+02 -1.59811097e+02
  -3.43144987e+01  1.95470649e+02]
 [ 4.60289743e+01 -1.32206225e+02 -3.92630436e+02 -1.48737868e+01
  -4.68266222e+01 -1.76420941e+01]
 [-3.66605219e+01  4.20720153e+00  6.04804770e+02 -1.41386996e+01
   8.52388414e+00  2.28502655e+01]
 [ 1.40761389e-01  3.56579880e+02 -2.08154815e+02 -1.27936463e+01
  -8.27364167e+00  1.90340679e+01]
 [ 5.00015736e+01  4.65138191e+01  1.44202223e+02 -1.82989806e+00
  -5.80249727e+01 -1.28249266e+00]
 [-2.66586431e+02 -2.85511018e+02 -2.76517929e+02 -7.38215287e+00
   2.81522779e+00  1.08855474e+01]
 [ 1.42951011e+01  3.94773137e+02  3.28474522e+02 -3.17807937e+01
   1.35381176e+02 -4.33388678e+01]
 [-6.06837990e+00 -2.19243684e+02  2.56373411e+02 -4.71550655e+01
  -4.37924323e-01 -4.33075875e+01]
 [ 8.22202916e+01 -2.19306245e+02  6.00566288e+02 -4.89849635e+01
   1.00253390e+01  2.23341405e+01]
 [ 1.04191580e+03  8.04998746e+02  6.44702804e+02 -4.89536832e+01
  -4.80152739e+00  4.38237126e+01]
 [ 2.88467007e+02  8.27676970e+01 -1.76311460e+02 -5.00015736e+01
  -4.87972817e+00  4.50436446e+01]
 [ 4.84375581e+02  6.42810345e+02  2.04291697e+02 -5.06897403e+01
   9.80637680e+00 -1.56401544e-02]
 [ 3.52138076e+02  1.05109658e+03 -3.44364919e+02 -6.62204136e+01
  -5.53661465e+00  4.64356184e+01]
 [ 3.82339214e+02  3.44349279e+02  4.20454270e+02 -5.34893280e+01
  -6.05273975e+00  4.72489064e+01]
 [ 2.06450038e+00  6.40558163e+02  6.88776759e+02 -5.45684986e+01
   8.14852043e+00  4.82498763e+01]
 [ 1.87681853e+00 -8.06562761e+01  3.96524834e+02 -8.03903935e+01
  -1.22775212e+01  6.82692739e+01]
 [-7.48537789e+01 -1.82395480e+02  8.77178058e+02 -8.16416059e+01
  -1.25434038e+01  6.90982021e+01]
 [ 9.66561541e+01 -4.04204150e+02  4.12618553e+02 -8.17823673e+01
  -1.22462409e+01  6.94422854e+01]
 [ 6.54524821e+02  5.70412070e+02  8.32932062e+02 -8.39094282e+01
  -1.28718471e+01  7.10063009e+01]
 [ 1.03573794e+03  9.65513650e+02  1.03716120e+03 -8.38625078e+01
  -1.27154455e+01  7.11783426e+01]
 [ 6.14814469e+02  5.44293013e+02  5.44902979e+02 -8.49260383e+01
  -1.30438888e+01  7.18352291e+01]]
[[0.62982833 0.77677675 0.75236731 0.76749577 0.59979072 0.64790917]
 [0.5773365  0.70644162 0.57608923 0.80348651 0.6945646  0.53954768]
 [0.6726403  0.77406578 0.79424238 0.85345803 0.80687257 0.82948902]
 [0.78873432 0.8493454  0.8285548  0.83122342 0.84323089 0.8600067 ]
 [0.84153738 0.84077823 0.88749982 0.84966581 0.84858517 0.83303438]
 [0.77161979 0.77809199 0.86298065 0.80597524 0.83063258 0.77320236]
 [0.75806037 0.75671567 0.74040503 0.70555776 0.67308454 0.66643007]
 [0.63756443 0.84007398 0.86511378 0.81531957 0.66080911 0.7957828 ]
 [0.74811472 0.76083647 0.7468288  0.81857184 0.83935535 0.65600105]
 [0.85156469 0.81823081 0.86343807 0.86258245 0.84924771 0.84471723]
 [0.81706908 0.75767876 0.86953568 0.84000709 0.87767818 0.82014223]
 [0.73175229 0.77029831 0.85653911 0.85610337 0.68266051 0.74132912]
 [0.76736686 0.76570254 0.86771827 0.85000848 0.68514173 0.67414185]
 [0.78847956 0.8257755  0.8746801  0.87503349 0.84387349 0.85493423]
 [0.84624229 0.84724204 0.93289006 0.93969099 0.92829951 0.89108326]
 [0.86290249 0.8511148  0.93432178 0.93254486 0.93313052 0.89721675]
 [0.84340874 0.84170433 0.92365669 0.91877931 0.91033016 0.88505468]
 [0.89322809 0.86249058 0.95011396 0.94158814 0.93632529 0.8856151 ]
 [0.9038838  0.88233514 0.93607073 0.95002208 0.93720703 0.90594311]
 [0.87778843 0.87739643 0.90883462 0.93087253 0.92047593 0.89961645]
 [0.60800087 0.75981388 0.6329627  0.66585003 0.58271038 0.65777595]
 [0.71511362 0.89178027 0.72180865 0.82017094 0.80124434 0.78161604]
 [0.71745401 0.83198925 0.73537114 0.78335336 0.72950841 0.78732965]
 [0.78751439 0.84312842 0.78616197 0.80067319 0.75547878 0.80829344]
 [0.72691166 0.84812847 0.77737287 0.78549372 0.7385571  0.76687713]
 [0.78201732 0.88369476 0.76489104 0.82951779 0.83410446 0.76985447]]
[[0.09195445 0.07515564 0.07639771 0.62868637 0.68296663 0.68418555]
 [0.07507139 0.06168901 0.09490234 0.79499818 0.60285842 0.57695717]
 [0.07659796 0.05946387 0.08140472 0.74682829 0.63174937 0.59011181]
 [0.06232146 0.07081852 0.08302202 0.09664629 0.37989728 0.23783292]
 [0.102013   0.07851408 0.09931923 0.21182305 0.30235082 0.24091466]
 [0.08199799 0.05676422 0.08551949 0.20555654 0.40518009 0.1997543 ]
 [0.07533579 0.08488013 0.0845811  0.35827235 0.4158479  0.22040305]
 [0.05515905 0.07398308 0.08675301 0.59165878 0.62400296 0.55859218]
 [0.07818258 0.05442293 0.08631925 0.05674032 0.04485081 0.40349079]
 [0.08497717 0.06824892 0.10771709 0.19017998 0.24135385 0.22502987]
 [0.1073537  0.06191909 0.11381537 0.22800771 0.13930562 0.47989572]
 [0.07641062 0.07198609 0.11753064 0.23604694 0.25403444 0.19533438]
 [0.07586185 0.07300502 0.12101883 0.21995608 0.27776301 0.2785478 ]
 [0.09672465 0.08218999 0.07177631 0.34128779 0.31072241 0.27393056]
 [0.04661366 0.05603728 0.07052959 0.73419146 0.68739327 0.71882318]
 [0.06374236 0.06665853 0.09883411 0.71003983 0.65601348 0.73509211]
 [0.05717604 0.06735745 0.08034756 0.42368594 0.40780072 0.35027116]
 [0.06886223 0.07458845 0.08641286 0.7873007  0.78604399 0.78936861]
 [0.06727275 0.0591851  0.07705009 0.73627352 0.71987617 0.76038748]
 [0.06342815 0.06639673 0.06819119 0.76919342 0.75007895 0.78952973]
 [0.09327135 0.06787646 0.08285837 0.32722777 0.41452416 0.35887881]
 [0.06860293 0.08340955 0.08617786 0.76052345 0.82473454 0.73281971]
 [0.07858032 0.06649049 0.08155051 0.75495761 0.75391884 0.72652801]
 [0.06814221 0.05833277 0.07026216 0.92986665 0.91561262 0.90739401]
 [0.09521795 0.10702754 0.12377197 0.90834791 0.89390763 0.89032708]
 [0.07720588 0.06618241 0.06720744 0.94268425 0.93906486 0.92917063]]
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1783_analysis_data.h5 already exists, checking if setup is up to date.

Processing Run 1783
None given as eventid, setting event 0 as template.
Event 28861 set as template
Calculating time delays:
(3/3)           



a52e4f
[[  65.37584531  -44.19907628   17.53261306 -109.54364128  -47.90579287
    61.62220826]
 [  66.48629627  -42.96350408   19.61275359 -109.4654405   -47.38966777
    62.07577273]
 [  87.78818653  -11.38603239   57.28988549  -98.59553321  -30.87366475
    67.87827001]]
[[ 951.54699247 -386.21797226 -224.26417365 -115.95610457  -46.38869789
    69.75508853]
 [ 675.0759835   562.57635304  632.5973242  -115.22101732  -45.16876585
    70.13045224]
 [1327.34862191 -206.32491658 1297.6636089  -104.44495095  -28.7622439
    75.76090782]]
[[0.77510182 0.76120413 0.69953395 0.85153975 0.85410922 0.8273125 ]
 [0.81609656 0.83873214 0.82409572 0.85631712 0.85515108 0.86051845]
 [0.61147331 0.518565   0.68108793 0.70349108 0.80336612 0.74497518]]
[[0.083143   0.06820834 0.08130717 0.66499273 0.77803667 0.67031213]
 [0.08099163 0.07719255 0.08242499 0.7171041  0.84142522 0.74202524]
 [0.06336892 0.06530053 0.07794064 0.82957209 0.88256237 0.83269091]]
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1784_analysis_data.h5 already exists, checking if setup is up to date.

Processing Run 1784
None given as eventid, setting event 0 as template.
Event 7255 set as template
Calculating time delays:
If satisfied with current slider location, press Enter to lock it down.
int of 4482 chosen representing time delay of 70.099 ns
Corresponding correlation value of 0.825 (max = 0.825)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)
QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)
QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)
QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)
QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2537 chosen representing time delay of -39.679 ns
Corresponding correlation value of 0.842 (max = 0.842)
If satisfied with current slider location, press Enter to lock it down.
int of 1565 chosen representing time delay of 24.477 ns
Corresponding correlation value of 0.861 (max = 0.861)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -7016 chosen representing time delay of -109.731 ns
Corresponding correlation value of 0.888 (max = 0.888)
If satisfied with current slider location, press Enter to lock it down.
int of -2923 chosen representing time delay of -45.716 ns
Corresponding correlation value of 0.895 (max = 0.895)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4091 chosen representing time delay of 63.984 ns
Corresponding correlation value of 0.844 (max = 0.844)
If satisfied with current slider location, press Enter to lock it down.
int of -16012 chosen representing time delay of -250.430 ns
Corresponding correlation value of 0.076 (max = 0.076)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -22624 chosen representing time delay of -353.843 ns
Corresponding correlation value of 0.085 (max = 0.085)
If satisfied with current slider location, press Enter to lock it down.
int of 14885 chosen representing time delay of 232.804 ns
Corresponding correlation value of 0.092 (max = 0.092)
If satisfied with current slider location, press Enter to lock it down.
int of -7402 chosen representing time delay of -115.768 ns
Corresponding correlation value of 0.746 (max = 0.746)
If satisfied with current slider location, press Enter to lock it down.
int of -2848 chosen representing time delay of -44.543 ns
Corresponding correlation value of 0.815 (max = 0.815)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4568 chosen representing time delay of 71.444 ns
Corresponding correlation value of 0.725 (max = 0.725)
If satisfied with current slider location, press Enter to lock it down.
int of 4978 chosen representing time delay of 77.857 ns
Corresponding correlation value of 0.695 (max = 0.695)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1824 chosen representing time delay of -28.528 ns
Corresponding correlation value of 0.643 (max = 0.643)
If satisfied with current slider location, press Enter to lock it down.
int of 2433 chosen representing time delay of 38.052 ns
Corresponding correlation value of 0.700 (max = 0.700)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -6774 chosen representing time delay of -105.946 ns
Corresponding correlation value of 0.747 (max = 0.747)
If satisfied with current slider location, press Enter to lock it down.
int of -2570 chosen representing time delay of -40.195 ns
Corresponding correlation value of 0.898 (max = 0.898)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4234 chosen representing time delay of 66.220 ns
Corresponding correlation value of 0.730 (max = 0.730)
If satisfied with current slider location, press Enter to lock it down.
int of 512 chosen representing time delay of 8.008 ns
Corresponding correlation value of 0.082 (max = 0.082)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -6654 chosen representing time delay of -104.070 ns
Corresponding correlation value of 0.096 (max = 0.096)
If satisfied with current slider location, press Enter to lock it down.
int of 78611 chosen representing time delay of 1229.488 ns
Corresponding correlation value of 0.083 (max = 0.083)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -7167 chosen representing time delay of -112.093 ns
Corresponding correlation value of 0.890 (max = 0.890)
If satisfied with current slider location, press Enter to lock it down.
int of -2423 chosen representing time delay of -37.896 ns
Corresponding correlation value of 0.937 (max = 0.937)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4739 chosen representing time delay of 74.119 ns
Corresponding correlation value of 0.867 (max = 0.867)
If satisfied with current slider location, press Enter to lock it down.
int of 5042 chosen representing time delay of 78.858 ns
Corresponding correlation value of 0.671 (max = 0.671)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -1695 chosen representing time delay of -26.510 ns
Corresponding correlation value of 0.545 (max = 0.545)
If satisfied with current slider location, press Enter to lock it down.
int of 2577 chosen representing time delay of 40.305 ns
Corresponding correlation value of 0.700 (max = 0.700)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -6710 chosen representing time delay of -104.945 ns
Corresponding correlation value of 0.745 (max = 0.745)
If satisfied with current slider location, press Enter to lock it down.
int of -2484 chosen representing time delay of -38.850 ns
Corresponding correlation value of 0.828 (max = 0.828)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4244 chosen representing time delay of 66.377 ns
Corresponding correlation value of 0.735 (max = 0.735)
If satisfied with current slider location, press Enter to lock it down.
int of 26994 chosen representing time delay of 422.190 ns
Corresponding correlation value of 0.070 (max = 0.070)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 19869 chosen representing time delay of 310.754 ns
Corresponding correlation value of 0.084 (max = 0.084)
If satisfied with current slider location, press Enter to lock it down.
int of 25533 chosen representing time delay of 399.340 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -7103 chosen representing time delay of -111.092 ns
Corresponding correlation value of 0.858 (max = 0.858)
If satisfied with current slider location, press Enter to lock it down.
int of -2349 chosen representing time delay of -36.739 ns
Corresponding correlation value of 0.916 (max = 0.916)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4748 chosen representing time delay of 74.259 ns
Corresponding correlation value of 0.849 (max = 0.849)
If satisfied with current slider location, press Enter to lock it down.
int of 5661 chosen representing time delay of 88.539 ns
Corresponding correlation value of 0.710 (max = 0.710)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -594 chosen representing time delay of -9.290 ns
Corresponding correlation value of 0.621 (max = 0.621)
If satisfied with current slider location, press Enter to lock it down.
int of 3753 chosen representing time delay of 58.697 ns
Corresponding correlation value of 0.686 (max = 0.686)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -6261 chosen representing time delay of -97.923 ns
Corresponding correlation value of 0.869 (max = 0.869)
If satisfied with current slider location, press Enter to lock it down.
int of -1915 chosen representing time delay of -29.951 ns
Corresponding correlation value of 0.880 (max = 0.880)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4350 chosen representing time delay of 68.035 ns
Corresponding correlation value of 0.874 (max = 0.874)
If satisfied with current slider location, press Enter to lock it down.
int of 56769 chosen representing time delay of 887.876 ns
Corresponding correlation value of 0.094 (max = 0.094)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 50110 chosen representing time delay of 783.728 ns
Corresponding correlation value of 0.098 (max = 0.098)
If satisfied with current slider location, press Enter to lock it down.
int of 60428 chosen representing time delay of 945.103 ns
Corresponding correlation value of 0.081 (max = 0.081)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -6649 chosen representing time delay of -103.991 ns
Corresponding correlation value of 0.784 (max = 0.784)
If satisfied with current slider location, press Enter to lock it down.
int of -1780 chosen representing time delay of -27.839 ns
Corresponding correlation value of 0.842 (max = 0.842)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4874 chosen representing time delay of 76.230 ns
Corresponding correlation value of 0.786 (max = 0.786)
If satisfied with current slider location, press Enter to lock it down.
int of 7043 chosen representing time delay of 110.154 ns
Corresponding correlation value of 0.651 (max = 0.651)
If satisfied with current slider location, press Enter to lock it down.
int of 2506 chosen representing time delay of 39.194 ns
Corresponding correlation value of 0.740 (max = 0.740)
If satisfied with current slider location, press Enter to lock it down.
int of 6954 chosen representing time delay of 108.762 ns
Corresponding correlation value of 0.664 (max = 0.664)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -4513 chosen representing time delay of -70.584 ns
Corresponding correlation value of 0.805 (max = 0.805)
If satisfied with current slider location, press Enter to lock it down.
int of -77 chosen representing time delay of -1.204 ns
Corresponding correlation value of 0.879 (max = 0.879)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4430 chosen representing time delay of 69.286 ns
Corresponding correlation value of 0.815 (max = 0.815)
If satisfied with current slider location, press Enter to lock it down.
int of 512 chosen representing time delay of 8.008 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -28294 chosen representing time delay of -442.523 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.
int of 66353 chosen representing time delay of 1037.771 ns
Corresponding correlation value of 0.064 (max = 0.064)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -4875 chosen representing time delay of -76.246 ns
Corresponding correlation value of 0.751 (max = 0.751)
If satisfied with current slider location, press Enter to lock it down.
int of 75 chosen representing time delay of 1.173 ns
Corresponding correlation value of 0.877 (max = 0.877)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4958 chosen representing time delay of 77.544 ns
Corresponding correlation value of 0.745 (max = 0.745)
If satisfied with current slider location, press Enter to lock it down.
int of 7245 chosen representing time delay of 113.313 ns
Corresponding correlation value of 0.392 (max = 0.392)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 5637 chosen representing time delay of 88.164 ns
Corresponding correlation value of 0.547 (max = 0.547)
If satisfied with current slider location, press Enter to lock it down.
int of 7471 chosen representing time delay of 116.848 ns
Corresponding correlation value of 0.397 (max = 0.397)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -4938 chosen representing time delay of -77.231 ns
Corresponding correlation value of 0.618 (max = 0.618)
If satisfied with current slider location, press Enter to lock it down.
int of -578 chosen representing time delay of -9.040 ns
Corresponding correlation value of 0.753 (max = 0.753)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 4382 chosen representing time delay of 68.535 ns
Corresponding correlation value of 0.589 (max = 0.589)
If satisfied with current slider location, press Enter to lock it down.
int of -1261 chosen representing time delay of -19.722 ns
Corresponding correlation value of 0.077 (max = 0.077)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -2433 chosen representing time delay of -38.052 ns
Corresponding correlation value of 0.088 (max = 0.088)
If satisfied with current slider location, press Enter to lock it down.
int of 27908 chosen representing time delay of 436.485 ns
Corresponding correlation value of 0.060 (max = 0.060)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of -3674 chosen representing time delay of -57.462 ns
Corresponding correlation value of 0.538 (max = 0.538)
If satisfied with current slider location, press Enter to lock it down.
int of 483 chosen representing time delay of 7.554 ns
Corresponding correlation value of 0.812 (max = 0.812)
If satisfied with current slider location, press Enter to lock it down.QXcbShmImage: shmget() failed (28: No space left on device) for size 3924000 (1000x981)

int of 5010 chosen representing time delay of 78.357 ns
Corresponding correlation value of 0.556 (max = 0.556)




acf975
[[  70.09917193  -39.67907166   24.4768416  -109.73132313  -45.71617125
    63.98387157]
 [  77.8566885   -28.52764159   38.05249561 -105.94640577  -40.19519676
    66.22041364]
 [  78.85765838  -26.51006167   40.30467784 -104.94543589  -38.85014348
    66.37681519]
 [  88.53891394   -9.2902517    58.69749939  -97.92300657  -29.95089564
    68.03467155]
 [ 110.1536073    39.19422688  108.76163356  -70.58401672   -1.20429189
    69.2858839 ]
 [ 113.31291848   88.16355024  116.84759337  -77.23108233   -9.04000923
    68.53515649]]
[[-2.50430152e+02 -3.53842853e+02  2.32803698e+02 -1.15768423e+02
  -4.45431597e+01  7.14442252e+01]
 [ 8.00775904e+00 -1.04069587e+02  1.22948818e+03 -1.12092986e+02
  -3.78960941e+01  7.41186916e+01]
 [ 4.22190327e+02  3.10754227e+02  3.99340062e+02 -1.11092017e+02
  -3.67387226e+01  7.42594530e+01]
 [ 8.87875924e+02  7.83728136e+02  9.45103249e+02 -1.03991386e+02
  -2.78394748e+01  7.62301124e+01]
 [ 8.00775904e+00 -4.42522528e+02  1.03777116e+03 -7.62457526e+01
   1.17301158e+00  7.75438854e+01]
 [-1.97222347e+01 -3.80524956e+01  4.36485428e+02 -5.74619272e+01
   7.55419457e+00  7.83571734e+01]]
[[0.82452304 0.84237292 0.86070192 0.88783822 0.89508353 0.84408384]
 [0.69477446 0.64290981 0.70018481 0.74739501 0.89817133 0.73033073]
 [0.67095742 0.54535095 0.70012358 0.74522961 0.82753476 0.73537265]
 [0.71017735 0.62077517 0.68634733 0.86860821 0.88005062 0.87374545]
 [0.65076326 0.74020585 0.66372195 0.80522205 0.87859943 0.81493234]
 [0.39197714 0.54749722 0.39682659 0.61800494 0.75251    0.58922157]]
[[0.07563383 0.08511365 0.09201061 0.74565473 0.81545159 0.72463237]
 [0.08151722 0.0958623  0.08309501 0.89035292 0.93663151 0.8665272 ]
 [0.06950143 0.08449305 0.06821813 0.85802633 0.91646282 0.84926463]
 [0.09422863 0.09829601 0.08132951 0.78374033 0.84211843 0.78580355]
 [0.06832285 0.06841074 0.06360799 0.75114916 0.87657423 0.74458405]
 [0.07721145 0.0884296  0.05952351 0.53792951 0.81185286 0.55579139]]






#### BELOW HERE IS THE SECOND ATTEMPT.  CAN TRUST 16/27 ONWARD

In [5]: %run find_phase_centers_from_planes.py                                  
No mode given.  Defaulting to hpol
Loading known plane locations.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1728_analysis_data.h5 already exists, checking if setup is up to date.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1773_analysis_data.h5 already exists, checking if setup is up to date.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1773_analysis_data.h5 already exists, checking if setup is up to date.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1774_analysis_data.h5 already exists, checking if setup is up to date.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1783_analysis_data.h5 already exists, checking if setup is up to date.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1784_analysis_data.h5 already exists, checking if setup is up to date.
Loading in cable delays.
Calculating time delays:

(1/6)           If satisfied with current slider location, press Enter to lock it down.
---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
~/Projects/Beacon/beacon/analysis/find_phase_centers_from_planes.py in <module>
    166             tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
    167             eventids = known_planes[key]['eventids'][:,1]
--> 168             time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids,align_method=9,hilbert=hilbert)
    169 
    170 

~/Projects/Beacon/beacon/objects/fftmath.py in calculateMultipleTimeDelays(self, eventids, align_method, hilbert, plot, hpol_cut, vpol_cut, colors)
    887                         indices, time_shift, corr_value, pairs, corrs = self.calculateTimeDelaysFromEvent(eventid,align_method=align_method,hilbert=hilbert,return_full_corrs=True)
    888                     else:
--> 889                         indices, time_shift, corr_value, pairs = self.calculateTimeDelaysFromEvent(eventid,align_method=align_method,hilbert=hilbert,return_full_corrs=False)
    890                 timeshifts.append(time_shift)
    891                 max_corrs.append(corr_value)

~/Projects/Beacon/beacon/objects/fftmath.py in calculateTimeDelaysFromEvent(self, eventid, return_full_corrs, align_method, hilbert)
    829         try:
    830             ffts, upsampled_waveforms = self.loadFilteredFFTs(eventid,load_upsampled_waveforms=True,hilbert=hilbert)
--> 831             return self.calculateTimeDelays(ffts, upsampled_waveforms, return_full_corrs=return_full_corrs, align_method=align_method, print_warning=False)
    832         except Exception as e:
    833             print('\nError in %s'%inspect.stack()[0][3])

~/Projects/Beacon/beacon/objects/fftmath.py in calculateTimeDelays(self, ffts, upsampled_waveforms, return_full_corrs, align_method, print_warning)
    771                     slider_roll.on_changed(update)
    772 
--> 773                     input('If satisfied with current slider location, press Enter to lock it down.')
    774                     plt.close(self.persistent_object[fig_index])
    775 

KeyboardInterrupt: 

In [6]:                                                                         

In [6]:                                                                         

In [6]:                                                                         

In [6]: plt.close('all')                                                        

In [7]: %run find_phase_centers_from_planes.py                                  
No mode given.  Defaulting to hpol
Loading known plane locations.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1728_analysis_data.h5 already exists, checking if setup is up to date.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1773_analysis_data.h5 already exists, checking if setup is up to date.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1773_analysis_data.h5 already exists, checking if setup is up to date.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1774_analysis_data.h5 already exists, checking if setup is up to date.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1783_analysis_data.h5 already exists, checking if setup is up to date.
/home/dsouthall/scratch-midway2/beacon_jan17_2020/run1784_analysis_data.h5 already exists, checking if setup is up to date.
Loading in cable delays.
Calculating time delays:
(1/6)           
If satisfied with current slider location, press Enter to lock it down.
int of 6353 chosen representing time delay of 99.362 ns
Corresponding correlation value of 0.543 (max = 0.543)
If satisfied with current slider location, press Enter to lock it down.
int of 1627 chosen representing time delay of 25.447 ns
Corresponding correlation value of 0.664 (max = 0.664)
If satisfied with current slider location, press Enter to lock it down.
int of 6143 chosen representing time delay of 96.077 ns
Corresponding correlation value of 0.529 (max = 0.529)
If satisfied with current slider location, press Enter to lock it down.
int of -4695 chosen representing time delay of -73.431 ns
Corresponding correlation value of 0.463 (max = 0.463)
If satisfied with current slider location, press Enter to lock it down.
int of -181 chosen representing time delay of -2.831 ns
Corresponding correlation value of 0.574 (max = 0.574)
If satisfied with current slider location, press Enter to lock it down.
int of 4515 chosen representing time delay of 70.615 ns
Corresponding correlation value of 0.688 (max = 0.688)

If satisfied with current slider location, press Enter to lock it down.

int of 6555 chosen representing time delay of 102.521 ns
Corresponding correlation value of 0.749 (max = 0.749)
If satisfied with current slider location, press Enter to lock it down.
int of 1494 chosen representing time delay of 23.366 ns
Corresponding correlation value of 0.721 (max = 0.721)

If satisfied with current slider location, press Enter to lock it down.
int of 6562 chosen representing time delay of 102.631 ns
Corresponding correlation value of 0.685 (max = 0.685)
If satisfied with current slider location, press Enter to lock it down.
int of -5056 chosen representing time delay of -79.077 ns
Corresponding correlation value of 0.780 (max = 0.780)

If satisfied with current slider location, press Enter to lock it down.
int of 8 chosen representing time delay of 0.125 ns
Corresponding correlation value of 0.850 (max = 0.850)
If satisfied with current slider location, press Enter to lock it down.
int of 5072 chosen representing time delay of 79.327 ns
Corresponding correlation value of 0.778 (max = 0.778)
(2/6)           

If satisfied with current slider location, press Enter to lock it down.
int of 5837 chosen representing time delay of 91.292 ns
Corresponding correlation value of 0.728 (max = 0.728)

If satisfied with current slider location, press Enter to lock it down.

int of 528 chosen representing time delay of 8.258 ns
Corresponding correlation value of 0.733 (max = 0.733)
If satisfied with current slider location, press Enter to lock it down.
int of 4994 chosen representing time delay of 78.107 ns
Corresponding correlation value of 0.678 (max = 0.678)
If satisfied with current slider location, press Enter to lock it down.
int of -5297 chosen representing time delay of -82.846 ns
Corresponding correlation value of 0.711 (max = 0.711)

If satisfied with current slider location, press Enter to lock it down.
int of -841 chosen representing time delay of -13.153 ns
Corresponding correlation value of 0.740 (max = 0.740)

If satisfied with current slider location, press Enter to lock it down.
int of 4482 chosen representing time delay of 70.099 ns
Corresponding correlation value of 0.726 (max = 0.726)

If satisfied with current slider location, press Enter to lock it down.
int of 6053 chosen representing time delay of 94.670 ns
Corresponding correlation value of 0.824 (max = 0.824)

If satisfied with current slider location, press Enter to lock it down.

int of 369 chosen representing time delay of 5.771 ns
Corresponding correlation value of 0.839 (max = 0.839)
If satisfied with current slider location, press Enter to lock it down.
int of 5376 chosen representing time delay of 84.081 ns
Corresponding correlation value of 0.818 (max = 0.818)
If satisfied with current slider location, press Enter to lock it down.

int of -5683 chosen representing time delay of -88.883 ns
Corresponding correlation value of 0.842 (max = 0.842)
If satisfied with current slider location, press Enter to lock it down.
int of -679 chosen representing time delay of -10.620 ns
Corresponding correlation value of 0.894 (max = 0.894)
If satisfied with current slider location, press Enter to lock it down.
int of 5007 chosen representing time delay of 78.310 ns
Corresponding correlation value of 0.839 (max = 0.839)
(3/6)           


If satisfied with current slider location, press Enter to lock it down.
int of 4472 chosen representing time delay of 69.943 ns
Corresponding correlation value of 0.790 (max = 0.790)

If satisfied with current slider location, press Enter to lock it down.
int of -1710 chosen representing time delay of -26.745 ns
Corresponding correlation value of 0.876 (max = 0.876)

If satisfied with current slider location, press Enter to lock it down.

int of 2392 chosen representing time delay of 37.411 ns
Corresponding correlation value of 0.916 (max = 0.916)
If satisfied with current slider location, press Enter to lock it down.
int of -6187 chosen representing time delay of -96.766 ns
Corresponding correlation value of 0.791 (max = 0.791)

If satisfied with current slider location, press Enter to lock it down.
int of -2075 chosen representing time delay of -32.453 ns
Corresponding correlation value of 0.804 (max = 0.804)

If satisfied with current slider location, press Enter to lock it down.
int of 4108 chosen representing time delay of 64.250 ns
Corresponding correlation value of 0.807 (max = 0.807)

If satisfied with current slider location, press Enter to lock it down.
int of 4693 chosen representing time delay of 73.399 ns
Corresponding correlation value of 0.889 (max = 0.889)

If satisfied with current slider location, press Enter to lock it down.
int of -1916 chosen representing time delay of -29.967 ns
Corresponding correlation value of 0.909 (max = 0.909)

If satisfied with current slider location, press Enter to lock it down.
int of 2757 chosen representing time delay of 43.120 ns
Corresponding correlation value of 0.892 (max = 0.892)
If satisfied with current slider location, press Enter to lock it down.
int of -6604 chosen representing time delay of -103.288 ns
Corresponding correlation value of 0.884 (max = 0.884)

If satisfied with current slider location, press Enter to lock it down.
int of -1936 chosen representing time delay of -30.279 ns
Corresponding correlation value of 0.871 (max = 0.871)



If satisfied with current slider location, press Enter to lock it down.
int of 4670 chosen representing time delay of 73.040 ns
Corresponding correlation value of 0.893 (max = 0.893)
(4/6)           
If satisfied with current slider location, press Enter to lock it down.
int of 4354 chosen representing time delay of 68.097 ns
Corresponding correlation value of 0.734 (max = 0.734)
If satisfied with current slider location, press Enter to lock it down.
int of -1859 chosen representing time delay of -29.075 ns
Corresponding correlation value of 0.866 (max = 0.866)

If satisfied with current slider location, press Enter to lock it down.
int of 2218 chosen representing time delay of 34.690 ns
Corresponding correlation value of 0.793 (max = 0.793)
If satisfied with current slider location, press Enter to lock it down.
int of -6216 chosen representing time delay of -97.219 ns
Corresponding correlation value of 0.752 (max = 0.752)

If satisfied with current slider location, press Enter to lock it down.

int of -2119 chosen representing time delay of -33.141 ns
Corresponding correlation value of 0.779 (max = 0.779)
If satisfied with current slider location, press Enter to lock it down.
int of 4072 chosen representing time delay of 63.687 ns
Corresponding correlation value of 0.724 (max = 0.724)
If satisfied with current slider location, press Enter to lock it down.

int of 4581 chosen representing time delay of 71.648 ns
Corresponding correlation value of 0.882 (max = 0.882)
If satisfied with current slider location, press Enter to lock it down.
int of -2056 chosen representing time delay of -32.156 ns
Corresponding correlation value of 0.924 (max = 0.924)

If satisfied with current slider location, press Enter to lock it down.
int of 2580 chosen representing time delay of 40.352 ns
Corresponding correlation value of 0.871 (max = 0.871)
If satisfied with current slider location, press Enter to lock it down.
int of -6637 chosen representing time delay of -103.804 ns
Corresponding correlation value of 0.896 (max = 0.896)
If satisfied with current slider location, press Enter to lock it down.
int of -2011 chosen representing time delay of -31.452 ns
Corresponding correlation value of 0.863 (max = 0.863)



If satisfied with current slider location, press Enter to lock it down.

int of 4632 chosen representing time delay of 72.445 ns
Corresponding correlation value of 0.882 (max = 0.882)
(5/6)           

If satisfied with current slider location, press Enter to lock it down.
int of 2384 chosen representing time delay of 37.286 ns
Corresponding correlation value of 0.557 (max = 0.557)

If satisfied with current slider location, press Enter to lock it down.
int of -4173 chosen representing time delay of -65.266 ns
Corresponding correlation value of 0.857 (max = 0.857)


If satisfied with current slider location, press Enter to lock it down.
int of -891 chosen representing time delay of -13.935 ns
Corresponding correlation value of 0.741 (max = 0.741)

If satisfied with current slider location, press Enter to lock it down.
int of -6564 chosen representing time delay of -102.662 ns
Corresponding correlation value of 0.631 (max = 0.631)

If satisfied with current slider location, press Enter to lock it down.
int of -3210 chosen representing time delay of -50.205 ns
Corresponding correlation value of 0.720 (max = 0.720)
If satisfied with current slider location, press Enter to lock it down.
int of 3305 chosen representing time delay of 51.691 ns
Corresponding correlation value of 0.702 (max = 0.702)
If satisfied with current slider location, press Enter to lock it down.
int of 2481 chosen representing time delay of 38.803 ns
Corresponding correlation value of 0.915 (max = 0.915)


If satisfied with current slider location, press Enter to lock it down.
int of -4345 chosen representing time delay of -67.956 ns
Corresponding correlation value of 0.939 (max = 0.939)

If satisfied with current slider location, press Enter to lock it down.
int of -599 chosen representing time delay of -9.368 ns
Corresponding correlation value of 0.903 (max = 0.903)
If satisfied with current slider location, press Enter to lock it down.
int of -6825 chosen representing time delay of -106.744 ns
Corresponding correlation value of 0.919 (max = 0.919)
If satisfied with current slider location, press Enter to lock it down.
int of -3076 chosen representing time delay of -48.109 ns
Corresponding correlation value of 0.920 (max = 0.920)
If satisfied with current slider location, press Enter to lock it down.
int of 3744 chosen representing time delay of 58.557 ns
Corresponding correlation value of 0.892 (max = 0.892)
(6/6)           



If satisfied with current slider location, press Enter to lock it down.
int of 2042 chosen representing time delay of 31.937 ns
Corresponding correlation value of 0.621 (max = 0.621)

If satisfied with current slider location, press Enter to lock it down.
int of -4449 chosen representing time delay of -69.583 ns
Corresponding correlation value of 0.832 (max = 0.832)

If satisfied with current slider location, press Enter to lock it down.
int of -1295 chosen representing time delay of -20.254 ns
Corresponding correlation value of 0.749 (max = 0.749)

If satisfied with current slider location, press Enter to lock it down.
int of -6503 chosen representing time delay of -101.708 ns
Corresponding correlation value of 0.616 (max = 0.616)
If satisfied with current slider location, press Enter to lock it down.
int of -3320 chosen representing time delay of -51.925 ns
Corresponding correlation value of 0.747 (max = 0.747)
If satisfied with current slider location, press Enter to lock it down.
int of 3141 chosen representing time delay of 49.126 ns
Corresponding correlation value of 0.755 (max = 0.755)
If satisfied with current slider location, press Enter to lock it down.
int of 2156 chosen representing time delay of 33.720 ns
Corresponding correlation value of 0.940 (max = 0.940)
If satisfied with current slider location, press Enter to lock it down.
int of -4621 chosen representing time delay of -72.273 ns
Corresponding correlation value of 0.941 (max = 0.941)
If satisfied with current slider location, press Enter to lock it down.

int of -1035 chosen representing time delay of -16.188 ns
Corresponding correlation value of 0.908 (max = 0.908)

If satisfied with current slider location, press Enter to lock it down.
int of -6777 chosen representing time delay of -105.993 ns
Corresponding correlation value of 0.934 (max = 0.934)
If satisfied with current slider location, press Enter to lock it down.
int of -3187 chosen representing time delay of -49.845 ns
Corresponding correlation value of 0.913 (max = 0.913)
If satisfied with current slider location, press Enter to lock it down.
int of 3583 chosen representing time delay of 56.039 ns
Corresponding correlation value of 0.905 (max = 0.905)


Calculating time delays:
(1/3)           


If satisfied with current slider location, press Enter to lock it down.
int of 5824 chosen representing time delay of 91.088 ns
Corresponding correlation value of 0.903 (max = 0.903)

If satisfied with current slider location, press Enter to lock it down.
int of 6982 chosen representing time delay of 109.200 ns
Corresponding correlation value of 0.601 (max = 0.601)

If satisfied with current slider location, press Enter to lock it down.
int of 9234 chosen representing time delay of 144.421 ns
Corresponding correlation value of 0.877 (max = 0.877)

If satisfied with current slider location, press Enter to lock it down.
int of 1209 chosen representing time delay of 18.909 ns
Corresponding correlation value of 0.674 (max = 0.674)

If satisfied with current slider location, press Enter to lock it down.
int of 3389 chosen representing time delay of 53.004 ns
Corresponding correlation value of 0.896 (max = 0.896)
If satisfied with current slider location, press Enter to lock it down.
int of 2153 chosen representing time delay of 33.673 ns
Corresponding correlation value of 0.593 (max = 0.593)

If satisfied with current slider location, press Enter to lock it down.
int of -899 chosen representing time delay of -14.060 ns
Corresponding correlation value of 0.104 (max = 0.104)

If satisfied with current slider location, press Enter to lock it down.
int of -16274 chosen representing time delay of -254.528 ns
Corresponding correlation value of 0.064 (max = 0.064)

If satisfied with current slider location, press Enter to lock it down.
int of 2571 chosen representing time delay of 40.211 ns
Corresponding correlation value of 0.076 (max = 0.076)
If satisfied with current slider location, press Enter to lock it down.
int of 668 chosen representing time delay of 10.448 ns
Corresponding correlation value of 0.439 (max = 0.439)
If satisfied with current slider location, press Enter to lock it down.
int of 3564 chosen representing time delay of 55.742 ns
Corresponding correlation value of 0.475 (max = 0.475)
If satisfied with current slider location, press Enter to lock it down.
int of 928 chosen representing time delay of 14.514 ns
Corresponding correlation value of 0.513 (max = 0.513)
(2/3)           

If satisfied with current slider location, press Enter to lock it down.
int of 6907 chosen representing time delay of 108.027 ns
Corresponding correlation value of 0.903 (max = 0.903)
If satisfied with current slider location, press Enter to lock it down.
int of 6379 chosen representing time delay of 99.769 ns
Corresponding correlation value of 0.880 (max = 0.880)
If satisfied with current slider location, press Enter to lock it down.
int of 9604 chosen representing time delay of 150.208 ns
Corresponding correlation value of 0.937 (max = 0.937)
If satisfied with current slider location, press Enter to lock it down.
int of -515 chosen representing time delay of -8.055 ns
Corresponding correlation value of 0.883 (max = 0.883)

If satisfied with current slider location, press Enter to lock it down.
int of 2704 chosen representing time delay of 42.291 ns
Corresponding correlation value of 0.942 (max = 0.942)
If satisfied with current slider location, press Enter to lock it down.
int of 3219 chosen representing time delay of 50.346 ns
Corresponding correlation value of 0.896 (max = 0.896)
If satisfied with current slider location, press Enter to lock it down.
int of -13582 chosen representing time delay of -212.425 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.
int of -3203 chosen representing time delay of -50.095 ns
Corresponding correlation value of 0.085 (max = 0.085)
If satisfied with current slider location, press Enter to lock it down.
int of -8448 chosen representing time delay of -132.128 ns
Corresponding correlation value of 0.067 (max = 0.067)
If satisfied with current slider location, press Enter to lock it down.
int of -875 chosen representing time delay of -13.685 ns
Corresponding correlation value of 0.507 (max = 0.507)
If satisfied with current slider location, press Enter to lock it down.
int of 2874 chosen representing time delay of 44.950 ns
Corresponding correlation value of 0.471 (max = 0.471)

If satisfied with current slider location, press Enter to lock it down.
int of 3779 chosen representing time delay of 59.104 ns
Corresponding correlation value of 0.426 (max = 0.426)
(3/3)           
If satisfied with current slider location, press Enter to lock it down.
int of 7122 chosen representing time delay of 111.389 ns
Corresponding correlation value of 0.719 (max = 0.719)
If satisfied with current slider location, press Enter to lock it down.
int of 6205 chosen representing time delay of 97.047 ns
Corresponding correlation value of 0.746 (max = 0.746)
If satisfied with current slider location, press Enter to lock it down.
int of 9587 chosen representing time delay of 149.942 ns
Corresponding correlation value of 0.791 (max = 0.791)
If satisfied with current slider location, press Enter to lock it down.
int of -914 chosen representing time delay of -14.295 ns
Corresponding correlation value of 0.754 (max = 0.754)

If satisfied with current slider location, press Enter to lock it down.
int of 2475 chosen representing time delay of 38.709 ns
Corresponding correlation value of 0.638 (max = 0.638)
If satisfied with current slider location, press Enter to lock it down.
int of 3369 chosen representing time delay of 52.692 ns
Corresponding correlation value of 0.795 (max = 0.795)
If satisfied with current slider location, press Enter to lock it down.
int of 816 chosen representing time delay of 12.762 ns
Corresponding correlation value of 0.069 (max = 0.069)

If satisfied with current slider location, press Enter to lock it down.
int of 14230 chosen representing time delay of 222.559 ns
Corresponding correlation value of 0.094 (max = 0.094)
If satisfied with current slider location, press Enter to lock it down.
int of 4979 chosen representing time delay of 77.872 ns
Corresponding correlation value of 0.084 (max = 0.084)
If satisfied with current slider location, press Enter to lock it down.
int of -1347 chosen representing time delay of -21.067 ns
Corresponding correlation value of 0.247 (max = 0.247)
If satisfied with current slider location, press Enter to lock it down.
int of 3517 chosen representing time delay of 55.006 ns
Corresponding correlation value of 0.325 (max = 0.325)
If satisfied with current slider location, press Enter to lock it down.
int of -15188 chosen representing time delay of -237.543 ns
Corresponding correlation value of 0.197 (max = 0.197)

Calculating time delays:
(1/5)           
If satisfied with current slider location, press Enter to lock it down.
int of 5972 chosen representing time delay of 93.403 ns
Corresponding correlation value of 0.951 (max = 0.951)
If satisfied with current slider location, press Enter to lock it down.
int of 806 chosen representing time delay of 12.606 ns
Corresponding correlation value of 0.958 (max = 0.958)
If satisfied with current slider location, press Enter to lock it down.
int of 4978 chosen representing time delay of 77.857 ns
Corresponding correlation value of 0.946 (max = 0.946)
If satisfied with current slider location, press Enter to lock it down.
int of -5169 chosen representing time delay of -80.844 ns
Corresponding correlation value of 0.939 (max = 0.939)
If satisfied with current slider location, press Enter to lock it down.
int of -993 chosen representing time delay of -15.531 ns
Corresponding correlation value of 0.932 (max = 0.932)
If satisfied with current slider location, press Enter to lock it down.
int of 4174 chosen representing time delay of 65.282 ns
Corresponding correlation value of 0.909 (max = 0.909)
If satisfied with current slider location, press Enter to lock it down.
int of 28308 chosen representing time delay of 442.741 ns
Corresponding correlation value of 0.063 (max = 0.063)
If satisfied with current slider location, press Enter to lock it down.
int of 22751 chosen representing time delay of 355.829 ns
Corresponding correlation value of 0.059 (max = 0.059)
If satisfied with current slider location, press Enter to lock it down.

int of 27475 chosen representing time delay of 429.713 ns
Corresponding correlation value of 0.062 (max = 0.062)
If satisfied with current slider location, press Enter to lock it down.
int of -5610 chosen representing time delay of -87.741 ns
Corresponding correlation value of 0.943 (max = 0.943)
If satisfied with current slider location, press Enter to lock it down.
int of -864 chosen representing time delay of -13.513 ns
Corresponding correlation value of 0.895 (max = 0.895)
If satisfied with current slider location, press Enter to lock it down.
int of 4742 chosen representing time delay of 74.166 ns
Corresponding correlation value of 0.894 (max = 0.894)
(2/5)           
If satisfied with current slider location, press Enter to lock it down.
int of 6045 chosen representing time delay of 94.545 ns
Corresponding correlation value of 0.924 (max = 0.924)
If satisfied with current slider location, press Enter to lock it down.
int of 824 chosen representing time delay of 12.887 ns
Corresponding correlation value of 0.910 (max = 0.910)
If satisfied with current slider location, press Enter to lock it down.
int of 5052 chosen representing time delay of 79.014 ns
Corresponding correlation value of 0.945 (max = 0.945)
If satisfied with current slider location, press Enter to lock it down.
int of -5225 chosen representing time delay of -81.720 ns
Corresponding correlation value of 0.914 (max = 0.914)
If satisfied with current slider location, press Enter to lock it down.
int of -997 chosen representing time delay of -15.593 ns
Corresponding correlation value of 0.934 (max = 0.934)
If satisfied with current slider location, press Enter to lock it down.
int of 4229 chosen representing time delay of 66.142 ns
Corresponding correlation value of 0.905 (max = 0.905)
If satisfied with current slider location, press Enter to lock it down.
int of 641 chosen representing time delay of 10.025 ns
Corresponding correlation value of 0.092 (max = 0.092)
If satisfied with current slider location, press Enter to lock it down.
int of 38138 chosen representing time delay of 596.484 ns
Corresponding correlation value of 0.081 (max = 0.081)
If satisfied with current slider location, press Enter to lock it down.
int of 42990 chosen representing time delay of 672.370 ns
Corresponding correlation value of 0.081 (max = 0.081)
If satisfied with current slider location, press Enter to lock it down.
int of -5665 chosen representing time delay of -88.601 ns
Corresponding correlation value of 0.870 (max = 0.870)
If satisfied with current slider location, press Enter to lock it down.
int of -856 chosen representing time delay of -13.388 ns
Corresponding correlation value of 0.860 (max = 0.860)
If satisfied with current slider location, press Enter to lock it down.
int of 4801 chosen representing time delay of 75.088 ns
Corresponding correlation value of 0.841 (max = 0.841)
(3/5)           
If satisfied with current slider location, press Enter to lock it down.
int of 6079 chosen representing time delay of 95.076 ns
Corresponding correlation value of 0.953 (max = 0.953)
If satisfied with current slider location, press Enter to lock it down.
int of 837 chosen representing time delay of 13.091 ns
Corresponding correlation value of 0.950 (max = 0.950)
If satisfied with current slider location, press Enter to lock it down.
int of 5086 chosen representing time delay of 79.546 ns
Corresponding correlation value of 0.932 (max = 0.932)
If satisfied with current slider location, press Enter to lock it down.
int of -5245 chosen representing time delay of -82.033 ns
Corresponding correlation value of 0.946 (max = 0.946)
If satisfied with current slider location, press Enter to lock it down.
int of -994 chosen representing time delay of -15.546 ns
Corresponding correlation value of 0.924 (max = 0.924)
If satisfied with current slider location, press Enter to lock it down.
int of 4253 chosen representing time delay of 66.518 ns
Corresponding correlation value of 0.905 (max = 0.905)
If satisfied with current slider location, press Enter to lock it down.
int of 19038 chosen representing time delay of 297.757 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.
int of 37752 chosen representing time delay of 590.447 ns
Corresponding correlation value of 0.074 (max = 0.074)
If satisfied with current slider location, press Enter to lock it down.
int of 18178 chosen representing time delay of 284.307 ns
Corresponding correlation value of 0.083 (max = 0.083)
If satisfied with current slider location, press Enter to lock it down.
int of -5681 chosen representing time delay of -88.852 ns
Corresponding correlation value of 0.926 (max = 0.926)
If satisfied with current slider location, press Enter to lock it down.
int of -856 chosen representing time delay of -13.388 ns
Corresponding correlation value of 0.897 (max = 0.897)
If satisfied with current slider location, press Enter to lock it down.
int of 4824 chosen representing time delay of 75.448 ns
Corresponding correlation value of 0.886 (max = 0.886)
(4/5)           
If satisfied with current slider location, press Enter to lock it down.
int of 6131 chosen representing time delay of 95.890 ns
Corresponding correlation value of 0.832 (max = 0.832)
If satisfied with current slider location, press Enter to lock it down.
int of 847 chosen representing time delay of 13.247 ns
Corresponding correlation value of 0.783 (max = 0.783)
If satisfied with current slider location, press Enter to lock it down.
int of 5120 chosen representing time delay of 80.078 ns
Corresponding correlation value of 0.881 (max = 0.881)
If satisfied with current slider location, press Enter to lock it down.
int of -5282 chosen representing time delay of -82.611 ns
Corresponding correlation value of 0.790 (max = 0.790)
If satisfied with current slider location, press Enter to lock it down.
int of -1014 chosen representing time delay of -15.859 ns
Corresponding correlation value of 0.865 (max = 0.865)
If satisfied with current slider location, press Enter to lock it down.
int of 4284 chosen representing time delay of 67.002 ns
Corresponding correlation value of 0.804 (max = 0.804)
If satisfied with current slider location, press Enter to lock it down.
int of 829 chosen representing time delay of 12.966 ns
Corresponding correlation value of 0.079 (max = 0.079)
If satisfied with current slider location, press Enter to lock it down.
int of 4585 chosen representing time delay of 71.710 ns
Corresponding correlation value of 0.069 (max = 0.069)
If satisfied with current slider location, press Enter to lock it down.
int of -9997 chosen representing time delay of -156.355 ns
Corresponding correlation value of 0.098 (max = 0.098)
If satisfied with current slider location, press Enter to lock it down.
int of -5708 chosen representing time delay of -89.274 ns
Corresponding correlation value of 0.453 (max = 0.453)
If satisfied with current slider location, press Enter to lock it down.
int of -820 chosen representing time delay of -12.825 ns
Corresponding correlation value of 0.525 (max = 0.525)
If satisfied with current slider location, press Enter to lock it down.
int of 5763 chosen representing time delay of 90.134 ns
Corresponding correlation value of 0.364 (max = 0.364)
(5/5)           
If satisfied with current slider location, press Enter to lock it down.
int of 6127 chosen representing time delay of 95.827 ns
Corresponding correlation value of 0.873 (max = 0.873)
If satisfied with current slider location, press Enter to lock it down.
int of 842 chosen representing time delay of 13.169 ns
Corresponding correlation value of 0.883 (max = 0.883)

If satisfied with current slider location, press Enter to lock it down.
int of 5121 chosen representing time delay of 80.093 ns
Corresponding correlation value of 0.887 (max = 0.887)



If satisfied with current slider location, press Enter to lock it down.
int of -5286 chosen representing time delay of -82.674 ns
Corresponding correlation value of 0.883 (max = 0.883)


If satisfied with current slider location, press Enter to lock it down.
int of -1005 chosen representing time delay of -15.718 ns
Corresponding correlation value of 0.915 (max = 0.915)
If satisfied with current slider location, press Enter to lock it down.
int of 4284 chosen representing time delay of 67.002 ns
Corresponding correlation value of 0.890 (max = 0.890)
If satisfied with current slider location, press Enter to lock it down.
int of -380 chosen representing time delay of -5.943 ns
Corresponding correlation value of 0.065 (max = 0.065)
If satisfied with current slider location, press Enter to lock it down.
int of 12680 chosen representing time delay of 198.317 ns
Corresponding correlation value of 0.056 (max = 0.056)
If satisfied with current slider location, press Enter to lock it down.
int of 14840 chosen representing time delay of 232.100 ns
Corresponding correlation value of 0.076 (max = 0.076)
If satisfied with current slider location, press Enter to lock it down.
int of -5712 chosen representing time delay of -89.337 ns
Corresponding correlation value of 0.825 (max = 0.825)
If satisfied with current slider location, press Enter to lock it down.
int of -851 chosen representing time delay of -13.310 ns
Corresponding correlation value of 0.825 (max = 0.825)
If satisfied with current slider location, press Enter to lock it down.
int of 4854 chosen representing time delay of 75.917 ns
Corresponding correlation value of 0.765 (max = 0.765)

Calculating time delays:
(1/26)          
If satisfied with current slider location, press Enter to lock it down.
int of -952 chosen representing time delay of -14.889 ns
Corresponding correlation value of 0.727 (max = 0.727)

If satisfied with current slider location, press Enter to lock it down.
int of 1091 chosen representing time delay of 17.063 ns
Corresponding correlation value of 0.777 (max = 0.777)

If satisfied with current slider location, press Enter to lock it down.
int of -753 chosen representing time delay of -11.777 ns
Corresponding correlation value of 0.752 (max = 0.752)
If satisfied with current slider location, press Enter to lock it down.

int of 2912 chosen representing time delay of 45.544 ns
Corresponding correlation value of 0.767 (max = 0.767)
If satisfied with current slider location, press Enter to lock it down.
int of 1061 chosen representing time delay of 16.594 ns
Corresponding correlation value of 0.600 (max = 0.600)
If satisfied with current slider location, press Enter to lock it down.
int of -2715 chosen representing time delay of -42.463 ns
Corresponding correlation value of 0.683 (max = 0.683)
If satisfied with current slider location, press Enter to lock it down.
int of -23678 chosen representing time delay of -370.328 ns
Corresponding correlation value of 0.092 (max = 0.092)
If satisfied with current slider location, press Enter to lock it down.
int of 39697 chosen representing time delay of 620.867 ns
Corresponding correlation value of 0.075 (max = 0.075)
If satisfied with current slider location, press Enter to lock it down.
int of 18692 chosen representing time delay of 292.346 ns
Corresponding correlation value of 0.076 (max = 0.076)
If satisfied with current slider location, press Enter to lock it down.
int of 2483 chosen representing time delay of 38.835 ns
Corresponding correlation value of 0.629 (max = 0.629)
If satisfied with current slider location, press Enter to lock it down.
int of 1276 chosen representing time delay of 19.957 ns
Corresponding correlation value of 0.683 (max = 0.683)
If satisfied with current slider location, press Enter to lock it down.
int of -1191 chosen representing time delay of -18.627 ns
Corresponding correlation value of 0.684 (max = 0.684)
(2/26)          
If satisfied with current slider location, press Enter to lock it down.
int of 53 chosen representing time delay of 0.829 ns
Corresponding correlation value of 0.723 (max = 0.723)
If satisfied with current slider location, press Enter to lock it down.
int of 1009 chosen representing time delay of 15.781 ns
Corresponding correlation value of 0.706 (max = 0.706)

If satisfied with current slider location, press Enter to lock it down.
int of 1881 chosen representing time delay of 29.419 ns
Corresponding correlation value of 0.662 (max = 0.662)

If satisfied with current slider location, press Enter to lock it down.
int of 1821 chosen representing time delay of 28.481 ns
Corresponding correlation value of 0.803 (max = 0.803)

If satisfied with current slider location, press Enter to lock it down.
int of 1824 chosen representing time delay of 28.528 ns
Corresponding correlation value of 0.740 (max = 0.740)

If satisfied with current slider location, press Enter to lock it down.
int of -1722 chosen representing time delay of -26.932 ns
Corresponding correlation value of 0.661 (max = 0.661)
If satisfied with current slider location, press Enter to lock it down.
int of 0 chosen representing time delay of 0.000 ns
Corresponding correlation value of 0.075 (max = 0.075)
If satisfied with current slider location, press Enter to lock it down.
int of 2294 chosen representing time delay of 35.879 ns
Corresponding correlation value of 0.062 (max = 0.062)
If satisfied with current slider location, press Enter to lock it down.
int of 3848 chosen representing time delay of 60.183 ns
Corresponding correlation value of 0.095 (max = 0.095)
If satisfied with current slider location, press Enter to lock it down.
int of 1399 chosen representing time delay of 21.881 ns
Corresponding correlation value of 0.802 (max = 0.802)
If satisfied with current slider location, press Enter to lock it down.
int of 2094 chosen representing time delay of 32.750 ns
Corresponding correlation value of 0.667 (max = 0.667)
If satisfied with current slider location, press Enter to lock it down.
int of 701 chosen representing time delay of 10.964 ns
Corresponding correlation value of 0.677 (max = 0.677)
(3/26)          
If satisfied with current slider location, press Enter to lock it down.
int of -709 chosen representing time delay of -11.089 ns
Corresponding correlation value of 0.809 (max = 0.809)
If satisfied with current slider location, press Enter to lock it down.
int of 1875 chosen representing time delay of 29.325 ns
Corresponding correlation value of 0.803 (max = 0.803)

If satisfied with current slider location, press Enter to lock it down.
int of -641 chosen representing time delay of -10.025 ns
Corresponding correlation value of 0.794 (max = 0.794)

If satisfied with current slider location, press Enter to lock it down.
int of 2573 chosen representing time delay of 40.242 ns
Corresponding correlation value of 0.853 (max = 0.853)

If satisfied with current slider location, press Enter to lock it down.
int of 1786 chosen representing time delay of 27.933 ns
Corresponding correlation value of 0.807 (max = 0.807)
If satisfied with current slider location, press Enter to lock it down.

int of -1634 chosen representing time delay of -25.556 ns
Corresponding correlation value of 0.829 (max = 0.829)
If satisfied with current slider location, press Enter to lock it down.
int of 77970 chosen representing time delay of 1219.463 ns
Corresponding correlation value of 0.077 (max = 0.077)
If satisfied with current slider location, press Enter to lock it down.
int of -7274 chosen representing time delay of -113.766 ns
Corresponding correlation value of 0.059 (max = 0.059)
If satisfied with current slider location, press Enter to lock it down.
int of -4611 chosen representing time delay of -72.117 ns
Corresponding correlation value of 0.081 (max = 0.081)
If satisfied with current slider location, press Enter to lock it down.
int of 1292 chosen representing time delay of 20.207 ns
Corresponding correlation value of 0.800 (max = 0.800)
If satisfied with current slider location, press Enter to lock it down.
int of 1181 chosen representing time delay of 18.471 ns
Corresponding correlation value of 0.632 (max = 0.632)
If satisfied with current slider location, press Enter to lock it down.
int of 793 chosen representing time delay of 12.403 ns
Corresponding correlation value of 0.636 (max = 0.636)
(4/26)          
If satisfied with current slider location, press Enter to lock it down.
int of -42 chosen representing time delay of -0.657 ns
Corresponding correlation value of 0.789 (max = 0.789)
If satisfied with current slider location, press Enter to lock it down.
int of 873 chosen representing time delay of 13.654 ns
Corresponding correlation value of 0.849 (max = 0.849)
If satisfied with current slider location, press Enter to lock it down.
int of 425 chosen representing time delay of 6.647 ns
Corresponding correlation value of 0.829 (max = 0.829)

If satisfied with current slider location, press Enter to lock it down.

int of 909 chosen representing time delay of 14.217 ns
Corresponding correlation value of 0.831 (max = 0.831)
If satisfied with current slider location, press Enter to lock it down.
int of 491 chosen representing time delay of 7.679 ns
Corresponding correlation value of 0.843 (max = 0.843)

If satisfied with current slider location, press Enter to lock it down.
int of -445 chosen representing time delay of -6.960 ns
Corresponding correlation value of 0.860 (max = 0.860)

If satisfied with current slider location, press Enter to lock it down.
int of -2973 chosen representing time delay of -46.498 ns
Corresponding correlation value of 0.062 (max = 0.062)
If satisfied with current slider location, press Enter to lock it down.
int of -2453 chosen representing time delay of -38.365 ns
Corresponding correlation value of 0.071 (max = 0.071)
If satisfied with current slider location, press Enter to lock it down.
int of -17935 chosen representing time delay of -280.506 ns
Corresponding correlation value of 0.083 (max = 0.083)
If satisfied with current slider location, press Enter to lock it down.
int of -5208 chosen representing time delay of -81.454 ns
Corresponding correlation value of 0.331 (max = 0.331)
If satisfied with current slider location, press Enter to lock it down.
int of 652 chosen representing time delay of 10.197 ns
Corresponding correlation value of 0.380 (max = 0.380)
If satisfied with current slider location, press Enter to lock it down.
int of 143 chosen representing time delay of 2.237 ns
Corresponding correlation value of 0.238 (max = 0.238)
(5/26)          
If satisfied with current slider location, press Enter to lock it down.
int of 196 chosen representing time delay of 3.065 ns
Corresponding correlation value of 0.842 (max = 0.842)
If satisfied with current slider location, press Enter to lock it down.
int of 844 chosen representing time delay of 13.200 ns
Corresponding correlation value of 0.841 (max = 0.841)

If satisfied with current slider location, press Enter to lock it down.
int of 607 chosen representing time delay of 9.494 ns
Corresponding correlation value of 0.887 (max = 0.887)
If satisfied with current slider location, press Enter to lock it down.
int of 637 chosen representing time delay of 9.963 ns
Corresponding correlation value of 0.850 (max = 0.850)
If satisfied with current slider location, press Enter to lock it down.
int of 414 chosen representing time delay of 6.475 ns
Corresponding correlation value of 0.849 (max = 0.849)
If satisfied with current slider location, press Enter to lock it down.
int of -239 chosen representing time delay of -3.738 ns
Corresponding correlation value of 0.833 (max = 0.833)

If satisfied with current slider location, press Enter to lock it down.
int of -1147 chosen representing time delay of -17.939 ns
Corresponding correlation value of 0.102 (max = 0.102)

If satisfied with current slider location, press Enter to lock it down.
int of -2036 chosen representing time delay of -31.843 ns
Corresponding correlation value of 0.079 (max = 0.079)

If satisfied with current slider location, press Enter to lock it down.
int of -3331 chosen representing time delay of -52.097 ns
Corresponding correlation value of 0.099 (max = 0.099)

If satisfied with current slider location, press Enter to lock it down.
int of 7777 chosen representing time delay of 121.633 ns
Corresponding correlation value of 0.212 (max = 0.212)

If satisfied with current slider location, press Enter to lock it down.

int of 2100 chosen representing time delay of 32.844 ns
Corresponding correlation value of 0.302 (max = 0.302)
If satisfied with current slider location, press Enter to lock it down.
int of 5097 chosen representing time delay of 79.718 ns
Corresponding correlation value of 0.241 (max = 0.241)
(6/26)          

If satisfied with current slider location, press Enter to lock it down.
int of 1245 chosen representing time delay of 19.472 ns
Corresponding correlation value of 0.826 (max = 0.826)

If satisfied with current slider location, press Enter to lock it down.
int of 855 chosen representing time delay of 13.372 ns
Corresponding correlation value of 0.778 (max = 0.778)
If satisfied with current slider location, press Enter to lock it down.
int of 726 chosen representing time delay of 11.355 ns
Corresponding correlation value of 0.863 (max = 0.863)

If satisfied with current slider location, press Enter to lock it down.
int of 451 chosen representing time delay of 7.054 ns
Corresponding correlation value of 0.806 (max = 0.806)

If satisfied with current slider location, press Enter to lock it down.
int of -515 chosen representing time delay of -8.055 ns
Corresponding correlation value of 0.843 (max = 0.843)

If satisfied with current slider location, press Enter to lock it down.

int of -111 chosen representing time delay of -1.736 ns
Corresponding correlation value of 0.773 (max = 0.773)
If satisfied with current slider location, press Enter to lock it down.

int of 299 chosen representing time delay of 4.676 ns
Corresponding correlation value of 0.082 (max = 0.082)
If satisfied with current slider location, press Enter to lock it down.
int of 20960 chosen representing time delay of 327.818 ns
Corresponding correlation value of 0.057 (max = 0.057)

If satisfied with current slider location, press Enter to lock it down.
int of 27664 chosen representing time delay of 432.669 ns
Corresponding correlation value of 0.086 (max = 0.086)

If satisfied with current slider location, press Enter to lock it down.
int of -10218 chosen representing time delay of -159.811 ns
Corresponding correlation value of 0.206 (max = 0.206)

If satisfied with current slider location, press Enter to lock it down.
int of -2194 chosen representing time delay of -34.314 ns
Corresponding correlation value of 0.405 (max = 0.405)


If satisfied with current slider location, press Enter to lock it down.
int of 12498 chosen representing time delay of 195.471 ns
Corresponding correlation value of 0.200 (max = 0.200)
(7/26)          

If satisfied with current slider location, press Enter to lock it down.
int of 357 chosen representing time delay of 5.584 ns
Corresponding correlation value of 0.758 (max = 0.758)

If satisfied with current slider location, press Enter to lock it down.
int of 844 chosen representing time delay of 13.200 ns
Corresponding correlation value of 0.757 (max = 0.757)
If satisfied with current slider location, press Enter to lock it down.
int of 721 chosen representing time delay of 11.277 ns
Corresponding correlation value of 0.740 (max = 0.740)
If satisfied with current slider location, press Enter to lock it down.
int of 474 chosen representing time delay of 7.413 ns
Corresponding correlation value of 0.720 (max = 0.720)
If satisfied with current slider location, press Enter to lock it down.
int of -536 chosen representing time delay of -8.383 ns
Corresponding correlation value of 0.756 (max = 0.756)
If satisfied with current slider location, press Enter to lock it down.
int of -109 chosen representing time delay of -1.705 ns
Corresponding correlation value of 0.666 (max = 0.666)
If satisfied with current slider location, press Enter to lock it down.
int of 2943 chosen representing time delay of 46.029 ns
Corresponding correlation value of 0.075 (max = 0.075)
If satisfied with current slider location, press Enter to lock it down.
int of -8453 chosen representing time delay of -132.206 ns
Corresponding correlation value of 0.085 (max = 0.085)
If satisfied with current slider location, press Enter to lock it down.
int of -25104 chosen representing time delay of -392.630 ns
Corresponding correlation value of 0.085 (max = 0.085)
If satisfied with current slider location, press Enter to lock it down.
int of -951 chosen representing time delay of -14.874 ns
Corresponding correlation value of 0.358 (max = 0.358)
If satisfied with current slider location, press Enter to lock it down.
int of -2994 chosen representing time delay of -46.827 ns
Corresponding correlation value of 0.416 (max = 0.416)
If satisfied with current slider location, press Enter to lock it down.
int of 1478 chosen representing time delay of 23.116 ns
Corresponding correlation value of 0.325 (max = 0.325)
(8/26)          
If satisfied with current slider location, press Enter to lock it down.
int of 1359 chosen representing time delay of 21.255 ns
Corresponding correlation value of 0.750 (max = 0.750)
If satisfied with current slider location, press Enter to lock it down.
int of 856 chosen representing time delay of 13.388 ns
Corresponding correlation value of 0.840 (max = 0.840)
If satisfied with current slider location, press Enter to lock it down.
int of 840 chosen representing time delay of 13.138 ns
Corresponding correlation value of 0.865 (max = 0.865)
If satisfied with current slider location, press Enter to lock it down.
int of 340 chosen representing time delay of 5.318 ns
Corresponding correlation value of 0.815 (max = 0.815)
If satisfied with current slider location, press Enter to lock it down.
int of -509 chosen representing time delay of -7.961 ns
Corresponding correlation value of 0.754 (max = 0.754)
If satisfied with current slider location, press Enter to lock it down.
int of -5 chosen representing time delay of -0.078 ns
Corresponding correlation value of 0.796 (max = 0.796)
If satisfied with current slider location, press Enter to lock it down.
int of -2344 chosen representing time delay of -36.661 ns
Corresponding correlation value of 0.055 (max = 0.055)
If satisfied with current slider location, press Enter to lock it down.
int of 269 chosen representing time delay of 4.207 ns
Corresponding correlation value of 0.074 (max = 0.074)
If satisfied with current slider location, press Enter to lock it down.
int of 38670 chosen representing time delay of 604.805 ns
Corresponding correlation value of 0.087 (max = 0.087)
If satisfied with current slider location, press Enter to lock it down.
int of -904 chosen representing time delay of -14.139 ns
Corresponding correlation value of 0.592 (max = 0.592)
If satisfied with current slider location, press Enter to lock it down.
int of 545 chosen representing time delay of 8.524 ns
Corresponding correlation value of 0.624 (max = 0.624)
If satisfied with current slider location, press Enter to lock it down.
int of 2277 chosen representing time delay of 35.613 ns
Corresponding correlation value of 0.607 (max = 0.607)
(9/26)          
If satisfied with current slider location, press Enter to lock it down.
int of 2707 chosen representing time delay of 42.338 ns
Corresponding correlation value of 0.789 (max = 0.789)
If satisfied with current slider location, press Enter to lock it down.
int of 811 chosen representing time delay of 12.684 ns
Corresponding correlation value of 0.761 (max = 0.761)
If satisfied with current slider location, press Enter to lock it down.
int of 2030 chosen representing time delay of 31.750 ns
Corresponding correlation value of 0.858 (max = 0.858)
If satisfied with current slider location, press Enter to lock it down.
int of -214 chosen representing time delay of -3.347 ns
Corresponding correlation value of 0.819 (max = 0.819)
If satisfied with current slider location, press Enter to lock it down.
int of -671 chosen representing time delay of -10.495 ns
Corresponding correlation value of 0.849 (max = 0.849)
If satisfied with current slider location, press Enter to lock it down.
int of -462 chosen representing time delay of -7.226 ns
Corresponding correlation value of 0.803 (max = 0.803)
If satisfied with current slider location, press Enter to lock it down.
int of 9 chosen representing time delay of 0.141 ns
Corresponding correlation value of 0.078 (max = 0.078)
If satisfied with current slider location, press Enter to lock it down.
int of 22799 chosen representing time delay of 356.580 ns
Corresponding correlation value of 0.054 (max = 0.054)
If satisfied with current slider location, press Enter to lock it down.
int of -13309 chosen representing time delay of -208.155 ns
Corresponding correlation value of 0.086 (max = 0.086)
If satisfied with current slider location, press Enter to lock it down.
int of 903 chosen representing time delay of 14.123 ns
Corresponding correlation value of 0.339 (max = 0.339)
If satisfied with current slider location, press Enter to lock it down.
int of 2116 chosen representing time delay of 33.095 ns
Corresponding correlation value of 0.413 (max = 0.413)
If satisfied with current slider location, press Enter to lock it down.
int of 1217 chosen representing time delay of 19.034 ns
Corresponding correlation value of 0.403 (max = 0.403)
(10/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 1384 chosen representing time delay of 21.646 ns
Corresponding correlation value of 0.852 (max = 0.852)
If satisfied with current slider location, press Enter to lock it down.

int of 760 chosen representing time delay of 11.887 ns
Corresponding correlation value of 0.818 (max = 0.818)
If satisfied with current slider location, press Enter to lock it down.
int of 1467 chosen representing time delay of 22.944 ns
Corresponding correlation value of 0.863 (max = 0.863)

If satisfied with current slider location, press Enter to lock it down.
int of -633 chosen representing time delay of -9.900 ns
Corresponding correlation value of 0.863 (max = 0.863)

If satisfied with current slider location, press Enter to lock it down.

int of 78 chosen representing time delay of 1.220 ns
Corresponding correlation value of 0.849 (max = 0.849)
If satisfied with current slider location, press Enter to lock it down.
int of 707 chosen representing time delay of 11.058 ns
Corresponding correlation value of 0.845 (max = 0.845)

If satisfied with current slider location, press Enter to lock it down.
int of 3197 chosen representing time delay of 50.002 ns
Corresponding correlation value of 0.085 (max = 0.085)

If satisfied with current slider location, press Enter to lock it down.
int of 2974 chosen representing time delay of 46.514 ns
Corresponding correlation value of 0.068 (max = 0.068)

If satisfied with current slider location, press Enter to lock it down.
int of 9220 chosen representing time delay of 144.202 ns
Corresponding correlation value of 0.108 (max = 0.108)

If satisfied with current slider location, press Enter to lock it down.
int of -117 chosen representing time delay of -1.830 ns
Corresponding correlation value of 0.190 (max = 0.190)

If satisfied with current slider location, press Enter to lock it down.
int of -3710 chosen representing time delay of -58.025 ns
Corresponding correlation value of 0.241 (max = 0.241)
If satisfied with current slider location, press Enter to lock it down.
int of -82 chosen representing time delay of -1.282 ns
Corresponding correlation value of 0.225 (max = 0.225)
(11/26)         

If satisfied with current slider location, press Enter to lock it down.
int of 2284 chosen representing time delay of 35.722 ns
Corresponding correlation value of 0.832 (max = 0.832)

If satisfied with current slider location, press Enter to lock it down.

int of 787 chosen representing time delay of 12.309 ns
Corresponding correlation value of 0.758 (max = 0.758)
If satisfied with current slider location, press Enter to lock it down.
int of 1485 chosen representing time delay of 23.226 ns
Corresponding correlation value of 0.870 (max = 0.870)

If satisfied with current slider location, press Enter to lock it down.
int of -639 chosen representing time delay of -9.994 ns
Corresponding correlation value of 0.840 (max = 0.840)

If satisfied with current slider location, press Enter to lock it down.
int of 71 chosen representing time delay of 1.110 ns
Corresponding correlation value of 0.878 (max = 0.878)

If satisfied with current slider location, press Enter to lock it down.

int of 702 chosen representing time delay of 10.979 ns
Corresponding correlation value of 0.820 (max = 0.820)
If satisfied with current slider location, press Enter to lock it down.
int of -17045 chosen representing time delay of -266.586 ns
Corresponding correlation value of 0.107 (max = 0.107)

If satisfied with current slider location, press Enter to lock it down.
int of -18255 chosen representing time delay of -285.511 ns
Corresponding correlation value of 0.062 (max = 0.062)
If satisfied with current slider location, press Enter to lock it down.
int of -17680 chosen representing time delay of -276.518 ns
Corresponding correlation value of 0.114 (max = 0.114)
If satisfied with current slider location, press Enter to lock it down.
int of 472 chosen representing time delay of 7.382 ns
Corresponding correlation value of 0.341 (max = 0.341)

If satisfied with current slider location, press Enter to lock it down.
int of -1433 chosen representing time delay of -22.412 ns
Corresponding correlation value of 0.333 (max = 0.333)
If satisfied with current slider location, press Enter to lock it down.
int of 696 chosen representing time delay of 10.886 ns
Corresponding correlation value of 0.480 (max = 0.480)
(12/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 2307 chosen representing time delay of 36.082 ns
Corresponding correlation value of 0.732 (max = 0.732)
If satisfied with current slider location, press Enter to lock it down.
int of 717 chosen representing time delay of 11.214 ns
Corresponding correlation value of 0.770 (max = 0.770)
If satisfied with current slider location, press Enter to lock it down.
int of 2965 chosen representing time delay of 46.373 ns
Corresponding correlation value of 0.857 (max = 0.857)
If satisfied with current slider location, press Enter to lock it down.
int of -1603 chosen representing time delay of -25.071 ns
Corresponding correlation value of 0.856 (max = 0.856)
If satisfied with current slider location, press Enter to lock it down.
int of -1021 chosen representing time delay of -15.969 ns
Corresponding correlation value of 0.770 (max = 0.770)
If satisfied with current slider location, press Enter to lock it down.
int of 573 chosen representing time delay of 8.962 ns
Corresponding correlation value of 0.775 (max = 0.775)
If satisfied with current slider location, press Enter to lock it down.
int of 914 chosen representing time delay of 14.295 ns
Corresponding correlation value of 0.076 (max = 0.076)
If satisfied with current slider location, press Enter to lock it down.
int of 25241 chosen representing time delay of 394.773 ns
Corresponding correlation value of 0.072 (max = 0.072)
If satisfied with current slider location, press Enter to lock it down.
int of 21002 chosen representing time delay of 328.475 ns
Corresponding correlation value of 0.118 (max = 0.118)
If satisfied with current slider location, press Enter to lock it down.
int of -2032 chosen representing time delay of -31.781 ns
Corresponding correlation value of 0.236 (max = 0.236)
If satisfied with current slider location, press Enter to lock it down.
int of 8656 chosen representing time delay of 135.381 ns
Corresponding correlation value of 0.254 (max = 0.254)
If satisfied with current slider location, press Enter to lock it down.
int of -2771 chosen representing time delay of -43.339 ns
Corresponding correlation value of 0.195 (max = 0.195)
(13/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 2376 chosen representing time delay of 37.161 ns
Corresponding correlation value of 0.767 (max = 0.767)
If satisfied with current slider location, press Enter to lock it down.
int of -173 chosen representing time delay of -2.706 ns
Corresponding correlation value of 0.774 (max = 0.774)
If satisfied with current slider location, press Enter to lock it down.
int of 3016 chosen representing time delay of 47.171 ns
Corresponding correlation value of 0.868 (max = 0.868)
If satisfied with current slider location, press Enter to lock it down.
int of -1676 chosen representing time delay of -26.213 ns
Corresponding correlation value of 0.850 (max = 0.850)
If satisfied with current slider location, press Enter to lock it down.
int of -1047 chosen representing time delay of -16.375 ns
Corresponding correlation value of 0.763 (max = 0.763)
If satisfied with current slider location, press Enter to lock it down.
int of 629 chosen representing time delay of 9.838 ns
Corresponding correlation value of 0.777 (max = 0.777)
If satisfied with current slider location, press Enter to lock it down.

int of -388 chosen representing time delay of -6.068 ns
Corresponding correlation value of 0.076 (max = 0.076)
If satisfied with current slider location, press Enter to lock it down.
int of -14018 chosen representing time delay of -219.244 ns
Corresponding correlation value of 0.073 (max = 0.073)
If satisfied with current slider location, press Enter to lock it down.
int of 16392 chosen representing time delay of 256.373 ns
Corresponding correlation value of 0.121 (max = 0.121)
If satisfied with current slider location, press Enter to lock it down.
int of -3015 chosen representing time delay of -47.155 ns
Corresponding correlation value of 0.220 (max = 0.220)
If satisfied with current slider location, press Enter to lock it down.
int of -28 chosen representing time delay of -0.438 ns
Corresponding correlation value of 0.278 (max = 0.278)
If satisfied with current slider location, press Enter to lock it down.
int of -2769 chosen representing time delay of -43.308 ns
Corresponding correlation value of 0.279 (max = 0.279)
(14/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 3446 chosen representing time delay of 53.896 ns
Corresponding correlation value of 0.788 (max = 0.788)
If satisfied with current slider location, press Enter to lock it down.
int of 707 chosen representing time delay of 11.058 ns
Corresponding correlation value of 0.826 (max = 0.826)
If satisfied with current slider location, press Enter to lock it down.
int of 2967 chosen representing time delay of 46.404 ns
Corresponding correlation value of 0.875 (max = 0.875)

If satisfied with current slider location, press Enter to lock it down.
int of -2739 chosen representing time delay of -42.838 ns
Corresponding correlation value of 0.875 (max = 0.875)
If satisfied with current slider location, press Enter to lock it down.
int of -476 chosen representing time delay of -7.445 ns
Corresponding correlation value of 0.844 (max = 0.844)

If satisfied with current slider location, press Enter to lock it down.
int of 2260 chosen representing time delay of 35.347 ns
Corresponding correlation value of 0.855 (max = 0.855)
If satisfied with current slider location, press Enter to lock it down.
int of 5257 chosen representing time delay of 82.220 ns
Corresponding correlation value of 0.097 (max = 0.097)

If satisfied with current slider location, press Enter to lock it down.
int of -14022 chosen representing time delay of -219.306 ns
Corresponding correlation value of 0.082 (max = 0.082)
If satisfied with current slider location, press Enter to lock it down.
int of 38399 chosen representing time delay of 600.566 ns
Corresponding correlation value of 0.072 (max = 0.072)
If satisfied with current slider location, press Enter to lock it down.
int of -3132 chosen representing time delay of -48.985 ns
Corresponding correlation value of 0.341 (max = 0.341)
If satisfied with current slider location, press Enter to lock it down.
int of 641 chosen representing time delay of 10.025 ns
Corresponding correlation value of 0.311 (max = 0.311)
If satisfied with current slider location, press Enter to lock it down.
int of 1428 chosen representing time delay of 22.334 ns
Corresponding correlation value of 0.274 (max = 0.274)
(15/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 3408 chosen representing time delay of 53.302 ns
Corresponding correlation value of 0.852 (max = 0.866)
If satisfied with current slider location, press Enter to lock it down.
int of 721 chosen representing time delay of 11.277 ns
Corresponding correlation value of 0.838 (max = 0.847)
If satisfied with current slider location, press Enter to lock it down.
int of 2946 chosen representing time delay of 46.076 ns
Corresponding correlation value of 0.933 (max = 0.933)
If satisfied with current slider location, press Enter to lock it down.
int of -2729 chosen representing time delay of -42.682 ns
Corresponding correlation value of 0.940 (max = 0.940)
If satisfied with current slider location, press Enter to lock it down.
int of -470 chosen representing time delay of -7.351 ns
Corresponding correlation value of 0.928 (max = 0.928)
If satisfied with current slider location, press Enter to lock it down.
int of 2249 chosen representing time delay of 35.175 ns
Corresponding correlation value of 0.891 (max = 0.891)
If satisfied with current slider location, press Enter to lock it down.
int of 66618 chosen representing time delay of 1041.916 ns
Corresponding correlation value of 0.047 (max = 0.047)
If satisfied with current slider location, press Enter to lock it down.
int of 51470 chosen representing time delay of 804.999 ns
Corresponding correlation value of 0.056 (max = 0.056)
If satisfied with current slider location, press Enter to lock it down.
int of 41221 chosen representing time delay of 644.703 ns
Corresponding correlation value of 0.071 (max = 0.071)
If satisfied with current slider location, press Enter to lock it down.
int of -3130 chosen representing time delay of -48.954 ns
Corresponding correlation value of 0.734 (max = 0.734)
If satisfied with current slider location, press Enter to lock it down.
int of -307 chosen representing time delay of -4.802 ns
Corresponding correlation value of 0.687 (max = 0.687)
If satisfied with current slider location, press Enter to lock it down.
int of 2802 chosen representing time delay of 43.824 ns
Corresponding correlation value of 0.719 (max = 0.719)
(16/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 3511 chosen representing time delay of 54.913 ns
Corresponding correlation value of 0.867 (max = 0.871)
If satisfied with current slider location, press Enter to lock it down.
int of 695 chosen representing time delay of 10.870 ns
Corresponding correlation value of 0.851 (max = 0.851)
If satisfied with current slider location, press Enter to lock it down.
int of 3010 chosen representing time delay of 47.077 ns
Corresponding correlation value of 0.934 (max = 0.934)
If satisfied with current slider location, press Enter to lock it down.
int of -2824 chosen representing time delay of -44.168 ns
Corresponding correlation value of 0.933 (max = 0.933)
If satisfied with current slider location, press Enter to lock it down.
int of -494 chosen representing time delay of -7.726 ns
Corresponding correlation value of 0.933 (max = 0.933)
If satisfied with current slider location, press Enter to lock it down.
int of 2320 chosen representing time delay of 36.285 ns
Corresponding correlation value of 0.897 (max = 0.897)
If satisfied with current slider location, press Enter to lock it down.
int of 18444 chosen representing time delay of 288.467 ns
Corresponding correlation value of 0.064 (max = 0.064)
If satisfied with current slider location, press Enter to lock it down.
int of 5292 chosen representing time delay of 82.768 ns
Corresponding correlation value of 0.067 (max = 0.067)
If satisfied with current slider location, press Enter to lock it down.
int of -11273 chosen representing time delay of -176.311 ns
Corresponding correlation value of 0.099 (max = 0.099)
If satisfied with current slider location, press Enter to lock it down.
int of -4121 chosen representing time delay of -64.453 ns
Corresponding correlation value of 0.729 (max = 0.729)
If satisfied with current slider location, press Enter to lock it down.
int of 583 chosen representing time delay of 9.118 ns
Corresponding correlation value of 0.706 (max = 0.706)
If satisfied with current slider location, press Enter to lock it down.
int of 2880 chosen representing time delay of 45.044 ns
Corresponding correlation value of 0.735 (max = 0.735)
(17/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 3538 chosen representing time delay of 55.335 ns
Corresponding correlation value of 0.843 (max = 0.843)
If satisfied with current slider location, press Enter to lock it down.
int of 692 chosen representing time delay of 10.823 ns
Corresponding correlation value of 0.842 (max = 0.842)
If satisfied with current slider location, press Enter to lock it down.
int of 3024 chosen representing time delay of 47.296 ns
Corresponding correlation value of 0.924 (max = 0.924)
If satisfied with current slider location, press Enter to lock it down.
int of -2856 chosen representing time delay of -44.668 ns
Corresponding correlation value of 0.919 (max = 0.919)
If satisfied with current slider location, press Enter to lock it down.
int of -509 chosen representing time delay of -7.961 ns
Corresponding correlation value of 0.910 (max = 0.910)
If satisfied with current slider location, press Enter to lock it down.
int of 2340 chosen representing time delay of 36.598 ns
Corresponding correlation value of 0.885 (max = 0.885)
If satisfied with current slider location, press Enter to lock it down.
int of 30970 chosen representing time delay of 484.376 ns
Corresponding correlation value of 0.057 (max = 0.057)
If satisfied with current slider location, press Enter to lock it down.
int of 41100 chosen representing time delay of 642.810 ns
Corresponding correlation value of 0.067 (max = 0.067)
If satisfied with current slider location, press Enter to lock it down.
int of 13062 chosen representing time delay of 204.292 ns
Corresponding correlation value of 0.080 (max = 0.080)
If satisfied with current slider location, press Enter to lock it down.
int of -3241 chosen representing time delay of -50.690 ns
Corresponding correlation value of 0.424 (max = 0.424)
If satisfied with current slider location, press Enter to lock it down.
int of 627 chosen representing time delay of 9.806 ns
Corresponding correlation value of 0.408 (max = 0.408)
If satisfied with current slider location, press Enter to lock it down.
int of 2891 chosen representing time delay of 45.216 ns
Corresponding correlation value of 0.307 (max = 0.350)
(18/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 3605 chosen representing time delay of 56.383 ns
Corresponding correlation value of 0.874 (max = 0.893)
If satisfied with current slider location, press Enter to lock it down.
int of 696 chosen representing time delay of 10.886 ns
Corresponding correlation value of 0.862 (max = 0.862)
If satisfied with current slider location, press Enter to lock it down.
int of 3094 chosen representing time delay of 48.391 ns
Corresponding correlation value of 0.950 (max = 0.950)
If satisfied with current slider location, press Enter to lock it down.
int of -2961 chosen representing time delay of -46.310 ns
Corresponding correlation value of 0.924 (max = 0.942)
If satisfied with current slider location, press Enter to lock it down.
int of -521 chosen representing time delay of -8.149 ns
Corresponding correlation value of 0.936 (max = 0.936)
If satisfied with current slider location, press Enter to lock it down.
int of 2407 chosen representing time delay of 37.646 ns
Corresponding correlation value of 0.886 (max = 0.886)
If satisfied with current slider location, press Enter to lock it down.
int of 22515 chosen representing time delay of 352.138 ns
Corresponding correlation value of 0.069 (max = 0.069)
If satisfied with current slider location, press Enter to lock it down.
int of 67205 chosen representing time delay of 1051.097 ns
Corresponding correlation value of 0.075 (max = 0.075)
If satisfied with current slider location, press Enter to lock it down.
int of -22018 chosen representing time delay of -344.365 ns
Corresponding correlation value of 0.086 (max = 0.086)
If satisfied with current slider location, press Enter to lock it down.
int of -3329 chosen representing time delay of -52.066 ns
Corresponding correlation value of 0.774 (max = 0.787)
If satisfied with current slider location, press Enter to lock it down.
int of -354 chosen representing time delay of -5.537 ns
Corresponding correlation value of 0.786 (max = 0.786)
If satisfied with current slider location, press Enter to lock it down.
int of 2969 chosen representing time delay of 46.436 ns
Corresponding correlation value of 0.789 (max = 0.789)
(19/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 3706 chosen representing time delay of 57.962 ns
Corresponding correlation value of 0.904 (max = 0.904)
If satisfied with current slider location, press Enter to lock it down.
int of 703 chosen representing time delay of 10.995 ns
Corresponding correlation value of 0.882 (max = 0.882)
If satisfied with current slider location, press Enter to lock it down.
int of 3158 chosen representing time delay of 49.392 ns
Corresponding correlation value of 0.936 (max = 0.936)
If satisfied with current slider location, press Enter to lock it down.
int of -3012 chosen representing time delay of -47.108 ns
Corresponding correlation value of 0.950 (max = 0.950)
If satisfied with current slider location, press Enter to lock it down.
int of -544 chosen representing time delay of -8.508 ns
Corresponding correlation value of 0.937 (max = 0.937)
If satisfied with current slider location, press Enter to lock it down.
int of 2462 chosen representing time delay of 38.506 ns
Corresponding correlation value of 0.906 (max = 0.906)
If satisfied with current slider location, press Enter to lock it down.
int of 24446 chosen representing time delay of 382.339 ns
Corresponding correlation value of 0.067 (max = 0.067)
If satisfied with current slider location, press Enter to lock it down.
int of 22017 chosen representing time delay of 344.349 ns
Corresponding correlation value of 0.059 (max = 0.059)
If satisfied with current slider location, press Enter to lock it down.
int of 26883 chosen representing time delay of 420.454 ns
Corresponding correlation value of 0.077 (max = 0.077)
If satisfied with current slider location, press Enter to lock it down.
int of -3420 chosen representing time delay of -53.489 ns
Corresponding correlation value of 0.736 (max = 0.736)
If satisfied with current slider location, press Enter to lock it down.
int of -387 chosen representing time delay of -6.053 ns
Corresponding correlation value of 0.720 (max = 0.720)
If satisfied with current slider location, press Enter to lock it down.
int of 3021 chosen representing time delay of 47.249 ns
Corresponding correlation value of 0.760 (max = 0.760)
(20/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 3778 chosen representing time delay of 59.089 ns
Corresponding correlation value of 0.869 (max = 0.878)
If satisfied with current slider location, press Enter to lock it down.
int of 694 chosen representing time delay of 10.854 ns
Corresponding correlation value of 0.877 (max = 0.877)
If satisfied with current slider location, press Enter to lock it down.
int of 3235 chosen representing time delay of 50.596 ns
Corresponding correlation value of 0.904 (max = 0.909)
If satisfied with current slider location, press Enter to lock it down.
int of -3108 chosen representing time delay of -48.610 ns
Corresponding correlation value of 0.931 (max = 0.931)
If satisfied with current slider location, press Enter to lock it down.
int of -562 chosen representing time delay of -8.790 ns
Corresponding correlation value of 0.920 (max = 0.920)
If satisfied with current slider location, press Enter to lock it down.
int of 2537 chosen representing time delay of 39.679 ns
Corresponding correlation value of 0.900 (max = 0.900)
If satisfied with current slider location, press Enter to lock it down.
int of 132 chosen representing time delay of 2.065 ns
Corresponding correlation value of 0.063 (max = 0.063)
If satisfied with current slider location, press Enter to lock it down.
int of 40956 chosen representing time delay of 640.558 ns
Corresponding correlation value of 0.066 (max = 0.066)
If satisfied with current slider location, press Enter to lock it down.
int of 44039 chosen representing time delay of 688.777 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.
int of -3489 chosen representing time delay of -54.568 ns
Corresponding correlation value of 0.769 (max = 0.769)
If satisfied with current slider location, press Enter to lock it down.
int of -423 chosen representing time delay of -6.616 ns
Corresponding correlation value of 0.715 (max = 0.750)
If satisfied with current slider location, press Enter to lock it down.
int of 3085 chosen representing time delay of 48.250 ns
Corresponding correlation value of 0.790 (max = 0.790)
(21/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 5535 chosen representing time delay of 86.568 ns
Corresponding correlation value of 0.508 (max = 0.608)
If satisfied with current slider location, press Enter to lock it down.
int of 672 chosen representing time delay of 10.510 ns
Corresponding correlation value of 0.760 (max = 0.760)
If satisfied with current slider location, press Enter to lock it down.
int of 4532 chosen representing time delay of 70.881 ns
Corresponding correlation value of 0.633 (max = 0.633)
If satisfied with current slider location, press Enter to lock it down.
int of -4796 chosen representing time delay of -75.010 ns
Corresponding correlation value of 0.666 (max = 0.666)
If satisfied with current slider location, press Enter to lock it down.
int of -964 chosen representing time delay of -15.077 ns
Corresponding correlation value of 0.583 (max = 0.583)
If satisfied with current slider location, press Enter to lock it down.
int of 3836 chosen representing time delay of 59.996 ns
Corresponding correlation value of 0.658 (max = 0.658)
If satisfied with current slider location, press Enter to lock it down.
int of 120 chosen representing time delay of 1.877 ns
Corresponding correlation value of 0.093 (max = 0.093)
If satisfied with current slider location, press Enter to lock it down.
int of -5157 chosen representing time delay of -80.656 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.
int of 25353 chosen representing time delay of 396.525 ns
Corresponding correlation value of 0.083 (max = 0.083)
If satisfied with current slider location, press Enter to lock it down.
int of -5140 chosen representing time delay of -80.390 ns
Corresponding correlation value of 0.327 (max = 0.327)
If satisfied with current slider location, press Enter to lock it down.
int of -785 chosen representing time delay of -12.278 ns
Corresponding correlation value of 0.415 (max = 0.415)
If satisfied with current slider location, press Enter to lock it down.
int of 4365 chosen representing time delay of 68.269 ns
Corresponding correlation value of 0.359 (max = 0.359)
(22/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 5424 chosen representing time delay of 84.832 ns
Corresponding correlation value of 0.489 (max = 0.715)
If satisfied with current slider location, press Enter to lock it down.
int of 680 chosen representing time delay of 10.635 ns
Corresponding correlation value of 0.892 (max = 0.892)
If satisfied with current slider location, press Enter to lock it down.
int of 4565 chosen representing time delay of 71.397 ns
Corresponding correlation value of 0.722 (max = 0.722)
If satisfied with current slider location, press Enter to lock it down.
int of -4855 chosen representing time delay of -75.933 ns
Corresponding correlation value of 0.820 (max = 0.820)
If satisfied with current slider location, press Enter to lock it down.
int of -981 chosen representing time delay of -15.343 ns
Corresponding correlation value of 0.801 (max = 0.801)
If satisfied with current slider location, press Enter to lock it down.
int of 3876 chosen representing time delay of 60.621 ns
Corresponding correlation value of 0.782 (max = 0.782)
If satisfied with current slider location, press Enter to lock it down.
int of -4786 chosen representing time delay of -74.854 ns
Corresponding correlation value of 0.069 (max = 0.069)
If satisfied with current slider location, press Enter to lock it down.
int of -11662 chosen representing time delay of -182.395 ns
Corresponding correlation value of 0.083 (max = 0.083)
If satisfied with current slider location, press Enter to lock it down.
int of 56085 chosen representing time delay of 877.178 ns
Corresponding correlation value of 0.086 (max = 0.086)
If satisfied with current slider location, press Enter to lock it down.
int of -5220 chosen representing time delay of -81.642 ns
Corresponding correlation value of 0.761 (max = 0.761)
If satisfied with current slider location, press Enter to lock it down.
int of -802 chosen representing time delay of -12.543 ns
Corresponding correlation value of 0.825 (max = 0.825)
If satisfied with current slider location, press Enter to lock it down.
int of 4418 chosen representing time delay of 69.098 ns
Corresponding correlation value of 0.733 (max = 0.733)
(23/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 5544 chosen representing time delay of 86.709 ns
Corresponding correlation value of 0.691 (max = 0.717)
If satisfied with current slider location, press Enter to lock it down.
int of 687 chosen representing time delay of 10.745 ns
Corresponding correlation value of 0.832 (max = 0.832)
If satisfied with current slider location, press Enter to lock it down.
int of 4619 chosen representing time delay of 72.242 ns
Corresponding correlation value of 0.708 (max = 0.735)
If satisfied with current slider location, press Enter to lock it down.
int of -4862 chosen representing time delay of -76.042 ns
Corresponding correlation value of 0.783 (max = 0.783)
If satisfied with current slider location, press Enter to lock it down.
int of -977 chosen representing time delay of -15.280 ns
Corresponding correlation value of 0.730 (max = 0.730)
If satisfied with current slider location, press Enter to lock it down.
int of 3890 chosen representing time delay of 60.840 ns
Corresponding correlation value of 0.787 (max = 0.787)
If satisfied with current slider location, press Enter to lock it down.
int of 6180 chosen representing time delay of 96.656 ns
Corresponding correlation value of 0.079 (max = 0.079)
If satisfied with current slider location, press Enter to lock it down.
int of -25844 chosen representing time delay of -404.204 ns
Corresponding correlation value of 0.066 (max = 0.066)
If satisfied with current slider location, press Enter to lock it down.
int of 26382 chosen representing time delay of 412.619 ns
Corresponding correlation value of 0.082 (max = 0.082)
If satisfied with current slider location, press Enter to lock it down.
int of -5229 chosen representing time delay of -81.782 ns
Corresponding correlation value of 0.755 (max = 0.755)
If satisfied with current slider location, press Enter to lock it down.
int of -783 chosen representing time delay of -12.246 ns
Corresponding correlation value of 0.754 (max = 0.754)
If satisfied with current slider location, press Enter to lock it down.
int of 4440 chosen representing time delay of 69.442 ns
Corresponding correlation value of 0.727 (max = 0.727)
(24/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 6553 chosen representing time delay of 102.490 ns
Corresponding correlation value of 0.788 (max = 0.788)
If satisfied with current slider location, press Enter to lock it down.
int of 740 chosen representing time delay of 11.574 ns
Corresponding correlation value of 0.843 (max = 0.843)
If satisfied with current slider location, press Enter to lock it down.
int of 4700 chosen representing time delay of 73.509 ns
Corresponding correlation value of 0.764 (max = 0.786)
If satisfied with current slider location, press Enter to lock it down.
int of -4911 chosen representing time delay of -76.809 ns
Corresponding correlation value of 0.768 (max = 0.801)
If satisfied with current slider location, press Enter to lock it down.
int of -999 chosen representing time delay of -15.625 ns
Corresponding correlation value of 0.755 (max = 0.755)
If satisfied with current slider location, press Enter to lock it down.
int of 3944 chosen representing time delay of 61.685 ns
Corresponding correlation value of 0.808 (max = 0.808)
If satisfied with current slider location, press Enter to lock it down.
int of 41849 chosen representing time delay of 654.525 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.
int of 36471 chosen representing time delay of 570.412 ns
Corresponding correlation value of 0.058 (max = 0.058)
If satisfied with current slider location, press Enter to lock it down.
int of 53256 chosen representing time delay of 832.932 ns
Corresponding correlation value of 0.070 (max = 0.070)
If satisfied with current slider location, press Enter to lock it down.
int of -5365 chosen representing time delay of -83.909 ns
Corresponding correlation value of 0.930 (max = 0.930)
If satisfied with current slider location, press Enter to lock it down.
int of -823 chosen representing time delay of -12.872 ns
Corresponding correlation value of 0.916 (max = 0.916)
If satisfied with current slider location, press Enter to lock it down.
int of 4540 chosen representing time delay of 71.006 ns
Corresponding correlation value of 0.907 (max = 0.907)
(25/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 7402 chosen representing time delay of 115.768 ns
Corresponding correlation value of 0.727 (max = 0.727)
If satisfied with current slider location, press Enter to lock it down.
int of 745 chosen representing time delay of 11.652 ns
Corresponding correlation value of 0.848 (max = 0.848)
If satisfied with current slider location, press Enter to lock it down.
int of 4663 chosen representing time delay of 72.930 ns
Corresponding correlation value of 0.769 (max = 0.777)
If satisfied with current slider location, press Enter to lock it down.
int of -5807 chosen representing time delay of -90.822 ns
Corresponding correlation value of 0.723 (max = 0.785)
If satisfied with current slider location, press Enter to lock it down.
int of -986 chosen representing time delay of -15.421 ns
Corresponding correlation value of 0.739 (max = 0.739)
If satisfied with current slider location, press Enter to lock it down.
int of 3964 chosen representing time delay of 61.998 ns
Corresponding correlation value of 0.759 (max = 0.767)
If satisfied with current slider location, press Enter to lock it down.
int of 66223 chosen representing time delay of 1035.738 ns
Corresponding correlation value of 0.095 (max = 0.095)
If satisfied with current slider location, press Enter to lock it down.
int of 61733 chosen representing time delay of 965.514 ns
Corresponding correlation value of 0.107 (max = 0.107)
If satisfied with current slider location, press Enter to lock it down.
int of 66314 chosen representing time delay of 1037.161 ns
Corresponding correlation value of 0.124 (max = 0.124)
If satisfied with current slider location, press Enter to lock it down.
int of -5362 chosen representing time delay of -83.863 ns
Corresponding correlation value of 0.908 (max = 0.908)
If satisfied with current slider location, press Enter to lock it down.
int of -813 chosen representing time delay of -12.715 ns
Corresponding correlation value of 0.894 (max = 0.894)
If satisfied with current slider location, press Enter to lock it down.
int of 4551 chosen representing time delay of 71.178 ns
Corresponding correlation value of 0.890 (max = 0.890)
(26/26)         
If satisfied with current slider location, press Enter to lock it down.
int of 5740 chosen representing time delay of 89.774 ns
Corresponding correlation value of 0.782 (max = 0.782)
If satisfied with current slider location, press Enter to lock it down.
int of 763 chosen representing time delay of 11.933 ns
Corresponding correlation value of 0.884 (max = 0.884)
If satisfied with current slider location, press Enter to lock it down.
int of 4723 chosen representing time delay of 73.868 ns
Corresponding correlation value of 0.761 (max = 0.765)
If satisfied with current slider location, press Enter to lock it down.
int of -4980 chosen representing time delay of -77.888 ns
Corresponding correlation value of 0.830 (max = 0.830)
If satisfied with current slider location, press Enter to lock it down.
int of -1002 chosen representing time delay of -15.671 ns
Corresponding correlation value of 0.834 (max = 0.834)
If satisfied with current slider location, press Enter to lock it down.
int of 4017 chosen representing time delay of 62.827 ns
Corresponding correlation value of 0.722 (max = 0.770)
If satisfied with current slider location, press Enter to lock it down.
int of 39310 chosen representing time delay of 614.814 ns
Corresponding correlation value of 0.077 (max = 0.077)
If satisfied with current slider location, press Enter to lock it down.
int of 34801 chosen representing time delay of 544.293 ns
Corresponding correlation value of 0.066 (max = 0.066)
If satisfied with current slider location, press Enter to lock it down.
int of 34840 chosen representing time delay of 544.903 ns
Corresponding correlation value of 0.067 (max = 0.067)
If satisfied with current slider location, press Enter to lock it down.
int of -5430 chosen representing time delay of -84.926 ns
Corresponding correlation value of 0.943 (max = 0.943)
If satisfied with current slider location, press Enter to lock it down.
int of -834 chosen representing time delay of -13.044 ns
Corresponding correlation value of 0.939 (max = 0.939)
If satisfied with current slider location, press Enter to lock it down.
int of 4593 chosen representing time delay of 71.835 ns
Corresponding correlation value of 0.929 (max = 0.929)

Calculating time delays:
(1/3)           
If satisfied with current slider location, press Enter to lock it down.
int of 4180 chosen representing time delay of 65.376 ns
Corresponding correlation value of 0.775 (max = 0.775)
If satisfied with current slider location, press Enter to lock it down.
int of -2826 chosen representing time delay of -44.199 ns
Corresponding correlation value of 0.761 (max = 0.761)
If satisfied with current slider location, press Enter to lock it down.
int of 1121 chosen representing time delay of 17.533 ns
Corresponding correlation value of 0.700 (max = 0.700)
If satisfied with current slider location, press Enter to lock it down.
int of -7004 chosen representing time delay of -109.544 ns
Corresponding correlation value of 0.852 (max = 0.852)
If satisfied with current slider location, press Enter to lock it down.
int of -3063 chosen representing time delay of -47.906 ns
Corresponding correlation value of 0.854 (max = 0.854)
If satisfied with current slider location, press Enter to lock it down.
int of 3940 chosen representing time delay of 61.622 ns
Corresponding correlation value of 0.827 (max = 0.827)
If satisfied with current slider location, press Enter to lock it down.
int of 60840 chosen representing time delay of 951.547 ns
Corresponding correlation value of 0.083 (max = 0.083)
If satisfied with current slider location, press Enter to lock it down.
int of -24694 chosen representing time delay of -386.218 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.
int of -14339 chosen representing time delay of -224.264 ns
Corresponding correlation value of 0.081 (max = 0.081)
If satisfied with current slider location, press Enter to lock it down.
int of -7414 chosen representing time delay of -115.956 ns
Corresponding correlation value of 0.665 (max = 0.665)
If satisfied with current slider location, press Enter to lock it down.
int of -2966 chosen representing time delay of -46.389 ns
Corresponding correlation value of 0.778 (max = 0.778)
If satisfied with current slider location, press Enter to lock it down.
int of 4460 chosen representing time delay of 69.755 ns
Corresponding correlation value of 0.670 (max = 0.670)
(2/3)           
If satisfied with current slider location, press Enter to lock it down.
int of 4251 chosen representing time delay of 66.486 ns
Corresponding correlation value of 0.816 (max = 0.816)
If satisfied with current slider location, press Enter to lock it down.
int of -2747 chosen representing time delay of -42.964 ns
Corresponding correlation value of 0.839 (max = 0.839)
If satisfied with current slider location, press Enter to lock it down.
int of 1254 chosen representing time delay of 19.613 ns
Corresponding correlation value of 0.824 (max = 0.824)
If satisfied with current slider location, press Enter to lock it down.
int of -6999 chosen representing time delay of -109.465 ns
Corresponding correlation value of 0.856 (max = 0.856)
If satisfied with current slider location, press Enter to lock it down.
int of -3030 chosen representing time delay of -47.390 ns
Corresponding correlation value of 0.855 (max = 0.855)
If satisfied with current slider location, press Enter to lock it down.
int of 3969 chosen representing time delay of 62.076 ns
Corresponding correlation value of 0.861 (max = 0.861)
If satisfied with current slider location, press Enter to lock it down.
int of 43163 chosen representing time delay of 675.076 ns
Corresponding correlation value of 0.081 (max = 0.081)
If satisfied with current slider location, press Enter to lock it down.
int of 35970 chosen representing time delay of 562.576 ns
Corresponding correlation value of 0.077 (max = 0.077)
If satisfied with current slider location, press Enter to lock it down.
int of 40447 chosen representing time delay of 632.597 ns
Corresponding correlation value of 0.082 (max = 0.082)
If satisfied with current slider location, press Enter to lock it down.
int of -7367 chosen representing time delay of -115.221 ns
Corresponding correlation value of 0.717 (max = 0.717)
If satisfied with current slider location, press Enter to lock it down.
int of -2888 chosen representing time delay of -45.169 ns
Corresponding correlation value of 0.841 (max = 0.841)
If satisfied with current slider location, press Enter to lock it down.
int of 4484 chosen representing time delay of 70.130 ns
Corresponding correlation value of 0.742 (max = 0.742)
(3/3)           
If satisfied with current slider location, press Enter to lock it down.
int of 5613 chosen representing time delay of 87.788 ns
Corresponding correlation value of 0.611 (max = 0.611)
If satisfied with current slider location, press Enter to lock it down.
int of -728 chosen representing time delay of -11.386 ns
Corresponding correlation value of 0.519 (max = 0.519)
If satisfied with current slider location, press Enter to lock it down.
int of 3663 chosen representing time delay of 57.290 ns
Corresponding correlation value of 0.681 (max = 0.681)
If satisfied with current slider location, press Enter to lock it down.
int of -6304 chosen representing time delay of -98.596 ns
Corresponding correlation value of 0.703 (max = 0.703)
If satisfied with current slider location, press Enter to lock it down.
int of -1974 chosen representing time delay of -30.874 ns
Corresponding correlation value of 0.803 (max = 0.803)
If satisfied with current slider location, press Enter to lock it down.
int of 4340 chosen representing time delay of 67.878 ns
Corresponding correlation value of 0.745 (max = 0.745)
If satisfied with current slider location, press Enter to lock it down.
int of 84868 chosen representing time delay of 1327.349 ns
Corresponding correlation value of 0.063 (max = 0.063)
If satisfied with current slider location, press Enter to lock it down.
int of -13192 chosen representing time delay of -206.325 ns
Corresponding correlation value of 0.065 (max = 0.065)
If satisfied with current slider location, press Enter to lock it down.
int of 82970 chosen representing time delay of 1297.664 ns
Corresponding correlation value of 0.078 (max = 0.078)
If satisfied with current slider location, press Enter to lock it down.
int of -6678 chosen representing time delay of -104.445 ns
Corresponding correlation value of 0.830 (max = 0.830)
If satisfied with current slider location, press Enter to lock it down.
int of -1839 chosen representing time delay of -28.762 ns
Corresponding correlation value of 0.883 (max = 0.883)
If satisfied with current slider location, press Enter to lock it down.
int of 4844 chosen representing time delay of 75.761 ns
Corresponding correlation value of 0.833 (max = 0.833)

Calculating time delays:
(1/6)           
If satisfied with current slider location, press Enter to lock it down.
int of 4482 chosen representing time delay of 70.099 ns
Corresponding correlation value of 0.825 (max = 0.825)
If satisfied with current slider location, press Enter to lock it down.
int of -2537 chosen representing time delay of -39.679 ns
Corresponding correlation value of 0.842 (max = 0.842)
If satisfied with current slider location, press Enter to lock it down.
int of 1565 chosen representing time delay of 24.477 ns
Corresponding correlation value of 0.861 (max = 0.861)
If satisfied with current slider location, press Enter to lock it down.
int of -7016 chosen representing time delay of -109.731 ns
Corresponding correlation value of 0.888 (max = 0.888)
If satisfied with current slider location, press Enter to lock it down.
int of -2923 chosen representing time delay of -45.716 ns
Corresponding correlation value of 0.895 (max = 0.895)
If satisfied with current slider location, press Enter to lock it down.
int of 4091 chosen representing time delay of 63.984 ns
Corresponding correlation value of 0.844 (max = 0.844)
If satisfied with current slider location, press Enter to lock it down.
int of -16012 chosen representing time delay of -250.430 ns
Corresponding correlation value of 0.076 (max = 0.076)
If satisfied with current slider location, press Enter to lock it down.
int of -22624 chosen representing time delay of -353.843 ns
Corresponding correlation value of 0.085 (max = 0.085)
If satisfied with current slider location, press Enter to lock it down.
int of 14885 chosen representing time delay of 232.804 ns
Corresponding correlation value of 0.092 (max = 0.092)
If satisfied with current slider location, press Enter to lock it down.
int of -7402 chosen representing time delay of -115.768 ns
Corresponding correlation value of 0.746 (max = 0.746)
If satisfied with current slider location, press Enter to lock it down.
int of -2848 chosen representing time delay of -44.543 ns
Corresponding correlation value of 0.815 (max = 0.815)
If satisfied with current slider location, press Enter to lock it down.
int of 4568 chosen representing time delay of 71.444 ns
Corresponding correlation value of 0.725 (max = 0.725)
(2/6)           
If satisfied with current slider location, press Enter to lock it down.
int of 4978 chosen representing time delay of 77.857 ns
Corresponding correlation value of 0.695 (max = 0.695)
If satisfied with current slider location, press Enter to lock it down.
int of -1824 chosen representing time delay of -28.528 ns
Corresponding correlation value of 0.643 (max = 0.643)
If satisfied with current slider location, press Enter to lock it down.
int of 2433 chosen representing time delay of 38.052 ns
Corresponding correlation value of 0.700 (max = 0.700)
If satisfied with current slider location, press Enter to lock it down.
int of -6774 chosen representing time delay of -105.946 ns
Corresponding correlation value of 0.747 (max = 0.747)
If satisfied with current slider location, press Enter to lock it down.
int of -2570 chosen representing time delay of -40.195 ns
Corresponding correlation value of 0.898 (max = 0.898)
If satisfied with current slider location, press Enter to lock it down.
int of 4234 chosen representing time delay of 66.220 ns
Corresponding correlation value of 0.730 (max = 0.730)
If satisfied with current slider location, press Enter to lock it down.
int of 512 chosen representing time delay of 8.008 ns
Corresponding correlation value of 0.082 (max = 0.082)
If satisfied with current slider location, press Enter to lock it down.
int of -6654 chosen representing time delay of -104.070 ns
Corresponding correlation value of 0.096 (max = 0.096)
If satisfied with current slider location, press Enter to lock it down.
int of 78611 chosen representing time delay of 1229.488 ns
Corresponding correlation value of 0.083 (max = 0.083)
If satisfied with current slider location, press Enter to lock it down.
int of -7167 chosen representing time delay of -112.093 ns
Corresponding correlation value of 0.890 (max = 0.890)
If satisfied with current slider location, press Enter to lock it down.
int of -2423 chosen representing time delay of -37.896 ns
Corresponding correlation value of 0.937 (max = 0.937)
If satisfied with current slider location, press Enter to lock it down.
int of 4739 chosen representing time delay of 74.119 ns
Corresponding correlation value of 0.867 (max = 0.867)
(3/6)           
If satisfied with current slider location, press Enter to lock it down.
int of 5042 chosen representing time delay of 78.858 ns
Corresponding correlation value of 0.671 (max = 0.671)
If satisfied with current slider location, press Enter to lock it down.
int of -1695 chosen representing time delay of -26.510 ns
Corresponding correlation value of 0.545 (max = 0.545)
If satisfied with current slider location, press Enter to lock it down.
int of 2577 chosen representing time delay of 40.305 ns
Corresponding correlation value of 0.700 (max = 0.700)
If satisfied with current slider location, press Enter to lock it down.
int of -6710 chosen representing time delay of -104.945 ns
Corresponding correlation value of 0.745 (max = 0.745)
If satisfied with current slider location, press Enter to lock it down.
int of -2484 chosen representing time delay of -38.850 ns
Corresponding correlation value of 0.828 (max = 0.828)
If satisfied with current slider location, press Enter to lock it down.
int of 4244 chosen representing time delay of 66.377 ns
Corresponding correlation value of 0.735 (max = 0.735)
If satisfied with current slider location, press Enter to lock it down.
int of 26994 chosen representing time delay of 422.190 ns
Corresponding correlation value of 0.070 (max = 0.070)
If satisfied with current slider location, press Enter to lock it down.
int of 19869 chosen representing time delay of 310.754 ns
Corresponding correlation value of 0.084 (max = 0.084)
If satisfied with current slider location, press Enter to lock it down.
int of 25533 chosen representing time delay of 399.340 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.
int of -7103 chosen representing time delay of -111.092 ns
Corresponding correlation value of 0.858 (max = 0.858)
If satisfied with current slider location, press Enter to lock it down.
int of -2349 chosen representing time delay of -36.739 ns
Corresponding correlation value of 0.916 (max = 0.916)
If satisfied with current slider location, press Enter to lock it down.
int of 4748 chosen representing time delay of 74.259 ns
Corresponding correlation value of 0.849 (max = 0.849)
(4/6)           
If satisfied with current slider location, press Enter to lock it down.
int of 5661 chosen representing time delay of 88.539 ns
Corresponding correlation value of 0.710 (max = 0.710)
If satisfied with current slider location, press Enter to lock it down.
int of -594 chosen representing time delay of -9.290 ns
Corresponding correlation value of 0.621 (max = 0.621)
If satisfied with current slider location, press Enter to lock it down.
int of 3753 chosen representing time delay of 58.697 ns
Corresponding correlation value of 0.686 (max = 0.686)
If satisfied with current slider location, press Enter to lock it down.
int of -6261 chosen representing time delay of -97.923 ns
Corresponding correlation value of 0.869 (max = 0.869)
If satisfied with current slider location, press Enter to lock it down.
int of -1915 chosen representing time delay of -29.951 ns
Corresponding correlation value of 0.880 (max = 0.880)
If satisfied with current slider location, press Enter to lock it down.
int of 4350 chosen representing time delay of 68.035 ns
Corresponding correlation value of 0.874 (max = 0.874)
If satisfied with current slider location, press Enter to lock it down.
int of 56769 chosen representing time delay of 887.876 ns
Corresponding correlation value of 0.094 (max = 0.094)
If satisfied with current slider location, press Enter to lock it down.
int of 50110 chosen representing time delay of 783.728 ns
Corresponding correlation value of 0.098 (max = 0.098)
If satisfied with current slider location, press Enter to lock it down.
int of 60428 chosen representing time delay of 945.103 ns
Corresponding correlation value of 0.081 (max = 0.081)
If satisfied with current slider location, press Enter to lock it down.
int of -6649 chosen representing time delay of -103.991 ns
Corresponding correlation value of 0.784 (max = 0.784)
If satisfied with current slider location, press Enter to lock it down.
int of -1780 chosen representing time delay of -27.839 ns
Corresponding correlation value of 0.842 (max = 0.842)
If satisfied with current slider location, press Enter to lock it down.
int of 4874 chosen representing time delay of 76.230 ns
Corresponding correlation value of 0.786 (max = 0.786)
(5/6)           
If satisfied with current slider location, press Enter to lock it down.
int of 7043 chosen representing time delay of 110.154 ns
Corresponding correlation value of 0.651 (max = 0.651)
If satisfied with current slider location, press Enter to lock it down.
int of 2506 chosen representing time delay of 39.194 ns
Corresponding correlation value of 0.740 (max = 0.740)
If satisfied with current slider location, press Enter to lock it down.
int of 6954 chosen representing time delay of 108.762 ns
Corresponding correlation value of 0.664 (max = 0.664)
If satisfied with current slider location, press Enter to lock it down.
int of -4513 chosen representing time delay of -70.584 ns
Corresponding correlation value of 0.805 (max = 0.805)
If satisfied with current slider location, press Enter to lock it down.
int of -77 chosen representing time delay of -1.204 ns
Corresponding correlation value of 0.879 (max = 0.879)
If satisfied with current slider location, press Enter to lock it down.
int of 4430 chosen representing time delay of 69.286 ns
Corresponding correlation value of 0.815 (max = 0.815)
If satisfied with current slider location, press Enter to lock it down.
int of 512 chosen representing time delay of 8.008 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.
int of -28294 chosen representing time delay of -442.523 ns
Corresponding correlation value of 0.068 (max = 0.068)
If satisfied with current slider location, press Enter to lock it down.
int of 66353 chosen representing time delay of 1037.771 ns
Corresponding correlation value of 0.064 (max = 0.064)
If satisfied with current slider location, press Enter to lock it down.
int of -4875 chosen representing time delay of -76.246 ns
Corresponding correlation value of 0.751 (max = 0.751)
If satisfied with current slider location, press Enter to lock it down.
int of 75 chosen representing time delay of 1.173 ns
Corresponding correlation value of 0.877 (max = 0.877)
If satisfied with current slider location, press Enter to lock it down.
int of 4958 chosen representing time delay of 77.544 ns
Corresponding correlation value of 0.745 (max = 0.745)
(6/6)           
If satisfied with current slider location, press Enter to lock it down.
int of 7245 chosen representing time delay of 113.313 ns
Corresponding correlation value of 0.392 (max = 0.392)
If satisfied with current slider location, press Enter to lock it down.
int of 3060 chosen representing time delay of 47.859 ns
Corresponding correlation value of 0.083 (max = 0.547)
If satisfied with current slider location, press Enter to lock it down.
int of 7471 chosen representing time delay of 116.848 ns
Corresponding correlation value of 0.397 (max = 0.397)
If satisfied with current slider location, press Enter to lock it down.
int of -4112 chosen representing time delay of -64.312 ns
Corresponding correlation value of 0.612 (max = 0.618)
If satisfied with current slider location, press Enter to lock it down.
int of -578 chosen representing time delay of -9.040 ns
Corresponding correlation value of 0.753 (max = 0.753)
If satisfied with current slider location, press Enter to lock it down.
int of 4382 chosen representing time delay of 68.535 ns
Corresponding correlation value of 0.589 (max = 0.589)
If satisfied with current slider location, press Enter to lock it down.
int of -1261 chosen representing time delay of -19.722 ns
Corresponding correlation value of 0.077 (max = 0.077)
If satisfied with current slider location, press Enter to lock it down.
int of -2433 chosen representing time delay of -38.052 ns
Corresponding correlation value of 0.088 (max = 0.088)
If satisfied with current slider location, press Enter to lock it down.
int of 27908 chosen representing time delay of 436.485 ns
Corresponding correlation value of 0.060 (max = 0.060)
If satisfied with current slider location, press Enter to lock it down.
int of -3674 chosen representing time delay of -57.462 ns
Corresponding correlation value of 0.538 (max = 0.538)
If satisfied with current slider location, press Enter to lock it down.
int of 483 chosen representing time delay of 7.554 ns
Corresponding correlation value of 0.812 (max = 0.812)
If satisfied with current slider location, press Enter to lock it down.
int of 5010 chosen representing time delay of 78.357 ns
Corresponding correlation value of 0.556 (max = 0.556)



'''
