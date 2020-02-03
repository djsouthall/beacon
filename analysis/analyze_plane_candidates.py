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
 [ 54.52157817  10.86990729  47.07686468 -44.16779597  -7.72623626
   36.28515816]
 [ 55.3348662   10.82298683  47.29582684 -44.66828091  -7.96083858
   36.59796125]
 [ 56.75812024  10.88554745  48.39063765 -45.76309172  -8.14852043
   37.64585159]
 [ 57.96241213  10.99502853  49.39160753 -47.10814499  -8.50824398
   38.50606008]
 [ 58.58801831  10.85426714  51.15894498 -48.60959981  -8.78976676
   39.67907166]
 [ 86.20853094  10.51018374  70.88117965 -75.01018041 -15.07710882
   59.9956322 ]
 [ 85.48908384  10.63530498  71.39730474 -75.93294951 -15.34299145
   60.62123838]
 [ 85.42652322  10.74478606  71.7883086  -76.04243059 -15.28043083
   60.84020054]
 [102.48993165  11.57371424  73.27412327 -91.35414173 -28.7622439
   61.68476887]
 [ 88.36687225  11.65191501  73.55564605 -77.27800279 -15.42119222
   62.26345459]
 [ 89.77448614  11.93343779  74.18125222 -77.88796881 -15.67143469
   61.65348857]]


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
 [ 288.46700738   82.76769698 -176.31146032  -50.51769865   -5.30201233
    45.04364461]
 [ 484.37558114  642.81034501  204.29169651  -50.68974035    9.8063768
    45.65361063]
 [ 352.13807586 1051.0965751  -344.36491914  -52.75424072   -5.53661465
    46.43561835]
 [ 382.33921397  344.34927898  420.45427019  -53.48932798   -6.05273975
    47.24890638]
 [   2.06450038  640.55816278  688.77675874  -54.56849863   -6.75654669
    48.24987626]
 [   1.87681853  -80.65627614  396.52483399  -80.39039351  -12.27752119
    68.26927387]
 [ -74.85377886 -182.39548038  877.1780584   -81.64160586  -12.54340381
    69.09820205]
 [  96.65615407 -404.20414979  412.61855285  -81.78236725  -12.24624088
    69.44228545]
 [ 654.52482064  570.41207039  832.93206165  -83.90942825  -12.87184705
    71.00630088]
 [1035.7379435   965.51365033 1037.16119755  -83.86250778  -12.71544551
    71.17834258]
 [ 614.81446867  544.29301257  544.9029786   -84.92603828  -13.04388875
    71.83522907]]

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
 [0.85879361 0.8511148  0.93432178 0.93254486 0.93313052 0.89721675]
 [0.84340874 0.84170433 0.92365669 0.91877931 0.91033016 0.88505468]
 [0.87844958 0.86249058 0.95011396 0.92865062 0.93632529 0.8856151 ]
 [0.9038838  0.88233514 0.93607073 0.95002208 0.93720703 0.90594311]
 [0.8270224  0.87739643 0.85520432 0.93087253 0.92047593 0.89961645]
 [0.54853226 0.75981388 0.6329627  0.66585003 0.58271038 0.65777595]
 [0.6162375  0.89178027 0.72180865 0.82017094 0.80124434 0.78161604]
 [0.54786448 0.83198925 0.72843138 0.78335336 0.72950841 0.78732965]
 [0.78751439 0.84312842 0.77640076 0.78734325 0.68034616 0.80829344]
 [0.69112644 0.84812847 0.75834261 0.78549372 0.7385571  0.74197444]
 [0.78201732 0.88369476 0.75535317 0.82951779 0.83410446 0.73752075]]

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
 [0.06374236 0.06665853 0.09883411 0.7067953  0.64468391 0.73509211]
 [0.05717604 0.06735745 0.08034756 0.42368594 0.40780072 0.31449616]
 [0.06886223 0.07458845 0.08641286 0.74389539 0.78604399 0.78936861]
 [0.06727275 0.0591851  0.07705009 0.73627352 0.71987617 0.76038748]
 [0.06342815 0.06639673 0.06819119 0.76919342 0.70656756 0.78952973]
 [0.09327135 0.06787646 0.08285837 0.32722777 0.41452416 0.35887881]
 [0.06860293 0.08340955 0.08617786 0.76052345 0.82473454 0.73281971]
 [0.07858032 0.06649049 0.08155051 0.75495761 0.75391884 0.72652801]
 [0.06814221 0.05833277 0.07026216 0.92986665 0.91561262 0.90739401]
 [0.09521795 0.10702754 0.12377197 0.90834791 0.89390763 0.89032708]
 [0.07720588 0.06618241 0.06720744 0.94268425 0.93906486 0.92917063]]


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
 [ 113.31291848   88.16355024  116.84759337  -63.76490941    4.78588724
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
 [-1.97222347e+01 -3.80524956e+01  4.36485428e+02 -7.01148121e+01
   7.55419457e+00  7.83571734e+01]]
[[0.82452304 0.84237292 0.86070192 0.88783822 0.89508353 0.84408384]
 [0.69477446 0.64290981 0.70018481 0.74739501 0.89817133 0.73033073]
 [0.67095742 0.54535095 0.70012358 0.74522961 0.82753476 0.73537265]
 [0.71017735 0.62077517 0.68634733 0.86860821 0.88005062 0.87374545]
 [0.65076326 0.74020585 0.66372195 0.80522205 0.87859943 0.81493234]
 [0.39197714 0.54749722 0.39682659 0.5929514  0.68362202 0.58922157]]
[[0.07563383 0.08511365 0.09201061 0.74565473 0.81545159 0.72463237]
 [0.08151722 0.0958623  0.08309501 0.89035292 0.93663151 0.8665272 ]
 [0.06950143 0.08449305 0.06821813 0.85802633 0.91646282 0.84926463]
 [0.09422863 0.09829601 0.08132951 0.78374033 0.84211843 0.78580355]
 [0.06832285 0.06841074 0.06360799 0.75114916 0.87657423 0.74458405]
 [0.07721145 0.0884296  0.05952351 0.53699158 0.81185286 0.55579139]]



'''
