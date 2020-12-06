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
from tools.fftmath import TimeDelayCalculator, TemplateCompareTool
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
confirmed_candidates = {\
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

all_candidates = {\
'1705-55163':{  'eventids':numpy.array([[1705,55163],[1705,55643]]),\
                'known_flight':'a405d9',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]}},\
'1720-71316':{  'eventids':numpy.array([[1720,71316],[1720,71324]]),\
                'known_flight':'a678ef',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]}},\
'1728-62026':{  'eventids':numpy.array([[1728,62026],[1728,62182],[1728,62370],[1728,62382],[1728,62552],[1728,62577]]),\
                'known_flight':'a44585',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]}},\
'1759-510':{  'eventids':numpy.array([[1759,510]]),\
                'known_flight':'a04abd',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]}},\
'1772-71053':{  'eventids':numpy.array([[1772,71053]]),\
                'known_flight':'ab81b5',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]}},\
'1773-14413':{  'eventids':numpy.array([[1773,14413],[1773,14540],[1773,14590]]),\
                'known_flight':'aa8c39',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1773-63659':{  'eventids':numpy.array([[1773,62999],[1773,63659],[1773,63707],[1773,63727],[1773,63752],[1773,63757]]),\
                'known_flight':'a28392',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1774-178':{  'eventids':numpy.array([[1774,178],[1774,381],[1774,1348],[1774,1485]]),\
                'known_flight':'a1c2b3',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1774-88800':{  'eventids':numpy.array([[1774,88800],[1774,88810],[1774,88815],[1774,88895],[1774,88913],[1774,88921],[1774,88923],[1774,88925],[1774,88944],[1774,88955],[1774,88959],[1774,88988],[1774,88993],[1774,89029],[1774,89030],[1774,89032],[1774,89034],[1774,89041],[1774,89043],[1774,89052],[1774,89172],[1774,89175],[1774,89181],[1774,89203],[1774,89204],[1774,89213]]),\
                'known_flight':'ab5f43',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1783-28830':{  'eventids':numpy.array([[1783,28830],[1783,28832],[1783,28842],[1783,28843],[1783,28861],[1783,28886]]),\
                'known_flight':'a52e4f',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1783-34681':{  'eventids':numpy.array([[1783,34681],[1783,34725],[1783,34730],[1783,34738],[1783,34778],[1783,34793],[1783,34811],[1783,34826]]),\
                'known_flight':None,\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1784-7166':{   'eventids':numpy.array([[1784,7166],[1784,7176],[1784,7179],[1784,7195],[1784,7203],[1784,7244],[1784,7255]]),\
                'known_flight':'acf975',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}}\
}

#THESE ARE JUST THE NEW ONES FROM THE UNBLINDED SEARCH.
all_candidates = {\
'1705-55163':{  'eventids':numpy.array([[1705,55163],[1705,55643]]),\
                'known_flight':'a405d9',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]}},\
'1720-71316':{  'eventids':numpy.array([[1720,71316],[1720,71324]]),\
                'known_flight':'a678ef',\
                'align_method':9,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]}},\
'1759-510':{  'eventids':numpy.array([[1759,510]]),\
                'known_flight':'a04abd',\
                'align_method':9,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]}},\
'1772-71053':{  'eventids':numpy.array([[1772,71053]]),\
                'known_flight':'ab81b5',\
                'align_method':9,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]}},\
'1773-63659':{  'eventids':numpy.array([[1773,62999]]),\
                'known_flight':'a28392',\
                'align_method':9,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1774-178':{  'eventids':numpy.array([[1774,178],[1774,381],[1774,1348],[1774,1485]]),\
                'known_flight':'a1c2b3',\
                'align_method':9,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1784-7166':{   'eventids':numpy.array([[1784,7203]]),\
                'known_flight':'acf975',\
                'align_method':9,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}}\
}

all_candidates = {\
'1774-178':{  'eventids':numpy.array([[1774,381]]),\
                'known_flight':'a1c2b3',\
                'align_method':9,\
                'baselines':{'hpol':[],'vpol':[[1,2],[1,3],[2,3]]}},\
}
'''


all_candidates = {\
'1773-14413':{  'eventids':numpy.array([[1773,14413],[1773,14540],[1773,14590]]),\
                'known_flight':'aa8c39',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1773-63659':{  'eventids':numpy.array([[1773,63659],[1773,63707],[1773,63727],[1773,63752],[1773,63757]]),\
                'known_flight':'a28392',\
                'align_method':0,\
                'baselines':{'hpol':[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],'vpol':[[1,2],[1,3],[2,3]]}},\
'1774-88800':{  'eventids':numpy.array([[1774,88800],[1774,88810],[1774,88815],[1774,88895],[1774,88913],[1774,88921],[1774,88923],[1774,88925],,[1774,88993],[1774,89029],[1774,89030],[1774,89032],[1774,89034],[1774,89041],[1774,89043],[1774,89052],[1774,89172],[1774,89175],[1774,89181],[1774,89203],[1774,89204],[1774,89213]]),\
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


'''


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


    crit_freq_low_pass_MHz = None#60 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = None#5

    crit_freq_high_pass_MHz = None#60
    high_pass_filter_order = None#6

    waveform_index_range = (0,400)

    apply_phase_response = False
    hilbert = False

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.03
    sine_subtract_max_freq_GHz = 0.09
    sine_subtract_percent = 0.03

    plot_filter = False
    plot_multiple = False
    plot_averaged_waveforms = False
    get_averaged_waveforms = True #If you want those calculations done but not plotted
    plot_averaged_waveforms_aligned = False
    plot_time_delays = True

    plot_plane_tracks = True #if True then plot_time_delays is set to True

    shorten_signals = False
    shorten_thresh = 0.7
    shorten_delay = 10.0
    shorten_length = 90.0

    filter_string = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_32768-align_0-shorten_signals-1-shorten_thresh-0.70-shorten_delay-10.00-shorten_length-90.00'#'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_%i-align_%i'%(final_corr_length,default_align_method)
    

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
                            tct = TemplateCompareTool(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False,apply_phase_response=apply_phase_response, sine_subtract=sine_subtract) #if sine_subtract is True a default sine_suibtract object is added and used for initial template.

                            #First pass alignment to make templates.  
                            times, averaged_waveforms = tct.averageAlignedSignalsPerChannel(eventids, template_eventid=eventids[-1], align_method=0, plot=plot_averaged_waveforms_aligned, sine_subtract=sine_subtract)

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
                            if sine_subtract:
                                tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
                            time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids,align_method=align_method,hilbert=hilbert,plot=plot_multiple,hpol_cut=None,vpol_cut=None, colors=calibrated_trigtime,shorten_signals = shorten_signals,shorten_thresh = shorten_thresh,shorten_delay = shorten_delay,shorten_length = shorten_length, sine_subtract=sine_subtract)
                            hpol_delays = time_shifts[0:6,:].T
                            vpol_delays = time_shifts[6:12,:].T
                            hpol_corrs = corrs[0:6,:].T
                            vpol_corrs = corrs[6:12,:].T
                        else:
                            hpol_delays = numpy.vstack((file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,1)][load_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,2)][load_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,3)][load_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(1,2)][load_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(1,3)][load_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(2,3)][load_cut])).T
                            vpol_delays = numpy.vstack((file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,1)][load_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,2)][load_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,3)][load_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(1,2)][load_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(1,3)][load_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(2,3)][load_cut])).T
                            hpol_corrs = numpy.vstack((file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(0,1)][load_cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(0,2)][load_cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(0,3)][load_cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(1,2)][load_cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(1,3)][load_cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(2,3)][load_cut])).T
                            vpol_corrs = numpy.vstack((file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(0,1)][load_cut],file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(0,2)][load_cut],file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(0,3)][load_cut],file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(1,2)][load_cut],file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(1,3)][load_cut],file['time_delays'][filter_string]['vpol_max_corr_%isubtract%i'%(2,3)][load_cut])).T
                        print('\n\n')

                        pp = pprint.PrettyPrinter(width=200,indent=0) #Width isn't really working which is lame.
                        pp.pprint(known_flight)
                        pp.pprint(hpol_delays)
                        pp.pprint(vpol_delays)
                        pp.pprint(hpol_corrs)
                        pp.pprint(vpol_corrs)


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
                
                
                


                for pol in ['Hpol']:#['Hpol','Vpol']:
                    pair_cut = numpy.array([pair in candidates[candidates_key]['baselines'][pol.lower()] for pair in [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]] ]) #Checks which pairs are worth looping over.
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
                        if pair_cut[pair_index] == True:
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
                                    plt.xlim(min(x),max(x))


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

                    plt.legend()


                        
    except Exception as e:
        print('Error in plotting.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    
