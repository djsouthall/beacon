'''
When calibrating the antenna positions I am seeing two peaks on correlation histograms for 
antennas 0 and 4.  I am using this to explore an characteristic differences between signals
in each peak. 
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info
from tools.fftmath import FFTPrepper

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()


if __name__ == '__main__':
    #plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = os.environ['BEACON_DATA']
    run = 5911
    if run == 1507:
        waveform_index_range = (1500,None) #Looking at the later bit of the waveform only, 10000 will cap off.  
    elif run == 1509:
            waveform_index_range = (2000,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
    elif run == 1511:
            waveform_index_range = (1250,2000) #Looking at the later bit of the waveform only, 10000 will cap off.  

    #Time delays greater than 50
    eventids_A = numpy.array([   2473, 2477, 2481, 2487, 2491, 2493, 2499, 2503, 2505, 2507, 2509,\
                               2511, 2513, 2515, 2517, 2521, 2523, 2525, 2527, 2529, 2531, 2533,\
                               2535, 2537, 2539, 2545, 2551, 2553, 2559, 2561, 2563, 2567, 2569,\
                               2571, 2575, 2577, 2579, 2581, 2583, 2585, 2587, 2589, 2591, 2593,\
                               2595, 2597, 2599, 2601, 2607, 2609, 2619, 2625, 2629, 2631, 2633,\
                               2635, 2637, 2639, 2641, 2643, 2645, 2647, 2649, 2651, 2653, 2655,\
                               2657, 2659, 2661, 2663, 2665, 2667, 2669, 2671, 2673, 2675, 2677,\
                               2679, 2687, 2689, 2691, 2709, 2711, 2713, 2715, 2717, 2719, 2721,\
                               2723, 2725, 2727, 2729, 2731, 2735, 2737, 2743, 2755, 2757, 2765,\
                               2767, 2771, 2773, 2775, 2777, 2779, 2781, 2783, 2785, 2787, 2789,\
                               2791, 2793, 2795, 2797, 2799, 2801, 2803, 2807, 2815, 2821, 2823,\
                               2829, 2831, 2833, 2835, 2839, 2843, 2845, 2847, 2849, 2851, 2853,\
                               2855, 2857, 2861, 2863, 2865, 2867, 2869, 2871, 2873, 2875, 2887,\
                               2899, 2905, 2907, 2909, 2911, 2913, 2915, 2917, 2919, 2921, 2923,\
                               2925, 2927, 2929, 2931, 2935, 2937, 2945, 2951, 2965, 2971, 2973,\
                               2975, 2977, 2979, 2981, 2983, 2985, 2987, 2989, 2991, 2993, 2995,\
                               3037, 3041, 3043, 3045, 3049, 3051, 3053, 3055, 3057, 3059, 3061,\
                               3067, 3079, 3097, 3099, 3103, 3107, 3111, 3115, 3117, 3119, 3135,\
                               3137, 3139, 3157, 3159, 3161, 3165, 3167, 3169, 3175, 3177, 3179,\
                               3181, 3185, 3187, 3189, 3191, 3193, 3195, 3203, 3213, 3215, 3221,\
                               3223, 3227, 3231, 3233, 3235, 3239, 3241, 3243, 3245, 3247, 3249,\
                               3251, 3253, 3255, 3257, 3259, 3261, 3263, 3271, 3291, 3295, 3299,\
                               3301, 3303, 3305, 3307, 3309, 3311, 3317, 3319, 3323, 3325, 3327,\
                               3329, 3333, 3345, 3355, 3359, 3363, 3365, 3367, 3371, 3373, 3375,\
                               3377, 3379, 3381, 3383, 3385, 3387, 3391, 3395, 3409])
    #Time delays less than 50
    eventids_B = numpy.array([   2475, 2479, 2483, 2485, 2489, 2495, 2497, 2501, 2519, 2541, 2543,\
                               2547, 2549, 2555, 2557, 2565, 2573, 2603, 2605, 2611, 2613, 2615,\
                               2617, 2621, 2623, 2627, 2681, 2683, 2685, 2693, 2695, 2697, 2699,\
                               2701, 2703, 2705, 2707, 2733, 2739, 2741, 2745, 2747, 2749, 2751,\
                               2753, 2759, 2761, 2763, 2769, 2805, 2809, 2811, 2813, 2817, 2819,\
                               2825, 2827, 2837, 2841, 2859, 2877, 2879, 2881, 2883, 2885, 2889,\
                               2891, 2893, 2895, 2897, 2901, 2903, 2933, 2939, 2941, 2943, 2947,\
                               2949, 2953, 2955, 2957, 2959, 2961, 2963, 2967, 2969, 2997, 2999,\
                               3001, 3003, 3005, 3007, 3009, 3011, 3013, 3015, 3017, 3019, 3021,\
                               3023, 3025, 3027, 3029, 3031, 3033, 3035, 3039, 3047, 3063, 3065,\
                               3069, 3071, 3073, 3075, 3077, 3081, 3083, 3085, 3087, 3089, 3091,\
                               3093, 3095, 3101, 3105, 3109, 3113, 3121, 3123, 3125, 3127, 3129,\
                               3131, 3133, 3141, 3143, 3145, 3147, 3149, 3151, 3153, 3155, 3163,\
                               3171, 3173, 3183, 3197, 3199, 3201, 3205, 3207, 3209, 3211, 3217,\
                               3219, 3225, 3229, 3237, 3265, 3267, 3269, 3273, 3275, 3277, 3279,\
                               3281, 3283, 3285, 3287, 3289, 3293, 3297, 3313, 3315, 3321, 3331,\
                               3335, 3337, 3339, 3341, 3343, 3347, 3349, 3351, 3353, 3357, 3361,\
                               3369, 3389, 3393, 3397, 3399, 3401, 3403, 3405, 3407, 3411])

    all_eventids = numpy.sort(numpy.append(eventids_A,eventids_B))
    cut_A = numpy.isin(all_eventids,eventids_A)
    cut_B = numpy.isin(all_eventids,eventids_B)

    print('Run %i'%run)
    final_corr_length = 2**18 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = 80 #This new pulser seems to peak in the region of 85 MHz or so
    crit_freq_high_pass_MHz = 65
    low_pass_filter_order = 3
    high_pass_filter_order = 6
    plot_filters = True

    reader = Reader(datapath,run)
    prep = FFTPrepper(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=plot_filters)

    wfs = {}
    rms = {}
    argmax = {}
    channels = [0,2]
    for channel in channels:
        wfs[channel] = numpy.zeros((len(all_eventids),prep.buffer_length))
        rms[channel] = numpy.zeros((len(all_eventids)))
        argmax[channel] = numpy.zeros((len(all_eventids)))

    t = prep.t()
    for event_index, eventid in enumerate(all_eventids):
        sys.stdout.write('(%i/%i)\r'%(event_index,len(all_eventids)))
        sys.stdout.flush()
        prep.setEntry(eventid)
        event_times = prep.t()
        for channel in channels:
            channel=int(channel)
            wfs[channel][event_index] = prep.wf(channel)
            rms[channel][event_index] = numpy.std(prep.wf(channel))
            argmax[channel][event_index] = numpy.argmax(prep.wf(channel))
            if False:
                fig = plt.figure()
                plt.plot(t,prep.wf(channel))
                plt.xlim(5000,5800)
                import pdb;pdb.set_trace()
                plt.close(fig)

    alpha = 0.2
    split_plots = False
    for channel in channels:
        plt.figure()
        plt.suptitle(str(channel))
        if split_plots == True:

            for index, wf in enumerate(wfs[channel]):
                if cut_A[index]:
                    plt.subplot(3,1,1)
                    plt.plot(t,wf,c='r',alpha=alpha)
                elif cut_B[index]:
                    plt.subplot(3,1,2)
                    plt.plot(t,wf,c='b',alpha=alpha)
            
            plt.subplot(3,1,1)
            plt.ylim((-45,45))
            plt.xlim((5000,5800))
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            plt.subplot(3,1,2)
            plt.ylim((-45,45))
            plt.xlim((5000,5800))
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            plt.subplot(3,1,3)
        else:
            plt.subplot(2,1,1)
            for index, wf in enumerate(wfs[channel]):
                if cut_A[index]:
                    plt.plot(t,wf,c='r',alpha=alpha)
                elif cut_B[index]:
                    plt.plot(t,wf,c='b',alpha=alpha)
            plt.ylim((-45,45))
            plt.xlim((5000,5800))
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            plt.subplot(2,1,2)
        bins = numpy.linspace(0,5,100)
        plt.hist(rms[channel][cut_A],bins=bins,color='r',alpha=0.5,label='td > 50 ns')
        plt.hist(rms[channel][cut_B],bins=bins,color='b',alpha=0.5,label='td < 50 ns')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.xlabel('rms')
        plt.ylabel('counts')
        plt.legend()


        plt.figure()
        plt.subplot(2,1,1)
        plt.scatter(eventids_A,argmax[channel][cut_A],color='r',alpha=0.5,label='td > 50 ns')
        plt.scatter(eventids_B,argmax[channel][cut_B],color='b',alpha=0.5,label='td < 50 ns')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.xlabel('eventid')
        plt.ylabel('argmax')
        plt.legend()

        plt.subplot(2,1,2)
        plt.scatter(eventids_A,rms[channel][cut_A],color='r',alpha=0.5,label='td > 50 ns')
        plt.scatter(eventids_B,rms[channel][cut_B],color='b',alpha=0.5,label='td < 50 ns')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.xlabel('eventid')
        plt.ylabel('rms')
        plt.legend()
