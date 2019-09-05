'''
This script uses the events found by find_good_saturating_signals.py to determine some good
expected time delays between antennas.  These can be used then as educated guesses for time
differences in the antenna_timings.py script. 
'''

import numpy
import scipy.spatial
import scipy.signal
import os
import sys
import csv

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()


#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.
known_pulser_ids = {
    'run792':{
        'eventids':numpy.array([115156, 115228, 115256, 115276, 115283, 115315, 115330, 115371,\
                                115447, 115612, 115872, 116230, 116262, 116462, 116473, 116479,\
                                116486, 116511, 116524, 116603, 116619, 116624, 116633, 116760,\
                                116790, 116816, 117026, 117050, 117175, 117195, 117237, 117247,\
                                117258, 117315, 117378, 117540, 117837, 117858, 117874, 117933,\
                                117949, 118116, 118139, 118167, 118208, 118219, 118227, 118241,\
                                118256, 118267, 118295, 118364, 118423, 118461, 118497, 118518,\
                                118644, 118662, 118676, 118685, 118719, 118752, 118856, 118872,\
                                118889, 118908, 118930, 118946, 118994, 119038, 119053, 119064,\
                                119070, 119094, 119150, 119161, 119177, 119208, 119223, 119304,\
                                119315, 119339, 119346, 119371, 119390, 119401, 119408, 119414,\
                                119431, 119434, 119458, 119472, 119478, 119508, 119517, 119555,\
                                119578, 119598, 119629, 119636, 119648, 119660, 119671, 119844,\
                                120009, 120107, 120115, 120202, 120225, 120241, 120249, 120263,\
                                120276, 120281, 120292, 120374, 120587, 120607, 120613, 120628,\
                                120632, 120905, 120910, 120916, 120925, 120941, 121019, 121081,\
                                121170, 121318, 121382, 121460, 121489, 121510, 121725, 121736,\
                                121741, 121751, 121765, 121769, 121803, 121876, 121981, 122001,\
                                122014, 122021, 122053, 122073, 122093, 122166, 122293, 122311,\
                                122403, 122455, 122508, 122551, 122560, 122579, 122723, 122761,\
                                122797]),
        'clock_rate':31249812.04283368
            },
    'run793':{
        'eventids':numpy.array([    96607,  96632,  96657,  96684,  96762,  96820,  96875,  96962,\
                                    97532,  97550,  97583,  97623,  97636,  97661,  97681,  97698,\
                                    97720,  97739,  97761,  97782,  97803,  97824,  97846,  97876,\
                                    97932,  97954,  97979,  98006,  98030,  98050,  98075,  98125,\
                                    98148,  98163,  98190,  98207,  98277,  98431,  98450,  98472,\
                                    98507,  98545,  98561,  98577,  98587,  98588,  98631,  98657,\
                                    98674,  98687,  98707,  98731,  98799,  98815,  99040,  99086,\
                                    99110,  99158,  99208,  99227,  99245,  99264,  99288,  99309,\
                                    99340,  99353,  99375,  99398,  99423,  99440,  99454,  99477,\
                                    99493,  99513,  99530,  99548,  99911,  99942,  99951,  99985,\
                                   100002, 100019, 100035, 100055, 100073, 100096, 100114, 100153,\
                                   100189, 100294, 100424, 100442, 100531, 100591, 100748, 100767,\
                                   100899, 100979, 101000, 101011, 101025, 101129, 101146, 101161,\
                                   101177, 101191, 101212, 101227, 101261, 101281, 101297, 101311,\
                                   101328, 101363, 101378, 101457, 101470, 101485, 101500, 101527,\
                                   101540, 101556, 101578, 101616, 101640, 101667, 101736, 101760,\
                                   101819, 102100, 102116, 102136, 102159, 102178, 102194, 102215,\
                                   102239, 102255, 102274, 102309, 102326, 102364, 102382, 102398,\
                                   102417, 102443, 102464, 102484, 102516, 102529, 102551, 102562,\
                                   102574, 102587, 102606, 102625, 102648, 102667, 102693, 102713,\
                                   102733, 102758, 102775, 102796, 102811, 102830, 102847, 102870,\
                                   102883, 102904, 102924, 102944, 102965, 102982, 102997, 103017,\
                                   103035, 103054, 103075, 103097, 103116, 103135, 103156, 103176,\
                                   103195, 103214, 103235, 103249, 103264, 103283, 103301, 103323,\
                                   103340, 103390, 103407, 103419, 103438, 103456, 103468, 103479,\
                                   103497, 103512, 103528, 103540, 103555, 103578, 103593, 103617,\
                                   103627, 103646, 103665, 103679, 103697, 103715, 103731, 103747,\
                                   103761, 103774, 103800, 103818, 103842, 103880, 103895, 103921,\
                                   103965, 103977, 103995, 104008, 104025, 104055, 104073, 104118,\
                                   104142, 104152, 104174, 104191, 104204, 104220, 104255, 104279,\
                                   104340, 104398, 104430, 104487, 104515, 104545, 104572, 104606,\
                                   104632, 104656, 104721, 104745, 104779, 104812, 104836, 105082,\
                                   105119, 105147, 105191, 105226, 105304, 105329, 105352, 105407,\
                                   105429, 105454, 105477, 105510, 105530, 105560, 105586, 105620,\
                                   105641, 105667, 105695, 105723, 105749, 105779, 105804, 105832,\
                                   105881, 105897, 105967, 105999, 106017, 106043, 106063, 106093,\
                                   106152, 106227, 106397, 106421, 106461, 106476, 106516, 106538,\
                                   106559, 106581, 106622, 106680, 106730, 106754, 106765, 106786,\
                                   106813, 106845, 106869, 106891, 106916, 106942, 106966, 107022,\
                                   107052, 107070, 107088, 107114, 107126, 107153, 107203, 107221,\
                                   107249, 107275, 107302, 107325, 107341, 107356, 107382, 107407,\
                                   107433, 107461, 107489, 107499, 107522, 107546, 107571, 107596,\
                                   107620, 107646, 107672, 107692, 107718, 107744, 107764, 107790,\
                                   107814, 107835, 107856, 107881, 107911, 107940, 108115, 108131,\
                                   108162, 108184, 108209, 108233, 108275, 108294, 108319, 108373,\
                                   108827, 108878, 108926, 108969, 108984, 109012, 109054, 109087,\
                                   109106, 109121, 109139, 109161, 109185, 109212, 109261, 110029,\
                                   110074, 110100, 110126, 110142, 110163, 110181, 110203, 110221,\
                                   110235, 110258, 110274, 110429, 110442, 110471, 110534, 110580,\
                                   110599, 110624, 110643, 110661, 110684, 110713, 110741, 110777,\
                                   110795, 110858, 110884, 110900, 110917, 110970, 110993, 111005,\
                                   111035, 111056, 111083, 111098, 111126, 111145, 111183, 111197,\
                                   111238, 111274, 111293, 111311, 111331, 111368, 111389, 111415,\
                                   111440, 111456, 111481, 111504, 111522, 111542, 111584, 111600,\
                                   111640, 111702, 111714, 111729, 111750, 111796, 111823, 111841,\
                                   111855, 111873, 111885, 111902, 111919, 111941, 111956, 111980,\
                                   111991, 112010, 112025, 112035, 112051, 112068, 112080, 112092,\
                                   112115, 112140, 112160, 112177, 112196, 112213, 112258, 112294,\
                                   112315, 112610, 112626, 112656, 112675, 112701, 112713, 112730,\
                                   112749, 112765, 112812, 112844, 112864, 112887, 112907, 112934,\
                                   112952, 112972, 113038, 113062, 113156, 113178, 113194, 113235,\
                                   113259, 113275, 113295, 113312, 113333, 113357, 113375, 113392,\
                                   113414, 113476, 113496, 113519, 113889, 113930, 113957, 114004,\
                                   114048, 114069, 114084, 114127, 114147, 114173, 114196, 114226,\
                                   114266, 114295, 114313, 114331, 114356, 114374, 114399, 114428,\
                                   114457, 114500, 114525, 114569, 114589, 114633, 114655, 114677,\
                                   114703, 114719, 114738, 114755, 114777, 114789, 114801, 114852,\
                                   114879, 114900, 114942, 114960, 114996, 115019, 115055, 115095,\
                                   115115, 115130, 115197, 115217, 115236, 115275, 115283, 115303,\
                                   115321, 115337, 115377, 115413, 115442, 115465, 115491, 115535,\
                                   115554, 115570, 115584, 115612, 115630, 115644, 115662, 115675,\
                                   115689, 115708, 115721, 115735, 115759, 115787, 115806, 115823,\
                                   115844, 115870, 115888, 115912, 115935, 115963, 115976, 115996,\
                                   116019, 116044, 116065, 116082, 116101, 116115, 116155, 116173,\
                                   116184]),
        'clock_rate':31249809.22371152,
        'ignore_eventids':numpy.array([ 96607,  96657,  96820,  96875,  98125,  98588,  99208, 100531,\
                                       101328, 101470, 101616, 101640, 101667, 102159, 102326, 102625,\
                                       103235, 103646, 103842, 103895, 103977, 104118, 104545, 105226,\
                                       105695, 105999, 106227, 106476, 106622, 106754, 106786, 106813,\
                                       106845, 107022, 107814, 108162, 110074, 110534, 110858, 111098,\
                                       111197, 111311, 111542, 111902, 111941, 112675, 112713, 112864,\
                                       112887, 113062, 113194, 113392, 113476, 113957, 114069, 114084,\
                                       114295, 114719, 114738, 114755, 114942, 115055, 115413, 115442,\
                                       115465, 115491, 115612, 116065])
            }
}

def rfftWrapper(waveform_times, *args, **kwargs):
    spec = numpy.fft.rfft(*args, **kwargs)
    real_power_multiplier = 2.0*numpy.ones_like(spec) #The factor of 2 because rfft lost half of the power except for dc and Nyquist bins (handled below).
    if len(numpy.shape(spec)) != 1:
        real_power_multiplier[:,[0,-1]] = 1.0
    else:
        real_power_multiplier[[0,-1]] = 1.0
    spec_dbish = 10.0*numpy.log10( real_power_multiplier*spec * numpy.conj(spec) / len(waveform_times)) #10 because doing power in log.  Dividing by N to match monutau. 
    freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
    return freqs, spec_dbish

def makeFilter(waveform_times,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order, plot_filter=False):
    dt = waveform_times[1] - waveform_times[0]
    freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
    b, a = scipy.signal.butter(filter_order, crit_freq_low_pass_MHz*1e6, 'low', analog=True)
    d, c = scipy.signal.butter(filter_order, crit_freq_high_pass_MHz*1e6, 'high', analog=True)

    filter_x_low_pass, filter_y_low_pass = scipy.signal.freqs(b, a,worN=freqs)
    filter_x_high_pass, filter_y_high_pass = scipy.signal.freqs(d, c,worN=freqs)
    filter_x = freqs
    filter_y = numpy.multiply(filter_y_low_pass,filter_y_high_pass)
    if plot_filter == True:
        plt.figure()
        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y)),color='k',label='final filter')
        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_low_pass)),color='r',linestyle='--',label='low pass')
        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_high_pass)),color='orange',linestyle='--',label='high pass')
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(crit_freq_low_pass_MHz, color='magenta',label='LP Crit') # cutoff frequency
        plt.axvline(crit_freq_high_pass_MHz, color='cyan',label='HP Crit') # cutoff frequency
        plt.xlim(0,200)
        plt.ylim(-50,10)
        plt.legend()
    return filter_y, freqs

def alignToTemplate(eventids,_upsampled_waveforms,_waveforms_corr, _template, _final_corr_length, _waveform_times_corr, _filter_y_corr=None, align_method=1, plot_wf=False, template_pre_filtered=False):
    '''
    _waveforms_corr should be a 2d array where each row is a waveform of a different event, already zero padded for cross correlation
    (i.e. must be over half zeros on the right side).

    _upsampled_waveforms are the waveforms that have been upsampled (BUT WERE NOT ORIGINALLY PADDED BY A FACTOR OF 2).  These are
    the waveforms that will be aligned based on the time delays from correlations performed with _waveforms_corr.
        
    If a filter is given then it will be applied to all signals.  It is assumed the upsampled signals are already filtered.

    This given _template must be in the same form upsampled nature as _waveforms_corr. 
    '''
    try:
        #Prepare template correlation storage
        max_corrs = numpy.zeros(len(eventids))
        index_delays = numpy.zeros(len(eventids),dtype=int) #Within the correlations for max corr
        corr_index_to_delay_index = -numpy.arange(-(_final_corr_length-1)//2,(_final_corr_length-1)//2 + 1) #Negative because with how it is programmed you would want to roll the template the normal amount, but I will be rolling the waveforms.
        rolled_wf = numpy.zeros_like(_upsampled_waveforms)

        if numpy.logical_and(_filter_y_corr is not None,template_pre_filtered == False):
            template_fft = numpy.multiply(numpy.fft.rfft(_template),_filter_y_corr)
            #template_fft = numpy.multiply(numpy.fft.rfft(_waveforms_corr[_template_event_index]),_filter_y_corr)
        else:
            template_fft = numpy.fft.rfft(_template)
            #template_fft = numpy.fft.rfft(_waveforms_corr[_template_event_index])
            
        scaled_conj_template_fft = numpy.conj(template_fft)/numpy.std(numpy.conj(template_fft)) 

        for event_index, eventid in enumerate(eventids):
            sys.stdout.write('(%i/%i)\r'%(event_index,len(eventids)))
            sys.stdout.flush()
            reader.setEntry(eventid)

            if _filter_y_corr is not None:
                fft = numpy.multiply(numpy.fft.rfft(_waveforms_corr[event_index]),_filter_y_corr)
            else:
                fft = numpy.fft.rfft(_waveforms_corr[event_index])

            corr_fft = numpy.multiply((fft/numpy.std(fft)),(scaled_conj_template_fft)) / (len(_waveform_times_corr)//2 + 1)
            corr = numpy.fft.fftshift(numpy.fft.irfft(corr_fft,n=_final_corr_length)) * (_final_corr_length//2 + 1) #Upsampling and keeping scale
            

            if align_method == 1:
                #Looks for best alignment within window after cfd trigger, cfd applied on hilber envelope.
                corr_hilbert = numpy.abs(scipy.signal.hilbert(corr))
                cfd_indices = numpy.where(corr_hilbert/numpy.max(corr_hilbert) > 0.5)[0]
                cfd_indices = cfd_indices[0:int(0.50*len(cfd_indices))] #Top 50% close past 50% max rise
                index_delays[event_index] = cfd_indices[numpy.argmax(corr[cfd_indices])]

                max_corrs[event_index] = corr[index_delays[event_index]]
                rolled_wf[event_index] = numpy.roll(_upsampled_waveforms[event_index],corr_index_to_delay_index[index_delays[event_index]])

            elif align_method == 2:
                #Of the peaks in the correlation plot within range of the max peak, choose the peak with the minimum index.
                index_delays[event_index] = min(scipy.signal.find_peaks(corr,height = 0.8*max(corr),distance=int(0.05*len(corr)))[0])

                max_corrs[event_index] = corr[index_delays[event_index]]
                rolled_wf[event_index] = numpy.roll(_upsampled_waveforms[event_index],corr_index_to_delay_index[index_delays[event_index]])

            elif align_method == 3:
                #Align to maximum correlation value.
                index_delays[event_index] = numpy.argmax(corr) #These are the indices within corr that are max.
                max_corrs[event_index] = corr[index_delays[event_index]]
                rolled_wf[event_index] = numpy.roll(_upsampled_waveforms[event_index],corr_index_to_delay_index[index_delays[event_index]])
        

        if False:
            #Use weighted averages for the correlations in the template
            upsampled_template_out = numpy.average(rolled_wf,axis=0,weights = max_corrs)
            upsampled_template_better = numpy.average(rolled_wf[max_corrs > 0.7],axis=0,weights = max_corrs[max_corrs > 0.7])
            upsampled_template_worse = numpy.average(rolled_wf[max_corrs <= 0.7],axis=0,weights = max_corrs[max_corrs <= 0.7])
        else:
            #DON'T Use weighted averages for the correlations in the template.
            upsampled_template_out = numpy.average(rolled_wf,axis=0)
            upsampled_template_better = numpy.average(rolled_wf[max_corrs > 0.7],axis=0)
            upsampled_template_worse = numpy.average(rolled_wf[max_corrs <= 0.7],axis=0)

        #import pdb;pdb.set_trace()
        #The template at this stage is in the upsampled waveforms which did not have the factor of 2 zeros added.  To make it an exceptable form
        #To be ran back into this function, it must be downsampled to the length before the factor of 2 was added.  Then add a factor of 2 of zeros.
        downsampled_template_out = numpy.zeros(numpy.shape(_waveforms_corr)[1])
        downsampled_template_out[0:numpy.shape(_waveforms_corr)[1]//2] = numpy.fft.irfft(numpy.fft.rfft(upsampled_template_out), n = numpy.shape(_waveforms_corr)[1]//2) * ((numpy.shape(_waveforms_corr)[1]//2)/numpy.shape(_upsampled_waveforms)[1])  #This can then be fed back into this function

        downsampled_template_worse = numpy.zeros(numpy.shape(_waveforms_corr)[1])
        downsampled_template_worse[0:numpy.shape(_waveforms_corr)[1]//2] = numpy.fft.irfft(numpy.fft.rfft(upsampled_template_worse), n = numpy.shape(_waveforms_corr)[1]//2) * ((numpy.shape(_waveforms_corr)[1]//2)/numpy.shape(_upsampled_waveforms)[1])  #This can then be fed back into this function

        downsampled_template_better = numpy.zeros(numpy.shape(_waveforms_corr)[1])
        downsampled_template_better[0:numpy.shape(_waveforms_corr)[1]//2] = numpy.fft.irfft(numpy.fft.rfft(upsampled_template_better), n = numpy.shape(_waveforms_corr)[1]//2) * ((numpy.shape(_waveforms_corr)[1]//2)/numpy.shape(_upsampled_waveforms)[1])  #This can then be fed back into this function


        if plot_wf:
            plt.figure()
            plt.title('Aligned Waveforms')
            for event_index, eventid in enumerate(eventids):
                plt.plot(numpy.linspace(0,1,len(rolled_wf[event_index])),rolled_wf[event_index],alpha=0.5)
            plt.plot(numpy.linspace(0,1,len(downsampled_template_out[0:len(downsampled_template_out)//2])),downsampled_template_out[0:len(downsampled_template_out)//2],color='r',linestyle='--')
            plt.ylabel('Adu',fontsize=16)
            plt.xlabel('Time (arb)',fontsize=16)

            plt.figure()
            for event_index, eventid in enumerate(eventids):
                if max_corrs[event_index] < 0.70:
                    plt.subplot(2,1,1)
                    plt.plot(numpy.linspace(0,1,len(rolled_wf[event_index])),rolled_wf[event_index],alpha=0.5)
                else:
                    plt.subplot(2,1,2)
                    plt.plot(numpy.linspace(0,1,len(rolled_wf[event_index])),rolled_wf[event_index],alpha=0.5)
            plt.subplot(2,1,1)
            #plt.plot(numpy.linspace(0,1,len(downsampled_template_out[0:len(downsampled_template_out)//2])),downsampled_template_out[0:len(downsampled_template_out)//2],color='r',linestyle='--')
            plt.plot(numpy.linspace(0,1,len(downsampled_template_worse[0:len(downsampled_template_worse)//2])),downsampled_template_worse[0:len(downsampled_template_worse)//2],color='r',linestyle='--')
            plt.ylabel('Adu',fontsize=16)
            plt.xlabel('Time (arb)',fontsize=16)

            plt.subplot(2,1,2)
            #plt.plot(numpy.linspace(0,1,len(downsampled_template_out[0:len(downsampled_template_out)//2])),downsampled_template_out[0:len(downsampled_template_out)//2],color='r',linestyle='--')
            plt.plot(numpy.linspace(0,1,len(downsampled_template_better[0:len(downsampled_template_better)//2])),downsampled_template_better[0:len(downsampled_template_better)//2],color='r',linestyle='--')
            plt.ylabel('Adu',fontsize=16)
            plt.xlabel('Time (arb)',fontsize=16)



            #plt.legend(fontsize=16)
        return index_delays, max_corrs, downsampled_template_out
    except Exception as e:
        print('Error in alignToTemplate')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    plt.close('all')

    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine

    #Filter settings
    final_corr_length = 2**16 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = 80
    crit_freq_high_pass_MHz = 20
    filter_order = 6
    use_filter = False
    align_method = 2
    #1. Looks for best alignment within window after cfd trigger, cfd applied on hilbert envelope.
    #2. Of the peaks in the correlation plot within range of the max peak, choose the peak with the minimum index.
    #3. Align to maximum correlation value.


    #Plotting info
    plot = True

    #Initial Template Selection Parameters
    manual_template = False #If true, templates waveforms will be printed out until the user chooses one as a started template for that channel.
    initial_std_precentage_window = 0.2 #The inital percentage of the window that the std will be calculated for.  Used for selecting interesting intial templates (ones that that start in the middle of the waveform and have low std)
    intial_std_threshold = 10.0 #The std in the first % defined above must be below this value for the event to be considered as a template.  
    
    #Template loop parameters
    iterate_limit = 1

    #Output
    save_templates = False

    #General Prep
    channels = numpy.arange(8,dtype=int)
    
    #Main loop
    for run_index, run in enumerate(runs):
        if 'run%i'%run in list(known_pulser_ids.keys()):
            try:
                if 'ignore_eventids' in list(known_pulser_ids['run%i'%run].keys()):
                    eventids = numpy.sort(known_pulser_ids['run%i'%run]['eventids'][~numpy.isin(known_pulser_ids['run%i'%run]['eventids'],known_pulser_ids['run%i'%run]['ignore_eventids'])])
                else:
                    eventids = numpy.sort(known_pulser_ids['run%i'%run]['eventids'])

                reader = Reader(datapath,run)
                reader.setEntry(eventids[0])
                
                waveform_times = reader.t()
                dt = waveform_times[1]-waveform_times[0]
                waveform_times_padded_to_power2 = numpy.arange(2**(numpy.ceil(numpy.log2(len(waveform_times)))))*dt #Rounding up to a factor of 2 of the len of the waveforms  USED FOR WAVEFORMS
                waveform_times_corr = numpy.arange(2*len(waveform_times_padded_to_power2))*dt #multiplying by 2 for cross correlation later. USED FOR CORRELATIONS
                
                if use_filter:
                    filter_y_corr,freqs_corr = makeFilter(waveform_times_corr,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order,plot_filter=True)
                    filter_y_wf,freqs_wf = makeFilter(waveform_times_padded_to_power2,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order,plot_filter=False)
                else:
                    freqs_corr = numpy.fft.rfftfreq(len(waveform_times_corr), d=(waveform_times_corr[1] - waveform_times_corr[0])/1.0e9)
                    freqs_wf = numpy.fft.rfftfreq(len(waveform_times_padded_to_power2), d=(waveform_times_padded_to_power2[1] - waveform_times_padded_to_power2[0])/1.0e9)

                df_corr = freqs_corr[1] - freqs_corr[0] #Note that this is the df for the padded correlation ffts and would not be the same as the one for the normal waveform ffts which have not been doubled in length. 
                final_dt_corr = 1e9/(2*(final_corr_length//2 + 1)*df_corr) #ns #This is the time step resulting from the cross correlation.  

                time_shifts_corr = numpy.arange(-(final_corr_length-1)//2,(final_corr_length-1)//2 + 1)*final_dt_corr #This results in the maxiumum of an autocorrelation being located at a time shift of 0.0

                #Load in waveforms:
                print('Loading Waveforms for Template:\n')
                exclude_eventids = []
                waveforms_corr = {}
                upsampled_waveforms = {}
                for channel in channels:
                    channel=int(channel)
                    waveforms_corr['ch%i'%channel] = numpy.zeros((len(eventids),len(waveform_times_corr)))
                    upsampled_waveforms['ch%i'%channel] = numpy.zeros((len(eventids),final_corr_length//2))

                for event_index, eventid in enumerate(eventids):
                    sys.stdout.write('(%i/%i)\r'%(event_index,len(eventids)))
                    sys.stdout.flush()
                    reader.setEntry(eventid)
                    event_times = reader.t()
                    for channel in channels:
                        channel=int(channel)
                        waveforms_corr['ch%i'%channel][event_index][0:reader.header().buffer_length] = reader.wf(channel)
                        #Below are the actual time domain waveforms_corr and should not have the factor of 2 padding.  The small rounding padding sticks around, so using waveform_times_padded_to_power2 times,
                        if use_filter:
                            upsampled_waveforms['ch%i'%channel][event_index] = numpy.fft.irfft(numpy.multiply(filter_y_wf,numpy.fft.rfft(waveforms_corr['ch%i'%channel][event_index][0:len(waveform_times_padded_to_power2)])),n=final_corr_length//2) * ((final_corr_length//2)/len(waveform_times_padded_to_power2))
                        else:
                            upsampled_waveforms['ch%i'%channel][event_index] = numpy.fft.irfft(numpy.fft.rfft(waveforms_corr['ch%i'%channel][event_index][0:len(waveform_times_padded_to_power2)]),n=final_corr_length//2) * ((final_corr_length//2)/len(waveform_times_padded_to_power2))
                        #upsampled_waveforms['ch%i'%channel][event_index], upsampled_times = scipy.signal.resample(waveforms_corr['ch%i'%channel][event_index][0:len(waveform_times_padded_to_power2)],final_corr_length//2,t=waveform_times_padded_to_power2)
                    
                '''
                print('Upsampling waveforms_corr')
                for channel in channels:
                    print(channel)
                    channel=int(channel)
                    upsampled_waveforms['ch%i'%channel], upsampled_times = scipy.signal.resample(waveforms_corr['ch%i'%channel],2*(final_corr_length//2 + 1),t=waveform_times_corr,axis = 1)
                '''
                print('\n')

                print('Making Templates')
                
                max_corrs = {}
                index_delays = {}
                templates = {}
                for channel in channels:
                    channel=int(channel)
                    sys.stdout.write('(%i/%i)\n'%(channel,len(channels)))
                    sys.stdout.flush()

                    #Find template to correlate to:
                    ratios = []
                    early_std = []
                    total_std = []
                    for index, waveform in enumerate(waveforms_corr['ch%i'%channel]): 
                        ratios.append(numpy.max(numpy.abs(scipy.signal.hilbert(waveform[0:len(waveform_times)])))/numpy.std(waveform[0:len(waveform_times)]))
                        early_std.append(numpy.std(waveform[0:len(waveform_times)][0:int(initial_std_precentage_window*len(waveform[0:len(waveform_times)]))]))
                        total_std.append(numpy.std(waveform[0:len(waveform_times)]))
                    early_std = numpy.array(early_std)
                    total_std = numpy.array(total_std)

                    exclude_eventids.append(eventids[numpy.logical_or(total_std < 20.0, total_std > 30.0)])

                    sorted_template_indices = numpy.argsort(ratios)[::-1][(early_std < intial_std_threshold)[numpy.argsort(ratios)[::-1]]] #Indices sorted from large to small in ratio, then cut on those templates with small enough initial std

                    if manual_template == True:
                        template_selected = False
                        template_index = 0
                        while template_selected == False:
                            fig = plt.figure()
                            plt.plot(waveform_times,waveforms_corr['ch%i'%channel][sorted_template_indices[template_index]][0:len(waveform_times)])
                            plt.show()
                            
                            acceptable_response = False
                            while acceptable_response == False:
                                response = input('Is this a good waveform to start?  (y/n)')
                                if response == 'y':
                                    acceptable_response = True
                                    template_selected = True
                                    plt.close(fig)
                                elif response == 'n':
                                    acceptable_response = True
                                    template_selected = False
                                    template_index += 1
                                    plt.close(fig)
                                    if template_index >= len(waveforms_corr['ch%i'%channel]):
                                        print('All possible starting templates cycled through.  Defaulting to first option.')
                                        template_selected = True
                                        template_index = 0
                                else:
                                    print('Response not accepted.  Please type either y or n')
                                    acceptable_response = False
                    else:
                        template_selected = True
                        template_index = 0

                    template_event_index = sorted_template_indices[template_index]
                    template_eventid = eventids[template_event_index]
                    template_waveform = waveforms_corr['ch%i'%channel][template_event_index]

                    #At this point there is an initial template, which is zero padded 2 some factor of two but is not significantly upsampled.  
                    #For less computation perhaps the template should be the conjugated one in the correlation?  Maybe not worth the confusion of shifting
                    #waveforms_corr before averaging.

                    if use_filter:
                        index_delays['ch%i'%channel], max_corrs['ch%i'%channel], downsampled_template_out = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],waveforms_corr['ch%i'%channel][template_event_index],final_corr_length,waveform_times_corr,_filter_y_corr = filter_y_corr,plot_wf = False)
                    else:
                        index_delays['ch%i'%channel], max_corrs['ch%i'%channel], downsampled_template_out = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],waveforms_corr['ch%i'%channel][template_event_index],final_corr_length,waveform_times_corr,_filter_y_corr = None,plot_wf = False)

                    #The above used the first pass at a template to align.  Now I use the produced templates iteratively, aligning signals to them to make a new template, repeat.

                    print('Using initial template for alignment of new template:')
                    template_count = 0
                    while template_count <= iterate_limit:
                        #I go one over iterate limit.  The last time it is just getting the correlation times of the events with the final template, and not using the output as a new template.
                        if template_count < iterate_limit:
                            sys.stdout.write('\t\t(%i/%i)\r'%(template_count+1,iterate_limit))
                            sys.stdout.flush()
                            if use_filter:
                                downsampled_template_out = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],downsampled_template_out,final_corr_length,waveform_times_corr,template_pre_filtered=True, _filter_y_corr = filter_y_corr, align_method=align_method,plot_wf = False)[2]
                            else:
                                downsampled_template_out = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],downsampled_template_out,final_corr_length,waveform_times_corr,template_pre_filtered=True, _filter_y_corr = None, align_method=align_method,plot_wf = False)[2]
                        else:
                            if use_filter:
                                index_delays['ch%i'%channel], max_corrs['ch%i'%channel], temp = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],downsampled_template_out,final_corr_length,waveform_times_corr,template_pre_filtered=True, _filter_y_corr = filter_y_corr,align_method=align_method,plot_wf = True)
                            else:
                                index_delays['ch%i'%channel], max_corrs['ch%i'%channel], temp = alignToTemplate(eventids,upsampled_waveforms['ch%i'%channel],waveforms_corr['ch%i'%channel],downsampled_template_out,final_corr_length,waveform_times_corr,template_pre_filtered=True, _filter_y_corr = None,align_method=align_method,plot_wf = True)
                        template_count += 1

                    templates['ch%i'%channel] = downsampled_template_out[0:len(waveform_times)] #cutting off the additional zero padding

                #Plotting

                if False:
                    plt.figure()
                    plt.suptitle('Averaged Templates for Run %i'%(run),fontsize=20)
                    for channel in channels:
                        channel=int(channel)
                        if channel == 0:
                            ax = plt.subplot(4,2,channel+1)
                        else:
                            plt.subplot(4,2,channel+1,sharex=ax)
                        plt.plot(waveform_times,templates['ch%i'%channel],label='ch%i'%channel)
                        plt.ylabel('Adu',fontsize=16)
                        plt.xlabel('Time (ns)',fontsize=16)
                        plt.legend(fontsize=16)

                if False:
                    plt.figure()
                    plt.suptitle('Averaged Templates for Run %i'%(run),fontsize=20)
                    plt.subplot(2,1,1)
                    for channel in channels:
                        channel=int(channel)
                        if channel == 0:
                            ax = plt.subplot(2,1,1)
                        elif channel %2 == 0:
                            plt.subplot(2,1,1)
                        elif channel %2 == 1:
                            plt.subplot(2,1,2)

                        template_freqs, template_fft_dbish = rfftWrapper(waveform_times,templates['ch%i'%channel])
                        plt.plot(template_freqs/1e6,template_fft_dbish,label='ch%i'%channel)
                    
                    plt.subplot(2,1,1)
                    plt.ylabel('dBish',fontsize=16)
                    plt.xlabel('Freq (MHz)',fontsize=16)
                    plt.legend(fontsize=16)
                    plt.xlim(0,250)
                    plt.ylim(-50,100)
                    plt.subplot(2,1,2)
                    plt.ylabel('dBish',fontsize=16)
                    plt.xlabel('Freq (MHz)',fontsize=16)
                    plt.legend(fontsize=16)
                    plt.xlim(0,250)
                    plt.ylim(-50,100)

                if True:
                    plt.figure()
                    plt.suptitle('Max Correlation Values for Run %i'%(run),fontsize=20)
                    for channel in channels:
                        channel=int(channel)
                        if channel == 0:
                            ax = plt.subplot(4,2,channel+1)
                        else:
                            plt.subplot(4,2,channel+1,sharex=ax)
                        plt.hist(max_corrs['ch%i'%channel],bins=100,range=(-1.05,1.05),label='ch%i'%channel)
                        plt.ylabel('Counts',fontsize=16)
                        plt.xlabel('Correlation Value',fontsize=16)
                        plt.legend(fontsize=16)

                if False:            
                    times, subtimes, trigtimes, all_eventids = cc.getTimes(reader)

                    eventids_in_template = eventids
                    indices_of_eventids_in_template = numpy.where(numpy.isin(all_eventids, eventids_in_template))[0]

                    adjusted_trigtimes = trigtimes%known_pulser_ids['run%i'%run]['clock_rate']
                    fig = plt.figure()
                    plt.scatter(times,adjusted_trigtimes,c='b',marker=',',s=(72./fig.dpi)**2)
                    plt.scatter(times[indices_of_eventids_in_template],adjusted_trigtimes[indices_of_eventids_in_template],c='r',marker=',',s=(72./fig.dpi)**2,label='In Template')
                    plt.ylabel('Trig times')
                    plt.xlabel('Times')
                    plt.legend()

                    for channel in channels:
                        channel=int(channel)
                        fig = plt.figure()
                        plt.title('Channel %i'%channel)
                        plt.scatter(times[indices_of_eventids_in_template],adjusted_trigtimes[indices_of_eventids_in_template],c=max_corrs['ch%i'%channel],marker=',',s=(72./fig.dpi)**2)
                        plt.ylabel('Trig times')
                        plt.xlabel('Times')
                        cbar = plt.colorbar()
                        cbar.set_label('Max Correlation Value')

                template_dir = '/home/dsouthall/Projects/Beacon/beacon/analysis/templates'
                if save_templates:
                    dir_made = False
                    attempt = 0
                    while dir_made == False:
                        try:
                            os.mkdir(template_dir + '/run793_%i'%attempt)
                            template_dir = template_dir + '/run793_%i'%attempt
                            dir_made = True
                            print('Templates being saved in ' + template_dir)
                        except Exception as e:
                            print('Dir exists, altering')
                            attempt += 1

                        for channel in channels:
                            channel=int(channel)
                            y = templates['ch%i'%channel]
                            x = waveform_times
                            with open(template_dir + '/ch%i.csv'%channel, mode='w') as file:
                                writer = csv.writer(file, delimiter=',')
                                for i in range(len(x)):
                                    writer.writerow([x[i],y[i]])

            except Exception as e:
                print('Error in main loop.')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


