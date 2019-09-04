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
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
        'clock_rate':31249809.22371152
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


if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine
    nearest_neighbor = 10 #Adjust until works.
    scale_subtimes = 10.0 #The larger this is the the less the nearest neighbor favors vertical lines.
    scale_times = 1.0  #The larger this is the the less the nearest neighbor favors horizontal lines.
    slope_bound = 1.0e-9
    percent_cut = 0.001
    nominal_clock_rate = 31.25e6
    lower_rate_bound = 31.2e6 #Don't make the bounds too large or the bisection method will overshoot and roll over.
    upper_rate_bound = 31.3e6 #Don't make the bounds too large or the bisection method will overshoot and roll over.
    plot = True
    verbose = False
    use_known_ids = True
    save_templates = True
    resample_factor = 200
    manual_template = False

    #Filter settings
    crit_freq_low_pass_MHz = 75
    crit_freq_high_pass_MHz = 15
    filter_order = 6
    plot_filter = True

    #Other parameters
    initial_std_precentage_window = 0.2
    intial_std_threshold = 10.0 #The std in the first % defined above must be below this value for the event to be considered as a template.  

    channels = numpy.arange(8,dtype=int)

    for run_index, run in enumerate(runs):
        reader = Reader(datapath,run)
        waveforms = {}

        reader.setEntry(eventids[0])
        waveform_times = reader.t()
        dt = waveform_times[1]-waveform_times[0]
        waveform_times_factor2 = numpy.arange(2**(numpy.ceil(numpy.log2(len(waveform_times)))))*dt #Rounding up to a factor of 2 of the len of the waveforms
        padded_times = numpy.arange(2*len(waveform_times_factor2))*dt #multiplying by 2 for cross correlation later.
        
        filter_y,freqs = makeFilter(padded_times,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order,plot_filter=True)
        filter_y = numpy.ones_like(filter_y)
        
        #Prepare filter
        reader.setEntry(0)
        wf = reader.wf(0)
        wf , waveform_times = scipy.signal.resample(wf,len(wf)*resample_factor,t=reader.t())
        freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
        b, a = scipy.signal.butter(filter_order, crit_freq_low_pass_MHz*1e6, 'low', analog=True)
        d, c = scipy.signal.butter(filter_order, crit_freq_high_pass_MHz*1e6, 'high', analog=True)
        
        filter_x_low_pass, filter_y_low_pass = scipy.signal.freqs(b, a,worN=freqs)
        filter_x_high_pass, filter_y_high_pass = scipy.signal.freqs(d, c,worN=freqs)
        filter_x = freqs
        filter_y = numpy.multiply(filter_y_low_pass,filter_y_high_pass)
        
        if plot_filter == True:
            plt.figure()
            plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_low_pass)),label='low pass')
            plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_high_pass)),label='high pass')
            plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y)),label='final filter')
            plt.title('Butterworth filter frequency response')
            plt.xlabel('Frequency [MHz]')
            plt.ylabel('Amplitude [dB]')
            plt.margins(0, 0.1)
            plt.grid(which='both', axis='both')
            plt.axvline(crit_freq_low_pass_MHz, color='green') # cutoff frequency
            plt.axvline(crit_freq_high_pass_MHz, color='cyan') # cutoff frequency
            plt.xlim(0,250)

        try:
            if 'run%i'%run in list(known_pulser_ids.keys()) and use_known_ids:
                try:
                    eventids = known_pulser_ids['run%i'%run]['eventids']
                    print('Loaded known pulser eventids.')
                except Exception as e:
                    print('Failed to load known eventids:')
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print('Attempting to determine pulser eventids.')
                    clock_rate, times, subtimes, trigtimes, eventids, indices = cc.getClockCorrection(reader, nearest_neighbor=nearest_neighbor, scale_subtimes=scale_subtimes, scale_times=scale_times, slope_bound=slope_bound, percent_cut=percent_cut, nominal_clock_rate=nominal_clock_rate, lower_rate_bound=lower_rate_bound, upper_rate_bound=upper_rate_bound, plot=plot, verbose=verbose)
                    eventids = numpy.sort(eventids[indices])
                    adjusted_trig_times = trig_times%clock_rate
            else:
                print('Attempting to determine pulser eventids.')
                clock_rate, times, subtimes, trigtimes, eventids, indices = cc.getClockCorrection(reader, nearest_neighbor=nearest_neighbor, scale_subtimes=scale_subtimes, scale_times=scale_times, slope_bound=slope_bound, percent_cut=percent_cut, nominal_clock_rate=nominal_clock_rate, lower_rate_bound=lower_rate_bound, upper_rate_bound=upper_rate_bound, plot=plot, verbose=verbose)
                eventids = numpy.sort(eventids[indices])
                adjusted_trig_times = trig_times%clock_rate

            for channel in channels:
                channel=int(channel)
                waveforms['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length*resample_factor))

            #template peak signal cut:
            peak_cut = 65
            std_cut = 65 #SELECTING ALL
            selected_waveform_indices = []
            print('Loading waveforms\n')
            for waveform_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\r'%(waveform_index,len(eventids)))
                sys.stdout.flush()
                reader.setEntry(eventid)
                event_times = reader.t()
                for channel in channels:
                    channel=int(channel) 
                    #Load waveform
                    wf = reader.wf(channel)
                    #Cut on waveform
                    #print('%0.3f,%0.3f'%(numpy.max(numpy.fabs(wf)),numpy.std(wf)))
                    if numpy.logical_and(~numpy.any(numpy.fabs(wf) >= peak_cut),numpy.std(wf) < std_cut):
                        selected_waveform_indices.append(waveform_index)
                    wf = scipy.signal.resample(wf,len(wf)*resample_factor)

                    #Apply filter
                    wf = numpy.fft.irfft(numpy.multiply(filter_y,numpy.fft.rfft(wf)))
                    
                    #Save waveform
                    waveforms['ch%i'%channel][waveform_index] = wf


            selected_waveform_indices = numpy.unique(selected_waveform_indices)
            eventids_in_template = eventids[selected_waveform_indices]

            for channel in channels:
                channel=int(channel) 
                waveforms['ch%i'%channel] = waveforms['ch%i'%channel][selected_waveform_indices]

            #Actual template calculations

            templates = {}
            delays_index = {}
            delays_times = {}
            dt = waveform_times[1]-waveform_times[0]
            max_corrs = {}
            print('Making Templates')

            for channel in channels:
                channel=int(channel)
                sys.stdout.write('(%i/8)\r'%(channel))
                sys.stdout.flush()

                #Find template to correlate to:
                ratios = []
                early_std = []
                for index, waveform in enumerate(waveforms['ch%i'%channel]): 
                    ratios.append(numpy.max(numpy.abs(scipy.signal.hilbert(waveform)))/numpy.std(waveform))
                    early_std.append(numpy.std(waveform[0:int(initial_std_precentage_window*len(waveform))]))
                early_std = numpy.array(early_std)

                
                sorted_template_indices = numpy.argsort(ratios)[::-1][(early_std < intial_std_threshold)[numpy.argsort(ratios)[::-1]]] #Indices sorted from large to small in ratio, then cut on those templates with small enough initial std

                template_selected = False
                template_index = 0
                if manual_template == True:
                    while template_selected == False:
                        fig = plt.figure()
                        plt.plot(waveform_times,waveforms['ch%i'%channel][sorted_template_indices[template_index]])
                        plt.show()
                        
                        acceptable_response = False
                        while acceptable_response == False:
                            response = input('Is this a good waveform to start?  (y/n)')
                            if response == 'y':
                                acceptable_response = True
                                template_selected = True
                                template_waveform = waveforms['ch%i'%channel][sorted_template_indices[template_index]]
                                plt.close(fig)
                            elif response == 'n':
                                acceptable_response = True
                                template_selected = False
                                template_index += 1
                                plt.close(fig)
                                if template_index >= len(waveforms['ch%i'%channel]):
                                    print('All possible starting templates cycled through.  Defaulting to first option.')
                                    template_selected = True
                                    template_index = 0
                                    template_waveform = waveforms['ch%i'%channel][sorted_template_indices[template_index]]
                            else:
                                print('Response not accepted.  Please type either y or n')
                                acceptable_response = False
                else:
                    template_selected = True
                    template_waveform = waveforms['ch%i'%channel][sorted_template_indices[template_index]]

                print('Run %i Channel %i template index is %i corresponding to event %i'%(run,channel,template_index,eventids[sorted_template_indices[template_index]]))

                if False: #Method 1
                    delays_index['ch%i'%channel] = []
                    delays_times['ch%i'%channel] = []
                    max_corrs['ch%i'%channel] = numpy.zeros((len(eventids_in_template)))
                    correlated_waveforms = numpy.zeros((len(eventids_in_template),reader.header().buffer_length*resample_factor),dtype=float)
                    for waveform_index, waveform in enumerate(waveforms['ch%i'%channel]):
                        corr = scipy.signal.correlate(template_waveform/numpy.std(template_waveform),waveform/numpy.std(waveform))/(len(template_waveform)) #should be roughly normalized between -1,1
                        #peak_indices = scipy.signal.find_peaks(corr,height=0.3,distance=500)[0]

                        sorted_corr = numpy.argsort(corr)
                        max_corrs['ch%i'%channel][waveform_index] = numpy.max(corr)
                        index_delay = int(numpy.argmax((corr))-numpy.size(corr)/2.)
                        delays_index['ch%i'%channel].append(index_delay)
                        delays_times['ch%i'%channel].append(index_delay*dt)
                        correlated_waveforms[waveform_index] = numpy.roll(waveform,index_delay)
                    final_template = numpy.average(correlated_waveforms,weights=max_corrs['ch%i'%channel],axis=0) #numpy.mean(correlated_waveforms,axis=0)
                if True: #Method 2
                    all_peak_indices = []
                    all_peak_weights = []
                    distance = 10.0 #The ns distance between peaks in the correlation. 
                    height = 0.25 #The minimum height in the correlation for it to count as a peak.
                    for waveform_index, waveform in enumerate(waveforms['ch%i'%channel]):
                        corr = scipy.signal.correlate(template_waveform/numpy.std(template_waveform),waveform/numpy.std(waveform))/(len(template_waveform)) #should be roughly normalized between -1,1
                        peak_indices = scipy.signal.find_peaks(corr,height=height,distance=distance/dt)[0]
                        all_peak_indices.append(peak_indices)
                        all_peak_weights.append(corr[peak_indices])
                        if False:
                            fig = plt.figure()
                            plt.plot(numpy.arange(len(corr)),corr)      
                            plt.scatter(peak_indices,corr[peak_indices],c='r')
                            input()
                            plt.close(fig)
                    peak_weights = numpy.concatenate(all_peak_weights)**2
                    hist,bin_edges = numpy.histogram(numpy.concatenate(all_peak_indices),weights=peak_weights,bins=numpy.arange(min(numpy.concatenate(all_peak_indices)),max(numpy.concatenate(all_peak_indices))+1e-10,distance/(2.0*dt)))
                    plt.figure()
                    plt.title('Weighted Correlation Peak Indices, Channel %i'%channel)
                    plt.hist(numpy.concatenate(all_peak_indices),weights=peak_weights,bins=numpy.arange(min(numpy.concatenate(all_peak_indices)),max(numpy.concatenate(all_peak_indices))+1e-10,distance/(2.0*dt)))
                    try:
                        align_to = (bin_edges[numpy.argmax(hist)] + bin_edges[numpy.argmax(hist)+1])/2.0
                    except Exception as e:
                        print('Tried selecting the location of the maximum correlation peak.  Made error.')
                    correlated_waveforms = []
                    weights = []
                    for waveform_index, waveform in enumerate(waveforms['ch%i'%channel]):
                        if numpy.size(all_peak_indices[waveform_index]) == 0:
                            continue
                        index_delay = int(all_peak_indices[waveform_index][numpy.argmin(all_peak_indices[waveform_index] - align_to)]-numpy.size(corr)/2.)
                        correlated_waveforms.append(numpy.roll(waveform,index_delay))
                        weights.append(all_peak_weights[waveform_index][numpy.argmin(all_peak_indices[waveform_index] - align_to)])
                    correlated_waveforms = numpy.array(correlated_waveforms)
                    final_template = numpy.average(correlated_waveforms,axis=0,weights=weights) #Should still probably weight these averages somehow?

                if True:
                    iterate_limit = 10
                    i = 0
                    while i < iterate_limit:
                        i += 1
                        #Optional second step of correlation with the first averaged template.
                        correlated_waveforms = numpy.zeros((len(eventids_in_template),reader.header().buffer_length*resample_factor))
                        delays_index['ch%i'%channel] = []
                        delays_times['ch%i'%channel] = []
                        max_corrs['ch%i'%channel] = numpy.zeros((len(eventids_in_template)))

                        for waveform_index, waveform in enumerate(waveforms['ch%i'%channel]):
                            corr = scipy.signal.correlate(final_template/numpy.std(final_template),waveform/numpy.std(waveform))/(len(final_template)) #should be roughly normalized between -1,1
                            max_corrs['ch%i'%channel][waveform_index] = numpy.max(corr)
                            index_delay = int(numpy.argmax((corr))-numpy.size(corr)/2.)
                            delays_index['ch%i'%channel].append(index_delay)
                            delays_times['ch%i'%channel].append(index_delay*dt)
                            correlated_waveforms[waveform_index] = numpy.roll(waveform,index_delay)

                        final_template = numpy.average(correlated_waveforms,weights=max_corrs['ch%i'%channel],axis=0)

                templates['ch%i'%channel] = final_template

            if True:
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

            if True:
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

            if True:            
                times, subtimes, trigtimes, all_eventids = cc.getTimes(reader)

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




