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

#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.
known_pulser_ids = {
    'run792':{
        'eventids':numpy.array([112256, 112889, 112111, 112127, 112149, 111733, 112275, 111689,\
            112235, 112125, 112193, 112895, 112167, 112162, 112201, 111774,\
            111888, 112106, 111008, 111018, 112875, 112305, 111878, 111798,\
            111788, 111025, 112100, 111507, 114356, 111825, 111803, 110971,\
            112091, 111674, 111902, 112854, 112311, 111061, 112075, 111476,\
            111654, 111605, 111636, 111037, 112907, 111155, 114628, 112322,\
            114565, 112054, 114610, 114427, 114593, 111051, 110926, 112333,\
            114530, 111087, 112918, 112768, 111942, 111620, 112048, 111929,\
            113517, 114559, 110946, 113521, 114701, 114547, 109405, 112761,\
            112342, 112922, 112019, 114326, 111445, 112028, 111437, 113526,\
            109475, 112932, 112506, 114323, 111424, 111207, 109902, 110760,\
            109394, 113247, 114978, 113231, 113533, 113006, 109879, 109676,\
            115023, 112666, 109501, 110746, 113226, 109489, 114842, 114825,\
            113546, 113244, 114930, 111408, 109651, 110020, 110095, 111289,\
            109372, 109493, 114299, 114312, 114290, 110676, 111241, 110691,\
            109622, 111271, 114303, 113208, 109366, 110364, 110085, 110656,\
            109361, 110076, 109351, 113620, 110038, 114089, 110209, 115062,\
            113760, 110146, 108250, 108348, 108421, 115070, 114070, 115088,\
            114042, 108162, 108267]),
        'clock_rate':31249812.04283368
            },
    'run793':{
        'eventids':numpy.array([    98958,  98382,  98477, 102883, 102771, 101954, 110633, 106755,\
                                   102421, 102550, 106845,  98182,  98270, 101931,  97476,  97880,\
                                   102863, 106911, 102605, 106996, 105160, 102495, 110519, 110415,\
                                   106927,  97920, 106808, 106887, 102815, 102701, 106938,  97491,\
                                    98706, 102523, 102629, 110465, 110372,  97982,  98075,  97898,\
                                    96868, 106972, 105121,  93236, 106285,  97534, 106265,  98421,\
                                   110561,  98325,  94907,  98723,  98402,  98305,  94885, 105138,\
                                   110747, 110645,  98287, 102796, 102676,  98911, 102582, 106744,\
                                   106822, 110245, 110330, 110544, 102407,  98813,  98943,  94105,\
                                    93254,  97938,  97862,  98737,  98041, 106873,  98439, 110359,\
                                    98364,  99008, 102201,  99250, 106787,  93295,  98457,  98928,\
                                    98001,  97839,  98971,  94856,  98061,  94872,  98093, 106855,\
                                   102560, 106949, 102446,  98131, 106896, 102102,  98143, 102840,\
                                   110390,  98346, 108060, 110665, 108130, 110317, 110483, 110687,\
                                   110606, 110711,  98752,  98685, 108075,  98696, 110499,  98893,\
                                    98490, 109453, 109471, 106395, 110438, 110765, 101900,  95429,\
                                    97963, 101976, 102025, 102131,  93275, 100910,  98163,  98201,\
                                    98221,  94789, 100597,  94130,  98237,  98251,  94832,  98504,\
                                    99804,  94063,  94810,  98109, 102227,  98025, 102653, 102392,\
                                   102726,  93199, 102745, 102344,  99025,  97850,  98835, 106714,\
                                   106771, 106671, 104185, 110347, 110273, 110260, 110584, 110778,\
                                   107683, 102470, 107572,  96573, 102370,  93215,  98763, 110402,\
                                   110300, 105617,  97796,  97808, 107664, 102048, 106334, 101875,\
                                   102000, 106697,  98799,  98862,  97829, 106373,  97747, 106310,\
                                   109438, 107632,  94777, 108094,  99333, 100573,  98777,  98844,\
                                   102319,  97766, 109413, 102275, 102292,  98983, 100881,  99827,\
                                   106727,  98880,  99267, 108110,  99176,  96559,  96883, 105176,\
                                   110731, 110227,  97511,  94748, 105101,  99310, 107587, 100552,\
                                    99048,  96856,  93623,  93315,  97567,  97458, 108037,  95397,\
                                    95502, 108207, 108153,  99691,  99205,  99363,  95463,  98560,\
                                    99193,  99098,  99084,  98650, 106412, 108931,  98522,  98623,\
                                   102165, 107015, 100934, 107031, 107527,  99746,  99891,  99720,\
                                   109392, 109519,  96624, 106349,  97652,  97549,  97727, 109487,\
                                   110075, 109530, 110005, 109083, 108953,  97600,  97702,\
                                   107065,  93335, 104162, 104092, 104211, 104110,  97616,  96940,\
                                    94922,  94728,  99379,  95480, 109049, 106230, 109508, 108906,\
                                   102903,  99221, 101821,  99342,  93575, 100619,  98539,  98639,\
                                    99391, 108170, 109066,  96539,  96639, 109175, 109123, 104141,\
                                   109268, 109147,  99165, 102148, 105190, 105645,  95444, 105086,\
                                   108186, 110185,  93156,  94038,  93356,  96482,  93180,  93385,\
                                    96497,  99664,  94024,  99123, 101269,  99779, 100856, 107049,\
                                   107538,  96526,  97684,  93144,  98668,  93987, 101773,  96511,\
                                    94705,  94146, 100499,  94158,  96605, 101009, 105672, 105536,\
                                    95519, 109329, 104124, 110816, 101844,  98576, 101854,  95413,\
                                    99439, 109021, 103217, 108982,  94053, 107724, 106194, 107705,\
                                   102190, 107554, 103264, 101564, 109371, 100344,  93551, 105595,\
                                   108833,  99914, 107750,  93954, 109203, 104061, 110160,  93459,\
                                   110109,  93532, 109106,  97669, 109310, 110022, 110138, 110037,\
                                   109244, 108887, 100962,  99609,  93438, 105578, 105706, 101024,\
                                   105723, 107767, 106431,  96802, 108266,  96904,  93645,  99406,\
                                   108846, 106211, 101212, 106090, 101578,  96830,  96963,  93483,\
                                    97436, 105996, 106103, 109986, 107983, 107998, 105923, 105061,\
                                   106176,  93659,  93509, 105560, 105414,  99462, 103241, 100756,\
                                   100776, 110833,  96715, 109624,  96816, 109905, 109578, 106627,\
                                    94670,  96701, 100453,  99578, 106142, 101717, 109604, 109924,\
                                   101678, 104021,  94942, 109715, 100477, 100651, 109942,  99522,\
                                   101122, 107849, 103994, 105912, 106010, 102927,  96790,  96652,\
                                    95365, 103980, 105036, 105474, 105387, 101139, 101343, 105518,\
                                   101164,  97055,  93933, 105951,  93106, 101091, 100699, 108699,\
                                   100724, 107484, 105808,  96378, 101616, 100801, 103167, 103196,\
                                    94177, 100670, 106607,  94687, 105890,  97417, 106054, 105783,\
                                   101183, 105839, 109737, 108812, 107107, 107502,  94652, 106524,\
                                    94604,  97402, 106582, 106471,  96365, 108770,  99948,  93075,\
                                   105212,  95357, 107786, 101526, 101657, 108285,  96668,  96989,\
                                   100829, 106593, 106486,  93058,  93915, 108725, 107962,\
                                   109979, 103347,  97078, 109809,  92215,  92373, 109681,  94958,\
                                    94265,  93768,  94280,  93685,  93701,  92241,  95556, 101637,\
                                   109664, 104258,  95690,  95830, 105971, 100003,  94200,  96355,\
                                   107872, 104993, 103893,  94975, 103305, 103152, 102956, 100302,\
                                    99980,  96335, 106036,  95914, 105823, 103844, 110843, 103323,\
                                   109842, 107141,  92317, 103939, 107456, 107161, 105375, 109828,\
                                   108660, 109771, 100275, 107441, 105227, 102984,  95328,  92191,\
                                   108308,  97134,  94492,  92426,  94559,  92509, 100239, 103399,\
                                    96260,  96137,  95813,  94535,  95590,  96119,  95971,  92167,\
                                   101504]),
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
    save_templates = False
    resample_factor = 100
    manual_template = False

    #Filter settings
    crit_freq_low_pass_MHz = 80
    crit_freq_high_pass_MHz = 35
    filter_order = 6
    plot_filter = True

    channels = numpy.arange(2,dtype=int)

    for run_index, run in enumerate(runs):
        reader = Reader(datapath,run)
        waveforms = {}

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
                    clock_rate, times, subtimes, trig_times, eventids = cc.getClockCorrection(reader, nearest_neighbor=nearest_neighbor, scale_subtimes=scale_subtimes, scale_times=scale_times, slope_bound=slope_bound, percent_cut=percent_cut, nominal_clock_rate=nominal_clock_rate, lower_rate_bound=lower_rate_bound, upper_rate_bound=upper_rate_bound, plot=plot, verbose=verbose)
                    adjusted_trig_times = trig_times%clock_rate
            else:
                print('Attempting to determine pulser eventids.')
                clock_rate, times, subtimes, trig_times, eventids = cc.getClockCorrection(reader, nearest_neighbor=nearest_neighbor, scale_subtimes=scale_subtimes, scale_times=scale_times, slope_bound=slope_bound, percent_cut=percent_cut, nominal_clock_rate=nominal_clock_rate, lower_rate_bound=lower_rate_bound, upper_rate_bound=upper_rate_bound, plot=plot, verbose=verbose)
                adjusted_trig_times = trig_times%clock_rate

            for channel in channels:
                channel=int(channel)
                waveforms['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length*resample_factor))

            #template peak signal cut:
            peak_cut = 63
            std_cut = 15
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
                for index, waveform in enumerate(waveforms['ch%i'%channel]): 
                    ratios.append(numpy.max(numpy.abs(scipy.signal.hilbert(waveform)))/numpy.std(waveform))
                template_index = numpy.argmax(ratios)

                sorted_template_indices = numpy.argsort(ratios)[::-1]

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
                                acceptable_response = False          +
                else:
                    template_selected = True
                    template_waveform = waveforms['ch%i'%channel][sorted_template_indices[template_index]]

                print('Run %i Channel %i template index is %i corresponding to event %i'%(run,channel,template_index,sorted_template_indices[template_index]))

                correlated_waveforms = numpy.zeros((len(eventids_in_template),reader.header().buffer_length*resample_factor),dtype=int)
                delays_index['ch%i'%channel] = []
                delays_times['ch%i'%channel] = []
                max_corrs['ch%i'%channel] = numpy.zeros((len(eventids_in_template)))

                for waveform_index, waveform in enumerate(waveforms['ch%i'%channel]):
                    corr = scipy.signal.correlate(template_waveform/numpy.std(template_waveform),waveform/numpy.std(waveform))/(len(template_waveform)) #should be roughly normalized between -1,1
                    #peak_indices = scipy.signal.find_peaks(corr,height=0.3,distance=500)[0]

                    sorted_corr = numpy.argsort(corr)
                    max_corrs['ch%i'%channel][waveform_index] = numpy.max(corr)
                    index_delay = int(numpy.argmax((corr))-numpy.size(corr)/2.)
                    delays_index['ch%i'%channel].append(index_delay)
                    delays_times['ch%i'%channel].append(index_delay*dt)
                    correlated_waveforms[waveform_index] = numpy.roll(waveform,index_delay)


                    if True:
                        fig = plt.figure()
                        plt.plot(numpy.arange(len(corr)),corr)      
                        plt.scatter(peak_indices,corr[peak_indices],c='r')
                        input()
                        plt.close(fig)

                    '''
                    max_corrs['ch%i'%channel][waveform_index] = numpy.max(corr)
                    index_delay = int(numpy.argmax((corr))-numpy.size(corr)/2.)
                    delays_index['ch%i'%channel].append(index_delay)
                    delays_times['ch%i'%channel].append(index_delay*dt)
                    correlated_waveforms[waveform_index] = numpy.roll(waveform,index_delay)
                    '''
                final_template = numpy.average(correlated_waveforms,weights=max_corrs['ch%i'%channel],axis=0) #numpy.mean(correlated_waveforms,axis=0)

                if True:
                    #Optional second step of correlation with the first averaged template.
                    correlated_waveforms = numpy.zeros((len(eventids_in_template),reader.header().buffer_length*resample_factor),dtype=int)
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
                plt.subplot(2,1,2)
                plt.ylabel('dBish',fontsize=16)
                plt.xlabel('Freq (MHz)',fontsize=16)
                plt.legend(fontsize=16)
                plt.xlim(0,250)

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
                plt.figure()
                plt.suptitle('Correlation Time Delays for Run %i'%(run),fontsize=20)
                for channel in channels:
                    channel=int(channel)
                    if channel == 0:
                        ax = plt.subplot(4,2,channel+1)
                    else:
                        plt.subplot(4,2,channel+1,sharex=ax)
                    plt.hist(delays_times['ch%i'%channel],bins=200,range=(-100,100),label='ch%i'%channel)
                    plt.ylabel('Counts',fontsize=16)
                    plt.xlabel('Max Correlation Time Delay from Template (ns)',fontsize=16)
                    plt.legend(fontsize=16)

            if False:            
                times, subtimes, trigtimes = cc.getTimes(reader)
                adjusted_trigtimes = trigtimes%known_pulser_ids['run%i'%run]['clock_rate']
                fig = plt.figure()
                plt.scatter(times,adjusted_trigtimes,c='b',marker=',',s=(72./fig.dpi)**2)
                plt.scatter(times[eventids_in_template],adjusted_trigtimes[eventids_in_template],c='r',marker=',',s=(72./fig.dpi)**2,label='In Template')
                plt.ylabel('Trig times')
                plt.xlabel('Times')
                plt.legend()

                for channel in channels:
                    channel=int(channel)
                    fig = plt.figure()
                    plt.title('Channel %i'%channel)
                    #print(eventids_in_template[numpy.where(adjusted_trigtimes[eventids_in_template] == numpy.min(adjusted_trigtimes[eventids_in_template]))])
                    plt.scatter(times[eventids_in_template],adjusted_trigtimes[eventids_in_template],c=max_corrs['ch%i'%channel],marker=',',s=(72./fig.dpi)**2)
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




            '''
            plt.figure()
            eid=100
            ax = plt.subplot(2,1,1)
            freqs, spec_dbish = rfftWrapper(waveform_times, waveforms['ch1'][eid])
            plt.plot(freqs/1e6,rfftWrapper(waveform_times, waveforms['ch1'][eid])[1],label='ch1')
            plt.plot(freqs/1e6,rfftWrapper(waveform_times, waveforms['ch7'][eid])[1],label='ch7')
            plt.ylabel('dBish')
            plt.xlabel('Freqs (MHz)')
            plt.legend()
            plt.subplot(2,1,2,sharex=ax)
            freqs, spec_dbish = rfftWrapper(waveform_times, waveforms['ch1'][eid])
            plt.plot(freqs/1e6,rfftWrapper(waveform_times, waveforms['ch3'][eid])[1],label='ch3')
            plt.plot(freqs/1e6,rfftWrapper(waveform_times, waveforms['ch5'][eid])[1],label='ch5')
            plt.ylabel('dBish')
            plt.xlabel('Freqs (MHz)')
            plt.legend()
            '''
        except Exception as e:
            print('Error in main loop.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)




