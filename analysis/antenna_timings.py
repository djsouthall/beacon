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
    
    resample_factor = 100

    #Filter settings
    crit_freq_low_pass_MHz = 80
    crit_freq_high_pass_MHz = 35
    filter_order = 6
    plot_filter = True

    for run_index, run in enumerate(runs):
        reader = Reader(datapath,run)
        waveform_times = reader.t()
        waveforms = {}
        
        try:
            if 'run%i'%run in list(known_pulser_ids.keys()) and use_known_ids:
                try:
                    eventids = known_pulser_ids['run%i'%run]['eventids']
                    print('Loaded known pulser eventids.')
                except Exception as e:
                    print('Failed to load known eventids:')
                    print(e)
                    print('Attempting to determine pulser eventids.')
                    clock_rate, times, subtimes, trig_times, eventids = cc.getClockCorrection(reader, nearest_neighbor=nearest_neighbor, scale_subtimes=scale_subtimes, scale_times=scale_times, slope_bound=slope_bound, percent_cut=percent_cut, nominal_clock_rate=nominal_clock_rate, lower_rate_bound=lower_rate_bound, upper_rate_bound=upper_rate_bound, plot=plot, verbose=verbose)
                    adjusted_trig_times = trig_times%clock_rate
            else:
                print('Attempting to determine pulser eventids.')
                clock_rate, times, subtimes, trig_times, eventids = cc.getClockCorrection(reader, nearest_neighbor=nearest_neighbor, scale_subtimes=scale_subtimes, scale_times=scale_times, slope_bound=slope_bound, percent_cut=percent_cut, nominal_clock_rate=nominal_clock_rate, lower_rate_bound=lower_rate_bound, upper_rate_bound=upper_rate_bound, plot=plot, verbose=verbose)
                adjusted_trig_times = trig_times%clock_rate

            for channel in range(8):
                waveforms['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length),dtype=int)

            #Cut out saturating signals or particularly noisy signals.
            peak_cut = 63
            std_cut = 15
            complying_waveforms = []
            print('Loading Waveforms')
            for waveform_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\r'%(waveform_index,len(eventids)))
                sys.stdout.flush()
                reader.setEntry(eventid)
                event_times = reader.t()
                for channel in range(8):
                    waveform = reader.wf(channel) 
                    if ~(numpy.any(numpy.fabs(waveform) >= peak_cut) or numpy.std(waveform) > std_cut):
                        complying_waveforms.append(waveform_index)
                    waveforms['ch%i'%channel][waveform_index] = reader.wf(channel)
            for channel in range(8):
                waveforms['ch%i'%channel] = waveforms['ch%i'%channel][numpy.unique(complying_waveforms)]
            print('\n%i/%i events passed cuts and are being used.'%(len(complying_waveforms),len(eventids)))
            eventids = eventids[numpy.unique(complying_waveforms)]
            dt = waveforms['ch0'][0][1]-waveforms['ch0'][0][0]


            #
            delays_even = numpy.zeros((len(eventids),4,4))
            max_corrs_even = numpy.zeros((len(eventids),4,4))
            delays_odd = numpy.zeros((len(eventids),4,4))
            max_corrs_odd = numpy.zeros((len(eventids),4,4))
            print('Cross correlating events between antennas.')
            for event_index in range(len(eventids)):
                sys.stdout.write('\r(%i/%i)'%(event_index,len(eventids)))
                sys.stdout.flush()
                for i in range(4):
                    x_even = waveforms['ch%i'%(2*i)][event_index]/numpy.std(waveforms['ch%i'%(2*i)][event_index])
                    x_odd = waveforms['ch%i'%(2*i+1)][event_index]/numpy.std(waveforms['ch%i'%(2*i+1)][event_index])
                    for j in range(4):
                        if j >= i:
                            continue
                        else:
                            y_even = waveforms['ch%i'%(2*j)][event_index]/numpy.std(waveforms['ch%i'%(2*j)][event_index])
                            y_odd = waveforms['ch%i'%(2*j+1)][event_index]/numpy.std(waveforms['ch%i'%(2*j+1)][event_index])
                            corr_even = scipy.signal.correlate(x_even,y_even)/(len(x_even)) #should be roughly normalized between -1,1
                            corr_odd = scipy.signal.correlate(x_odd,y_odd)/(len(x_odd)) #should be roughly normalized between -1,1

                            max_corrs_even[event_index][i][j] = numpy.max(corr_even)
                            delays_even[event_index][i][j] = int(numpy.argmax((corr_even))-numpy.size(corr_even)/2.)*dt
                            max_corrs_odd[event_index][i][j] = numpy.max(corr_odd)
                            delays_odd[event_index][i][j] = int(numpy.argmax((corr_odd))-numpy.size(corr_odd)/2.)*dt

            mean_corrs_even = numpy.mean(max_corrs_even,axis=0)
            mean_corrs_odd = numpy.mean(max_corrs_odd,axis=0)


            for i in range(4):
                for j in range(4):
                    if j >= i:
                        continue
                    else:
                        plt.figure()
                        plt.suptitle('Cross Correlation Times Between Antenna %i and %i'%(i,j))

                        ax = plt.subplot(2,1,1)
                        plt.hist(delays_even[:,i,j],label=('Channel %i and %i'%(2*i,2*j)),bins=100)
                        plt.xlabel('Delay (ns)',fontsize=16)
                        plt.ylabel('Counts',fontsize=16)
                        plt.legend(fontsize=16)

                        plt.subplot(2,1,2,sharex=ax)
                        plt.hist(delays_odd[:,i,j],label=('Channel %i and %i'%(2*i+1,2*j+1)),bins=100)
                        plt.xlabel('Delay (ns)',fontsize=16)
                        plt.ylabel('Counts',fontsize=16)
                        plt.legend(fontsize=16)

        except Exception as e:
            print('Error in main loop.')
            print(e)






