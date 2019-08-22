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
template_dirs = {
    'run793':{  'dir':'/home/dsouthall/Projects/Beacon/beacon/analysis/templates/run793_4',
                'resample_factor' : 200,
                'crit_freq_low_pass_MHz' : 75,
                'crit_freq_high_pass_MHz' : 15,
                'filter_order' : 6,
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

def loadTemplates(template_path):
    waveforms = {}
    for channel in range(8):
        with open(template_path + '/ch%i.csv'%channel) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            x = []
            y = []
            for row in csv_reader:
                x.append(float(row[0]))
                y.append(float(row[1]))
                line_count += 1
            waveforms['ch%i'%channel] = numpy.array(y)
    return x,waveforms


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


if __name__ == '__main__':
    #plt.close('all')
    # If your data is elsewhere, pass it as an argument
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine
    event_limit = 1
    save_fig = True

    for run_index, run in enumerate(runs):
        reader = Reader(datapath,run)
        eventids = cc.getTimes(reader)[3]

        if event_limit is not None:
            if event_limit < len(eventids):
                eventids = eventids[0:event_limit]

        waveform_times, templates = loadTemplates(template_dirs['run%i'%run]['dir'])
        original_wf_len = int(reader.header().buffer_length)
        upsample_wf_len = original_wf_len*template_dirs['run%i'%run]['resample_factor']
        corr_delay_times = numpy.arange(-upsample_wf_len+1,upsample_wf_len)
        #Setup Filter
        freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
        b, a = scipy.signal.butter(template_dirs['run%i'%run]['filter_order'], template_dirs['run%i'%run]['crit_freq_low_pass_MHz']*1e6, 'low', analog=True)
        d, c = scipy.signal.butter(template_dirs['run%i'%run]['filter_order'], template_dirs['run%i'%run]['crit_freq_high_pass_MHz']*1e6, 'high', analog=True)

        filter_x_low_pass, filter_y_low_pass = scipy.signal.freqs(b, a,worN=freqs)
        filter_x_high_pass, filter_y_high_pass = scipy.signal.freqs(d, c,worN=freqs)
        filter_x = freqs
        filter_y = numpy.multiply(filter_y_low_pass,filter_y_high_pass)
        filter_y = numpy.tile(filter_y,(8,1))
        templates_scaled = {} #precomputing
        len_template = len(waveform_times)

        templates_scaled_2d = numpy.zeros((8,len_template))
        for channel in range(8):
            templates_scaled['ch%i'%channel] = templates['ch%i'%channel]/numpy.std(templates['ch%i'%channel])
            templates_scaled_2d[channel] = templates['ch%i'%channel]/numpy.std(templates['ch%i'%channel])

        if True:
            plt.figure()
            plt.suptitle('Averaged Templates for Run %i'%(run),fontsize=20)
            for channel in range(8):
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
            for channel in range(8):
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
            plt.xlim(0,250)
            plt.ylim(-50,100)
            plt.legend(fontsize=16)
            plt.subplot(2,1,2)
            plt.ylabel('dBish',fontsize=16)
            plt.xlabel('Freq (MHz)',fontsize=16)
            plt.legend(fontsize=16)
            plt.xlim(0,250)
            plt.ylim(-50,100)
        '''
        max_corrs = {}
        delays = {}
        for channel in range(8):
            max_corrs['ch%i'%channel] = numpy.zeros(len(eventids))
            delays['ch%i'%channel] = numpy.zeros(len(eventids))
        '''
        
        max_corrs = numpy.zeros((len(eventids),8))
        delays = numpy.zeros((len(eventids),8))

        for event_index, eventid in enumerate(eventids):
            sys.stdout.write('\r(%i/%i)'%(event_index+1,len(eventids)))
            sys.stdout.flush()
            reader.setEntry(eventid) 

            wfs = numpy.zeros((8,original_wf_len))
            for channel in range(8):
                wfs[channel] = reader.wf(channel)
            
            wfs = scipy.signal.resample(wfs,upsample_wf_len,axis=1)

            #Apply filter
            wfs = numpy.fft.irfft(numpy.multiply(filter_y,numpy.fft.rfft(wfs,axis=1)),axis=1)

            #Can't find a way to vectorize corr
            scaled_wfs = wfs/numpy.tile(numpy.std(wfs,axis=1),(numpy.shape(wfs)[1],1)).T
            corr = numpy.zeros((8,numpy.shape(wfs)[1]*2 - 1))
            for channel in range(8):
                corr[channel] = scipy.signal.correlate(templates_scaled_2d[channel],scaled_wfs[channel])/(len_template) #should be roughly normalized between -1,1
            
            max_corrs[event_index] = numpy.max(corr,axis=1)
            delays[event_index] = corr_delay_times[numpy.argmax(corr,axis=1)]

        if 'run%i'%run in list(known_pulser_ids.keys()):
            print('Testing against known pulser events')
            pulser_max_corrs = numpy.zeros((len(known_pulser_ids['run%i'%run]['eventids']),8))
            pulser_delays = numpy.zeros((len(known_pulser_ids['run%i'%run]['eventids']),8))

            for event_index, eventid in enumerate(known_pulser_ids['run%i'%run]['eventids']):
                sys.stdout.write('\r(%i/%i)'%(event_index+1,len(known_pulser_ids['run%i'%run]['eventids'])))
                sys.stdout.flush()
                reader.setEntry(eventid) 
                wfs = numpy.zeros((8,original_wf_len))
                for channel in range(8):
                    wfs[channel] = reader.wf(channel)

                wfs = scipy.signal.resample(wfs,upsample_wf_len,axis=1)

                #Apply filter
                wfs = numpy.fft.irfft(numpy.multiply(filter_y,numpy.fft.rfft(wfs,axis=1)),axis=1)

                #Can't find a way to vectorize corr
                scaled_wfs = wfs/numpy.tile(numpy.std(wfs,axis=1),(numpy.shape(wfs)[1],1)).T
                corr = numpy.zeros((8,numpy.shape(wfs)[1]*2 - 1))
                for channel in range(8):
                    corr[channel] = scipy.signal.correlate(templates_scaled_2d[channel],scaled_wfs[channel])/(len_template) #should be roughly normalized between -1,1
                pulser_max_corrs[event_index] = numpy.max(corr,axis=1)
                pulser_delays[event_index] = corr_delay_times[numpy.argmax(corr,axis=1)]





        if True:
            fig = plt.figure(figsize=(16,12))
            plt.suptitle('Max Correlation Values for Run %i'%(run),fontsize=20)

            times, subtimes, trigtimes, all_eventids = cc.getTimes(reader)


            for channel in range(8):
                if channel == 0:
                    ax = plt.subplot(4,2,channel+1)
                else:
                    plt.subplot(4,2,channel+1,sharex=ax)
                max_bin = numpy.max(numpy.histogram(max_corrs[:,channel],bins=100,range=(-1.05,1.05))[0])
                plt.hist(max_corrs[:,channel],weights=numpy.ones_like(max_corrs[:,channel])/max_bin,bins=100,range=(-1.05,1.05),label='ch%i all events'%channel,alpha=0.5)
                if 'run%i'%run in list(known_pulser_ids.keys()):
                    max_bin = numpy.max(numpy.histogram(pulser_max_corrs[:,channel],bins=100,range=(-1.05,1.05))[0])
                    plt.hist(pulser_max_corrs[:,channel],weights=numpy.ones_like(pulser_max_corrs[:,channel])/max_bin,bins=100,range=(-1.05,1.05),label='ch%i pulser events'%channel,alpha=0.5)
                plt.ylabel('Counts',fontsize=16)
                plt.xlabel('Correlation Value',fontsize=16)
                plt.legend(fontsize=16)


                plt.figure()
                plt.plot(times[numpy.isin(all_eventids,known_pulser_ids['run%i'%run]['eventids'])],pulser_max_corrs[:,channel])

                plt.figure()
                plt.subplot(2,1,1)
                for index in numpy.where(max_corrs[:,channel] < 0.7):
                    plt.plot(waveform_times,waveforms['ch%i'%channel][index],alpha=0.5)        
                plt.subplot(2,1,2)
                for index in numpy.where(max_corrs[:,channel] > 0.8):
                    plt.plot(waveform_times,waveforms['ch%i'%channel][index],alpha=0.5)        


            if save_fig:
                fig_saved = False
                attempt = 0
                while fig_saved == False:
                    filename = 'template_search_%i.png'%attempt
                    if os.path.exists(filename):
                        print('%s exists, altering name.'%filename)
                        attempt += 1
                    else:
                        try:
                            fig.savefig(filename)
                            print('%s saved'%filename)
                            fig_saved = True

                        except Exception as e:
                            print('Error while saving figure.')
                            print(e)
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)

                        

