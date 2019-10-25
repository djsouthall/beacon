'''
This file contains the classes used for time delays and cross correlations.
The goal is to centralize essential calculations such that I am at least consistent
between various scripts. 
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv
import inspect

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info
from objects.fftmath import TemplateCompareTool,TimeDelayCalculator

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()


if __name__ == '__main__':
    try:
         #plt.close('all')
        # If your data is elsewhere, pass it as an argument
        datapath = os.environ['BEACON_DATA']
        default_site = 1

        if len(sys.argv) == 2:
            if int(sys.argv[1]) in [1,2,3]:
                site = int(sys.argv[1])
            else:
                print('Given site not in options.  Defaulting to %i'%default_site)
                site = default_site
        else:
            print('No site given.  Defaulting to %i'%default_site)
            site = default_site


        if site == 1:
            waveform_index_range = (1500,None) #Looking at the later bit of the waveform only, 10000 will cap off.  
            run = 1507
            cfd_thresh = 0.8

            expected_time_differences_physical  =  [((0, 1), -55.818082350306895), ((0, 2), 82.39553727998077), ((0, 3), 18.992683496782092), ((1, 2), 138.21361963028767), ((1, 3), 74.81076584708899), ((2, 3), -63.40285378319868)]
            max_time_differences_physical  =  [((0, 1), -129.37157110284315), ((0, 2), 152.48216718324971), ((0, 3), 184.0463943150346), ((1, 2), 139.82851662731397), ((1, 3), 104.08254793314117), ((2, 3), -81.41340851163496)]
            expected_time_differences_hpol  =  [((0, 1), -48.64981133952688), ((0, 2), 98.30134449320167), ((0, 3), 28.04902069144964), ((1, 2), 146.95115583272855), ((1, 3), 76.69883203097652), ((2, 3), -70.25232380175203)]
            max_time_differences_hpol  =  [((0, 1), -126.12707694881749), ((0, 2), 161.95012224782622), ((0, 3), 175.32227710151258), ((1, 2), 152.8775395859053), ((1, 3), 115.57550659269941), ((2, 3), -83.00271992904541)]
            expected_time_differences_vpol  =  [((0, 1), -39.7774564691897), ((0, 2), 103.04747677383648), ((0, 3), 33.348616734485404), ((1, 2), 142.82493324302618), ((1, 3), 73.1260732036751), ((2, 3), -69.69886003935108)]
            max_time_differences_vpol  =  [((0, 1), -116.30631267571034), ((0, 2), 167.91381462435598), ((0, 3), 173.23375803172078), ((1, 2), 148.37193587533608), ((1, 3), 107.24435568878336), ((2, 3), -78.60480502528327)]

            antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=1)
            pulser_location = info.loadPulserLocationsENU()['run1507'] #ENU


        elif site == 2:
            waveform_index_range = (2000,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
            run = 1509
            cfd_thresh = 0.8

            expected_time_differences_physical  =  [((0, 1), -96.21228508039667), ((0, 2), 21.36317970746586), ((0, 3), -56.5419782996255), ((1, 2), 117.57546478786253), ((1, 3), 39.67030678077117), ((2, 3), -77.90515800709136)]
            max_time_differences_physical  =  [((0, 1), -129.37157110284315), ((0, 2), 152.48216718324971), ((0, 3), -184.0463943150346), ((1, 2), 139.82851662731397), ((1, 3), 104.08254793314117), ((2, 3), -81.41340851163496)]
            expected_time_differences_hpol  =  [((0, 1), -87.02691548238317), ((0, 2), 34.2401194012441), ((0, 3), -47.2064288815136), ((1, 2), 121.26703488362728), ((1, 3), 39.82048660086957), ((2, 3), -81.4465482827577)]
            max_time_differences_hpol  =  [((0, 1), -126.12707694881749), ((0, 2), 161.95012224782622), ((0, 3), -175.32227710151258), ((1, 2), 152.8775395859053), ((1, 3), 115.57550659269941), ((2, 3), -83.00271992904541)]
            expected_time_differences_vpol  =  [((0, 1), -82.68067312409039), ((0, 2), 38.11479814558243), ((0, 3), -36.25022423298833), ((1, 2), 120.79547126967282), ((1, 3), 46.43044889110206), ((2, 3), -74.36502237857076)]
            max_time_differences_vpol  =  [((0, 1), -116.30631267571034), ((0, 2), 167.91381462435598), ((0, 3), -173.23375803172078), ((1, 2), 148.37193587533608), ((1, 3), 107.24435568878336), ((2, 3), -78.60480502528327)]


            antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=1)
            pulser_location = info.loadPulserLocationsENU()['run1509'] #ENU

        elif site == 3:
            waveform_index_range = (1250,2000) #Looking at the later bit of the waveform only, 10000 will cap off.  
            run = 1511
            cfd_thresh = 0.8

            expected_time_differences_physical  =  [((0, 1), -103.1812449168724), ((0, 2), -142.99918760162836), ((0, 3), -183.12401361615616), ((1, 2), -39.81794268475596), ((1, 3), -79.94276869928376), ((2, 3), -40.1248260145278)]
            max_time_differences_physical  =  [((0, 1), -129.37157110284315), ((0, 2), -152.48216718324971), ((0, 3), -184.0463943150346), ((1, 2), -139.82851662731397), ((1, 3), -104.08254793314117), ((2, 3), -81.41340851163496)]
            expected_time_differences_hpol  =  [((0, 1), -101.15104066991898), ((0, 2), -147.48113054454006), ((0, 3), -173.9731100821191), ((1, 2), -46.33008987462108), ((1, 3), -72.82206941220011), ((2, 3), -26.491979537579027)]
            max_time_differences_hpol  =  [((0, 1), -126.12707694881749), ((0, 2), -161.95012224782622), ((0, 3), -175.32227710151258), ((1, 2), -152.8775395859053), ((1, 3), -115.57550659269941), ((2, 3), -83.00271992904541)]
            expected_time_differences_vpol  =  [((0, 1), -99.18571126908932), ((0, 2), -151.7951737712349), ((0, 3), -171.08543756381596), ((1, 2), -52.60946250214556), ((1, 3), -71.89972629472663), ((2, 3), -19.290263792581072)]
            max_time_differences_vpol  =  [((0, 1), -116.30631267571034), ((0, 2), -167.91381462435598), ((0, 3), -173.23375803172078), ((1, 2), -148.37193587533608), ((1, 3), -107.24435568878336), ((2, 3), -78.60480502528327)]


            antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=1)
            pulser_location = info.loadPulserLocationsENU()['run1511'] #ENU

        #Filter settings
        final_corr_length = 2**17 #Should be a factor of 2 for fastest performance
        crit_freq_low_pass_MHz = 70 #This new pulser seems to peak in the region of 85 MHz or so
        crit_freq_high_pass_MHz = None#20
        low_pass_filter_order = 12
        high_pass_filter_order = None#8
        use_filter = True
        plot_filters= True

        known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
        eventids = {}
        eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])
        eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])
        all_eventids = numpy.sort(numpy.append(eventids['hpol'],eventids['vpol']))

        hpol_eventids_cut = numpy.isin(all_eventids,eventids['hpol'])
        vpol_eventids_cut = numpy.isin(all_eventids,eventids['vpol'])

        reader = Reader(datapath,run)
        tct = TemplateCompareTool(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=plot_filters)

        #group_delay_freqs, group_delays, weighted_group_delays = tct.calculateGroupDelaysFromEvent( eventids['hpol'][0], apply_filter=False, plot=True,event_type='hpol')
        
        times, hpol_waveforms = tct.averageAlignedSignalsPerChannel(eventids['hpol'], align_method=0, template_eventid=eventids['hpol'][0], plot=False,event_type='hpol')
        times, vpol_waveforms = tct.averageAlignedSignalsPerChannel(eventids['vpol'], align_method=0, template_eventid=eventids['vpol'][0], plot=False,event_type='vpol')

        averaged_waveforms = numpy.zeros_like(hpol_waveforms)
        for channel in range(8):
            if channel%2 == 0:
                averaged_waveforms[channel] = hpol_waveforms[channel]
            elif channel%2 == 1:
                averaged_waveforms[channel] = vpol_waveforms[channel]

        fig = plt.figure()
        fig.canvas.set_window_title('Hpol and Vpol Average Waveforms')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.xlabel('t (ns)')
        plt.ylabel('Adu')
        for channel in range(8):
            plt.plot(times, averaged_waveforms[channel],alpha=0.7,label=str(channel))
        plt.legend()

        group_delay_freqs, group_delays, weighted_group_delays = tct.calculateGroupDelays(times, averaged_waveforms, plot=True,event_type=None,group_delay_band=(45e6,80e6))

        fig = plt.figure()
        fig.canvas.set_window_title('Group Delay Rolled Hpol and Vpol Average Waveforms')
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.xlabel('t (ns)')
        plt.ylabel('Adu')
        for channel in range(8):
            roll = -int((weighted_group_delays[channel] - min(weighted_group_delays))/(times[1]-times[0]))
            plt.plot(times, numpy.roll(averaged_waveforms[channel],roll),alpha=0.7,label=str(channel))
        plt.legend()

        tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False)
        corr_time_shifts, max_corrs, pairs, corrs = tdc.calculateTimeDelays(numpy.fft.rfft(averaged_waveforms,axis=1), averaged_waveforms, return_full_corrs=True, align_method=0)

        rolls = []
        for index, wf in enumerate(averaged_waveforms):
            rolls.append(- min(numpy.where(wf > max(wf*cfd_thresh))[0]))
        rolls = numpy.array(rolls) - max(rolls) 

        fig = plt.figure()
        fig.canvas.set_window_title('%0.2f Cfd Rolled Hpol and Vpol Average Waveforms'%cfd_thresh)
        plt.subplot(2,1,1)
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.xlabel('t (ns)')
        plt.ylabel('Normalized to max() = 1')
        plt.subplot(2,1,2)
        plt.minorticks_on()
        plt.grid(b=True, which='major', color='k', linestyle='-')
        plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        plt.xlabel('t (ns)')
        plt.ylabel('Normalized to max() = 1')

        for channel in range(8):
            plt.subplot(2,1,channel%2 + 1)
            roll = rolls[channel]
            plt.plot(times, numpy.roll(averaged_waveforms[channel],roll)/max(averaged_waveforms[channel]),alpha=0.7,label=str(channel))
            plt.legend()



        '''
        So:
        Find average waveform for each signal using the above code (in some other script).  The using that
        apply the group delay code (below), which will need to be generalized to work for given waveforms
        rather than just eventid.  There should be two: calculateGroupDelays and calculateGroupDelaysForEvent,
        the latter just calls the former for with a particular events waveforms.  This way it can easily be
        called for, but is more generically available.  

        Output times with averaged waveforms for input into group delay calculator
        '''
    except Exception as e:
        print('Error in FFTPrepper.__main__()')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
