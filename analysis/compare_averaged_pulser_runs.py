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

        all_averaged_waveforms = {}

        for site in [1,2,3]:
            if site == 1:
                waveform_index_range = (1500,None) #Looking at the later bit of the waveform only, 10000 will cap off.  
                run = 1507
                antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=1)
                pulser_location = info.loadPulserLocationsENU()['run1507'] #ENU

            elif site == 2:
                waveform_index_range = (2000,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
                run = 1509

                antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=1)
                pulser_location = info.loadPulserLocationsENU()['run1509'] #ENU
            elif site == 3:
                waveform_index_range = (1250,2000) #Looking at the later bit of the waveform only, 10000 will cap off.  
                run = 1511

                antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=1)
                pulser_location = info.loadPulserLocationsENU()['run1511'] #ENU

            #Filter settings
            final_corr_length = 2**17 #Should be a factor of 2 for fastest performance
            
            crit_freq_low_pass_MHz = None#70 #This new pulser seems to peak in the region of 85 MHz or so
            low_pass_filter_order = None#24
            
            crit_freq_high_pass_MHz = None#35
            high_pass_filter_order = None#4
            
            use_filter = True
            plot_filters= True

            known_pulser_ids = info.loadPulserEventids(remove_ignored=True)

            #Prepare eventids

            eventids = {}
            eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])
            eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])
            all_eventids = numpy.sort(numpy.append(eventids['hpol'],eventids['vpol']))

            hpol_eventids_cut = numpy.isin(all_eventids,eventids['hpol'])
            vpol_eventids_cut = numpy.isin(all_eventids,eventids['vpol'])


            #Set up tempalte compare tool used for making averaged waveforms for first pass alignment. 
            reader = Reader(datapath,run)
            tct = TemplateCompareTool(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=site==1)

            #First pass alignment to make templates.  
            times, _hpol_waveforms = tct.averageAlignedSignalsPerChannel(eventids['hpol'], align_method=0, template_eventid=eventids['hpol'][0], plot=False,event_type='hpol')
            times, _vpol_waveforms = tct.averageAlignedSignalsPerChannel(eventids['vpol'], align_method=0, template_eventid=eventids['vpol'][0], plot=False,event_type='vpol')

            upsample_factor = 1
            #Upsample so frequency spectra has more points in region I care about. 
            hpol_waveforms = numpy.zeros((numpy.shape(_hpol_waveforms)[0],numpy.shape(_hpol_waveforms)[1]*upsample_factor))
            vpol_waveforms = numpy.zeros((numpy.shape(_vpol_waveforms)[0],numpy.shape(_vpol_waveforms)[1]*upsample_factor))

            for channel in range(8):
                hpol_waveforms[channel][0:numpy.shape(_hpol_waveforms)[1]] = _hpol_waveforms[channel]
                vpol_waveforms[channel][0:numpy.shape(_vpol_waveforms)[1]] = _vpol_waveforms[channel]

            times = numpy.arange(numpy.shape(vpol_waveforms)[1]) * numpy.diff(times)[0]


            averaged_waveforms = numpy.zeros_like(hpol_waveforms)
            resampled_averaged_waveforms = numpy.zeros((8,len(tct.waveform_times_corr)))
            for channel in range(8):
                if channel%2 == 0:
                    averaged_waveforms[channel] = hpol_waveforms[channel]
                elif channel%2 == 1:
                    averaged_waveforms[channel] = vpol_waveforms[channel]
                #Resampling averaged waveforms to be more compatible with cross correlation framework. 
                all_averaged_waveforms[run] = averaged_waveforms
                resampled_averaged_waveforms[channel] = scipy.interpolate.interp1d(times,averaged_waveforms[channel],kind='cubic',bounds_error=False,fill_value=0)(tct.waveform_times_corr)

        #Plot the spectra of each averaged waveform.

        for antenna_index in range(4):
            h = int(2*antenna_index)
            v = int(h+1)


            fig = plt.figure(figsize=(10,5))
            fig.canvas.set_window_title('Antenna %i Average Spectra'%antenna_index)
            plt.suptitle('Antenna %i Average Spectra'%antenna_index)
            plt.subplot(2,1,1)
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            plt.xlabel('Freqs')
            plt.ylabel('Hpol dBish')

            plt.subplot(2,1,2)
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            plt.xlabel('Freqs (MHz)')
            plt.ylabel('Vpol dBish')

            for _site , run in enumerate([1507,1509,1511]):
                site = _site+1
                freqs, spec_dbish_h, spec_h = tct.rfftWrapper(times, all_averaged_waveforms[run][h])
                freqs, spec_dbish_v, spec_v = tct.rfftWrapper(times, all_averaged_waveforms[run][v])
                
                min_freq = 5#MHz
                max_freq = 200#MHz

                if min_freq is not None:
                    f_low_cut = freqs/1e6 > min_freq
                else:
                    f_low_cut = numpy.ones_like(freqs)
                if max_freq is not None:
                    f_high_cut = freqs/1e6 < max_freq
                else:
                    f_high_cut = numpy.ones_like(freqs)

                f_cut = numpy.logical_and(f_low_cut,f_high_cut)

                plt.subplot(2,1,1)
                plt.plot(freqs[f_cut]/1e6,spec_dbish_h[f_cut],label=str(site))
                plt.ylim(-60,60)
                plt.legend()

                plt.subplot(2,1,2)
                plt.plot(freqs[f_cut]/1e6,spec_dbish_v[f_cut],label=str(site))
                plt.ylim(-60,60)
                plt.legend()
            
            #fig.savefig('ant%ispectra-LP%s-%s-HP%s-%s.png'%(antenna_index,str(crit_freq_low_pass_MHz), str(low_pass_filter_order), str(crit_freq_high_pass_MHz), str(high_pass_filter_order)))



    except Exception as e:
        print('Error in FFTPrepper.__main__()')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
