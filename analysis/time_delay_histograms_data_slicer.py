'''
This is a script uses the dataSlicer class to determine if cutting on time delays is more reasonable (due to poor 
calibration) than cutting on correlation maps.  
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv
import h5py

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info
from tools.data_slicer import dataSlicerSingleRun
from tools.fftmath import FFTPrepper
from tools.correlator import Correlator
from tools.data_handler import createFile

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
plt.ion()

datapath = os.environ['BEACON_DATA']



if __name__=="__main__":
    plt.close('all')
    trigger_types = [2]#[2]
    if len(sys.argv) >= 2:
        run = int(sys.argv[1])
        if len(sys.argv) == 3:
            trigger_types = [int(sys.argv[1])]
    else:
        run = 1650

    datapath = os.environ['BEACON_DATA']

    impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1'#'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1'

    # 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1'
    # 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1'
    # 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0'

    crit_freq_low_pass_MHz = None#100 #This new pulser seems to peak in the region of 85 MHz or so
    low_pass_filter_order = None#8

    crit_freq_high_pass_MHz = None
    high_pass_filter_order = None

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.03
    sine_subtract_max_freq_GHz = 0.09
    sine_subtract_percent = 0.03

    hilbert=False
    final_corr_length = 2**10

    db_subset_plot_ranges = []#[[0,30],[30,40],[40,50]] #Used as bin edges.  
    plot_maps = True

    subset_cm = plt.cm.get_cmap('autumn', 10)
    subset_colors = subset_cm(numpy.linspace(0, 1, len(db_subset_plot_ranges)))[0:len(db_subset_plot_ranges)]

    try:
        run = int(run)

        reader = Reader(datapath,run)
        
        try:
            print(reader.status())
        except Exception as e:
            print('Status Tree not present.  Returning Error.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(1)
        
        filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
        
        if filename is not None:
            with h5py.File(filename, 'r') as file:
                eventids = file['eventids'][...]

                trigger_type_cut = numpy.isin(file['trigger_type'][...], trigger_types)

                dsets = list(file.keys()) #Existing datasets

                print('Time delay keys')
                print(list(file['time_delays'].keys()))

                print('Impulsivity keys')
                print(list(file['impulsivity'].keys()))

                print('Map keys')
                print(list(file['map_direction'].keys()))

                if not numpy.isin('cw',dsets):
                    print('cw dataset does not exist for this run.')
                else:
                    cw_dsets = list(file['cw'].keys())
                    print(list(file['cw'].attrs))
                    prep = FFTPrepper(reader, final_corr_length=int(file['cw'].attrs['final_corr_length']), crit_freq_low_pass_MHz=float(file['cw'].attrs['crit_freq_low_pass_MHz']), crit_freq_high_pass_MHz=float(file['cw'].attrs['crit_freq_high_pass_MHz']), low_pass_filter_order=float(file['cw'].attrs['low_pass_filter_order']), high_pass_filter_order=float(file['cw'].attrs['high_pass_filter_order']), waveform_index_range=(None,None), plot_filters=False)
                    prep.addSineSubtract(file['cw'].attrs['sine_subtract_min_freq_GHz'], file['cw'].attrs['sine_subtract_max_freq_GHz'], file['cw'].attrs['sine_subtract_percent'], max_failed_iterations=3, verbose=False, plot=False)
        

                    #Add attributes for future replicability. 
                    
                    raw_freqs = prep.rfftWrapper(prep.t(), numpy.ones_like(prep.t()))[0]
                    df = raw_freqs[1] - raw_freqs[0]
                    freq_bins = (numpy.append(raw_freqs,raw_freqs[-1]+df) - df/2)/1e6 #MHz

                    freq_hz = file['cw']['freq_hz'][...][trigger_type_cut]
                    linear_magnitude = file['cw']['linear_magnitude'][...][trigger_type_cut]
                    binary_cw_cut = file['cw']['has_cw'][...][trigger_type_cut]
                    if not numpy.isin('dbish',cw_dsets):
                        dbish = 10.0*numpy.log10( linear_magnitude[binary_cw_cut]**2 / len(prep.t()))
                    else:
                        dbish = file['cw']['dbish'][...][trigger_type_cut][binary_cw_cut]
                    if plot_maps:
                        cor = Correlator(reader,  upsample=2**16, n_phi=720, n_theta=720, waveform_index_range=(None,None),crit_freq_low_pass_MHz=float(file['cw'].attrs['crit_freq_low_pass_MHz']), crit_freq_high_pass_MHz=float(file['cw'].attrs['crit_freq_high_pass_MHz']), low_pass_filter_order=float(file['cw'].attrs['low_pass_filter_order']), high_pass_filter_order=float(file['cw'].attrs['high_pass_filter_order']), plot_filter=False,apply_phase_response=True, tukey=False, sine_subtract=True)
                        cor.prep.addSineSubtract(file['cw'].attrs['sine_subtract_min_freq_GHz'], file['cw'].attrs['sine_subtract_max_freq_GHz'], file['cw'].attrs['sine_subtract_percent'], max_failed_iterations=3, verbose=False, plot=False)

                    sine_subtract_min_freq_MHz = 1000*float(file['cw'].attrs['sine_subtract_min_freq_GHz'])
                    sine_subtract_max_freq_MHz = 1000*float(file['cw'].attrs['sine_subtract_max_freq_GHz'])

                file.close()
            
            
            ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                    curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

            # ds.addROI('48 MHz',{'cw_freq_Mhz':[47,49]})
            # ds.addROI('42 MHz',{'cw_freq_Mhz':[41,43]})
            # ds.addROI('52 MHz',{'cw_freq_Mhz':[51,53]})
            # ds.addROI('88 MHz',{'cw_freq_Mhz':[88,90]})

            if True:
                #if trigger_types == [2]:
                #This set of plots aims to compare high correlation with template events across a few plots.  
                ds.addROI('Simple Template V > 0.7',{'cr_template_search_v':[0.7,1.0]})# Adding 2 ROI in different rows and appending as below allows for "OR" instead of "AND"
                ds.addROI('Simple Template H > 0.7',{'cr_template_search_h':[0.7,1.0]})
                #Done for OR condition
                _eventids = numpy.sort(numpy.unique(numpy.append(ds.getCutsFromROI('Simple Template H > 0.7',load=False,save=False),ds.getCutsFromROI('Simple Template V > 0.7',load=False,save=False))))
                #_eventids = eventids = ds.getEventidsFromTriggerType()#numpy.arange(reader.N())
                #Now that eventids are gotten from initial cuts, I want to clear out ROI such that they are not plotted
                #on the below plots (all plotted events will be in these ROI so I don't need to print again.)
                
                # ds.addROI('High SNR',{'snr_h':[21,22.5]})
                # _eventids = ds.getCutsFromROI('High SNR',load=False,save=False)#numpy.sort(numpy.unique(numpy.append(ds.getCutsFromROI('High SNR',load=False,save=False))))
                
                ds.resetAllROI()
                
                #Add actual ROI
                #ds.addROI('Biggest Cluster',{'phi_best_h':[30,35],'elevation_best_h':[-27,-10.0]})
                # ds.addROI('Biggest Cluster TD',{'time_delay_0subtract3_h':[-180,-165],'time_delay_1subtract2_h':[22,30]})
                # ds.addROI('Lowest',{'phi_best_h':[44,50],'elevation_best_h':[-38,-29]})
                # ds.addROI('Middle',{'phi_best_h':[36,42.5],'elevation_best_h':[-38,-25]})
                # ds.addROI('Small Patch',{'phi_best_h':[-32,-26],'elevation_best_h':[-43,-38]})
                # ds.addROI('Array Plane Patch A',{'phi_best_h':[8,12],'elevation_best_h':[-21,-18]})
                # ds.addROI('Array Plane Patch B',{'phi_best_h':[-12,6],'elevation_best_h':[-20,-15]})
                # ds.addROI('Array Plane Patch C',{'phi_best_h':[-21,-13],'elevation_best_h':[-18,-14.5]})

                use_sources = ['East Dyer Substation','Goldfield KGFN-FM','Tonopah KTPH','Solar Plant','Silver Peak Substation']
                sources_ENU, data_slicer_cut_dict = info.loadValleySourcesENU()
                for source_key, cut_dict in data_slicer_cut_dict.items():
                    if not(source_key in use_sources):
                        continue #Skipping calculating that one.
                    else:
                        ds.addROI(source_key,cut_dict)

                # ds.addROI('Solar Plant Substation',{'time_delay_0subtract1_h':[-127,-123],'time_delay_0subtract2_h':[-127,-123.5]})
                # # 'Solar Plant Substation'
                # # Some crosspol. definitely mostly hpol
                # ds.addROI('Dyer or Tonopah',{'time_delay_0subtract1_h':[-135,-131],'time_delay_0subtract2_h':[-111,-105]}) #I think tonopah
                # # 'Dyer or Tonopah'
                # # Very crosspol, good for both calibrations
                # ds.addROI('Beatty Airport VORTAC',{'time_delay_0subtract1_h':[-124.5,-121],'time_delay_0subtract2_h':[22.5,28.5]})
                # # 'Beatty Airport VORTAC'
                # # Should work for vpol
                # ds.addROI('Silver Peak or Distant Substation',{'time_delay_0subtract1_h':[-140.5,-137],'time_delay_0subtract2_h':[-90,-83.5],'time_delay_0subtract3_h':[-167,-161],'time_delay_1subtract2_h':[46,55]})
                # # 'Silver Peak or Distant Substation'
                # # Basically no vpol
                # ds.addROI('Palmetto Tower',{'time_delay_0subtract1_h':[-138,-131.7],'time_delay_0subtract2_h':[-7,-1]})
                # # 'Palmetto Tower'
                # # Cross pol, not the most impulsive
                # ds.addROI('Goldfield Radio',{'hilbert_phi_best_h':[6,12],'time_delay_0subtract1_h':[-143,-139]})
                # ds.addROI('Cedar Peak',{'time_delay_0subtract1_h':[-143,-140],'time_delay_0subtract2_h':[-60.1,-57.4]})
                # # 'Goldfield Radio'
                # # Impulsive cross pol, better for hpol though
                

                # ds.addROI('Goldfield Radio',{'phi_best_h':[8.4,11.7],'elevation_best_h':[-65,-63]})
                if False:
                    try:
                        ds.addROI('A',{'hilbert_phi_best_h':[-20,-13],'time_delay_0subtract1_h':[-125,-120]})
                        ds.addROI('B',{'hilbert_phi_best_h':[-11,-5],'time_delay_0subtract1_h':[-135,-132]})
                        ds.addROI('C',{'hilbert_phi_best_h':[-5,1],'time_delay_0subtract1_h':[-143,-137]})
                        ds.addROI('D',{'hilbert_phi_best_h':[6,12],'time_delay_0subtract1_h':[-143,-139]})
                        ds.addROI('E',{'hilbert_phi_best_h':[18,23],'time_delay_0subtract1_h':[-141,-138]})
                        ds.addROI('F',{'hilbert_phi_best_h':[30,36],'time_delay_0subtract1_h':[-135,-132]})
                        ds.addROI('G',{'hilbert_phi_best_h':[36,42],'time_delay_0subtract1_h':[-132,-129]})
                        ds.addROI('H',{'hilbert_phi_best_h':[44,48],'time_delay_0subtract1_h':[-128,-123]})
                    except:
                        print('Using hilbert ROI failed, using normal map ROI')
                        ds.addROI('A',{'hilbert_phi_best_h':[-20,-13],'time_delay_0subtract1_h':[-125,-120]})
                        ds.addROI('B',{'hilbert_phi_best_h':[-11,-5],'time_delay_0subtract1_h':[-135,-132]})
                        ds.addROI('C',{'hilbert_phi_best_h':[-5,1],'time_delay_0subtract1_h':[-143,-137]})
                        ds.addROI('D',{'hilbert_phi_best_h':[6,12],'time_delay_0subtract1_h':[-143,-139]})
                        ds.addROI('E',{'hilbert_phi_best_h':[18,23],'time_delay_0subtract1_h':[-141,-138]})
                        ds.addROI('F',{'hilbert_phi_best_h':[30,36],'time_delay_0subtract1_h':[-135,-132]})
                        ds.addROI('G',{'hilbert_phi_best_h':[36,42],'time_delay_0subtract1_h':[-132,-129]})
                        ds.addROI('H',{'hilbert_phi_best_h':[44,48],'time_delay_0subtract1_h':[-128,-123]})

                plot_param_pairs = [['cr_template_search_h','cr_template_search_v'],['hilbert_phi_best_h','hilbert_elevation_best_h'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v'],['phi_best_h','time_delay_0subtract1_h'],['time_delay_0subtract1_h','time_delay_0subtract2_h'],['time_delay_0subtract3_h','time_delay_1subtract2_h']]
                for key_x, key_y in plot_param_pairs:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    if 'cr_template_search' in key_x:
                        fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                        fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=_eventids, cmap='coolwarm', include_roi=True)
                    else:
                        fig, ax = ds.plotROI2dHist(key_x, key_y, eventids=_eventids, cmap='coolwarm', include_roi=True)

                all_figs = []
                all_axs = []
                for roi_key in list(ds.roi.keys()):
                    roi_eventids = numpy.intersect1d(ds.getCutsFromROI(roi_key),_eventids)
                    roi_impulsivity = ds.getDataFromParam(roi_eventids,'impulsivity_h')
                    roi_impulsivity_sort = numpy.argsort(roi_impulsivity)
                    if len(roi_eventids) == 0:
                        print('%s:  No events in both ROI and predefined cuts.'%(roi_key))
                    else:
                        #eventid = numpy.random.choice(roi_eventids)
                        eventid = roi_eventids[roi_impulsivity_sort[-1]] #plot highest impulsivity in cut
                        #fig, ax = prep.plotEvent(eventid, channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=False, sine_subtract=False, apply_tukey=None)
                        #ax.set_title('%s: eventid = '%(roi_key,eventid))
                        fig, ax = prep.plotEvent(eventid, channels=[0,1,2,3,4,5,6,7], apply_filter=True, hilbert=False, sine_subtract=True, apply_tukey=None,additional_title_text=roi_key)

                        if plot_maps:
                            try:
                                #Calculate the predicted az el
                                distance_m = numpy.linalg.norm(sources_ENU[roi_key])
                                zenith_deg = numpy.rad2deg(numpy.arccos(sources_ENU[roi_key][2]/distance_m))
                                elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(sources_ENU[roi_key][2]/distance_m))
                                azimuth_deg = numpy.rad2deg(numpy.arctan2(sources_ENU[roi_key][1],sources_ENU[roi_key][0]))
                                include_baselines = numpy.array([0,1,2,3,4,5])

                                # mean_corr_values, fig, ax = cor.map(eventid, 'hpol', include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=True, zenith_cut_array_plane=None, interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg)#, zenith_cut_ENU=[90,180]
                                # all_figs.append(fig)
                                # all_axs.append(ax)

                                mean_corr_values, fig, ax = cor.map(eventid, 'hpol', include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=False, zenith_cut_array_plane=None, interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg)#, zenith_cut_ENU=[90,180]
                                all_figs.append(fig)
                                all_axs.append(ax)
                            except:
                                mean_corr_values, fig, ax = cor.map(eventid, 'hpol', include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=True, zenith_cut_array_plane=None, interactive=True)#, zenith_cut_ENU=[90,180]
                                all_figs.append(fig)
                                all_axs.append(ax)

                                mean_corr_values, fig, ax = cor.map(eventid, 'hpol', include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=False, zenith_cut_array_plane=None, interactive=True)#, zenith_cut_ENU=[90,180]
                                all_figs.append(fig)
                                all_axs.append(ax)



        else:
            print('filename is None, indicating empty tree.  Skipping run %i'%run)
    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


