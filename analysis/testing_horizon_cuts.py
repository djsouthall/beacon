'''
This is a script plot the current parameters for the range of runs given, and save them to an output directory in the
given path.  
'''
import os
import sys
import inspect
import warnings
import datetime
import numpy
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.correlator import Correlator
from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

datapath = os.environ['BEACON_DATA']
figsize = (16,9)


if __name__=="__main__":
    plt.close('all')#

    #Main Control Parameters
    runs = numpy.arange(1643,1729)#numpy.arange(1643,1646)
    # plot_param_pairs = [\
    #     ['impulsivity_h', 'impulsivity_v'],\
    #     ['cr_template_search_h', 'cr_template_search_v'],\
    #     ['std_h', 'std_v'],\
    #     ['p2p_h', 'p2p_v'],\
    #     ['snr_h', 'snr_v'],\
    #     ['time_delay_0subtract1_h', 'time_delay_0subtract2_h'],\
    #     ['time_delay_0subtract3_h', 'time_delay_1subtract2_h'],\
    #     ['time_delay_1subtract3_h', 'time_delay_2subtract3_h'],\
    #     ['time_delay_0subtract1_v', 'time_delay_0subtract2_v'],\
    #     ['time_delay_0subtract3_v', 'time_delay_1subtract2_v'],\
    #     ['time_delay_1subtract3_v', 'time_delay_2subtract3_v'],\
    #     ['cw_freq_Mhz','cw_dbish'],\
    #     ['phi_best_h','cw_freq_Mhz'],\
    #     ['phi_best_h','cw_dbish'],\
    #     ['phi_best_h_allsky', 'elevation_best_h_allsky'],\
    #     ['phi_best_v_allsky', 'elevation_best_v_allsky'],\
    #     ]

    plot_param_pairs = [\
        ['impulsivity_h', 'impulsivity_v'],\
        ['phi_best_h_allsky', 'elevation_best_h_allsky'],\
        ['phi_best_v_allsky', 'elevation_best_v_allsky'],\
        ['phi_best_h_belowhorizon', 'elevation_best_h_belowhorizon'],\
        ['phi_best_v_belowhorizon', 'elevation_best_v_belowhorizon'],\
        ['p2p_h', 'p2p_v'],\
        ['cr_template_search_h', 'cr_template_search_v'],\
        ]

    # plot_param_pairs = [\
    #     ['phi_best_h', 'elevation_best_h'],\
    #     ['phi_best_v', 'elevation_best_v'],\
    #     ['phi_best_h_allsky', 'elevation_best_h_allsky'],\
    #     ['phi_best_v_allsky', 'elevation_best_v_allsky'],\
    #     ['phi_best_h_abovehorizon', 'elevation_best_h_abovehorizon'],\
    #     ['phi_best_v_abovehorizon', 'elevation_best_v_abovehorizon'],\
    #     ['phi_best_h_belowhorizon', 'elevation_best_h_belowhorizon'],\
    #     ['phi_best_v_belowhorizon', 'elevation_best_v_belowhorizon'],\
    #     ]


    # Other Parameters
    time_delays_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    impulsivity_dset_key = time_delays_dset_key
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_30-n_phi_1440-min_phi_neg180-max_phi_180-n_theta_720-min_theta_0-max_theta_180-scope_allsky'

    trigger_types = [2]

    n_phi = int(map_direction_dset_key.split('-n_phi_')[-1].split('-')[0])
    range_phi_deg = (float(map_direction_dset_key.split('-min_phi_')[-1].split('-')[0].replace('neg','-')) , float(map_direction_dset_key.split('max_phi_')[-1].split('-')[0].replace('neg','-')))
    n_theta = int(map_direction_dset_key.split('-n_theta_')[-1].split('-')[0])
    range_theta_deg = (float(map_direction_dset_key.split('-min_theta_')[-1].split('-')[0].replace('neg','-')) , float(map_direction_dset_key.split('max_theta_')[-1].split('-')[0].replace('neg','-')))
    upsample = int(map_direction_dset_key.split('-upsample_')[-1].split('-')[0])
    max_method = int(map_direction_dset_key.split('-maxmethod_')[-1].split('-')[0])
    crit_freq_low_pass_MHz = float(map_direction_dset_key.split('LPf_')[-1].split('-')[0])
    low_pass_filter_order = int(map_direction_dset_key.split('-LPo_')[-1].split('-')[0])
    crit_freq_high_pass_MHz = float(map_direction_dset_key.split('-HPf_')[-1].split('-')[0])
    high_pass_filter_order = int(map_direction_dset_key.split('-HPo_')[-1].split('-')[0])


    lognorm = True
    cmap = 'binary'#'YlOrRd'#'binary'#'coolwarm'

    #Correlation Map Settings

    plot_filter=False

    apply_phase_response=True
    sine_subtract = bool(map_direction_dset_key.split('-sinesubtract_')[-1].split('-')[0])
    deploy_index = map_direction_dset_key.replace('deploy_calibration','deploycalibration').split('-deploycalibration_')[-1].split('-')[0]
    if deploy_index.isdigit():
        deploy_index = int(deploy_index)
    sine_subtract_min_freq_GHz = 0.03
    sine_subtract_max_freq_GHz = 0.09
    sine_subtract_percent = 0.05
    try:
        ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, remove_incomplete_runs=True,\
                curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                cr_template_n_bins_h=1000,cr_template_n_bins_v=1000,\
                impulsivity_n_bins_h=1000,impulsivity_n_bins_v=1000,\
                time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-200,max_time_delays_val=200,\
                std_n_bins_h=200,std_n_bins_v=200,max_std_val=None,\
                p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=None,\
                n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg)

        #INITIAL CUTS!  THESE SET THE EVENTS THAT WILL BE CONSIDERED BY FURTHER RFI EVENTS
        min_impulsivity_h = 0.5
        min_impulsivity_v = 0.0

        ds.addROI('above horizon',{'elevation_best_h_allsky':[0,90], 'impulsivity_h': [min_impulsivity_h,1],'impulsivity_v': [min_impulsivity_v,1]})
        ds.addROI('below horizon',{'elevation_best_h_allsky':[-90,numpy.nextafter(0.0,-90)], 'impulsivity_h': [min_impulsivity_h,1],'impulsivity_v': [min_impulsivity_v,1]}) #nextafter just makes this 0 but not including 0 basically
        eventids_dict = ds.getCutsFromROI('above horizon')
        initial_count = len(ds.concatenateParamDict(eventids_dict))
        background_eventids_dict = ds.getCutsFromROI('below horizon')

        ds.resetAllROI()#Already selected the subset of events, don't need to continue to consider it an "ROI" when passing eventids_dict



        if False:       
            #Attempts to generate passed_eventids_dict based on a range of event count thresholds.  
            #counts = ds.get2dHistCounts('phi_best_h_belowhorizon', 'elevation_best_h_belowhorizon', eventids_dict, set_bins=True)
            
            background_counts = ds.get2dHistCounts('phi_best_h_belowhorizon', 'elevation_best_h_belowhorizon', background_eventids_dict, set_bins=True)
            background_percent = background_counts/numpy.sum(background_counts)

            #This all needs to be explored based on cuts in both vpol and hpol, OR at least an hpol impulsivity cut must be made.

            background_cut_threshs = numpy.linspace(0,0.01,101)
            final_counts = []

            for thresh_index, background_cut_thresh in enumerate(background_cut_threshs):
                print(thresh_index/len(background_cut_threshs))
                background_cut = background_counts/numpy.sum(background_counts) < background_cut_thresh  #True if less than this percent of background events are in that bin.  Hard to know what is reasonable without plotting.

                counts = ds.get2dHistCounts('phi_best_h_belowhorizon', 'elevation_best_h_belowhorizon', eventids_dict, set_bins=False)
                final_count = numpy.sum(numpy.multiply(counts,background_cut))
                final_counts.append(final_count)

            final_counts = numpy.asarray(final_counts)

            if True:
                plt.figure()
                plt.plot(background_cut_threshs,final_counts)
                plt.ylabel('Counts Passing Cut')
                plt.xlabel('Bin Prominence Cut Threshold')
                plt.axhline(initial_count,label='Initial Count',linestyle='--',c='k')

            passed_eventids_dict = {}
            if len(background_cut_threshs[final_counts > 1]) > 0:
                background_cut_thresh = background_cut_threshs[final_counts > 1][0]#background_cut_threshs[final_counts > initial_count*.50][0] #Choosing the first cut where 50% of events pass. 
            else:
                background_cut_thresh = background_cut_threshs[-1]
            background_cut = background_counts/numpy.sum(background_counts) < background_cut_thresh  #True if less than this percent of background events are in that bin.  Hard to know what is reasonable without plotting.
            for run in list(eventids_dict.keys()):
                passed_eventids_dict[run] = []
                for eventid in eventids_dict[run]:
                    counts = ds.get2dHistCounts('phi_best_h_belowhorizon', 'elevation_best_h_belowhorizon', {run:[eventid]}, set_bins=False)
                    if int(numpy.sum(numpy.multiply(counts,background_cut))) == 1:
                        passed_eventids_dict[run].append(eventid)
        elif True:
            #This aims to cut out sub horizon RFI using many pointed source cuts.
            #define in hpol, automate duplicate dicts in vpol for OR condition

            rfi_box_index = 0
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[54,61],'elevation_best_h_belowhorizon':[-11,-2]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[46.5,54],'elevation_best_h_belowhorizon':[-6,0.5]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[41.5,45.5],'elevation_best_h_belowhorizon':[-3.5,0.5]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[30.5,35],'elevation_best_h_belowhorizon':[-4,0]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[25,29],'elevation_best_h_belowhorizon':[-2.5,-0.5]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[22.5,24.5],'elevation_best_h_belowhorizon':[-2.5,-0.5]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[16.5,20.5],'elevation_best_h_belowhorizon':[-3.5,-0.5]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[6.5,10.5],'elevation_best_h_belowhorizon':[-6.5,-3.5]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[2,4],'elevation_best_h_belowhorizon':[-7,-4.5]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[-3,0],'elevation_best_h_belowhorizon':[-9,-4]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[-5.5,-4],'elevation_best_h_belowhorizon':[-8,-5.5]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[-13,-10],'elevation_best_h_belowhorizon':[-9.5,-5.5]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[-5.5,-4],'elevation_best_h_belowhorizon':[-8,-5.5]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[-15.5,-12.5],'elevation_best_h_belowhorizon':[-5,-1]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[-17,-15.5],'elevation_best_h_belowhorizon':[-5.5,-2]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[7.5,9],'elevation_best_h_belowhorizon':[-14,-12]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[13,15],'elevation_best_h_belowhorizon':[-23,-21]}); rfi_box_index += 1
            ds.addROI('rfi box %i hpol'%rfi_box_index,{'phi_best_h_belowhorizon':[22,24],'elevation_best_h_belowhorizon':[-10,-7.5]}); rfi_box_index += 1

            roi_dict = dict(ds.roi)
            for cut_key, cut_dict in roi_dict.items():
                cut_key_vpol = cut_key.replace('hpol','vpol')
                cut_dict_vpol = {}
                for sub_dict_key, cut_list in cut_dict.items():
                    cut_dict_vpol[sub_dict_key.replace('_h','_v')] = cut_list
                ds.addROI(cut_key_vpol,cut_dict_vpol)

            rfi_dict = {} #Eventids flagged as RFI
            for key, item in ds.roi.items():
                if 'rfi' in key:
                    rfi_dict_i = ds.getCutsFromROI(key)
                    rfi_dict = ds.returnUniqueEvents(rfi_dict, rfi_dict_i)

            passed_eventids_dict = ds.returnEventsAWithoutB(eventids_dict, rfi_dict)
            passed_eventids_dict = ds.removeEmptyRunsFromDict(passed_eventids_dict)

        else:
            #This applies an extremely generous box cut.
            ds.addROI('rfi box 1',{'phi_best_h_belowhorizon':[-15,60],'elevation_best_h_belowhorizon':[-25,0]})
            ds.addROI('rfi box 2',{'phi_best_v_belowhorizon':[-15,60],'elevation_best_v_belowhorizon':[-25,0]})
            rfi_dict_1 = ds.getCutsFromROI('rfi box 1')
            rfi_dict_2 = ds.getCutsFromROI('rfi box 2')
            rfi_dict = ds.returnUniqueEvents(rfi_dict_1, rfi_dict_2)
            passed_eventids_dict = ds.returnEventsAWithoutB(eventids_dict, rfi_dict)
            passed_eventids_dict = ds.removeEmptyRunsFromDict(passed_eventids_dict)

        passed_count = len(ds.concatenateParamDict(passed_eventids_dict))
        print(passed_eventids_dict)
        print('total passed_count = ',passed_count)

        if False:
            ds.setCurrentPlotBins('elevation_best_h_belowhorizon', 'elevation_best_h_belowhorizon', background_eventids_dict) #to be used for 1d hist below
            label_x = ds.current_label_x
            bin_edges_x = ds.current_bin_edges_x

            below_horizon_secondary_vals = ds.concatenateParamDict(ds.getDataFromParam(eventids_dict, 'elevation_best_h_belowhorizon', verbose=False)) #events that may be above horizon on initial reconstruction, but might be contaminated by below horizon backgrounds
            below_horizon_background_vals = ds.concatenateParamDict(ds.getDataFromParam(background_eventids_dict, 'elevation_best_h_belowhorizon', verbose=False)) #Events that are very likely to be background (below horizon)


            if True:
                plt.figure()
                all_counts, all_bin_edges, patches = plt.hist(below_horizon_background_vals,bins = bin_edges_x,label='background events')
                secondary_counts, secondary_bin_edges, patches = plt.hist(below_horizon_secondary_vals,bins = bin_edges_x,label='above horizon events\n(best below horizon sidelobe)')
            else:
                all_counts, all_bin_edges = numpy.histogram(below_horizon_background_vals,bins = bin_edges_x)
                secondary_counts, secondary_bin_edges = numpy.histogram(below_horizon_background_vals,bins = bin_edges_x)

        if True:
            if passed_count > 0:
                for key_x, key_y in plot_param_pairs:
                    print('Generating %s plot'%(key_x + ' vs ' + key_y))
                    fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=passed_eventids_dict,include_roi=len(list(ds.roi.keys()))!=0, lognorm=lognorm, suppress_legend=True)
                    fig.set_size_inches(figsize[0], figsize[1])
                    plt.tight_layout()
            else:
                print('No events passed the specified cuts.')

        if False:
            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None,include_roi=len(list(ds.roi.keys()))!=0, lognorm=lognorm, mask_top_N_bins=0, fill_value=0, suppress_legend=True)
                fig.set_size_inches(figsize[0], figsize[1])
                plt.tight_layout()

                # fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None,include_roi=len(list(ds.roi.keys()))!=0, lognorm=lognorm, mask_top_N_bins=1000, fill_value=0)
                # fig.set_size_inches(figsize[0], figsize[1])
                # plt.tight_layout()

        print(passed_eventids_dict)
        print('total passed_count = ',passed_count)

        
        if passed_count < 10:
            datapath = os.environ['BEACON_DATA']
            all_figs = []
            all_axs = []
            all_cors = []
            for run, eventids in passed_eventids_dict.items():
                reader = Reader(datapath,run)
                waveform_index_range = (None,None)
                cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, n_theta=n_theta, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=plot_filter,apply_phase_response=apply_phase_response, deploy_index=deploy_index)
                if sine_subtract:
                    cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                for eventid in eventids:
                    for mode in ['hpol','vpol']:
                        mean_corr_values, fig, ax = cor.map(eventid, mode, include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, plot_corr=False, hilbert=False,interactive=True, max_method=0, waveforms=None, verbose=True, mollweide=False, zenith_cut_ENU=None, zenith_cut_array_plane=None, center_dir='E', circle_zenith=None, circle_az=None, time_delay_dict={},window_title=None,add_airplanes=True)
                        all_figs.append(fig)
                        all_axs.append(ax)
                        if False:
                            cor.plotPointingResolution(mode, snr=5, bw=50e6, plot_map=True, mollweide=False,center_dir='E', window_title=None, include_baselines=[0,1,2,3,4,5])                    
                all_cors.append(cor)
        else:
            print('Passed count greater than set limit, maps not plotted.')


    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


