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
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_slicer import dataSlicer
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
    runs = numpy.arange(1643,1646)#numpy.arange(1643,1729)#numpy.arange(1643,1646)
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
        ['phi_best_v_allsky', 'elevation_best_v_allsky'],\
        ['impulsivity_h', 'impulsivity_v'],\
        ['phi_best_h_allsky', 'elevation_best_h_allsky'],\
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

    lognorm = True
    cmap = 'binary'#'YlOrRd'#'binary'#'coolwarm'

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

        min_impulsivity_h = 0.6
        min_impulsivity_v = 0.0


        ds.addROI('above horizon',{'elevation_best_h_allsky':[0,90], 'impulsivity_h': [min_impulsivity_h,1],'impulsivity_v': [min_impulsivity_v,1]})
        ds.addROI('below horizon',{'elevation_best_h_allsky':[-90,numpy.nextafter(0.0,-90)], 'impulsivity_h': [min_impulsivity_h,1],'impulsivity_v': [min_impulsivity_v,1]}) #nextafter just makes this 0 but not including 0 basically
        eventids_dict = ds.getCutsFromROI('above horizon')
        initial_count = len(ds.concatenateParamDict(eventids_dict))
        background_eventids_dict = ds.getCutsFromROI('below horizon')

        ds.resetAllROI()#Already selected the subset of events, don't need to continue to consider it an "ROI" when passing eventids_dict

        
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

        if False:
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

        if False:
            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=passed_eventids_dict,include_roi=len(list(ds.roi.keys()))!=0, lognorm=lognorm)
                fig.set_size_inches(figsize[0], figsize[1])
                plt.tight_layout()

        else:
            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None,include_roi=len(list(ds.roi.keys()))!=0, lognorm=lognorm, mask_top_N_bins=0, fill_value=0)
                fig.set_size_inches(figsize[0], figsize[1])
                plt.tight_layout()

                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None,include_roi=len(list(ds.roi.keys()))!=0, lognorm=lognorm, mask_top_N_bins=1000, fill_value=0)
                fig.set_size_inches(figsize[0], figsize[1])
                plt.tight_layout()

    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


