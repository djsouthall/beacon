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
import scipy
import matplotlib
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

datapath = os.environ['BEACON_DATA']


if __name__=="__main__":
    plt.close('all')
    interactive_mode = True
    if interactive_mode == False:
        outpath = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'figures', 'parameter_plots_' + str(datetime.datetime.now()).replace(' ', '_').replace('.','p').replace(':','-'))
        matplotlib.use('Agg')
        os.mkdir(outpath)
        outpath_made = True
    else:
        outpath = None
        plt.ion()

    #Main Control Parameters
    runs = numpy.arange(1643,1645)#1729)
    #runs = runs[runs != 1663]
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
    #     ['phi_best_h', 'elevation_best_h'],\
    #     ['phi_best_v', 'elevation_best_v'],\
    #     ]

    # plot_param_pairs = [\
    #     ['impulsivity_h', 'impulsivity_v'],\
    #     ['cr_template_search_h', 'cr_template_search_v'],\
    #     ['std_h', 'std_v'],\
    #     ['p2p_h', 'p2p_v'],\
    #     ['snr_h', 'snr_v'],\
    #     ['cw_freq_Mhz','cw_dbish'],\
    #     ['phi_best_h','cw_freq_Mhz'],\
    #     ['phi_best_h','cw_dbish'],\
    #     ['phi_best_h', 'elevation_best_h'],\
    #     ['phi_best_v', 'elevation_best_v'],\
    #     ]

    plot_param_pairs = [\
        ['impulsivity_h', 'impulsivity_v'],\
        ['cr_template_search_h', 'cr_template_search_v'],\
        ['std_h', 'std_v'],\
        ['p2p_h', 'p2p_v'],\
        ['snr_h', 'snr_v'],\
        ['time_delay_0subtract1_h', 'time_delay_0subtract2_h'],\
        ['time_delay_0subtract3_h', 'time_delay_1subtract2_h'],\
        ['time_delay_1subtract3_h', 'time_delay_2subtract3_h'],\
        ['time_delay_0subtract1_v', 'time_delay_0subtract2_v'],\
        ['time_delay_0subtract3_v', 'time_delay_1subtract2_v'],\
        ['time_delay_1subtract3_v', 'time_delay_2subtract3_v'],\
        ['cw_freq_Mhz','cw_dbish'],\
        ['phi_best_h','cw_freq_Mhz'],\
        ['phi_best_h','cw_dbish'],\
        ['phi_best_h_allsky', 'elevation_best_h_allsky'],\
        ['phi_best_h_abovehorizon', 'elevation_best_h_abovehorizon'],\
        ['phi_best_h_belowhorizon', 'elevation_best_h_belowhorizon'],\
        ]

    # plot_param_pairs = [\
    #     ['impulsivity_h', 'impulsivity_v'],\
    #     ['std_h', 'std_v'],\
    #     ['p2p_h', 'p2p_v'],\
    #     ['phi_best_h_allsky', 'elevation_best_h_allsky'],\
    #     ['phi_best_h_abovehorizon', 'elevation_best_h_abovehorizon'],\
    #     ['phi_best_h_belowhorizon', 'elevation_best_h_belowhorizon'],\
    #     ]

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
    figsize = (8,6)
    dpi = 108*4


    # Other Parameters
    time_delays_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    impulsivity_dset_key = time_delays_dset_key
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_chi2_optimized_deploy_from_rtk-gps-day1-june20-2021.json-n_phi_1440-min_phi_neg180-max_phi_180-n_theta_720-min_theta_0-max_theta_180-scope_allsky'#'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_chi2_optimized_deploy_from_rtk-gps-day1-june20-2021.json-n_phi_1440-min_phi_neg180-max_phi_180-n_theta_720-min_theta_0-max_theta_180-scope_allsky'#'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploycalibration_'#'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_30-n_phi_1440-min_phi_neg180-max_phi_180-n_theta_720-min_theta_0-max_theta_180-scope_allsky'

    trigger_types = [2]

    try:
        n_phi = int(map_direction_dset_key.split('-n_phi_')[-1].split('-')[0])
        range_phi_deg = (float(map_direction_dset_key.split('-min_phi_')[-1].split('-')[0].replace('neg','-')) , float(map_direction_dset_key.split('max_phi_')[-1].split('-')[0].replace('neg','-')))
        n_theta = int(map_direction_dset_key.split('-n_theta_')[-1].split('-')[0])
        range_theta_deg = (float(map_direction_dset_key.split('-min_theta_')[-1].split('-')[0].replace('neg','-')) , float(map_direction_dset_key.split('max_theta_')[-1].split('-')[0].replace('neg','-')))
    except:
        #Something went wrong naming the map so here is the information.
        n_phi       = 1440
        min_phi     = -180
        max_phi     = 180

        n_theta     = 720
        min_theta   = 0
        max_theta   = 180

        range_phi_deg = (min_phi, max_phi)
        range_theta_deg = (min_theta, max_theta)


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

        if False:
            ds.addROI('Above Horizon',{'elevation_best_h_allsky':[0,90]})# , 'cr_template_search_h': [0.7,1]
            #Done for OR condition
            eventids_dict = ds.getCutsFromROI('Above Horizon',load=False,save=False)
        elif True:
            #ds.addROI('RFI Source',{'phi_best_h_allsky':[36,39],'elevation_best_h_allsky':[-12,-6]})# , 'cr_template_search_h': [0.7,1]

            ds.addROI('RFI Source',{'phi_best_h_allsky':[45,47],'elevation_best_h_allsky':[-9,-4],'snr_h':[15,100],'snr_v':[15,100],'time_delay_0subtract1_h':[-127,-123], 'time_delay_0subtract2_h':[-129,-124]})
            #ds.addROI('RFI Source 2',{'std_h':[2,6],'std_v':[6,8]})# , 'cr_template_search_h': [0.7,1]

            print('ROI COLOR IN HEX IS: %s'%str(matplotlib.colors.to_hex(ds.roi_colors[0])))
            eventids_dict = None

            if True:
                n_phi = int(map_direction_dset_key.split('-n_phi_')[-1].split('-')[0])
                range_phi_deg = (float(map_direction_dset_key.split('-min_phi_')[-1].split('-')[0].replace('neg','-')) , float(map_direction_dset_key.split('max_phi_')[-1].split('-')[0].replace('neg','-')))
                n_theta = int(map_direction_dset_key.split('-n_theta_')[-1].split('-')[0])
                range_theta_deg = (float(map_direction_dset_key.split('-min_theta_')[-1].split('-')[0].replace('neg','-')) , float(map_direction_dset_key.split('max_theta_')[-1].split('-')[0].replace('neg','-')))

                thetas_deg = numpy.linspace(min(range_theta_deg),max(range_theta_deg),n_theta) #Zenith angle
                phis_deg = numpy.linspace(min(range_phi_deg),max(range_phi_deg),n_phi) #Azimuth angle

                bin_width_theta = numpy.diff(thetas_deg)[0]
                bin_width_phi = numpy.diff(phis_deg)[0]

                bin_edges_theta = numpy.arange(-1,1 + bin_width_theta,bin_width_theta)
                x_theta = numpy.arange(-1,1 + bin_width_theta/10,bin_width_theta/10)

                bin_edges_phi = numpy.arange(-1,1 + bin_width_phi,bin_width_phi)
                x_phi = numpy.arange(-1,1 + bin_width_phi/10,bin_width_phi/10)


                all_theta_best = ds.concatenateParamDict(ds.getDataFromParam(ds.getCutsFromROI('RFI Source'),'elevation_best_h_allsky'))
                all_phi_best = ds.concatenateParamDict(ds.getDataFromParam(ds.getCutsFromROI('RFI Source'),'phi_best_h_allsky'))

                stacked_variables = numpy.vstack((all_phi_best,all_theta_best)) #x,y
                covariance_matrix = numpy.cov(stacked_variables)
                sig_phi = numpy.sqrt(covariance_matrix[0,0])
                sig_theta = numpy.sqrt(covariance_matrix[1,1])
                rho_phi_theta = covariance_matrix[0,1]/(sig_phi*sig_theta)
                mean_phi = numpy.mean(all_phi_best)
                mean_theta = numpy.mean(all_theta_best)


                fig = plt.figure()
                plt.subplot(1,2,1)
                plt.xlabel('Azimuth Distribution (Degrees)\nCentered on Mean')
                plt.hist(all_phi_best - mean_phi, bins=bin_edges_phi, log=False, edgecolor='black', linewidth=1.0,label='Mean = %0.3f\nSigma = %0.3f'%(mean_phi,sig_phi),density=False)
                plt.plot(x_phi,scipy.stats.norm.pdf(x_phi,0,sig_phi)*bin_width_phi*len(all_phi_best),label='Gaussian Fit')
                plt.xlim(min(all_phi_best - mean_phi) - 1.0,max(all_phi_best - mean_phi) + 1.0)

                plt.legend(loc = 'upper right',fontsize=10)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.subplot(1,2,2)
                plt.xlabel('Elevation Distribution (Degrees)\nCentered on Mean')
                plt.hist(all_theta_best - mean_theta, bins=bin_edges_theta, log=False, edgecolor='black', linewidth=1.0,label='Mean = %0.3f\nSigma = %0.3f'%(mean_theta,sig_theta),density=False)
                plt.plot(x_theta,scipy.stats.norm.pdf(x_theta,0,sig_theta)*bin_width_theta*len(all_theta_best),label='Gaussian Fit')
                plt.xlim(min(all_theta_best - mean_theta) - 1.0,max(all_theta_best - mean_theta) + 1.0)

                plt.legend(loc = 'upper right',fontsize=10)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)




        else:
            eventids_dict = None


        # for key_x, key_y in plot_param_pairs:
        #     print('Generating %s plot'%(key_x + ' vs ' + key_y))
        #     fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=eventids_dict,include_roi=len(list(ds.roi.keys()))!=0, lognorm=lognorm)
        #     fig.set_size_inches(figsize[0], figsize[1])
        #     plt.tight_layout()
        #     if interactive_mode == False:
        #         fig.savefig(os.path.join(outpath,key_x + '-vs-' + key_y + '.png'),dpi=dpi)




    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


