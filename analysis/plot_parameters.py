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
matplotlib.use('Agg')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

datapath = os.environ['BEACON_DATA']

if __name__=="__main__":
    plt.close('all')

    #Main Control Parameters
    runs = numpy.arange(1643,1729)
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
        ['phi_best_h', 'elevation_best_h'],\
        ['phi_best_v', 'elevation_best_v'],\
        ]
    outpath = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'figures', 'parameter_plots_' + str(datetime.datetime.now()).replace(' ', '_').replace('.','p').replace(':','-'))
    figsize = (16,9)
    dpi = 108*10


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
        os.mkdir(outpath)

        ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, remove_incomplete_runs=True,\
                curve_choice=0, trigger_types=trigger_types,included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                cr_template_n_bins_h=1000,cr_template_n_bins_v=1000,\
                impulsivity_n_bins_h=1000,impulsivity_n_bins_v=1000,\
                time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-200,max_time_delays_val=200,\
                std_n_bins_h=200,std_n_bins_v=200,max_std_val=None,\
                p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=None,\
                n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg)

        for key_x, key_y in plot_param_pairs:
            print('Generating %s plot'%(key_x + ' vs ' + key_y))
            fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=None,include_roi=len(list(ds.roi.keys()))!=0, lognorm=lognorm)
            fig.set_size_inches(figsize[0], figsize[1])
            plt.tight_layout()
            fig.savefig(os.path.join(outpath,key_x + '-vs-' + key_y + '.png'),dpi=dpi)


    except Exception as e:
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


