'''
Obtains eventids from the data slicer, and then saves them as a csv. 
'''

#General Imports
import numpy
import itertools
import os
import sys
import csv
import scipy
import scipy.interpolate
import pymap3d as pm
import inspect

#Personal Imports
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.info as info
from tools.data_slicer import dataSlicerSingleRun

#Plotting Imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


#Settings
from pprint import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()
datapath = os.environ['BEACON_DATA']

if __name__ == '__main__':
    try:
        output_directory = os.environ['BEACON_ANALYSIS_DIR'] + 'data/eventids/'
        run = 1650
        use_sources = ['East Dyer Substation','Goldfield KGFN-FM','Tonopah KTPH','Solar Plant','Silver Peak Substation']

        #I think adding an absolute time offset for each antenna and letting that vary could be interesting.  It could be used to adjust the cable delays.
        print('Potential RFI Source Locations.')
        sources_ENU, data_slicer_cut_dict = info.loadValleySourcesENU()

        for key in list(sources_ENU.keys()):
            if not(key in use_sources):
                del sources_ENU[key]
                del data_slicer_cut_dict[key]


        print('Source Directions based on ENU, sorted North to South are:')

        keys = list(sources_ENU.keys())
        azimuths = []

        for source_key in keys:
            azimuths.append(numpy.rad2deg(numpy.arctan2(sources_ENU[source_key][1],sources_ENU[source_key][0])))

        sort_cut = numpy.argsort(azimuths)[::-1]
        for index in sort_cut:
            print('%s : %0.3f'%(keys[index], azimuths[index]))


        reader = Reader(datapath,run)
        impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        map_direction_dset_key = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0'
        ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                    curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                    impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                    p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

        ds.addROI('Simple Template V > 0.7',{'cr_template_search_v':[0.7,1.0]})# Adding 2 ROI in different rows and appending as below allows for "OR" instead of "AND"
        ds.addROI('Simple Template H > 0.7',{'cr_template_search_h':[0.7,1.0]})
        #Done for OR condition
        _eventids = numpy.sort(numpy.unique(numpy.append(ds.getCutsFromROI('Simple Template H > 0.7',load=False,save=False),ds.getCutsFromROI('Simple Template V > 0.7',load=False,save=False))))

        eventid_dict = {}
        for index in sort_cut:
            source_key = keys[index]
            azimuth = azimuths[index]
            cut_dict = data_slicer_cut_dict[source_key]
            ds.addROI(source_key,cut_dict)
            roi_eventids = numpy.intersect1d(ds.getCutsFromROI(source_key),_eventids)
            numpy.savetxt('./run%i_roi_eventids_%s.csv'%(run,source_key.replace(' ','-') + '_rough_estimate_azimuth_from_east_%0.1fdeg'%azimuth),roi_eventids) 

    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






