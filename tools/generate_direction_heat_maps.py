#!/usr/bin/env python3
'''
Given a range of runs this well create an event direction heatmap from the stored pointing values for that map.
Then the map will be smoothed out and stored such that it can used to obtain heat values per event.  This will
be done per scope (abovehorizon, belowhorizon, allsky)
'''

import sys
import os
import inspect
import h5py
import copy
from pprint import pprint
import textwrap

import numpy
import scipy
import scipy.signal
import scipy.ndimage

from beacon.tools.data_slicer import dataSlicer

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)


'''
'impulsivity_h','impulsivity_v', 'cr_template_search_h', 'cr_template_search_v', 'std_h', 'std_v', 'p2p_h', 'p2p_v', 'snr_h', 'snr_v',\
'time_delay_0subtract1_h','time_delay_0subtract2_h','time_delay_0subtract3_h','time_delay_1subtract2_h','time_delay_1subtract3_h','time_delay_2subtract3_h',\
'time_delay_0subtract1_v','time_delay_0subtract2_v','time_delay_0subtract3_v','time_delay_1subtract2_v','time_delay_1subtract3_v','time_delay_2subtract3_v',
'cw_present','cw_freq_Mhz','cw_linear_magnitude','cw_dbish','theta_best_h','theta_best_v','elevation_best_h','elevation_best_v','phi_best_h','phi_best_v',\
'calibrated_trigtime','triggered_beams','beam_power','hpol_peak_to_sidelobe','vpol_peak_to_sidelobe','hpol_max_possible_map_value','vpol_max_possible_map_value',\
'map_max_time_delay_0subtract1_h','map_max_time_delay_0subtract2_h','map_max_time_delay_0subtract3_h',\
'map_max_time_delay_1subtract2_h','map_max_time_delay_1subtract3_h','map_max_time_delay_2subtract3_h',\
'map_max_time_delay_0subtract1_v','map_max_time_delay_0subtract2_v','map_max_time_delay_0subtract3_v',\
'map_max_time_delay_1subtract2_v','map_max_time_delay_1subtract3_v','map_max_time_delay_2subtract3_v'
'''




if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length



    if False:
        #Quicker for testing
        runs = numpy.arange(5910,5921)
    else:
        runs = numpy.arange(5733,5974)
        #runs = runs[runs != 5864]

    print("Preparing dataSlicer")

    #Should match the source data resolutions, in corrolator they are handled as centers rather than bin edges, so similarly in dataslicer they will be treated as the center of bins despite that not generally not being how bins are handled in ds. 
    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                    n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)

    if False:
        #Forces both all baseline map to point in box
        ds.addROI('above horizon',{'elevation_best_all':[10,90],'phi_best_all':[-90,90]})
        above_horizon_eventids_dict = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
        ds.addROI('above horizon full',{'elevation_best_all':[10,90],'phi_best_all':[-90,90],'similarity_count_h':[0,10],'similarity_count_v':[0,10],'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,10],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]})
        above_horizon_full_eventids_dict = ds.getCutsFromROI('above horizon full',load=False,save=False,verbose=True, return_successive_cut_counts=False, return_total_cut_counts=False)
        skymap_mode = 'all'
    elif False:
        #Forces both maps to point in box
        ds.addROI('above horizon',{'elevation_best_h':[10,90],'phi_best_h':[-90,90],'elevation_best_v':[10,90],'phi_best_v':[-90,90]})
        above_horizon_eventids_dict = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
        ds.addROI('above horizon full',{'elevation_best_h':[10,90],'phi_best_h':[-90,90],'elevation_best_v':[10,90],'phi_best_v':[-90,90],'similarity_count_h':[0,10],'similarity_count_v':[0,10],'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,10],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]})
        above_horizon_full_eventids_dict = ds.getCutsFromROI('above horizon full',load=False,save=False,verbose=True, return_successive_cut_counts=False, return_total_cut_counts=False)
        skymap_mode = 'h'
    elif True:
        #Forces uses the "best map" method
        ds.addROI('above horizon',{'elevation_best_choice':[10,90],'phi_best_choice':[-90,90]})
        above_horizon_eventids_dict = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
        ds.addROI('above horizon full',{'elevation_best_choice':[10,90],'phi_best_choice':[-90,90],'similarity_count_h':[0,1],'similarity_count_v':[0,1],'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':[1.2,10000],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERMAXcr_template_search_v':[0.5,100],'min_snr_hSLICERADDmin_snr_v':[10,1000]})#'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]
        above_horizon_full_eventids_dict = ds.getCutsFromROI('above horizon full',load=False,save=False,verbose=True, return_successive_cut_counts=False, return_total_cut_counts=False)
        skymap_mode = 'best'
    else:
        #Forces both all baseline maps to point in box
        ds.addROI('above horizon',{'elevation_best_all':[10,90],'phi_best_all':[-90,90]})
        above_horizon_eventids_dict = ds.getCutsFromROI('above horizon',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
        ds.addROI('above horizon full',{'elevation_best_all':[10,90],'phi_best_all':[-90,90],'similarity_count_h':[0,10],'similarity_count_v':[0,10],'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,10],'impulsivity_hSLICERADDimpulsivity_v':[0.4,100],'cr_template_search_hSLICERADDcr_template_search_v':[0.8,100]})
        above_horizon_full_eventids_dict = ds.getCutsFromROI('above horizon full',load=False,save=False,verbose=True, return_successive_cut_counts=False, return_total_cut_counts=False)
        skymap_mode = 'h'

    above_horizon_full_eventids_array = []
    above_horizon_full_runs_array = []
    for run in above_horizon_full_eventids_dict.keys():
        above_horizon_full_eventids_array.append(above_horizon_full_eventids_dict[run])
        above_horizon_full_runs_array.append(numpy.ones(len(above_horizon_full_eventids_dict[run]))*run)
    above_horizon_full_eventids_array = numpy.vstack((numpy.concatenate(above_horizon_full_runs_array),numpy.concatenate(above_horizon_full_eventids_array))).T
    del above_horizon_full_runs_array

    if skymap_mode in ['h','v', 'all']:
        elevation_key = 'elevation_best_%s_allsky'%skymap_mode
        azimuth_key = 'phi_best_%s_allsky'%skymap_mode
    elif skymap_mode == 'best':
        elevation_key = 'elevation_best_choice'
        azimuth_key = 'phi_best_choice'

    el_full = ds.getDataArrayFromParam(elevation_key, eventids_dict=above_horizon_full_eventids_dict)
    phi_full = ds.getDataArrayFromParam(azimuth_key, eventids_dict=above_horizon_full_eventids_dict)

    
    for pol in ['h','v', 'all']:
        ds.plotROI2dHist('phi_best_%s_belowhorizon'%pol, 'elevation_best_%s_belowhorizon'%pol, cmap='cool', eventids_dict=above_horizon_full_eventids_dict, return_counts=True, include_roi=False)

    fig_og, ax_og, counts = ds.plotROI2dHist(azimuth_key, elevation_key, cmap='cool', eventids_dict=above_horizon_eventids_dict, return_counts=True, include_roi=False)
    #Consider making this from a later cut.

    sigma_sets = [1.0]#[0.25,0.5,1.0]#[1.0,0.15,[0.05, 0.15]]
    for sigma_set in sigma_sets:
        smoothed_counts = scipy.ndimage.gaussian_filter(counts/numpy.max(counts), sigma_set, order=0, output=None, mode=['wrap','nearest'], cval=0.0, truncate=10.0)

        fig = plt.figure()
        ax = plt.gca(sharex=ax_og,sharey=ax_og)
        im = ax.pcolormesh(ds.current_bin_edges_mesh_x, ds.current_bin_edges_mesh_y, smoothed_counts,cmap=cmap, norm=colors.LogNorm(vmin=0.0001, vmax=smoothed_counts.max()))
        plt.title('Smoothed With Gaussian\nSigma = %s'%str(sigma_set))
        cbar = fig.colorbar(im)
        cbar.set_label('Counts/Max(Counts)')

        phi_label = ds.current_label_x
        theta_label = ds.current_label_y
        plt.xlabel(phi_label)
        plt.ylabel(theta_label)
        plt.grid(which='both', axis='both')
        ax.minorticks_on()
        ax.grid(b=True, which='major', color='k', linestyle='-')
        ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


        interp = scipy.interpolate.interp2d(ds.current_bin_centers_mesh_x[0,:], ds.current_bin_centers_mesh_y[:,0], smoothed_counts, kind='cubic', copy=False, bounds_error=False)
        interp_arb = lambda x, y : numpy.array([interp(_x,_y) for _x, _y in zip(numpy.asarray(x).flatten(),numpy.asarray(y).flatten())]).reshape(numpy.shape(numpy.asarray(x))) #Replicates the output of interp but doesn't assume 2 1d arrays forming a mesh.  Slower but better for interpolating arbitrary input coords.
        interpolated_values = interp(ds.current_bin_centers_mesh_x[0,:], ds.current_bin_centers_mesh_y[:,0])

        fig = plt.figure()
        ax = plt.gca(sharex=ax_og,sharey=ax_og)
        im = ax.pcolormesh(ds.current_bin_edges_mesh_x, ds.current_bin_edges_mesh_y, interpolated_values,cmap=cmap, norm=colors.LogNorm(vmin=0.0001, vmax=smoothed_counts.max()))
        plt.title('Interpolated value output')
        cbar = fig.colorbar(im)
        cbar.set_label('Counts/Max(Counts)')

        plt.xlabel(phi_label)
        plt.ylabel(theta_label)
        plt.grid(which='both', axis='both')
        ax.minorticks_on()
        ax.grid(b=True, which='major', color='k', linestyle='-')
        ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


        #Make a plot saying what percent of event cut by certain cut values.  I am considering trying to do it by evenly sampling on sphere and feeding into the interp_arb.
        sample_points = 10000000
        theta = 90.0 - numpy.arange(ds.roi['above horizon'][elevation_key][0], ds.roi['above horizon'][elevation_key][0])
        phi = numpy.arange(ds.roi['above horizon'][azimuth_key][0], ds.roi['above horizon'][azimuth_key][0])

        cos_bounds = numpy.cos(numpy.deg2rad(90.0 - numpy.array(ds.roi['above horizon'][elevation_key])))
        sampled_elevation = 90.0 - numpy.rad2deg(numpy.arccos(numpy.random.uniform(min(cos_bounds), max(cos_bounds), sample_points)))
        sampled_phi = numpy.random.uniform(min(ds.roi['above horizon'][azimuth_key]),max(ds.roi['above horizon'][azimuth_key]),sample_points)
        sampled_values = interp_arb(sampled_phi,sampled_elevation)

        #bins = numpy.linspace(0.0,0.00005,1000)
        bins = numpy.linspace(0.0,0.5,int(10000*sigma_set))
        centers = 0.5*(bins[1:]+bins[:-1])

        plt.figure()
        plt.suptitle('Sigmas = ' + str(sigma_set))
        ax1 = plt.subplot(3,1,1)
        ax1.set_yscale('log')
        ax1.set_ylabel('Counts for Uniformly Sampled\nSky Heatmap Values')

        plt.grid(which='both', axis='both')
        ax1.minorticks_on()
        ax1.grid(b=True, which='major', color='k', linestyle='-')
        ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        counts= ax1.hist(sampled_values,bins=bins,color='r')[0]
        percentage_cut = numpy.cumsum(counts[::-1])[::-1] #for each
        percent_in_each_bin = 100*counts/len(sampled_values)
        percentage_with_value_greater_than_bin = numpy.cumsum(counts[::-1])[::-1]*100/len(sampled_values)

        ax2 = plt.subplot(3,1,2, sharex=ax1)
        plt.suptitle('Sigmas = ' + str(sigma_set))

        plt.xlabel('Interpolated Heat Map Value\nCounts/Max(Counts)')
        plt.ylabel('Counts')
        plt.grid(which='both', axis='both')
        ax2.minorticks_on()
        ax2.grid(b=True, which='major', color='k', linestyle='-')
        ax2.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        ax2.set_yscale('log')
        ax2.set_yscale('log')

        ax3 = plt.subplot(3,1,3, sharex=ax1)

        plt.grid(which='both', axis='both')
        ax3.minorticks_on()
        ax3.grid(b=True, which='major', color='k', linestyle='-')
        ax3.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

        plt.ylabel('Percentage')
        plt.xlabel('Heat Map Value')
        plt.plot(centers, percentage_with_value_greater_than_bin, label='% Of Sky Above X Val', c='r')

        sky_percent_interp = scipy.interpolate.interp1d(centers,percentage_with_value_greater_than_bin,kind='cubic',bounds_error=False)


        #Plot the values that pass each dict and where they lie
        dicts = [above_horizon_eventids_dict,above_horizon_full_eventids_dict]
        dict_names = ['Forward Cut', 'Full Cuts']

        for index, eventids_dict in enumerate(dicts):
            el = ds.getDataArrayFromParam(elevation_key, eventids_dict=eventids_dict)
            phi = ds.getDataArrayFromParam(azimuth_key, eventids_dict=eventids_dict)
            val = interp_arb(phi,el)
            ax2.hist(val,bins=bins,label=dict_names[index])
            ax3.plot(centers, 100*numpy.cumsum(numpy.histogram(val,bins=bins)[0])/len(val), label='%% Of Events Remaining for UL Cut at X\n%s'%dict_names[index], c=plt.rcParams['axes.prop_cycle'].by_key()['color'][index])
            #ax3.hist(val, bins=bins, cumulative=True,weights=numpy.ones(len(val))*100/len(val), label='%% Of Events Remaining for UL Cut at X\n%s'%dict_names[index],alpha=0.8)
            ax3.set_xlim(0,0.05)

        plt.legend()

        plt.sca(ax3)
        plt.legend(loc = 'upper right')

        vals_full = interp_arb(el_full, phi_full)

        sorted_indices = numpy.argsort(vals_full)
        n_cut = min(500, len(vals_full))
        if n_cut == len(vals_full):
            cut_val = numpy.max(vals_full)
        else:
            cut_val = numpy.mean([vals_full[sorted_indices[n_cut - 1]], vals_full[sorted_indices[n_cut]]])

        print('For Sigma Set = %s here is the list of the %i most isolated events passing the quality cuts:'%(str(sigma_set),n_cut))
        print('(This cut corresponds to a map val of %f, ignoring %f percent of the forward box.'%(cut_val,sky_percent_interp(cut_val)))
        
        out_array = above_horizon_full_eventids_array[sorted_indices[0:n_cut]].astype(int)
        out_array = out_array[numpy.lexsort((out_array[:,1],out_array[:,0]))]

        pprint(out_array)

        min_val_eventid_dict = {}
        for run in numpy.unique(out_array[:,0]):
            min_val_eventid_dict[int(run)] = out_array[out_array[:,0] == run][:,1].astype(int)
            for eventid in out_array[out_array[:,0] == run][:,1].astype(int):
                print('https://users.rcc.uchicago.edu/~cozzyd/monutau/#event&run=%i&entry=%i'%(run,eventid))

        ds.plotROI2dHist(azimuth_key, elevation_key, cmap='cool', eventids_dict=min_val_eventid_dict, return_counts=True, include_roi=False)

        ds.eventInspector(min_val_eventid_dict) #only run on sinteractive with a lot of memory

    