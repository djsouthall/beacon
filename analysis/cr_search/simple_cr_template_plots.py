#!/usr/bin/env python3
'''
The purpose of this script is to generate plots based on the simple_cr_template_search.py results.

The documention for each function needs to be updated to represent the new functionality.  Additionally this should
all be moved to the tools folder and just be called here.  

'''
import sys
import os
import inspect
import h5py
import ast

import numpy
import scipy
import scipy.signal
import scipy.signal

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_handler import createFile
from tools.data_slicer import dataSlicerSingleRun

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.patches import Rectangle
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

datapath = os.environ['BEACON_DATA']

def main():
    print('WARNING THIS IS DEPRECATED AND EXPECTS OLDER FORM OF CLASS')
    plt.close('all')
    #Parameters:
    #General Params:
    curve_choice = 0 #Which curve to select from correlation data.


    #1dhist Params:
    plot_1dhists = True
    bins_1dhist = numpy.linspace(0,1,201)

    #2dhist Params:
    plot_2dhists = True
    bins_2dhist_h = numpy.linspace(0,1,201)
    bins_2dhist_v = numpy.linspace(0,1,201)
    
    bin_centers_mesh_h, bin_centers_mesh_v = numpy.meshgrid((bins_2dhist_h[:-1] + bins_2dhist_h[1:]) / 2, (bins_2dhist_v[:-1] + bins_2dhist_v[1:]) / 2)

    trigger_types = [2]#[1,2,3]
    #In progress.
    #ROI  List
    #This should be a list with coords x1,x2, y1, y2 for each row.  These will be used to produce plots pertaining to 
    #these specific regions of interest.  x and y should be correlation values, and ordered.
    plot_roi = True
    roi = numpy.array([])
    # roi = numpy.array([ [0.16,0.4,0.16,0.45],
    #                     [0.56,0.745,0.88,0.94],
    #                     [0.67,0.79,0.53,0.60],
    #                     [0.7,0.85,0.65,0.80],
    #                     [0.70,0.82,0.38,0.46],
    #                     [0.12,0.38,0.05,0.12],
    #                     [0.55,0.66,0.38,0.44],
    #                     [0.50,0.75,0.20,0.32],
    #                     [0.48,0.63,0.55,0.63]])
    roi_colors = [cm.rainbow(x) for x in numpy.linspace(0, 1, numpy.shape(roi)[0])]

    #Impulsivity Plot Params:
    plot_impulsivity = True
    impulsivity_dset = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_0-Hilb_0-corlen_262144-align_0'
    impulsivity_bin_edges_h = numpy.linspace(0,1,201)
    impulsivity_bin_edges_v = numpy.linspace(0,1,201)
    
    impulsivity_bin_centers_mesh_h, impulsivity_bin_centers_mesh_v = numpy.meshgrid((impulsivity_bin_edges_h[:-1] + impulsivity_bin_edges_h[1:]) / 2, (impulsivity_bin_edges_v[:-1] + impulsivity_bin_edges_v[1:]) / 2)

    #60 Hz Background plotting Params:
    show_only_60hz_bg = False
    window_s = 20.0
    TS_cut_level = 0.15
    normalize_by_window_index = True
    plot_background_60hz_TS_hist = True
    bins_background_60hz_TS_hist = numpy.linspace(0,1,201) #Unsure what a good value is here yet.



    if len(sys.argv) >= 2:
        runs = numpy.array(sys.argv[1:],dtype=int)
        #runs = numpy.array(int(sys.argv[1]))
    else:
        runs = numpy.array([1642,1643,1644,1645,1646,1647])

    try:
        if plot_1dhists:
            hpol_counts = numpy.zeros((3,len(bins_1dhist)-1)) #3 rows, one for each trigger type.  
            vpol_counts = numpy.zeros((3,len(bins_1dhist)-1))
        if plot_2dhists:
            hv_counts = numpy.zeros((3,len(bins_2dhist_v)-1,len(bins_2dhist_h)-1)) #v going to be plotted vertically, h horizontall, with 3 of such matrices representing the different trigger types.
        if plot_impulsivity:
            impulsivity_hv_counts = numpy.zeros((3,len(bins_2dhist_v)-1,len(bins_2dhist_h)-1)) #v going to be plotted vertically, h horizontall, with 3 of such matrices representing the different trigger types.
            impulsivity_roi_counts = numpy.zeros((numpy.shape(roi)[0],3,len(bins_2dhist_v)-1,len(bins_2dhist_h)-1)) #A seperate histogram for each roi.  The above is a total histogram.
        if show_only_60hz_bg:
            print('ONLY SHOWING EVENTS THAT ARE EXPECTED TO BE 60HZ BACKGROUND SIGNALS')
        else:
            plot_background_60hz_TS_hist = False            

        if plot_background_60hz_TS_hist:
            background_60hz_TS_counts = numpy.zeros((3,len(bins_background_60hz_TS_hist)-1))

        for run in runs:
            #Prepare to load correlation values.
            reader = Reader(datapath,run)
            try:
                print(reader.status())
            except Exception as e:
                print('Status Tree not present.  Returning Error.')
                print('\nError in %s'%inspect.stack()[0][3])
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                sys.exit(1)
            filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
            
            #Load correlation values            
            if filename is not None:
                with h5py.File(filename, 'a') as file:
                    eventids = file['eventids'][...]
                    dsets = list(file.keys()) #Existing datasets
                    try:
                        this_dset = 'bi-delta-curve-choice-%i'%curve_choice

                        event_times = file['calibrated_trigtime'][...]
                        trigger_type = file['trigger_type'][...]
                        output_correlation_values = file['cr_template_search'][this_dset][...]
                        if plot_impulsivity:
                            # impulsivity_dsets = list(file['impulsivity'].keys())
                            # print(impulsivity_dsets)
                            hpol_output_impulsivity = file['impulsivity'][impulsivity_dset]['hpol'][...]
                            vpol_output_impulsivity = file['impulsivity'][impulsivity_dset]['vpol'][...]


                        #Apply cuts
                        #These will be applied to all further calculations and are convenient for investigating particular sources. 
                        if show_only_60hz_bg:
                            trig2_cut = file['trigger_type'][...] == 2 #60Hz algorithm should only run on RF events.  Set others to -1.
                            metric = numpy.ones(len(trigger_type))*-1.0
                            metric[trig2_cut] = diffFromPeriodic(event_times[trig2_cut],window_s=window_s, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=False)
                            show_only_cut = metric > TS_cut_level #Great than means it IS a 60Hz likely
                        else:
                            show_only_cut = numpy.ones(len(event_times),dtype=bool)
                        file.close()
                    except Exception as e:
                        print('Error loading data.')
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)

            max_output_correlation_values = numpy.max(output_correlation_values,axis=1)
            
            if numpy.any([plot_1dhists,plot_2dhists,plot_impulsivity,plot_background_60hz_TS_hist]):
                for trig_index in range(3):
                    trigger_cut = trigger_type == trig_index+1
                    trigger_cut_indices = numpy.where(numpy.logical_and(trigger_cut,show_only_cut))[0]
                    trigger_cut_indices_raw = numpy.where(trigger_cut)[0] #Without the selected background cut

                    max_output_correlation_values_h = numpy.max(output_correlation_values[trigger_cut_indices][:,[0,2,4,6]],axis=1)
                    max_output_correlation_values_v = numpy.max(output_correlation_values[trigger_cut_indices][:,[1,3,5,7]],axis=1)
                    if plot_1dhists:
                        hpol_counts[trig_index] += numpy.histogram(max_output_correlation_values_h,bins=bins_1dhist)[0]
                        vpol_counts[trig_index] += numpy.histogram(max_output_correlation_values_v,bins=bins_1dhist)[0]
                    if plot_2dhists:
                        hv_counts[trig_index] += numpy.histogram2d(max_output_correlation_values_h, max_output_correlation_values_v, bins = [bins_2dhist_h,bins_2dhist_v])[0].T 
                    if plot_impulsivity:
                        impulsivity_hv_counts[trig_index] += numpy.histogram2d(hpol_output_impulsivity[trigger_cut_indices], vpol_output_impulsivity[trigger_cut_indices], bins = [impulsivity_bin_edges_h,impulsivity_bin_edges_v])[0].T 
                        if plot_roi:
                            for roi_index, roi_coords in enumerate(roi):
                                roi_cut = numpy.logical_and(numpy.logical_and(max_output_correlation_values_h >= roi_coords[0], max_output_correlation_values_h <= roi_coords[1]),numpy.logical_and(max_output_correlation_values_v >= roi_coords[2], max_output_correlation_values_v <= roi_coords[3]))
                                roi_cut_indices = trigger_cut_indices[roi_cut]
                                # if roi_index == 0:
                                #     import pdb; pdb.set_trace()
                                impulsivity_roi_counts[roi_index][trig_index] += numpy.histogram2d(hpol_output_impulsivity[roi_cut_indices], vpol_output_impulsivity[roi_cut_indices], bins = [impulsivity_bin_edges_h,impulsivity_bin_edges_v])[0].T 
                    if plot_background_60hz_TS_hist:
                        background_60hz_TS_counts[trig_index] += numpy.histogram(metric[trigger_cut_indices_raw],bins=bins_background_60hz_TS_hist)[0]

        if plot_1dhists:
            summed_counts = hpol_counts + vpol_counts

            fig1, ax1 = plt.subplots()
            plt.title('Runs = %s'%str(runs))
            if show_only_60hz_bg == True:
                plt.title('Max Correlation Values\nBoth Polarizations\n[ONLY SUSPECTED 60Hz BACKGROUND EVENTS]')
            else:
                plt.title('Max Correlation Values\nBoth Polarizations')
            ax1.bar(bins_1dhist[:-1], numpy.sum(summed_counts,axis=0), width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='All Triggers')
            plt.ylabel('Counts')
            plt.xlabel('Correlation Value with bi-delta CR Template')
            plt.legend(loc='upper left')

            fig2, ax2 = plt.subplots()
            if show_only_60hz_bg == True:
                plt.title('Runs = %s\n[ONLY SUSPECTED 60Hz BACKGROUND EVENTS]'%str(runs))
            else:
                plt.title('Runs = %s'%str(runs))
            if 1 in trigger_types:
                ax2.bar(bins_1dhist[:-1], summed_counts[0], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = Software',alpha=0.7,)
            if 2 in trigger_types:
                ax2.bar(bins_1dhist[:-1], summed_counts[1], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = RF',alpha=0.7,)
            if 3 in trigger_types:
                ax2.bar(bins_1dhist[:-1], summed_counts[2], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = GPS',alpha=0.7,)
            plt.legend(loc='upper left')
            plt.ylabel('Counts')
            plt.xlabel('Correlation Value with bi-delta CR Template')
            
            fig3 = plt.figure()
            if show_only_60hz_bg == True:
                plt.title('Runs = %s\n[ONLY SUSPECTED 60Hz BACKGROUND EVENTS]'%str(runs))
            else:
                plt.title('Runs = %s'%str(runs))
            ax_a = plt.subplot(2,1,1)
            for pol_index, pol in enumerate(['hpol','vpol']):
                ax_b = plt.subplot(2,1,pol_index+1,sharex=ax_a,sharey=ax_a)
                if pol == 'hpol':
                    max_output_correlation_values = hpol_counts
                else:
                    max_output_correlation_values = vpol_counts

                if plot_roi:
                    for roi_index, roi_coords in enumerate(roi): 
                        ax_b.axvspan(roi_coords[0+2*pol_index],roi_coords[1+2*pol_index],alpha=0.4, color=roi_colors[roi_index],label='roi %i'%roi_index)
                if 1 in trigger_types:
                    ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[0], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = Software',alpha=0.7,)
                if 2 in trigger_types:
                    ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[1], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = RF',alpha=0.7,)
                if 3 in trigger_types:
                    ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[2], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = GPS',alpha=0.7,)
                plt.ylabel('%s Counts'%pol.title())
                plt.xlabel('Correlation Value with bi-delta CR Template')

                plt.legend(loc='upper left')



        if plot_2dhists:
            for trig_index in range(3):
                if trig_index + 1 in trigger_types:
                    fig4, ax4 = plt.subplots()
                    if show_only_60hz_bg == True:
                        plt.title('bi-delta CR Template Correlations, Runs = %s\nTrigger = %s\n[ONLY SUSPECTED 60Hz BACKGROUND EVENTS]'%(str(runs),['Software','RF','GPS'][trig_index]))
                    else:
                        plt.title('bi-delta CR Template Correlations, Runs = %s\nTrigger = %s'%(str(runs),['Software','RF','GPS'][trig_index]))
                    im = ax4.pcolormesh(bin_centers_mesh_h, bin_centers_mesh_v, hv_counts[trig_index],norm=colors.LogNorm(vmin=0.5, vmax=hv_counts[trig_index].max()))#cmap=plt.cm.coolwarm
                    plt.xlabel('HPol Correlation Values')
                    plt.xlim(0,1)
                    plt.ylabel('VPol Correlation Values')
                    plt.ylim(0,1)
                    try:
                        cbar = fig4.colorbar(im)
                        cbar.set_label('Counts')
                    except Exception as e:
                        print('Error in colorbar, often caused by no events.')
                        print(e)
                    if plot_roi:
                        for roi_index, roi_coords in enumerate(roi): 
                            ax4.add_patch(Rectangle((roi_coords[0], roi_coords[2]), roi_coords[1] - roi_coords[0], roi_coords[3] - roi_coords[2],fill=False, edgecolor=roi_colors[roi_index]))
                            plt.text((roi_coords[1]+roi_coords[0])/2, roi_coords[3]+0.02,'roi %i'%roi_index,color=roi_colors[roi_index],fontweight='bold')

        if plot_impulsivity:
            for trig_index in range(3):
                if trig_index + 1 in trigger_types:
                    fig5, ax5 = plt.subplots()
                    if show_only_60hz_bg == True:
                        plt.title('bi-delta CR Impulsivity Values, Runs = %s\nTrigger = %s\n[ONLY SUSPECTED 60Hz BACKGROUND EVENTS]'%(str(runs),['Software','RF','GPS'][trig_index]))
                    else:
                        plt.title('bi-delta CR Impulsivity Values, Runs = %s\nTrigger = %s'%(str(runs),['Software','RF','GPS'][trig_index]))

                    im = ax5.pcolormesh(impulsivity_bin_centers_mesh_h, impulsivity_bin_centers_mesh_v, impulsivity_hv_counts[trig_index],norm=colors.LogNorm(vmin=0.5, vmax=impulsivity_hv_counts[trig_index].max()),cmap='Greys')#cmap=plt.cm.coolwarm
                    plt.xlabel('HPol Impulsivity Values')
                    #plt.xlim(0,1)
                    plt.ylabel('VPol Impulsivity Values')
                    #plt.ylim(0,1)
                    try:
                        cbar = fig5.colorbar(im)
                        cbar.set_label('Counts')
                    except Exception as e:
                        print('Error in colorbar, often caused by no events.')
                        print(e)
                    if plot_roi:
                        legend_properties = []
                        legend_labels = []
                        for roi_index, roi_coords in enumerate(roi):
                            levels = numpy.linspace(0,numpy.max(impulsivity_roi_counts[roi_index][trig_index]),6)[1:7] #Not plotting bottom contour because it is often background and clutters plot.
                            cs = ax5.contour(bin_centers_mesh_h, bin_centers_mesh_v, impulsivity_roi_counts[roi_index][trig_index], colors=[roi_colors[roi_index]],levels=levels)#,label='roi %i'%roi_index)
                            legend_properties.append(cs.legend_elements()[0][0])
                            legend_labels.append('roi %i'%roi_index)

                        plt.legend(legend_properties,legend_labels)

        if plot_background_60hz_TS_hist:
            fig_bg_60, ax_bg_60 = plt.subplots()
            plt.title('60 Hz Background Event Cut')
            for trig_index in range(3):
                if trig_index + 1 in trigger_types:
                    if 1 in trigger_types:
                        ax_bg_60.bar(bins_background_60hz_TS_hist[:-1], background_60hz_TS_counts[trig_index], width=numpy.diff(bins_background_60hz_TS_hist), edgecolor='black', align='edge',label='Trigger Type = Software',alpha=0.7,)
                    if 2 in trigger_types:
                        ax_bg_60.bar(bins_background_60hz_TS_hist[:-1], background_60hz_TS_counts[trig_index], width=numpy.diff(bins_background_60hz_TS_hist), edgecolor='black', align='edge',label='Trigger Type = RF',alpha=0.7,)
                    if 3 in trigger_types:
                        ax_bg_60.bar(bins_background_60hz_TS_hist[:-1], background_60hz_TS_counts[trig_index], width=numpy.diff(bins_background_60hz_TS_hist), edgecolor='black', align='edge',label='Trigger Type = GPS',alpha=0.7,)

            ax_bg_60.axvspan(TS_cut_level,max(bins_background_60hz_TS_hist[:-1]),color='g',label='Events Plotted',alpha=0.5)
            plt.legend()
            plt.xlabel('Test Statistic\n(Higher = More Likely 60 Hz)')
            plt.ylabel('Counts')
    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':
    
    plot_param_pairs = [['impulsivity_h','impulsivity_v'], ['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v']]
    
    #plt.close('all')
    #runs = [1642,1643,1644,1645,1646,1647]
    if len(sys.argv) == 2:
        runs = [int(sys.argv[1])]
    else:
        runs = [1650]

    for run in runs:
        reader = Reader(datapath,run)
        impulsivity_dset_key = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_262144-align_0'#'LPf_None-LPo_None-HPf_None-HPo_None-Phase_0-Hilb_0-corlen_262144-align_0'
        print("Preparing dataSlicerSingleRun")
        ds = dataSlicerSingleRun(reader, impulsivity_dset_key, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        std_n_bins_h=300,std_n_bins_v=300,max_std_val=9,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False)

        print('Adding ROI to dataSlicerSingleRun')

        #ds.addROI('corr A',{'cr_template_search_h':[0.575,0.665],'cr_template_search_v':[0.385,0.46]})
        #ds.addROI('high v imp',{'impulsivity_h':[0.479,0.585],'impulsivity_h':[0.633,0.7]})
        #ds.addROI('small h.4 v.4 imp',{'impulsivity_h':[0.34,0.45],'impulsivity_h':[0.42,0.5]})

        #ds.addROI('imp cluster',{'impulsivity_h':[0.35,0.46],'impulsivity_v':[0.45,0.50]})

        # roi_key = 'imp cluster'
        # with h5py.File(ds.analysis_filename, 'r') as file:
        #     cut_dict = ast.literal_eval(file['ROI'][roi_key].attrs['dict'])
        #     print('cut_dict = ', cut_dict)
        #     included_antennas = file['ROI'][roi_key].attrs['included_antennas']
        #     print('included_antennas = ', included_antennas)
        #     cr_template_curve_choice = file['ROI'][roi_key].attrs['cr_template_curve_choice']
        #     print('cr_template_curve_choice = ', cr_template_curve_choice)
        #     trigger_types = file['ROI'][roi_key].attrs['trigger_types']
        #     print('trigger_types = ', trigger_types)
        #     file.close()

        print('Generating plots:')
        fig_dict = {}
        ax_dict = {}

        if False:
            #Getting clusters for Run 1650.  Should maybe do this for more runs stacked to have better statistics and better grouping.
            #But would need to develop the tool further to achieve this, and I want to do it this way for now, to understand what tools I need to build.
            #Percieved SNR Clusters
            test_ROI_index = 0
            ds.addROI('SNR ROI %i'%test_ROI_index,{'snr_h':[14,20.25],'snr_v':[21,27.5]}); test_ROI_index += 1
            ds.addROI('SNR ROI %i'%test_ROI_index,{'snr_h':[24.5,28.5],'snr_v':[15,18]}); test_ROI_index += 1
            ds.addROI('SNR ROI %i'%test_ROI_index,{'snr_h':[15,21],'snr_v':[10.5,15]}); test_ROI_index += 1
            ds.addROI('SNR ROI %i'%test_ROI_index,{'snr_h':[21,24],'snr_v':[12,17]}); test_ROI_index += 1

            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                # fig_dict[key_x + ' vs ' + key_y] = fig
                # ax_dict[key_x + ' vs ' + key_y] = ax
            ds.resetAllROI()

        if False:
            #Percieved Impulsivity Clusters
            test_ROI_index = 0
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.51,0.61],'impulsivity_v':[0.34,0.44]}); test_ROI_index += 1
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.45,0.54],'impulsivity_v':[0.20,0.25]}); test_ROI_index += 1
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.38,0.50],'impulsivity_v':[0.28,0.40]}); test_ROI_index += 1
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.36,0.42],'impulsivity_v':[0.47,0.52]}); test_ROI_index += 1
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.44,0.58],'impulsivity_v':[0.50,0.59]}); test_ROI_index += 1
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.50,0.56],'impulsivity_v':[0.64,0.68]}); test_ROI_index += 1 #partial zone with overlap
            ds.addROI('Impulsivity ROI %i'%test_ROI_index,{'impulsivity_h':[0.57,0.68],'impulsivity_v':[0.63,0.71]}); test_ROI_index += 1
            
            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                # fig_dict[key_x + ' vs ' + key_y] = fig
                # ax_dict[key_x + ' vs ' + key_y] = ax
            ds.resetAllROI()

        if True:
            #Percieved CR Template Clusters
            test_ROI_index = 0
            ds.addROI('CR ROI %i'%test_ROI_index,{'cr_template_search_h':[0.56,0.71],'cr_template_search_v':[0.31,0.39]}); test_ROI_index += 1
            ds.addROI('CR ROI %i'%test_ROI_index,{'cr_template_search_h':[0.59,0.68],'cr_template_search_v':[0.39,0.50]}); test_ROI_index += 1
            ds.addROI('CR ROI %i'%test_ROI_index,{'cr_template_search_h':[0.61,0.71],'cr_template_search_v':[0.895,0.93]}); test_ROI_index += 1 #partial zone with overlap
            ds.addROI('CR ROI %i'%test_ROI_index,{'cr_template_search_h':[0.50,0.625],'cr_template_search_v':[0.85,0.89]}); test_ROI_index += 1 #partial zone with overlap

            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                # fig_dict[key_x + ' vs ' + key_y] = fig
                # ax_dict[key_x + ' vs ' + key_y] = ax
            ds.resetAllROI()

        if False:
            #Percieved STD Clusters #There are too many STD groupings to easily group, and many of them are understanably
            #linear relationships between the 2.  Which is hard to use a box for.  
            # test_ROI_index = 0
            # ds.addROI('STD ROI %i'%test_ROI_index,{'std_h':[2.5,5.0],'std_v':[6.0,8.0]}); test_ROI_index += 1
            # ds.addROI('STD ROI %i'%test_ROI_index,{'std_h':[1.5,2.0],'std_v':[1.2,1.5]}); test_ROI_index += 1
            # ds.addROI('STD ROI %i'%test_ROI_index,{'std_h':[2.2,2.9],'std_v':[0.73,0.83]}); test_ROI_index += 1
            # ds.addROI('STD ROI %i'%test_ROI_index,{'std_h':[2.5,5.0],'std_v':[6.0,8.0]}); test_ROI_index += 1

            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                # fig_dict[key_x + ' vs ' + key_y] = fig
                # ax_dict[key_x + ' vs ' + key_y] = ax
            ds.resetAllROI()
            
        if False:
            #Percieved CR Template Clusters
            test_ROI_index = 0
            ds.addROI('p2p ROI %i'%test_ROI_index,{'p2p_h':[52,74],'p2p_v':[66,90]}); test_ROI_index += 1
            ds.addROI('p2p ROI %i'%test_ROI_index,{'p2p_h':[35,55],'p2p_v':[45,62]}); test_ROI_index += 1
            ds.addROI('p2p ROI %i'%test_ROI_index,{'p2p_h':[52,70],'p2p_v':[6,11]}); test_ROI_index += 1
            #There are many more but too hard to seperate.  
            
            for key_x, key_y in plot_param_pairs:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                # fig_dict[key_x + ' vs ' + key_y] = fig
                # ax_dict[key_x + ' vs ' + key_y] = ax
            ds.resetAllROI()


        #addint a relationship parameter to the given params would be interesting.  Hard to apply though.


        
        





    #main() #This is the analysis before it was turned into a class.
    
  