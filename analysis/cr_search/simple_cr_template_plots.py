#!/usr/bin/env python3
'''
The purpose of this script is to generate plots based on the simple_cr_template_search.py results.
'''

import sys
import os
import inspect
import h5py

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.info as info
import tools.cosmic_ray_template as crt
from tools.data_handler import createFile
from objects.fftmath import TemplateCompareTool

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.ion()

import numpy
import scipy
import scipy.signal
import scipy.signal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

datapath = os.environ['BEACON_DATA']



if __name__ == '__main__':
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

    #In progress.


    if len(sys.argv) >= 2:
        runs = numpy.array(sys.argv[1:],dtype=int)
        #runs = numpy.array(int(sys.argv[1]))
    else:
        runs = numpy.array([1649])

    try:
        if plot_1dhists:
            hpol_counts = numpy.zeros((3,len(bins_1dhist)-1)) #3 rows, one for each trigger type.  
            vpol_counts = numpy.zeros((3,len(bins_1dhist)-1))
        if plot_2dhists:
            hpol_counts = numpy.zeros((3,len(bins_2dhist_h)-1)) #3 rows, one for each trigger type.  
            vpol_counts = numpy.zeros((3,len(bins_2dhist_v)-1))
            hv_counts = numpy.zeros((3,len(bins_2dhist_v)-1,len(bins_2dhist_h)-1)) #v going to be plotted vertically, h horizontall, with 3 of such matrices representing the different trigger types.

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
                        file.close()
                    except Exception as e:
                        print('Error loading data.')
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)

            max_output_correlation_values = numpy.max(output_correlation_values,axis=1)
            
            if plot_1dhists:
                for trig_index in range(3):
                    hpol_counts[trig_index] += numpy.histogram(numpy.max(output_correlation_values[numpy.where(trigger_type == trig_index+1)[0]][:,[0,2,4,6]],axis=1),bins=bins_1dhist)[0]
                    vpol_counts[trig_index] += numpy.histogram(numpy.max(output_correlation_values[numpy.where(trigger_type == trig_index+1)[0]][:,[1,3,5,7]],axis=1),bins=bins_1dhist)[0]

            if plot_2dhists:
                for trig_index in range(3):
                    hv_counts[trig_index] += numpy.histogram2d(numpy.max(output_correlation_values[numpy.where(trigger_type == trig_index+1)[0]][:,[0,2,4,6]],axis=1), numpy.max(output_correlation_values[numpy.where(trigger_type == trig_index+1)[0]][:,[1,3,5,7]],axis=1), bins = [bins_2dhist_h,bins_2dhist_v])[0].T 

        if plot_1dhists:
            summed_counts = hpol_counts + vpol_counts

            fig1, ax1 = plt.subplots()
            plt.title('Runs = %s'%str(runs))
            plt.title('Max Correlation Values\nBoth Polarizations')
            ax1.bar(bins_1dhist[:-1], numpy.sum(summed_counts,axis=0), width=numpy.diff(bins_1dhist), edgecolor='black', align='edge')


            fig2, ax2 = plt.subplots()
            plt.title('Runs = %s'%str(runs))
            ax2.bar(bins_1dhist[:-1], summed_counts[0], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = Software',alpha=0.7,)
            ax2.bar(bins_1dhist[:-1], summed_counts[1], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = RF',alpha=0.7,)
            ax2.bar(bins_1dhist[:-1], summed_counts[2], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = GPS',alpha=0.7,)
            plt.legend(loc='upper left')
            plt.ylabel('Counts')
            plt.xlabel('Correlation Value with bi-delta CR Template')
            
            fig3 = plt.figure()
            plt.suptitle('Runs = %s'%str(runs))
            ax_a = plt.subplot(2,1,1)
            for pol_index, pol in enumerate(['hpol','vpol']):
                ax_b = plt.subplot(2,1,pol_index+1,sharex=ax_a,sharey=ax_a)
                if pol == 'hpol':
                    max_output_correlation_values = hpol_counts
                else:
                    max_output_correlation_values = vpol_counts

                ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[0], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = Software',alpha=0.7,)
                ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[1], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = RF',alpha=0.7,)
                ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[2], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = GPS',alpha=0.7,)
                plt.legend(loc='upper left')
                plt.ylabel('%s Counts'%pol.title())
                plt.xlabel('Correlation Value with bi-delta CR Template')



        if plot_2dhists:
            for trig_index in range(3):
                fig4, ax4 = plt.subplots()
                plt.title('bi-delta CR Template Correlations, Runs = %s\nTrigger = %s'%(str(runs),['Software','RF','GPS'][trig_index]))
                im = ax4.pcolormesh(bin_centers_mesh_h, bin_centers_mesh_v, hv_counts[trig_index],norm=colors.LogNorm(vmin=0.5, vmax=hv_counts[trig_index].max()))#cmap=plt.cm.coolwarm
                plt.xlabel('HPol Correlation Values')
                plt.xlim(0,1)
                plt.ylabel('VPol Correlation Values')
                plt.ylim(0,1)
                cbar = fig4.colorbar(im)
                cbar.set_label('Counts')

    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
  