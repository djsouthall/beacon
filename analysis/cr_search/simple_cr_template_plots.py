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
from matplotlib.widgets import LassoSelector
from matplotlib.patches import Rectangle
from matplotlib import cm

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



class SelectFromCollection(object):
    """
    This is an adapted bit of code that prints out information
    about the lasso'd data points. This code was originally written
    in plane_hunt.py and is being adapted for use with 2d histograms.

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.
    """

    def __init__(self, ax, collection, eventids, runnum, total_hpol_delays,total_vpol_delays,alpha_other=0.3):

        self.canvas = ax.figure.canvas
        self.eventids = eventids
        self.runnum = runnum
        self.id = numpy.array(list(zip(self.runnum,self.eventids)))
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

        self.total_hpol_delays = total_hpol_delays
        self.total_vpol_delays = total_vpol_delays




    def onselect(self, verts):
        path = Path(verts)
        self.ind = numpy.nonzero(path.contains_points(self.xys))[0]
        print('Selected run/eventids:')
        print(repr(self.id[self.ind]))
        txtToClipboard(repr(self.id[self.ind]))
        event_info = self.id[self.ind]
        #print(event_info)
        print('Coordinates:')
        print(repr(self.xys[self.ind]))
        self.canvas.draw_idle()

        self.alignSelectedEvents()

    def plotTimeDelays(self,times,total_hpol_delays,total_vpol_delays):
        '''
        This is intended to reproduce the plot that Kaeli used to make, with the time
        delays being plotted all on the same plot. 
        '''
        times = times - times[0]
        for pol in ['Hpol','Vpol']:
            _fig = plt.figure()
            _fig.canvas.set_window_title('%s Delays'%pol)
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            plt.title(pol)
            plt.ylabel('Time Delay (ns)')
            plt.xlabel('Readout Time (s)')

            for pair_index, pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
                if pol == 'Hpol':
                    y = total_hpol_delays[:,pair_index]
                else:
                    y = total_vpol_delays[:,pair_index]

                #plt.plot(times, y,linestyle = '--',alpha=0.5)
                plt.scatter(times, y,label='A%i and A%i'%(pair[0],pair[1]))
            plt.legend(loc='upper right')

    def alignSelectedEvents(self, plot_aligned_wf=False,save_template=False,plot_timedelays=True):
        '''
        My plan is for this to be called when some events are circled in the plot.
        It will take those wf, align them, and plot the averaged waveforms.  No
        filters will be applied. 
        '''
        if plot_timedelays == True:
            runs, counts = numpy.unique(self.id[self.ind][:,0],return_counts=True)
            run = runs[numpy.argmax(counts)]
            print('Only calculating template from run with most points circled: run %i with %i events circled'%(run,max(counts)))
            eventids = self.id[self.ind][:,1][self.id[self.ind][:,0] == run]
            coords = self.xys[self.ind]

            self.plotTimeDelays(self.xys[self.ind][:,0]*60,self.total_hpol_delays[self.ind],self.total_vpol_delays[self.ind])

        _reader = Reader(datapath,run)
        
        crit_freq_low_pass_MHz = None
        low_pass_filter_order = None
        
        crit_freq_high_pass_MHz = None# 45
        high_pass_filter_order = None# 12
        
        waveform_index_range = (None,None)
        
        final_corr_length = 2**18

        tct = TemplateCompareTool(_reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False,apply_phase_response=True)
        tdc = TimeDelayCalculator(_reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False,apply_phase_response=True)
        self.cor = Correlator(_reader,  upsample=2**15, n_phi=360, n_theta=360, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=True)
        
        if True:
            print('TRYING TO MAKE CORRELATOR PLOT.')
            print(eventids)
            self.cor.animatedMap(eventids, 'both', '', plane_zenith=None,plane_az=None,hilbert=False, max_method=None,center_dir='E',save=False,dpi=300)

        times, averaged_waveforms = tct.averageAlignedSignalsPerChannel( eventids, align_method=0, template_eventid=eventids[0], plot=plot_aligned_wf,event_type=None)
        
        resampled_averaged_waveforms_original_length = numpy.zeros((8,len(_reader.t())))
        for channel in range(8):
            resampled_averaged_waveforms_original_length[channel] = scipy.interpolate.interp1d(times,averaged_waveforms[channel],kind='cubic',bounds_error=False,fill_value=0)(reader.t())

        if False:
            for channel in range(8):
                plt.figure()
                plt.title(str(channel))
                for eventid in eventids:
                    tct.setEntry(eventid)
                    plt.plot(tct.t(),tct.wf(channel),label=str(eventid),alpha=0.8)
                plt.legend()
                plt.xlabel('t (ns)')
                plt.ylabel('adu')


        if save_template == True:
            filename_index = 0 
            filename = './generated_event_template_%i.csv'%filename_index
            existing_files = numpy.array(glob.glob('./*.csv'))

            while numpy.isin(filename,existing_files):
                filename_index += 1
                filename = './generated_event_template_%i.csv'%filename_index
            numpy.savetxt(filename,resampled_averaged_waveforms_original_length, delimiter=",")
            print('Genreated template saved as:\n%s'%filename)



        tdc.calculateMultipleTimeDelays(eventids, align_method=8,hilbert=False,plot=True, colors=numpy.array(coords)[:,0])

        return resampled_averaged_waveforms_original_length

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()








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
    #ROI  List
    #This should be a list with coords x1,x2, y1, y2 for each row.  These will be used to produce plots pertaining to 
    #these specific regions of interest.  x and y should be correlation values, and ordered.
    plot_roi = True
    roi = numpy.array([[0.16,0.36,0.16,0.45],[0.56,0.745,0.88,0.94],[0.67,0.79,0.53,0.60],[0.70,0.82,0.38,0.46]])
    roi_colors = [cm.rainbow(x) for x in numpy.linspace(0, 1, numpy.shape(roi)[0])]

    #Impulsivity Plot Params:
    plot_impulsivity = True
    impulsivity_dset = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_0-Hilb_0-corlen_262144-align_0'
    bins_impulsivity_h = numpy.linspace(0,1,201)
    bins_impulsivity_v = numpy.linspace(0,1,201)
    
    impulsivity_bin_centers_mesh_h, impulsivity_bin_centers_mesh_v = numpy.meshgrid((bins_impulsivity_h[:-1] + bins_impulsivity_h[1:]) / 2, (bins_impulsivity_v[:-1] + bins_impulsivity_v[1:]) / 2)




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
                        file.close()
                    except Exception as e:
                        print('Error loading data.')
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)

            max_output_correlation_values = numpy.max(output_correlation_values,axis=1)
            
            if numpy.any([plot_1dhists,plot_2dhists,plot_impulsivity]):
                for trig_index in range(3):
                    trigger_cut = trigger_type == trig_index+1
                    trigger_cut_indices = numpy.where(trigger_cut)[0]

                    max_output_correlation_values_h = numpy.max(output_correlation_values[trigger_cut_indices][:,[0,2,4,6]],axis=1)
                    max_output_correlation_values_v = numpy.max(output_correlation_values[trigger_cut_indices][:,[1,3,5,7]],axis=1)
                    if plot_1dhists:
                        hpol_counts[trig_index] += numpy.histogram(max_output_correlation_values_h,bins=bins_1dhist)[0]
                        vpol_counts[trig_index] += numpy.histogram(max_output_correlation_values_v,bins=bins_1dhist)[0]
                    if plot_2dhists:
                        hv_counts[trig_index] += numpy.histogram2d(max_output_correlation_values_h, max_output_correlation_values_v, bins = [bins_2dhist_h,bins_2dhist_v])[0].T 
                    if plot_impulsivity:
                        impulsivity_hv_counts[trig_index] += numpy.histogram2d(hpol_output_impulsivity[trigger_cut_indices], vpol_output_impulsivity[trigger_cut_indices], bins = [bins_impulsivity_h,bins_impulsivity_v])[0].T 
                        if plot_roi:
                            for roi_index, roi_coords in enumerate(roi):
                                roi_cut = numpy.logical_and(numpy.logical_and(max_output_correlation_values_h >= roi_coords[0], max_output_correlation_values_h <= roi_coords[1]),numpy.logical_and(max_output_correlation_values_v >= roi_coords[2], max_output_correlation_values_v <= roi_coords[3]))
                                roi_cut_indices = trigger_cut_indices[roi_cut]
                                # if roi_index == 0:
                                #     import pdb; pdb.set_trace()
                                impulsivity_roi_counts[roi_index][trig_index] += numpy.histogram2d(hpol_output_impulsivity[roi_cut_indices], vpol_output_impulsivity[roi_cut_indices], bins = [bins_impulsivity_h,bins_impulsivity_v])[0].T 
        if plot_1dhists:
            summed_counts = hpol_counts + vpol_counts

            fig1, ax1 = plt.subplots()
            plt.title('Runs = %s'%str(runs))
            plt.title('Max Correlation Values\nBoth Polarizations')
            ax1.bar(bins_1dhist[:-1], numpy.sum(summed_counts,axis=0), width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='All Triggers')
            plt.legend(loc='upper left')

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

                if plot_roi:
                    for roi_index, roi_coords in enumerate(roi): 
                        ax_b.axvspan(roi_coords[0+2*pol_index],roi_coords[1+2*pol_index],alpha=0.4, color=roi_colors[roi_index],label='roi %i'%roi_index)
                ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[0], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = Software',alpha=0.7,)
                ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[1], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = RF',alpha=0.7,)
                ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[2], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = GPS',alpha=0.7,)
                plt.ylabel('%s Counts'%pol.title())
                plt.xlabel('Correlation Value with bi-delta CR Template')

                plt.legend(loc='upper left')



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
                if plot_roi:
                    for roi_index, roi_coords in enumerate(roi): 
                        ax4.add_patch(Rectangle((roi_coords[0], roi_coords[2]), roi_coords[1] - roi_coords[0], roi_coords[3] - roi_coords[2],fill=False, edgecolor=roi_colors[roi_index]))
                        plt.text((roi_coords[1]+roi_coords[0])/2, roi_coords[3]+0.02,'roi %i'%roi_index,color=roi_colors[roi_index],fontweight='bold')

        if plot_impulsivity:
            for trig_index in range(3):
                fig4, ax4 = plt.subplots()
                plt.title('bi-delta CR Impulsivity Values, Runs = %s\nTrigger = %s'%(str(runs),['Software','RF','GPS'][trig_index]))
                im = ax4.pcolormesh(impulsivity_bin_centers_mesh_h, impulsivity_bin_centers_mesh_v, impulsivity_hv_counts[trig_index],norm=colors.LogNorm(vmin=0.5, vmax=impulsivity_hv_counts[trig_index].max()))#cmap=plt.cm.coolwarm
                plt.xlabel('HPol Impulsivity Values')
                #plt.xlim(0,1)
                plt.ylabel('VPol Impulsivity Values')
                #plt.ylim(0,1)
                cbar = fig4.colorbar(im)
                cbar.set_label('Counts')
                if plot_roi:
                    legend_properties = []
                    legend_labels = []
                    for roi_index, roi_coords in enumerate(roi):
                        levels = numpy.linspace(0,numpy.max(impulsivity_roi_counts[roi_index][trig_index]),6)[1:7] #Not plotting bottom contour because it is often background and clutters plot.
                        cs = ax4.contour(bin_centers_mesh_h, bin_centers_mesh_v, impulsivity_roi_counts[roi_index][trig_index], colors=[roi_colors[roi_index]],levels=levels)#,label='roi %i'%roi_index)
                        legend_properties.append(cs.legend_elements()[0][0])
                        legend_labels.append('roi %i'%roi_index)
                        # strs = ['', '', '', '', 'roi %i'%roi_index, '', '']
                        # fmt = {}
                        # for l, s in zip(cs.levels, strs):
                        #     fmt[l] = s
                        #cs.levels[4] = 'roi %i'%roi_index

                        # try:
                        #     ax4.clabel(cs,cs.levels,fmt=fmt,fontsize=16,inline=False)
                        # except Exception as e:
                        #     print('Error when adding contour.')
                        #     print(e)
                        #     exc_type, exc_obj, exc_tb = sys.exc_info()
                        #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        #     print(exc_type, fname, exc_tb.tb_lineno)
                    #import pdb; pdb.set_trace()
                    plt.legend(legend_properties,legend_labels)
                # if plot_roi:
                #Need to turn this into a contour plotted over top using 2dhist with points just form each ROI
                #     for roi_index, roi_coords in enumerate(roi): 
                #         ax4.add_patch(Rectangle((roi_coords[0], roi_coords[2]), roi_coords[1] - roi_coords[0], roi_coords[3] - roi_coords[2],fill=False, edgecolor=roi_colors[roi_index]))
                #         plt.text((roi_coords[1]+roi_coords[0])/2, roi_coords[3]+0.02,'roi %i'%roi_index,color=roi_colors[roi_index],fontweight='bold')

    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
  