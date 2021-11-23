#!/usr/bin/env python3
'''
The purpose of this script is to perform a simple template search using a simple template from cosmic_ray_template.py.

This should be executable to be run using farm.py.  Saving the resulting data to the existing hdf5 files.  
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
from tools.fftmath import TemplateCompareTool, TimeDelayCalculator

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.ioff()

import numpy
import scipy
import scipy.signal
import scipy.signal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

datapath = os.environ['BEACON_DATA']



if __name__ == '__main__':
    if len(sys.argv) >= 2:
        try:
            print(sys.argv)
            if len(sys.argv) == 3:
                print(bool(int(sys.argv[2])))
                farm_mode = bool(int(sys.argv[2]))
            else:
                farm_mode = False
            
            if farm_mode == True:
                print('Farm Mode = True, no plots will be available')
                calculate_correlation_values = True #If True then the values we be newly calculated, if false then will try to load them from the existing files
            else:
                print('Farm Mode = False')
                calculate_correlation_values = False #If True then the values we be newly calculated, if false then will try to load them from the existing files
            #Parameters:

            crit_freq_low_pass_MHz = 80
            low_pass_filter_order = 14

            crit_freq_high_pass_MHz = 20
            high_pass_filter_order = 4

            plot_filter=False

            sine_subtract = True
            sine_subtract_min_freq_GHz = 0.00
            sine_subtract_max_freq_GHz = 0.25
            sine_subtract_percent = 0.03

            apply_phase_response = True

            shorten_signals = False
            shorten_thresh = 0.7
            shorten_delay = 10.0
            shorten_length = 90.0

            align_method = None

            hilbert=False
            final_corr_length = 2**17

            filter_string = ''

            if crit_freq_low_pass_MHz is None:
                filter_string += 'LPf_%s-'%('None')
            else:
                filter_string += 'LPf_%0.1f-'%(crit_freq_low_pass_MHz)

            if low_pass_filter_order is None:
                filter_string += 'LPo_%s-'%('None')
            else:
                filter_string += 'LPo_%i-'%(low_pass_filter_order)

            if crit_freq_high_pass_MHz is None:
                filter_string += 'HPf_%s-'%('None')
            else:
                filter_string += 'HPf_%0.1f-'%(crit_freq_high_pass_MHz)

            if high_pass_filter_order is None:
                filter_string += 'HPo_%s-'%('None')
            else:
                filter_string += 'HPo_%i-'%(high_pass_filter_order)

            if apply_phase_response is None:
                filter_string += 'Phase_%s-'%('None')
            else:
                filter_string += 'Phase_%i-'%(apply_phase_response)

            if hilbert is None:
                filter_string += 'Hilb_%s-'%('None')
            else:
                filter_string += 'Hilb_%i-'%(hilbert)

            if final_corr_length is None:
                filter_string += 'corlen_%s-'%('None')
            else:
                filter_string += 'corlen_%i-'%(final_corr_length)

            if align_method is None:
                filter_string += 'align_%s-'%('None')
            else:
                filter_string += 'align_%i-'%(align_method)

            if shorten_signals is None:
                filter_string += 'shortensignals-%s-'%('None')
            else:
                filter_string += 'shortensignals-%i-'%(shorten_signals)
            if shorten_thresh is None:
                filter_string += 'shortenthresh-%s-'%('None')
            else:
                filter_string += 'shortenthresh-%0.2f-'%(shorten_thresh)
            if shorten_delay is None:
                filter_string += 'shortendelay-%s-'%('None')
            else:
                filter_string += 'shortendelay-%0.2f-'%(shorten_delay)
            if shorten_length is None:
                filter_string += 'shortenlength-%s-'%('None')
            else:
                filter_string += 'shortenlength-%0.2f-'%(shorten_length)

            filter_string += 'sinesubtract_%i'%(int(sine_subtract))

            print(filter_string)



            #Curve choice is a parameter in the bi-delta template model that changes the timing of the input dela signal.
            curve_choice = 0
            upsample_factor = 4
            save_data = True

            if farm_mode == False:
                plt.close('all')
            run = int(sys.argv[1])
            reader = Reader(datapath,run)

            #Prepare for Correlations
            reader.setEntry(0)
            waveform_times = reader.t()
            waveform_sample = reader.wf(0)
            waveform_sample, waveform_times = scipy.signal.resample(waveform_sample,len(waveform_sample)*upsample_factor,t=waveform_times) #upsample times to desired amount.

            cr_gen = crt.CosmicRayGenerator(waveform_times,t_offset=800.0,model='bi-delta')
            template_t, template_E = cr_gen.eFieldGenerator(plot=True,curve_choice=curve_choice)
            
            len_t = len(template_t)
            template_E = template_E/(numpy.std(template_E)*len_t) #Pre dividing to handle normalization of cross correlation.
            

            if calculate_correlation_values == True:
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
                

                if filename is not None:
                    with h5py.File(filename, 'a') as file:
                        eventids = file['eventids'][...]
                        dsets = list(file.keys()) #Existing datasets

                        if not numpy.isin('cr_template_search',dsets):
                            file.create_group('cr_template_search')
                        else:
                            print('cr_template_search group already exists in file %s'%filename)

                        cr_search_dsets = list(file['cr_template_search'].keys())
                        
                        this_dset = 'bi-delta-curve-choice-%i'%curve_choice
                        event_times = file['calibrated_trigtime'][...]
                        trigger_type = file['trigger_type'][...]

                        if not numpy.isin(this_dset,cr_search_dsets):
                            file['cr_template_search'].create_dataset(this_dset, (file.attrs['N'],8), dtype='f', compression='gzip', compression_opts=4, shuffle=True)
                        else:
                            print('cr_template_search["%s"] group already exists in file %s'%(this_dset,filename))

                        file['cr_template_search'][this_dset][...] = numpy.zeros((file.attrs['N'],8),dtype=float)

                        file['cr_template_search'][this_dset].attrs['filter_string'] = filter_string

                        output_correlation_values = numpy.zeros((file.attrs['N'],8),dtype=float) #Fill this, write to hdf5 once.

                        tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=(None,None),plot_filters=plot_filter,apply_phase_response=apply_phase_response)
                        if sine_subtract:
                            tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                        
                        for eventid_index, eventid in enumerate(eventids): 
                            if eventid%500 == 0:
                                sys.stdout.write('(%i/%i)\r'%(eventid,len(eventids)))
                                sys.stdout.flush()

                            #CALCULATE CORRELATION VALUE
                            tdc.setEntry(eventid)
                            for channel in range(8):
                                wf = tdc.wf(channel, apply_filter=True, hilbert=hilbert, tukey=True, sine_subtract=sine_subtract, return_sine_subtract_info=False, ss_first=True)
                                wf = scipy.signal.resample(wf,len_t) #I don't need the times.
                                cc = scipy.signal.correlate(template_E, wf)/wf.std() #template_E already normalized for cc
                                output_correlation_values[eventid,channel] = numpy.max(numpy.abs(cc))


                        file['cr_template_search'][this_dset][...] = output_correlation_values
                        file.close()
            else:

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
            if farm_mode == False:
                bins = numpy.linspace(0,1,201)
                

                plt.figure()
                plt.title('Run = %i'%run)
                plt.title('Max Correlation Values\nBoth Polarizations')
                plt.hist(max_output_correlation_values)
                plt.figure()
                plt.hist(max_output_correlation_values[trigger_type == 1],label='Trigger Type = Software',alpha=0.7,bins=bins)
                plt.hist(max_output_correlation_values[trigger_type == 2],label='Trigger Type = RF',alpha=0.7,bins=bins)
                plt.hist(max_output_correlation_values[trigger_type == 3],label='Trigger Type = GPS',alpha=0.7,bins=bins)
                plt.legend()
                plt.ylabel('Counts')
                plt.xlabel('Correlation Value with bi-delta CR Template')
                
                plt.figure()
                plt.suptitle('Run = %i'%run)
                ax = plt.subplot(2,1,1)
                for pol_index, pol in enumerate(['hpol','vpol']):
                    plt.subplot(2,1,pol_index+1,sharex=ax,sharey=ax)
                    if pol == 'hpol':
                        max_output_correlation_values = numpy.max(output_correlation_values[:,[0,2,4,6]],axis=1) 
                    else:
                        max_output_correlation_values = numpy.max(output_correlation_values[:,[1,3,5,7]],axis=1)

                    plt.hist(max_output_correlation_values[trigger_type == 1],label='Trigger Type = Software',alpha=0.7,bins=bins)
                    plt.hist(max_output_correlation_values[trigger_type == 2],label='Trigger Type = RF',alpha=0.7,bins=bins)
                    plt.hist(max_output_correlation_values[trigger_type == 3],label='Trigger Type = GPS',alpha=0.7,bins=bins)
                    plt.legend()
                    plt.ylabel('%s Counts'%pol.title())
                    plt.xlabel('Correlation Value with bi-delta CR Template')
            

                # eventids = numpy.arange(100)
                # output_correlation_values = numpy.zeros((len(eventids),8),dtype=float)

                # for eventid_index, eventid in enumerate(eventids):
                #     if eventid%10 == 0:
                #         sys.stdout.write('(%i/%i)\r'%(eventid,len(eventids)))
                #         sys.stdout.flush()

                #     #CALCULATE CORRELATION VALUE
                #     reader.setEntry(eventid)
                #     for channel in range(8):
                #         wf = scipy.signal.resample(reader.wf(channel),len_t) #I don't need the times.
                #         cc = scipy.signal.correlate(template_E, wf)/wf.std() #template_E already normalized for cc
                #         output_correlation_values[eventid,channel] = numpy.max(numpy.abs(cc))

                # plt.figure()
                # plt.plot(cc)

        except Exception as e:
            print('Error in main loop.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    else:
        try:
            plt.close('all')
            #Get timing info from real BEACON data for testing.
            run = 1509
            known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
            eventid = known_pulser_ids['run%i'%run]['hpol'][0]
            reader = Reader(datapath,run)
            reader.setEntry(eventid)
            test_t = reader.t()
            test_pulser_adu = reader.wf(0)

            #Creating test signal
            cr_gen = crt.CosmicRayGenerator(test_t,t_offset=800.0,model='bi-delta')
            for curve_choice in range(4):
                out_t, out_E = cr_gen.eFieldGenerator(plot=True,curve_choice=curve_choice)
                      
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(test_t,test_pulser_adu,label='Pulser Signal')
            plt.ylabel('E (adu)')
            plt.xlabel('t (ns)')

            plt.legend()
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            plt.subplot(2,1,2)
            plt.plot(out_t,out_E,label='Test CR Signal')
            plt.scatter(out_t,out_E,c='r')
            plt.ylabel('E (adu)')
            plt.xlabel('t (ns)')

            plt.legend()
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
        
        except Exception as e:
            print('Error in main loop.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    
