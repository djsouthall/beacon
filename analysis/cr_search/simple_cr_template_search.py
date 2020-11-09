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
from tools.fftmath import TemplateCompareTool

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

                        output_correlation_values = numpy.zeros((file.attrs['N'],8),dtype=float) #Fill this, write to hdf5 once.
                        
                        for eventid_index, eventid in enumerate(eventids): 
                            if eventid%500 == 0:
                                sys.stdout.write('(%i/%i)\r'%(eventid,len(eventids)))
                                sys.stdout.flush()

                            #CALCULATE CORRELATION VALUE
                            reader.setEntry(eventid)
                            for channel in range(8):
                                wf = scipy.signal.resample(reader.wf(channel),len_t) #I don't need the times.
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


    
