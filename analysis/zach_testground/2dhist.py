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
from tools.data_handler import createFile, getEventTimes,loadTriggerTypes
from tools.fftmath import TemplateCompareTool



import numpy
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import scipy.signal

datapath = os.environ['BEACON_DATA']

plt.ion()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        try:
            plt.close('all')
            run = int(sys.argv[1])
            reader = Reader(datapath,run)

            #Parameters:
            #Curve choice is a parameter in the bi-delta template model that changes the timing of the input dela signal.
            curve_choice = 0
            upsample_factor = 1
            save_data = True

            #Prepare for Correlations
            reader.setEntry(0)
            waveform_times = reader.t()
            waveform_sample = reader.wf(0)
            waveform_sample, waveform_times = scipy.signal.resample(waveform_sample,len(waveform_sample)*upsample_factor,t=waveform_times) #upsample times to desired amount.
            template_t = waveform_times
            template_E = waveform_sample
            
            len_t = len(template_t)
            template_E = template_E/(numpy.std(template_E)*len_t) #Pre dividing to handle normalization of cross correlation.

            all_times = getEventTimes(reader)
            all_trigger_types = loadTriggerTypes(reader)
            eventids = numpy.where(all_trigger_types == 3)[0][0:500]
            #eventids = numpy.where(all_trigger_types == 3)[4082:4082 + 1000]
            #eventids = numpy.arange(500) + 4082

            all_x = all_times[eventids]

            x_values = numpy.array([])
            y_values = numpy.array([])
            z_values = numpy.array([])


            plt.figure()
            for event_index, eventid in enumerate(eventids):
                if event_index%10 == 0:
                    sys.stdout.write('(%i/%i)\r'%(event_index,len(eventids)))
                    sys.stdout.flush()

                #CALCULATE CORRELATION VALUE
                reader.setEntry(eventid)
                wf1 = scipy.signal.resample(reader.wf(1),len_t) #I don't need the times.
                wf3 = scipy.signal.resample(reader.wf(3),len_t) #I don't need the times.

                cc = scipy.signal.correlate(wf1, wf3)/(wf1.std()*wf3.std()*len(wf3)) #template_E already normalized for cc
                cc_times = (template_t[1]-template_t[0])*(numpy.arange(len(cc))-len(cc)//2)
                x = numpy.ones_like(cc)*all_x[event_index]

                x_values = numpy.append(x_values,x)
                y_values = numpy.append(y_values,cc_times)
                z_values = numpy.append(z_values,cc)



                # plt.scatter(x,cc_times,c=cc,cmap='coolwarm')

            # cbar = plt.colorbar()
            # cbar.set_label('Correlation Value', rotation=90)
            # plt.figure()
            # plt.subplot(2,1,1)
            # plt.plot(wf1)
            # plt.plot(wf3)
            # plt.subplot(2,1,2)
            # plt.plot(cc_times,cc)

            plt.figure()
            plt.hist2d(x_values,y_values,weights=z_values,bins=[100,2000],cmap='coolwarm')
            cbar = plt.colorbar()
            cbar.set_label('Mean Correlation Value', rotation=90)
            plt.ylim(-800,800)


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


    
