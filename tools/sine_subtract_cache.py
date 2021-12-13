#!/usr/bin/env python3
'''
This file contains code written to handle pre-calculating sine subtraction.  The goal is to save runtime on code
by storing the sine subtraction output values and calling them for each event as needed.  This will also hopefully
be done for the TV subtraction when that is included.

Note that the levels of ROOT idiosyncrasies here is frustrating and is the cause of weird workarounds.
'''

import numpy
import os
import sys
from array import array
import inspect
import ROOT
ROOT.gSystem.Load(os.environ['LIB_ROOT_FFTW_WRAPPER_DIR'] + 'build/libRootFftwWrapper.so.3')
ROOT.gInterpreter.ProcessLine('#include "%s"'%(os.environ['LIB_ROOT_FFTW_WRAPPER_DIR'] + 'include/FFTtools.h'))
ROOT.gInterpreter.ProcessLine("void dobranch(TTree *t,  const char * name, long long addr) { t->Branch(name, (FFTtools::SineSubtractResult*) addr); }");

from ROOT import FFTtools

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_handler import loadTriggerTypes

import matplotlib.pyplot as plt
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
plt.ion()

datapath = os.environ['BEACON_DATA']
#cache_dir = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'sine_subtraction_cache')
cache_dir = os.environ['BEACON_SINE_SUBTRACTION_CACHE']#os.path.join(os.environ['BEACON_PROCESSED_DATA'],'sine_subtraction_cache')


def prepareStandardSineSubtractions():
    '''
    This should be standard and consistent from when the ROOT file was generated and when the results are being called.
    '''
    sine_subtract_min_freq_GHz = 0.00
    sine_subtract_max_freq_GHz = 0.25
    sine_subtract_percent = 0.03
    max_failed_iterations = 5
    sine_subtracts = [None]*8
    for channel in range(8):
        sine_subtracts[channel] = FFTtools.SineSubtract(max_failed_iterations, sine_subtract_percent,False)
        sine_subtracts[channel].setVerbose(False)
        sine_subtracts[channel].setFreqLimits(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz)
        sine_subtracts[channel].getResult()

    return sine_subtracts


class sineSubtractedReader(Reader):
    '''
    This is the same as Reader on all accounts except it will interface with the stored sine subtraction values.
    '''
    def __init__(self, base_dir, run, debug=False):
        self.debug=debug
        self.ss_event_entry = -1
        self.ss_filename = os.path.join(cache_dir, 'sinsub%i.root'%run)
        
        if not os.path.exists(cache_dir):
            print('WARNING! The supplied sine subtraction directory does not exist!  This should be set correctly as environment BEACON_SINE_SUBTRACTION_CACHE if you want to utilize pre-calculated sine subtraction information for quicker analysis.')
            self.ss_event_file = None
            self.ss_event_tree = None
        elif not os.path.exists(self.ss_filename):
            print('WARNING! SINE SUBTRACTION CACHE FILE DOES NOT EXIST: %s'%self.ss_filename)
            self.ss_event_file = None
            self.ss_event_tree = None
        else:
            self.ss_event_file = ROOT.TFile.Open(self.ss_filename,"READ")
            self.ss_event_tree = self.ss_event_file.Get("sinsubcache")
            self.sine_subtracts = prepareStandardSineSubtractions()
            self.r = ROOT.FFTtools.SineSubtractResult()

            for channel in range(8):
                #self.ss_event_tree.Branch("result_ch%i"%channel, ROOT.AddressOf(self.sine_subtracts[channel].getResult()), "result_ch%i"%channel)
                self.ss_event_tree.SetBranchAddress("result_ch%i"%channel,self.r)

        super().__init__(base_dir, run)
        

    def event(self,force_reload = False):
        '''
        Does the required preparations for the event to be loaded.  By default this does nothing if the event is already properly set.
        
        Parameters
        ----------
        force_reload : bool
            Will force this to reset entry info.
        '''
        if self.ss_event_file is None:
            return super().event(force_reload=force_reload)
        else:
            self.ss_event_entry = self.event_entry
            super().event(force_reload=force_reload)

            try:
                if (self.ss_event_entry != self.current_entry or force_reload):
                    self.ss_event_tree.GetEntry(self.current_entry)
                    self.ss_event_entry = self.current_entry 
            except Exception as e:
                print('\nError in %s'%inspect.stack()[0][3])
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

            return self.evt

    def raw_wf(self,channel):
        '''
        Calls the super wf function without performing sine subtraction.
        '''
        return super().wf(channel)

    def wf(self, channel):
        '''
        This will pull the waveform data (returned in adu) for the requested channel.  The channel mapping for the
        2019 prototype is: 
        0: NE, Hpol
        1: NE, Vpol
        2: NW, Hpol
        3: NW, Vpol
        4: SE, Hpol
        5: SE, Vpol
        6: SW, Hpol
        7: SW, Vpol
        
        This is subject to change so always cross reference the run with with those in the know to be sure.
        
        Parameters
        ----------
        ch : int
          The channel you specifically want to read out a signal for.
        '''
        ## stupid hack because for some reason it doesn't always report the right buffer length 
        try:
            if self.ss_event_file is None:
                return super().wf(channel)
            else:
                #ev, ss_ev = self.event() #Want to call the newly defined one to update both trees rather than the original.
                ev = self.event() #Want to call the newly defined one to update both trees rather than the original.

                #Load original wf and make output shell for processed wf.
                original_wf = numpy.copy(numpy.frombuffer(ev.getData(channel), numpy.dtype('float64'), ev.getBufferLength()))
                original_wf -= numpy.mean(original_wf)
                original_wf = original_wf.astype(numpy.double)
                len_wf = len(original_wf)

                #Do the sine subtraction
                output_wf = numpy.zeros(len_wf,dtype=numpy.double)
                if self.debug == True:
                    #self.sine_subtracts[channel].getResult().Print()
                    print(self.sine_subtracts[channel].getResult())
                self.sine_subtracts[channel].subtractCW(len_wf,original_wf.data,len_wf,output_wf, self.sine_subtracts[channel].getResult())

                if False:
                    plt.figure()
                    plt.subplot(2,1,1)
                    plt.plot(original_wf)
                    plt.plot(output_wf)
                    plt.subplot(2,1,2)  

                    #Crude just for testing

                    original_spec = numpy.fft.rfft(original_wf)
                    original_db = 10.0*numpy.log10( 2*original_spec * numpy.conj(original_spec) / len_wf)

                    output_spec = numpy.fft.rfft(output_wf)
                    output_db = 10.0*numpy.log10( 2*output_spec * numpy.conj(output_spec) / len_wf)

                    plt.plot(original_db)
                    plt.plot(output_db)

                return output_wf
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    def sswf(self, channel):
        '''
        This will pull the waveform data (returned in adu) for the requested channel.  The channel mapping for the
        2019 prototype is: 
        0: NE, Hpol
        1: NE, Vpol
        2: NW, Hpol
        3: NW, Vpol
        4: SE, Hpol
        5: SE, Vpol
        6: SW, Hpol
        7: SW, Vpol
        
        This is subject to change so always cross reference the run with with those in the know to be sure.
        
        Parameters
        ----------
        ch : int
          The channel you specifically want to read out a signal for.
        '''
        ## stupid hack because for some reason it doesn't always report the right buffer length 
        try:
            #ev, ss_ev = self.event() #Want to call the newly defined one to update both trees rather than the original.
            ev = self.event() #Want to call the newly defined one to update both trees rather than the original.

            #Load original wf and make output shell for processed wf.
            original_wf = numpy.copy(numpy.frombuffer(ev.getData(channel), numpy.dtype('float64'), ev.getBufferLength()))
            original_wf -= numpy.mean(original_wf)
            original_wf = original_wf.astype(numpy.double)
            len_wf = len(original_wf)

            #Do the sine subtraction
            output_wf = numpy.zeros(len_wf,dtype=numpy.double)
            if self.debug == True:
                #self.sine_subtracts[channel].getResult().Print()
                print(self.sine_subtracts[channel].getResult())
            self.sine_subtracts[channel].subtractCW(len_wf,original_wf.data,len_wf,output_wf)
            print(self.sine_subtracts[channel].getNSines())

            if True:
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(original_wf)
                plt.plot(output_wf)
                plt.subplot(2,1,2)  

                #Crude just for testing

                original_spec = numpy.fft.rfft(original_wf)
                original_db = 10.0*numpy.log10( 2*original_spec * numpy.conj(original_spec) / len_wf)

                output_spec = numpy.fft.rfft(output_wf)
                output_db = 10.0*numpy.log10( 2*output_spec * numpy.conj(output_spec) / len_wf)

                plt.plot(original_db)
                plt.plot(output_db)

            return output_wf
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run = int(sys.argv[1])

        print('Attempting to precalculate sine subtraction terms for run %i'%run)
        #Prepare event reader and sine subtracts
        reader = Reader(datapath,run)
        filename = os.path.join(cache_dir, 'sinsub%i.root'%run)

        sine_subtracts = prepareStandardSineSubtractions()

        #Check if the cache directory exists.  If not make it.  This is where the sin subtraction ROOT files will be stored.
        if not os.path.exists(cache_dir):
            try:
                os.mkdir(cache_dir)
                print('Created cache directory: %s'%cache_dir)
            except Exception as e:
                print('Error in sine_subtraction_cache.__main__() when attempting to make cache directory.')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                sys.exit(0)


        #By here I want the ROOT file created, whether or not I had to do it or it aready existed.
        f = ROOT.TFile.Open(filename,"RECREATE") #Change to CREATE once development is over.
        t = ROOT.TTree("sinsubcache","sinsubcache")
        
        len_wf = len(reader.t())
        output_wf = numpy.zeros(len_wf,dtype=numpy.double)#output_wf is the output array for the subtractCW function, and must be predefined.  Not sure if I can use the same one for all events. 

        for channel in range(8):
            ROOT.dobranch(t,"result_ch%i"%channel,ROOT.addressof(sine_subtracts[channel].getResult()))


        for eventid in range(reader.N()):
            if (eventid + 1) % 100 == 0:
                sys.stdout.write('(%i/%i)\t\t\t\n'%(eventid+1,reader.N()))
                sys.stdout.flush()


            reader.setEntry(eventid)
            for channel in range(8):
                #Get waveform and perform sine subtraction
                temp_wf = reader.wf(int(channel))
                temp_wf -= numpy.mean(temp_wf)
                temp_wf = temp_wf.astype(numpy.double)

                #Do the sine subtraction
                sine_subtracts[channel].subtractCW(len_wf,temp_wf.data,len_wf,output_wf)#*1e-9,output_wf)#dt_ns_original

            #Because the branch already knows to expect the result at the specified address of each sine_subtract, that is where it will look when Fill is called for this entry.
            t.Fill() # fill the tree, it will update all 8 results
        t.Write()
        f.Close()

    else:
        '''
        Testing
        '''
        run = 5732
        eventid = 0
        channel = 1


        reader = Reader(datapath,run)
        reader.setEntry(eventid)
        wf = reader.wf(channel)
        fig = plt.figure()
        ax = plt.gca()
        plt.plot(wf,label='raw')
        
        reader2 = sineSubtractedReader(datapath,run,debug=True)
        reader2.setEntry(eventid)
        wf2 = reader2.wf(channel)
        plt.sca(ax)
        plt.plot(wf2,label='ss_stored')

        wf3 = reader2.sswf(channel)
        plt.sca(ax)
        plt.plot(wf3,label='ss_new')
        plt.legend()

        if numpy.all(wf2 == wf3):
            print('Seems like sine subtraction from stored values is working.')
        #wf2 should equal wf3        
        