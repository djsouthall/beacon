#! /usr/bin/env python3

'''
This is a script originally written by Cosmin and untested by others.
'''
# EXAMPLE FOR MAKING FAKE DATA


import numpy
import scipy.interpolate
import ROOT
import sys 
import os

#load beaconroot 
ROOT.gSystem.Load("libbeaconroot.so") # must be in your LD_LIBRARY_PATH

class FakeDataMaker: 
    def __init__(self, input_run, output_run, real_data_dir = None, mc_data_dir = None): 
        #inputs
        self.input_run = input_run
        self.output_run = output_run
        self.real_data_dir = os.environ['BEACON_ROOT_DATA'] if real_data_dir is None else real_data_dir
        self.mc_data_dir = os.environ['BEACON_MC_DATA'] if mc_data_dir is None else mc_data_dir


        #setup up input files 
        self.data_ev_file = ROOT.TFile.Open("%s/run%d/event.root" % (self.real_data_dir, self.input_run)); 
        self.data_hd_file = ROOT.TFile.Open("%s/run%d/header.root" % (self.real_data_dir, self.input_run)); 

        self.data_ev_tree = self.data_ev_file.Get("event"); 
        self.data_hd_tree = self.data_hd_file.Get("header"); 

        self.header = ROOT.beacon.Header() 
        self.realevt = ROOT.beacon.Event() 

        self.data_hd_tree.SetBranchAddress("header", self.header)
        self.data_ev_tree.SetBranchAddress("event", self.realevt)

        #   Get first entry, we need the event length 
        self.data_ev_tree.GetEntry(0); 
        self.data_hd_tree.GetEntry(0); 

        self.data_index = -1 # even though we got the first entry, we will need to scan for next force trigger
        self.num_processed = 0


        #setup output filesand director
        os.makedirs("%s/run%d" % (self.mc_data_dir, self.output_run))

        self.mc_hd_file = ROOT.TFile.Open("%s/run%d/header.root" % (self.mc_data_dir, self.output_run), "CREATE") 
        #    self.mc_hd_file.cd()
        self.mc_hd_tree = ROOT.TTree("header","header") 
        # we can use the same header from the raw data, though we'll typically update it 
        #print("About to branch header")
        self.mc_hd_tree.Branch("header", "beacon::Header",  self.header)



        self.mc_ev_file = ROOT.TFile.Open("%s/run%d/event.root" % (self.mc_data_dir, self.output_run), "CREATE") 
        #    self.mc_ev_file.cd()
        self.mc_event_tree = ROOT.TTree("event","event")
        self.mc_event = ROOT.beacon.SimEvent(self.output_run, self.realevt.getBufferLength())
        #print("About to branch event")
        self.mc_event_tree.Branch("event", "beacon::SimEvent",self.mc_event)

# sets up the next force trigger, returning false if there is none
def nextForceTrigger(self): 
    while True: 
        if (self.data_index+1 >= self.data_hd_tree.GetEntries()):
            return False 


        self.data_index+=1 
        self.data_hd_tree.GetEntry(self.data_index)

        if (self.header.trigger_type == ROOT.beacon.TRIGGER_SW or self.header.trigger_type == ROOT.beacon.TRIGGER_EXT): 

            #modify the header appropriately 
            self.header.trigger_type = ROOT.beacon.TRIGGER_RF # now it's an RF trigger! 
            self.header.trigger_pol = ROOT.beacon.TRIGGER_HPOL # assume this is the only valid thing
            self.num_processed+=1
            self.header.event_number = self.output_run*1000000000 + self.num_processed; 
            self.header.trig_number = self.header.event_number 

            #TODO: jitter the trigger time, etc. 

            # get the waveforms
            self.data_ev_tree.GetEntry(self.data_index) 

            return True




# truth_dict contains  'vertex' (3 numbers), 'polarization' (3 numbers), 'signal_amp' (8 numbers), 'arrival_times' (8 numbers), 'phi', 'theta', and 'E'
# wf_V is the "truth" waveform to add, that will be scaled by signal_amp and delayed by arrival_times (it can either be one waveform or one per channel, but the transformations are applied regardless) 
# wf_dt is the sample period length of wf
# wf_t0 is the t0 of the truth (should be somewhere in the middle of the real waveform 
def makeFakeEvent(self,  truth_dict, wf_V, wf_dt, wf_t0 = 300): 
    #start by incrementing the event number
    self.mc_event.incrementEventNumber() 

    truth = ROOT.beacon.MCTruth()

    if 'signal_amp' not in truth_dict or len(truth_dict['signal_amp'])!=8: 
        raise ValueError('signal_amp not as expected in truth')  

    if 'arrival_times' not in truth_dict or len(truth_dict['arrival_times'])!=8: 
        raise ValueError('arrival_times not as expected in truth')  

    for i in range(8):
        truth.arrival_times[i] = float(truth_dict['arrival_times'][i])
        truth.signal_amp[i] = float(truth_dict['signal_amp'][i])

    #more optional things
    if 'vertex' in truth_dict and len(truth_dict['vertex']) == 3: 
        for i in range(3):
            truth.vertex[i] = float(truth_dict['vertex'][i])

    if 'pol' in truth_dict and len(truth_dict['pol']) == 3: 
        for i in range(3):
            truth.pol[i] = float(truth_dict['pol'][i])

    if 'E' in truth_dict: 
        truth.E = truth_dict['E']

    if 'array_phi' in truth_dict: 
        truth.array_phi = truth_dict['array_phi']
    if 'array_theta' in truth_dict: 
        truth.array_theta = truth_dict['array_theta']

    individual_events = wf_V.ndim == 2

    truth.true_dt = wf_dt
    truth.true_t0 = wf_t0 

    for ichan in range(8): 
        if individual_events: 
            truth.setTrueWF(ichan, wf_V[ichan].size, wf_V[ichan])
        else: 
            #print(wf_V)
            truth.setTrueWF(ichan, wf_V.size, wf_V)

    self.mc_event.setTruth(truth) 

    for ichan in range(8): 
        forced_wf = numpy.frombuffer(self.realevt.getData(ichan), numpy.dtype('float64'), self.realevt.getBufferLength())

        wf = wf_V[ichan] if wf_V.ndim == 2 else wf_V

        #we need to add the delayed and scaled truewf to the fake data. 

        #####TODO TODO TODO TODO
        # the right way to do this is to use the Fourier Shift theorem and add in the fourier domain (to properly handle fractional delays). 
        #But then you have to deal with resampling, and I don't feel like doing that right now

        # I'm going to be really lazy and just interpolate for now, and someone will fix this eventually!
        # If the truth waveform is way oversampled, then this is going to be pretty close anyway
        #####TODO TODO TODO TODO

        wf_t = numpy.arange(wf_t0, wf_t0 + wf.size * wf_dt, wf_dt) 
        interp = scipy.interpolate.interp1d(wf_t, wf, fill_value = 0, bounds_error = False, assume_sorted ='True')

        data_t = numpy.arange(truth.arrival_times[ichan], truth.arrival_times[ichan] + self.realevt.getBufferLength() * 2, 2)  # 2 ns spacing 
        forced_wf += interp( data_t) * truth.signal_amp[ichan]
        self.mc_event.digitize(ichan, forced_wf) 

    print("Filled ",self.mc_event.getEventNumber())
    #make sure we fill 
    #  self.mc_ev_file.cd() 
    self.mc_event_tree.Fill() 
    #  self.mc_hd_file.cd()
    self.mc_hd_tree.Fill() 



    def finalize(self):
        self.mc_ev_file.cd() 
        self.mc_event_tree.Write() 
        self.mc_ev_file.Close()

        self.mc_hd_file.cd() 
        self.mc_hd_tree.Write() 
        self.mc_hd_file.Close()





if __name__=="__main__": 
    maker = FakeDataMaker(6500,1000, mc_data_dir = './test')

    while maker.nextForceTrigger(): 
        truth_dict = {}
        truth_dict['E'] = 1e19 #why not
        truth_dict['pol'] = (1,2,3) # not the real polarization
        truth_dict['array_phi'] = 0 # not the true phi
        truth_dict['array_theta'] = 0 # not the true theta
        truth_dict['vertex'] = (1,2,3) # note the real vertex
        truth_dict['signal_amp'] = (1.1,0.2,1.1,0.4,1.1,0.6,1.7,0.8) # lol 
        truth_dict['arrival_times'] = (2,2,4,4,6,6,8,8) # lol 

        totally_a_cosmic_ray = numpy.array([0,25,-20,15,-10,2,-2,0], dtype='float64')
        maker.makeFakeEvent(truth_dict, totally_a_cosmic_ray, 5, 300) 

    maker.finalize() 
