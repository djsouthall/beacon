#!/usr/bin/env python3
'''
This script will run through the root files for the runids given, and print out how many events are in each event type. 
'''
import os
import sys
from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
import numpy
import inspect

def loadTriggerTypes(reader):
    '''
    Will get a list of trigger types corresponding to all eventids for the given reader
    trigger_type:
    1 Software
    2 RF
    3 GPS
    '''
    #trigger_type = numpy.zeros(reader.N())
    '''
    N = reader.head_tree.Draw("raw_approx_trigger_time_nsecs:raw_approx_trigger_time:trig_time:Entry$","","goff") 
    #ROOT.gSystem.ProcessEvents()
    subtimes = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N)
    times = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N) 
    trigtimes = numpy.frombuffer(reader.head_tree.GetV3(), numpy.dtype('float64'), N)
    eventids = numpy.frombuffer(reader.head_tree.GetV4(), numpy.dtype('float64'), N).astype(int)
    '''


    try:
        N = reader.head_tree.Draw("trigger_type","","goff") 
        trigger_type = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N).astype(int)
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print('Error while trying to copy header elements to attrs.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return e
    return trigger_type

if __name__ == "__main__":
    runs = numpy.arange(1630,1729)
    datapath = os.environ['BEACON_DATA']

    total_1 = 0
    total_2 = 0
    total_3 = 0

    non_zero_trig2 = []

    for run in runs:
        reader = Reader(datapath,run)
        if reader.failed_setup == False:
            trigger_type = numpy.asarray(loadTriggerTypes(reader))
            c1 = sum(trigger_type == 1)
            c2 = sum(trigger_type == 2)
            c3 = sum(trigger_type == 3)
            exists = 'T'
        else:
            c1 = 0
            c2 = 0
            c3 = 0
            exists = 'F'

        total_1 += c1
        total_2 += c2
        total_3 += c3

        if c2 > 0:
            non_zero_trig2.append(run)

        print('run {:<5d} exists: {:s}    1: {:<8d}    2: {:<8d}    3: {:<8d}'.format(run,exists,c1,c2,c3))

    print('\nAll runs               1: {:<8d}    2: {:<8d}    3: {:<8d}'.format(total_1,total_2,total_3))

    print('\nRuns with > 0 Trigger Type = 2:')
    print(non_zero_trig2)
