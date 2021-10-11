#!/usr/bin/env python3
'''
This script is intended to save eventids for force trigger events for Steven Prohira to work with.
'''
import os
import sys
import numpy
import csv
from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_handler import createFile, getTimes, loadTriggerTypes, getEventTimes
raw_datapath = os.environ['BEACON_DATA']

if __name__ == '__main__':
    runs = numpy.arange(5733,5790)#numpy.array([5733])
    all_runs = numpy.array([],dtype=int)
    all_eventids = numpy.array([],dtype=int)
    all_trigger_types = numpy.array([],dtype=int)
    for run in runs:
        reader = Reader(raw_datapath,run)
        t = loadTriggerTypes(reader)
        cut = t !=  2
        all_runs = numpy.append(all_runs, run*numpy.ones(sum(cut)))
        all_eventids = numpy.append(all_eventids,numpy.where(cut)[0])
        all_trigger_types = numpy.append(all_trigger_types, t[cut])
        print('%i Force Triggers = %0.2f Percent'%(run, 100*sum(cut)/len(cut)))
    out = numpy.vstack((all_runs, all_eventids, all_trigger_types)).T

    if len(runs) == 1:
        savename = os.path.join('./', 'force_trigger_events_run_%i.csv'%runs[0])
    else:
        savename = os.path.join('./', 'force_trigger_events_runs_%i-%i.csv'%(min(runs), max(runs)))
    numpy.savetxt(savename, out.astype(int), delimiter=',', fmt='%i', header = 'run, eventid, trigtype')
