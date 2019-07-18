'''
This is a script that contains the work of trying to coordinate times listed 
in the BEACON spreadsheet (July 2019) with events in the data.
'''
import numpy
import os
import sys
from pytz import timezone,utc
from datetime import datetime
from pprint import pprint
import glob

sys.path.append(os.environ['BEACON_INSTALL_DIR']) 
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.interpret import getReaderDict, getHeaderDict, getStatusDict #Must be imported before matplotlib or else plots don't load.

import matplotlib.pyplot as plt
plt.ion()

tz = timezone('US/Pacific')
raw_days = {}
raw_days['d1'] = {}
raw_days['d1']['tz'] = tz
raw_days['d1']['date'] = '6-26-2019'
raw_days['d1']['times'] = \
    [   'Pulsing Start Time: 2:53 PM',
        'Pulsing End Time:   3:55 PM',
        'Pulsing Start Time: 4:11 PM',
        'Pulsing End Time:   4:53 PM',
        'Pulsing Start Time: 4:53 PM',
        'Pulsing End Time:   5:24 PM',
        'Pulsing Start Time: 7:17 PM',
        'Pulsing End Time:   7:55 PM',
        'Pulsing Start Time: 8:00 PM',
        'Pulsing End Time:   8:39 PM']
raw_days['d1']['cfg'] = numpy.array([\
                            'site-1 cfg-1',
                            'site-1 cfg-2',
                            'site-1 cfg-3',
                            'site-2 cfg-1',
                            'site-2 cfg-2'])

raw_days['d2'] = {}
raw_days['d2']['tz'] = tz
raw_days['d2']['date'] = '6-27-2019'
raw_days['d2']['times'] = \
    [   'Pulsing Start Time: 3:45 PM',
        'Pulsing End Time:   4:16 PM',
        'Pulsing Start Time: 4:17 PM',
        'Pulsing End Time:   5:27 PM']
raw_days['d2']['cfg'] = numpy.array([\
                            'site-2 cfg-1',
                            'site-2 cfg-2'])

raw_days['d3'] = {}
raw_days['d3']['tz'] = tz
raw_days['d3']['date'] = '6-28-2019'
raw_days['d3']['times'] = \
    [   'Pulsing Start Time: 6:12 PM',
        'Pulsing End Time:   7:02 PM']
raw_days['d3']['cfg'] = numpy.array([\
                            'site-A cfg-1'])

raw_days['d4'] = {}
raw_days['d4']['tz'] = tz
raw_days['d4']['date'] = '6-29-2019'
raw_days['d4']['times'] = \
    [   'Pulsing Start Time: 10:58 AM',
        'Pulsing End Time:   11:32 AM',
        'Pulsing Start Time: 11:40 AM',
        'Pulsing End Time:   11:59 AM',
        'Pulsing Start Time: 12:00 PM',
        'Pulsing End Time:   12:48 PM',
        'Pulsing Start Time: 12:49 PM',
        'Pulsing End Time:   1:45 PM',
        'Pulsing Start Time: 4:55 PM',
        'Pulsing End Time:   5:42 PM',
        'Pulsing Start Time: 5:43 PM',
        'Pulsing End Time:   5:54 PM',
        'Pulsing Start Time: 5:57 PM',
        'Pulsing End Time:   6:51 PM',
        'Pulsing Start Time: 6:59 PM',
        'Pulsing End Time:   7:08 PM',
        'Pulsing Start Time: 7:09 PM',
        'Pulsing End Time:   7:21 PM']
raw_days['d4']['cfg'] = numpy.array([\
                            'site-4ish cfg-1a',
                            'site-4ish cfg-1b',
                            'site-4ish cfg-1c',
                            'site-4ish cfg-2',
                            'site-LOS cfg-1a',
                            'site-LOS cfg-1b',
                            'site-LOS cfg-2',
                            'site-LOS cfg-3',
                            'site-LOS cfg-4'])



raw_days['d5'] = {}
raw_days['d5']['tz'] = tz
raw_days['d5']['date'] = '7-1-2019'
raw_days['d5']['times'] = \
    [   'Pulsing Start Time: 11:19 AM', 
        'Pulsing End Time:   11:22 AM',
        'Pulsing Start Time: 11:23 AM', 
        'Pulsing End Time:   11:38 AM',
        'Pulsing Start Time: 11:59 AM',
        'Pulsing End Time:   12:01 PM',
        'Pulsing Start Time: 12:07 PM',
        'Pulsing End Time:   12:10 PM',
        'Pulsing Start Time: 12:11 PM',
        'Pulsing End Time:   12:15 PM',
        'Pulsing Start Time: 12:17 PM',
        'Pulsing End Time:   12:21 PM',
        'Pulsing Start Time: 12:37 PM',
        'Pulsing End Time:   2:21 PM',
        'Pulsing Start Time: 2:22 PM',
        'Pulsing End Time:   3:26 PM',
        'Pulsing Start Time: 3:28 PM',
        'Pulsing End Time:   3:29 PM',
        'Pulsing Start Time: 4:39 PM',
        'Pulsing End Time:   4:43 PM',
        'Pulsing Start Time: 4:45 PM',
        'Pulsing End Time:   5:32 PM',
        'Pulsing Start Time: 5:34 PM',
        'Pulsing End Time:   6:15 PM',
        'Pulsing Start Time: 6:24 PM',
        'Pulsing End Time:   6:37 PM',
        'Pulsing Start Time: 6:37 PM',
        'Pulsing End Time:   6:42 PM',
        'Pulsing Start Time: 6:42 PM',
        'Pulsing End Time:   6:54 PM',
        'Pulsing Start Time: 7:11 PM',
        'Pulsing End Time:   7:27 PM',
        'Pulsing Start Time: 7:30 PM',
        'Pulsing End Time:   7:40 PM']
raw_days['d5']['cfg'] = numpy.array([\
                            'site-mtn1 cfg-1',
                            'site-mtn1 cfg-2a',
                            'site-mtn1 cfg-2b',
                            'site-mtn1 cfg-2c',
                            'site-mtn1 cfg-3a',
                            'site-mtn1 cfg-3b',
                            'site-mtn1 cfg-4',
                            'site-mtn1 cfg-5',
                            'site-mtn1 cfg-6',
                            'site-mtn1 cfg-7',
                            'site-mtn1 cfg-8',
                            'site-mtn1 cfg-9',
                            'site-mtn1 cfg-10',
                            'site-mtn1 cfg-11',
                            'site-mtn1 cfg-12',
                            'site-mtn1 cfg-13',
                            'site-mtn1 cfg-14'])

raw_days['d6'] = {}
raw_days['d6']['tz'] = tz
raw_days['d6']['date'] = '7-2-2019'
raw_days['d6']['times'] = \
    [   'Pulsing Start Time: 11:15 AM',
        'Pulsing End Time:   11:25 AM',
        'Pulsing Start Time: 11:27 AM',
        'Pulsing End Time:   11:39 AM',
        'Pulsing Start Time: 11:39 AM',
        'Pulsing End Time:   11:55 AM',
        'Pulsing Start Time: 11:58 AM',
        'Pulsing End Time:   1:26 PM',
        'Pulsing Start Time: 1:29 PM',
        'Pulsing End Time:   1:46 PM']
raw_days['d6']['cfg'] = numpy.array([\
                            'site-mtn2 cfg-1',
                            'site-mtn2 cfg-2',
                            'site-mtn2 cfg-3',
                            'site-mtn2 cfg-4',
                            'site-mtn2 cfg-5'])

clean_days = {}
for day_key in list(raw_days.keys()):
    rd = raw_days[day_key]['times']
    starts = []
    ends = []
    for t in rd:
        new_t = t.replace(' ','').split('Time:')[-1]
        
        year = int(raw_days[day_key]['date'].split('-')[2])
        month = int(raw_days[day_key]['date'].split('-')[0])
        day = int(raw_days[day_key]['date'].split('-')[1])
        base_hour = int(new_t.split(':')[0])
        if 'PM' in new_t:
            if base_hour == 12:
                hour = 12
            else:
                hour = base_hour + 12
        else:
            if base_hour == 12:
                hour = 0
            else:
                hour = base_hour
        minute = int(new_t.split(':')[1].replace('AM','').replace('PM',''))
        dt = datetime( year = year, month = month, day = day, hour = hour, minute = minute )
        dt = raw_days[day_key]['tz'].localize(dt,is_dst=True).timestamp()
        
        if 'Start' in t:
            starts.append(dt)
        elif 'End' in t:
            ends.append(dt)
    clean_days[day_key] = {}
    clean_days[day_key]['starts'] = numpy.array(starts)
    clean_days[day_key]['ends'] = numpy.array(ends)

if __name__ == '__main__':
    #print(clean_days)
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    run_labels = numpy.array(glob.glob(datapath + '/run*'))
    run_labels = run_labels[numpy.argsort([int(run_label.split('run')[-1]) for run_label in run_labels])] #sorting by run number.
    run_times_unix = numpy.zeros(len(run_labels))


    flagged_runs = {}
    for day_label in list(clean_days.keys()):
        flagged_runs[day_label] = []
    flagged_runs_cfg = {}
    for day_label in list(clean_days.keys()):
        flagged_runs_cfg[day_label] = []


    min_ts = []
    max_ts = []
    run_ids = []
    print('')
    for run_index,run_label in enumerate(run_labels):
        if 'run' in run_label:
            run = int(run_label.split('run')[-1])
            reader = Reader(datapath,run)
            if reader.N() == 0:
                continue

            sys.stdout.write('\r%i/%i'%(run_index+1,len(run_labels)))
            sys.stdout.flush()
            min_t = reader.head_tree.GetMinimum('readout_time')#utc.localize(datetime.fromtimestamp(reader.head_tree.GetMinimum('readout_time')))
            max_t = reader.head_tree.GetMaximum('readout_time')#utc.localize(datetime.fromtimestamp(reader.head_tree.GetMaximum('readout_time')))
            min_ts.append(min_t)
            max_ts.append(max_t)
            run_ids.append(run)

            for day_label in list(clean_days.keys()):
                cfg = numpy.array([],dtype=int)

                #Leading end in window.
                if numpy.any(numpy.logical_and(min_t > clean_days[day_label]['starts'],min_t < clean_days[day_label]['ends'])):
                    cfg = numpy.append(cfg, numpy.where(numpy.logical_and(min_t > clean_days[day_label]['starts'],min_t < clean_days[day_label]['ends']))[0])
                
                #Tail end in window.
                if numpy.any(numpy.logical_and(max_t > clean_days[day_label]['starts'],max_t < clean_days[day_label]['ends'])):
                    cfg = numpy.append(cfg, numpy.where(numpy.logical_and(max_t > clean_days[day_label]['starts'],max_t < clean_days[day_label]['ends']))[0])
                
                #Run overlaps window entirely.
                if numpy.any(numpy.logical_and(min_t < clean_days[day_label]['starts'],max_t > clean_days[day_label]['ends'])):
                    cfg = numpy.append(cfg, numpy.where(numpy.logical_and(min_t < clean_days[day_label]['starts'],max_t > clean_days[day_label]['ends']))[0])
                
                if len(cfg) > 0:
                    cfg = numpy.unique(cfg.flatten())
                    flagged_runs[day_label].append(run)
                    flagged_runs_cfg[day_label].append(cfg)

    print('')
    min_ts = numpy.array(min_ts)
    max_ts = numpy.array(max_ts)
    run_ids = numpy.array(run_ids)

    print('')

    #pprint(flagged_runs)
    for day_label in list(flagged_runs.keys()):
        print(raw_days[day_label]['date'])
        for i, run in enumerate(flagged_runs[day_label]):
            print('  %i'%run)
            for cfg in raw_days[day_label]['cfg'][flagged_runs_cfg[day_label][i]]:
                print('    ' + cfg)
        print('')

    interest_time_unix = 1561768000
    #interest_time = utc.localize(datetime.fromtimestamp(interest_time_unix),is_dst=None)
    cut = numpy.logical_and(min_ts < interest_time_unix, max_ts > interest_time_unix)
    print('Unix timestamp %i is in run %i'%(1561768000,numpy.array(run_ids)[cut][0]))

    '''
    #Runs by date with configurations in each run. 
    6-26-2019
      734
        site-1 cfg-1
      735
        site-1 cfg-2
      736
        site-1 cfg-2
        site-1 cfg-3
      737
        site-1 cfg-3
      739
        site-2 cfg-1
        site-2 cfg-2
      740
        site-2 cfg-2

    6-27-2019
      746
        site-2 cfg-1
        site-2 cfg-2
      747
        site-2 cfg-2

    6-28-2019
      756
        site-A cfg-1
      757
        site-A cfg-1

    6-29-2019
      762
        site-4ish cfg-1a
      763
        site-4ish cfg-1a
        site-4ish cfg-1b
        site-4ish cfg-1c
      764
        site-4ish cfg-1c
        site-4ish cfg-2
      766
        site-LOS cfg-1a
        site-LOS cfg-1b
      767
        site-LOS cfg-1b
        site-LOS cfg-2
      768
        site-LOS cfg-2
      769
        site-LOS cfg-2
      770
        site-LOS cfg-3
        site-LOS cfg-4

    7-1-2019
      781
        site-mtn1 cfg-1
        site-mtn1 cfg-2a
      782
        site-mtn1 cfg-2b
        site-mtn1 cfg-2c
        site-mtn1 cfg-3a
        site-mtn1 cfg-3b
        site-mtn1 cfg-4
      783
        site-mtn1 cfg-4
      784
        site-mtn1 cfg-4
      785
        site-mtn1 cfg-4
        site-mtn1 cfg-5
      786
        site-mtn1 cfg-5
      787
        site-mtn1 cfg-5
      788
        site-mtn1 cfg-5
        site-mtn1 cfg-6
        site-mtn1 cfg-7
        site-mtn1 cfg-8
        site-mtn1 cfg-9
      789
        site-mtn1 cfg-9
        site-mtn1 cfg-10
        site-mtn1 cfg-11
        site-mtn1 cfg-12
      790
        site-mtn1 cfg-13
        site-mtn1 cfg-14

    7-2-2019
      792
        site-mtn2 cfg-1
        site-mtn2 cfg-2
        site-mtn2 cfg-3
        site-mtn2 cfg-4
      793
        site-mtn2 cfg-4
        site-mtn2 cfg-5
    '''