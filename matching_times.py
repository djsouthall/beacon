'''
This is a script that contains the work of trying to coordinate times listed 
in the BEACON spreadsheet (July 2019) with events in the data.
'''
import numpy
import os
import sys
from pytz import timezone,utc
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader
from analysis.tools.interpret import getReaderDict, getHeaderDict, getStatusDict
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
plt.ion()
import glob

tz = timezone('US/Pacific')
raw_days = {}
raw_days['d1'] = {}
raw_days['d1']['tz'] = tz
raw_days['d1']['date'] = '6-26-2019'
raw_days['d1']['times'] = \
    [  'Pulsing Start Time: 2:53 PM',
        'Pulsing End Time:   3:55 PM',
        'Pulsing Start Time: 4:11 PM',
        'Pulsing End Time:   4:53 PM',
        'Pulsing Start Time: 4:53 PM',
        'Pulsing End Time:   5:24 PM',
        'Pulsing Start Time: 7:17 PM',
        'Pulsing End Time:   7:55 PM',
        'Pulsing Start Time: 8:00 PM',
        'Pulsing End Time:   8:39 PM']

raw_days['d2'] = {}
raw_days['d2']['tz'] = tz
raw_days['d2']['date'] = '6-27-2019'
raw_days['d2']['times'] = \
    [  'Pulsing Start Time: 3:45 PM',
        'Pulsing End Time:   4:16 PM',
        'Pulsing Start Time: 4:17 PM',
        'Pulsing End Time:   5:27 PM']

raw_days['d3'] = {}
raw_days['d3']['tz'] = tz
raw_days['d3']['date'] = '6-28-2019'
raw_days['d3']['times'] = \
    [  'Pulsing Start Time: 6:12 PM',
        'Pulsing End Time:   7:02 PM']

raw_days['d4'] = {}
raw_days['d4']['tz'] = tz
raw_days['d4']['date'] = '6-29-2019'
raw_days['d4']['times'] = \
    [  'Pulsing Start Time: 10:58 AM',
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


raw_days['d5'] = {}
raw_days['d5']['tz'] = tz
raw_days['d5']['date'] = '7-1-2019'
raw_days['d5']['times'] = \
    [  'Pulsing Start Time: 11:19 AM',
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

raw_days['d6'] = {}
raw_days['d6']['tz'] = tz
raw_days['d6']['date'] = '7-2-2019'
raw_days['d6']['times'] = \
    [  'Pulsing Start Time: 11:15 AM',
        'Pulsing End Time:   11:25 AM',
        'Pulsing Start Time: 11:27 AM',
        'Pulsing End Time:   11:39 AM',
        'Pulsing Start Time: 11:39 AM',
        'Pulsing End Time:   11:55 AM',
        'Pulsing Start Time: 11:58 AM',
        'Pulsing End Time:   1:26 PM',
        'Pulsing Start Time: 1:29 PM',
        'Pulsing End Time:   1:46 PM']

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
        dt = raw_days[day_key]['tz'].localize(dt,is_dst=True)
        
        if 'Start' in t:
            starts.append(dt)
        elif 'End' in t:
            ends.append(dt)
    clean_days[day_key] = {}
    clean_days[day_key]['starts'] = numpy.array(starts)
    clean_days[day_key]['ends'] = numpy.array(ends)

if __name__ == '__main__':
    print(clean_days)
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    run_labels = glob.glob(datapath + '/*')
    run_times_unix = numpy.zeros(len(run_labels))
    
    test = 0
    
    flagged_runs = {}
    for day_label in list(clean_days.keys()):
        flagged_runs[day_label] = []


    for run_label in run_labels:
        if 'run' in run_label:
            run = int(run_label.split('run')[-1])
            reader = Reader(datapath,run)
            if reader.N() == 0:
                continue
            min_t = utc.localize(datetime.fromtimestamp(reader.head_tree.GetMinimum('readout_time')),is_dst=None)
            max_t = utc.localize(datetime.fromtimestamp(reader.head_tree.GetMaximum('readout_time')),is_dst=None)
            for day_label in list(clean_days.keys()):
                flag = False
                if numpy.any(numpy.logical_and(min_t > clean_days[day_label]['starts'],min_t < clean_days[day_label]['ends'])):
                    flag = True
                if numpy.any(numpy.logical_and(max_t > clean_days[day_label]['starts'],max_t < clean_days[day_label]['ends'])):
                    flag = True
                if numpy.any(numpy.logical_and(min_t < clean_days[day_label]['starts'],max_t > clean_days[day_label]['ends'])):
                    flag = True
                if flag == True:
                    flagged_runs[day_label].append(run)

    pprint(flagged_runs)
    flagged_runs_dates = {}
    for day_label in list(flagged_runs.keys()):
        flagged_runs_dates[raw_days[day_label]['date']] = flagged_runs[day_label]
    pprint(flagged_runs_dates)