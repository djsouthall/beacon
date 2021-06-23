import glob
import os
import sys
import numpy
import sys
import warnings
from pprint import pprint

if __name__=="__main__":
    search_dir = '/home/dsouthall/Projects/Beacon/beacon/tools/' #where to find the slurm files
    min_slurm_id = 0 #Any slurmid before this will be ignored. 
    files = numpy.array(glob.glob(search_dir + 'slurm*.out')) 
    slurmids = numpy.array([int(f.split('-')[-1].replace('.out','')) for f in files])
    files = files[numpy.argsort(slurmids)][numpy.sort(slurmids) >= min_slurm_id] #Sorting by slurm id.

    failed_runs = []
    for f in files:
        with open(f, "r") as file:
            first_line = file.readline()
            last_attempted = ''
            for last_line in file:
                if 'Attempting'in last_line:
                    last_attempted = last_line.replace('\n','')
            #Ends on last line leaving last line as last line.  last attempted will read the last attempted script. 
            try:
                run = int(first_line.split('Run ')[-1].replace('\n',''))
                if 'CANCELLED AT' in last_line:
                    print('%i Cancelled due to: %s  |  %s  |  %s'%(run, last_line.split('DUE TO ')[-1].replace(' ***\n',''), f, last_attempted))
                    failed_runs.append(run)
            except:
                print('Failed Parsing of Run Number:')
                print('\t' + first_line)
    failed_runs = numpy.array(failed_runs)
    print('All failed runs = ')
    pprint(failed_runs)