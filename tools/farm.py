#!/usr/bin/env python3
'''
This script is meant to submit multiple similar jobs as sbatch to slurm. 
'''
import os
import sys
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import subprocess
import time
import numpy
import yaml

if __name__ == "__main__":

    ###------------###
    ### Parameters ###
    ###------------###

    username = 'dsouthall'
    #mem = '16G'
    partition = 'broadwl'
    #Approximately the First week of september: numpy.array(5733,5790) 
    runs = numpy.array(5733,5790)#numpy.arange(5159,5200)#numpy.arange(1643,1729)#numpy.arange(5150,5250)#numpy.arange(1643,1729)#numpy.arange(5150,5250)#numpy.array([1663])#numpy.arange(1643,1729)
    deploy_index = '/home/dsouthall/Projects/Beacon/beacon/config/september_2021_minimized_calibration.json'
    done_runs = numpy.array([])

    ###--------###
    ### Script ###
    ###--------###


    for run in runs:
        if run in done_runs:
            continue
        jobname = 'bcn%i'%run

        batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname)

        command = os.environ['BEACON_ANALYSIS_DIR'] + 'analysis/all_analysis_part1.sh %i %s'%(run, deploy_index)#'analysis/time_averaged_spectrum.py %i'%(run)#'analysis/all_analysis.sh %i'%(run)#'tools/data_handler.py %i'%(run)#'analysis/time_averaged_spectrum.py %i'%(run)#'tools/data_handler.py %i redo'%(run)#'analysis/cr_search/simple_cr_template_search.py %i 1'%(run)#'tools/data_handler.py %i'%(run)#'analysis/rf_bg_search.py %i'%(run)
        #command = os.environ['BEACON_ANALYSIS_DIR'] + 'analysis/rf_bg_search.py %i'%(run)#'analysis/all_analysis.sh %i'%(run)#'analysis/rf_bg_search.py %i'%(run)#'analysis/all_analysis.sh %i'%(run)#'analysis/time_averaged_spectrum.py %i'%(run)#'analysis/all_analysis.sh %i'%(run)#'tools/data_handler.py %i'%(run)#'analysis/time_averaged_spectrum.py %i'%(run)#'tools/data_handler.py %i redo'%(run)#'analysis/cr_search/simple_cr_template_search.py %i 1'%(run)#'tools/data_handler.py %i'%(run)#'analysis/rf_bg_search.py %i'%(run)

        command_queue = batch + command
        print(command_queue)    
        os.system(command_queue) # Submit to queue

        # Avoid overwhelming the queue with jobs
        while False:

            # Clean up log files
            n_output = subprocess.Popen('ls slurm*.out | wc', shell=True, 
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].split()[0]
            if n_output.isdigit():
                os.system('rm slurm*.out')

            n_submitted = int(subprocess.Popen('squeue -u %s | wc\n'%username, shell=True, 
                                               stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate()[0].split()[0]) - 1
            
            # Check to see whether to enter holding pattern
            if n_submitted < 200:
                break
            else:
                print('%i jobs already in queue, waiting ...'%(n_submitted), time.asctime(time.localtime()))
                time.sleep(60)
                