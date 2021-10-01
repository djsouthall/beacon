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
    #runs = numpy.arange(5733,5790)#numpy.arange(5159,5200)#numpy.arange(1643,1729)#numpy.arange(5150,5250)#numpy.arange(1643,1729)#numpy.arange(5150,5250)#numpy.array([1663])#numpy.arange(1643,1729)
    #runs = [5737,5744,5745,5747,5748,5752,5753,5754,5755,5759,5760,5761,5762,5764,5765,5766,5767,5768,5769,5770,5772,5777,5778,5779,5780,5781,5782,5783,5784,5785,5786,5787,5788]
    deploy_index = '/home/dsouthall/Projects/Beacon/beacon/config/september_2021_minimized_calibration.json'

    if False:
        runs = [5737,5744,5745,5747,5748,5752,5753,5754,5755,5759,5760,5761,5762,5764,5765,5766,5767,5768,5769,5770,5772,5777,5778,5779,5780,5781,5782,5783,5784,5785,5786,5787,5788]
        done_runs = numpy.array([5732,5733,5734,5735,5736,5738,5739,5740,5741,5742,5743,5746,5749,5750,5751,5756,5757,5758,5763,5771,5773,5774,5775,5776,5789])
        done_runs = numpy.append(done_runs , [5762,5764]) #TEMPORARY BECAUSE ALL OTHER FILES FINISHED RUNNING PART 1
        pol = 'vpol' #currently only used in rf_bg_search because it runs so long it is split up for safety. 
    elif False:
        runs = [5762,5764]
        done_runs = numpy.array([5732,5733,5734,5735,5736,5738,5739,5740,5741,5742,5743,5746,5749,5750,5751,5756,5757,5758,5763,5771,5773,5774,5775,5776,5789])
        #runs = [5762,5764]
        pol = 'vpol' #currently only used in rf_bg_search because it runs so long it is split up for safety. 
    else:
        runs = numpy.array([5630, 5631, 5632, 5638, 5639, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5656, 5657, 5659, 5660], dtype=int)
        done_runs = numpy.array([])
        pol = 'hpol'


    ###--------###
    ### Script ###
    ###--------###

    for run in runs:
        if run in done_runs:
            continue
        jobname = 'bcn%i'%run

        batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname)

        command = os.environ['BEACON_ANALYSIS_DIR'] + 'analysis/all_analysis_part1.sh %i %s %s'%(run, deploy_index, pol)#'analysis/time_averaged_spectrum.py %i'%(run)#'analysis/all_analysis.sh %i'%(run)#'tools/data_handler.py %i'%(run)#'analysis/time_averaged_spectrum.py %i'%(run)#'tools/data_handler.py %i redo'%(run)#'analysis/cr_search/simple_cr_template_search.py %i 1'%(run)#'tools/data_handler.py %i'%(run)#'analysis/rf_bg_search.py %i'%(run)
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
                