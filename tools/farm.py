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
        analysis_part = 2
    elif False:
        runs = [5762,5764]
        done_runs = numpy.array([5732,5733,5734,5735,5736,5738,5739,5740,5741,5742,5743,5746,5749,5750,5751,5756,5757,5758,5763,5771,5773,5774,5775,5776,5789])
        #runs = [5762,5764]
        pol = 'vpol' #currently only used in rf_bg_search because it runs so long it is split up for safety. 
        analysis_part = 2
    elif False:
        runs = numpy.arange(5790,5974,dtype=int)
        done_runs = numpy.array([])
        pol = 'hpol'
        analysis_part = 1
    elif False:
        #runs = numpy.arange(5790,5974,dtype=int)
        runs = numpy.arange(5733,5790,dtype=int)
        done_runs = numpy.array([])
        analysis_part = 2
        pol == 'both'
    elif True:
        #runs = numpy.arange(5790,5974,dtype=int)
        runs = numpy.arange(5733,5974,dtype=int)
        done_runs = numpy.array([])
        analysis_part = 1 #Need to run 2 at some point after 11//23/2021 if things worked on part 1
        pol = 'both' #Always treated as both when analysis_part == 3
    else:
        runs = numpy.array([5630, 5631, 5632, 5638, 5639, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5656, 5657, 5659, 5660], dtype=int)
        done_runs = numpy.array([])
        pol = 'vpol'
        analysis_part = 2


    ###--------###
    ### Script ###
    ###--------###

    for run in runs:
        if run in done_runs:
            continue

        jobname = 'bcn%i'%run

        if True and analysis_part == 2:
            script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part2.sh')

            if pol == 'both':
            
                #Prepare Hpol Job
                batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname+'h')
                command = '%s %i %s %s'%(script, run, deploy_index, 'hpol')
                command_queue = batch + command

                #Submit hpol job and get the jobid to then submit vpol with dependency
                print(command_queue)
                hpol_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

                
                #Prepare Vpol Job
                batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition,jobname+'v', hpol_jobid)
                command = '%s %i %s %s'%(script, run, deploy_index, 'vpol')
                command_queue = batch + command

                #Submit vpol job
                print(command_queue)
                vpol_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

                print('Run %i jobs submitted --> HPol jid:%i\tVPol jid:%i'%(run,hpol_jobid,vpol_jobid))
            elif pol == 'hpol':
                #Prepare Hpol Job
                batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname+'h')
                command = '%s %i %s %s'%(script, run, deploy_index, 'hpol')
                command_queue = batch + command

                #Submit hpol job and get the jobid to then submit vpol with dependency
                print(command_queue)
                hpol_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))
            elif pol == 'vpol':
                #Prepare Vpol Job
                batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname+'v')
                command = '%s %i %s %s'%(script, run, deploy_index, 'vpol')
                command_queue = batch + commandW

                #Submit hpol job and get the jobid to then submit vpol with dependency
                print(command_queue)
                hpol_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

        elif analysis_part == 1:
            script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part1.sh')
            batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname)
            command = script + ' %i %s'%(run, deploy_index)
            command_queue = batch + command
            print(command_queue)    
            os.system(command_queue) # Submit to queue
        elif analysis_part == 3:
            script1 = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part1.sh')
            script2 = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part2.sh')

            #Prepare Script 1
            batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname+'s1')
            command = '%s %i'%(script1, run)
            command_queue = batch + command

            #Submit script1 job and get the jobid to then submit vpol with dependency
            print(command_queue)
            script1_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

            #Prepare Hpol Job
            batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition,jobname+'h', script1_jobid)
            command = '%s %i %s %s'%(script2, run, deploy_index, 'hpol')
            command_queue = batch + command

            #Submit hpol job and get the jobid to then submit vpol with dependency
            print(command_queue)
            hpol_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

            
            #Prepare Vpol Job
            batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition,jobname+'v', hpol_jobid)
            command = '%s %i %s %s'%(script2, run, deploy_index, 'vpol')
            command_queue = batch + command

            #Submit vpol job
            print(command_queue)
            vpol_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

            print('Run %i jobs submitted --> HPol jid:%i\tVPol jid:%i'%(run,hpol_jobid,vpol_jobid))

        else:
            script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'analyze_event_rate_frequency.py')
            batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname)
            command = script + ' %i'%(run)
            command_queue = batch + command
            print(command_queue)    
            os.system(command_queue) # Submit to queue




