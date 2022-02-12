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
from pprint import pprint


def parseJobIDs(partition,user,max_expected_column_width=8):
    '''
    Will return a dictionary of job ids for the set of jobs currently queued for user on partition.
    '''
    truncated_user = user[0:min(max_expected_column_width,len(user))]
    text = subprocess.check_output(['squeue','--user=dsouthall']).decode('utf-8')

    out_dict = {}
    for line in text.replace(' ','').split('\n'):
        if 'JOBID' in line or partition not in line:
            continue
        #import pdb; pdb.set_trace()
        # print(line)
        # print('---')
        jobid = line.split(partition)[0]
        key = line.split(partition)[1].split(truncated_user)[0]
        out_dict[key] = jobid
    return out_dict




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
    elif False:
        #THE ONE I LIkE TO DO NOW
        runs = numpy.arange(5733,5974,dtype=int)#numpy.array([5864])#numpy.arange(5733,5974,dtype=int)
        done_runs = numpy.array([5775])
        analysis_part = 3
        # Jan/15/2022
        # redo for just all numpy.array([5821,5808])
        # redo for hv and all numpy.array([5954,5949,5947,5925,5921,5943,5905,5903,5941,5901,5899,5893,5936])
        #runs = numpy.array([5954,5949,5947,5925,5921,5943,5905,5903,5941,5901,5899,5893,5936,5821,5808])
    elif False:
        runs = numpy.arange(5733,5974,dtype=int)
        done_runs = numpy.array([])
        analysis_part = 5 # Sine subtraction
    elif True:
        # This will run sine subtraction, the analysis, then maps.  Makes 4 jobs per run, so limit to less than 125 runs
        # at a time.

        # Add 1 to batch_number to get next set of runs, starting at 0, submits 400 jobs, and a total allowed is 500,
        # so don't submit unless < 100 jobs in queue.
        batch_number = 5 

        # batch_number = 0 executed on 2/ 4/2022 , 5974 - 6073
        # batch_number = 1 executed on 2/ 6/2022 , 6074 - 6173
        # batch_number = 2 executed on 2/ 8/2022 , 6174 - 6273
        # batch_number = 3 executed on 2/ 9/2022 , 6274 - 6373
        # batch_number = 4 executed on 2/11/2022 , 6374 - 6473
        # batch_number = 5 not yet executed      , 6474 - 6573
        # batch_number = 6 not yet executed      , 6574 - 6640


        batch_length = 100
        max_run_to_include = 6640
        runs = 5974 + batch_number*batch_length + numpy.arange(batch_length)
        runs = runs[runs <= max_run_to_include]
        done_runs = numpy.array([])
        analysis_part = 4

        bad_node_numbers = [15]
        if len(runs) == 0:
            print('You dont need to submit any more jobs, you have done it all.')
            import pdb; pdb.set_trace()
        else:
            print('Submitting Jobs for Runs:')
            pprint(runs)
            print('Is this okay?  Press c to proceed, crtl-d to exit.')
            import pdb; pdb.set_trace()

        if len(bad_node_numbers) == 0:
            bad_node_string = ''
            bad_node_list = []
        else:
            bad_node_string = "--exclude=midway2-%s"%str(['{:04d}'.format(node) for node in bad_node_numbers]).replace("'","")
            bad_node_list = ["midway2-{:04d}".format(node) for node in bad_node_numbers]


    else:
        runs = numpy.array([5630, 5631, 5632, 5638, 5639, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5656, 5657, 5659, 5660], dtype=int)
        done_runs = numpy.array([])
        pol = 'vpol'
        analysis_part = 2


    ###--------###
    ### Script ###
    ###--------###


    max_expected_column_width = 8
    jobid_dict = parseJobIDs(partition,username,max_expected_column_width=max_expected_column_width)

    for run in runs:
        print('\nRun %i'%run)
        if run in done_runs:
            continue

        jobname = 'bcn%i'%run

        if analysis_part == 1:
            script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part1.sh')
            batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname)
            command = script + ' %i %s'%(run, deploy_index)
            command_queue = batch + command
            print(command_queue)    
            os.system(command_queue) # Submit to queue

        elif True and analysis_part == 2:
            #Run hpol and vpol jobs in same job, and run the 'all' case as a seperate job. 

            script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part2.sh')

            
            #Prepare Hpol Job
            batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname+'hv')
            command = '%s %i %s %s'%(script, run, deploy_index, 'both')
            command_queue = batch + command

            #Submit hpol job and get the jobid to then submit vpol with dependency
            print(command_queue)
            both_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

            
            #Prepare Vpol Job
            batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition,jobname+'all', both_jobid)
            command = '%s %i %s %s'%(script, run, deploy_index, 'all')
            command_queue = batch + command

            #Submit all job
            print(command_queue)
            all_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

            print('Run %i jobs submitted --> Both jid:%i\tAll jid:%i'%(run,both_jobid,all_jobid))

        elif False and analysis_part == 2:

            # ***********   FOR USE EXLCUSIVELY ON 1/15/2022
            #Run hpol and vpol jobs in same job, and run the 'all' case as a seperate job. 

            # Jan/15/2022
            # redo for just all numpy.array([5821,5808])
            # redo for hv and all numpy.array([5954,5949,5947,5925,5921,5943,5905,5903,5941,5901,5899,5893,5936])

            if False:
                if numpy.isin(run,numpy.array([5954,5949,5947,5925,5921,5943,5905,5903,5941,5901,5899,5893,5936])):

                    script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part2.sh')

                    
                    #Prepare Both Job
                    batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname+'hv')
                    command = '%s %i %s %s'%(script, run, deploy_index, 'both')
                    command_queue = batch + command

                    #Submit Both job and get the jobid to then submit vpol with dependency
                    print(command_queue)
                    both_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

                    
                    #Prepare All Job
                    batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition,jobname+'all', both_jobid)
                    command = '%s %i %s %s'%(script, run, deploy_index, 'all')
                    command_queue = batch + command

                    #Submit all job
                    print(command_queue)
                    all_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

                    print('Run %i jobs submitted --> Both jid:%i\tAll jid:%i'%(run,both_jobid,all_jobid))
                elif numpy.isin(run, numpy.array([5821,5808])):
                    script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part2.sh')

                    #Prepare ALL Job
                    batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname+'all')
                    command = '%s %i %s %s'%(script, run, deploy_index, 'all')
                    command_queue = batch + command

                    #Submit all job
                    print(command_queue)
                    all_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

                    print('Run %i jobs submitted --> All jid:%i'%(run,all_jobid))


        elif False and analysis_part == 2:
            # Assumes h and v may already be started, and checks if it is running, then will pull the jobid from that for dependancy rather than running it again
            _jobname = jobname+'hv'
            
            script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part2.sh')

            #Run hpol and vpol jobs in same job, and run the 'all' case as a seperate job. 
            
            #Prepare Hpol Job
            batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,_jobname)
            command = '%s %i %s %s'%(script, run, deploy_index, 'both')
            command_queue = batch + command

            if _jobname[0:min(8,len(_jobname))] in list(jobid_dict.keys()):
                print('Skipping Executing the following, as a job already matches the jobname %s'%(_jobname))
                print('\t' + command_queue)
                both_jobid = int(jobid_dict[_jobname[0:min(8,len(_jobname))]])
            else:
                #Submit hpol job and get the jobid to then submit vpol with dependency
                if False:
                    print(command_queue)
                    both_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))
                else:
                    #I don't want to run them in this scenario, they are already run and finished, just not in queue.
                    both_jobid = 0

            
            #Prepare Vpol Job
            _jobname = jobname+'all'
            if both_jobid == 0:
                batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,_jobname)
            else:
                batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition,_jobname, both_jobid)
            command = '%s %i %s %s'%(script, run, deploy_index, 'all')
            command_queue = batch + command

            #Submit all job
            if _jobname[0:min(8,len(_jobname))] in list(jobid_dict.keys()):
                print('Skipping Executing the following, as a job already matches the jobname %s'%(_jobname))
                print('\t' + command_queue)
                all_jobid = int(jobid_dict[_jobname[0:min(8,len(_jobname))]])
            else:
                print(command_queue)
                #all_jobid = 1
                all_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

            print('Run %i jobs submitted --> Both jid:%i\tAll jid:%i'%(run,both_jobid,all_jobid))
            
        elif False and analysis_part == 2:
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
            elif pol == 'all':
                #Prepare all Job
                batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname+'all')
                command = '%s %i %s %s'%(script, run, deploy_index, 'all')
                command_queue = batch + commandW

                #Submit hpol job and get the jobid to then submit all with dependency
                print(command_queue)
                hpol_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n','')) 

        elif False and analysis_part == 3:
            # *** This one will check for existing jobs running with name 'a' and use their jobids as necessary to start.
            # If 'a' (all pol) jobs are not available it will confirm that no jobs are running for that job, then just
            # submit it normally.

            #Runs h then v, then all
            #Run hpol and vpol jobs in same job, and run the 'all' case as a seperate job. 

            check_str = str(jobname+'all')[0:max_expected_column_width] #imposing characterlimit of dict

            if check_str in list(jobid_dict.keys()):
                jobid = int(jobid_dict[check_str])
            elif numpy.any(numpy.array(['bcn%i'%run in a for a in list(jobid_dict.keys())])):
                print('A job showed up for run %i but ALL was not visible.  This is unexpected, and this run is being ignored.'%run)
                continue
            else:
                jobid = 0

            #Prepare Job

            if jobid == 0:
                batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition, jobname)
            else:
                batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition, jobname, jobid)

            script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part3.sh')

            command = '%s %i %s %s'%(script, run, deploy_index, 'both')
            command_queue = batch + command

            #Submit job and get the jobid
            print(command_queue)
            _jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))
            
            if jobid == 0:
                print('Run %i jobs submitted --> jid:%i'%(run,_jobid))
            else:
                print('Run %i jobs submitted --> jid:%i dependant on jid:%i'%(run,_jobid, jobid))
        elif False and analysis_part == 3:
            #Runs h then v, then all
            #Run hpol and vpol jobs in same job, and run the 'all' case as a seperate job. 

            script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part3.sh')            
            #Prepare Job
            batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname)
            command = '%s %i %s %s'%(script, run, deploy_index, 'both')
            command_queue = batch + command

            #Submit job and get the jobid
            print(command_queue)
            jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))
            
            print('Run %i jobs submitted --> jid:%i'%(run,jobid))
            
        elif analysis_part == 4:
            # Execute each script, but assuming the they are dependant on order.

            first   = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'tools', 'sine_subtract_cache.py')
            second  = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part1.sh')
            third   = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part2.sh')

            #Prepare Sine Subtraction

            batch = 'sbatch --partition=%s %s --job-name=%s --time=12:00:00 '%(partition, bad_node_string, jobname + 'ss')
            command = first + ' %i'%(run)
            command_queue = batch + command
            
            #Submit sine subtraction and get the jobid
            print(command_queue)
            first_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))


            #Prepare Non-Map Analysis
            batch = 'sbatch --partition=%s %s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition, bad_node_string, jobname, first_jobid)
            command = '%s %i'%(second, run)
            command_queue = batch + command

            #Submit Non-Map Analysis and get the jobid
            print(command_queue)
            second_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

            #Prepare Maps for H and V pol Job
            batch = 'sbatch --partition=%s %s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition, bad_node_string, jobname+'hv', second_jobid)
            command = '%s %i %s %s'%(third, run, deploy_index, 'both')
            command_queue = batch + command

            #Submit hpol job and get the jobid to then submit vpol with dependency
            print(command_queue)
            both_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))


            #All job must be done second, because "best map" selection is call when all is, so hv must already be done.
            #Prepare All Job
            batch = 'sbatch --partition=%s %s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition, bad_node_string, jobname+'all', both_jobid)
            command = '%s %i %s %s'%(third, run, deploy_index, 'all')
            command_queue = batch + command

            #Submit All job
            print(command_queue)
            all_jobid = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

            print('Run %i jobs submitted --> \nSine Subtraction jid:%i\nNon-Map Analysis jid:%i\tBoth jid:%i\tAll jid:%i'%(run,first_jobid,second_jobid,both_jobid,all_jobid))

        elif analysis_part == 5:
            script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'tools', 'sine_subtract_cache.py')
            batch = 'sbatch --partition=%s --job-name=%s --time=12:00:00 '%(partition,jobname)
            command = script + ' %i'%(run)
            command_queue = batch + command
            print(command_queue)    
            os.system(command_queue) # Submit to queue

        else:
            script = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'analyze_event_rate_frequency.py')
            batch = 'sbatch --partition=%s --job-name=%s --time=36:00:00 '%(partition,jobname)
            command = script + ' %i'%(run)
            command_queue = batch + command
            print(command_queue)    
            os.system(command_queue) # Submit to queue




