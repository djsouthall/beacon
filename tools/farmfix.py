#!/usr/bin/env python3
'''
This will cancel jobs and attempt to resubmit them where they were left off.  This is mainly meant to fix any jobs
that are running on known bad nodes.
'''
import os
import sys
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import subprocess
import time
import numpy
import yaml
from pprint import pprint


def parseJobIDs(partition,user):
    '''
    Will return a dictionary of job ids for the set of jobs currently queued for user on partition.
    '''
    structured_dtype = numpy.dtype([('jobname', numpy.unicode_, 16), ('run', int), ('jobtype', numpy.unicode_, 16), ('jobid', int), ('node_reason', numpy.unicode_, 16)])
    out_array = numpy.array([], dtype=structured_dtype)

    text = subprocess.check_output(['squeue','--format="%.18i split %.30j split %R"','--user=dsouthall']).decode('utf-8')
    for line in text.replace(' ','').replace('"','').split('\n'):
        if 'JOBID' in line or len(line) == 0:
            continue
        try:
            # import pdb; pdb.set_trace()
            jobid = int(line.split('split')[0])
            jobname = str(line.split('split')[1])

            run = int(''.join(filter(str.isdigit, jobname)))

            jobtype = jobname.replace('bcn','').replace(str(run),'')
            
            node_reason = str(str(line.split('split')[2]))

            a = numpy.array([(jobname, run, jobtype, jobid, node_reason)], dtype=structured_dtype)
            out_array = numpy.append(out_array, a)
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()

    return out_array


if __name__ == "__main__":

    ###------------###
    ### Parameters ###
    ###------------###
    debug = False #Disables actually sending commands to bash

    username = 'dsouthall'
    partition = 'broadwl'
    deploy_index = '/home/dsouthall/Projects/Beacon/beacon/config/september_2021_minimized_calibration.json'

    bad_node_numbers = [15,227]
    bad_node_string = "--exclude=midway2-%s"%str(['{:04d}'.format(node) for node in bad_node_numbers]).replace("'","").replace(' ','')

    bad_node_list = ["midway2-{:04d}".format(node) for node in bad_node_numbers]


    ###--------###
    ### Script ###
    ###--------###

    out_array = parseJobIDs(partition,username)

    expected_jobname_order = ['ss','','hv','all'] #'bcn%i%s'%(run,expected_jobname_order[i])

    flagged_runs = numpy.unique(out_array['run'][numpy.isin(out_array['node_reason'], bad_node_list)])

    print('Number of flagged runs = ', len(flagged_runs))
    print(flagged_runs)
    print('Continue?')
    import pdb; pdb.set_trace()
    print('Are you sure?')
    import pdb; pdb.set_trace()

    # Execute each script, but assuming the they are dependant on order.

    first   = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'tools', 'sine_subtract_cache.py')
    second  = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part1.sh')
    third   = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'all_analysis_part2.sh')

    for run in flagged_runs:
        print('\nRun %i'%run)
        
        jobs_to_run = out_array[out_array['run'] == run]['jobtype']


        jobname = 'bcn%i'%run
        past_job_id = None
        for index in range(len(expected_jobname_order)):
            current_entry = out_array[numpy.logical_and(out_array['run'] == run, out_array['jobtype'] == expected_jobname_order[index])]
            if len(current_entry) == 1:
                current_entry = current_entry[0]

            if index == 0 and expected_jobname_order[index] in jobs_to_run:
                cancel_command = 'scancel %i'%(current_entry['jobid'])
                print('Cancelling current job:')
                print(cancel_command)
                if debug == False:
                    # print('Is this okay?')
                    # import pdb; pdb.set_trace()
                    subprocess.Popen(cancel_command.split(' '))

                #Prepare Sine Subtraction

                batch = 'sbatch --partition=%s %s --job-name=%s --time=12:00:00 '%(partition, bad_node_string, jobname + 'ss')
                command = first + ' %i'%(run)
                command_queue = batch + command

                #Submit sine subtraction and get the jobid
                print(command_queue)
                if debug == False:
                    past_job_id = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

            elif index == 1 and expected_jobname_order[index] in jobs_to_run:
                cancel_command = 'scancel %i'%(current_entry['jobid'])
                print('Cancelling current job:')
                print(cancel_command)
                if debug == False:
                    # print('Is this okay?')
                    # import pdb; pdb.set_trace()
                    subprocess.Popen(cancel_command.split(' '))

                if past_job_id is not None:
                    #Prepare Non-Map Analysis
                    batch = 'sbatch --partition=%s %s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition, bad_node_string, jobname, past_job_id)
                    command = '%s %i'%(second, run)
                    command_queue = batch + command

                    #Submit Non-Map Analysis and get the jobid
                    print(command_queue)
                    if debug == False:
                        past_job_id = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))
                else:
                    #Prepare Non-Map Analysis
                    batch = 'sbatch --partition=%s %s --job-name=%s --time=36:00:00 '%(partition, bad_node_string, jobname)
                    command = '%s %i'%(second, run)
                    command_queue = batch + command

                    #Submit Non-Map Analysis and get the jobid
                    print(command_queue)
                    if debug == False:
                        past_job_id = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))
            elif index == 2 and expected_jobname_order[index] in jobs_to_run:
                cancel_command = 'scancel %i'%(current_entry['jobid'])
                print('Cancelling current job:')
                print(cancel_command)
                if debug == False:
                    # print('Is this okay?')
                    # import pdb; pdb.set_trace()
                    subprocess.Popen(cancel_command.split(' '))

                if past_job_id is not None:
                    #Prepare Maps for H and V pol Job
                    batch = 'sbatch --partition=%s %s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition, bad_node_string, jobname+'hv', past_job_id)
                    command = '%s %i %s %s'%(third, run, deploy_index, 'both')
                    command_queue = batch + command

                    #Submit hpol job and get the jobid to then submit vpol with dependency
                    print(command_queue)
                    if debug == False:
                        past_job_id = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))
                else:
                    #Prepare Maps for H and V pol Job
                    batch = 'sbatch --partition=%s %s --job-name=%s --time=36:00:00 '%(partition, bad_node_string, jobname+'hv')
                    command = '%s %i %s %s'%(third, run, deploy_index, 'both')
                    command_queue = batch + command

                    #Submit hpol job and get the jobid to then submit vpol with dependency
                    print(command_queue)
                    if debug == False:
                        past_job_id = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))
            elif index == 3 and expected_jobname_order[index] in jobs_to_run:
                cancel_command = 'scancel %i'%(current_entry['jobid'])
                print('Cancelling current job:')
                print(cancel_command)
                if debug == False:
                    # print('Is this okay?')
                    # import pdb; pdb.set_trace()
                    subprocess.Popen(cancel_command.split(' '))

                if past_job_id is not None:
                    #All job must be done second, because "best map" selection is call when all is, so hv must already be done.
                    #Prepare All Job
                    batch = 'sbatch --partition=%s %s --job-name=%s --time=36:00:00 --dependency=afterok:%i '%(partition, bad_node_string, jobname+'all', past_job_id)
                    command = '%s %i %s %s'%(third, run, deploy_index, 'all')
                    command_queue = batch + command

                    #Submit All job
                    print(command_queue)
                    if debug == False:
                        past_job_id = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))
                else:
                    #All job must be done second, because "best map" selection is call when all is, so hv must already be done.
                    #Prepare All Job
                    batch = 'sbatch --partition=%s %s --job-name=%s --time=36:00:00 '%(partition, bad_node_string, jobname+'all')
                    command = '%s %i %s %s'%(third, run, deploy_index, 'all')
                    command_queue = batch + command

                    #Submit All job
                    print(command_queue)
                    if debug == False:
                        past_job_id = int(subprocess.check_output(command_queue.split(' ')).decode("utf-8").replace('Submitted batch job ','').replace('\n',''))

            if past_job_id is not None:
                print('Submitted jobid %i'%past_job_id)



