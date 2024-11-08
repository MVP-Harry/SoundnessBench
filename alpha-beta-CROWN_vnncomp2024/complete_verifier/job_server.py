#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
### preprocessor-hint: private-file
# A parallel job server for CPU and/or GPU by Huan Zhang.

import os
import sys
import multiprocessing
from multiprocessing import Pool, Array, cpu_count
from collections import defaultdict
from termcolor import colored, cprint
import waitGPU
import time
import subprocess


# Initialization function that will be run by each subprocess in thread pool.
def worker_init(counter, pool_njobs, max_jobs, delay, gpu, id_queue,
                utilization_, memory_ratio_, args=None):
    global gpu_counter
    global njobs
    global max_jobs_per_gpu
    global use_delay
    global use_gpu
    global utilization
    global memory_ratio
    gpu_counter = counter
    njobs = pool_njobs
    max_jobs_per_gpu = max_jobs
    use_delay = delay
    use_gpu = gpu
    utilization = utilization_
    memory_ratio = memory_ratio_
    # Obtain an unique ID for this worker thread
    index = id_queue.get()
    if not use_gpu:
        ncpus = cpu_count() // 2
        cpu_list = []
        task_ncpus = 1 if args is None else args.ncpus
        start_cpuid = 0 if args is None else args.start_cpuid
        set_affinity = False if args is None else args.set_affinity
        if set_affinity:
            for i in range(task_ncpus):
                # main core
                cpu_list.append((start_cpuid + (task_ncpus * index + i)) % ncpus)
                # SMT core
                cpu_list.append((start_cpuid + (task_ncpus * index + i)) % ncpus + ncpus)
            print(f'Worker {index} using CPU {cpu_list}')
            os.sched_setaffinity(0, cpu_list)
        time.sleep(3)


# Main function for each subprocess worker.
def worker(cmd):
    global gpu_counter
    global njobs
    global max_jobs_per_gpu
    global use_delay
    global use_gpu
    global utilization
    global memory_ratio
    jid = cmd[0]
    cmd = cmd[1]
    if use_delay:
        delay = (jid % njobs) * 2 + 2
        cprint('job {} sleeping for {} seconds'.format(jid, delay), 'blue')
        time.sleep(delay)
        if use_gpu:
            cprint('job {} waiting for available GPU'.format(jid), 'blue')
    if use_gpu:
        while True:
            blacklist = []  # If some GPU is broken, we can add them to the blacklist.
            usedlist = defaultdict(int)
            with gpu_counter.get_lock():
                for i, c in enumerate(gpu_counter):
                    if c >= max_jobs_per_gpu:  # We only run at most max_jobs_per_gpu jobs per GPU. For most tasks it is 1.
                        blacklist.append(i)
                    if c > 0:
                        usedlist[i] += c
            cprint('Current GPUs in use: {}'.format(usedlist), 'blue')
            # Wait for idle GPUs. Set your criterion here. Wait for GPU utilization < 50%, memory usage < 1/3, occupied by less than 5 processes
            gpu_ok = waitGPU.wait(utilization=utilization,
                                  memory_ratio=memory_ratio,
                                  interval=10, nproc=1, ngpu=1, max_count=6,
                                  blacklist=blacklist, usedlist=usedlist)
            if not gpu_ok:
                # Reached maximum timeout, or all GPUs blacklisted
                time.sleep(2)
                continue
            gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES']) if 'CUDA_VISIBLE_DEVICES' in os.environ else int(os.environ['HIP_VISIBLE_DEVICES'])
            with gpu_counter.get_lock():
                if gpu_counter[gpu_id] >= max_jobs_per_gpu:
                    cprint("Currently {} jobs are running on GPU {}, exceeding limit {}".format(gpu_counter[gpu_id], gpu_id, max_jobs_per_gpu), 'blue')
                    time.sleep(2)
                    continue
                else:
                    gpu_counter[gpu_id] += 1
                    break
        cprint("Running job {} on GPU {}".format(cmd, gpu_id), 'blue')
    else:
        cprint("Running job {} without GPU".format(cmd), 'blue')
        # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        # os.environ['HIP_VISIBLE_DEVICES'] = "-1"
    sys.stdout.flush()
    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        cprint("Job {} created".format(jid), 'blue')
        logfile = open('job_{}.log'.format(jid), 'wb')
        output_buffer = []
        for line in process.stdout:
            logfile.write(line)
            logfile.flush()
            line = line.decode('utf-8')
            output_buffer.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            # Program finished. Kill it after timeout.
            if '[END-END-END]' in line:
                cprint("Job {} END detected".format(jid), 'blue')
                break
    except (KeyboardInterrupt, SystemExit, SyntaxError):
        raise
    except Exception as e:
        cprint("Job {} error".format(jid), 'red')
        cprint("Command {} on GPU {}".format(cmd, gpu_id), 'red')
        print(e)
    # Try terminate the process in case they get stuck after finishing.
    try:
        try:
            process.communicate(timeout=5)
        except subprocess.SubprocessError:
            cprint("Killing job".format(jid), 'red')
            process.terminate()
            time.sleep(3)
            process.kill()
            time.sleep(3)
            process.kill()
            cprint("Killed job".format(jid), 'red')
    except (KeyboardInterrupt, SystemExit, SyntaxError):
        raise
    except Exception as e:
        cprint("Job Error!".format(jid), 'red')
        print(e)
        pass
    if use_gpu:
        with gpu_counter.get_lock():
            gpu_counter[gpu_id] -=1
    cprint("Job {} Done".format(jid), 'blue')
    return output_buffer


class JobServer():
    def __init__(self, njobs, max_jobs_per_gpu=1, use_delay=True, use_gpu=True,
                 utilization=5, memory_ratio=0.05,
                 additional_args=None):
        # keep record of how many processes are running on a certain GPU
        gpu_counter = Array('i', [0]*32, lock=True)
        if use_gpu:
            cprint("starting job server with {} jobs, max {} jobs per GPU".format(njobs, max_jobs_per_gpu), 'blue')
        else:
            cprint("starting job server with {} jobs".format(njobs), 'blue')
        # Generate a ID for each worker.
        manager = multiprocessing.Manager()
        id_queue = manager.Queue()
        for i in range(njobs):
            id_queue.put(i)
        self.pool = Pool(njobs, initializer=worker_init,
                         initargs=(gpu_counter, njobs, max_jobs_per_gpu,
                                   use_delay, use_gpu, id_queue,
                                   utilization, memory_ratio, additional_args))
        self.njobs = njobs
        self.use_gpu = use_gpu

    def gpu_reset(self):
        cprint("Resetting GPUs", 'blue')
        for i in range(8):
            subprocess.check_output('/opt/rocm/bin/gpureset {}'.format(i+1).split())

    def run_commands(self, commands):
        sync = False  # Run in synchronized mode - wait for all jobs done before the next batch.
        results = []
        if sync:
            cprint("Applying job server workaround to run {} jobs each batch".format(self.njobs), 'blue')
            self.gpu_reset()
            idx = 0
            while idx < len(commands):
                cprint("Running batch {} to {}".format(idx, idx+self.njobs), 'blue')
                results.extend(self.pool.map(worker, enumerate(commands[idx:idx + self.njobs]), chunksize=1))
                idx += self.njobs
                self.gpu_reset()
            return results
        else:
            return self.pool.map(worker, enumerate(commands), chunksize=1)
