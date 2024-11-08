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
# WaitGPU utility by Eric Wong, modified by Huan Zhang

import os
import gpustat
import time
import re
import subprocess
from collections import defaultdict
import random

def rocm_get_info():
    total_mem = {}
    used_mem = {}
    utilization = {}
    mem_output = subprocess.check_output('rocm-smi --showmeminfo vram'.split())
    gpu_output = subprocess.check_output('rocm-smi --showuse'.split())
    for l in mem_output.decode('utf-8').split('\n'):
        m = re.match(r'GPU\[(\d)\]\s+: vram Total Memory \(B\): (\d+)', l)
        if m:
            gpu_id = int(m.group(1))
            mem = int(m.group(2))
            total_mem[gpu_id] = int(mem / 1024 / 1024)
        m = re.match(r'GPU\[(\d)\]\s+: vram Total Used Memory \(B\): (\d+)', l)
        if m:
            gpu_id = int(m.group(1))
            mem = int(m.group(2))
            used_mem[gpu_id] = int(mem / 1024 / 1024)
    for l in gpu_output.decode('utf-8').split('\n'):
        m = re.match(r'GPU\[(\d)\]\s+: GPU use \(%\): (\d+)', l)
        if m:
            gpu_id = int(m.group(1))
            util = int(m.group(2))
            utilization[gpu_id] = util
    results = []
    class Entry(object):
        def __init__(self, results):
            self.entry = results
    min_ind = min(total_mem.keys())
    for k in total_mem.keys():
        results.append(Entry({'index': k - min_ind,
                              'memory.total': total_mem[k],
                              'memory.used': used_mem[k],
                              'utilization.gpu': utilization[k],
                              'processes': []}))
    return results

def proc_sat(gpu, nproc):
    """ Return true if the number of processes on gpu is at most nproc"""
    return len(gpu.entry['processes']) <= nproc

def util_sat(gpu, util):
    """ Return true if the gpu utilization is at most util """
    return float(gpu.entry['utilization.gpu']) <= util

def mem_ratio_sat(gpu, mem_ratio):
    """ Return true if the memory utilization is at most mem_ratio """
    r = float(gpu.entry['memory.used'])/float(gpu.entry['memory.total'])
    return r <= mem_ratio

def avail_mem_sat (gpu, mem):
    """ Return true if there is at least mem available memory """
    avail_mem = float(gpu.entry['memory.total'])-float(gpu.entry['memory.used'])
    return mem <= avail_mem

def gpu_id_sat(gpu, gpu_ids):
    gid = int(gpu.entry['index'])
    return gid in gpu_ids

def wait(utilization=None, memory_ratio=None, available_memory=None,
         interval=10, gpu_ids=None, nproc=None, ngpu=1, max_count=0, blacklist=None, usedlist=None):
    print("waitGPU: Waiting for the following conditions, checking every {} seconds. "
          .format(interval))
    conditions = []
    if utilization is not None:
        conditions.append(lambda gpu: util_sat(gpu, utilization))
        print("+ utilization <= {}".format(utilization))
    if memory_ratio is not None:
        conditions.append(lambda gpu: mem_ratio_sat(gpu, memory_ratio))
        print("+ memory_ratio <= {}".format(memory_ratio))
    if available_memory is not None:
        conditions.append(lambda gpu: avail_mem_sat(gpu, available_memory))
        print("+ available_memory >= {}".format(available_memory))
    if gpu_ids is not None:
        conditions.append(lambda gpu: gpu_id_sat(gpu, gpu_ids))
        print("+ GPU id is {}".format(gpu_ids))
    if nproc is not None:
        conditions.append(lambda gpu: proc_sat(gpu, nproc))
        print("+ n_processes <= {}".format(nproc))

    while True:
        free_gpu_ids = []
        if os.path.exists('/dev/nvidia0'):
            # NVIDIA GPUs
            use_nvidia = True
            stats = gpustat.GPUStatCollection.new_query().gpus
        else:
            # AMD ROCm
            stats = rocm_get_info()
            use_nvidia = False
        # shuffle to avoid selecting the same GPU
        random.shuffle(stats)
        # prioritize unused GPUs. (key is the smaller the better)
        if usedlist is None:
            usedlist = defaultdict(int)
        stats = sorted(stats, key=lambda g: (usedlist[g.entry['index']], -g.entry['memory.total'], g.entry['utilization.gpu'], g.entry['memory.used']))
        all_blacklisted = True
        for gpu in stats:
            index = int(gpu.entry['index'])
            if blacklist is not None and index in blacklist:
                print('GPU {} blacklisted'.format(index))
                continue
            all_blacklisted = False
            if all(c(gpu) for c in conditions):
                free_gpu_ids.append(index)
            if len(free_gpu_ids) == ngpu:
                break
        if all_blacklisted:
            print("waitGPU: all GPUs blacklisted. Return failure")
            return False
        max_count -= 1
        if max_count == 0:
            print("waitGPU: max wait count has reached. Return failure.")
            return False
        if len(free_gpu_ids) < ngpu:
            time.sleep(interval)
        else:
            break

    print("waitGPU: Setting GPU to: {}".format(free_gpu_ids))
    if use_nvidia:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, free_gpu_ids))
    else:
        os.environ['HIP_VISIBLE_DEVICES'] = ",".join(map(str, free_gpu_ids))
    return True

if __name__ == "__main__":
    print(rocm_get_info())
    wait(utilization=10, memory_ratio=0.2, nproc=1)

