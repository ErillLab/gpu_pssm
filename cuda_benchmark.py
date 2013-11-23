# -*- coding: utf-8 -*-
"""
Created on Thu Nov 07 02:48:39 2013

@author: Talmo

This file can be used to test the performance of the CUDA PSSM sliding window
implementation.

"""
import numpy as np
from numbapro import cuda
from gpu_pssm import score_sequence
from matplotlib.pyplot import *
from time import time

# Genome size vs bases/sec
def test1():
    start = time()
    genome_sizes = range(int(5e6), int(80e6 + 1), int(5e6)) # 5 to 80mi bases at steps of 5mi
    trials = 3 # trials per size
    w = 16 # window size
    pssm = np.random.rand(4 * w) # generate PSSM
    data = []
    for n in genome_sizes * trials:
        print n
        # Generate random genome
        seq = np.random.randint(0, 3, int(n))
        
        # Score
        __, __, run_info = score_sequence(seq, pssm, benchmark = True)
        data.append((n, run_info["genome_size"] / run_info["runtime"]))
    print "\n\nBenchmarked %g bp in %.2f seconds." % (int64(sum(genome_sizes)) * trials, time() - start)
    
    x = [pt[0] for pt in data]
    y = [pt[1] for pt in data]
    fit_fn = poly1d(polyfit(x, y, 1))
    plot(x, y, 'x', x, fit_fn(x), '--r')
    xlabel('Genome Size'), ylabel('Bases per second')
    
# bpg vs tpb
#def test2():
#n = 50e6 # See test1() for why this is an unbiased genome size
#trials = 20
#w = 16 # window size
#pssm = np.random.rand(4 * w) # generate PSSM
#bpg_range = (16, cuda.get_current_device().MAX_GRID_DIM_X*0.5)
#tpb_range = (32, cuda.get_current_device().MAX_BLOCK_DIM_X*0.5)
#
#data = []
#for t in range(trials):
#    for bpg_exp in range(int(math.ceil(math.log(bpg_range[0], 2))), int(math.ceil(math.log(bpg_range[1], 2)) + 1)):
#        bpg = 2 ** bpg_exp
#        for tpb_exp in range(int(math.ceil(math.log(tpb_range[0], 2))), int(math.ceil(math.log(tpb_range[1], 2)) + 1)):
#            tpb = 2 ** tpb_exp
#            print t, bpg, tpb
#            seq = np.random.randint(0, 3, int(n))
#            __, __, run_info = score_sequence(seq, pssm, False, True, bpg, tpb)
#            data.append((bpg, tpb, run_info["genome_size"] / run_info["runtime"]))
#x = [pt[0] for pt in data]
#y = [pt[1] for pt in data]
#z = [pt[2] for pt in data]

#xlabel('Blocks per Grid'), ylabel('Threads per Block')