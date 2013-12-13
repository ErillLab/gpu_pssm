# -*- coding: utf-8 -*-
"""
Created on Thu Nov 07 02:48:39 2013

@author: Talmo

This file can be used to test the performance of the CUDA PSSM sliding window
implementation (gpu_pssm).

"""
import numpy as np
from numbapro import cuda
from gpu_pssm import *
from matplotlib.pyplot import *
from time import time

# Genome size vs bases/sec
def test1():
    """
    This function generates a plot of CUDA performance vs size of the
    sequence being scored.
    """
    start = time()
    genome_sizes = range(int(5e6), int(80e6 + 1), int(5e6)) # 5 to 80mi bases at steps of 5mi
    trials = 3 # trials per size
    w = 16 # window size
    pssm = np.random.rand(4 * w) # generate PSSM
    data = []
    for n in genome_sizes * trials:
        # Generate random genome
        seq = np.random.randint(0, 3, int(n))
        
        # Score
        __, run_info = score_sequence(seq, pssm, keep_strands = False, benchmark = True)
        data_point = (n, run_info["genome_size"] / run_info["runtime"])
        print "%d -> %g bp/s" % data_point
        data.append(data_point)
    print "\n\nBenchmarked %g bp in %.2f seconds." % (np.int64(sum(genome_sizes)) * trials, time() - start)
    
    x = [pt[0] for pt in data]
    y = [pt[1] for pt in data]
    print x
    print y
    #fit_fn = poly1d(polyfit(x, y, 1))
    #plot(x, y, 'x', x, fit_fn(x), '--r')
    #xlabel('Genome Size'), ylabel('Bases per second')
    
# bpg vs tpb
def test2():
    """
    This function is aimed at testing the performance of initializing the CUDA
    kernel with differing values of blocks per grid and threads per block.
    
    Careful: It is very unstable and will most likely crash your display driver
    unless you set safe ranges. If you're on Windows 7 or newer, it will just
    be restarted, however, but you will need to restart the Python session to
    restart the device hook from NumbaPro if you're running this interactively.
    
    TODO: Generate a better visualization for this -- 3d plots are bad...
    """
    n = 50e6 # See test1() for why this is an unbiased genome size
    trials = 20
    w = 16 # window size
    pssm = np.random.rand(4 * w) # generate PSSM
    bpg_range = (16, cuda.get_current_device().MAX_GRID_DIM_X*0.5)
    tpb_range = (32, cuda.get_current_device().MAX_BLOCK_DIM_X*0.5)
    
    data = []
    for t in range(trials):
        for bpg_exp in range(int(math.ceil(math.log(bpg_range[0], 2))), int(math.ceil(math.log(bpg_range[1], 2)) + 1)):
            bpg = 2 ** bpg_exp
            for tpb_exp in range(int(math.ceil(math.log(tpb_range[0], 2))), int(math.ceil(math.log(tpb_range[1], 2)) + 1)):
                tpb = 2 ** tpb_exp
                print t, bpg, tpb
                seq = np.random.randint(0, 3, int(n))
                __, __, run_info = score_sequence(seq, pssm, False, True, bpg, tpb)
                data.append((bpg, tpb, run_info["genome_size"] / run_info["runtime"]))
    x = [pt[0] for pt in data]
    y = [pt[1] for pt in data]
    z = [pt[2] for pt in data]
    
    
    xlabel('Blocks per Grid'), ylabel('Threads per Block')

def test3():
    """
    This function genertes a plot comparing CPU versus GPU performance.
    """
    start = time()
    genome_sizes = range(int(100e3), int(1e6 + 1), int(100e3))
    trials = 5 # trials per size
    w = 16 # window size
    pssm = np.random.rand(4 * w) # generate PSSM
    data = []
    for n in genome_sizes * trials:
        print n
        # Generate random genome
        seq = np.random.randint(0, 3, int(n))
        
        # Score
        __, __, gpu_run_info = score_sequence(seq, pssm, benchmark = True)
        __, __, cpu_run_info = score_sequence_with_cpu(seq, pssm, benchmark = True)
        
        # Save data
        data_point = (n, n / gpu_run_info["runtime"], n / cpu_run_info["runtime"])
        print data_point
        data.append(data_point)
    print "\n\nBenchmarked %g bp in %.2f seconds." % (np.int64(sum(genome_sizes)) * trials, time() - start)
    
    # Plot data
    x = [pt[0] for pt in data]
    y1 = [pt[1] for pt in data]
    y2 = [pt[2] for pt in data]
    fit_fn1 = poly1d(polyfit(x, y1, 1))
    fit_fn2 = poly1d(polyfit(x, y2, 1))
    plot(x, y1, 'rx', x, fit_fn1(x), 'r--', x, y2, 'x', x, fit_fn2(x), 'b--')
    plt.yscale('log')
    xlabel('Genome Size'), ylabel('Bases per second')
    
test1()