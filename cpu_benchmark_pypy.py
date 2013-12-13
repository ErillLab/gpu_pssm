# -*- coding: utf-8 -*-
"""
Created on Fri Dec 06 00:45:45 2013

@author: Talmo

This file can be used to benchmark the performance of the CPU in a naive
sliding window scoring routine.
"""
#import numpypy
#import numpy as np
import random
from time import time, clock
import sys


if sys.platform == 'win32':
    default_timer = clock # On Windows, the best timer is time.clock
else:
    default_timer = time # On most other platforms the best timer is time.time


# Genome size vs bases/sec
def test1():
    start = default_timer()
    genome_sizes = range(int(125e3), int(2000e3 + 1), int(125e3))
    trials = 15 # trials per size
    w = 16 # window size
    bases = "ACGT"
    pssm = [{base: random.uniform(-7.5, 2.0) for base in bases} for _ in range(w)] # generate PSSM
    data = []
    for n in genome_sizes * trials:
        # Generate random genome
        seq = "".join([random.choice(bases) for _ in range(n)])
        
        # Score
        _, time_elapsed = score_sequence(seq, pssm)
        data_point = (n, n / time_elapsed)
        print "%d -> %g bp/s" % data_point
        data.append(data_point)
    print "\n\nBenchmarked %g bp in %.2f seconds." % (int(sum(genome_sizes)) * trials, default_timer() - start)
    
    x = [pt[0] for pt in data]
    y = [pt[1] for pt in data]

    print "x =", x
    print "y =", y
    fit_fn = poly1d(polyfit(x, y, 1))
    plot(x, y, 'x', x, fit_fn(x), '--r')
    xlabel('Genome Size'), ylabel('Bases per second')


def score_sequence(sequence, pssm):
    start = default_timer()
    
    # Get reverse PSSM
    pssm_wc = wc_matrix(pssm)
    
    # Scoring function
    score = lambda seq, matrix, matrix_wc: max(sum([matrix[i][base] for i, base in enumerate(seq)]), sum([matrix_wc[i][base] for i, base in enumerate(seq)]))
    
    # Score using sliding window
    scores = [score(seq, pssm, pssm_wc) for seq in sliding_window(sequence, len(pssm), seq_only=True)]
    
    # Time!
    time_elapsed = default_timer() - start
    return scores, time_elapsed



def sliding_window(sequence, width=2, step=1, seq_only=False):
    """Generator for a sliding window walker; yields a new substring of sequence
    as well as the position of the left end of the window."""
    for position in range(0, len(sequence) - width, step):
        yield sequence[position: position + width] if seq_only else (position, sequence[position: position + width])


def wc_matrix(matrix):
    """Returns the reverse complement of a position-weight matrix."""
    return [{"A": position["T"], "T": position["A"], "C": position["G"], "G": position["C"]} for position in matrix[::-1]]




test1()