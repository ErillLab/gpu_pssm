# -*- coding: utf-8 -*-
"""
Created on Thu Nov 07 17:52:15 2013

@author: Talmo

This file provides functions that can be used to score a sequence of
nucleotide bases using a GPU-based PSSM approach.
"""

import math
import numpy as np
from numbapro import cuda
from time import time, clock


def score_sequence(seq, pssm, verbose = False, benchmark = False, blocks_per_grid = -1, threads_per_block = -1):
    """
    This function will score a sequence of nucleotides based on a PSSM by using
    a sliding window parallelized on a GPU.
    
    Args:
        seq: This must be an integer representation of the nucleotide sequence,
            where the alphabet is (A = 0, C = 1, G = 2, T = 3). It must be a 
            vector (1D array) of integers that can be cast to int32 (See: 
            numpy.int32).
        pssm: This must a vectorized PSSM where every four elements correspond 
            to one position. Make sure this can be cast to an array of float32.
        verbose: Set this to True to print performance information.
        benchmark: If set to True, the function will return information about
            the run in a dictionary at the third output variable.
        blocks_per_grid: This is the blocks per grid that will be assigned to 
            the CUDA kernel. See this SO question for info on choosing this
            value: http://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
            It defaults to the length of the sequence or the maximum number of
            blocks per grid supported by the GPU, whichever is lower.
        threads_per_block: Threads per block. See above. It defaults to 55% of
            the maximum number of threads per block supported by the GPU, a
            value determined experimentally. Higher values will likely result
            in failure to allocate resources to the kernel (since there will
            not be enough register space for each thread).
        
    Returns:
        scores: 1D float32 array of length (n - w + 1), where n is the length
            of the sequence and w is the window size. The value at index i of
            this array corresponds to the score of the n-mer at position i in 
            the sequence.
        strands: 1D int32 array of length (n - w + 1). The value at position i
            is either 0 or 1 corresponding to the strand of the score at that
            position where 0 means the forward strand and 1 means reverse.
        run_info: This is a dictionary that is returned if the benchmark
            parameter is set to True. It contains the following:
            >>> run_info.keys()
            ['memory_used', 'genome_size', 'runtime', 'threads_per_block', 'blocks_per_grid']
            
    Example:
        >>> pssm = np.tile([1, 2, 3, 4], 16)
        >>> seq = np.random.randint(0, 3, 30e6)
        >>> scores, strands, run_info = score_sequence(seq, pssm, verbose=True)
        tpb = 563, bpg = 53286, global threads = 30000018
        genome size: 3e+07 bp time: 241.00 ms 1.24481e+08 bp/s
        global memory: 361234432 bytes used (33.64% of total)
        >>> scores
        array([ 40.,  36.,  45., ...,  40.,  38.,  35.], dtype=float32)
        >>> strands
        array([1, 1, 0, ..., 1, 0, 1])
    """
    w = len(pssm) / 4 # width of PSSM
    n = len(seq) # length of the sequence

    # Calculate the appropriate threads per block and blocks per grid    
    if threads_per_block < 0 or blocks_per_grid < 0:
        # We don't use the max number of threads to avoid running out of
        # register space by saturating the streaming multiprocessors
        threads_per_block = int(cuda.get_current_device().MAX_BLOCK_DIM_X * 0.55)
        
        # We saturate our grid and let the dynamic scheduler assign the blocks
        # to the discrete CUDA cores/streaming multiprocessors
        blocks_per_grid = int(math.ceil(float(n) / threads_per_block))
        if blocks_per_grid > cuda.get_current_device().MAX_GRID_DIM_X:
            blocks_per_grid = cuda.get_current_device().MAX_GRID_DIM_X
    
    if verbose:
        print 'tpb = %d, bpg = %d, global threads = %d' % (threads_per_block, blocks_per_grid, threads_per_block * blocks_per_grid)
    
    
    # Calculate the reverse-complement of the PSSM
    pssm_r = np.array([pssm[i / 4 + (3 - (i % 4))] for i in range(w*4)][::-1])
    
    # Collect benchmarking info
    s = clock()
    start_mem = cuda.get_current_device().get_memory_info()[0]
    
    # Start a stream
    stream = cuda.stream()
    
    # Copy data to device
    d_pssm = cuda.to_device(pssm.astype(np.float32), stream)
    d_pssm_r = cuda.to_device(pssm_r.astype(np.float32), stream)
    d_seq = cuda.to_device(seq, stream)
    
    # Allocate memory on device to store results
    d_scores = cuda.device_array(n - w + 1, dtype=np.float32, stream=stream)
    d_strands = cuda.device_array(n - w + 1, dtype=np.int32, stream=stream)
    
    # Run the kernel
    cuda_score[blocks_per_grid, threads_per_block](d_pssm, d_pssm_r, d_seq, d_scores, d_strands)
    
    # Copy results back to host
    scores = d_scores.copy_to_host(stream=stream)
    strands = d_strands.copy_to_host(stream=stream)
    stream.synchronize()
    
    # Collect benchmarking info
    end_mem = cuda.get_current_device().get_memory_info()[0]
    t = clock() - s
    
    # Output info on the run if verbose parameter is true
    if verbose:
        print "genome size: %g bp" % n,
        print "time: %.2f ms" % (t * 1000),
        print "%g bp/s" % (n / t)
        print "global memory: %d bytes used (%.2f%% of total)" % \
            (start_mem - end_mem, float(start_mem - end_mem) * 100 / cuda.get_current_device().get_memory_info()[1])
    
    # Return the run information for benchmarking
    if benchmark:
        run_info = {"genome_size": n, "runtime": t, "memory_used": start_mem - end_mem, \
            "blocks_per_grid": blocks_per_grid, "threads_per_block": threads_per_block}
        return (scores, strands, run_info)
        
    return (scores, strands)


def score_long_sequence(sequence, pssm, chunk_size=7e7, keep_strands=True):
    """
    This function is a wrapper for score_sequence that splits a very long
    sequence into chunks that can be copied to the GPU safely without running
    out of memory.
    
    Args:
        sequence: A sequence of bases to be scored.
        pssm: PSSM to be used for scoring.
        chunk_size: The length of each chunk. Change this to the highest value
            your GPU can stably handle. Recommended value for a GPU with
            1024 MB of memory is 6e7.
        keep_strands: Whether memory should be allocated for storing which
            strand the scores come from. Set this to False if you just want the
            scores and the strands array will not be returned.
            
    Returns:
        scores: An array of scores of the length of the sequence minus the
            window size (width of the PSSM).
        strands: An array indicating which strand the score corresponds to.
        
    See score_sequence for (a lot) more information on these parameters and
    return values.
    """
    
    # Pre-allocate memory for scores and strands    
    scores = np.empty(sequence.size - pssm.size / 4 + 1, np.float32)
    if keep_strands:
        strands = np.empty(sequence.size - pssm.size / 4 + 1, np.int32)    
    
    for chunk_start in range(0, sequence.size, int(chunk_size)):
        # Score chunk
        chunk_scores, chunk_strands = score_sequence(sequence[chunk_start:chunk_start + chunk_size], pssm)
        
        # Save output to array segment if keep_strands:
        scores[chunk_start:chunk_start + chunk_scores.size] = chunk_scores
        if keep_strands:
            strands[chunk_start:chunk_start + chunk_strands.size] = chunk_strands
    
    if keep_strands:
        return scores, strands
    else:
        return scores


def score_sequence_with_cpu(sequence, pssm, benchmark=True):
    """
    This is a wrapper for the cpu_score() function and is analogous to
    score_sequence(). It uses a CPU-based implementation of the sliding window
    scorer. The main purpose is to use it for comparison with the CUDA-based
    implementation.
    
    Args:
        sequence: A sequence of bases to be scored.
        pssm: PSSM to be used for scoring.
        benchmark: Determines whether the runtime for the scoring function will
            be returned.
            
    Returns:
        scores: An array of scores of the length of the sequence minus the
            window size (width of the PSSM).
        strands: An array indicating which strand the score corresponds to.
        runtime: The time it took to score the sequence. It is essentially the
            same as timing the function call, except it does not include the
            time needed to allocate memory. Will only be returned if the
            benchmark argument is set to True.
    """
        
    # Calculate the reverse-complement of the PSSM
    pssm_r = np.array([pssm[i / 4 + (3 - (i % 4))] for i in range(pssm.size)][::-1])
    
    # Pre-allocate memory for scores and strands
    scores = np.empty(sequence.size - pssm.size / 4 + 1, np.float32)
    strands = np.empty(sequence.size - pssm.size / 4 + 1, np.int32)    
    
    # Score and save output to the pre-allocated arrays
    s = clock()
    cpu_score(pssm, pssm_r, sequence, scores, strands)
    runtime = clock() - s
    
    if benchmark:
        return scores, strands, runtime
    else:
        return scores, strands


def create_pssm(motif_filename, genome_frequencies = [0.25] * 4, epsilon = 1e-50):
    """
    The Position-Specific Scoring Matrix (PSSM) reflects the information
    content represented by the occurrence of each base at each position of a
    sequence of n length in the genome. See:
        http://en.wikipedia.org/wiki/Position-Specific_Scoring_Matrix
    
    This function will generate a PSSM that can be used with the score_sequences()
    function in this script.
    
    Args:
        motif_filename: A file containing the sequences that make up the motif.
            Typically this either a FASTA or plain-text file with one sequence
            per line, where each sequence is a binding site for the DNA-binding
            protein. These sequences must be aligned and of the same length.
        genome_frequencies: The background frequencies of each base in the
            genome. This is the nucleotide composition of the genome in the
            order: A, C, G, T
        epsilon: A small number added to prevent log(0) situations where there
            is an infinitely great negative log-likelihood of a base occuring
            at a position.
        
    Returns:
        pssm: A float64 array of length w * 4, where w is the width of the
            motif.
    """    
    
    # Initialize language and counts
    bases = "ACGT"
    counts = np.array([])
    
    # Read in file "without slurping"
    with open(motif_filename, "r+") as f:
        for line in f:
            if ">" not in line:
                # Initialize counts array
                if counts.size is 0:
                    counts = np.zeros(len(line.strip()) * 4)
                
                # Add to nucleotide count per position
                for pos, base in enumerate(line.strip()):
                    counts[pos * 4 + bases.index(base)] += 1
    
    # Intialize PSSM array
    pssm = np.zeros(counts.size)
    
    # Calculate PSSM
    for pos in range(counts.size / 4):
        total_count = sum(counts[pos:pos + 4])
        for b in range(4):
            # p = frequency of this base in the genome
            p = genome_frequencies[b]
            
            # pseudo_freq = frequency of this base at this position using a
            # Laplacian pseudocount
            pseudo_freq = (counts[pos * 4 + b] + p) / (total_count + 1)
            
            # f = frequency of this base at this position
            #f = counts[pos * 4 + b] / total_count
            #print f
            
            # Each entry in the PSSM is the log-likelihood of that base at that
            # position.
            pssm[pos * 4 + b] = math.log(pseudo_freq / p, 2)
            
            # We add a very small epsilon to avoid log(0) situations, which
            # should yield -Inf, but is not supported by Python. This is known
            # as a computational pseudocount
            #pssm[pos * 4 + b] = math.log(f / p + epsilon, 2)
    return pssm
    
    
# CUDA device kernel
@cuda.jit('void(float32[:], float32[:], int32[:], float32[:], int32[:])')
def cuda_score(pssm, pssm_r, seq, scores, strands):
    """
    This is a CUDA kernel that will score a sequence using a PSSM sliding
    window approach. It parallelizes across all threads to score the entire
    sequence.
    
    Make sure that all the arguments are pointers to device arrays (not host).
    
    Args:
        pssm: This is a float32 array of w * 4 length where w is the width of
            the window. Every four elements correspond to the PSSM score of A,
            C, G, T (in that order).
        pssm_r: This is a float32 array that is the reverse complement of the
            the forward PSSM.
        seq: This is a nucleotide sequence represented in integer form where,
            A = 0, C = 1, G = 2, and T = 3.
        scores: This is the output vector for the score of the window starting
            at ech position.
        strands: This is the output vector for a binary vector that represents 
            which strand each score corresponds to, where 0 = forward and
            1 = reverse strand.
    """
    # Get the unique 1D thread index. Numerically equivalent to:
    # cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    i = cuda.grid(1)
    
    # We may have allocated slightly more threads than necessary, in which case
    # don't continue. scores.shape[0] == len(scores) == len(seq) - w + 1
    # We use .shape since the Python Dialect for CUDA in NumbaPro doesn't
    # support __len__. See: http://docs.continuum.io/numbapro/CUDAPySpec.html
    while i < scores.shape[0]:
        # Initialize scores
        score = 0.0
        score_r = 0.0 # reverse strand
        
        # Loop through each position of our window of length w where:
        # w = width of PSSM = the length of the n-mer we're scoring
        for pos in range(pssm.shape[0] / 4):
            pssm_idx = pos * 4 + seq[i + pos] # index of the cell in the PSSM
            score += pssm[pssm_idx]
            score_r += pssm_r[pssm_idx]
        
        # We keep whichever strand scored the highest and save it
        if score >= score_r:
            scores[i] = score
            strands[i] = 0
        else:
            scores[i] = score_r
            strands[i] = 1
        
        # Iterate to the next chunk
        i += cuda.gridDim.x * cuda.blockDim.x


def cpu_score(pssm, pssm_r, seq, scores, strands):
    """
    This is a CPU-based approach to scoring using a PSSM. This is mainly
    intended for comparison with the GPU approach.
    
    Args:
        pssm: This is a float32 array of w * 4 length where w is the width of
            the window. Every four elements correspond to the PSSM score of A,
            C, G, T (in that order).
        pssm_r: This is a float32 array that is the reverse complement of the
            the forward PSSM.
        seq: This is a nucleotide sequence represented in integer form where,
            A = 0, C = 1, G = 2, and T = 3.
        scores: This is the output vector for the score of the window starting
            at ech position.
        strands: This is the output vector for a binary vector that represents 
            which strand each score corresponds to, where 0 = forward and
            1 = reverse strand.
    """
    # Window size
    w = pssm.size / 4
    
    for pos in range(scores.size):
        window = seq[pos:pos + w]
        
        scores = [pssm[i * 4 + base] for i, base in enumerate(window)]
        score = sum(scores)
        scores_r = [pssm_r[i * 4 + base] for i, base in enumerate(window)]
        score_r = sum(scores_r)
        
        # We keep whichever strand scored the highest and save it
        if score >= score_r:
            scores[i] = score
            strands[i] = 0
        else:
            scores[i] = score_r
            strands[i] = 1    
    

#w = 16
#N = 82e6
##tpb = int(cuda.get_current_device().MAX_THREADS_PER_BLOCK/2)
##bpg = int(math.ceil(float(N)/tpb))
#
#pssm = np.tile([1, 2, 3, 4], w)
#seq = np.random.randint(0, 3, N)
##seq = np.tile([0, 0, 1])
#scores, strands = score_sequence(seq, pssm, verbose=True)
