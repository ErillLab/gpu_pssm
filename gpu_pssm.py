# -*- coding: utf-8 -*-
"""
Created on Thu Nov 07 17:52:15 2013

@author: Talmo

This file provides functions that can be used to score a sequence of
nucleotide bases using a GPU-based PSSM approach.

See readme.md for information on dependencies and what is needed to run this.

See the documentation for score_sequence() or score_long_sequence() for usage
examples and more help:
    >>> import gpu_pssm
    >>> help(gpu_pssm.score_sequence)
    Help on function score_sequence in module gpu_pssm:
    
    score_sequence(seq, pssm, verbose=False, keep_strands=True, benchmark=False, blocks_per_grid=-1, threads_per_block=-1)
        This function will score a sequence of nucleotides based on a PSSM by using
        a sliding window parallelized on a GPU.
    ...
"""

import sys
import math
import numpy as np
from numbapro import cuda
from time import time, clock

"""
Beware all ye who enter here: Here lies module level code!

Justification:
In Windows, clock() provides a more accurate reading into the timing value,
with resolution of less than 1 microsecond (1e-6) since it returns the actual
time spent in the function through QueryPerformanceCounter. See:
http://msdn.microsoft.com/en-us/library/windows/desktop/ms644904(v=vs.85).aspx
Additionally,

In Unix, we prefer to use time() since clock() counts CPU time, NOT the actual
time. This means that any time spent "sleeping" will not be counted. 

The major disadvantage is that time() returns the time since epoch, but can be
inaccurate as a timer if the the system time changes in between calls.

This preference is reflected in Python's timeit module (see default_timer()).

For a further discussion:
http://www.pythoncentral.io/measure-time-in-python-time-time-vs-time-clock/
http://docs.python.org/2.7/library/timeit.html#timeit.default_timer
http://docs.python.org/2.7/library/time.html#time.clock
"""
# From the timeit module:
if sys.platform == 'win32':
    default_timer = clock # On Windows, the best timer is time.clock
else:
    default_timer = time # On most other platforms the best timer is time.time


def score_sequence(seq, pssm, verbose = False, keep_strands = True, benchmark = False, blocks_per_grid = -1, threads_per_block = -1):
    """
    This function will score a sequence of nucleotides based on a PSSM by using
    a sliding window parallelized on a GPU.
    
    Args:
        seq: This must be an integer representation of the nucleotide sequence,
            where the alphabet is (A = 0, C = 1, G = 2, T = 3). It must be a 
            vector (1D array) of integers that can be cast to int32 (See: 
            numpy.int32).
        pssm: This must a vectorized PSSM where every four elements correspond 
            to one position. Make sure this can be cast to an array of float64.
        verbose: Set this to True to print performance information.
        benchmark: If set to True, the function will return information about
            the run in a dictionary at the third output variable.
        keep_strands: Whether memory should be allocated for storing which
            strand the scores come from. Set this to False if you just want the
            scores and the strands array will not be returned.
            NOTE: If this and benchmark are set to False, then the scores will
            not be returned in a tuple, meaning:
                >>> score_sequence
        blocks_per_grid: This is the blocks per grid that will be assigned to 
            the CUDA kernel. See this SO question for info on choosing this
            value: http://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
            It defaults to the length of the sequence or the maximum number of
            blocks per grid supported by the GPU, whichever is lower.
            Set this to a negative number
        threads_per_block: Threads per block. See above. It defaults to 55% of
            the maximum number of threads per block supported by the GPU, a
            value determined experimentally. Higher values will likely result
            in failure to allocate resources to the kernel (since there will
            not be enough register space for each thread).
        
    Returns:
        scores: 1D float64 array of length (n - w + 1), where n is the length
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
            Note that the memory_used is rather misleading if running the
            function more than once. CUDA is optimized to not transfer the same
            data from the host to the device so it will not always change. It
            may also unload other assets from memory, so the memory changed can
            be negative.
            TODO: Find a better method of calculating memory usage.
            
    Example:
        >>> pssm = np.random.uniform(-7.5, 2.0, 4 * 16) # Window size of 16
        >>> seq = np.random.randint(0, 3, 30e6) # Generate random 30 million bp sequence
        >>> scores, strands, run_info = score_sequence(seq, pssm, benchmark=True, verbose=True)
        Threads per block = 563
        Blocks per grid = 53286
        Total threads = 30000018
        Scoring... Done.
        Genome size: 3e+07 bp
        Time: 605.78 ms
        Speed: 4.95229e+07 bp/sec
        >>> scores
        array([-16.97089798, -33.48925866, -21.80381526, ..., -10.27919401,
               -32.64575614, -23.97110103])
        >>> strands
        array([1, 1, 1, ..., 1, 1, 0])
        >>> run_info
        {'memory_used': 426508288L, 'genome_size': 30000000, 'runtime': 0.28268090518054123, 'threads_per_block': 563, 'blocks_per_grid': 53286}
        
    A more interesting interpretation of the run information for performance 
    analysis is the number of bases score per second:
        >>> print "%g bases/sec" % run_info["genome_size"] / run_info["runtime"]
        1.06127e+08 bases/sec
    """
    w = int(pssm.size / 4) # width of PSSM
    n = int(seq.size) # length of the sequence being scored
    
    # Calculate the reverse-complement of the PSSM
    pssm_r = np.array([pssm[i / 4 + (3 - (i % 4))] for i in range(pssm.size)][::-1])

    # Calculate the appropriate threads per block and blocks per grid    
    if threads_per_block <= 0 or blocks_per_grid <= 0:
        # We don't use the max number of threads to avoid running out of
        # register space by saturating the streaming multiprocessors
        # ~55% was found empirically, but your mileage may vary with different GPUs
        threads_per_block = int(cuda.get_current_device().MAX_BLOCK_DIM_X * 0.55)
        
        # We saturate our grid and let the dynamic scheduler assign the blocks
        # to the discrete CUDA cores/streaming multiprocessors
        blocks_per_grid = int(math.ceil(float(n) / threads_per_block))
        if blocks_per_grid > cuda.get_current_device().MAX_GRID_DIM_X:
            blocks_per_grid = cuda.get_current_device().MAX_GRID_DIM_X
    
     
    
    if verbose:
        print "Threads per block = %d" % threads_per_block
        print "Blocks per grid = %d" % blocks_per_grid
        print "Total threads = %d" % (threads_per_block * blocks_per_grid)
    


    i
    
    # Collect benchmarking info
    s = default_timer()
    start_mem = cuda.get_current_device().get_memory_info()[0]    
    
    # Start a stream
    stream = cuda.stream()
    
    # Copy data to device
    d_pssm = cuda.to_device(pssm.astype(np.float64), stream)
    d_pssm_r = cuda.to_device(pssm_r.astype(np.float64), stream)
    d_seq = cuda.to_device(seq.astype(np.int32), stream)
    
    # Allocate memory on device to store results
    d_scores = cuda.device_array(n - w + 1, dtype=np.float64, stream=stream)
    if keep_strands:
        d_strands = cuda.device_array(n - w + 1, dtype=np.int32, stream=stream)
        
    # Run the kernel
    if keep_strands:
        cuda_score[blocks_per_grid, threads_per_block](d_pssm, d_pssm_r, d_seq, d_scores, d_strands)
    else:
        cuda_score_without_strands[blocks_per_grid, threads_per_block](d_pssm, d_pssm_r, d_seq, d_scores)
    
    # Copy results back to host
    scores = d_scores.copy_to_host(stream=stream)
    if keep_strands:
        strands = d_strands.copy_to_host(stream=stream)
    stream.synchronize()
    
    # Collect benchmarking info
    end_mem = cuda.get_current_device().get_memory_info()[0]
    t = default_timer() - s
    
    # Output info on the run if verbose parameter is true
    if verbose:
        print "Genome size: %g bp" % n
        print "Time: %.2f ms (using time.%s())" % (t * 1000, default_timer.__name__)
        print "Speed: %g bp/sec" % (n / t)
        print "Global memory: %d bytes used (%.2f%% of total)" % \
            (start_mem - end_mem, float(start_mem - end_mem) * 100 / cuda.get_current_device().get_memory_info()[1])
    
    # Return the run information for benchmarking
    run_info = {"genome_size": n, "runtime": t, "memory_used": start_mem - end_mem, \
                "blocks_per_grid": blocks_per_grid, "threads_per_block": threads_per_block}
                
    # I'm so sorry BDFL, please don't hunt me down for returning different size
    # tuples in my function
    if keep_strands:
        if benchmark:
            return (scores, strands, run_info)
        else:
            return (scores, strands)
    else:
        if benchmark:
            return (scores, run_info)
        else:
            # Careful! This won't return a tuple, so you don't need to do
            # score_sequence[0] to get the scores
            return scores


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
        
    See score_sequence() for (a lot) more information on these parameters and
    return values.
    """
    
    # Pre-allocate memory for scores and (maybe) strands
    scores = np.empty(sequence.size - pssm.size / 4 + 1, np.float64)
    if keep_strands:
        strands = np.empty(sequence.size - pssm.size / 4 + 1, np.int32)    
    
    for chunk_start in range(0, sequence.size, int(chunk_size)):
        # Score chunk
        if keep_strands:
            chunk_scores, chunk_strands = score_sequence(sequence[chunk_start:chunk_start + chunk_size], pssm)
        else:
            chunk_scores = score_sequence(sequence[chunk_start:chunk_start + chunk_size], pssm, keep_strands=False)
        
        # Save output to array segment
        scores[chunk_start:chunk_start + chunk_scores.size] = chunk_scores
        if keep_strands:
            strands[chunk_start:chunk_start + chunk_strands.size] = chunk_strands
    
    if keep_strands:
        return scores, strands
    else:
        return scores


def score_sequence_with_cpu(sequence, pssm, keep_strands=True, benchmark=False):
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
            TODO: Calculate the memory used; other useful benchmarks.
            
    Returns:
        scores: An array of scores of the length of the sequence minus the
            window size (width of the PSSM).
        strands: An array indicating which strand the score corresponds to.
        run_info: This dictionary is only returned if benchmark is set to True.
            It contains benchmarking information in the following keys:
            >>> run_info.keys()
            ['genome_size', 'runtime']
            
            
    """
        
    # Calculate the reverse-complement of the PSSM
    pssm_r = np.array([pssm[i / 4 + (3 - (i % 4))] for i in range(pssm.size)][::-1])
    
    # Pre-allocate memory for scores and strands
    scores = np.empty(sequence.size - pssm.size / 4 + 1, np.float64)
    strands = np.empty(sequence.size - pssm.size / 4 + 1, np.int32)    
    
    # Score and save output to the pre-allocated arrays
    s = default_timer()
    cpu_score(pssm, pssm_r, sequence, scores, strands)
    t = default_timer() - s
    
    run_info = {"genome_size": n, "runtime": t}    
    
    if keep_strands:
        if benchmark:
            return (scores, strands, run_info)
        else:
            return (scores, strands)
    else:
        if benchmark:
            return (scores, run_info)
        else:
            # Careful! This won't return a tuple, so you don't need to do
            # score_sequence[0] to get the scores
            return scores


def create_pssm(motif_filename, genome_frequencies = [0.25] * 4):
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
            order: A, C, G, T. If not specified, they are assumed to be
            equiprobable.
        
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
        total_count = sum(counts[pos * 4:pos * 4 + 4])
        for b in range(4):
            # p = frequency of this base in the genome
            p = genome_frequencies[b]
            
            ##### Laplacian pseudocount
            # pseudo_freq = frequency of this base at this position with the
            # added probability of one more event
            pseudo_freq = (counts[pos * 4 + b] + p) / (total_count + 1)
            
            # Each entry in the PSSM is the log-likelihood of that base at that
            # position.
            pssm[pos * 4 + b] = math.log(pseudo_freq / p, 2)
            
            
            ##### Computational pseudocount
            # f = frequency of this base at this position in the motif
            #f = counts[pos * 4 + b] / total_count
            
            # We add a very small epsilon to avoid log(0) situations, which
            # should yield -Inf, but is not supported by Python
            #epsilon = 1e-50
            #pssm[pos * 4 + b] = math.log(f / p + epsilon, 2)
    return pssm


@cuda.jit('void(float64[:], float64[:], int32[:], float64[:], int32[:])')
def cuda_score(pssm, pssm_r, seq, scores, strands):
    """
    This is a CUDA kernel that will score a sequence using a PSSM sliding
    window approach. It parallelizes across all threads to score the entire
    sequence.
    
    Make sure that all the arguments are pointers to device arrays (not host).
    
    Args:
        pssm: This is a float64 array of w * 4 length where w is the width of
            the window. Every four elements correspond to the PSSM score of A,
            C, G, T (in that order).
        pssm_r: This is a float64 array that is the reverse complement of the
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


@cuda.jit('void(float64[:], float64[:], int32[:], float64[:])')
def cuda_score_without_strands(pssm, pssm_r, seq, scores):
    """
    This is a CUDA kernel that is virtually identical to cuda_score, with the
    exception that it does not save any information about which strand the
    score comes from. This saves a couple of operations and a fair amount of
    memory depending on the size of the sequence.
    
    It DOES, however, still compare both the forward and reverse strands, it
    just keeps the highest one without keeping track of where it came from.
    
    This should be used when scoring for the purposes of analyzing the
    distribution of scores and getting the actual sites is not too important.
    
    Make sure that all the arguments are pointers to device arrays (not host).
    
    Args:
        pssm: This is a float64 array of w * 4 length where w is the width of
            the window. Every four elements correspond to the PSSM score of A,
            C, G, T (in that order).
        pssm_r: This is a float64 array that is the reverse complement of the
            the forward PSSM.
        seq: This is a nucleotide sequence represented in integer form where,
            A = 0, C = 1, G = 2, and T = 3.
        scores: This is the output vector for the score of the window starting
            at ech position.
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
        scores[i] = max(score, score_r)
        
        # Iterate to the next chunk
        i += cuda.gridDim.x * cuda.blockDim.x


def cpu_score(pssm, pssm_r, seq, scores, strands):
    """
    This is a CPU-based approach to scoring using a PSSM. This is mainly
    intended for comparison with the GPU approach.
    
    Note: This function is very slow! It's intended to be directly compared
    with the CUDA implementation in cuda_score(), not for "daily use".
    
    Args:
        pssm: This is a float64 array of w * 4 length where w is the width of
            the window. Every four elements correspond to the PSSM score of A,
            C, G, T (in that order).
        pssm_r: This is a float64 array that is the reverse complement of the
            the forward PSSM.
        seq: This is a nucleotide sequence represented in integer form where,
            A = 0, C = 1, G = 2, and T = 3.
        scores: This is the output vector for the score of the window starting
            at each position.
        strands: This is the output vector for a binary vector that represents 
            which strand each score corresponds to, where 0 = forward and
            1 = reverse strand.
    """
    for i in range(scores.size):
        score = 0.0
        score_r = 0.0
        for pos in range(0, pssm.size / 4):
            base = seq[i + pos]
            score += pssm[pos * 4 + base]
            score_r += pssm_r[pos * 4 + base]
        
        # We keep whichever strand scored the highest and save it
        if score >= score_r:
            scores[i] = score
            strands[i] = 0
        else:
            scores[i] = score_r
            strands[i] = 1    

