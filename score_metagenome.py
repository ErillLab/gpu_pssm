# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:50:31 2013

@author: Talmo

This script scores an entire metagenome (the MetaHit database) using a PSSM.
"""
import os
import glob
import time
import numpy as np
import gpu_pssm

total_read_time = 0
total_score_time = 0
convert_time = 0
splice_in_mem_time = 0
scaffold_time = 0
total_bases_in_patients = 0

def score_patient(patient_filename):
    global total_read_time, total_score_time, convert_time, splice_in_mem_time, scaffold_time, total_bases_in_patients
    # Parameters
    motif_file = "../lexA.seq.fa" # FASTA or plain-text file with binding sites
    score_threshold = 8.36 # scores below this threshold will not be reported
    cuda_chunk = 60e6 # how many bases will be sent to the GPU at a time
    
    # Initialize
    bases = "ACGT"
    #bases_char_codes = np.array(bases, "c").view(np.uint8)
    scaffolds = []
    
    # Read patient file
    start = time.time()
    print "Reading in patient file [%s]..." % patient_filename,
    with open(patient_filename, "r+") as f:
        # Pre-allocate metagenome to size of file
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0, os.SEEK_SET)
        metagenome = np.tile(-1, size)
        
        # Process the lines in the file
        pos = 0
        for line in f:
            if ">" not in line:
                s = time.time()
                # Keep sequence as the charcode integers
                seq = np.array(line.strip(), "c").view(np.uint8)
                convert_time += time.time() - s
                
                s = time.time()
                # Replace the sequence chunk in memory
                metagenome[pos:pos + len(seq)] = seq
                splice_in_mem_time += time.time() - s
                
                s = time.time()                
                # Keep track of the beginning of scaffolds
                scaffolds.append(pos)
                scaffold_time += time.time() - s                
                
                # Update position in the metagenome array
                pos += len(seq)
                
    # Truncate the metagenome array to fit the data
    metagenome = metagenome[0:np.where(metagenome == -1)[0][0]]
    
    # Replace character codes with 0-3 int representation of the nucleotides
    for value, base in enumerate(bases):
        metagenome[np.where(metagenome == ord(base))] = value
    
    # Timing for benchmarking
    runtime = time.time() - start
    total_read_time += runtime
    print "Done. [%.5fs]" % runtime
    total_bases_in_patients += metagenome.size
    
    # Calculate frequencies in the genome
    genome_frequencies = np.bincount(metagenome).astype(np.float) / metagenome.size
    
    # Get PSSM
    pssm = gpu_pssm.create_pssm(motif_file, genome_frequencies)
    
    # Pre-allocate scores and strands arrays
    scores = np.empty(metagenome.size, np.float32)
    strands = np.empty(metagenome.size, np.int32)
    
    # Score the genome
    start = time.time()
    print "Scoring...",
    for chunk_start in range(0, metagenome.size, int(cuda_chunk)):
        chunk_scores, chunk_strands = gpu_pssm.score_sequence(metagenome[chunk_start:chunk_start + cuda_chunk], pssm)
        scores[chunk_start:chunk_start + chunk_scores.size] = chunk_scores
        strands[chunk_start:chunk_start + chunk_strands.size] = chunk_strands
    runtime = time.time() - start
    total_score_time += runtime
    print "Done. [%.2fs]" % runtime
    
    # Invalidate scores at positions that cross scaffold boundaries
    w = pssm.size / 4
    for pos in scaffolds:
        if pos - w + 1 > 0:
            scores[pos - w + 1:pos] = -np.inf
    
    # Keep only the scores above score threshold
    high_scores = np.where(scores >= score_threshold)
    scores = scores[high_scores]
    strands = strands[high_scores]
    
    return scores


s = time.time()
# Get filenames
patients = glob.glob("../MetaHit/MH[0-9]*.seq.fa")
#patients = [patients[0]] # For debugging, only score the first patient

# Initialize scores array
scores = np.array([], dtype=np.float32)

# Score each patient
for patient in patients:
    # Score using PSSM (GPU sliding window)
    patient_scores = score_patient(patient)
    
    # Save scores above thrshold
    scores = np.append(scores, patient_scores)
total_time = time.time() - s

print "Scored %d patients in %.2fs." % (len(patients), total_time)
print "Total reading time: %.2fs" % total_read_time
print "Total scoring time: %.2fs" % total_score_time
print "Total bases: %d" % total_bases_in_patients
print "Overall rate: %f bases/sec" % (total_bases_in_patients / total_time)
