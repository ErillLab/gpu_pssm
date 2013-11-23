GPU PSSM Sliding Window
=======================

This is a GPU-based implementation of a sliding window matrix-scoring method in Python.

It speeds up scoring of large amounts of sequence data, useful for larger datasets like metagenomes.

In this repository, you will find:

- `gpu_pssm.py`: Contains the main scoring function, `score_sequence()`. See the function definition for further documentation and usage examples.
- `cuda_benchmark.py`: Benchmarking code used to help evaluate the best parameters to use with *your* CUDA-capable card.
- `score_metagenome.py`: Utilizes the GPU approach to scoring a large dataset (the [MetaHit database](http://www.nature.com/nature/journal/v464/n7285/full/nature08821.html)) against a motif of [LexA](http://en.wikipedia.org/wiki/Repressor_lexA) binding sites.

The dependencies for these scripts are:
- [NumPy](http://www.numpy.org/)
- [NumbaPro](http://docs.continuum.io/numbapro/): This was chosen as it is NVIDIA's endorsed Python implementation of CUDA. It has good support and is in active development, plus the license is free to academics (anyone with a .edu email address) so it can be freely used for research. To get NumbaPro, first install [Anaconda](https://store.continuum.io/cshop/academicanaconda) and then follow the instructions on the [NumbaPro installation page](http://docs.continuum.io/numbapro/install.html). It's a surprisingly painless install!
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): NVIDIA's CUDA toolkit which is necessary to provide functionality for GPU programming.
- A [CUDA-capable GPU](https://developer.nvidia.com/cuda-gpus) with compute-capability of 2.0 or higher.

This implementation is based on CUDA 5.5 and hopefully will get upgraded when CUDA 6 comes out with [Unified Memory](http://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/).
