#!/bin/bash

# mpicc -o out/apm1 src/apm1.c
# salloc -N 2 -n 8 mpirun out/apm1 1 obj AAA AAAAA

mpicc -fopenmp -c src/flexible_mpi.c -o out/flexible_mpi
nvcc -I. -c src/flexible_mpi.cu -o out/flexible_mpi_cu
mpicc -fopenmp out/flexible_mpi out/flexible_mpi_cu -lcudart -L/usr/local/cuda/lib64 -o out/flexible_mpi_exec
# THREAD_PER_BLOCK=1 BLOCK_PER_GRID=1 USE_GPU=1 salloc -N 2 -n 8 mpirun out/flexible_exec 1 obj AAA AAAAA

USE_GPU=0 DISTRIBUTE_PATTERNS=1 salloc -N 2 -n 8 mpirun /users/eleves-a/2020/$(whoami)/TD/INF560/Distributed-Text-Search/out/flexible_mpi_exec 1 /users/eleves-a/2020/$(whoami)/TD/INF560/Distributed-Text-Search/obj CUTH Q Q Q Q