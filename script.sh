#!/bin/bash


mpicc -o out/apm1 src/apm1.c
salloc -N 2 -n 8 mpirun out/apm1 1 obj AAA AAAAA

mpicc -fopenmp -c src/flexible_mpi.c -o out/flexible_mpi
nvcc -I. -c src/flexible_mpi.cu -o out/flexible_mpi_cu
mpicc -fopenmp out/flexible_mpi out/flexible_mpi_cu -lcudart -L/usr/local/cuda/lib64 -o out/flexible_mpi_exec

OMP_NUM_THREADS=8 ONLY_RANK_0=1 PERCENTAGE_GPU=27 DISTRIBUTE_PATTERNS=1 salloc -N 2 -n 8 mpirun out/flexible_mpi_exec 1 obj CUTH Q Q Q Q