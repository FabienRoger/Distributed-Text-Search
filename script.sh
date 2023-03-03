#!/bin/bash


mpicc -c src/flexible_mpi.c -o out/flexible_mpi
nvcc -I. -c src/flexible_mpi.cu -o out/flexible_mpi_cu
mpicc out/flexible_mpi out/flexible_mpi_cu -lcudart -L/usr/local/cuda/lib64 -o out/flexible_exec
salloc -N 2 -n 8 mpirun out/flexible_exec 1 obj ABC