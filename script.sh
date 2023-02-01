#!/bin/bash


mpicc -o out/apm1 src/apm1.c -Wall;
mpicc -o out/apm2 src/apm2.c -Wall;
./apm 1 dna/large GGCCAGGGGCACGTGGAAGAAGCTATCGTGGCAAAGGGAGCAGTCATATC;
salloc -N 1 -n 1 mpirun out/apm1 1 dna/large GGCCAGGGGCACGTGGAAGAAGCTATCGTGGCAAAGGGAGCAGTCATATC;
salloc -N 1 -n 1 mpirun out/apm2 1 dna/large GGCCAGGGGCACGTGGAAGAAGCTATCGTGGCAAAGGGAGCAGTCATATC;