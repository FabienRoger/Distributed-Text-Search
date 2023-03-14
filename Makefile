OUT_DIR=out
SRC_DIR=src

NAME=flexible_mpi
OUT=$(OUT_DIR)/$(NAME)_exec
TESTNAME=apm1
TESTOUT=$(OUT_DIR)/apm1
CUDAPATH=/usr/local/cuda/lib64

all:
	mkdir -p $(OUT_DIR)
	mpicc -fopenmp -c $(SRC_DIR)/$(NAME).c -o out/$(NAME)
	nvcc -I. -c $(SRC_DIR)/$(NAME).cu -o out/$(NAME)_cu
	mpicc -fopenmp out/$(NAME) out/$(NAME)_cu -lcudart -L$(CUDAPATH) -o $(OUT)

test:
	mkdir -p $(OUT_DIR)
	mpicc -o $(TESTOUT) $(SRC_DIR)/${TESTNAME}.c

clean:
	rm -rf $(OUT_DIR)
