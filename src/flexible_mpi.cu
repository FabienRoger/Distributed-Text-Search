#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

// mem limited to ~40kB per block -> 256 x len < 40kB, len < 128!
// TODO: make this smarter
#define MAX_BLOCK_PER_GRID 65535
#define MAX_PATTERN_LENGTH 16

int THREAD_PER_BLOCK;

__device__ int levenshtein(char *s1, char *s2, int len, int *column)
{
    unsigned int x, y, lastdiag, olddiag;

    for (y = 1; y <= len; y++)
    {
        column[y] = y;
    }
    for (x = 1; x <= len; x++)
    {
        column[0] = x;
        lastdiag = x - 1;
        for (y = 1; y <= len; y++)
        {
            olddiag = column[y];
            column[y] = MIN3(
                column[y] + 1,
                column[y - 1] + 1,
                lastdiag + (s1[y - 1] == s2[x - 1] ? 0 : 1));
            lastdiag = olddiag;
        }
    }
    return (column[len]);
}

__global__ void compute_matches_kernel(char *buf, int start, int end, int n_bytes, int length, char *pattern, int approx_factor, int *n_matches, int max_pattern_length)
{
    extern __shared__ int column[];
    int *my_column = &column[threadIdx.x * max_pattern_length];
    int j;
    int skip_size = blockDim.x * gridDim.x;
    for (j = start + blockIdx.x * blockDim.x + threadIdx.x; j < end; j += skip_size)
    {
        int distance = 0;
        int size;

        size = length;
        if (n_bytes - j < length)
        {
            size = n_bytes - j;
        }

        distance = levenshtein(pattern, &buf[j], size, my_column);

        if (distance <= approx_factor)
        {
            atomicAdd(n_matches, 1);
        }
    }
}

extern "C" void compute_matches_gpu(char *buf, int start, int end, int n_bytes, char **patterns, int starti, int endi, int approx_factor, int max_pattern_length, int *n_matches)
{
    // shifts the buffer and patterns to the start position
    buf = buf + start;
    end = end - start;
    n_bytes = n_bytes - start;
    start = 0;
    patterns = patterns + starti;
    n_matches = n_matches + starti;
    endi = endi - starti;
    starti = 0;

    int i;
    /* Allocate & transfer */
    char *d_buf;
    int *d_n_matches;

    cudaSetDevice(0);
    cudaMalloc((void **)&d_buf, sizeof(char) * n_bytes);
    cudaMalloc((void **)&d_n_matches, sizeof(int) * endi);
    cudaMemcpy(d_buf, buf, sizeof(char) * n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_matches, n_matches, sizeof(int) * endi, cudaMemcpyHostToDevice);

    /* Traverse the patterns */
    for (i = starti; i < endi; i++)
    {
        int length = strlen(patterns[i]);
        char *pattern = patterns[i];
        char *d_pattern;
        cudaMalloc((void **)&d_pattern, sizeof(char) * length);
        cudaMemcpy(d_pattern, pattern, sizeof(char) * length, cudaMemcpyHostToDevice);

        int block_size = THREAD_PER_BLOCK;
        int num_blocks = (end - start + block_size - 1) / block_size;
        if (num_blocks > MAX_BLOCK_PER_GRID)
        {
            num_blocks = MAX_BLOCK_PER_GRID;
        }
        compute_matches_kernel<<<num_blocks, block_size, MAX_PATTERN_LENGTH * THREAD_PER_BLOCK * sizeof(int)>>>(d_buf, start, end, n_bytes, length, d_pattern, approx_factor, d_n_matches + i, max_pattern_length);

        cudaFree(d_pattern);
    }

    /* Transfer back */
    cudaMemcpy(n_matches, d_n_matches, sizeof(int) * endi, cudaMemcpyDeviceToHost);
    /* Free */
    cudaFree(d_buf);
    cudaFree(d_n_matches);

    cudaDeviceSynchronize();
}

extern "C" int big_enough_gpu_available(int max_pattern_length)
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 1)
    {
        return false;
    }
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_shared_memory = prop.sharedMemPerBlock;
    int required_shared_memory = max_pattern_length * THREAD_PER_BLOCK * sizeof(int);

    return required_shared_memory < max_shared_memory;
}