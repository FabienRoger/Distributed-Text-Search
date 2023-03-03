#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

int levenshtein(char *s1, char *s2, int len, int *column)
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

extern "C" void compute_matches_gpu(char *buf, int start, int end, int n_bytes, char **pattern, int starti, int endi, int approx_factor, int max_pattern_length, int *n_matches)
{
    int i, j;
    /* Allocate compute buffer */
    int *column;
    column = (int *)malloc((max_pattern_length + 1) * sizeof(int));
    if (column == NULL)
    {
        printf("Error: unable to allocate memory for column (%ldB)\n",
               (max_pattern_length + 1) * sizeof(int));
        exit(1);
    }
    /* Traverse the patterns */
    for (i = starti; i < endi; i++)
    {
        /* Traverse the input data up to the end of the file */
        for (j = start; j < end; j++)
        {
            int size_pattern = strlen(pattern[i]);
            int distance = 0;
            int size;

            size = size_pattern;
            if (n_bytes - j < size_pattern)
            {
                size = n_bytes - j;
            }

            distance = levenshtein(pattern[i], &buf[j], size, column);

            if (distance <= approx_factor)
            {
                n_matches[i]++;
            }
        }
    }
    free(column);
}

__global__ void sum_all_kernel(int *a, int *sum, int n)
{
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? a[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        sum[blockIdx.x] = sdata[0];
    }
}

extern "C" int sum_all(int *a, int n)
{
    int *d_a, *d_sum;
    int sum = 0;

    cudaMalloc((void **)&d_a, sizeof(int) * n);
    cudaMalloc((void **)&d_sum, sizeof(int) * n);

    cudaMemcpy(d_a, a, sizeof(int) * n, cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    sum_all_kernel<<<num_blocks, block_size, block_size * sizeof(int)>>>(d_a, d_sum, n);

    while (num_blocks > 1)
    {
        int threads = block_size;
        int blocks = (num_blocks + threads - 1) / threads;
        sum_all_kernel<<<blocks, threads, threads * sizeof(int)>>>(d_sum, d_sum, num_blocks);
        num_blocks = blocks;
    }

    cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_sum);

    return sum;
}