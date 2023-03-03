#include <string.h>
#include <stdio.h>
#include <stdlib.h>

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