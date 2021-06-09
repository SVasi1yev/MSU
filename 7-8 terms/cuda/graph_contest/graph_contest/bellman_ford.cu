#include "cuda_error_hadling.h"
#include "bellman_ford.cuh"

#include <iostream>
#include <cuda_runtime.h>



using namespace std;

__global__ void initKernel(float *d_distances, int source, int ver_num)
{
    int it = threadIdx.x;
    int ib = blockIdx.x;
    long long gindex = it + ib * blockDim.x;
    if(gindex < ver_num)
    {
        double max_val = FLT_MAX;
        if(gindex == source)
        {
            d_distances[gindex] = 0;
        }
        else
        {
            d_distances[gindex] = max_val;
        }
    }
}

__global__ void bellmanFordGPU(int *d_scr_ids, int *d_dst_ids, float *d_weight,
    float *d_distances, int source, long long edge_num, int ver_num, int iter_num, int *changes)
{
    int it = threadIdx.x;
    int ib = blockIdx.x;
    long long gindex = it + ib * blockDim.x;
    changes[0] = 0;
    if(gindex < edge_num)
    {
        // if(iter_num == 0)
        // {    
        //     double max_val = FLT_MAX;
        //     if(gindex < ver_num)
        //     {
        //         if(gindex == source)
        //         {
        //             d_distances[gindex] = 0;
        //         }
        //         else
        //         {
        //             d_distances[gindex] = max_val;
        //         }
        //     }
        // }
        float scr_dist = d_distances[d_scr_ids[gindex]];
        float dst_dist = d_distances[d_dst_ids[gindex]];
        float weight = d_weight[gindex];
        if(dst_dist > scr_dist + weight)
        {
            d_distances[d_dst_ids[gindex]] = scr_dist + weight;
            changes[0] = 1;
        }
    }
}

void user_algorithm(Graph graph, int *d_src_ids, int* d_dst_ids,
    float *d_weights, float *d_distances, int source_vertex, int block_num)
{
    // printf("!!!\n");
    int iter_num = 0;
    int* changes;
    int* d_changes;
    changes = (int*)malloc(sizeof(int));
    cudaMalloc((void **)&d_changes, sizeof(int));
    // printf("!!!\n");
    changes[0] = 1;
    // cout << changes[0] << " 1\n";

    int init_block_num = graph.vertices_count / BLOCKSIZE + 1;

    SAFE_KERNEL_CALL((initKernel<<<init_block_num, BLOCKSIZE>>>(d_distances, source_vertex, graph.vertices_count)))

    while(changes[0])
    {
        // printf("%d\n", iter_num);
        SAFE_KERNEL_CALL((bellmanFordGPU<<<block_num, BLOCKSIZE>>>(
                d_src_ids, d_dst_ids, d_weights, d_distances, source_vertex,
                graph.edges_count, graph.vertices_count, iter_num, d_changes)))
        cudaDeviceSynchronize();
        cudaMemcpy(changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);
        iter_num++;
        // cout << changes[0] << " 3\n";
    }

    free(changes);
    cudaFree(d_changes);
}