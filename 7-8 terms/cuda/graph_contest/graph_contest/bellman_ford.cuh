#pragma once
#include <cuda_runtime.h>
#include <float.h>
#include "graph.h"
#define BLOCKSIZE 1024
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void bellmanFordGPU(int *d_scr_ids, int *d_dst_ids, float *d_weight,
    float *d_distances, int source, int edge_num, int ver_num, int iter_num, bool *changes);

    void user_algorithm(Graph graph, int *d_src_ids, int* d_dst_ids,
        float *d_weights, float *d_distances, int source_vertex, int block_num);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

