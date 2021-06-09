/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bellman-Ford algorithm
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cuda_runtime.h>
#include <float.h>

void cpu_bellman_ford_edges_list(Graph _graph, float *_distances, int _source_vertex)
{
    int vertices_count = _graph.vertices_count;
    long long edges_count = _graph.edges_count;
    
    int *src_ids = _graph.src_ids;
    int *dst_ids = _graph.dst_ids;
    float *weights = _graph.weights;
    
    //double t1 = omp_get_wtime();
    double max_val = FLT_MAX;
    bool changes = true;
    int iterations_count = 0;
    for (int i = 0; i < vertices_count; i++)
    {
        _distances[i] = max_val;
    }
    _distances[_source_vertex] = 0;
        
    // do bellman-ford algorithm
    while (changes)
    {
        iterations_count++;
        changes = false;
            
        for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
        {
            int sr = src_ids[cur_edge];
            int dst = dst_ids[cur_edge];
            float weight = weights[cur_edge];
                
            float src_distance = _distances[sr];
            float dst_distance = _distances[dst];
                
            if (dst_distance > src_distance + weight)
            {
                _distances[dst] = src_distance + weight;
                changes = true;
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
