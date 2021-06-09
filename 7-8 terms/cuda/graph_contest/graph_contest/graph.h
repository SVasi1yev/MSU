//
//  graph.h
//  
//
//  Created by Elijah Afanasiev on 23.09.2018.
//
//

#ifndef graph_h
#define graph_h

struct Graph
{
public:
    int *src_ids;
    int *dst_ids;
    float *weights;
    int vertices_count;
    long long edges_count;
};


#endif /* graph_h */
