/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename _T>
_T rand_uniform_val(int _upper_border)
{
    return (_T)(rand() % _upper_border);
}

template <>
int rand_uniform_val(int _upper_border)
{
    return (int)(rand() % _upper_border);
}

template <>
float rand_uniform_val(int _upper_border)
{
    return (float)(rand() % _upper_border) / _upper_border;
}

template <>
double rand_uniform_val(int _upper_border)
{
    return (double)(rand() % _upper_border) / _upper_border;
}

float rand_flt()
{
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

Graph generate_random_uniform(int _vertices_count, int _average_degree)
{
	// check input parameters correctness
	if (_average_degree > _vertices_count)
		throw "average connections in graph is greater than number of vertices";

	long long edges_count = (long long)_vertices_count * _average_degree;
    
    Graph graph;
    graph.src_ids = new int[edges_count];
    graph.dst_ids = new int[edges_count];
    graph.weights = new float[edges_count];
    graph.vertices_count = _vertices_count;
    graph.edges_count = edges_count;
	for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
	{
		int from = rand() % _vertices_count;
		int to = rand() % _vertices_count;
        
        graph.src_ids[cur_edge] = from;
        graph.src_ids[cur_edge] = to;
        graph.weights[cur_edge] = rand_flt();
	}
    
    return graph;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Graph generate_SSCA2(int _vertices_count, int _max_clique_size)
{
    uint32_t TotVertices;
	uint32_t* clusterSizes;
	uint32_t* firstVsInCluster;
	uint32_t estTotClusters, totClusters;
	
	uint32_t *startVertex, *endVertex;
	long long numEdges;
    long long numIntraClusterEdges, numInterClusterEdges;
    uint32_t MaxCliqueSize;
    uint32_t MaxParallelEdges = 1;
    double ProbUnidirectional = 1.0;
    double ProbIntercliqueEdges = 0.5;
	uint32_t i_cluster, currCluster;
	uint32_t *startV, *endV, *d;
	long long estNumEdges, edgeNum;

	long long i, j, k, t, t1, t2, dsize;
	double p;
	uint32_t* permV;

    // initialize RNG 

	TotVertices = _vertices_count;

	// generate clusters
	MaxCliqueSize = _max_clique_size;
	estTotClusters = 1.25 * TotVertices / (MaxCliqueSize/2);
	clusterSizes = (uint32_t *) malloc(estTotClusters*sizeof(uint32_t));

	for(i = 0; i < estTotClusters; i++) 
	{
		clusterSizes[i] = 1 + (rand_uniform_val<double>(10000.0) *MaxCliqueSize);
	}
	
	totClusters = 0;

	firstVsInCluster = (uint32_t *) malloc(estTotClusters*sizeof(uint32_t));

	firstVsInCluster[0] = 0;
	for (i=1; i<estTotClusters; i++) 
	{
		firstVsInCluster[i] = firstVsInCluster[i-1] + clusterSizes[i-1];
		if (firstVsInCluster[i] > TotVertices-1)
			break;
	}

	totClusters = i;

	clusterSizes[totClusters-1] = TotVertices - firstVsInCluster[totClusters-1];

	// generate intra-cluster edges
	estNumEdges = (uint32_t) ((TotVertices * (double) MaxCliqueSize * (2-ProbUnidirectional)/2) +
		      	        (TotVertices * (double) ProbIntercliqueEdges/(1-ProbIntercliqueEdges))) * (1+MaxParallelEdges/2);

	if ((estNumEdges > ((1<<30) - 1)) && (sizeof(uint32_t*) < 8)) 
	{
		fprintf(stderr, "ERROR: long* should be 8 bytes for this problem size\n");
		fprintf(stderr, "\tPlease recompile the code in 64-bit mode\n");
		exit(-1);
	}

	edgeNum = 0;
	p = ProbUnidirectional;

	fprintf (stderr, "[allocating %3.3f GB memory ... ", (double) 2*estNumEdges*8/(1<<30));

    cout << "alloc of " << sizeof(uint32_t)*estNumEdges / (1024*1024) << " MB memory" << endl;
	startV = (uint32_t *) malloc(estNumEdges*sizeof(uint32_t));
	endV = (uint32_t *) malloc(estNumEdges*sizeof(uint32_t));

	fprintf(stderr, "done] ");  

	for (i_cluster=0; i_cluster < totClusters; i_cluster++)
	{
		for (i = 0; i < clusterSizes[i_cluster]; i++)
		{
			for (j = 0; j < i; j++) 
			{
				for (k = 0; k<1 + ((uint32_t)(MaxParallelEdges - 1) * rand_uniform_val<double>(10000.0)); k++)
				{
					startV[edgeNum] = j + \
					firstVsInCluster[i_cluster];	
					endV[edgeNum] = i + \
					firstVsInCluster[i_cluster];
					edgeNum++;
				}	
			}
			
		}
	}
	numIntraClusterEdges = edgeNum;
	
	//connect the clusters
	dsize = (uint32_t) (log((double)TotVertices)/log(2));
	d = (uint32_t *) malloc(dsize * sizeof(uint32_t));
	for (i = 0; i < dsize; i++) {
		d[i] = (uint32_t) pow(2, (double) i);
	}

	currCluster = 0;

	for (i = 0; i < TotVertices; i++) 
	{
		p = ProbIntercliqueEdges;	
		for (j = currCluster; j<totClusters; j++) 
		{
			if ((i >= firstVsInCluster[j]) && (i < firstVsInCluster[j] + clusterSizes[j]))
			{
				currCluster = j;
				break;
			}	
		}
		for (t = 1; t < dsize; t++)
		{
			j = (i + d[t] + (uint32_t)(rand_uniform_val<double>(10000.0) * (d[t] - d[t - 1]))) % TotVertices;
			if ((j<firstVsInCluster[currCluster]) || (j>=firstVsInCluster[currCluster] + clusterSizes[currCluster]))
			{
				for (k = 0; k<1 + ((uint32_t)(MaxParallelEdges - 1)* rand_uniform_val<double>(10000.0)); k++)
				{
					if (p >  rand_uniform_val<double>(10000.0)) 
					{
						startV[edgeNum] = i;
						endV[edgeNum] = j;
						edgeNum++;	
					}	
				}	
			}
			p = p/2;
		}
	}
	
	numEdges = edgeNum;
	numInterClusterEdges = numEdges - numIntraClusterEdges;	

	free(clusterSizes);  
	free(firstVsInCluster);
	free(d);

	fprintf(stderr, "done\n");
	fprintf(stderr, "\tNo. of inter-cluster edges - %d\n", numInterClusterEdges);
	fprintf(stderr, "\tTotal no. of edges - %d\n", numEdges);

	// shuffle vertices to remove locality	
	fprintf(stderr, "Shuffling vertices to remove locality ... ");
	fprintf(stderr, "[allocating %3.3f GB memory ... ", (double)(TotVertices + 2 * numEdges) * 8 / (1 << 30));

	permV = (uint32_t *)malloc(TotVertices*sizeof(uint32_t));
	startVertex = (uint32_t *)malloc(numEdges*sizeof(uint32_t));
	endVertex = (uint32_t *)malloc(numEdges*sizeof(uint32_t));

	for (i = 0; i<TotVertices; i++) 
	{
		permV[i] = i;
	}

	for (i = 0; i<TotVertices; i++) 
	{
		t1 = i + rand_uniform_val<double>(10000.0) * (TotVertices - i);
		if (t1 != i)
		{
			t2 = permV[t1];
			permV[t1] = permV[i];
			permV[i] = t2;
		}
	}

	for (i = 0; i<numEdges; i++) 
	{
		startVertex[i] = permV[startV[i]];
		endVertex[i] = permV[endV[i]];
	}

	free(startV);
	free(endV);
	free(permV);

	// generate edge weights
  
	vector<vector<uint32_t>> dests(TotVertices);

    Graph graph;
    graph.src_ids = new int[numEdges];
    graph.dst_ids = new int[numEdges];
    graph.weights = new float[numEdges];
    graph.vertices_count = _vertices_count;
    graph.edges_count = numEdges;
    
    // add edges to graph
	for (uint32_t i = 0; i < numEdges; i++) 
	{
        graph.src_ids[i] = startVertex[i];
        graph.dst_ids[i] = endVertex[i];
        graph.weights[i] = rand_flt();
	}
	fprintf(stderr, "done\n");
    
    return graph;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Graph generate_R_MAT(int _vertices_count, int _average_connections)
{
    int n = (int)log2(_vertices_count);
    int vertices_count = _vertices_count;
    
    int _a_prob = 57;
    int _b_prob = 19;
    int _c_prob = 19;
    int _d_prob = 5;
    
    long long edges_count = ((long long)_vertices_count) * _average_connections;
    
    Graph graph;
    graph.src_ids = new int[edges_count];
    graph.dst_ids = new int[edges_count];
    graph.weights = new float[edges_count];
    graph.vertices_count = _vertices_count;
    graph.edges_count = edges_count;
    
    int _omp_threads = omp_get_max_threads();
    
    // generate and add edges to graph
    unsigned int seed = 0;
    #pragma omp parallel num_threads(_omp_threads) private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();
        
        #pragma omp for schedule(static)
        for (long long cur_edge = 0; cur_edge < edges_count; cur_edge++)
        {
            int x_middle = _vertices_count / 2, y_middle = _vertices_count / 2;
            for (long long i = 1; i < n; i++)
            {
                int a_beg = 0, a_end = _a_prob;
                int b_beg = _a_prob, b_end = b_beg + _b_prob;
                int c_beg = _a_prob + _b_prob, c_end = c_beg + _c_prob;
                int d_beg = _a_prob + _b_prob + _c_prob, d_end = d_beg + _d_prob;
            
                int step = (int)pow(2, n - (i + 1));
            
                int probability = rand_r(&seed) % 100;
                if (a_beg <= probability && probability < a_end)
                {
                    x_middle -= step, y_middle -= step;
                }
                else if (b_beg <= probability && probability < b_end)
                {
                    x_middle -= step, y_middle += step;
                }
                else if (c_beg <= probability && probability < c_end)
                {
                    x_middle += step, y_middle -= step;
                }
                else if (d_beg <= probability && probability < d_end)
                {
                    x_middle += step, y_middle += step;
                }
            }
            if (rand_r(&seed) % 2 == 0)
                x_middle--;
            if (rand_r(&seed) % 2 == 0)
                y_middle--;
        
            int from = x_middle;
            int to = y_middle;
        
            graph.src_ids[cur_edge] = from;
            graph.dst_ids[cur_edge] = to;
            graph.weights[cur_edge] = rand_flt();
        }
    }
    
    return graph;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
