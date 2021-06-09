#include "verification.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_error_hadling.h"

#define TEST_ITERATION_COUNT 10
#define BLOCKSIZE 1024

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vector<string> split(const string& str, int delimiter(int) = ::isspace){
  vector<string> result;
  auto e=str.end();
  auto i=str.begin();
  while(i!=e){
    i=find_if_not(i,e, delimiter);
    if(i==e) break;
    auto j=find_if(i,e, delimiter);
    result.push_back(string(i,j));
    i=j;
  }
  return result;
}
 
void parse_cmd_params(int _argc, char **_argv, int &_scale, int &_avg_degree,
                      bool &_check, string &_graph_type)
{
    // get params from cmd line
    string all_params;
    for (int i = 1; i < _argc; i++)
    {
        string option(_argv[i]);
        all_params += option + " ";
    }
     
    std::vector<std::string>vstrings = split(all_params);
     
    for (int i = 0; i < vstrings.size(); i++)
    {
        string option = vstrings[i];
         
        cout << "option: " << option << endl;
         
        if (option.compare("--s") == 0)
        {
            _scale = atoi(vstrings[++i].c_str());
        }
         
        if (option.compare("--e") == 0)
        {
            _avg_degree = atoi(vstrings[++i].c_str());
        }
         
        if (option.compare("--nocheck") == 0)
        {
            _check = false;
        }
         
        if (option.compare("--rmat") == 0)
        {
            _graph_type = "rmat";
        }
         
        if (option.compare("--random_uniform") == 0)
        {
            _graph_type = "random_uniform";
        }
         
        if (option.compare("--ssca2") == 0)
        {
            _graph_type = "ssca2";
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void user_copy_graph_to_device(Graph graph, int **d_src_ids, int **d_dst_ids,
    float **d_weights, float **d_distances)
{
    SAFE_CALL((cudaHostRegister(graph.src_ids, graph.edges_count * sizeof(int), 0)));
    SAFE_CALL((cudaHostRegister(graph.dst_ids, graph.edges_count * sizeof(int), 0)));
    SAFE_CALL((cudaHostRegister(graph.weights, graph.edges_count * sizeof(float), 0)));

    SAFE_CALL((cudaMalloc((void **)d_src_ids, graph.edges_count * sizeof(int))));
    SAFE_CALL((cudaMalloc((void **)d_dst_ids, graph.edges_count * sizeof(int))));
    SAFE_CALL((cudaMalloc((void **)d_weights, graph.edges_count * sizeof(float))));
    SAFE_CALL((cudaMalloc((void **)d_distances, graph.vertices_count * sizeof(float))));

    SAFE_CALL((cudaMemcpy(*d_src_ids, graph.src_ids, graph.edges_count * sizeof(int), cudaMemcpyHostToDevice)));
    SAFE_CALL((cudaMemcpy(*d_dst_ids, graph.dst_ids, graph.edges_count * sizeof(int), cudaMemcpyHostToDevice)));
    SAFE_CALL((cudaMemcpy(*d_weights, graph.weights, graph.edges_count * sizeof(float), cudaMemcpyHostToDevice)));
}



void user_copy_result_back_to_host_and_free_memory(float *user_distances, int *d_src_ids,
    int* d_dst_ids, float *d_weights, float *d_distances, int vertices_count)
{
    cudaMemcpy(user_distances, d_distances, vertices_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_src_ids);
    cudaFree(d_dst_ids);
    cudaFree(d_weights);
    cudaFree(d_distances);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	try
	{
        // считываем параметры командной сторки
        int scale = 12;
        int avg_degree = 15;
        string graph_type = "rmat";
        bool check = true;
        parse_cmd_params(argc, argv, scale, avg_degree, check, graph_type);
        cout << "cmd parameters parsed" << endl;
        
        // генерируем граф
        Graph graph;
        if (graph_type == "rmat")
            graph = generate_R_MAT(pow(2.0, scale), avg_degree);
        else if (graph_type == "random_uniform")
            graph = generate_random_uniform(pow(2.0, scale), avg_degree);
        else if (graph_type == "ssca2")
            graph = generate_SSCA2(pow(2.0, scale), avg_degree);
        else
        {
            cout << "Unknown graph type" << endl;
            return 1;
        }
        cout << "graph generated" << endl;
        
        // выделяем память под ответ
        float *user_result = new float[graph.vertices_count];
        
        // запускаем копирования данных и алгоритм
        double t1 = omp_get_wtime();
        
        int last_source = 0;

        int *d_src_ids;
        int *d_dst_ids;
        float *d_weights;
        float *d_distances;

        int block_size = (((int)(graph.edges_count)) / BLOCKSIZE) + 1;

        user_copy_graph_to_device(graph, &d_src_ids, &d_dst_ids, &d_weights,
            &d_distances);
        for(int i = 0; i < TEST_ITERATION_COUNT; i++)
        {
            last_source = i % (graph.vertices_count);
            user_algorithm(graph, d_src_ids, d_dst_ids, d_weights,
                d_distances, last_source, block_size);
        }
        user_copy_result_back_to_host_and_free_memory(user_result, d_src_ids, d_dst_ids,
            d_weights, d_distances, graph.vertices_count);
        
        double t2 = omp_get_wtime();
        cout << "#algorithm executed!" << endl;
        cout << "#perf: " << ((double)(TEST_ITERATION_COUNT) * graph.edges_count) / ((t2 - t1) * 1e6) << endl;
        cout << "#time: " << t2 - t1 << endl;
        cout << "#check: " << check << endl;
        
        // делаем проверку корректности каждый раз
        if(check)
        {
            verify_result(graph, user_result, last_source);
        }
        
        delete []graph.src_ids;
        delete []graph.dst_ids;
        
        // освобождаем память
        delete []user_result;
        
	}
	catch (const char *error)
	{
		cout << error << endl;
	}
	catch (...)
	{
		cout << "unknown error" << endl;
	}

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
