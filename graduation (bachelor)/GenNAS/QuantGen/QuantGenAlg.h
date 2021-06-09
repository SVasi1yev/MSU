#ifndef QUANTGENALG_H
#define QUANTGENALG_H

#include "QuantIndivid.h"

#include <vector>
#include <random>
#include <omp.h>
#include <memory>
#include <functional>
#include <ctime>
#include <iostream>
#include <mpi.h>
#include <unordered_map>

#define PI 3.1415926

namespace QuantGen {

    double angle_function(char x, char b, double fitness_x, double fitness_b, QuantBit qbit);

    class QuantGenAlg {
    private:
        const int max_pop_size = 999999;
        int global_pop_size;
        int local_pop_size;
        int individ_size;
        std::vector<QuantIndivid> population;

        bool need_mutation;
        double mutation_probability;
        bool need_crossover;
        double crossover_probability;
        std::function<double(const char*, int)> fitness_function;
        std::function<double(char, char, double, double, QuantBit)> angle_function;

        double best_fitness;
        char* best_observation;

        int mpi_size;
        int mpi_rank;
        int omp_size;

        std::unordered_map<int, int> individ_proc_num;
        std::vector<int> proc_pop_size;
        std::vector<int> displs;
        int min_individ_num;

        double start_time = MPI_Wtime();
        double cur_time;
        int seed;
        std::vector<std::mt19937_64> rands;
        std::vector<std::uniform_real_distribution<>> dists;
    public:
        QuantGenAlg(
                int global_pop_size,
                int individ_size,
                bool need_mutation,
                double mutation_probability,
                bool need_crossover,
                double crossover_probability,
                std::function<double(const char*, int)> fitness_function,
                std::function<double(char, char, double, double, QuantBit)> angle_function,
                int mpi_size,
                int mpi_rank,
                int omp_size,
                int seed
        );

        void startAlgorithm(int iter_num);

        void makeObservations();
        void makeRotation();
        void makeMutation();
        void makeCrossover();

        void printPopulation();

        void printParams();

        ~QuantGenAlg() {
            delete[] best_observation;
        }
    };

}

#endif //QUANTGENALG_H
