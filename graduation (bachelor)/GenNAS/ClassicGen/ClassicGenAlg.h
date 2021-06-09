#ifndef CLASSICGENALG_H
#define CLASSICGENALG_H

#include "ClassicIndivid.h"

#include <vector>
#include <unordered_map>
#include <random>
#include <functional>
#include <omp.h>
#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace ClassicGen {

    class ClassicGenAlg {
    private:
        const int max_pop_size = 999999;
        int global_pop_size;
        int local_pop_size;
        int individ_size;
        std::vector<ClassicIndivid> population;

        double mutation_probability;
        double crossover_probability;
        std::function<double(const char*, int)> fitness_funtion;

        char* best_individ;
        double best_fitness;

        int mpi_size;
        int mpi_rank;
        int omp_size;

        std::unordered_map<int, int> individ_proc_num;
        std::vector<int> proc_pop_size;
        std::vector<int> displs;
        int min_individ_num;

        int seed;
        std::vector<std::mt19937_64> rands;
        std::vector<std::uniform_real_distribution<>> dists;
    public:
        ClassicGenAlg(
                int global_pop_size,
                int individ_size,
                double mutation_probability,
                double crossover_probability,
                std::function<double(const char*, int)> fitness_function,
                int mpi_size,
                int mpi_rank,
                int omp_size,
                int seed
        );

        void startAlgorithm(int iter_num);

        void countFitnesses();
        void makeSelection();
        void makeMutation();
        void makeCrossover();

        void printPopulation();

        void printParams();
    };

}

#endif //CLASSICGENALG_H
