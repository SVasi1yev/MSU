#include "ClassicIndivid.cpp"

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

using namespace ClassicGen;

ClassicGenAlg::ClassicGenAlg(
        int global_pop_size,
        int individ_size,
        double mutation_probability,
        double crossover_probability,
        std::function<double(const char *, int)> fitness_function,
        int mpi_size,
        int mpi_rank,
        int omp_size,
        int seed
):
        global_pop_size(global_pop_size),
        individ_size(individ_size),
        mutation_probability(mutation_probability),
        crossover_probability(crossover_probability),
        fitness_funtion(fitness_function),
        mpi_size(mpi_size),
        mpi_rank(mpi_rank),
        omp_size(omp_size),
        seed(seed),
        best_fitness(-std::numeric_limits<double>::infinity())
{
    if (global_pop_size > max_pop_size) {
        throw std::runtime_error("Max population size = 999999");
    }
    if (mpi_rank < global_pop_size % mpi_size) {
        local_pop_size = global_pop_size / mpi_size + 1;
    } else {
        local_pop_size = global_pop_size / mpi_size;
    }

    for (int i = 0; i < omp_size; i++) {
        rands.push_back(std::mt19937_64((mpi_rank * 1000 + i + 1) * seed));
        dists.push_back(std::uniform_real_distribution<>(0, 1));
    }

    for (int i = 0; i < local_pop_size; i++) {
        population.push_back(ClassicIndivid(individ_size));
    }
    best_individ = new char[individ_size];

    omp_set_num_threads(omp_size);
    omp_set_dynamic(0);
    omp_set_schedule(omp_sched_dynamic, 1);
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < local_pop_size; i++) {
            population[i].uniformInit(rands[thread_num], dists[thread_num]);
        }
    }

    int individ_num = 0;
    for (int i = 0; i < mpi_size; i++) {
        int temp_local_size;
        if (i < global_pop_size % mpi_size) {
            temp_local_size = global_pop_size / mpi_size + 1;
        } else {
            temp_local_size = global_pop_size / mpi_size;
        }

        proc_pop_size.push_back(temp_local_size);
        displs.push_back(individ_num);
        for (int j = 0; j < temp_local_size; j++) {
            individ_proc_num[individ_num] = i;
            individ_num++;
        }
    }
    min_individ_num = displs[mpi_rank];
    MPI_Barrier(MPI_COMM_WORLD);
}

void ClassicGenAlg::startAlgorithm(int iter_num) {
    countFitnesses();
    printPopulation();
    for (int i = 0; i < iter_num; i++) {
        makeSelection();
        makeCrossover();
        makeMutation();
        countFitnesses();
        printPopulation();
    }
}

void ClassicGenAlg::countFitnesses() {
    double cur_best_fitness = best_fitness;
    int cur_best_individ_ind = -1;

#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < local_pop_size; i++) {
            population[i].countFitness(fitness_funtion);
#pragma omp critical
            {
                if (population[i].getFitness() > cur_best_fitness) {
                    cur_best_fitness = population[i].getFitness();
                    cur_best_individ_ind = i;
                }
            }
        }
    }

    struct {
        double fitness;
        int rank;
    } local, global;

    local.fitness = cur_best_fitness;
    local.rank = mpi_rank;

    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

    if (global.rank == mpi_rank) {
        if (cur_best_fitness > best_fitness) {
            best_fitness = cur_best_fitness;
            std::copy(
                    population[cur_best_individ_ind].getIndivid(),
                    population[cur_best_individ_ind].getIndivid() + individ_size,
                    best_individ
            );
            MPI_Bcast(
                    best_individ, individ_size,
                    MPI_UNSIGNED_CHAR, global.rank, MPI_COMM_WORLD
            );
        }
    } else {
        if (global.fitness > best_fitness) {
            best_fitness = global.fitness;
            MPI_Bcast(
                    best_individ, individ_size,
                    MPI_UNSIGNED_CHAR, global.rank, MPI_COMM_WORLD
            );
        }
    }
}

void ClassicGenAlg::makeSelection() {
    double* local_fitnesses = new double[local_pop_size];
    for (int i = 0; i < local_pop_size; i++) {
        local_fitnesses[i] = population[i].getFitness();
    }

    double* global_fitnesses;
    if (mpi_rank == 0) {
        global_fitnesses = new double[global_pop_size];
    }
    MPI_Gatherv(
            local_fitnesses, local_pop_size, MPI_DOUBLE,
            global_fitnesses, proc_pop_size.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD
    );
    delete[] local_fitnesses;

    int* selection_pairs;
    int selection_pairs_size;

    if (mpi_rank == 0) {
        std::vector<int> individs_perm;
        for (int i = 0; i < global_pop_size; i++) {
            individs_perm.push_back(i);
        }
        std::shuffle(individs_perm.begin(), individs_perm.end(), rands[0]);
        int max = global_fitnesses[0];
        int argmax = 0;
        for (int i = 1; i < global_pop_size; i++) {
            if (global_fitnesses[i] > max) {
                max = global_fitnesses[i];
                argmax = i;
            }
        }
        if (individs_perm.size() % 2 == 1) {
            if (individs_perm[individs_perm.size() - 1] == argmax) {
                std::swap(individs_perm[0], individs_perm[individs_perm.size() - 1]);
            }
            individs_perm.pop_back();
        }

        std::vector<int>* proc_num_individs = new std::vector<int>[mpi_size];
        for (int i = 0; i < individs_perm.size(); i+= 2) {
            int selection_pair[2] = {individs_perm[i], individs_perm[i + 1]};
            if (global_fitnesses[selection_pair[0]] < global_fitnesses[selection_pair[1]]) {
                std::swap(selection_pair[0], selection_pair[1]);
            }
            int dist1 = individ_proc_num[selection_pair[0]];
            int dist2 = individ_proc_num[selection_pair[1]];
            if (dist1 != dist2) {
                proc_num_individs[dist1].push_back(selection_pair[0]);
                proc_num_individs[dist1].push_back(selection_pair[1]);
                proc_num_individs[dist2].push_back(selection_pair[0]);
                proc_num_individs[dist2].push_back(selection_pair[1]);
            } else {
                proc_num_individs[dist1].push_back(selection_pair[0]);
                proc_num_individs[dist1].push_back(selection_pair[1]);
            }
        }

        for (int i = 1; i < mpi_size; i++) {
            MPI_Send(
                    proc_num_individs[i].data(), proc_num_individs[i].size(),
                    MPI_INT, i, 0, MPI_COMM_WORLD
            );
        }
        selection_pairs_size = proc_num_individs[0].size();
        selection_pairs = new int[selection_pairs_size];
        for (int i = 0; i < selection_pairs_size; i++) {
            selection_pairs[i] = proc_num_individs[0][i];
        }
        delete[] proc_num_individs;
        delete[] global_fitnesses;
    } else {
        MPI_Status status;
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &selection_pairs_size);
        selection_pairs = new int[selection_pairs_size];
        MPI_Recv(
                selection_pairs, selection_pairs_size,
                MPI_INT, 0, 0, MPI_COMM_WORLD, &status
        );
    }

    int thread_num = omp_get_thread_num();
    for (int i = 0; i < selection_pairs_size / 2; i++) {
        if (individ_proc_num[selection_pairs[2 * i]] == mpi_rank
            && individ_proc_num[selection_pairs[2 * i + 1]] != mpi_rank) {
            int my_individ = selection_pairs[2 * i];
            int other_individ = selection_pairs[2 * i + 1];

            MPI_Request request;
            MPI_Isend(
                    population[my_individ - min_individ_num].getIndivid(), individ_size,
                    MPI_UNSIGNED_CHAR, individ_proc_num[other_individ], my_individ,
                    MPI_COMM_WORLD, &request
            );
            MPI_Request_free(&request);
        }
    }
    for (int i = 0; i < selection_pairs_size / 2; i++) {
        if (individ_proc_num[selection_pairs[2 * i + 1]] == mpi_rank) {
            if (individ_proc_num[selection_pairs[2 * i]] == mpi_rank) {
                population[selection_pairs[2 * i + 1] - min_individ_num]
                        .copyIndivid(population[selection_pairs[2 * i] - min_individ_num].getIndivid());
            } else {

                int my_individ = selection_pairs[2 * i + 1];
                int other_individ = selection_pairs[2 * i];
                char* temp = new char[individ_size];
                MPI_Status status;
                MPI_Recv(
                        temp, individ_size, MPI_UNSIGNED_CHAR,
                        individ_proc_num[other_individ], other_individ,
                        MPI_COMM_WORLD, &status
                );
                population[my_individ - min_individ_num].setIndivid(temp);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void ClassicGenAlg::makeMutation() {
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < local_pop_size; i++) {
            population[i].mutate(mutation_probability, rands[thread_num], dists[thread_num]);
        }
    }
}

void ClassicGenAlg::makeCrossover() {
    int* crossover_pairs;
    int crossover_pairs_size;
    if (mpi_rank == 0) {
        std::vector<int> individs_perm;
        for (int i = 0; i < global_pop_size; i++) {
            individs_perm.push_back(i);
        }
        std::shuffle(individs_perm.begin(), individs_perm.end(), rands[0]);
        if (individs_perm.size() % 2 == 1) {
            individs_perm.pop_back();
        }

        std::vector<int>* proc_num_individs = new std::vector<int>[mpi_size];
        for (int i = 0; i < individs_perm.size(); i += 2) {
            if (dists[0](rands[0]) < crossover_probability) {
                int crossover_pair[2] = {individs_perm[i], individs_perm[i + 1]};
                int dist1 = individ_proc_num[crossover_pair[0]];
                int dist2 = individ_proc_num[crossover_pair[1]];
                if (dist1 != dist2) {
                    proc_num_individs[dist1].push_back(crossover_pair[0]);
                    proc_num_individs[dist1].push_back(crossover_pair[1]);
                    proc_num_individs[dist2].push_back(crossover_pair[1]);
                    proc_num_individs[dist2].push_back(crossover_pair[0]);
                } else {
                    if (crossover_pair[0] > crossover_pair[1]) {
                        std::swap(crossover_pair[0], crossover_pair[1]);
                    }
                    proc_num_individs[dist1].push_back(crossover_pair[0]);
                    proc_num_individs[dist1].push_back(crossover_pair[1]);
                }
            }
        }

        for (int i = 1; i < mpi_size; i++) {
            MPI_Send(
                    proc_num_individs[i].data(), proc_num_individs[i].size(),
                    MPI_INT, i, 0, MPI_COMM_WORLD
            );
        }
        crossover_pairs_size = proc_num_individs[0].size();
        crossover_pairs = new int[crossover_pairs_size];
        for (int i = 0; i < crossover_pairs_size; i++) {
            crossover_pairs[i] = proc_num_individs[0][i];
        }
        delete[] proc_num_individs;
    } else {
        MPI_Status status;
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_INT, &crossover_pairs_size);
        crossover_pairs = new int[crossover_pairs_size];
        MPI_Recv(
                crossover_pairs, crossover_pairs_size,
                MPI_INT, 0, 0, MPI_COMM_WORLD, &status
        );
    }

    typedef struct {
        bool same_owner = false;
        char* mask;
        int crossover_part_size;
        char* crossover_part;
    } CrossoverInfo;

    CrossoverInfo* info = new CrossoverInfo[crossover_pairs_size / 2];

    int thread_num = omp_get_thread_num();

    for (int i = 0; i < crossover_pairs_size / 2; i++) {
        int my_individ = crossover_pairs[2 * i];
        int other_individ = crossover_pairs[2 * i + 1];

        info[i].mask = new char[individ_size];

        if (my_individ < other_individ) {
            info[i].crossover_part_size = 0;
            double prob = dists[thread_num](rands[thread_num]) / 2;
            for (int j = 0; j < individ_size; j++) {
                if (dists[thread_num](rands[thread_num]) < prob) {
                    info[i].mask[j] = 1;
                    info[i].crossover_part_size += 1;
                } else {
                    info[i].mask[j] = 0;
                }
            }

            info[i].crossover_part = population[my_individ - min_individ_num]
                    .getPartByMask(info[i].mask, info[i].crossover_part_size);

            if (individ_proc_num[my_individ] == individ_proc_num[other_individ]) {
                info[i].same_owner = true;
                char* temp = population[other_individ - min_individ_num]
                        .getPartByMask(info[i].mask, info[i].crossover_part_size);
                population[my_individ - min_individ_num].setPartByMask(info[i].mask, temp);
                population[other_individ - min_individ_num].setPartByMask(info[i].mask, info[i].crossover_part);
                delete[] temp;
                delete[] info[i].mask;
                delete[] info[i].crossover_part;
                continue;
            }

            MPI_Request request;
            MPI_Isend(
                    info[i].mask, individ_size, MPI_UNSIGNED_CHAR,
                    individ_proc_num[other_individ], my_individ, MPI_COMM_WORLD, &request
            );
            MPI_Isend(
                    info[i].crossover_part, info[i].crossover_part_size, MPI_UNSIGNED_CHAR,
                    individ_proc_num[other_individ], max_pop_size + my_individ, MPI_COMM_WORLD, &request
            );
        }
    }

    for (int i = 0; i < crossover_pairs_size / 2; i++) {
        if (info[i].same_owner) { continue; }
        int my_individ = crossover_pairs[2 * i];
        int other_individ = crossover_pairs[2 * i + 1];
        if (my_individ > other_individ) {
            MPI_Status status;
            MPI_Recv(
                    info[i].mask, individ_size, MPI_UNSIGNED_CHAR,
                    individ_proc_num[other_individ], other_individ, MPI_COMM_WORLD, &status
            );
            MPI_Probe(individ_proc_num[other_individ], max_pop_size + other_individ, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &info[i].crossover_part_size);
            info[i].crossover_part = new char[info[i].crossover_part_size];
            MPI_Recv(
                    info[i].crossover_part, info[i].crossover_part_size, MPI_UNSIGNED_CHAR,
                    individ_proc_num[other_individ], max_pop_size + other_individ, MPI_COMM_WORLD, &status
            );
            char* temp = population[my_individ - min_individ_num]
                    .getPartByMask(info[i].mask, info[i].crossover_part_size);
            population[my_individ - min_individ_num].setPartByMask(info[i].mask, info[i].crossover_part);
            delete[] info[i].crossover_part;
            info[i].crossover_part = temp;
            MPI_Request request;
            MPI_Isend(
                    info[i].crossover_part, info[i].crossover_part_size, MPI_UNSIGNED_CHAR,
                    individ_proc_num[other_individ], max_pop_size + my_individ, MPI_COMM_WORLD, &request
            );
        }
    }

    for (int i = 0; i < crossover_pairs_size / 2; i++) {
        if (info[i].same_owner) { continue; }
        int my_individ = crossover_pairs[2 * i];
        int other_individ = crossover_pairs[2 * i + 1];
        if (my_individ < other_individ) {
            MPI_Status status;
            MPI_Recv(
                    info[i].crossover_part, info[i].crossover_part_size, MPI_UNSIGNED_CHAR,
                    individ_proc_num[other_individ], max_pop_size + other_individ, MPI_COMM_WORLD, &status
            );
            population[my_individ - min_individ_num].setPartByMask(info[i].mask, info[i].crossover_part);
        }
    }

    for (int i = 0; i < crossover_pairs_size / 2; i++) {
        if (!info[i].same_owner) {
            delete[] info[i].mask;
            delete[] info[i].crossover_part;
        }
    }

    delete[] info;
    delete[] crossover_pairs;
}

void ClassicGenAlg::printPopulation() {
    if (mpi_rank == 0) {
        std::vector<ClassicIndivid> all_population;
        for (int i = 0; i < local_pop_size; i++) {
            all_population.push_back(population[i]);
        }

        MPI_Status status;
        for (int i = 1; i < mpi_size; i++) {
            for (int j = 0; j < proc_pop_size[i]; j++) {
                char* temp_individ = new char[individ_size];
                double temp_fitness;
                MPI_Recv(
                        temp_individ, individ_size, MPI_UNSIGNED_CHAR,
                        i, 0, MPI_COMM_WORLD, &status
                );
                MPI_Recv(
                        &temp_fitness, 1, MPI_DOUBLE,
                        i, 0, MPI_COMM_WORLD, &status
                );

                all_population.push_back(
                        ClassicIndivid(
                                individ_size,
                                temp_individ,
                                temp_fitness
                        )
                );
            }
        }

        for (int i = 0; i < all_population.size(); i++) {
            all_population[i].print();
        }
        std::cout << "#####" << std::endl;
        for (int i = 0; i < individ_size; i++) {
            std::cout << int(best_individ[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << best_fitness;
        std::cout << std::endl << std::endl;
    } else {
        for (int i = 0; i < local_pop_size; i++) {
            MPI_Send(
                    population[i].getIndivid(), individ_size,
                    MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD
            );
            double temp_fitness = population[i].getFitness();
            MPI_Send(
                    &temp_fitness, 1,
                    MPI_DOUBLE, 0, 0, MPI_COMM_WORLD
            );
        }
    }
}

void ClassicGenAlg::printParams() {
    for (int i = 0; i < mpi_size; i++) {
        if (mpi_rank == i) {
            std::cout << "Process: " << i << '\n';
            std::cout << "global_pop_size: " << global_pop_size << '\n';
            std::cout << "local_pop_size: " << local_pop_size << '\n';
            std::cout << "individ_size: " << individ_size << '\n';
            std::cout << "mutation_probability: " << mutation_probability << '\n';
            std::cout << "crossover_probability: " << crossover_probability << '\n';
            std::cout << "mpi_size: " << mpi_size << '\n';
            std::cout << "mpi_rank: " << mpi_rank << '\n';
            std::cout << "omp_size: " << omp_size << '\n';
            std::cout << "individ_proc_num:\n";
            for (auto e: individ_proc_num) {
                std::cout << e.first << " --- " << e.second << '\n';
            }
            std::cout << "proc_pop_size: ";
            for (int i = 0; i < proc_pop_size.size(); i++) {
                std::cout << proc_pop_size[i] << ' ';
            }
            std::cout << '\n';
            std::cout << "displs: ";
            for (int i = 0; i < displs.size(); i++) {
                std::cout << displs[i] << ' ';
            }
            std::cout << '\n';
            std::cout << "min_individ_num: " << min_individ_num << "\n\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}