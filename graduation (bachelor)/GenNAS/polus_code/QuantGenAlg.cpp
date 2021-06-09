#include "QuantIndivid.cpp"

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

using namespace QuantGen;

double QuantGen::angle_function(char x, char b, double fitness_x, double fitness_b, QuantBit qbit) {
    if (x == b) {
        return 0;
    }
    double sign = 0;
    if (x == 0) {
        if (fitness_x < fitness_b) {
            if (qbit.getAlpha() * qbit.getBeta() > 0) {
                sign = 1;
            } else if (qbit.getAlpha() * qbit.getBeta() < 0) {
                sign = -1;
            } else if (qbit.getAlpha() < 0.00001) {
                return 0;
            } else {
                sign = 1;
            }
        } else {
            if (qbit.getAlpha() * qbit.getBeta() > 0) {
                sign = -1;
            } else if (qbit.getAlpha() * qbit.getBeta() < 0) {
                sign = 1;
            } else if (qbit.getAlpha() < 0.00001) {
                return 1;
            } else {
                return 0;
            }
        }
    } else {
        if (fitness_x < fitness_b) {
            if (qbit.getAlpha() * qbit.getBeta() > 0) {
                sign = -1;
            } else if (qbit.getAlpha() * qbit.getBeta() < 0) {
                sign = 1;
            } else if (qbit.getAlpha() < 0.00001) {
                return 1;
            } else {
                return 0;
            }
        } else {
            if (qbit.getAlpha() * qbit.getBeta() > 0) {
                sign = 1;
            } else if (qbit.getAlpha() * qbit.getBeta() < 0) {
                sign = -1;
            } else if (qbit.getAlpha() < 0.00001) {
                return 0;
            } else {
                sign = 1;
            }
        }
    }

    double delta = 0.005 * PI
                   + (0.05 * PI - 0.005 * PI)
                     * std::abs(fitness_x - fitness_b) / std::max(fitness_x, fitness_b);

    return sign * delta;
}

QuantGenAlg::QuantGenAlg(
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
):
        global_pop_size(global_pop_size),
        individ_size(individ_size),
        need_mutation(need_mutation),
        mutation_probability(mutation_probability),
        need_crossover(need_crossover),
        crossover_probability(crossover_probability),
        fitness_function(fitness_function),
        angle_function(angle_function),
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
        population.push_back(QuantIndivid(individ_size));
    }

    best_observation = new char[individ_size];

    omp_set_num_threads(omp_size);
    omp_set_dynamic(0);
    omp_set_schedule(omp_sched_dynamic, 1);
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < local_pop_size; i++) {
            population[i].uniformInit();
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

void QuantGenAlg::startAlgorithm(int iter_num) {
    start_time = MPI_Wtime();
    makeObservations();
    printPopulation();
    for (int i = 0; i < iter_num; i++) {
        makeRotation();
        if (need_mutation) {
            makeMutation();
        }
        if (need_crossover) {
            makeCrossover();
        }
        makeObservations();
        printPopulation();
    }
}

void QuantGenAlg::makeObservations() {
    double cur_best_fitness = best_fitness;
    int cur_best_observation_ind = -1;
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < local_pop_size; i++) {
            population[i].observe(fitness_function, rands[thread_num], dists[thread_num]);
#pragma omp critical
            {
                if (population[i].getFitness() > cur_best_fitness) {
                    cur_best_fitness = population[i].getFitness();
                    cur_best_observation_ind = i;
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
                    population[cur_best_observation_ind].getObservation(),
                    population[cur_best_observation_ind].getObservation() + individ_size,
                    best_observation
            );
            MPI_Bcast(
                    best_observation, individ_size,
                    MPI_UNSIGNED_CHAR, global.rank, MPI_COMM_WORLD
            );
        }
    } else {
        if (global.fitness > best_fitness) {
            best_fitness = global.fitness;
            MPI_Bcast(
                    best_observation, individ_size,
                    MPI_UNSIGNED_CHAR, global.rank, MPI_COMM_WORLD
            );
        }
    }
}

void QuantGenAlg::makeRotation() {
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < local_pop_size; i++) {
            population[i].rotate(best_fitness, best_observation, angle_function);
        }
    }
}

void QuantGenAlg::makeMutation() {
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < local_pop_size; i++) {
            population[i].mutate(mutation_probability, rands[thread_num], dists[thread_num]);
        }
    }
}

void QuantGenAlg::makeCrossover() {
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
        double* crossover_part;
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
                    info[i].crossover_part_size += 2;
                } else {
                    info[i].mask[j] = 0;
                }
            }

            info[i].crossover_part = population[my_individ - min_individ_num]
                    .getPartByMask(info[i].mask, info[i].crossover_part_size);

            if (individ_proc_num[my_individ] == individ_proc_num[other_individ]) {
                info[i].same_owner = true;
                double *temp = population[other_individ - min_individ_num]
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
                    info[i].crossover_part, info[i].crossover_part_size, MPI_DOUBLE,
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
            MPI_Get_count(&status, MPI_DOUBLE, &info[i].crossover_part_size);
            info[i].crossover_part = new double[info[i].crossover_part_size];
            MPI_Recv(
                    info[i].crossover_part, info[i].crossover_part_size, MPI_DOUBLE,
                    individ_proc_num[other_individ], max_pop_size + other_individ, MPI_COMM_WORLD, &status
            );
            double* temp = population[my_individ - min_individ_num]
                    .getPartByMask(info[i].mask, info[i].crossover_part_size);
            population[my_individ - min_individ_num].setPartByMask(info[i].mask, info[i].crossover_part);
            delete[] info[i].crossover_part;
            info[i].crossover_part = temp;
            MPI_Request request;
            MPI_Isend(
                    info[i].crossover_part, info[i].crossover_part_size, MPI_DOUBLE,
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
                    info[i].crossover_part, info[i].crossover_part_size, MPI_DOUBLE,
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

void QuantGenAlg::printPopulation() {
    if (mpi_rank == 0) {
        std::vector<QuantIndivid> all_population;
        for (int i = 0; i < local_pop_size; i++) {
            all_population.push_back(population[i]);
        }

        MPI_Status status;
        double* temp_double_individ = new double[2 * individ_size];
        for (int i = 1; i < mpi_size; i++) {
            for (int j = 0; j < proc_pop_size[i]; j++) {
                char* temp_observation = new char[individ_size];
                double temp_fitness;
                MPI_Recv(
                        temp_double_individ, 2 * individ_size, MPI_DOUBLE,
                        i, 0, MPI_COMM_WORLD, &status
                );
                MPI_Recv(
                        temp_observation, individ_size, MPI_UNSIGNED_CHAR,
                        i, 0, MPI_COMM_WORLD, &status
                );
                MPI_Recv(
                        &temp_fitness, 1, MPI_DOUBLE,
                        i, 0, MPI_COMM_WORLD, &status
                );

                QuantBit* temp_individ = new QuantBit[individ_size];
                for (int k = 0; k < individ_size; k++) {
                    temp_individ[k] = QuantBit(
                            temp_double_individ[2 * k],
                            temp_double_individ[2 * k + 1]
                    );
                }

                all_population.push_back(
                        QuantIndivid(
                                individ_size,
                                temp_individ,
                                temp_observation,
                                temp_fitness
                        )
                );
            }
        }
        delete[] temp_double_individ;

        for (int i = 0; i < all_population.size(); i++) {
            all_population[i].print();
        }
        std::cout << "#####" << std::endl;
        for (int i = 0; i < individ_size; i++) {
            std::cout << int(best_observation[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << best_fitness << std::endl;
        std::cout << "time: " << MPI_Wtime() - start_time;
        std::cout << std::endl << std::endl;
    } else {
        double* temp_double_individ = new double[2 * individ_size];
        for (int i = 0; i < local_pop_size; i++) {
            QuantBit* temp_individ = population[i].getIndivid();
            for (int j = 0; j < individ_size; j++) {
                temp_double_individ[2 * j] = temp_individ[j].getAlpha();
                temp_double_individ[2 * j + 1] = temp_individ[j].getBeta();
            }

            MPI_Send(
                    temp_double_individ, 2 * individ_size,
                    MPI_DOUBLE, 0, 0, MPI_COMM_WORLD
            );
            MPI_Send(
                    population[i].getObservation(), individ_size,
                    MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD
            );
            double temp_fitness = population[i].getFitness();
            MPI_Send(
                    &temp_fitness, 1,
                    MPI_DOUBLE, 0, 0, MPI_COMM_WORLD
            );
        }
        delete[] temp_double_individ;
    }
}

void QuantGenAlg::printParams() {
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
