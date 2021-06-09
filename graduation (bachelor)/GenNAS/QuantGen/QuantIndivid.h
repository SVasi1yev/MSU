#ifndef QUANTINDIVID_H
#define QUANTINDIVID_H

#include "QuantBit.h"

#include <vector>
#include <random>
#include <functional>
#include <memory>
#include <iostream>

namespace QuantGen {

    class QuantIndivid {
    private:
        int individ_size;
        QuantBit *individ;

        double fitness;
        char *observation;
    public:
        QuantIndivid(int individ_size);

        QuantIndivid(const QuantIndivid &other);

        QuantIndivid(int individ_size, QuantBit *individ, char *observation, double fitness);

        QuantIndivid &operator=(const QuantIndivid &other);

        void uniformInit();

        void observe(
                std::function<double(const char *, int)> fitness_func,
                std::mt19937_64 &rand,
                std::uniform_real_distribution<> &dist
        );

        void rotate(
                double best_fitness,
                const char *best_observation,
                std::function<double(char, char, double, double, QuantBit)> angle_function
        );

        void mutate(
                double mutation_probability,
                std::mt19937_64 &rand,
                std::uniform_real_distribution<> &dist
        );

        double *getPartByMask(const char *mask, int part_size);

        void setPartByMask(const char *mask, const double *part);

        inline QuantBit *getIndivid() const { return individ; }

        inline char *getObservation() const { return observation; }

        inline double getFitness() { return fitness; }

        void print();

        ~QuantIndivid() {
            delete[] individ;
            delete[] observation;
        }
    };

}

#endif //QUANTINDIVID_H
