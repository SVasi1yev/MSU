#include "QuantBit.cpp"

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

using namespace QuantGen;

QuantIndivid::QuantIndivid(int individ_size):
        individ_size(individ_size),
        fitness(-std::numeric_limits<double>::infinity())
{
    individ = new QuantBit[individ_size];
    observation = new char[individ_size];
}

QuantIndivid::QuantIndivid(const QuantIndivid& other):
        individ_size(other.individ_size),
        fitness(other.fitness)
{
    individ = new QuantBit[individ_size];
    observation = new char[individ_size];
    for (int i = 0; i < individ_size; i++) {
        individ[i] = other.getIndivid()[i];
        observation[i] = other.getObservation()[i];
    }
}

QuantIndivid::QuantIndivid(int individ_size, QuantBit* individ, char* observation, double fitness):
        individ_size(individ_size),
        individ(individ),
        observation(observation),
        fitness(fitness) {}

QuantIndivid& QuantIndivid::operator=(const QuantIndivid &other) {
    if (this != &other) {
        delete[] individ;
        delete[] observation;
        individ = new QuantBit[individ_size];
        observation = new char[individ_size];
        for (int i = 0; i < individ_size; i++) {
            individ[i] = other.getIndivid()[i];
            observation[i] = other.getObservation()[i];
        }
    }

    return *this;
}

void QuantIndivid::uniformInit() {
    for (int i = 0; i < individ_size; i++) {
        individ[i] = QuantBit(1/sqrt(2), 1/sqrt(2));
    }
}

void QuantIndivid::observe(
        std::function<double(const char*, int)> fitness_func,
        std::mt19937_64& rand,
        std::uniform_real_distribution<>& dist
)
{
    for (int i = 0; i < individ_size; i++) {
        observation[i] = (dist(rand) < (individ[i].getAlpha() * individ[i].getAlpha()) ? 0 : 1);
    }
    fitness = fitness_func(observation, individ_size);
}

void QuantIndivid::rotate(
        double best_fitness,
        const char* best_observation,
        std::function<double(char, char, double, double, QuantBit)> angle_function
)
{
    for (int i = 0; i < individ_size; i++) {
        double angle = angle_function(
                observation[i], best_observation[i],
                fitness, best_fitness,
                individ[i]
        );
        individ[i].rotate(angle);
    }
}

void QuantIndivid::mutate(
        double mutation_probability,
        std::mt19937_64& rand,
        std::uniform_real_distribution<>& dist
)
{
    for (int i = 0; i < individ_size; i++) {
        if (dist(rand) < mutation_probability) {
            individ[i].inverse();
        }
    }
}

double* QuantIndivid::getPartByMask(const char* mask, int part_size) {
    double* part = new double[part_size];
    for (int i = 0, k = 0; i < individ_size; i++) {
        if (mask[i] == 1) {
            part[k++] = individ[i].getAlpha();
            part[k++] = individ[i].getBeta();
        }
    }
    return part;
}

void QuantIndivid::setPartByMask(const char* mask, const double* part) {
    for (int i = 0, k = 0; i < individ_size; i++) {
        if (mask[i] == 1) {
            individ[i].setAmplitudes(part[k], part[k + 1]);
            k += 2;
        }
    }
}

void QuantIndivid::print() {
    for (int i = 0; i < individ_size; i++) {
        std::cout << "(" << individ[i].getAlpha() << ", "
                  << individ[i].getBeta() << ") ";
    }
    std::cout << std::endl;
    for (int i = 0; i < individ_size; i++) {
        std::cout << int(observation[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << fitness;
    std::cout << std::endl;
}
