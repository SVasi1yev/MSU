#include <vector>
#include <random>
#include <iostream>
#include <functional>

namespace ClassicGen {

    class ClassicIndivid {
    private:
        int individ_size;
        char* individ;

        double fitness;
    public:
        ClassicIndivid(int individ_size);
        ClassicIndivid(const ClassicIndivid& other);
        ClassicIndivid(int individ_size, char* individ, double fitness);
        ClassicIndivid& operator=(const ClassicIndivid& other);

        void uniformInit(
                std::mt19937_64& rand,
                std::uniform_real_distribution<>& dist
        );

        void countFitness(std::function<double(const char*, int)> fitness_function);

        void mutate(
                double mutation_probability,
                std::mt19937_64& rand,
                std::uniform_real_distribution<>& dist
        );

        char* getPartByMask(const char* mask, int part_size);

        void setPartByMask(const char* mask, const char* part);

        inline char* getIndivid() { return individ; }
        inline double getFitness() { return fitness; }

        void setIndivid(char* new_individ);
        void copyIndivid(char* new_individ);

        void print();

        ~ClassicIndivid() { delete[] individ; }
    };

}

using namespace ClassicGen;

ClassicIndivid::ClassicIndivid(int individ_size):
        individ_size(individ_size),
        fitness(-std::numeric_limits<double>::infinity())
{
    individ = new char[individ_size];
}

ClassicIndivid::ClassicIndivid(const ClassicIndivid& other):
        individ_size(other.individ_size),
        fitness(other.fitness)
{
    individ = new char[individ_size];
    for (int i = 0; i < individ_size; i++) {
        individ[i] = other.individ[i];
    }
}

ClassicIndivid::ClassicIndivid(int individ_size, char* individ, double fitness):
        individ_size(individ_size),
        individ(individ),
        fitness(fitness) {}

ClassicIndivid& ClassicIndivid::operator=(const ClassicIndivid &other) {
    if (this != &other) {
        delete[] individ;
        individ = new char[individ_size];
        for (int i = 0; i < individ_size; i++) {
            individ[i] = other.individ[i];
        }
    }

    return *this;
}

void ClassicIndivid::uniformInit(
        std::mt19937_64& rand,
        std::uniform_real_distribution<>& dist
)
{
    for (int i = 0; i < individ_size; i++) {
        individ[i] = (dist(rand) < 0.5) ? 0 : 1;
    }
}

void ClassicIndivid::countFitness(std::function<double(const char*, int)> fitness_function) {
    fitness = fitness_function(individ, individ_size);
}

void ClassicIndivid::mutate(
        double mutation_probability,
        std::mt19937_64& rand,
        std::uniform_real_distribution<>& dist
)
{
    for (int i = 0; i < individ_size; i++) {
        if (dist(rand) < mutation_probability) {
            individ[i] = (individ[i] + 1) % 2;
        }
    }
}

char* ClassicIndivid::getPartByMask(const char *mask, int part_size) {
    char* part= new char[part_size];
    for (int i = 0, k = 0; i < individ_size; i++) {
        if (mask[i] == 1) {
            part[k++] = individ[i];
        }
    }
    return part;
}

void ClassicIndivid::setPartByMask(const char *mask, const char *part) {
    for (int i = 0, k = 0; i < individ_size; i++) {
        if (mask[i] == 1) {
            individ[i] = part[k++];
        }
    }
}

void ClassicIndivid::setIndivid(char* new_individ) {
    delete[] individ;
    individ = new_individ;
}

void ClassicIndivid::copyIndivid(char* new_individ) {
    for (int i = 0; i < individ_size; i++) {
        individ[i] = new_individ[i];
    }
}

void ClassicIndivid::print() {
    for (int i = 0; i < individ_size; i++) {
        std::cout << int(individ[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << fitness;
    std::cout << std::endl;
}
