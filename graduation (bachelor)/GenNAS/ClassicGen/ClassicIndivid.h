#ifndef CLASSICINDIVID_H
#define CLASSICINDIVID_H

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

#endif //CLASSICINDIVID_H
