#ifndef REFLECTOR_H
#define REFLECTOR_H

#include <iostream>
#include <math.h>
#include <mpi.h>
#include <iomanip>
#include <limits>
#include <functional>

class Reflector {
private:
    int n;
    int local_size;

    double** matrix;
    double* x;
    double* b;

    int mpi_size;
    int mpi_rank;
    bool last_process;

    double t1, t2;

    bool needProcessColumn(int i);
    double* conctructXiUpdateAi(int i);
    void updateRestMatrix(const double* x, int i);
    void countX();

    inline bool myColumn(int i) { return ((i - mpi_rank) % mpi_size) == 0; }
    inline int glob2loc(int i) {
        if (myColumn(i)) {
            return (i - mpi_rank) / mpi_size;
        } else {
            return -1;
        }
    }
    inline int loc2glob(int i) { return i * mpi_size + mpi_rank; }
    inline int glob2procNum(int i) { return i % mpi_size; }

    double init_matr_func(int i, int j) {
        double res;
        res = 1.0 / (loc2glob(i) + j + 1);
//        if (loc2glob(i) == j) {
//            res += 2;
//        }
        return res;
    }

    double init_b_func(int i) {
        double res;
        res = i + 1;
        return res;
    }
public:
    Reflector(int n, int mpi_rank, int mpi_size);

    void start();

    void printMatrix();
    void printX();
    void printResidual();
    void printDiff(const double* right_x);
    void printTimes() {
        if (mpi_rank == 0) {
            std::cout << "T1: " << t1 << "\tT2: "
                << t2 << "\tT_all: " << t1 + t2 << std::endl;
        }
    }

    ~Reflector();
};


#endif //REFLECTOR_H
