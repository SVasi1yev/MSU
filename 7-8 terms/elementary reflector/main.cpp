#include "Reflector.h"

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = std::atoi(argv[1]);

    // Reflector(int n, int mpi_rank, int mpi_size)
    Reflector reflector = Reflector(n, rank, size);
//    reflector.printMatrix();
    reflector.start();
//    reflector.printMatrix();
    reflector.printX();
    reflector.printResidual();

//    double right_x[4] = {-0.75, -0.25, 0.25, 0.75};
//    int local_size = 0;
//    if (rank < (n % size)) {
//        local_size = n / size + 1;
//    } else {
//        local_size = n / size;
//    }
//
//    double part_right_x[local_size];
//    for (int i = 0; i < local_size; i++) {
//        part_right_x[i] = right_x[i * size + rank];
//    }
//
//    reflector.printDiff(part_right_x);

    reflector.printTimes();

    MPI_Finalize();
}