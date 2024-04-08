#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <math.h>
#include <cmath>

double target_func(double x, double y, double z) {
    double res;
    if ((x + y + z) > 1) {
        res = 0;
    } else {
        res = 1 / pow((1 + x + y + z), 3);
    }
    return res;
}

double* gen_points(
    double x1, double x2, 
    double y1, double y2,
    double z1, double z2,
    int n 
) {
    double* points = new double[3 * n];
    for (int i = 0; i < n; i++) {
        points[3 * i] = (double)rand() / RAND_MAX;
        points[3 * i + 1] = (double)rand() / RAND_MAX;
        points[3 * i + 2] = (double)rand() / RAND_MAX;
    }

    return points;
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    int points_num = 1000;
    double* points;
    double* all_points;
    double true_value = 0.03407359;
    double error = std::atof(argv[1]);
    double cur_error = error * 1000;
    double part = 0;
    double sum = 0;
    double res = 0;
    int total_points = 0;
    int sendcounts[size];
    int displs[size];

    sendcounts[0] = 0;
    displs[0] = 0;
    int r = points_num % (size - 1);
    int q = 0;
    for (int i = 1; i < size; i++) {
        if ((i - 1) < r) {
            sendcounts[i] = ((points_num / (size - 1)) + 1) * 3;
            displs[i] = q;
            q += sendcounts[i];
        } else {
            sendcounts[i] = ((points_num / (size - 1))) * 3;
            displs[i] = q;
            q += sendcounts[i];
        }
    }

    double start = MPI_Wtime();
    if (rank == 0) {
        srand(0);
        points = new double[1];

        all_points = gen_points(
            0, 1, 0, 1, 0, 1, points_num
        );
        int k = 0;
        while (cur_error > error) {
            MPI_Scatterv(
                all_points, sendcounts, displs, 
                MPI_DOUBLE , points , sendcounts[rank], 
                MPI_DOUBLE , 0 , MPI_COMM_WORLD
            );
            total_points += points_num;
            delete[] all_points;
            all_points = gen_points(
                0, 1, 0, 1, 0, 1, points_num
            );
            part = 0;
            sum = 0;
            MPI_Allreduce(&part, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            res += sum;
            cur_error = std::abs(true_value - res / total_points);
        }

        delete[] all_points;
    } else {
        all_points = new double[1];
        points = new double[sendcounts[rank]];

        while (cur_error > error) {
            MPI_Scatterv(
                all_points, sendcounts, displs, 
                MPI_DOUBLE , points , sendcounts[rank], 
                MPI_DOUBLE , 0 , MPI_COMM_WORLD
            );
            total_points += points_num;
            part = 0;
            sum = 0;
            for (int i = 0; i < sendcounts[rank] / 3; i++) {
                part += target_func(
                    points[3 * i], 
                    points[3 * i + 1],
                    points[3 * i + 2]
                );
            }
            MPI_Allreduce(&part, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            res += sum;
            cur_error = std::abs(true_value - res / total_points);
        }

        delete[] points;
    }

    double part_total_time = MPI_Wtime() - start;
    double total_time = 0;
    MPI_Reduce(
        &part_total_time , &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        res /= total_points;
        std::cout << "Result: " << res 
                << "; Error: " << std::abs(res - true_value) 
                << "; Points number: " << total_points
                << "; Time: " << total_time
                << std::endl;
    }

    MPI_Finalize();

    return 0;
}
