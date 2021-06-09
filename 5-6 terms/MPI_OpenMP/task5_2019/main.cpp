#include <iostream>
#include "q_gates.h"
#include <mpich/mpi.h>
#include <omp.h>
#include <random>
#include <string.h>

using namespace std;
using complexd = complex<double>;

const double pi = 3.1415926535;

int main(int argc, char* argv[]) {
    int num_threads = atoi(argv[1]);
    omp_set_num_threads(num_threads);
    int n = atoi(argv[2]);
    int vec_size = 1 << n;
    mt19937_64 rnd;
    uniform_int_distribution<> uid(-10, 10);

    int rank = 0, size = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int cur_vec_size = vec_size / size;
    complexd* a = new complexd[cur_vec_size];
    complexd* temp = nullptr;

    if (!strcmp(argv[3], "random")) {
        #pragma omp parallel private(rnd)
        {
            int thread_num = omp_get_thread_num();
            rnd.seed((rank * 100 + thread_num + 1) * time(0));

            #pragma omp for
            for (int i = 0; i < cur_vec_size; i++) {
                a[i] = complexd(uid(rnd), uid(rnd));
            }

            // #pragma omp for
            // for (int i = 0; i < cur_vec_size; i++) {
            //     a[i] = complexd(rank * cur_vec_size + i, rank * cur_vec_size + i);
            // }

            double part_len = 0;
            #pragma omp parallel for reduction (+:part_len)
            for (int i = 0; i < cur_vec_size; i++) {
                part_len += a[i].real() * a[i].real() + a[i].imag() * a[i].imag();
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double len = 0;
            MPI_Allreduce(&part_len, &len, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            len = sqrt(len);
            #pragma omp parallel for
            for (int i = 0; i < cur_vec_size; i++) {
                a[i] /= len;
            }
        }
    } else {
        MPI_Status status;
        MPI_File input_file;
        MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
        MPI_Offset disp = sizeof(int) + cur_vec_size * rank * sizeof(complexd);
        MPI_File_set_view(input_file, disp, MPI_CXX_DOUBLE_COMPLEX,
                            MPI_CXX_DOUBLE_COMPLEX, "native", MPI_INFO_NULL);
        MPI_File_read(input_file, a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, &status);
        MPI_File_close(&input_file);
    }

    // for (int i = 0; i < size; i++) {
    //     if (i == rank) {
    //         cout << "rank = " << rank << endl;
    //         for (int j = 0; j < cur_vec_size; j++) {
    //             cout << a[j] << " ";
    //         }
    //         cout << "\n\n";
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    double time = MPI_Wtime();
    for (int i = 0; i < n - 1; i++) {
        temp = a;
        a = adamar_gate(a, n, i);
        // MPI_Barrier(MPI_COMM_WORLD);
        delete[] temp;
        for (int j = 1; j < n - i; j++) {
            temp = a;
            a = c_rw_gate(a, n, i, i + j,
                        complexd(pi / pow(2, j)));
            // MPI_Barrier(MPI_COMM_WORLD);
            delete[] temp;
        }
    }
    temp = a;
    a = adamar_gate(a, n, n - 1);
    delete[] temp;

    time = MPI_Wtime() - time;

    for (int i = 0; i < n / 2; i++) {
        temp = a;
        a = swap_gate(a, n, i, n - i - 1);
        delete[] temp;
    }

    if (!rank) {
        cout << "time = " << time << endl;
    }

    for (int i = 0; i < size; i++) {
        if (i == rank) {
            cout << "rank = " << rank << endl;
            for (int j = 0; j < cur_vec_size; j++) {
                cout << a[j] << " ";
            }
            cout << "\n\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (strcmp(argv[4], "null")) {
        MPI_Status status;
        MPI_File output_file;
        MPI_File_open(MPI_COMM_WORLD, argv[4],
                MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &output_file);
        MPI_Offset disp = sizeof(int) + cur_vec_size * rank * sizeof(complexd);
        if (!rank) {
            MPI_File_write(output_file, &vec_size, 1, MPI_INT, &status);
        }
        MPI_File_set_view(output_file, disp, MPI_CXX_DOUBLE_COMPLEX,
                MPI_CXX_DOUBLE_COMPLEX, "native", MPI_INFO_NULL);
        MPI_File_write(output_file, a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, &status);
        MPI_File_close(&output_file);
    }

    delete[] a;

    MPI_Finalize();
    return 0;
}
