#include <string.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#include <complex>
#include <random>
#include <mpi/mpi.h>
#include <omp.h>
#include <fstream>

using namespace std;
typedef complex<double> complexd;

double normal_dis_gen() {
    double s = 0;
    for (int i = 0; i < 12; i++) {
        s += (double)rand() / RAND_MAX;
    }
    return s - 6;
}

complexd* single_qubit_transform (complexd *a, int n, complexd u[2][2], int k) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int vec_size = 1 << n;
    int cur_vec_size = vec_size / size;
    MPI_Status status;

    complexd *out = new complexd[cur_vec_size];
    int temp = size;
    int size_power = -1;
    while (temp > 0) {
        temp >>= 1;
        size_power++;
    }

    if (k >= size_power) {
        int shift = n - 1 - k;
        int mask = 1 << shift;
        #pragma omp parallel for
        for (int i = 0; i < cur_vec_size; i++) {
            int i0 = i & ~mask;
            int i1 = i | mask;
            int iq = (i & mask) >> shift;
            out[i] = u[iq][0] * a[i0] + u[iq][1] * a[i1];
        }
    } else {
        int zero_one = rank & (1 << (size_power - k - 1));
        int partner = rank ^ (1 << (size_power - k - 1));
        complexd *a_partner = new complexd[cur_vec_size];
        MPI_Sendrecv(a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner, 0, 
                a_partner, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner, 0, MPI_COMM_WORLD, &status);
        if (zero_one) {
            #pragma omp parallel for
            for (int i = 0; i < cur_vec_size; i++) {
                out[i] = u[1][0] * a_partner[i] + u[1][1] * a[i];
            }
        } else {
            #pragma omp parallel for
            for (int i = 0; i < cur_vec_size; i++) {
                out[i] = u[0][0] * a[i] + u[0][1] * a_partner[i];
            }
        }
        delete[] a_partner;
    }

    return out;
} 

int main (int argc, char* argv[]) {
    int num_threads = atoi(argv[1]);
    omp_set_num_threads(num_threads);
    int n = atoi(argv[2]);
    int vec_size = 1 << n;
    mt19937_64 rnd;
    uniform_int_distribution<> uid(-5, 5);

    complexd u[2][2] = {
        complexd(1.0 / pow(2, 0.5)),
        complexd(1.0 / pow(2, 0.5)),
        complexd(1.0 / pow(2, 0.5)),
        complexd(-1.0 / pow(2, 0.5))
    };

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int cur_vec_size = vec_size / size;
    srand(time(0));
    double e = atof(argv[3]);
    int m = atoi(argv[6]);
    double avg = 0;
    for (int z = 0; z < m; z++)
    {
        complexd *a = new complexd[cur_vec_size];

        if (!strcmp(argv[4], "random")) {
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
                
            }
        } else {
            MPI_Status status;
            MPI_File input_file;
            MPI_File_open(MPI_COMM_WORLD, argv[4], MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
            MPI_Offset disp = sizeof(int) + cur_vec_size * rank * sizeof(complexd);
            MPI_File_set_view(input_file, disp, MPI_CXX_DOUBLE_COMPLEX, MPI_CXX_DOUBLE_COMPLEX, "native", MPI_INFO_NULL);
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
        
        
        double part_len = 0;
        #pragma omp parallel for reduction (+:part_len)
        for (int i = 0; i < cur_vec_size; i++) {
            part_len += a[i].real() * a[i].real() + a[i].imag() * a[i].imag();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double len = 0;
        MPI_Allreduce(&part_len, &len, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        len = sqrt(len);
        // if (!rank) {
        //     cout << "len = " << len << endl;
        // }
        #pragma omp parallel for
        for (int i = 0; i < cur_vec_size; i++) {
            a[i] /= len;
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

        complexd* b = new complexd[cur_vec_size];
        #pragma omp parallel for
        for (int i = 0; i < cur_vec_size; i++) {
            b[i] = a[i];
        }

        // for (int i = 0; i < size; i++) {
        //     if (i == rank) {
        //         cout << "rank = " << rank << endl;
        //         for (int j = 0; j < cur_vec_size; j++) {
        //             cout << b[j] << " ";
        //         }
        //         cout << "\n\n";
        //     }
        //     MPI_Barrier(MPI_COMM_WORLD);
        // }

        MPI_Barrier(MPI_COMM_WORLD);
        double time = MPI_Wtime();
        for (int i = 0; i < n; i++) {
            double* h = new double[4];

            if (!rank) {
                double teta = e * normal_dis_gen();
                h[0] = cos(teta);
                h[1] = sin(teta);
                h[2] = -sin(teta);
                h[3] = cos(teta);
            }

            MPI_Bcast(h, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            complexd ue[2][2] = {
                u[0][0] * h[0] + u[0][1] * h[2],
                u[0][0] * h[1] + u[0][1] * h[3],
                u[1][0] * h[0] + u[1][1] * h[2],
                u[1][0] * h[1] + u[1][1] * h[3]
            };

            complexd* temp = a;
            a = single_qubit_transform(temp, n, ue, i);
            delete[] temp;
            delete[] h;
        }
        time = MPI_Wtime() - time;

        for (int i = 0; i < n; i++) {
            complexd* temp = b;
            b = single_qubit_transform(temp, n, u, i);
            delete[] temp;
        }

        // for (int i = 0; i < size; i++) {
        //     if (i == rank) {
        //         cout << "rank = " << rank << endl;
        //         for (int j = 0; j < cur_vec_size; j++) {
        //             cout << b[j] << " ";
        //         }
        //         cout << "\n\n";
        //     }
        //     MPI_Barrier(MPI_COMM_WORLD);
        // }

        // part_len = 0;
        // #pragma omp parallel for reduction (+:part_len)
        // for (int i = 0; i < cur_vec_size; i++) {
        //     part_len += a[i].real() * a[i].real() + a[i].imag() * a[i].imag();
        // }
        // len = 0;
        // MPI_Allreduce(&part_len, &len, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // len = sqrt(len);
        // if (!rank) {
        //     cout << "len = " << len << endl;
        // }

        // part_len = 0;
        // #pragma omp parallel for reduction (+:part_len)
        // for (int i = 0; i < cur_vec_size; i++) {
        //     part_len += b[i].real() * b[i].real() + b[i].imag() * b[i].imag();
        // }
        // len = 0;
        // MPI_Allreduce(&part_len, &len, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // len = sqrt(len);
        // if (!rank) {
        //     cout << "len = " << len << endl;
        // }

        /*
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
        */

        double real = 0, imag = 0;
        #pragma omp parallel for reduction(+: real, imag)
        for (int i = 0; i < cur_vec_size; i++) {
            real += a[i].real() * b[i].real() + a[i].imag() * b[i].imag();
            imag += a[i].imag() * b[i].real() - a[i].real() * b[i].imag();
        }
        complexd temp = complexd(real, imag);
        complexd temp0;
        MPI_Reduce(&temp, &temp0, 1, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
        if (!rank) {
            //cout << temp0 << endl;
            double fidenlity = temp0.real() * temp0.real() + temp0.imag() * temp0.imag();
            avg += 1 - fidenlity;
            ofstream fout("result.txt", ios_base::app);
            fout << 1.0 - fidenlity << endl;
            fout.close();
        }
        
        if (strcmp(argv[5], "null")) {
            MPI_Status status;
            MPI_File output_file;
            MPI_File_open(MPI_COMM_WORLD, argv[4], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &output_file);
            MPI_Offset disp = sizeof(int) + cur_vec_size * rank * sizeof(complexd);
            if (!rank) {
                MPI_File_write(output_file, &vec_size, 1, MPI_INT, &status);
            }
            MPI_File_set_view(output_file, disp, MPI_CXX_DOUBLE_COMPLEX, MPI_CXX_DOUBLE_COMPLEX, "native", MPI_INFO_NULL);
            MPI_File_write(output_file, a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, &status);
            MPI_File_close(&output_file);
        }

        // if (!rank) {
        //     cout << "time = " << time << endl;
        // }

        delete[] a;
        delete[] b;
        // if (!rank) {
        //     cout << z << endl;
        // }
    }

    cout << "avg = " << avg / m << endl;

    MPI_Finalize();
    return 0;
}
