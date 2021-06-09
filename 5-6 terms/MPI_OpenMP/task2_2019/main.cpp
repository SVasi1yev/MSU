#include <string.h>
#include <ctime>
#include <iostream>
#include <complex>
#include <random>
#include <mpi/mpi.h>

using namespace std;
typedef complex<double> complexd;

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
            for (int i = 0; i < cur_vec_size; i++) {
                out[i] = u[1][0] * a_partner[i] + u[1][1] * a[i];
            }
        } else {
            for (int i = 0; i < cur_vec_size; i++) {
                out[i] = u[0][0] * a[i] + u[0][1] * a_partner[i];
            }
        }
    }

    return out;
} 

int main (int argc, char* argv[]) {
    int n = atoi(argv[1]);
    int k = atoi(argv[2]) - 1;
    int vec_size = 1 << n;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int cur_vec_size = vec_size / size;

    mt19937_64 rnd;
    uniform_int_distribution<> uid(-5, 5);
    rnd.seed((rank + 1) * time(0));

    complexd u[2][2] = {
        complexd(1.0 / pow(2, 0.5)),
        complexd(1.0 / pow(2, 0.5)),
        complexd(1.0 / pow(2, 0.5)),
        complexd(-1.0 / pow(2, 0.5))
    };

    complexd *a = new complexd[cur_vec_size];

    if (!strcmp(argv[3], "random")) {
        for (int i = 0; i < cur_vec_size; i++) {
            a[i] = complexd(uid(rnd), uid(rnd));
            // a[i] = complexd(cur_vec_size * rank + i, cur_vec_size * rank + i);
        }
    } else {
        MPI_Status status;
        MPI_File input_file;
        MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
        MPI_Offset disp = sizeof(int) + cur_vec_size * rank * sizeof(complexd);
        MPI_File_set_view(input_file, disp, MPI_CXX_DOUBLE_COMPLEX, MPI_CXX_DOUBLE_COMPLEX, "native", MPI_INFO_NULL);
        MPI_File_read(input_file, a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, &status);
        MPI_File_close(&input_file);
    }

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

    double part_len = 0;
    for (int i = 0; i < cur_vec_size; i++) {
        part_len += a[i].real() * a[i].real() + a[i].imag() * a[i].imag();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double len = 0;
    MPI_Allreduce(&part_len, &len, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    len = sqrt(len);
    if (!rank) {
        cout << "len = " << len << endl;
    }
    for (int i = 0; i < cur_vec_size; i++) {
        a[i] /= len;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();
    complexd *out = single_qubit_transform(a, n, u, k);
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime() - time;

    if (!rank) {
        cout << "time = " << time << endl;
    }

    /*
    part_len = 0;
    for (int i = 0; i < cur_vec_size; i++) {
        part_len = a[i].real() * a[i].real() + a[i].imag() * a[i].imag();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    len = 0;
    MPI_Allreduce(&part_len, &len, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    len = sqrt(len);
    cout << len << endl;
    */
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
    
    // for (int i = 0; i < size; i++) {
    //     if (i == rank) {
    //         cout << "rank = " << rank << endl;
    //         for (int j = 0; j < cur_vec_size; j++) {
    //             cout << out[j] << " ";
    //         }
    //         cout << "\n\n";
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    
    
    if (strcmp(argv[4], "null")) {
        MPI_Status status;
        MPI_File output_file;
        MPI_File_open(MPI_COMM_WORLD, argv[4], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &output_file);
        MPI_Offset disp = sizeof(int) + cur_vec_size * rank * sizeof(complexd);
        if (!rank) {
            MPI_File_write(output_file, &vec_size, 1, MPI_INT, &status);
        }
        MPI_File_set_view(output_file, disp, MPI_CXX_DOUBLE_COMPLEX, MPI_CXX_DOUBLE_COMPLEX, "native", MPI_INFO_NULL);
        MPI_File_write(output_file, out, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, &status);
        MPI_File_close(&output_file);
    }

    delete[] a;
    delete[] out;   

    MPI_Finalize();
    return 0;
}