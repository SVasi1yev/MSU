#include "q_gates.h"
#include <complex>
#include <mpi/mpi.h>
#include <iostream>
#include <random>
#include <fstream>
#include <omp.h>

int main(int argc, char *argv[]) {
    int n = 5;
    int k = 0;
    int l = 1;
    complexd phi(3.14 / 4, 0.0);
    int vec_size = 1 << n;
    mt19937_64 rnd;
    rnd.seed(0);
    uniform_int_distribution<> uid(-5, 5);

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int cur_vec_size = vec_size / size;

    complexd* a_full = new complexd[vec_size];
    if (!rank) {
        for (int i = 0; i < vec_size; i++) {
            a_full[i] = complexd(uid(rnd), uid(rnd));
            // if (i % 4 == 0) { cout << '\n'; }
            // cout << a_full[i] << ' ';
        }
        // cout << "\n\n";
    }

    MPI_Bcast(a_full, vec_size,
            MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    complexd* a_part = a_full + rank * cur_vec_size;
    complexd* out_part = nullptr, *out_full = nullptr, *temp = nullptr;
    if (!rank) {
        out_full = new complexd[vec_size];
        temp = new complexd[vec_size];
    }

    out_part = adamar_gate(a_part, n, k);
    // for (int i = 0; i < cur_vec_size; i++) {
    //     cout << out_part[i] << '\n';
    // }
    MPI_Gather(out_part, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            out_full, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            0, MPI_COMM_WORLD);
    if (!rank) {
        ifstream f("test_dir/adamar_gate.dat", ios::binary | ios::in);
        f.read((char*) temp, vec_size * sizeof(complexd));
        bool flag = true;
        for (int i = 0; i < vec_size; i++) {
            // cout << out_full[i] << '\t' << temp[i] << '\n';
            if (out_full[i] != temp[i]) {
                flag = false;
                break;
            }
        }
        if (flag) {
            cout << "ADAMAR_GATE TEST PASSED\n";
        } else {
            cout << "ADAMAR_GATE TEST FAILED\n";
            exit(1);
        }
        f.close();
    }
    delete[] out_part;

    out_part = n_adamar_gate(a_part, n);
    MPI_Gather(out_part, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            out_full, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            0, MPI_COMM_WORLD);
    if (!rank) {
        ifstream f("test_dir/n_adamar_gate.dat", ios::binary | ios::in);
        f.read((char*) temp, vec_size * sizeof(complexd));
        bool flag = true;
        for (int i = 0; i < vec_size; i++) {
            if (out_full[i] != temp[i]) {
                flag = false;
                break;
            }
        }
        if (flag) {
            cout << "N_ADAMAR_GATE TEST PASSED\n";
        } else {
            cout << "N_ADAMAR_GATE TEST FAILED\n";
            exit(1);
        }
        f.close();
    }
    delete[] out_part;

    out_part = rw_gate(a_part, n, k, phi);
    MPI_Gather(out_part, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            out_full, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            0, MPI_COMM_WORLD);
    if (!rank) {
        ifstream f("test_dir/rw_gate.dat", ios::binary | ios::in);
        f.read((char*) temp, vec_size * sizeof(complexd));
        bool flag = true;
        for (int i = 0; i < vec_size; i++) {
            if (out_full[i] != temp[i]) {
                flag = false;
                break;
            }
        }
        if (flag) {
            cout << "RW_GATE TEST PASSED\n";
        } else {
            cout << "RW_GATE TEST FAILED\n";
            exit(1);
        }
        f.close();
    }
    delete[] out_part;

    out_part = c_rw_gate(a_part, n, k, l, phi);
    MPI_Gather(out_part, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            out_full, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            0, MPI_COMM_WORLD);
    if (!rank) {
        ifstream f("test_dir/c_rw_gate.dat", ios::binary | ios::in);
        f.read((char*) temp, vec_size * sizeof(complexd));
        bool flag = true;
        for (int i = 0; i < vec_size; i++) {
            if (out_full[i] != temp[i]) {
                flag = false;
                break;
            }
        }
        if (flag) {
            cout << "C_RW_GATE TEST PASSED\n";
        } else {
            cout << "C_RW_GATE TEST FAILED\n";
            exit(1);
        }
        f.close();
    }
    delete[] out_part;

    out_part = not_gate(a_part, n, k);
    MPI_Gather(out_part, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            out_full, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            0, MPI_COMM_WORLD);
    if (!rank) {
        ifstream f("test_dir/not_gate.dat", ios::binary | ios::in);
        f.read((char*) temp, vec_size * sizeof(complexd));
        bool flag = true;
        for (int i = 0; i < vec_size; i++) {
            if (out_full[i] != temp[i]) {
                flag = false;
                break;
            }
        }
        if (flag) {
            cout << "NOT_GATE TEST PASSED\n";
        } else {
            cout << "NOT_GATE TEST FAILED\n";
            exit(1);
        }
        f.close();
    }
    delete[] out_part;

    out_part = c_not_gate(a_part, n, k, l);
    MPI_Gather(out_part, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            out_full, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX,
            0, MPI_COMM_WORLD);
    if (!rank) {
        ifstream f("test_dir/c_not_gate.dat", ios::binary | ios::in);
        f.read((char*) temp, vec_size * sizeof(complexd));
        bool flag = true;
        for (int i = 0; i < vec_size; i++) {
            if (out_full[i] != temp[i]) {
                flag = false;
                break;
            }
        }
        if (flag) {
            cout << "C_NOT_GATE TEST PASSED\n";
        } else {
            cout << "C_NOT_GATE TEST FAILED\n";
            exit(1);
        }
        f.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    delete[] out_part;
    delete[] temp;

    return 0;
}
