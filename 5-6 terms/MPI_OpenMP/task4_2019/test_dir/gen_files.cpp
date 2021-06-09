#include <iostream>
#include <random>
#include <fstream>
#include <complex>
#include "./test_q_gates.h"

int main() {
    int n = 5;
    int k = 0;
    int l = 1;
    complexd phi(3.14 / 4, 0.0);
    int vec_size = 1 << n;
    mt19937_64 rnd;
    rnd.seed(0);
    uniform_int_distribution<> uid(-5, 5);

    complexd* a = new complexd[vec_size];
    for (size_t i = 0; i < vec_size; i++) {
        a[i] = complexd(uid(rnd), uid(rnd));
    }

    complexd* out;
    ofstream f;

    out = new complexd[vec_size];
    out = test_adamar_gate(a, n, k);
    f.open("adamar_gate.dat", ios::binary | ios::out);
    f.write((char*) out, vec_size * sizeof(complexd));
    f.close();
    delete[] out;

    out = new complexd[vec_size];
    out = test_n_adamar_gate(a, n);
    f.open("n_adamar_gate.dat", ios::binary | ios::out);
    f.write((char*) out, vec_size * sizeof(complexd));
    f.close();
    delete[] out;

    out = new complexd[vec_size];
    out = test_rw_gate(a, n, k, phi);
    f.open("rw_gate.dat", ios::binary | ios::out);
    f.write((char*) out, vec_size * sizeof(complexd));
    f.close();
    delete[] out;

    out = new complexd[vec_size];
    out = test_c_rw_gate(a, n, k, l, phi);
    f.open("c_rw_gate.dat", ios::binary | ios::out);
    f.write((char*) out, vec_size * sizeof(complexd));
    f.close();
    delete[] out;

    out = new complexd[vec_size];
    out = test_not_gate(a, n, k);
    f.open("not_gate.dat", ios::binary | ios::out);
    f.write((char*) out, vec_size * sizeof(complexd));
    f.close();
    delete[] out;

    out = new complexd[vec_size];
    out = test_c_not_gate(a, n, k, l);
    f.open("c_not_gate.dat", ios::binary | ios::out);
    f.write((char*) out, vec_size * sizeof(complexd));
    f.close();
    delete[] out;

    delete[] a;
}