#include <iostream>
#include "test_q_gates.h"
#include <random>
#include <string.h>
#include <fstream>

using namespace std;
using complexd = complex<double>;

const double pi = 3.1415926535;

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    int vec_size = 1 << n;
    mt19937_64 rnd;
    uniform_int_distribution<> uid(-10, 10);

    complexd* a = new complexd[vec_size];
    complexd* temp = nullptr;

    ifstream in(argv[2], ios::binary | ios::in);
    in.read((char*) a, sizeof(int));
    in.read((char*) a, vec_size * sizeof(complexd));
    in.close();

    // for (int i = 0; i < vec_size; i++) {
    //     if (i % 4 == 0) {
    //         cout <<'\n';
    //     }
    //     cout << a[i] << ' ';
    // }

    for (int i = 0; i < n - 1; i++) {
        temp = a;
        a = test_adamar_gate(a, n, i);
        delete[] temp;
        for (int j = 1; j < n - i; j++) {
            temp = a;
            a = test_c_rw_gate(a, n, i, i + j,
                        complexd(pi / pow(2, j)));
            delete[] temp;
        }
    }
    temp = a;
    a = test_adamar_gate(a, n, n - 1);
    delete[] temp;

    for (int i = 0; i < n / 2; i++) {
        temp = a;
        a = test_swap_gate(a, n, i, n - i - 1);
        delete[] temp;
    }

    // for (int i = 0; i < vec_size; i++) {
    //     if (i % 4 == 0) {
    //         cout <<'\n';
    //     }
    //     cout << a[i] << ' ';
    // }

    ofstream out(argv[3], ios::binary | ios::out);
    out.write((char*) &vec_size, sizeof(int));
    out.write((char*) a, vec_size * sizeof(complexd));
    out.close();

    delete[] a;

    return 0;
}