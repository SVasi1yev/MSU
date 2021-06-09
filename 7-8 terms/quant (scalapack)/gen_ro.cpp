#include <iostream>
#include <fstream>
#include <complex>

typedef std::complex<double> complex_d;

using namespace std;

int main(int argc, char* argv[]) {
    int N = atoi(argv[1]);
    int n = pow(2, N);

    complex_d* d = new complex_d[n * n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                d[i * n + j] = complex_d(0.0 / n, 0);
            } else {
                d[i * n + j] = complex_d(0, 0);
            }
        }
    }
    d[0] = complex_d(1,0);
    ofstream out("ro.dat", ios::binary | ios::out);
    out.write((char*) d, n * n * sizeof(complex_d));
}