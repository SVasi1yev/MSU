#include <iostream>
#include <complex>
#include <fstream>
#include <ctime>
#include <random>

using namespace std;
typedef complex<double> complexd;

int main (int argc, char *argv[]) {
    int n = 1 << atoi(argv[1]);
    mt19937_64 rnd;
    uniform_int_distribution<> uid(-5, 5);
    rnd.seed(time(0));

    complexd *a = new complexd[n];

    for (int i = 0; i < n; i++) {
        a[i] = complexd(uid(rnd), uid(rnd));
    }

    ofstream output_file;
    output_file.open(argv[2], ios::binary | ios::out | ios::trunc);
    output_file.write((char*) &n, sizeof(int));
    output_file.write((char*) a, n * sizeof(complexd));

    output_file.close();
    delete[] a;

    return 0;
}