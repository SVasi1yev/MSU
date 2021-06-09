#include <iostream>
#include <complex>
#include <fstream>
#include <ctime>
#include <random>

using namespace std;
typedef complex<double> complexd;

int main (int argc, char *argv[]) {
    ifstream input_file(argv[1], ios::binary | ios::in);
    

    int n;
    input_file.read((char*) &n, sizeof(int));
    cout << n << endl;

    complexd *a = new complexd[n];
    input_file.read((char*) a, n * sizeof(complexd));
    for (int i = 0; i < n; i += 4) {
        for (int j = 0; j < 4; j++) {
            cout << a[i + j] << " ";
        }
        cout << endl;
    }

    input_file.close();
    delete[] a;

    return 0;
}