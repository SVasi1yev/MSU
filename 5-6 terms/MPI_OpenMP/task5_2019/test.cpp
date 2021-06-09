#include <iostream>
#include <complex>
#include <fstream>

using namespace std;
typedef complex<double> complexd;

int main(int argc, char *argv[]) {
    ifstream f1(argv[1], ios::binary | ios::in);
    int n1;
    f1.read((char*) &n1, sizeof(int));
    complexd *a = new complexd[n1];
    f1.read((char*) a, n1 * sizeof(complexd));
    f1.close();

    ifstream f2(argv[2], ios::binary | ios::in);
    int n2;
    f2.read((char*) &n2, sizeof(int));
    complexd *b = new complexd[n2];
    f2.read((char*) b, n2 * sizeof(complexd));
    f2.close();

    if (n1 != n2) {
        cout << "TEST FAILED!\n";
        exit(1);
    }

    for (int i = 0; i < n1; i++) {
        if (abs((a[i] - b[i])) > 1e-6) {
            cout << "TEST FAILED!\n";
            exit(1);
        }
    }

    cout << "TEST PASSED!\n";

    delete[] a;
    delete[] b;

    return 0;
}
