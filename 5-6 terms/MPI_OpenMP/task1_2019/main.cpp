#include <omp.h>
#include <ctime>
#include <iostream>
#include <complex>
#include <random>

using namespace std;
typedef complex<double> complexd;

complexd* single_qubit_transform (complexd *a, int n, complexd u[2][2], int k, int num_treads) {
    int size = 1 << n;
    complexd *out = new complexd[size];

    int shift = n - k - 1;
    int mask = 1 << (shift);
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        int i0 = i & ~mask;
        int i1 = i | mask;
        int iq = (i & mask) >> shift; 
        out[i] = u[iq][0] * a[i0] + u[iq][1] * a[i1];
    }

    return out;
} 

int main (int argc, char* argv[]) {
    double time0 = omp_get_wtime();
    mt19937_64 rnd;
    uniform_int_distribution<> uid(-100, 100);
    omp_set_dynamic(0);
    int num_treads = atoi(argv[1]);
    omp_set_num_threads(num_treads);

    int n = atoi(argv[2]);
    int k = atoi(argv[3]) - 1;
    int size = 1 << n;
    complexd *a = new complexd[size];

    #pragma omp parallel private(rnd)
    {
        int temp = omp_get_thread_num();
        rnd.seed((temp + 1) * time(0));
        #pragma omp for
        for (int i = 0; i < size; i ++) {
            a[i] = complexd(uid(rnd), uid(rnd));
            // a[i] = complexd(i, i);
        }
    }

    
    // for (int i = 0; i < size; i++) {
    //     cout << a[i] << " ";
    // }
    // cout << endl;
    

    double len = 0;
    #pragma omp parallel for reduction (+: len)
    for (int i = 0; i < size; i++) {
        len += a[i].real() * a[i].real() + a[i].imag() * a[i].imag();
    }
    len = sqrt(len);
    // cout << "len = " << len << endl;
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        a[i] /= len;
    }

    complexd u[2][2] = {
        complexd(1.0 / pow(2, 0.5)),
        complexd(1.0 / pow(2, 0.5)),
        complexd(1.0 / pow(2, 0.5)),
        complexd(-1.0 / pow(2, 0.5))
    };

    complexd *out = single_qubit_transform(a, n, u, k, num_treads);
    time0 = omp_get_wtime() - time0;

    /*
    cout << "size = " << size << endl;;
    for (int i = 0; i < size; i++) {
        cout << a[i] << " ";
    }
    cout << endl;
    */
    
    // len = 0;
    // #pragma omp parallel for reduction (+: len)
    // for (int i = 0; i < size; i++) {
    //     len += a[i].real() * a[i].real() + a[i].imag() * a[i].imag();
    // }
    // len = sqrt(len);
    // cout << "len = " << len << endl;
    
    /*
    for (int i = 0; i < size; i++) {
        cout << a[i] << " ";
    }
    cout << endl;
    */
    // for (int i = 0; i < size; i++) {
    //     cout << out[i] << " ";
    // }
    // cout << endl;
    
    cout << "time = " << time0 << endl;
}
