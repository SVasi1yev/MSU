#include "scalapack.h"
#include <iostream>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    init_scalapack(scala_info);

    int N = atoi(argv[1]);
    int n = pow(2, N);

    Matrix ro(n);
    ro.read_from_file(argv[2]);

    double dT = atof(argv[3]);
    Matrix H(n);
    H.read_from_file(argv[4]);
    int step_num = atoi(argv[5]);

    double *einvals = new double[n];
    Matrix einvecs(n);
    Matrix::get_eigen_vals_vecs(H, einvals, einvecs);

    ro.print_diag();
    for (int i = 1; i <= step_num; i++) {
        Matrix expD(n);
        expD.make_zero_matrix();
        for (int j = 0; j < n; j++) {
            expD.set(j, j, exp(complex_d(0, -1 * dT * i * einvals[j])));
        }
        Matrix temp = Matrix::multiply(einvecs, expD);
        Matrix U = Matrix::multiply(temp, einvecs, 'N', 'C');
        temp = Matrix::multiply(U, ro);
        temp = Matrix::multiply(temp, U, 'N', 'C');
        temp.print_diag();
    }

    Cblacs_gridexit(scala_info.ctx);
    Cblacs_exit(0);
//    MPI_Finalize();
}
