#ifndef SCALAPACK_H
#define SCALAPACK_H

#include <complex>
#include <mpi/mpi.h>
#include <iostream>

using namespace std;

typedef std::complex<float> complex_s;
typedef std::complex<double> complex_d;

extern "C" {
    void Cblacs_pcoord(int, int, int*, int*);

	void Cblacs_pinfo( int* mypnum, int* nprocs);
	void Cblacs_get(int context, int request, int* value);
	int Cblacs_gridinit( int* context, char * order, int np_row, int np_col);
	void Cblacs_gridinfo( int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);
	void Cblacs_gridexit( int context);
	void Cblacs_exit( int error_code);
	void Cblacs_gridmap( int* context, int* map, int ld_usermap, int np_row, int np_col);

	int npreroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
	int numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);

	void descinit_(int *idescal, int *m, int *n, int *mb, int *nb, int *dummy1 , int *dummy2 , int *icon, int *procRows, int *info);

	void psgesvd_(char *jobu, char *jobvt, int *m, int *n, float *a, int *ia, int *ja, int *desca, float *s, float *u, int *iu, int *ju, int *descu, float *vt, int *ivt, int *jvt, int *descvt, float *work, int *lwork, int *info);
	void pdgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *ia, int *ja, int *desca, double *s, double *u, int *iu, int *ju, int *descu, double *vt, int *ivt, int *jvt, int *descvt, double *work, int *lwork, int *info);
	void pcgesvd_(char *jobu, char *jobvt, int *m, int *n, complex_s *a, int *ia, int *ja, int *desca, float *s, complex_s *u, int *iu, int *ju, int *descu, complex_s *vt, int *ivt, int *jvt, int *descvt, complex_s *work, int *lwork, float *rwork, int *info);
	void pzgesvd_(char *jobu, char *jobvt, int *m, int *n, complex_d *a, int *ia, int *ja, int *desca, double *s, complex_d *u, int *iu, int *ju, int *descu, complex_d *vt, int *ivt, int *jvt, int *descvt, complex_d *work, int *lwork, double *rwork, int *info);
    void pdgemm_(char *transa, char *transb, int *M, int *N, int *K, double *alpha, double *A, int *ia, int *ja, int *descA,
             double *B, int *ib, int *jb, int *descB, double *beta, double *C, int *ic, int *jc, int *descC);
    void pzgemm_ (char *jobu, char *jobvt, int *M, int *N, int *K, double *ALPHA, complex_d *A, int *IA, int *JA, int *DESCA, complex_d *B, int *IB, int *JB, int *DESCB, double *BETA, complex_d *C, int *IC, int *JC, int *DESCC);
    void pzheevd_(char *jobz, char *uplo, int *n, complex_d *a, int *ia, int *ja, int *desa, double *w,
              complex_d *z, int *iz, int *jz, int *descz, complex_d *work, int *lwork,  double *rwork, int *lrwork, int *iwork, int *liwork, int *info);
}

int iZero = 0;
int iOne = 1;
double dZero = 0.0;
double dOne = 1.0;

struct ScalapackInfo {
    int rank, size;
    int dims[2] ={0};
    int ctx;
    int np_row, np_col;
    int my_row, my_col;
    int info;
};

void init_scalapack(ScalapackInfo& info) {
    MPI_Comm_size(MPI_COMM_WORLD, &info.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &info.rank);

    MPI_Dims_create(info.size, 2, info.dims);
    info.np_row = info.dims[0], info.np_col = info.dims[1];

    Cblacs_pinfo(&info.rank, &info.size);
    Cblacs_get(-1, 0, &info.ctx);
    Cblacs_gridinit(&info.ctx, "Row-major", info.np_row, info.np_col);
    Cblacs_gridinfo(info.ctx, &info.np_row, &info.np_col, &info.my_row, &info.my_col);
    swap(info.np_row, info.np_col);
    swap(info.my_row, info.my_col);
}

ScalapackInfo scala_info;

class Matrix {
public:
    complex_d* data;
    int desc[9];
    int global_N;
    int local_N;
    int local_M;
    int my_n;
    int my_m;
    int data_len;

    Matrix(int n): global_N(n) {
        local_N = global_N % scala_info.np_row == 0 ?
                  global_N / scala_info.np_row : global_N / scala_info.np_row + 1;
        local_M = global_N % scala_info.np_col == 0 ?
                  global_N / scala_info.np_col : global_N / scala_info.np_col + 1;
//        local_N = local_M = max(local_N, local_M);
        my_n = numroc_(&global_N, &local_N, &scala_info.my_row, &iZero, &scala_info.np_row);
        my_m = numroc_(&global_N, &local_M, &scala_info.my_col, &iZero, &scala_info.np_col);
        descinit_(desc, &global_N, &global_N, &local_M, &local_N, &iZero, &iZero,
                  &scala_info.ctx, &my_m, &scala_info.info);
        data_len = my_n * my_m;
        data = new complex_d[data_len];
    }

    Matrix(const Matrix& other): global_N(other.global_N), local_N(other.local_N),
        local_M(other.local_M), my_n(other.my_n), my_m(other.my_m), data_len(other.data_len)
    {
        for (int i = 0; i < 9; i++) {
            desc[i] = other.desc[i];
        }
        data = new complex_d[data_len];
        for (int i = 0; i < data_len; i++) {
            data[i] = other.data[i];
        }
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            global_N = other.global_N;
            local_N = other.local_N;
            local_M = other.local_M;
            my_n = other.my_n;
            my_m = other.my_m;
            data_len = other.data_len;
            for (int i = 0; i < 9; i++) {
                desc[i] = other.desc[i];
            }
            delete[] data;
            data = new complex_d[data_len];
            for (int i = 0; i < data_len; i++) {
                data[i] = other.data[i];
            }
        }

        return *this;
    }

    void read_from_file(char* file_name) {
        make_zero_matrix();
        MPI_File file;
        MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
        MPI_Datatype file_type;
        int global_size[2] = {global_N, global_N};
//        int local_size[2] = {my_m, my_n};
        int local_size[2] = {my_n, my_m};
//        int start[2] = {local_M * scala_info.my_col, local_N * scala_info.my_row};
        int start[2] = {local_N * scala_info.my_row, local_M * scala_info.my_col};
        MPI_Type_create_subarray(2, global_size, local_size, start,
                                 MPI_ORDER_C, MPI_CXX_DOUBLE_COMPLEX, &file_type);
        MPI_Type_commit(&file_type);
        MPI_File_set_view(file, 0, MPI_CXX_DOUBLE_COMPLEX, file_type,
                          "native", MPI_INFO_NULL);
        MPI_File_read(file, data, my_n * my_m, MPI_CXX_DOUBLE_COMPLEX, MPI_STATUS_IGNORE);
        MPI_Type_free(&file_type);
        MPI_File_close(&file);
    }

    void print_diag() {
        // ТОЛЬКО ДЛЯ КВАДРАТНОГО ГРИДА
        if (scala_info.my_row == scala_info.my_col) {
            if (scala_info.rank != scala_info.size - 1) {
                complex_d* d = new complex_d[local_N];
                for (int i = 0; i < local_N; i++) {
                    d[i] = data[local_N * i + i];
                }
//                cout << "SEND FROM " << scala_info.rank << "\n\n";
                MPI_Send(d, local_N, MPI_CXX_DOUBLE_COMPLEX, scala_info.size - 1,
                         scala_info.my_row, MPI_COMM_WORLD);
            } else {
                complex_d* d = new complex_d[global_N];
                for (int i = 0; i < scala_info.np_row - 1; i++) {
//                    cout << "RECV FROM " << i * scala_info.np_col + i << "\n\n";
                    MPI_Recv(d + i * local_N, local_N, MPI_CXX_DOUBLE_COMPLEX,
                             i * scala_info.np_col + i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                for (int i = 0; i < my_n; i++) {
                    d[local_N * (scala_info.np_row - 1) + i] = data[i * my_m + i];
                }

                cout << "DIAG: ";
                for (int i = 0; i < global_N; i++) {
                    cout << d[i] << ' ';
                }
                complex_d trace = complex_d(0, 0);
                for (int i = 0; i < global_N; i++) {
                    trace += d[i];
                }
                cout << "=== TRACE: " << trace;
                cout << "\n\n";
            }
        }
    }

    bool get(int i, int j, complex_d& val) {
        if (i >= global_N || j >= global_N) {
            return false;
        }
        int proc_i, proc_j;
        int offset_i, offset_j;
        proc_i = i / local_N;
        proc_j = j / local_M;
        offset_i = i - proc_i * local_N;
        offset_j = j - proc_j * local_M;

        if (scala_info.my_row == proc_i && scala_info.my_col == proc_j) {
            val = data[my_m * offset_i + offset_j];
            return true;
        }
        return false;
    }

    bool set(int i, int j, complex_d val) {
        if (i >= global_N || j >= global_N) {
            return false;
        }
        int proc_i, proc_j;
        int offset_i, offset_j;
        proc_i = i / local_N;
        proc_j = j / local_M;
        offset_i = i - proc_i * local_N;
        offset_j = j - proc_j * local_M;

        if (scala_info.my_row == proc_i && scala_info.my_col == proc_j) {
            data[my_m * offset_i + offset_j] = val;
            return true;
        }
        return false;
    }

    void make_zero_matrix() {
        for (int i = 0; i < data_len; i++) {
            data[i] = 0.0;
        }
    }

    static Matrix multiply(Matrix& A, Matrix& B, char transa = 'N', char transb = 'N') {
        Matrix C(A.global_N);
        C.make_zero_matrix();

        pzgemm_(&transa, &transb, &A.global_N, &A.global_N, &B.global_N,
                &dOne,
                A.data, &iOne, &iOne, A.desc,
                B.data, &iOne, &iOne, B.desc,
                &dZero,
                C.data, &iOne, &iOne, C.desc);

        return C;
    }

    static void get_eigen_vals_vecs(Matrix& A, double* eigenvals, Matrix& eigenvecs) {
        // ТОЛЬКО ДЛЯ КВАДРАТНОГО ГРИДА
        char jobz = 'V';
        char uplo = 'U';

        int lwork = -1;
        complex_d* work = new complex_d[1];
        int lrwork = -1;
        double* rwork = new double[1];
        int liwork = -1;
        int* iwork = new int[1];

        pzheevd_(&jobz, &uplo, &A.global_N,
             A.data, &iOne, &iOne, A.desc,
             eigenvals,
             eigenvecs.data, &iOne, &iOne, eigenvecs.desc,
             work, &lwork, rwork, &lrwork, iwork, &liwork,
             &scala_info.info
        );

        lwork = work[0].real();
        lrwork = rwork[0];
        liwork = iwork[0];

        delete[] work;
        delete[] rwork;
        delete[] iwork;

        work = new complex_d[lwork];
        rwork = new double[lrwork];
        iwork = new int[liwork];

        pzheevd_(&jobz, &uplo, &A.global_N,
             A.data, &iOne, &iOne, A.desc,
             eigenvals,
             eigenvecs.data, &iOne, &iOne, eigenvecs.desc,
             work, &lwork, rwork, &lrwork, iwork, &liwork,
             &scala_info.info
        );

        delete[] work;
        delete[] rwork;
        delete[] iwork;
    }

    ~Matrix(){
        delete[] data;
    }
};
#endif
