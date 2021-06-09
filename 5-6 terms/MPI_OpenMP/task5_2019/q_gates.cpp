#include "q_gates.h"
#include <mpi/mpi.h>
#include <omp.h>
#include <complex>
#include <cmath>

using namespace std;
using complexd = complex<double>;

complexd* single_q_t(complexd* a, int n,
                    complexd u[2][2], int k) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int vec_size = 1 << n;
    int cur_vec_size = vec_size / size;
    MPI_Status status;

    complexd *out = new complexd[cur_vec_size];
    int temp = size;
    int size_power = -1;
    while (temp > 0) {
        temp >>= 1;
        size_power++;
    }

    if (k >= size_power) {
        int shift = n - 1 - k;
        int mask = 1 << shift;

        #pragma omp parallel for
        for (int i = 0; i < cur_vec_size; i++) {
            int i0 = i & ~mask;
            int i1 = i | mask;
            int iq = (i & mask) >> shift;
            out[i] = u[iq][0] * a[i0] + u[iq][1] * a[i1];
        }
    } else {
        int zero_one = rank & (1 << (size_power - k - 1));
        int partner = rank ^ (1 << (size_power - k - 1));
        complexd *a_partner = new complexd[cur_vec_size];
        MPI_Sendrecv(a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner, 0,
                a_partner, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner,
                0, MPI_COMM_WORLD, &status);

        if (zero_one) {
            #pragma omp parallel for
            for (int i = 0; i < cur_vec_size; i++) {
                out[i] = u[1][0] * a_partner[i] + u[1][1] * a[i];
            }
        } else {
            #pragma omp parallel for
            for (int i = 0; i < cur_vec_size; i++) {
                out[i] = u[0][0] * a[i] + u[0][1] * a_partner[i];
            }
        }
    }

    return out;
}

complexd* double_q_t(complexd* a, int n,
                    complexd u[4][4], int k, int l) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int vec_size = 1 << n;
    int cur_vec_size = vec_size / size;
    MPI_Status status;

    complexd* out = new complexd[cur_vec_size];
    int temp = size;
    int size_power = -1;
    while (temp > 0) {
        temp >>= 1;
        size_power++;
    }

    if ((k >= size_power) && (l >= size_power)) {
        int shift_k = n - 1 - k;
        int shift_l = n - 1 - l;
        int mask_k = 1 << shift_k;
        int mask_l = 1 << shift_l;

        #pragma omp parallel for
        for (int i = 0; i < cur_vec_size; i++) {
            int i00 = i & ~mask_k & ~mask_l;
            int i01 = (i & ~mask_k) | mask_l;
            int i10 = (i | mask_k) & ~mask_l;
            int i11 = i | mask_k | mask_l;

            int iq_k = (i & mask_k) >> shift_k;
            int iq_l = (i & mask_l) >> shift_l;
            int iq = (iq_k << 1) + iq_l;

            out[i] = u[iq][0] * a[i00] + u[iq][1] * a[i01]
                    + u[iq][2] * a[i10] + u[iq][3] * a[i11];
        }
    } else if ((k < size_power) && (l >= size_power)) {
        int zero_one = rank & (1 << (size_power - k - 1));
        int partner_rank = rank ^ (1 << (size_power - k - 1));
        complexd* partner_a = new complexd[cur_vec_size];
        MPI_Sendrecv(a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_rank, 0,
                partner_a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_rank,
                0, MPI_COMM_WORLD, &status);

        int shift_l = n - 1 - l;
        int mask_l = 1 << shift_l;
        if (zero_one) {
            #pragma omp parallel for
            for (int i = 0; i < cur_vec_size; i++) {
                int i0 = i & ~mask_l;
                int i1 = i | mask_l;

                int iq_l = (i & mask_l) >> shift_l;
                int iq = 2 + iq_l;
                out[i] = u[iq][0] * partner_a[i0] + u[iq][1] * partner_a[i1]
                        + u[iq][2] * a[i0] + u[iq][3] * a[i1];
            }
        } else {
            #pragma omp parallel for
            for (int i = 0; i < cur_vec_size; i++) {
                int i0 = i & ~mask_l;
                int i1 = i | mask_l;

                int iq_l = (i & mask_l) >> shift_l;
                int iq = iq_l;
                out[i] = u[iq][0] * a[i0] + u[iq][1] * a[i1]
                        + u[iq][2] * partner_a[i0] + u[iq][3] * partner_a[i1];
            }
        }
        delete[] partner_a;
    } else if ((k >= size_power) && (l < size_power)) {
        int zero_one = rank & (1 << (size_power - l - 1));
        int partner_rank = rank ^ (1 << (size_power - l - 1));
        complexd* partner_a = new complexd[cur_vec_size];
        MPI_Sendrecv(a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_rank, 0,
                partner_a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_rank,
                0, MPI_COMM_WORLD, &status);

        int shift_k = n - 1 - k;
        int mask_k = 1 << shift_k;
        if (zero_one) {
            #pragma omp parallel for
            for (int i = 0; i < cur_vec_size; i++) {
                int i0 = i & ~mask_k;
                int i1 = i | mask_k;

                int iq_k = (i & mask_k) >> shift_k;
                int iq = (iq_k << 1) + 1;
                out[i] = u[iq][0] * partner_a[i0] + u[iq][1] * a[i0]
                        + u[iq][2] * partner_a[i1] + u[iq][3] * a[i1];
            }
        } else {
            #pragma omp parallel for
            for (int i = 0; i < cur_vec_size; i++) {
                int i0 = i & ~mask_k;
                int i1 = i | mask_k;

                int iq_k = (i & mask_k) >> shift_k;
                int iq = iq_k << 1;
                out[i] = u[iq][0] * a[i0] + u[iq][1] * partner_a[i0]
                        + u[iq][2] * a[i1] + u[iq][3] * partner_a[i1];
            }
        }
        delete[] partner_a;
    } else {
        int zero_one_k = (rank & (1 << (size_power - k - 1)))
                        >> (size_power - k - 1);
        int zero_one_l = (rank & (1 << (size_power - l - 1)))
                        >> (size_power - l - 1);

        int partner_00_rank = (rank & ~(1 << (size_power - k - 1)))
                                & ~(1 << (size_power - l - 1));
        int partner_01_rank = (rank & ~(1 << (size_power - k - 1)))
                                | (1 << (size_power - l - 1));
        int partner_10_rank = (rank | (1 << (size_power - k - 1)))
                                & ~(1 << (size_power - l - 1));
        int partner_11_rank = (rank | (1 << (size_power - k - 1)))
                                | (1 << (size_power - l - 1));

        complexd* partner_00_a = nullptr,
                *partner_01_a = nullptr,
                *partner_10_a = nullptr,
                *partner_11_a = nullptr;

        if (rank == partner_00_rank) {
            partner_00_a = a;
        } else {
            partner_00_a = new complexd[cur_vec_size];
            MPI_Sendrecv(a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_00_rank, 0,
                partner_00_a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_00_rank,
                0, MPI_COMM_WORLD, &status);
        }
        if (rank == partner_01_rank) {
            partner_01_a = a;
        } else {
            partner_01_a = new complexd[cur_vec_size];
            MPI_Sendrecv(a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_01_rank, 0,
                partner_01_a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_01_rank,
                0, MPI_COMM_WORLD, &status);
        }
        if (rank == partner_10_rank) {
            partner_10_a = a;
        } else {
            partner_10_a = new complexd[cur_vec_size];
            MPI_Sendrecv(a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_10_rank, 0,
                partner_10_a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_10_rank,
                0, MPI_COMM_WORLD, &status);
        }
        if (rank == partner_11_rank) {
            partner_11_a = a;
        } else {
            partner_11_a = new complexd[cur_vec_size];
            MPI_Sendrecv(a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_11_rank, 0,
                partner_11_a, cur_vec_size, MPI_CXX_DOUBLE_COMPLEX, partner_11_rank,
                0, MPI_COMM_WORLD, &status);
        }

        int iq = (zero_one_k << 1) + zero_one_l;
        #pragma omp parallel for
        for (int i = 0; i < cur_vec_size; i++) {
            out[i] = u[iq][0] * partner_00_a[i] + u[iq][1] * partner_01_a[i]
                    + u[iq][2] * partner_10_a[i] + u[iq][3] * partner_11_a[i];
        }

        if (rank != partner_00_rank) {
            delete[] partner_00_a;
        }
        if (rank != partner_01_rank) {
            delete[] partner_01_a;
        }
        if (rank != partner_10_rank) {
            delete[] partner_10_a;
        }
        if (rank != partner_11_rank) {
            delete[] partner_11_a;
        }
    }

    return out;
}

complexd* adamar_gate(complexd* a, int n, int k) {
    complexd u[2][2] = {
        complexd(1.0 / pow(2, 0.5)),
        complexd(1.0 / pow(2, 0.5)),

        complexd(1.0 / pow(2, 0.5)),
        complexd(-1.0 / pow(2, 0.5))
    };

    return single_q_t(a, n, u, k);
}

complexd* n_adamar_gate(complexd* a, int n) {
    complexd* out = adamar_gate(a, n, 0);
    complexd* temp = out;
    for (int i = 1; i < n; i++) {
        out = adamar_gate(temp, n, i);
        delete[] temp;
        temp = out;
    }

    return out;
}

complexd* rw_gate(complexd* a, int n, int k, complexd phi) {
    complexd u[2][2] = {
        complexd(1.0),
        complexd(0.0),

        complexd(0.0),
        exp(complexd(0.0, 1.0) * phi)
    };

    return single_q_t(a, n, u, k);
}

complexd* c_rw_gate(complexd* a, int n, int k,
                    int l, complexd phi) {
    complexd u[4][4] = {
        complexd(1.0), complexd(0.0),
        complexd(0.0), complexd(0.0),

        complexd(0.0), complexd(1.0),
        complexd(0.0), complexd(0.0),

        complexd(0.0), complexd(0.0),
        complexd(1.0), complexd(0.0),

        complexd(0.0), complexd(0.0),
        complexd(0.0), exp(complexd(0.0, 1.0) * phi)
    };

    return double_q_t(a, n, u, k, l);
}

complexd* not_gate(complexd* a, int n, int k) {
    complexd u[2][2] = {
        complexd(0.0),
        complexd(1.0),

        complexd(1.0),
        complexd(0.0)
    };

    return single_q_t(a, n, u, k);
}

complexd* c_not_gate(complexd* a, int n, int k,
                int l) {
    complexd u[4][4] = {
        complexd(1.0), complexd(0.0),
        complexd(0.0), complexd(0.0),

        complexd(0.0), complexd(1.0),
        complexd(0.0), complexd(0.0),

        complexd(0.0), complexd(0.0),
        complexd(0.0), complexd(1.0),

        complexd(0.0), complexd(0.0),
        complexd(1.0), complexd(0.0)
    };

    return double_q_t(a, n, u, k, l);
}

complexd* swap_gate(complexd* a, int n, int k,
                int l) {
    complexd u[4][4] = {
        complexd(1.0), complexd(0.0),
        complexd(0.0), complexd(0.0),

        complexd(0.0), complexd(0.0),
        complexd(1.0), complexd(0.0),

        complexd(0.0), complexd(1.0),
        complexd(0.0), complexd(0.0),

        complexd(0.0), complexd(0.0),
        complexd(0.0), complexd(1.0)
    };

    return double_q_t(a, n, u, k, l);
}