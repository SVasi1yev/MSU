#include <cmath>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <omp.h>

using namespace std;

const double pi = 3.1415926535897932;
double Lx, Ly, Lz;
double hx, hy, hz, tau;
const int ndims = 3;
int dim_loc_size[ndims];

inline double x(int glob_idx) { 
    return glob_idx * hx; 
}
inline double y(int glob_idx) { 
    return glob_idx * hy; 
}
inline double z(int glob_idx) { 
    return glob_idx * hz; 
}

double phi(double x, double y, double z) {
    return sin(3 * pi * x / Lx) * sin(2 * pi * y / Ly) * sin(2 * pi * z / Lz);
}

double a() {
    return pi * sqrt(9. / (Lx * Lx) + 4. / (Ly * Ly) + 4. / (Lz * Lz));
}

double u(double x, double y, double z, double t) {
    return phi(x, y, z) * cos(a() * t + 4 * pi);
}
int idx (int i, int j, int k) {
    return i * dim_loc_size[1] * dim_loc_size[2] + j * dim_loc_size[2] + k;
} 
double laplace(double data[], int i, int j, int k) {
    double res = 0;
    int prev, curr = idx(i, j, k), next;

    prev = idx(i-1, j, k);
    next = idx(i+1, j, k);
    res += (data[prev] - 2 * data[curr] + data[next]) / (hx * hx);

    prev = idx(i, j-1, k);
    next = idx(i, j+1, k);
    res += (data[prev] - 2 * data[curr] + data[next]) / (hy * hy);

    prev = idx(i, j, k-1);
    next = idx(i, j, k+1);
    res += (data[prev] - 2 * data[curr] + data[next]) / (hz * hz);

    return res;
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Status status;
    int N, K;
    int dx, dy, dz;
    double max_error = 0;

    N = atoi(argv[1]);
    K = atoi(argv[2]);
    Lx = atof(argv[3]);
    Ly = atof(argv[4]);
    Lz = atof(argv[5]);
    hx = Lx / (N - 1); 
    hy = Ly / (N - 1);
    hz = Lz / (N - 1);
    tau = 5 * 1e-5; 
    const int t_steps = K;

    int bcond_type[ndims];
    bcond_type[0] = 0;
    bcond_type[1] = 1;
    bcond_type[2] = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    omp_set_num_threads(4);
    double t_start = MPI_Wtime();

    int grid_dims[ndims];
    MPI_Dims_create(size, ndims, grid_dims);
    int periods[ndims];
    for (int i = 0; i < ndims; i++) {
        periods[i] = 1;
    }

    MPI_Comm grid_comm;
    int grid_coords[ndims];
    MPI_Cart_create(MPI_COMM_WORLD, ndims, grid_dims, periods, 0, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, ndims, grid_coords);

    int rank_source[ndims];
    int rank_dest[ndims];
    for (int i = 0; i < ndims; i++) {
        MPI_Cart_shift(grid_comm, i, 1, &rank_source[i], &rank_dest[i]);
    }
    bool bfisrt[ndims];
    bool blast[ndims];
    for (int i = 0; i < ndims; i++) {
        bfisrt[i] = (grid_coords[i] == 0);
        blast[i] = (grid_coords[i] == (grid_dims[i] - 1));
    }

    int first_idx[ndims];
    for (int i = 0; i < ndims; i++) {
        int r = N % grid_dims[i];
        int base = N / grid_dims[i];
        first_idx[i] = base * grid_coords[i] + min(r, grid_coords[i]);
        dim_loc_size[i] = base + ((grid_coords[i] < r) ? 1 : 0) + 2; //для граничных
    }
    int total_loc_size = 1;
    for (int i = 0; i < ndims; i++) { total_loc_size *= dim_loc_size[i]; }
    double* data[t_steps];
    for (int i = 0; i < t_steps; i++) {
        data[i] = new double[total_loc_size];
    }

#pragma omp parallel for collapse(3) num_threads(4)
    for (int i = 0; i < dim_loc_size[0]; i++) {
        for (int j = 0; j < dim_loc_size[1]; j++) {
            for (int k = 0; k < dim_loc_size[2]; k++) {
                data[0][idx(i, j, k)] = phi(
                    x(first_idx[0] + i - 1),
                    y(first_idx[1] + j - 1),
                    z(first_idx[2] + k - 1)
                );
            }
        }
    }

    max_error = 0;
    int max_i = 0;
    int max_j = 0;
    int max_k = 0;
#pragma omp parallel for collapse(3) num_threads(4)
    for (int i = 1; i < dim_loc_size[0] - 1; i++) {
        for (int j = 1; j < dim_loc_size[1] - 1; j++) {
            for (int k = 1; k < dim_loc_size[2] - 1; k++) {
                double u_true = u(
                    x(first_idx[0] + i - 1),
                    y(first_idx[1] + j - 1),
                    z(first_idx[2] + k - 1),
                    0 * tau
                );
#pragma omp critical
                {
                    double err = fabs(u_true - data[0][idx(i, j, k)]);
                    if (err > max_error) {
                        max_error = err;
                        max_i = i;
                        max_j = j;
                        max_k = k;
                    }
                }
            }
        }
    }
    double temp_error = max_error;
    MPI_Reduce(&temp_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm);
    if (!rank) {
        cout << first_idx[0] << ' ' << dim_loc_size[0] << endl;
        cout << "T_STEP: " << 0 << endl;
        cout << "ERROR: " << max_error << endl;
        cout << "TIME: " << MPI_Wtime() - t_start << endl;
    }
    MPI_Barrier(grid_comm);
    for (int p = 0; p < size; p++) {
        if (rank == p) {
            cout
            << " RANK: " << rank 
            << " i: " << max_i
            << " j: " << max_j
            << " k: " << max_k
            << endl;
        }
        MPI_Barrier(grid_comm);
    }

    for (int t = 1; t < t_steps; t++) {
#pragma omp parallel for collapse(3) num_threads(4)
        for (int i = 1; i < dim_loc_size[0] - 1; i++) {
            for (int j = 1; j < dim_loc_size[1] - 1; j++) {
                for (int k = 1; k < dim_loc_size[2] - 1; k++) {
                    // if (t == 0) {
                    //     data[t][idx(i, j, k)] = phi(
                    //         x(first_idx[0] + i - 1),
                    //         y(first_idx[1] + j - 1),
                    //         z(first_idx[2] + k - 1)
                    //     );
                    // }
                    // else 
                    if (t == 1) {     
                        data[t][idx(i, j, k)] = data[t-1][idx(i, j, k)] + 0.5 * tau * tau *
                            laplace(data[t-1], i, j, k);
                    } else {
                        data[t][idx(i, j, k)] = 2 * data[t-1][idx(i, j, k)] - data[t-2][idx(i, j, k)]
                            + tau * tau 
                            * laplace(data[t-1], i, j, k);
                    }
                }
            }
        }

        if (bfisrt[0] && blast[0]) {
            for (int j = 0; j < dim_loc_size[1]; j++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                    data[t][idx(0, j, k)] = data[t][idx(dim_loc_size[0] - 1, j, k)];
                }
            }
        } else {
            double send_buffer[dim_loc_size[1] * dim_loc_size[2]];
            double recv_buffer[dim_loc_size[1] * dim_loc_size[2]];
            for (int j = 0; j < dim_loc_size[1]; j++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                    send_buffer[j * dim_loc_size[2] + k] 
                        = data[t][idx(dim_loc_size[0] - 2, j, k)];
                }
            }
            MPI_Sendrecv(
                send_buffer, dim_loc_size[1] * dim_loc_size[2], 
                MPI_DOUBLE, rank_dest[0], 1, 
                recv_buffer, dim_loc_size[1] * dim_loc_size[2],
                MPI_DOUBLE, rank_source[0], 1,
                grid_comm, &status
            );
            for (int j = 0; j < dim_loc_size[1]; j++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                    data[t][idx(0, j, k)]
                        = recv_buffer[j * dim_loc_size[2] + k];
                }
            }
        }

        if (bfisrt[0] && blast[0]) {
            for (int j = 0; j < dim_loc_size[1]; j++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                    data[t][idx(dim_loc_size[0] - 1, j, k)] = data[t][idx(1, j, k)];
                }
            }
        } else {
            double send_buffer[dim_loc_size[1] * dim_loc_size[2]];
            double recv_buffer[dim_loc_size[1] * dim_loc_size[2]];
            for (int j = 0; j < dim_loc_size[1]; j++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                    send_buffer[j * dim_loc_size[2] + k] 
                        = data[t][idx(1, j, k)];
                }
            }
            MPI_Sendrecv(
                send_buffer, dim_loc_size[1] * dim_loc_size[2], 
                MPI_DOUBLE, rank_source[0], 1, 
                recv_buffer, dim_loc_size[1] * dim_loc_size[2],
                MPI_DOUBLE, rank_dest[0], 1,
                grid_comm, &status);
            for (int j = 0; j < dim_loc_size[1]; j++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                    data[t][idx(dim_loc_size[0] - 1, j, k)]
                        = recv_buffer[j * dim_loc_size[2] + k];
                }
            }
        }

        if (bfisrt[1] && blast[1]) {
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                    data[t][idx(i, 1, k)] = data[t][idx(i, dim_loc_size[1] - 2, k)];
                }
            }
        } else {
            double send_buffer[dim_loc_size[0] * dim_loc_size[2]];
            double recv_buffer[dim_loc_size[0] * dim_loc_size[2]];
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                        send_buffer[i * dim_loc_size[2] + k] 
                            = data[t][idx(i, dim_loc_size[1] - 2, k)];
                }
            }
            MPI_Sendrecv(
                send_buffer, dim_loc_size[0] * dim_loc_size[2], 
                MPI_DOUBLE, rank_dest[1], 1,
                recv_buffer, dim_loc_size[0] * dim_loc_size[2],
                MPI_DOUBLE, rank_source[1], 1,
                grid_comm, &status);
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                    if (bfisrt[1]) {
                        data[t][idx(i, 1, k)]
                            = recv_buffer[i * dim_loc_size[2] + k];  
                    } else {
                        data[t][idx(i, 0, k)]
                            = recv_buffer[i * dim_loc_size[2] + k];
                    }
                }
            }
        }

        if (bfisrt[1] && blast[1]) {
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                    data[t][idx(i, dim_loc_size[1] - 1, k)] = data[t][idx(i, 2, k)];
                }
            }
        } else {
            double send_buffer[dim_loc_size[0] * dim_loc_size[2]];
            double recv_buffer[dim_loc_size[0] * dim_loc_size[2]];
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                    if (bfisrt[1]) {
                        send_buffer[i * dim_loc_size[2] + k] 
                            = data[t][idx(i, 2, k)];
                    } else {
                        send_buffer[i * dim_loc_size[2] + k] 
                            = data[t][idx(i, 1, k)];
                    }
                }
            }
            MPI_Sendrecv(
                send_buffer, dim_loc_size[0] * dim_loc_size[2], 
                MPI_DOUBLE, rank_source[1], 1,
                recv_buffer, dim_loc_size[0] * dim_loc_size[2],
                MPI_DOUBLE, rank_dest[1], 1,
                grid_comm, &status);
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int k = 0; k < dim_loc_size[2]; k++) {
                    data[t][idx(i, dim_loc_size[1] - 1, k)]
                        = recv_buffer[i * dim_loc_size[2] + k];
                }
            }
        }

        if (bfisrt[2] && blast[2]) {
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int j = 0; j < dim_loc_size[1]; j++) {
                    data[t][idx(i, j, 1)] = data[t][idx(i, j, dim_loc_size[2] - 2)];
                }
            }
        } else {
            double send_buffer[dim_loc_size[0] * dim_loc_size[1]];
            double recv_buffer[dim_loc_size[0] * dim_loc_size[1]];
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int j = 0; j < dim_loc_size[1]; j++) {
                    send_buffer[i * dim_loc_size[1] + j] 
                        = data[t][idx(i, j, dim_loc_size[2] - 2)];
                }
            }
            MPI_Sendrecv(
                send_buffer, dim_loc_size[0] * dim_loc_size[1], 
                MPI_DOUBLE, rank_dest[2], 1,
                recv_buffer, dim_loc_size[0] * dim_loc_size[1],
                MPI_DOUBLE, rank_source[2], 1,
                grid_comm, &status);
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int j = 0; j < dim_loc_size[1]; j++) {
                    if (bfisrt[2]) {
                        data[t][idx(i, j, 1)]
                            = recv_buffer[i * dim_loc_size[1] + j];
                    } else {
                        data[t][idx(i, j, 0)]
                            = recv_buffer[i * dim_loc_size[1] + j];
                    }
                }
            }
        }

        if (bfisrt[2] && blast[2]) {
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int j = 0; j < dim_loc_size[1]; j++) {
                    data[t][idx(i, j, dim_loc_size[2] - 1)] = data[t][idx(i, j, 2)];
                }
            }
        } else {
            double send_buffer[dim_loc_size[0] * dim_loc_size[1]];
            double recv_buffer[dim_loc_size[0] * dim_loc_size[1]];
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int j = 0; j < dim_loc_size[1]; j++) {
                    if (bfisrt[1]) {
                        send_buffer[i * dim_loc_size[1] + j] 
                            = data[t][idx(i, j, 2)];
                    } else {
                        send_buffer[i * dim_loc_size[1] + j] 
                            = data[t][idx(i, j, 1)];
                    }
                }
            }
            MPI_Sendrecv(
                send_buffer, dim_loc_size[0] * dim_loc_size[1], 
                MPI_DOUBLE, rank_source[2], 1,
                recv_buffer, dim_loc_size[0] * dim_loc_size[1],
                MPI_DOUBLE, rank_dest[2], 1,
                grid_comm, &status);
            for (int i = 0; i < dim_loc_size[0]; i++) {
                for (int j = 0; j < dim_loc_size[1]; j++) {
                    data[t][idx(i, j, dim_loc_size[2] - 1)]
                        = recv_buffer[i * dim_loc_size[1] + j];
                }
            }
        }

        if (bcond_type[0] == 0) {
            if (bfisrt[0]) {
                for (int j = 0; j < dim_loc_size[1]; j++) {
                    for (int k = 0; k < dim_loc_size[2]; k++) {
                        // data[t][idx(0, j, k)] = 0;
                        data[t][idx(1, j, k)] = 0;
                    }
                }
            }
            if (blast[0]) {
                for (int j = 0; j < dim_loc_size[1]; j++) {
                    for (int k = 0; k < dim_loc_size[2]; k++) {
                        // data[t][idx(dim_loc_size[0] - 1, j, k)] = 0;
                        data[t][idx(dim_loc_size[0] - 2, j, k)] = 0;
                    }
                }
            }
        }

        // if (bcond_type[1] == 0) {
        //     if (bfisrt[1]) {
        //         for (int i = 0; i < dim_loc_size[0]; i++) {
        //             for (int k = 0; k < dim_loc_size[2]; k++) {
        //                 data[t][idx(i, 0, k)] = 0;
        //                 data[t][idx(i, 1, k)] = 0;
        //             }
        //         }
        //     }
        //     if (blast[1]) {
        //         for (int i = 0; i < dim_loc_size[0]; i++) {
        //             for (int k = 0; k < dim_loc_size[2]; k++) {
        //                 data[t][idx(i, dim_loc_size[1] - 1, k)] = 0;
        //                 data[t][idx(i, dim_loc_size[1] - 2, k)] = 0;
        //             }
        //         }
        //     }
        // }

        // if (bcond_type[2] == 0) {
        //     if (bfisrt[2]) {
        //         for (int i = 0; i < dim_loc_size[0]; i++) {
        //             for (int j = 0; j < dim_loc_size[1]; j++) {
        //                 data[t][idx(i, j, 0)] = 0;
        //                 data[t][idx(i, j, 1)] = 0;
        //             }
        //         }
        //     }
        //     if (blast[2]) {
        //         for (int i = 0; i < dim_loc_size[0]; i++) {
        //             for (int j = 0; j < dim_loc_size[1]; j++) {
        //                 data[t][idx(i, j, dim_loc_size[2] - 1)] = 0;
        //                 data[t][idx(i, j, dim_loc_size[2] - 2)] = 0;
        //             }
        //         }
        //     }
        // }

        max_error = 0;
        double max_true = 0;
        double max_pred = 0;
        int max_i = 0;
        int max_j = 0;
        int max_k = 0;
#pragma omp parallel for collapse(3) num_threads(4)
        for (int i = 1; i < dim_loc_size[0] - 1; i++) {
            for (int j = 1; j < dim_loc_size[1] - 1; j++) {
                for (int k = 1; k < dim_loc_size[2] - 1; k++) {
                    double u_true = u(
                        x(first_idx[0] + i - 1),
                        y(first_idx[1] + j - 1),
                        z(first_idx[2] + k - 1),
                        t * tau
                    );
#pragma omp critical
                    {
                        double err = fabs(u_true - data[t][idx(i, j, k)]);
                        if (err > max_error) {
                            max_error = err;
                            max_i = i;
                            max_j = j;
                            max_k = k;
                            max_true = u_true;
                            max_pred = data[t][idx(i, j, k)];
                        }
                    }
                }
            }
        }
        double temp_error = max_error;
        MPI_Reduce(&temp_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, grid_comm);
        if (!rank) {
            // cout << first_idx[0] << ' ' 
            //     << dim_loc_size[0] << ' ' 
            //     << max_true << ' ' 
            //     << max_pred << ' '
            //     << u(63. * (1. / 63.), 16. * (1. / 63.), 16. * (1. / 63.), t*tau) << ' '
            //     << sin(3 * pi * 63. * (1. / 63.) / 1.) << ' '
            //     << endl;
            cout << "T_STEP: " << t << endl;
            cout << "ERROR: " << max_error << endl;
            cout << "TIME: " << MPI_Wtime() - t_start << endl;
            // if (t == t_steps-1) {
            //     ofstream out_pred;
            //     ofstream out_true;
            //     ofstream out_err;
            //     out_pred.open("pred.txt");
            //     out_true.open("true.txt");
            //     out_err.open("err.txt");
            //     for (int i = 1; i < dim_loc_size[0] - 1; i++) {
            //         for (int j = 1; j < dim_loc_size[1] - 1; j++) {
            //             for (int k = 1; k < dim_loc_size[2] - 1; k++) {
            //                 double u_true = u(
            //                     x(first_idx[0] + i - 1),
            //                     y(first_idx[1] + j - 1),
            //                     z(first_idx[2] + k - 1),
            //                     t * tau
            //                 );
            //                 out_pred << data[t][idx(i, j, k)] << '\n';
            //                 out_true << u_true << '\n';
            //                 out_err << fabs(data[t][idx(i, j, k)] - u_true) << '\n';
            //             }
            //         }
            //     }
            // }
        }
        MPI_Barrier(grid_comm);
        for (int p = 0; p < size; p++) {
            if (rank == p) {
                cout
                << " RANK: " << rank 
                << " i: " << max_i
                << " j: " << max_j
                << " k: " << max_k
                << endl;
            }
            MPI_Barrier(grid_comm);
        }
    }

    for (int t = 0; t < t_steps; t++) {
        delete data[t];
    }

    MPI_Finalize();

    return 0;
}