#include "Reflector.h"

Reflector::Reflector(int n, int mpi_rank, int mpi_size):
    n(n), mpi_rank(mpi_rank), mpi_size(mpi_size) {

    last_process = mpi_rank == (mpi_size - 1);

    if (mpi_rank < (n % mpi_size)) {
        local_size = n / mpi_size + 1;
    } else {
        local_size = n / mpi_size;
    }

    matrix = new double*[local_size];
    x = new double[local_size];

    for (int i = 0; i < local_size; i++) {
        matrix[i] = new double[n];
        for (int j = 0; j < n; j++) {
            //TODO initialization
            matrix[i][j] = init_matr_func(i, j);
        }
        x[i] = 0;
    }

    if (last_process) {
        b = new double[n];
        for (int i = 0; i < n; i++) {
            double part = 0;
            double sum;
            for (int j = 0; j < local_size; j++) {
                part += matrix[j][i];
            }
            MPI_Reduce(&part, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_size - 1, MPI_COMM_WORLD);
            b[i] = sum;
        }
    } else {
        for (int i = 0; i < n; i++) {
            double part = 0;
            double sum;
            for (int j = 0; j < local_size; j++) {
                part += matrix[j][i];
            }
            MPI_Reduce(&part, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_size - 1, MPI_COMM_WORLD);
        }
    }

    const int digits = std::numeric_limits<double>::digits10;
//    std::cout << std::setfill(' ') << std::setw(digits);
    std::cout << std::fixed << std::setprecision(digits - 11);
}

bool Reflector::needProcessColumn(int i) {
    char res;
    if (!myColumn(i)) {
        MPI_Bcast(&res, 1, MPI_UNSIGNED_CHAR, glob2procNum(i), MPI_COMM_WORLD);
    } else {
        res = 0;
        for (int j = i + 1; j < n; j++) {
            if (matrix[glob2loc(i)][j] != 0) {
                res = 1;
                break;
            }
        }
        MPI_Bcast(&res, 1, MPI_UNSIGNED_CHAR, glob2procNum(i), MPI_COMM_WORLD);
    }
    return (res == 0) ? false : true;
}

double* Reflector::conctructXiUpdateAi(int i) {
    double* x_i = new double[n - i];
    if (!myColumn(i)) {
        MPI_Bcast(x_i, n - i, MPI_DOUBLE, glob2procNum(i), MPI_COMM_WORLD);
    } else {
        double s_i = 0;
        for (int j = i + 1; j < n; j++) {
            s_i += matrix[glob2loc(i)][j] * matrix[glob2loc(i)][j];
        }
        double norm_a_i = sqrt(s_i + matrix[glob2loc(i)][i] * matrix[glob2loc(i)][i]);
        x_i[0] = matrix[glob2loc(i)][i] - norm_a_i;
        matrix[glob2loc(i)][i] = norm_a_i;
        for (int j = i + 1; j < n; j++) {
            x_i[j - i] = matrix[glob2loc(i)][j];
            matrix[glob2loc(i)][j] = 0;
        }

        double norm_x_i = sqrt(s_i + x_i[0] * x_i[0]);
        for (int j = 0; j < (n - i); j++) {
            x_i[j] /= norm_x_i;
        }

        MPI_Bcast(x_i, n - i, MPI_DOUBLE, glob2procNum(i), MPI_COMM_WORLD);
    }

    return x_i;
}

void Reflector::updateRestMatrix(const double* x, int i) {
    for (int j = 0; j < local_size; j++) {
        if (loc2glob(j) <= i) {
            continue;
        }
        // (I - 2xx^t)a = a - 2(x, a)x
        double* a = matrix[j] + i;
        double prod = a[0] * x[0];
        for (int k = 1; k < (n - i); k++) {
            prod += a[k] * x[k];
        }
        prod *= 2;
        for (int k = 0; k < (n - i); k++) {
            a[k] -= prod * x[k];
        }
    }

    if (last_process) {
        double* a = b + i;
        double prod = 0;
        for (int k = 0; k < (n - i); k++) {
            prod += a[k] * x[k];
        }
        for (int k = 0; k < (n - i); k++) {
            a[k] -= 2 * prod * x[k];
        }
    }
}

void Reflector::countX() {
    for (int i = n - 1; i >= 0; i--) {
        double part = 0;
        for (int j = 0; j < local_size; j++) {
            part -= matrix[j][i] * x[j];
        }
        if (last_process) {
            part += b[i];
        }

        double full;
        MPI_Reduce(
            &part, &full, 1, MPI_DOUBLE, MPI_SUM,
            glob2procNum(i), MPI_COMM_WORLD
        );
        if (myColumn(i)) {
            x[glob2loc(i)] = full / matrix[glob2loc(i)][i];
        }
    }
}

void Reflector::start() {
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for (int i = 0; i < n; i++) {
        if (!needProcessColumn(i)) { continue; }
        double* x_i = conctructXiUpdateAi(i);
        updateRestMatrix(x_i, i);

        delete[] x_i;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime() - t1;

    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    countX();
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime() - t2;
}

void Reflector::printMatrix() {
    if (mpi_rank == 0) {
        double** temp_matr = new double*[n];
        for (int i = 0; i < n; i++) {
            temp_matr[i] = new double[n];
            if (myColumn(i)) {
                std::copy(matrix[glob2loc(i)], matrix[glob2loc(i)] + n, temp_matr[i]);
            } else {
                MPI_Status status;
                MPI_Recv(
                    temp_matr[i], n, MPI_DOUBLE,
                    glob2procNum(i), 0, MPI_COMM_WORLD, &status
                );
            }
        }

        double* temp_b = new double[n];
        if (last_process) {
            std::copy(b, b + n, temp_b);
        } else {
            MPI_Status status;
            MPI_Recv(
                    temp_b, n, MPI_DOUBLE,
                    mpi_size - 1, 0, MPI_COMM_WORLD, &status
            );
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << temp_matr[j][i] << "\t\t";
            }
            std::cout << "||\t\t" << temp_b[i] << std::endl;
        }
        std::cout << std::endl;

        for (int i = 0; i < n; i++) {
            delete[] temp_matr[i];
        }
        delete[] temp_matr;
        delete[] temp_b;
    } else {
        for (int i = 0; i < local_size; i++) {
            MPI_Send(matrix[i], n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        if (last_process) {
            MPI_Send(b, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

void Reflector::printX() {
    if (mpi_rank != 0) {
        MPI_Send(x, local_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        double *temp_x = new double[n];
        for (int i = 0; i < local_size; i++) {
            temp_x[i * mpi_size] = x[i];
        }

        double *temp_part = new double[local_size];
        for (int i = 1; i < mpi_size; i++) {
            MPI_Status status;
            MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
            int part_size;
            MPI_Get_count(&status, MPI_DOUBLE, &part_size);
            MPI_Recv(temp_part, part_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            for (int j = 0; j < part_size; j++) {
                temp_x[j * mpi_size + i] = temp_part[j];
            }
        }

        std::cout << "X:";
        for (int i = 0; i < n; i++) {

            std::cout << temp_x[i] << "\t";
        }
        std::cout << std::endl;

        delete[] temp_x;
        delete[] temp_part;
    }
}

void Reflector::printResidual() {
    double** init_matrix = new double*[local_size];
    for (int i = 0; i < local_size; i++) {

        init_matrix[i] = new double[n];

        for (int j = 0; j < n; j++) {
            //TODO initialization
            init_matrix[i][j] = init_matr_func(i, j);
        }
    }

    double* init_b;
    if (last_process) {
        init_b = new double[n];
        for (int i = 0; i < n; i++) {
            double part = 0;
            double sum;
            for (int j = 0; j < local_size; j++) {
                part += init_matrix[j][i];
            }
            MPI_Reduce(&part, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_size - 1, MPI_COMM_WORLD);
            init_b[i] = sum;
        }
    } else {
        for (int i = 0; i < n; i++) {
            double part = 0;
            double sum;
            for (int j = 0; j < local_size; j++) {
                part += init_matrix[j][i];
            }
            MPI_Reduce(&part, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_size - 1, MPI_COMM_WORLD);
        }
    }

    double* part = new double[n];
    for (int i = 0; i < n; i++) {
        part[i] = 0;
        for (int j = 0; j < local_size; j++) {
            part[i] += init_matrix[j][i] * x[j];
        }
    }

    double* full;
    if (last_process) {
        full = new double[n];
    }

    MPI_Reduce(part, full, n, MPI_DOUBLE, MPI_SUM, mpi_size - 1, MPI_COMM_WORLD);

    if (last_process) {
        double residual = 0;
        for (int i = 0; i < n; i++) {
            residual += (full[i] - init_b[i]) * (full[i] - init_b[i]);
        }
        residual = sqrt(residual);
        std::cout << "Residual: " << std::scientific << residual << std::endl;
    }

    for (int i = 0; i < local_size; i++) {
        delete[] init_matrix[i];
    }
    delete[] init_matrix;
    delete[] part;
    if (last_process) {
        delete[] init_b;
        delete[] full;
    }
}

void Reflector::printDiff(const double *right_x) {
    double part = 0;
    for (int i = 0; i < local_size; i++) {
        part += (x[i] - right_x[i]) * (x[i] - right_x[i]);
    }

    double full;

    MPI_Reduce(&part, &full, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    full = sqrt(full);
    if (mpi_rank == 0) {
        std::cout << "Diff: " << full << std::endl;
    }
}

Reflector::~Reflector() {

    for (int i = 0; i < local_size; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] x;
    if (last_process) {
        delete[] b;
    }

}
