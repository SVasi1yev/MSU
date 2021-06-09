#include <iostream>
#include <mpi/mpi.h>
#include <fstream>


using namespace std;


double *multiplic(double *matrix, double* vector, size_t n, size_t m) {
    double *c = new double[n];
    fill_n(c, n, 0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            c[i] += matrix[i * m + j] * vector[j];
        }
    }

    return c;
}


int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *c0;

    if (!rank) {
        ifstream matrix_file(argv[1], ios::binary | ios::in);
        char m_type; size_t m_rows; size_t m_cols;
        matrix_file.read(&m_type, sizeof(char));
        matrix_file.read((char*) &m_rows, sizeof(size_t));
        matrix_file.read((char*) &m_cols, sizeof(size_t));
        double *matrix = new double[m_rows * m_cols];
        matrix_file.read((char*) matrix, m_rows * m_cols * sizeof(double));

        ifstream vector_file(argv[2], ios::binary | ios::in);
        char v_type; size_t v_rows; size_t v_cols;
        vector_file.read(&v_type, sizeof(char));
        vector_file.read((char*) &v_rows, sizeof(size_t));
        vector_file.read((char*) &v_cols, sizeof(size_t));
        double *vector = new double[v_rows * v_cols];
        vector_file.read((char*) vector, v_rows * v_cols * sizeof(double));

        c0 = multiplic(matrix, vector, m_rows, m_cols);

        matrix_file.close();
        vector_file.close();
    }

    double time;

    ifstream A_file(argv[1], ios::binary | ios::in);
    char A_type; size_t A_rows; size_t A_cols;
    A_file.read(&A_type, sizeof(char));
    A_file.read((char*) &A_rows, sizeof(size_t));
    A_file.read((char*) &A_cols, sizeof(size_t));
    double *A;

    ifstream b_file(argv[2], ios::binary | ios:: in);
    char b_type; size_t b_rows; size_t b_cols;
    b_file.read(&b_type, sizeof(char));
    b_file.read((char*) &b_rows, sizeof(size_t));
    b_file.read((char*) &b_cols, sizeof(size_t));
    double *b;

    double *c;
    if (!rank) {
        c = new double[A_rows];
    }

    if (A_rows > A_cols) {
        size_t rows_num;
        if (rank == size - 1) {
            rows_num = A_rows - rank * (A_rows / size);
        } else {
            rows_num = A_rows / size;
        }

        A_file.seekg(A_cols * rank * (A_rows / size) * sizeof(double), ios::cur);
        A = new double[A_cols * rows_num];
        A_file.read((char*) A, A_cols * rows_num * sizeof(double));
        b = new double[b_rows];        
        b_file.read((char*) b, b_rows * sizeof(double));

	    MPI_Barrier(MPI_COMM_WORLD);
        if (!rank) {
            time = MPI_Wtime();
        }

        double *c_part = multiplic(A, b, rows_num, A_cols);

        A_file.close();
        b_file.close();
        delete[] A;
        delete[] b;

        int *recv_counts = new int[size];
        fill_n(recv_counts, size - 1, A_rows / size);
        recv_counts[size - 1] = A_rows - (size - 1) * (A_rows / size);
        int *displs = new int[size];
        fill_n(displs, size, 0);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < i; j++) {
                displs[i] += recv_counts[j];
            }
        }
        MPI_Gatherv(c_part, rows_num, MPI_DOUBLE, c, recv_counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        delete[] c_part;
        delete[] recv_counts;
        delete[] displs;
    } else {
        size_t cols_num;
        if (rank == size - 1) {
            cols_num = A_cols - rank * (A_cols / size);
        } else {
            cols_num = A_cols / size;
        }

        A = new double[A_rows * cols_num];
        double *temp = A;
        for (int i = 0; i < A_rows; i++) {
            A_file.seekg(sizeof(char) + 2 * sizeof(size_t) + (A_cols * i + rank * (A_cols / size)) * sizeof(double), ios::beg);
            A_file.read((char*) temp, cols_num * sizeof(double));
            temp = temp + cols_num;
        }
        b = new double[cols_num];      
        b_file.seekg(rank * (b_rows / size) * sizeof(double), ios::cur);  
        b_file.read((char*) b, cols_num * sizeof(double));

	    MPI_Barrier(MPI_COMM_WORLD);
        if (!rank) {
            time = MPI_Wtime();
        }

        double *c_part = multiplic(A, b, A_rows, cols_num);

        A_file.close();
        b_file.close();
        delete[] A;
        delete[] b;

        MPI_Reduce(c_part, c, A_rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        delete[] c_part;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) {
        time = MPI_Wtime() - time;
        cout << "time -- " << time << endl;
        for (int i = 0; i < A_rows; i++) {
            if (c0[i] != c[i]) {
                cout << "error -- " << c[i] << " -- " << c0[i] << endl;
            } else {
                cout << "OK -- " << c[i] << " -- " << c0[i] << endl;
            }
        }

        ofstream c_file(argv[3], ios::binary | ios::out | ios::trunc);
        c_file.write(&A_type, sizeof(char));
        c_file.write((char*) &A_rows, sizeof(size_t));
        c_file.write((char*) &b_cols, sizeof(size_t));
        c_file.write((char*) c, A_rows * sizeof(double));

        c_file.close();
        delete[] c;
    }


    MPI_Finalize();

    return 0;
}
