#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <math.h>
#include <fstream>


using namespace std;


int main(int argc, char **argv){
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    MPI_Datatype filetype;
    MPI_Offset disp;

    double time_start, time_finish, time = 0;
    double file_time_start, file_time_finish, file_time = 0;

    const int cube_ndims = 3;
    int cube_dims[cube_ndims];
    int cube_side = round(pow(size, (double)1 / 3));
if (rank < cube_side * cube_side * cube_side) {
    // if (!rank) {
    //     cout << "cube_side -- " << cube_side << endl;
    // }

    fill_n(cube_dims, cube_ndims, cube_side);
    int cube_periods[cube_ndims] = {0, 0, 0};
    int cube_reorder = 0;
    MPI_Comm cube_comm;
    MPI_Cart_create(MPI_COMM_WORLD, cube_ndims, cube_dims, cube_periods, cube_reorder, &cube_comm);
    int cube_coords[cube_ndims];
    MPI_Cart_coords(cube_comm, rank, cube_ndims, cube_coords);

    // cout << cube_coords[0] << " -- " << cube_coords[1] << " -- " << cube_coords[2] << " -- " << rank << endl;

    MPI_Comm i_line_comm;
    MPI_Comm j_line_comm;
    MPI_Comm k_line_comm;
    MPI_Comm ij_flat_comm;

    int remain_dims[] = {1, 0, 0};
    MPI_Cart_sub(cube_comm, remain_dims, &i_line_comm);

    remain_dims[0] = 0; remain_dims[1] = 1;
    MPI_Cart_sub(cube_comm, remain_dims, &j_line_comm);

    remain_dims[1] = 0; remain_dims[2] = 1;
    MPI_Cart_sub(cube_comm, remain_dims, &k_line_comm);

    remain_dims[2] = 0; remain_dims[0] = 1; remain_dims[1] = 1;
    MPI_Cart_sub(cube_comm, remain_dims, &ij_flat_comm);

    MPI_Barrier(cube_comm);
    file_time_start = MPI_Wtime();
    int N = 0;
    if (!rank) {
        ifstream A_file(argv[1], ios::binary | ios::in);
        char A_type;
        A_file.read(&A_type, sizeof(char));
        A_file.read((char*) &N, sizeof(size_t));
        A_file.close();
    }
    MPI_Barrier(cube_comm);
    file_time_finish = MPI_Wtime();
    file_time += file_time_finish - file_time_start;

    MPI_Bcast(&N, 1, MPI_INT, 0, cube_comm);

    // cout << rank << " -- " << N << endl;

    MPI_Barrier(cube_comm);
    file_time_start = MPI_Wtime();

    int first_row, first_col, rows, cols;
    double *A_part;
    double *B_part;

    if (!cube_coords[2]) {
        if (cube_coords[0] < N % cube_side) {
            rows = N / cube_side + 1;
            first_row = rows * cube_coords[0];
        } else {
            rows = N / cube_side;
            first_row = (rows + 1) * (N % cube_side) + rows * (cube_coords[0] - N % cube_side);
        }
        if (cube_coords[1] < N % cube_side) {
            cols = N / cube_side + 1;
            first_col = cols * cube_coords[1];
        } else {
            cols = N / cube_side;
            first_col = (cols + 1) * (N % cube_side) + cols * (cube_coords[1] - N % cube_side);
        }

        // cout << rank << " -- " << rows << " " << cols << " " << first_row << " " << first_col << endl;

        MPI_File A_file, B_file;
        disp = sizeof(char) + 2 * sizeof(int) + (first_row * N + first_col) * sizeof(double);
        int stride = N;
        MPI_Type_vector(rows, cols, stride, MPI_DOUBLE, &filetype); 
        MPI_Type_commit(&filetype);
        MPI_File_open(ij_flat_comm, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &A_file);
        MPI_File_open(ij_flat_comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &B_file);
        MPI_File_set_view(A_file, disp, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
        MPI_File_set_view(B_file, disp, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);

        A_part = new double[rows * cols];
        B_part = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            MPI_File_read(A_file, A_part + i * cols, cols, MPI_DOUBLE, &status);
            MPI_File_read(B_file, B_part + i * cols, cols, MPI_DOUBLE, &status);
        }

        MPI_File_close(&A_file);
        MPI_File_close(&B_file);

        // if ((cube_coords[0] == 0) && (cube_coords[1] == 1)) {
        //     for (int i = 0; i < rows * cols; i++) {
        //         cout << A_part[i] << endl;
        //     }
        // }
    }

    MPI_Barrier(cube_comm);
    file_time_finish = MPI_Wtime();
    file_time += file_time_finish - file_time_start;

    MPI_Barrier(cube_comm);
    time_start = MPI_Wtime();

    int A_rows, A_cols, B_rows, B_cols;

    if (cube_coords[0] < N % cube_side) {
        A_rows = N / cube_side + 1;
    } else {
        A_rows = N / cube_side;
    }
    if (cube_coords[2] < N % cube_side) {
        A_cols = N / cube_side + 1;
    } else {
        A_cols = N / cube_side; 
    }

    if (cube_coords[2] < N % cube_side) {
        B_rows = N / cube_side + 1;
    } else {
        B_rows = N / cube_side;
    }
    if (cube_coords[1] < N % cube_side) {
        B_cols = N / cube_side + 1;
    } else {
        B_cols = N / cube_side;
    }

    MPI_Barrier(cube_comm);

    if (!cube_coords[2] && (cube_coords[1] > 0)) {
        int dest_rank;
        int dest_coords[] = {cube_coords[0], cube_coords[1], cube_coords[1]};
        MPI_Cart_rank(cube_comm, dest_coords, &dest_rank);
        MPI_Send(A_part, rows * cols, MPI_DOUBLE, dest_rank, 1, cube_comm);
        delete[] A_part;
    }
    if (!cube_coords[2] && (cube_coords[0] > 0)) {
        int dest_rank;
        int dest_coords[] = {cube_coords[0], cube_coords[1], cube_coords[0]};
        MPI_Cart_rank(cube_comm, dest_coords, &dest_rank);
        MPI_Send(B_part, rows * cols, MPI_DOUBLE, dest_rank, 2, cube_comm);
        delete[] B_part;
    }

    if ((cube_coords[1] != 0) || (cube_coords[2] != 0)) {
        A_part = new double[A_rows * A_cols];
    }
    if ((cube_coords[0] != 0) || (cube_coords[2] != 0)) {
        B_part = new double[B_rows * B_cols];
    }

    if (cube_coords[2] && (cube_coords[1] == cube_coords[2])) {
        int source_rank;
        int source_coords[] = {cube_coords[0], cube_coords[1], 0};
        MPI_Cart_rank(cube_comm, source_coords, &source_rank);
        MPI_Recv(A_part, A_rows * A_cols, MPI_DOUBLE, source_rank, 1, cube_comm, &status);
    }
    if (cube_coords[2] && (cube_coords[0] == cube_coords[2])) {
        int source_rank;
        int source_coords[] = {cube_coords[0], cube_coords[1], 0};
        MPI_Cart_rank(cube_comm, source_coords, &source_rank);
        MPI_Recv(B_part, B_rows * B_cols, MPI_DOUBLE, source_rank, 2, cube_comm, &status);
    }

    MPI_Barrier(cube_comm);

    int source_rank;
    MPI_Cart_rank(j_line_comm, &cube_coords[2], &source_rank);
    MPI_Bcast(A_part, A_rows * A_cols, MPI_DOUBLE, source_rank, j_line_comm);

    MPI_Cart_rank(i_line_comm, &cube_coords[2], &source_rank);
    MPI_Bcast(B_part, B_rows * B_cols, MPI_DOUBLE, source_rank, i_line_comm);

    // if ((cube_coords[0] == 0) && (cube_coords[1] == 1) && (cube_coords[2] == 1)) {
    //     for (int i =0 ; i < B_rows * B_cols; i++) {
    //         cout << B_part[i] << endl;
    //     }
    // }

    double *C_part = new double[A_rows * B_cols];
    fill_n(C_part, A_rows * B_cols, 0);
    for (int i = 0; i < A_rows; i++) {
        for (int k = 0; k < A_cols; k++) {
            for (int j = 0; j < B_cols; j++) {
                C_part[i * B_cols + j] += A_part[i * A_cols + k] * B_part[k * B_cols + j];
            }
        }
    }

    double *recv;
    if (!cube_coords[2]) {
        recv = new double[A_rows * B_cols];
    }

    int temp = 0;
    MPI_Cart_rank(k_line_comm, &temp, &source_rank);
    MPI_Reduce(C_part, recv, A_rows * B_cols, MPI_DOUBLE, MPI_SUM, source_rank, k_line_comm);

    MPI_Barrier(cube_comm);
    time_finish = MPI_Wtime();
    time += time_finish - time_start;

    MPI_Barrier(cube_comm);
    file_time_start = MPI_Wtime();

    if (!cube_coords[2]) {
        MPI_File C_file;
        MPI_File_open(ij_flat_comm, argv[3], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &C_file);
        if (!rank) {
            char type = 'd';
            int C_rows = N;
            MPI_File_write(C_file, &type, 1, MPI_CHAR, &status);
            MPI_File_write(C_file, &C_rows, 1, MPI_INT, &status);
            MPI_File_write(C_file, &C_rows, 1, MPI_INT, &status);
        }
        MPI_File_set_view(C_file, disp, MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
        for(int i = 0; i < rows; ++i) {
            MPI_File_write(C_file, recv + i * cols, cols, MPI_DOUBLE, &status);
        }
        MPI_File_close(&C_file);
    } 

    MPI_Barrier(cube_comm);
    file_time_finish = MPI_Wtime();
    file_time += file_time_finish - file_time_start;

    if (!rank) {
        ofstream time_file;
        time_file.open("times.dat", ios::app);
        time_file << size << "\t" << N << "\t" << time << "\t" << file_time << endl;
    }   

    delete[] A_part;
    delete[] B_part;
    delete[] C_part;

    if (!cube_coords[2]) {
        delete[] recv;
        MPI_Type_free(&filetype);
        MPI_Comm_free(&ij_flat_comm);        
    }

    MPI_Comm_free(&i_line_comm);
    MPI_Comm_free(&j_line_comm);
    MPI_Comm_free(&k_line_comm);
    MPI_Comm_free(&cube_comm);
}
    MPI_Finalize();
    return 0;
}
