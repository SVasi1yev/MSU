#include <iostream>
#include <cstdlib>
#include <mpi/mpi.h>
#include <math.h>
#include <vector>
#include <fstream>


using namespace std;


int main(int argc, char **argv){
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int low_board = strtol(argv[1], NULL, 10);
    int high_board = strtol(argv[2], NULL, 10);
    int root = (int)pow(high_board, 0.5) + 1;
    int init_low_board = low_board;
    low_board = max(low_board, root);
    int d = high_board - low_board;
    int interval = d / size;

    if (!rank) {
        vector<int> mask1(root - 2);
        vector<int> primes1;
        fill(mask1.begin(), mask1.end(), true);
        auto* times = new double[size];

        times[0] = MPI_Wtime();

        for (int i = 2; i < (int)pow(root, 0.5) + 1; i++) {
            if (mask1[i - 2]) {
                primes1.push_back(i);
                for (int j = i - 2; j < root - 2; j += i) {
                    mask1[j] = false;
                }
            }
        }
        for (int i = (int)pow(root, 0.5) + 1; i < root; i++) {
            if (mask1[i - 2]) {
                primes1.push_back(i);
            }
        }

        vector<bool> mask2(interval);
        vector<int> primes2;
        fill(mask2.begin(), mask2.end(), true);

        for (auto x : primes1) {
        	if (x >= init_low_board) {
        		primes2.push_back(x);
        	}
            for (int i = low_board % x ? x - low_board % x : 0; i < mask2.size(); i += x) {
                mask2[i] = false;
            }
        }

        for (int i = 0; i < mask2.size(); i++) {
            if (mask2[i]) {
                primes2.push_back(i + low_board);
            }
        }

        int** other_primes = new int*[size - 1];
        int* buf_sizes = new int[size - 1];
        MPI_Status status;
        for (int i = 1; i < size; i ++) {
            MPI_Probe(i, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &buf_sizes[i - 1]);
            other_primes[i - 1] = new int[buf_sizes[i - 1]];
            MPI_Recv(other_primes[i - 1], buf_sizes[i - 1], MPI_INT, i, 0, MPI_COMM_WORLD, &status);
        }

        times[0] = MPI_Wtime() - times[0];

        for (int i = 1; i < size; i ++) {
            MPI_Recv(&times[i], 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
        }

        double times_sum = 0;
        double max_time = 0;
        for (int i = 0; i < size; i++) {
            if (times[i] > max_time) {
                max_time = times[i];
            }
            times_sum += times[i];
        }

        /*for (auto x : primes1) {
            cout << x << ' ';
        }
        cout << endl;
        for (auto x : primes2) {
            cout << x << ' ';
        }
        cout << endl;
        for (int i = 0; i < size - 1; i++) {
            for (int j = 0; j < buf_sizes[i]; j++) {
                cout << other_primes[i][j] << ' ';
            }
            cout << endl;
        }*/

        int sum = primes2.size();
        for (int i = 0; i < size - 1; i++) {
            sum += buf_sizes[i];
        }
        cout << sum << endl;

        ofstream output_file;
        output_file.open(argv[3]);
        for (auto x : primes2) {
            output_file << x << ' ';
        }
        for (int i = 0; i < size - 1; i++) {
            for (int j = 0; j < buf_sizes[i]; j++) {
                output_file << other_primes[i][j] << ' ';
            }
        }
        output_file.close();

        output_file.open(argv[4], ios_base::app | ios_base::out);
        output_file << size << '\t' << max_time << '\t' << times_sum << '\n';
        output_file.close();

        delete[] buf_sizes;
        for (int i = 0; i < size - 1; i++) {
            delete[] other_primes[i];
        }
        delete[] other_primes;
    } else {
        double time;
        vector<bool> mask1(root - 2);
        vector<int> primes1;
        fill(mask1.begin(), mask1.end(), true);

        time = MPI_Wtime();

        for (int i = 2; i < (int)pow(root, 0.5) + 1; i++) {
            if (mask1[i - 2]) {
                primes1.push_back(i);
                for (int j = i - 2; j < root - 2; j += i) {
                    mask1[j] = false;
                }
            }
        }
        for (int i = (int)pow(root, 0.5) + 1; i < root; i++) {
            if (mask1[i - 2]) {
                primes1.push_back(i);
            }
        }

        vector<int> primes2;
        vector<bool> mask2;
        if (rank == (size - 1)) {
            mask2.resize(d - rank * (d / size));
            fill(mask2.begin(), mask2.end(), true);
            for (auto x : primes1) {
                for (int i = (low_board + rank * interval) % x ? x - (low_board + rank * interval) % x : 0; i < mask2.size(); i += x) {
                    mask2[i] = false;
                }
            }
            for (int i = 0; i < mask2.size(); i++) {
                if (mask2[i]) {
                    primes2.push_back(i + low_board + rank * interval);
                }
            }
        } else {
            mask2.resize(interval);
            fill(mask2.begin(), mask2.end(), true);
            for (auto x : primes1) {
                for (int i = (low_board + rank * interval) % x ? x - (low_board + rank * interval) % x : 0; i < mask2.size(); i += x) {
                    mask2[i] = false;
                }
            }
            for (int i = 0; i < mask2.size(); i++) {
                if (mask2[i]) {
                    primes2.push_back(i + low_board + rank * interval);
                }
            }
        }

        time = MPI_Wtime() - time;
        MPI_Send(&primes2[0], primes2.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&time, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
