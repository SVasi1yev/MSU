#include <pthread.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <time.h>
#include <numeric>

using namespace std;

pthread_mutex_t lock;

struct args {
    long rank;
    long size;
    int low_board;
    int high_board;
    vector<int>* primes1;
    vector<int>* primes2;
    double* times;
};

void* thread (void* ptr) {
    struct timespec begin, end;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &begin);

    struct args* temp = (struct args*)ptr;
    int d = temp->high_board - temp->low_board;
    int interval = d / temp->size;

    if (temp->rank == (temp->size - 1)) {
        vector<bool> mask2(d - temp->rank * interval);
        fill(mask2.begin(), mask2.end(), true);
        for (auto x : *(temp->primes1)) {
            for (int i = (temp->low_board + temp->rank * interval) % x ? x - (temp->low_board + temp->rank * interval) % x : 0; i < mask2.size(); i += x) {
                mask2[i] = false;
            }
        }
        for (int i = 0; i < mask2.size(); i++) {
            if (mask2[i]) {
                pthread_mutex_lock(&lock);
                (temp->primes2)->push_back(i + temp->low_board + temp->rank * interval);
                pthread_mutex_unlock(&lock);
            }
        }
    } else {
        vector<bool> mask2(interval);
        fill(mask2.begin(), mask2.end(), true);
        for (auto x : *(temp->primes1)) {
            for (int i = (temp->low_board + temp->rank * interval) % x ? x - (temp->low_board + temp->rank * interval) % x : 0; i < mask2.size(); i += x) {
                mask2[i] = false;
            }
        }
        for (int i = 0; i < mask2.size(); i++) {
            if (mask2[i]) {
                pthread_mutex_lock(&lock);
                (temp->primes2)->push_back(i + temp->low_board + temp->rank * interval);
                pthread_mutex_unlock(&lock);
            }
        }
    }

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);

    (temp->times)[temp->rank] = (end.tv_sec - begin.tv_sec) + 1e-9 * (end.tv_nsec - begin.tv_nsec);

    return NULL;
}

int main (int argc, char** argv) {
    struct timespec begin, end;
    long size = strtol(argv[5], NULL, 10);
    double times[size];
    double times_sum = 0;
    double max_time = 0;
    int low_board = strtol(argv[1], NULL, 10);
    int high_board = strtol(argv[2], NULL, 10);
    int root = (int)pow(high_board, 0.5) + 1;
    int new_low_board = max(low_board, root);
    int d = high_board - new_low_board;
    int interval = d / size;

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &begin);

    vector<int> mask1(root - 2);
    vector<int> primes1;
    vector<int> primes2;
    fill(mask1.begin(), mask1.end(), true);

    for (int i = 2; i < (int)pow(root, 0.5) + 1; i++) {
        if (mask1[i - 2]) {
            primes1.push_back(i);
            if (i >= low_board) {
                primes2.push_back(i);
            }
            for (int j = i - 2; j < root - 2; j += i) {
                mask1[j] = false;
            }
        }
    }

    for (int i = (int)pow(root, 0.5) + 1; i < root; i++) {
        if (mask1[i - 2]) {
            primes1.push_back(i);
            if (i >= low_board) {
                primes2.push_back(i);
            }
        }
    }

    pthread_t threads[size - 1];
    pthread_mutex_init(&lock, NULL);

    struct args a[size - 1];
    for (int i = 0; i < size - 1; i++) {
        a[i] = {i + 1, size, new_low_board, high_board, &primes1, &primes2, times};
    }

    for (long id = 0; id < size - 1; id++) {
        pthread_create(threads + id, NULL, thread, a + id);
    }

    vector<bool> mask2(interval);
    fill(mask2.begin(), mask2.end(), true);

    for (auto x : primes1) {
        for (int i = new_low_board % x ? x - new_low_board % x : 0; i < mask2.size(); i += x) {
            mask2[i] = false;
        }
    }

    for (int i = 0; i < mask2.size(); i++) {
        if (mask2[i]) {
            pthread_mutex_lock(&lock);
            primes2.push_back(i + new_low_board);
            pthread_mutex_unlock(&lock);
        }
    }

    for (long i = 0; i < size - 1; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_mutex_destroy(&lock);

    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
    times[0] = (end.tv_sec - begin.tv_sec) + 1e-9 * (end.tv_nsec - begin.tv_nsec);

    cout << primes2.size() << endl;

    ofstream output_file;
    output_file.open(argv[3]);
    for (auto x : primes2) {
        output_file << x << ' ';
    }
    output_file.close();

    for (int i = 0; i < size; i++) {
        times_sum += times[i];
        if (times[i] > max_time) {
            max_time = times[i];
        }
    }

    output_file.open(argv[4], ios_base::app | ios_base::out);
    output_file << size << '\t' << max_time << '\t' << times_sum << '\n';
    output_file.close();

    return 0;
}