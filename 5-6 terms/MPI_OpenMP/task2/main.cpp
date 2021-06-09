#include <iostream>
#include <fstream>
#include <ctime>
#include "papi.h"


using namespace std;


const char* test(const float* matrix, const float* testMatrix, size_t dim) {
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if (matrix[i * dim + j] != testMatrix[i * dim + j]) {
                return "ERROR";
            }
        }
    }
    return "OK";
}


int main(int argc, char *argv[]) {
    const int optBlockSize = 70;
    const int defBlockSize = 32;

    auto deltaTimes = new double[3];
    fill_n(deltaTimes, 3, 0);
    auto L1Misses = new long long[3];
    fill_n(L1Misses, 3, 0);
    auto L2Misses = new long long[3];
    fill_n(L2Misses, 3, 0);
    auto cycles = new long long[3];
    fill_n(cycles, 3, 0);
    auto FLOPs = new long long[3];
    fill_n(FLOPs, 3, 0);

    clock_t startTime;
    int eventsNum = 3;
    int eventsNum2 = 1;
    long long values[3];
    long long values2[1];
    fill_n(values, 3, 0);
    fill_n(values2, 1, 0);
    int events[] = {PAPI_L1_DCM, PAPI_L2_DCM, PAPI_TOT_CYC};
    int events2[] = {PAPI_FP_OPS};
    PAPI_library_init(PAPI_VER_CURRENT);

    ifstream aInputFile(argv[1], ios::binary | ios::in);
    char aType; size_t aRows; size_t aCols;
    aInputFile.read(&aType, sizeof(char));
    aInputFile.read((char*) &aRows, sizeof(size_t));
    aInputFile.read((char*) &aCols, sizeof(size_t));
    auto *aMatrix = new float[aRows * aCols * sizeof(float)];
    aInputFile.read((char*) aMatrix, aRows * aCols * sizeof(float));

    ifstream bInputFile(argv[2], ios::binary | ios::in);
    char bType; size_t bRows; size_t bCols;
    bInputFile.read(&bType, sizeof(char));
    bInputFile.read((char*) &bRows, sizeof(size_t));
    bInputFile.read((char*) &bCols, sizeof(size_t));
    auto *bMatrix = new float[bRows * bCols * sizeof(float)];
    bInputFile.read((char*) bMatrix, bRows * bCols * sizeof(float));

    aInputFile.close();
    bInputFile.close();

    char cType = 'f'; size_t cRows = aRows; size_t cCols = bCols;
    auto *cMatrix = new float[cRows * cCols * sizeof(float)];

    /*auto* testCMatrix = new float[cRows * cCols * sizeof(float)];
    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < bCols; j++) {
            for (int k = 0; k < aCols; k++) {
                testCMatrix[i * cCols + j] += aMatrix[i * cCols + k] * bMatrix[k * cCols + j];
            }
        }
    }*/

    size_t n = aRows;
    int err1;
    fill_n(cMatrix, cRows * cCols, 0.0);
    if ((err1 = PAPI_start_counters(events, eventsNum)) != PAPI_OK) {
        cout << err1 << endl;
        cout << "PAPI_error" << endl;
    }
    startTime = clock();
    for (int i = 0; i < aRows; i += defBlockSize) {
        for (int j = 0; j < bCols; j += defBlockSize) {
            for (int k = 0; k < aCols; k += defBlockSize) {
                for (int i1 = i; (i1 < i + defBlockSize) && (i1 < aRows); i1++) {
                    for (int j1 = j; (j1 < j + defBlockSize) && (j1 < bCols); j1++) {
                        for (int k1 = k; (k1 < k + defBlockSize) && (k1 < aCols); k1++) {
                            cMatrix[i1 * cCols + j1] += aMatrix[i1 * aCols + k1] * bMatrix[k1 * bCols + j1];
                        }
                    }
                }
            }
        }
    }
    deltaTimes[0] += (double) (clock() - startTime) / CLOCKS_PER_SEC;
    if (PAPI_stop_counters(values, eventsNum) != PAPI_OK) {
        cout << "PAPI_error" << endl;
    }
    L1Misses[0] = values[0];
    L2Misses[0] = values[1];
    cycles[0] = values[2];

    //cout << test(cMatrix, testCMatrix, cCols) << endl;

    fill_n(cMatrix, cRows * cCols, 0.0);
    if (PAPI_start_counters(events2, eventsNum2) != PAPI_OK) {
        cout << err1 << endl;
        cout << "PAPI_error" << endl;
    }
    for (int i = 0; i < aRows; i += defBlockSize) {
        for (int j = 0; j < bCols; j += defBlockSize) {
            for (int k = 0; k < aCols; k += defBlockSize) {
                for (int i1 = i; (i1 < i + defBlockSize) && (i1 < aRows); i1++) {
                    for (int j1 = j; (j1 < j + defBlockSize) && (j1 < bCols); j1++) {
                        for (int k1 = k; (k1 < k + defBlockSize) && (k1 < aCols); k1++) {
                            cMatrix[i1 * cCols + j1] += aMatrix[i1 * aCols + k1] * bMatrix[k1 * bCols + j1];
                        }
                    }
                }
            }
        }
    }
    if (PAPI_stop_counters(values2, eventsNum2) != PAPI_OK) {
        cout << "PAPI_error" << endl;
    }
    FLOPs[0] = values2[0];

    //cout << test(cMatrix, testCMatrix, cCols) << endl;

    fill_n(cMatrix, cRows * cCols, 0.0);
    if (PAPI_start_counters(events, eventsNum) != PAPI_OK) {
        cout << "PAPI_error" << endl;
    }
    startTime = clock();
    for (int i = 0; i < aRows; i += defBlockSize) {
        for (int k = 0; k < aCols; k += defBlockSize) {
            for (int j = 0; j < bCols; j += defBlockSize) {
                for (int i1 = i; (i1 < i + defBlockSize) && (i1 < aRows); i1++) {
                    for (int k1 = k; (k1 < k + defBlockSize) && (k1 < aCols); k1++) {
                        for (int j1 = j; (j1 < j + defBlockSize) && (j1 < bCols); j1++) {
                            cMatrix[i1 * cCols + j1] += aMatrix[i1 * aCols + k1] * bMatrix[k1 * bCols + j1];
                        }
                    }
                }
            }
        }
    }
    deltaTimes[1] += (double) (clock() - startTime) / CLOCKS_PER_SEC;
    if (PAPI_stop_counters(values, eventsNum) != PAPI_OK) {
        cout << "PAPI_error" << endl;
    }
    L1Misses[1] = values[0];
    L2Misses[1] = values[1];
    cycles[1] = values[2];

    //cout << test(cMatrix, testCMatrix, cCols) << endl;

    fill_n(cMatrix, cRows * cCols, 0.0);
    if (PAPI_start_counters(events2, eventsNum2) != PAPI_OK) {
        cout << "PAPI_error" << endl;
    }
    for (int i = 0; i < aRows; i += defBlockSize) {
        for (int k = 0; k < aCols; k += defBlockSize) {
            for (int j = 0; j < bCols; j += defBlockSize) {
                for (int i1 = i; (i1 < i + defBlockSize) && (i1 < aRows); i1++) {
                    for (int k1 = k; (k1 < k + defBlockSize) && (k1 < aCols); k1++) {
                        for (int j1 = j; (j1 < j + defBlockSize) && (j1 < bCols); j1++) {
                            cMatrix[i1 * cCols + j1] += aMatrix[i1 * aCols + k1] * bMatrix[k1 * bCols + j1];
                        }
                    }
                }
            }
        }
    }
    if (PAPI_stop_counters(values2, eventsNum2) != PAPI_OK) {
        cout << "PAPI_error" << endl;
    }
    FLOPs[1] = values2[0];

    //cout << test(cMatrix, testCMatrix, cCols) << endl;

    fill_n(cMatrix, cRows * cCols, 0.0);
    if (PAPI_start_counters(events, eventsNum) != PAPI_OK) {
        cout << "PAPI_error" << endl;
    }
    startTime = clock();
    for (int i = 0; i < aRows; i += optBlockSize) {
        for (int k = 0; k < aCols; k += optBlockSize) {
            for (int j = 0; j < bCols; j += optBlockSize) {
                for (int i1 = i; (i1 < i + optBlockSize) && (i1 < aRows); i1++) {
                    for (int k1 = k; (k1 < k + optBlockSize) && (k1 < aCols); k1++) {
                        for (int j1 = j; (j1 < j + optBlockSize) && (j1 < bCols); j1++) {
                            cMatrix[i1 * cCols + j1] += aMatrix[i1 * aCols + k1] * bMatrix[k1 * bCols + j1];
                        }
                    }
                }
            }
        }
    }
    deltaTimes[2] += (double) (clock() - startTime) / CLOCKS_PER_SEC;
    if (PAPI_stop_counters(values, eventsNum) != PAPI_OK) {
        cout << "PAPI_error" << endl;
    }
    L1Misses[2] = values[0];
    L2Misses[2] = values[1];
    cycles[2] = values[2];

    //cout << test(cMatrix, testCMatrix, cCols) << endl;

    fill_n(cMatrix, cRows * cCols, 0.0);
    if (PAPI_start_counters(events2, eventsNum2) != PAPI_OK) {
        cout << "PAPI_error" << endl;
    }
    for (int i = 0; i < aRows; i += optBlockSize) {
        for (int k = 0; k < aCols; k += optBlockSize) {
            for (int j = 0; j < bCols; j += optBlockSize) {
                for (int i1 = i; (i1 < i + optBlockSize) && (i1 < aRows); i1++) {
                    for (int k1 = k; (k1 < k + optBlockSize) && (k1 < aCols); k1++) {
                        for (int j1 = j; (j1 < j + optBlockSize) && (j1 < bCols); j1++) {
                            cMatrix[i1 * cCols + j1] += aMatrix[i1 * aCols + k1] * bMatrix[k1 * bCols + j1];
                        }
                    }
                }
            }
        }
    }
    if (PAPI_stop_counters(values2, eventsNum2) != PAPI_OK) {
        cout << "PAPI_error" << endl;
    }
    FLOPs[2] = values2[0];

    //cout << test(cMatrix, testCMatrix, cCols) << endl;

    //delete[] testCMatrix;
    delete[] cMatrix;
    delete[] aMatrix;
    delete[] bMatrix;

    for (int i = 0; i < 3; i ++) {
        cout << deltaTimes[i] << ' ' << cycles[i] << ' ' << L1Misses[i] << ' ' << L2Misses[i] << ' ' << FLOPs[i] << ' ' << endl;
    }

    ofstream outputFile;
    outputFile.open(argv[3], ios_base::app | ios_base::binary | ios_base::out);
    for (int i = 0; i < 3; i++) {
        outputFile.write((char*)(deltaTimes + i), sizeof(double));
    }
    for (int i = 0; i < 3; i++) {
        outputFile.write((char*)(cycles + i), sizeof(long long));
    }
    for (int i = 0; i < 3; i++) {
        outputFile.write((char*)(L1Misses + i), sizeof(long long));
    }
    for (int i = 0; i < 3; i++) {
        outputFile.write((char*)(L2Misses + i), sizeof(long long));
    }
    for (int i = 0; i < 3; i++) {
        outputFile.write((char*)(FLOPs + i), sizeof(long long));
    }
    outputFile.close();


    delete[] deltaTimes;
    delete[] cycles;
    delete[] L1Misses;
    delete[] L2Misses;
    delete[] FLOPs;

    return 0;
}