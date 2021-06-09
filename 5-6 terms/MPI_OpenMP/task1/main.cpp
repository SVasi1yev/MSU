#include <iostream>
#include <fstream>
#include <ctime>

#define tripleFor(COUNTER1, BOARD1, COUNTER2, BOARD2, COUNTER3, BOARD3) \
    for (int (COUNTER1) = 0; (COUNTER1) < (BOARD1); (COUNTER1)++) \
        for (int (COUNTER2) = 0; (COUNTER2) < (BOARD2); (COUNTER2)++) \
            for (int (COUNTER3) = 0; (COUNTER3) < (BOARD3); (COUNTER3)++)

using namespace std;

int main(int argc, char *argv[]) {
    ifstream aInputFile(argv[1], ios::binary | ios::in);
    char aType; size_t aRows; size_t aCols;
    aInputFile.read(&aType, sizeof(char));
    aInputFile.read((char*) &aRows, sizeof(size_t));
    aInputFile.read((char*) &aCols, sizeof(size_t));
    char *aMatrix;
    if (aType == 'f') {
        aMatrix = new char[aRows * aCols * sizeof(float)];
        aInputFile.read(aMatrix, aRows * aCols * sizeof(float));
    } else if (aType == 'd') {
        aMatrix = new char[aRows * aCols * sizeof(double)];
        aInputFile.read(aMatrix, aRows * aCols * sizeof(double));
    }

    ifstream bInputFile(argv[2], ios::binary | ios::in);
    char bType; size_t bRows; size_t bCols;
    bInputFile.read(&bType, sizeof(char));
    bInputFile.read((char*) &bRows, sizeof(size_t));
    bInputFile.read((char*) &bCols, sizeof(size_t));
    char *bMatrix;
    if (bType == 'f') {
        bMatrix = new char[bRows * bCols * sizeof(float)];
        bInputFile.read(bMatrix, bRows * bCols * sizeof(float));
    } else if (bType == 'd') {
        bMatrix = new char[bRows * bCols * sizeof(double)];
        bInputFile.read(bMatrix, bRows * bCols * sizeof(double));
    }

    aInputFile.close();
    bInputFile.close();

    clock_t startTime = 0, endTime = 0, deltaTime = 0;

    if (aCols != bRows) {
        return -1;
    }

    char cType; size_t cRows = aRows; size_t cCols = bCols;
    char *cMatrix;

    if ((aType == 'd') || (bType == 'd')) {
        auto *newAMatrix = new double[aRows * aCols];
        if (aType == 'f') {
            for (int i = 0; i < aRows * aCols; i++) {
                newAMatrix[i] = *((float*)(aMatrix + i * sizeof(float)));
            }
        } else {
            for (int i = 0; i < aRows * aCols; i++) {
                newAMatrix[i] = *((double*)(aMatrix + i * sizeof(double)));
            }
        }

        auto *newBMatrix = new double[bRows * bCols];
        if (bType == 'f') {
            for (int i = 0; i < bRows * bCols; i++) {
                newBMatrix[i] = *((float*)(bMatrix + i * sizeof(float)));
            }
        } else {
            for (int i = 0; i < bRows * bCols; i++) {
                newBMatrix[i] = *((double*)(bMatrix + i * sizeof(double)));
            }
        }

        cType = 'd';
        auto *newCMatrix = new double[cRows * cCols];
        for (int i = 0; i < cRows * cCols; i++){
            newCMatrix[i]= 0.0;
        }

        switch (argv[4][0] - '0') {
            case 0:
                startTime = clock();
                tripleFor(i, cRows, j, cCols, k, aCols) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            case 1:
                startTime = clock();
                tripleFor(i, cRows, k, aCols, j, cCols) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            case 2:
                startTime = clock();
                tripleFor(k, aCols, i, cRows, j, cCols) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            case 3:
                startTime = clock();
                tripleFor(j, cCols, i, cRows, k, aCols) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            case 4:
                startTime = clock();
                tripleFor(j, cCols, k, aCols, i, cRows) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            case 5:
                startTime = clock();
                tripleFor(k, aCols, j, cCols, i, cRows) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            default:
                break;
        }

        cMatrix = (char*) newCMatrix;
        delete[] newAMatrix;
        delete[] newBMatrix;
    } else {
        auto *newAMatrix = new float[aRows * aCols];
        for (int i = 0; i < aRows * aCols; i++) {
            newAMatrix[i] = *((float*)(aMatrix + i * sizeof(float)));
        }

        auto *newBMatrix = new float[bRows * bCols];
        for (int i = 0; i < bRows * bCols; i++) {
            newBMatrix[i] = *((float*)(bMatrix + i * sizeof(float)));
        }

        cType = 'f';
        auto *newCMatrix = new float[cRows * cCols];
        for (int i = 0; i < cRows * cCols; i++){
            newCMatrix[i]= 0.0;
        }

        switch (argv[4][0] - '0') {
            case 0:
                startTime = clock();
                tripleFor(i, cRows, j, cCols, k, aCols) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            case 1:
                startTime = clock();
                tripleFor(i, cRows, k, aCols, j, cCols) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            case 2:
                startTime = clock();
                tripleFor(k, aCols, i, cRows, j, cCols) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            case 3:
                startTime = clock();
                tripleFor(j, cCols, i, cRows, k, aCols) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            case 4:
                startTime = clock();
                tripleFor(j, cCols, k, aCols, i, cRows) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            case 5:
                startTime = clock();
                tripleFor(k, aCols, j, cCols, i, cRows) {
                    newCMatrix[i * cCols + j] += newAMatrix[i * aCols + k] * newBMatrix[k * bCols + j];
                }
                endTime = clock();
                break;
            default:
                break;
        }

        cMatrix = (char*) newCMatrix;
        delete[] newAMatrix;
        delete[] newBMatrix;
    }

    deltaTime = endTime - startTime;
    ofstream outputFile;
    outputFile.open(argv[3], ios::binary | ios::out | ios::trunc);
    outputFile.write(&cType, sizeof(char));
    outputFile.write((char*) &cRows, sizeof(size_t));
    outputFile.write((char*) &cCols, sizeof(size_t));
    if (cType == 'f') {
        outputFile.write(cMatrix, cRows * cCols * sizeof(float));
    } else {
        outputFile.write(cMatrix, cRows * cCols * sizeof(double));
    }
    outputFile.close();
    delete[] aMatrix;
    delete[] bMatrix;
    delete[] cMatrix;

    clock_t averageTime[6];
    int nums[6];
    char c;
    int n = 0;
    ifstream inputTimeFile;
    inputTimeFile.open("Time.dat");

    if (!inputTimeFile.is_open()) {
        return 0;
    }

    for (int i = 0; i < 6; i++) {
        inputTimeFile.read(&c, 1);
        inputTimeFile.read(&c, 1);
        inputTimeFile.read(&c, 1);
        while (c != '\t') {
            n = n * 10 + (c - '0');
            inputTimeFile.read(&c, 1);
        }
        averageTime[i] = n;
        n = 0;
        inputTimeFile.read(&c, 1);
        while (c != '\n') {
            n = n * 10 + (c - '0');
            inputTimeFile.read(&c, 1);
        }
        nums[i] = n;
        n = 0;
    }

    averageTime[argv[4][0] - '0'] = (averageTime[argv[4][0] - '0'] * nums[argv[4][0] - '0']
            + deltaTime) / ++nums[argv[4][0] - '0'];

    inputTimeFile.close();

    ofstream outputTimeFile;
    outputTimeFile.open("Time.dat");

    for (int i = 0; i < 6; i++) {
        outputTimeFile << i << '\t' << averageTime[i] << '\t' << nums[i] << '\n';
    }

    outputTimeFile.close();
}