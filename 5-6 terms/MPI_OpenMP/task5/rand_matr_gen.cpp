#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <fstream>

using namespace std;

template <typename T>
void genMatrixFile(char type, int rows, int cols, char* outputFileName) {
    T *matrix = new T[rows * cols];
    srand(time(0));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = -5 + rand() % 11;
    }
    ofstream outputFile;
    outputFile.open(outputFileName, ios::binary | ios::out | ios::trunc);
    outputFile.write(&type, sizeof(char));
    outputFile.write((char*) &rows, sizeof(int));
    outputFile.write((char*) &cols, sizeof(int));
    outputFile.write((char*) matrix, rows * cols * sizeof(T));
    outputFile.close();
    delete[] matrix;
}

int main (int argc, char *argv[]) {
    char type  = *argv[1];
    int rows = 0;
    for (char *c = argv[2]; *c != '\0'; c++) {
        rows = rows * 10 + (*c - '0');
    }
    int cols = 0;
    for (char *c = argv[3]; *c != '\0'; c++) {
        cols = cols * 10 + (*c - '0');
    }
    char* outputFileName = argv[4];

    if (type == 'f') {
        genMatrixFile<float>(type, rows, cols, outputFileName);
    } else {
        genMatrixFile<double>(type, rows, cols, outputFileName);
    }
}