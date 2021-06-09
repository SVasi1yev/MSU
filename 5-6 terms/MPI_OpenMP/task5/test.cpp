#include <iostream>
#include <fstream>
#include <cstdlib>


using namespace std;


int main (int argc, char *argv[]) {
    ifstream aInputFile(argv[1], ios::binary | ios::in);
    char aType; int aRows; int aCols;
    aInputFile.read(&aType, sizeof(char));
    aInputFile.read((char*) &aRows, sizeof(int));
    aInputFile.read((char*) &aCols, sizeof(int));
    double *aMatrix = new double[aRows * aCols];
    aInputFile.read((char*) aMatrix, aRows * aCols * sizeof(double));

    ifstream bInputFile(argv[2], ios::binary | ios::in);
    char bType; int bRows; int bCols;
    bInputFile.read(&bType, sizeof(char));
    bInputFile.read((char*) &bRows, sizeof(int));
    bInputFile.read((char*) &bCols, sizeof(int));
    double *bMatrix = new double[bRows * bCols];
    bInputFile.read((char*) bMatrix, bRows * bCols);

    ifstream cInputFile(argv[3], ios::binary | ios::in);
    char cType; int cRows; int cCols;
    bInputFile.read(&cType, sizeof(char));
    bInputFile.read((char*) &cRows, sizeof(int));
    bInputFile.read((char*) &cCols, sizeof(int));
    double *cMatrix = new double[cRows * cCols];
    bInputFile.read((char*) cMatrix, cRows * cCols);

    aInputFile.close();
    bInputFile.close();
    cInputFile.close();

    double *testMatrix = new double[aRows * bCols];
    fill_n(testMatrix, aRows * bCols, 0);
    for (int i = 0; i < aRows; i++) {
        for (int k = 0; k < aCols; k++) {
            for (int j = 0; j < bCols; j++) {
                testMatrix[i * bCols + j] += aMatrix[i * aCols + k] * bMatrix[k * bCols + j];
            }
        }
    }

    bool fl = true;

    for (int i = 0; i < cRows * cCols; i++) {
        if (cMatrix[i] != testMatrix[i]) {
            fl = false;
            break;
        }
    }

    if (fl) {
        cout << "RIGHT!!!" << endl;
    } else {
        cout << "WRONG!!!" << endl;
    }

    delete[] aMatrix;
    delete[] bMatrix;
    delete[] cMatrix;
    delete[] testMatrix;

    return 0;
}