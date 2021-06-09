#include <fstream>
#include <iostream>

using namespace std;

const double eps = 0.0001;

int main(int argc, char *argv[]) {
    ifstream aInputFile(argv[1], ios::binary | ios::in);
    char aType; size_t aRows; size_t aCols;
    aInputFile.read(&aType, sizeof(char));
    aInputFile.read((char*) &aRows, sizeof(size_t));
    aInputFile.read((char*) &aCols, sizeof(size_t));

    ifstream bInputFile(argv[2], ios::binary | ios::in);
    char bType; size_t bRows; size_t bCols;
    bInputFile.read(&bType, sizeof(char));
    bInputFile.read((char*) &bRows, sizeof(size_t));
    bInputFile.read((char*) &bCols, sizeof(size_t));

    bool equal = true;

    if ((aType != bType) || (aRows != bRows) || (aCols != bCols)) {
        equal = false;
    } else {
        if (aType == 'f') {
            float *aMatrix = new float[aRows * aCols];
            aInputFile.read((char*) aMatrix, aRows * aCols * sizeof(float));
            float *bMatrix = new float[bRows * bCols];
            bInputFile.read((char*) bMatrix, bRows * bCols * sizeof(float));
            for (int i = 0; i < aRows; i++) {
                for (int j = 0; j < aCols; j++) {
                    if ((aMatrix[i * aCols + j] - bMatrix[i * bCols + j]) > eps) {
                        equal = false;
                        break;
                    }
                }
            }
            delete[] aMatrix;
            delete[] bMatrix;
        } else {
            double *aMatrix = new double[aRows * aCols];
            aInputFile.read((char*) &aMatrix, aRows * aCols * sizeof(double));
            double *bMatrix = new double[bRows * bCols];
            bInputFile.read((char*) &bMatrix, bRows * bCols * sizeof(double));
            for (int i = 0; i < aRows; i++) {
                for (int j = 0; j < aCols; j++) {
                    if (aMatrix[i * aCols + j] != bMatrix[i * bCols + j]) {
                        equal = false;
                        break;
                    }
                }
            }
            delete[] aMatrix;
            delete[] bMatrix;
        }
    }

    aInputFile.close();
    bInputFile.close();

    if (equal) {
        cout << "Matrixes are equal" << endl;
        return 0;
    } else {
        cout << "Matrixes aren't equal" << endl;
        return -1;
    }
}