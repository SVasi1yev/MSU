#include <iostream>
#include <fstream>

using namespace std;

int main (int argc, char *argv[]) {
    char* inputFileName = argv[1];
    ifstream inputFile(inputFileName, ios::binary | ios::in);

    char type;
    inputFile.read((char*) &type, sizeof(char));
    cout << type << endl;
    size_t rows;
    inputFile.read((char*) &rows, sizeof(size_t));
    cout << rows << endl;
    size_t cols;
    inputFile.read((char*) &cols, sizeof(size_t));
    cout << cols << endl;

    if (type == 'f') {
        float *matrix = new float[rows * cols];
        inputFile.read((char*) matrix, rows * cols * sizeof(float));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << matrix[i * cols + j] << ' ';
            }
            cout << endl;
        }
        delete[] matrix;
        inputFile.close();
    } else {
        double *matrix = new double[rows * cols];
        inputFile.read((char*) matrix, rows * cols * sizeof(double));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cout << matrix[i * cols + j] << ' ';
            }
            cout << endl;
        }
        delete[] matrix;
        inputFile.close();
    }
}