#include <fstream>
#include <iostream>

using namespace std;

int main (int argc, char* argv[]) {
    ifstream inputFile;
    inputFile.open(argv[1], ios_base::in | ios_base::binary);
    ofstream outputFile;
    outputFile.open(argv[2]);
    char* stats = new char[sizeof(double) * 15 + sizeof(long long) * 60];
    inputFile.read(stats, sizeof(double) * 15 + sizeof(long long) * 60);
    for (int i = 0; i < 5; i++) {
        outputFile << i << '\t';
        cout << i << '\t';
        for (int k = 0; k < 3; k++) {
            outputFile << *((double*)(stats + sizeof(double) * (i * 3 + k) + sizeof(long long) * i * 12)) << '\t';
            cout << *((double*)(stats + sizeof(double) * (i * 3 + k) + sizeof(long long) * i * 12)) << '\t';
        }
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                outputFile << *((long long*)(stats + sizeof(double) * (i + 1) * 3 + sizeof(long long) * (i * 12 + j * 3 + k))) << '\t';
                cout << *((long long*)(stats + sizeof(double) * (i + 1) * 3 + sizeof(long long) * (i * 12 + j * 3 + k))) << '\t';
            }
        }
        outputFile << '\n';
        cout << '\n';
    }
    delete[] stats;
    outputFile.close();
    inputFile.close();
}
