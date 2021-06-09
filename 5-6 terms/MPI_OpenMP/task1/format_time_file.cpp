#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

int main () {
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

    double doubleAverageTime[6];

    for (int i = 0; i < 6; i++) {
        doubleAverageTime[i] = (double)averageTime[i] / CLOCKS_PER_SEC;
    }

    inputTimeFile.close();

    ofstream outputTimeFile;
    outputTimeFile.open("Time.dat");

    for (int i = 0; i < 6; i++) {
        outputTimeFile << i << '\t' << doubleAverageTime[i] << '\t' << nums[i] << '\n';
    }

    outputTimeFile.close();
}
