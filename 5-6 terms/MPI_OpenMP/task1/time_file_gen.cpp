#include <fstream>

using namespace std;

int main() {
    ofstream timeFile;
    timeFile.open("Time.dat");

    for (int i = 0; i < 6; i++) {
        timeFile << i << '\t' << 0 << '\t' << 0 << '\n';
    }

    timeFile.close();
}
