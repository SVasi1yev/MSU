#include <complex>
#include <fstream>
#include <cmath>
#include <iostream>
#include <string>
typedef std::complex<double> complexd;

class Basis{
public:
    int** vec;
    int size;
    int logsize;
    Basis(int n = 2) : size(n), logsize((int)pow(2,n)){
        vec = new int*[logsize];
        for(int i = 0; i < logsize; i++){
            vec[i] = new int[n + 1];
            int temp = i;
            int csum = 0;
            for(int j=0; temp>0; j++)
            {
                vec[i][j] = temp%2;
                temp/=2;
                csum += vec[i][j];
            }
            vec[i][n] = size - csum;
        }
    }
    ~Basis(){
        for(int i = 0; i < logsize; i++){
            delete[] vec[i];
        }
        delete[] vec;
    }
};

template<typename T>
T** genH(Basis& b, T wc, T wa, T g){
    auto** H = new T*[b.logsize];
    for(int i = 0; i < b.logsize; i++){
        H[i] = new T[b.logsize];
        for(int j = 0; j < b.logsize; j ++){
            if (i == j){
                H[i][j] = double(b.vec[i][b.size]) * wc + double(b.size - b.vec[i][b.size]) * wa;
            } else{
                int csum = 0;
                for(int k = 0; k < b.size; k++){
                    csum += abs(b.vec[i][k] - b.vec[j][k]);
                }
                if (csum == 1){
                    H[i][j] = g;
                }
                else{
                    H[i][j] = 0;
                }
            }
        }
    }
    return H;
}


int main(int argc, char** argv){
    if (argc < 6){
        std::cout<<"./gen n ofileName wcreal|wcimagine wareal|waimagine greal|gimagine"<<std::endl;
        return -1;
    }
    int n = atoi(argv[1]);
    Basis b(n);
    int dposwc = std::string(argv[3]).find('|');
    int dposwa = std::string(argv[4]).find('|');
    int dposg = std::string(argv[5]).find('|');
    complexd wc = complexd(atof(std::string(argv[3]).substr(0,dposwc).c_str()), atof(std::string(argv[3]).substr(dposwc + 1, std::string::npos).c_str()));
    complexd wa = complexd(atof(std::string(argv[4]).substr(0,dposwa).c_str()), atof(std::string(argv[4]).substr(dposwa + 1, std::string::npos).c_str()));
    complexd g = complexd(atof(std::string(argv[5]).substr(0,dposg).c_str()), atof(std::string(argv[5]).substr(dposg + 1, std::string::npos).c_str()));
    auto** H = genH(b, wc,wa,g);
    std::ofstream fout;
    fout.open(argv[2],std::ios::binary | std::ios::out);
    for(int i = 0; i < b.logsize; i ++){
        for(int j = 0; j < b.logsize; j++){
            fout.write((char*) (H[i] + j), sizeof(complexd));
            std::cout<<*(H[i] + j)<<' ';
        }
        std::cout << '\n';
        delete[] H[i];
    }
    delete[] H;
    return 0;
}