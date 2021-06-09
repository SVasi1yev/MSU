#ifndef QUANTBIT_H
#define QUANTBIT_H

#include <cstdint>
#include <math.h>
#include <algorithm>
#include <stdexcept>

namespace QuantGen {

    class QuantBit {
    private:
        double alpha;
        double beta;
    public:
        QuantBit() {
            alpha = 1.0;
            beta = 0.0;
        }

        QuantBit(double alpha, double beta):
                alpha(alpha), beta(beta) {}

        QuantBit(const QuantBit& other) {
            alpha = other.getAlpha();
            beta = other.getBeta();
        }

        QuantBit& operator=(const QuantBit& other) {
            if (this != &other) {
                alpha = other.getAlpha();
                beta = other.getBeta();
            }

            return *this;
        }

        void rotate(double angle);

        void inverse() { std::swap(alpha, beta); }

        inline double getAlpha() const { return alpha; }
        inline double getBeta() const { return beta; }

        void setAmplitudes(double alpha, double beta);
    };

}

#endif //QUANTBIT_H
