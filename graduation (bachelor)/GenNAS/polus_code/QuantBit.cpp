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

using namespace QuantGen;

void QuantBit::rotate(double angle) {
    double old_alpha = alpha;
    double old_beta = beta;

    alpha = std::cos(angle) * old_alpha - std::sin(angle) * old_beta;
    beta = std::sin(angle) * old_alpha + std::cos(angle) * old_beta;
}

void QuantBit::setAmplitudes(double alpha, double beta) {
    if (std::abs(alpha * alpha + beta * beta - 1) < 0.00001) { //!!!
        this->alpha = alpha;
        this->beta = beta;
    } else {
        throw std::runtime_error("Unnormalized amplitudes");
    }
}