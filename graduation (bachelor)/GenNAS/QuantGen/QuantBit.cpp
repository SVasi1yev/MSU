#include "QuantBit.h"

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
